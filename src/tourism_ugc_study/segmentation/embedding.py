"""固定模型身份的同步语义向量接口。

本模块仅负责原文 token 计数和 Qwen 推理，不决定切分边界。模型依赖
延迟导入，因此旧规则和纯向量测试不需要加载 torch 或模型权重。
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
from typing import Any, Protocol, Sequence

import numpy as np
from .contracts import EmbeddingError


class TokenCounter(Protocol):
    """仅计数正文并提供码点位置，不允许字符数估计或 decode 重建正文。"""

    max_input_tokens: int

    def count(self, text: str) -> int: ...

    def input_count(self, text: str) -> int: ...

    def offsets(self, text: str) -> tuple[tuple[int, int], ...]: ...


class EmbeddingProvider(Protocol):
    """以输入顺序返回二维向量，身份包含所有影响实际输入／推理的配置。"""

    identity: dict[str, Any]

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def identity_sha256(identity: dict[str, Any]) -> str:
    """对有限 JSON 身份计算规范摘要；拒绝 NaN 等非标准 JSON 值。"""
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_vectors(vectors: Any, *, count: int, dimension: int) -> np.ndarray:
    """核验数量、维度、有限值与非零范数，返回连续 float32 副本。

    不自动归一化第三方输出：修正数据会掩盖 provider 或缓存损坏。
    """
    value = np.asarray(vectors)
    if value.ndim != 2 or value.shape != (count, dimension) or dimension < 1:
        raise EmbeddingError("invalid_vector_shape")
    if not np.issubdtype(value.dtype, np.floating):
        raise EmbeddingError("invalid_vector_dtype")
    if not np.isfinite(value).all():
        raise EmbeddingError("nonfinite_vector")
    if count and np.any(np.linalg.norm(value.astype(np.float64), axis=1) <= 1e-12):
        raise EmbeddingError("zero_vector")
    converted = np.ascontiguousarray(value, dtype=np.float32)
    if not np.isfinite(converted).all():
        raise EmbeddingError("float32_vector_overflow")
    return converted.copy()


def _validate_model_parameters(model: Any, *, device: str, expected_dtype: Any,
                               cuda_device_index: int | None = None) -> dict[str, Any]:
    """核验实际浮点参数的设备与精度，不能以加载请求代替实际身份。

    非浮点参数不构成精度证据；旋转位置编码等内部缓冲区可合法使用不同
    精度，因此这里只核验模型参数。任一浮点参数漂移或没有浮点参数都失败。
    """
    dtypes, devices = set(), set()
    count = 0
    for parameter in model.parameters():
        if not parameter.dtype.is_floating_point:
            continue
        count += 1
        if parameter.dtype != expected_dtype:
            raise EmbeddingError("loaded_model_parameter_dtype_mismatch")
        actual_device = parameter.device
        if (actual_device.type != device
                or (device == "cuda" and actual_device.index != cuda_device_index)):
            raise EmbeddingError("loaded_model_parameter_device_mismatch")
        dtypes.add(str(parameter.dtype).removeprefix("torch."))
        devices.add(str(actual_device))
    if not count:
        raise EmbeddingError("loaded_model_has_no_floating_parameters")
    return {"loaded_parameter_dtypes": sorted(dtypes), "loaded_parameter_devices": sorted(devices),
            "floating_parameter_count": count}


class QwenTokenCounter:
    """只加载同 revision 的 tokenizer/config，提供精确正文及实际输入计数。

    revision 必须是完整提交 SHA。原文不规范化；指令模板仅用于模型，
    EOS 沿用官方原生 Transformers 示例的 tokenizer 行为，不手动追加。
    """

    def __init__(self, *, model_id: str, revision: str, max_input_tokens: int = 256,
                 instruction: str = "Retrieve semantically similar text",
                 local_files_only: bool = True, cache_dir: str | None = None) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise EmbeddingError("revision_must_be_commit_sha")
        if isinstance(max_input_tokens, bool) or not isinstance(max_input_tokens, int) or max_input_tokens < 1:
            raise EmbeddingError("invalid_max_input_tokens")
        if not isinstance(instruction, str) or not instruction.strip():
            raise EmbeddingError("empty_instruction")
        from transformers import AutoConfig, AutoTokenizer

        self._asset_paths: dict[str, Path] = {}
        self._asset_stats: dict[str, tuple[int, int, int, int, int]] = {}
        self.identity = {"assets_sha256": {}}
        self._asset_kwargs = dict(model_id=model_id, revision=revision,
                                  local_files_only=local_files_only, cache_dir=cache_dir)
        self._bind_assets(("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"))
        asset_hashes = self.identity["assets_sha256"]
        kwargs = dict(revision=revision, local_files_only=local_files_only,
                      cache_dir=cache_dir, trust_remote_code=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", **kwargs)
        self.model_config = AutoConfig.from_pretrained(model_id, **kwargs)
        self.verify_assets(strong=False)
        if not self.tokenizer.is_fast or self.tokenizer.padding_side != "left":
            raise EmbeddingError("fast_left_padding_tokenizer_required")
        if self.tokenizer.pad_token_id is None:
            raise EmbeddingError("missing_padding_token")
        resolved = getattr(self.model_config, "_commit_hash", None)
        if resolved != revision:
            raise EmbeddingError("unresolved_model_revision")
        self.max_input_tokens = max_input_tokens
        if max_input_tokens > self.model_config.max_position_embeddings:
            raise EmbeddingError("input_limit_exceeds_model_context")
        self.instruction = instruction
        self.identity = {
            "provider": "qwen-transformers-local-v1", "model_id": model_id,
            "model_revision": revision, "tokenizer_revision": revision,
            "dimension": int(self.model_config.hidden_size),
            "instruction": instruction, "template": "Instruct: {instruction}\nQuery:{text}",
            "special_tokens": "tokenizer_native_no_manual_eos",
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id, "padding_side": "left",
            "max_input_tokens": max_input_tokens, "truncation": False,
            "pooling": "last_valid_token", "normalization": "l2_float32",
            "quantization": None,
            "native_normalizer": str(self.tokenizer.backend_tokenizer.normalizer),
            "offset_policy": "original_grapheme_envelope",
            "assets_sha256": asset_hashes,
            "dependencies": {name: importlib.metadata.version(name)
                             for name in ("transformers", "tokenizers", "huggingface-hub", "numpy")},
        }

    @staticmethod
    def _file_sha(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _stat_identity(path: Path) -> tuple[int, int, int, int, int]:
        value = path.stat()
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    def _bind_assets(self, names: Sequence[str]) -> None:
        """绑定本地实际资产字节，并防止摘要计算中资产被替换。"""
        from huggingface_hub import hf_hub_download

        for name in names:
            args = self._asset_kwargs
            path = Path(hf_hub_download(args["model_id"], name, revision=args["revision"],
                                       local_files_only=args["local_files_only"], cache_dir=args["cache_dir"]))
            before = self._stat_identity(path)
            digest = self._file_sha(path)
            if self._stat_identity(path) != before:
                raise EmbeddingError("model_asset_changed_during_hash")
            self._asset_paths[name] = path
            self._asset_stats[name] = before
            self.identity["assets_sha256"][name] = digest

    def verify_assets(self, *, strong: bool = True) -> None:
        """运行前后可强哈希复验；每次计数／推理用 stat 检测资产改变。"""
        for name, path in self._asset_paths.items():
            if self._stat_identity(path) != self._asset_stats[name]:
                raise EmbeddingError("model_asset_changed")
            if strong and self._file_sha(path) != self.identity["assets_sha256"][name]:
                raise EmbeddingError("model_asset_digest_changed")

    def format_input(self, text: str) -> str:
        """返回精确模型输入；不删除空白或改变原文字符。"""
        if not isinstance(text, str):
            raise EmbeddingError("text_must_be_string")
        return f"Instruct: {self.instruction}\nQuery:{text}"

    def count(self, text: str) -> int:
        """正文 token 数，不含指令及特殊 token；空串返回零。"""
        if not isinstance(text, str):
            raise EmbeddingError("text_must_be_string")
        self.verify_assets(strong=False)
        return len(self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])

    def input_count(self, text: str) -> int:
        """实际模型输入长度，包含模板及 tokenizer 原生特殊 token。"""
        self.verify_assets(strong=False)
        return len(self.tokenizer(self.format_input(text), add_special_tokens=True,
                                  truncation=False)["input_ids"])

    def offsets(self, text: str) -> tuple[tuple[int, int], ...]:
        """返回正文 token 的 Unicode 码点区间；允许字节 BPE 的重复区间。

        所有非空正文必须覆盖从首到尾且不留空洞。区间仅用于提出候选，
        上层仍须回退至字素边界并重计完整原文子串。
        """
        if not isinstance(text, str):
            raise EmbeddingError("text_must_be_string")
        import regex

        self.verify_assets(strong=False)
        result = self.tokenizer(text, add_special_tokens=False, truncation=False,
                                return_offsets_mapping=True)
        raw_positions = tuple((int(a), int(b)) for a, b in result["offset_mapping"])
        # 原生 NFC 的 offset 有时遗漏同簇末尾组合符号；以原文字素包络
        # 扩大候选而不修改正文或 token 计数，最后仍要求无空洞覆盖。
        envelopes = {}
        for match in regex.finditer(r"\X", text):
            for index in range(match.start(), match.end()):
                envelopes[index] = (match.start(), match.end())
        if any(not (0 <= start < end <= len(text)) for start, end in raw_positions):
            raise EmbeddingError("invalid_token_offsets")
        positions = tuple((envelopes[a][0], envelopes[b - 1][1]) for a, b in raw_positions)
        if len(positions) != len(result["input_ids"]):
            raise EmbeddingError("invalid_token_offsets")
        previous_start = 0
        covered = 0
        for start, end in positions:
            if not (0 <= start < end <= len(text)) or start < previous_start or start > covered:
                raise EmbeddingError("invalid_token_offsets")
            previous_start, covered = start, max(covered, end)
        if covered != len(text):
            raise EmbeddingError("incomplete_token_offsets")
        return positions


class QwenEmbeddingProvider(QwenTokenCounter):
    """显式加载一次固定 Qwen 模型，同步按原顺序编码。

    只接受 CPU/MPS/CUDA 和明确的浮点精度；无量化、自动设备分配、
    自动重试或模型降级。每次 encode 都实际推理，缓存由独立模块控制。
    """

    def __init__(self, *, model_id: str, revision: str, device: str, dtype: str,
                 batch_size: int, max_input_tokens: int = 256,
                 instruction: str = "Retrieve semantically similar text",
                 local_files_only: bool = True, cache_dir: str | None = None) -> None:
        if device not in {"cpu", "mps", "cuda"}:
            raise EmbeddingError("unsupported_device")
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise EmbeddingError("unsupported_dtype")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise EmbeddingError("invalid_batch_size")
        import torch
        from transformers import AutoModel

        if device == "mps" and not torch.backends.mps.is_available():
            raise EmbeddingError("mps_unavailable")
        if device == "cuda" and not torch.cuda.is_available():
            raise EmbeddingError("cuda_unavailable")
        if (device == "cuda" and dtype == "bfloat16"
                and not torch.cuda.is_bf16_supported(including_emulation=False)):
            raise EmbeddingError("cuda_bfloat16_not_natively_supported")
        if device == "cuda":
            # FP32 声明不应借由 TF32 计算静默降精度；不改变既有 MPS 身份。
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        super().__init__(model_id=model_id, revision=revision, instruction=instruction,
                         max_input_tokens=max_input_tokens, local_files_only=local_files_only,
                         cache_dir=cache_dir)
        if device == "mps":
            # 本机统一内存同时服务系统；任何调用入口都保留保护上限。
            torch.mps.set_per_process_memory_fraction(0.5)
        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

        # 与Transformers默认checkpoint选择一致：单文件存在时优先使用它，
        # 只有单文件缺失才选择索引及分片，否则身份会遗漏实际加载的权重。
        try:
            self._bind_assets(("model.safetensors",))
        except (EntryNotFoundError, LocalEntryNotFoundError):
            self._bind_assets(("model.safetensors.index.json",))
            # 先绑定索引，再从同一已绑定路径列出分片；否则索引读取后换文件
            # 会留下“新索引摘要＋旧分片摘要”，无法证明实际加载的新权重。
            # 绑定之后的变化仍由既有加载后资产复验拒绝，不重新接受新身份。
            index = self._asset_paths["model.safetensors.index.json"]
            weight_names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
            self._bind_assets(weight_names)
        self.device, self.batch_size = device, batch_size
        self.model = AutoModel.from_pretrained(
            model_id, revision=revision, local_files_only=local_files_only,
            cache_dir=cache_dir, trust_remote_code=False, dtype=getattr(torch, dtype),
            attn_implementation="eager", use_cache=False,
        ).to(device)
        self.model.eval()
        actual_parameters = _validate_model_parameters(
            self.model, device=device, expected_dtype=getattr(torch, dtype),
            cuda_device_index=torch.cuda.current_device() if device == "cuda" else None)
        self.verify_assets(strong=False)
        if getattr(self.model.config, "_commit_hash", None) != revision:
            raise EmbeddingError("loaded_model_revision_mismatch")
        self.model.config.use_cache = False
        self.identity.update(device=device, dtype=dtype, attention_backend="eager", use_cache=False)
        self.identity["mps_memory_fraction"] = 0.5 if device == "mps" else None
        self.identity["torch_num_threads"] = torch.get_num_threads()
        self.identity["dependencies"]["torch"] = importlib.metadata.version("torch")
        if device == "cuda":
            self.identity.update(actual_parameters)
            properties = torch.cuda.get_device_properties(torch.cuda.current_device())
            self.identity["cuda_device_name"] = properties.name
            self.identity["cuda_compute_capability"] = [properties.major, properties.minor]
            observed_uuid = getattr(properties, "uuid", None)
            self.identity["cuda_device_uuid"] = str(observed_uuid) if observed_uuid is not None else None
            self.identity["cuda_runtime_version"] = torch.version.cuda
            self.identity["float32_matmul_precision"] = torch.get_float32_matmul_precision()
            self.identity["cuda_matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
            self.identity["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
        self.model_call_count = 0

    @property
    def batch_size(self) -> int:
        """当前实际批大小；变化形成新的缓存身份。"""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise EmbeddingError("invalid_batch_size")
        self._batch_size = value
        self.identity["batch_size"] = value

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """返回 float32[N,D] 的 L2 向量；超限在任何模型调用前拒绝。"""
        import torch
        import torch.nn.functional as functional

        if isinstance(texts, (str, bytes)):
            raise EmbeddingError("texts_must_be_sequence")
        values = list(texts)
        self.verify_assets(strong=False)
        if any(not isinstance(text, str) or not text for text in values):
            raise EmbeddingError("empty_or_invalid_text")
        if any(self.input_count(text) > self.max_input_tokens for text in values):
            raise EmbeddingError("model_input_limit_exceeded")
        dimension = self.identity["dimension"]
        if not values:
            return np.empty((0, dimension), dtype=np.float32)
        output = []
        for begin in range(0, len(values), self.batch_size):
            batch = self.tokenizer([self.format_input(t) for t in values[begin:begin + self.batch_size]],
                                   padding=True, truncation=False, add_special_tokens=True,
                                   return_tensors="pt").to(self.device)
            if batch["input_ids"].shape[1] > self.max_input_tokens:
                raise EmbeddingError("model_input_limit_exceeded")
            with torch.inference_mode():
                self.model_call_count += 1
                hidden = self.model(**batch, use_cache=False).last_hidden_state
                mask = batch["attention_mask"]
                # 最大有效下标同样适用于左右 padding，避免把 padding 当末 token。
                last = (torch.arange(mask.shape[1], device=mask.device)[None, :] * mask).max(dim=1).values
                pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last].float()
                norm = torch.linalg.vector_norm(pooled, dim=1)
                if not torch.isfinite(pooled).all().item() or (norm <= 1e-12).any().item():
                    raise EmbeddingError("invalid_model_vector")
                normalized = functional.normalize(pooled, p=2, dim=1)
                output.append(normalized.cpu().numpy().copy())
        return validate_vectors(np.concatenate(output, axis=0), count=len(values), dimension=dimension)
