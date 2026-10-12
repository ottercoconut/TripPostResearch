"""语义切分的固定经验参数和严格配置解析，不加载模型或打开数据库。

默认值是首轮工程起点，不代表科研最优值。配置含算法与embedding两部分，
所有字段进入谱系；路径由CLI显式提供，不在配置中存凭据或源正文。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re

RULE_VERSION = "semantic-depth-ugc-0.1"
IMPLEMENTATION_VERSION = "semantic-depth-ugc-0.1-original-1"
RUN_KIND = "semantic-text-segmentation-run"


def validate_semantic_rule_identity(manifest: dict) -> None:
    """只认可当前已登记的语义规则与实现配对，不把合法文件摘要当算法身份。

    本门不检查开发／全量范围，也不打开文件或加载模型；开发运行核验及跨模型
    汇总均可复用。缺失、未知或错配的身份抛 ValueError，不能静默解释为当前规则。
    """
    if (not isinstance(manifest, dict)
            or (manifest.get("rule_version"), manifest.get("implementation_version"))
            != (RULE_VERSION, IMPLEMENTATION_VERSION)):
        raise ValueError("semantic_rule_identity_mismatch")


@dataclass(frozen=True)
class SemanticConfig:
    """边界计算与候选单元参数；任何非法值在模型运行之前拒绝。

    token限制约束正文，模型输入含模板上限由EmbeddingConfig约束。
    epsilon用于峰值／平台容差，不能作为跨硬件效果验收阈值。
    """

    unit_max_tokens: int = 64
    window_tokens: int = 64
    absolute_cohesion_threshold: float = 0.55
    depth_floor: float = 0.15
    depth_std_factor: float = 0.5
    min_segment_nonspace_chars: int = 12
    max_segment_tokens: int = 384
    epsilon: float = 1e-6
    smoothing: int = 0

    def __post_init__(self) -> None:
        """拒绝bool伪装整数、非有限阈值和无意义的长度组合。"""
        for name in ("unit_max_tokens", "window_tokens", "min_segment_nonspace_chars", "max_segment_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"invalid_algorithm_parameter:{name}")
        for name in ("absolute_cohesion_threshold", "depth_floor", "depth_std_factor", "epsilon"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"invalid_algorithm_parameter:{name}")
        if not -1 <= self.absolute_cohesion_threshold <= 1:
            raise ValueError("invalid_cohesion_threshold")
        if not 0 < self.depth_floor <= 4 or self.depth_std_factor < 0 or not 0 < self.epsilon < 0.01:
            raise ValueError("invalid_depth_parameters")
        if self.unit_max_tokens > self.max_segment_tokens:
            raise ValueError("unit_limit_exceeds_segment_limit")
        if type(self.smoothing) is not int or self.smoothing != 0:
            raise ValueError("smoothing_not_supported")


@dataclass(frozen=True)
class EmbeddingConfig:
    """显式本地Qwen运行身份；同配置不得静默更换设备、模型或精度。"""

    model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    revision: str = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    device: str = "mps"
    dtype: str = "float32"
    batch_size: int = 1
    max_input_tokens: int = 256
    instruction: str = "Retrieve semantically similar text"

    def __post_init__(self) -> None:
        """型号仅限定已计划的Qwen系列，revision必须完整固定提交。"""
        if self.model_id not in {f"Qwen/Qwen3-Embedding-{s}" for s in ("0.6B", "4B", "8B")}:
            raise ValueError("unsupported_embedding_model")
        if not isinstance(self.revision, str) or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("embedding_revision_must_be_commit")
        if self.device not in {"cpu", "mps", "cuda"} or self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported_embedding_runtime")
        if any(type(v) is not int or v < 1 for v in (self.batch_size, self.max_input_tokens)):
            raise ValueError("invalid_embedding_limit")
        if self.max_input_tokens > 32768 or not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("invalid_embedding_input")


@dataclass(frozen=True)
class SemanticSettings:
    """同一运行的算法与模型声明，实际权重及设备身份另由provider补齐。"""

    algorithm: SemanticConfig
    embedding: EmbeddingConfig

    @property
    def fingerprint(self) -> str:
        """返回规范化配置摘要，不是完成运行后的batch_id。"""
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def load_semantic_settings(path: Path) -> SemanticSettings:
    """解析JSON，两层均拒绝未知字段；不允许缺少算法或模型声明。"""
    return parse_semantic_settings(json.loads(path.read_text(encoding="utf-8")))


def parse_semantic_settings(raw: object) -> SemanticSettings:
    """从清单内JSON对象恢复同一配置契约；不接受未知字段或隐式类型转换。"""
    if not isinstance(raw, dict) or set(raw) != {"algorithm", "embedding"}:
        raise ValueError("semantic_config_fields_invalid")
    values = []
    for key, cls in (("algorithm", SemanticConfig), ("embedding", EmbeddingConfig)):
        item = raw[key]
        if not isinstance(item, dict) or set(item) - set(cls.__dataclass_fields__):
            raise ValueError(f"semantic_config_fields_invalid:{key}")
        values.append(cls(**item))
    return SemanticSettings(*values)


def validate_identity_settings(identity: dict, settings: EmbeddingConfig) -> None:
    """比较实际provider身份与声明，防止测试注入或配置漂移伪造科研谱系。

    完整资产摘要另由provider复验；这里不打开模型文件，可用于运行及发布包
    的独立验收。测试provider也必须提供同一显式身份，不能例外发布。
    """
    expected = {
        "provider": "qwen-transformers-local-v1",
        "model_id": settings.model_id, "model_revision": settings.revision,
        "tokenizer_revision": settings.revision, "device": settings.device,
        "dtype": settings.dtype, "batch_size": settings.batch_size,
        "max_input_tokens": settings.max_input_tokens, "instruction": settings.instruction,
        "template": "Instruct: {instruction}\nQuery:{text}",
        "dimension": {"0.6B": 1024, "4B": 2560, "8B": 4096}[settings.model_id.rsplit("-", 1)[-1]],
        "special_tokens": "tokenizer_native_no_manual_eos", "padding_side": "left",
        "truncation": False, "pooling": "last_valid_token", "normalization": "l2_float32",
        "quantization": None, "attention_backend": "eager", "use_cache": False,
    }
    if not isinstance(identity, dict) or any(
        key not in identity or type(identity[key]) is not type(value) or identity[key] != value
        for key, value in expected.items()
    ):
        raise ValueError("embedding_identity_settings_mismatch")
