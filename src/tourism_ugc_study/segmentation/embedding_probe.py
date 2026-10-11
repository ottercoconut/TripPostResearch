"""在独立进程验证本地 Qwen 接口，并监督内存、swap 与运行超时。

仅使用模块内自造中文样例，不访问科研正文。结果、向量和失败日志写入
调用者指定的私有目录；CPU 失败后不自动转到 MPS 或换模型重试。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np

from .embedding import EmbeddingError, QwenEmbeddingProvider, identity_sha256
from .embedding_cache import EmbeddingCache
from .semantic_config import EmbeddingConfig
from .semantic_resources import (
    LinuxResourcePolicy, SemanticResourceError, cuda_worker_resources,
    process_peak_rss_bytes, require_linux_resources, sample_linux_resources,
)

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
# FP32工程一致性门槛，在本次真实推理前固定；不代表主题效果或分段容差。
REPEAT_TOLERANCE = 1e-5
RUNTIME_TOLERANCE = 1e-4
NORM_TOLERANCE = 1e-5
SYNTHETIC_TEXTS = (
    "清晨的海边阳光明媚，沿着沙滩散步很舒服。",
    "早上去海滩走走，阳光照在海面上，让人心情舒畅。",
    "从车站乘坐公交到景区需要半小时，末班车在晚上八点。",
    "门票可以提前网上购买，学生凭证件享受半价。",
    "清晨的海边阳光明媚，沿着沙滩散步很舒服。",
    "一家人👨‍👩‍👧‍👦在海边拍照，孩子看见了彩虹🌈。",
    "第一天：参观博物馆。\n第二天：乘船游览海湾。",
    "这家餐厅的海鲜很新鲜，不过周末排队时间比较长。",
    "海景特别漂亮！交通也很方便！",
    "好看！方便！",
    "票价3.5元，路线可以查看https://example.org/?x=1;y=2。",
    "咖啡馆的招牌写着cafe\u0301，窗边能看见老街。",
)


def parse_smoke_tolerances(dtype: str, raw: object | None) -> dict[str, float]:
    """在资产核验／worker 启动前固定数值门，不允许运行后放宽。

    FP32 使用既有固定门；低精度必须显式声明 repeat/batch/norm 三个有限
    正数。未知键、bool、NaN、无穷值或不完整声明均以稳定 ValueError 拒绝。
    """
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("unsupported_smoke_dtype")
    fixed = {"repeat": REPEAT_TOLERANCE, "batch": RUNTIME_TOLERANCE, "norm": NORM_TOLERANCE}
    if raw is None:
        if dtype != "float32":
            raise ValueError("lower_precision_smoke_tolerances_required")
        return fixed
    if not isinstance(raw, dict) or set(raw) != set(fixed):
        raise ValueError("invalid_smoke_numerical_tolerances")
    try:
        valid = all(type(v) in {int, float} and math.isfinite(v) and v > 0 for v in raw.values())
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("invalid_smoke_numerical_tolerances")
    if dtype == "float32" and raw != fixed:
        raise ValueError("fp32_smoke_tolerances_fixed")
    return {key: float(value) for key, value in raw.items()}


def _synthetic_inputs_sha256(texts: list[str]) -> str:
    """绑定固定输入的规范 JSON 字节，不混同文件排版或原始研究语料。"""
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _require_unit_vectors(vectors: np.ndarray, *, norm_tolerance: float = NORM_TOLERANCE) -> None:
    """独立检查完整输出形状、float32、有限值和L2归一化，失败拒绝通过。"""
    if (not isinstance(vectors, np.ndarray) or vectors.ndim != 2
            or min(vectors.shape) < 1 or vectors.dtype != np.float32
            or not np.isfinite(vectors).all()):
        raise EmbeddingError("invalid_smoke_vectors")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if np.any(np.abs(norms - 1) > norm_tolerance):
        raise EmbeddingError("smoke_vector_not_normalized")


def _comparison_metrics(base: np.ndarray, other: np.ndarray, *, norm_tolerance: float = NORM_TOLERANCE) -> dict[str, float]:
    """按同输入顺序比较实际输出，使用float64计算避免诊断本身放大舍入。"""
    _require_unit_vectors(base, norm_tolerance=norm_tolerance)
    _require_unit_vectors(other, norm_tolerance=norm_tolerance)
    if base.shape != other.shape:
        raise EmbeddingError("smoke_comparison_shape_mismatch")
    left, right = base.astype(np.float64), other.astype(np.float64)
    return {"max_absolute_vector_difference": float(np.max(np.abs(left - right))),
            "max_pairwise_cosine_difference": float(np.max(np.abs(left @ left.T - right @ right.T)))}


def _require_numerical_consistency(metrics: dict, *, tolerance: float, label: str) -> None:
    """强制预先确定的数值门；缺项、非有限数和超差均不能标passed。"""
    keys = {"max_absolute_vector_difference", "max_pairwise_cosine_difference"}
    if (set(metrics) != keys or not np.isfinite(tolerance) or tolerance <= 0
            or any(type(v) not in (float, int) or not np.isfinite(v) or v < 0 for v in metrics.values())):
        raise EmbeddingError("invalid_smoke_comparison_metrics")
    if any(value > tolerance for value in metrics.values()):
        raise EmbeddingError(f"smoke_numerical_difference_exceeded:{label}")


def _pressure_level() -> int:
    """读取macOS内存压力等级；诊断与swap增量一起保留，不混同空闲页数。"""
    return int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True).strip())


def _write_json(path: Path, value: Any) -> None:
    """原子更新监督状态；不让监督者读到半个 JSON。"""
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def _sha_file(path: Path) -> str:
    """流式计算资产摘要，避免把全部权重读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_local_assets(*, embedding: EmbeddingConfig | None = None) -> dict[str, Any]:
    """核验已下载固定 snapshot 的必需文件及远端已知 LFS 摘要。

    本函数不下载、不改 revision；缺失资产必须由模型准备阶段明确解决。
    """
    from huggingface_hub import snapshot_download

    model_id, revision = (MODEL_ID, REVISION) if embedding is None else (embedding.model_id, embedding.revision)
    if embedding is None:
        required = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                    "vocab.json", "merges.txt", "README.md", "1_Pooling/config.json",
                    "modules.json", "config_sentence_transformers.json")
        snapshot = Path(snapshot_download(model_id, revision=revision, local_files_only=True,
                                          allow_patterns=list(required)))
    else:
        required = ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
        snapshot = Path(snapshot_download(model_id, revision=revision, local_files_only=True,
            allow_patterns=[*required, "model.safetensors", "model.safetensors.index.json", "model-*.safetensors"]))
        index = snapshot / "model.safetensors.index.json"
        if index.exists():
            value = json.loads(index.read_text(encoding="utf-8"))
            weight_map = value.get("weight_map") if isinstance(value, dict) else None
            if (not isinstance(weight_map, dict) or not weight_map
                    or any(not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors")
                           for name in weight_map.values())):
                raise EmbeddingError("invalid_weight_index")
            required = (*required, "model.safetensors.index.json", *sorted(set(weight_map.values())))
        else:
            required = (*required, "model.safetensors")
    assets = []
    for relative in required:
        path = snapshot / relative
        assets.append({"file": relative, "size": path.stat().st_size, "sha256": _sha_file(path)})
    expected = {
        "model.safetensors": "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
        "tokenizer.json": "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
    }
    if ((model_id, revision) == (MODEL_ID, REVISION)
            and any(asset["file"] in expected and asset["sha256"] != expected[asset["file"]] for asset in assets)):
        raise EmbeddingError("model_asset_digest_mismatch")
    return {"model_id": model_id, "revision": revision, "snapshot": str(snapshot), "assets": assets}


def _execute_worker(output_dir: Path, device: str, *, embedding: EmbeddingConfig | None = None,
                    numerical_tolerances: dict[str, float] | None = None,
                    contract_sha256: str | None = None) -> None:
    """单设备实际推理，重复/batch 对照全部绕过缓存，缓存另行检验。"""
    import torch

    os.umask(0o077)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o700)
    status = output_dir / "status.json"
    step = 0

    def phase(name: str) -> None:
        nonlocal step
        step += 1
        _write_json(status, {"phase": name, "step": step, "monotonic": time.monotonic()})

    phase("loading")
    torch.set_num_threads(2)
    torch.manual_seed(20261011)
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(0.5)
    configured = embedding or EmbeddingConfig(device=device)
    tolerances = parse_smoke_tolerances(configured.dtype, numerical_tolerances)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise EmbeddingError("cuda_unavailable")
        if torch.cuda.device_count() != 1:
            raise EmbeddingError("cuda_visible_device_count_mismatch")
        torch.cuda.set_device(0)
    provider_args = asdict(configured)
    provider_args["batch_size"] = 1
    provider = QwenEmbeddingProvider(**provider_args, local_files_only=True)
    _write_json(output_dir / "model-identity.json", provider.identity)
    texts = list(SYNTHETIC_TEXTS[:2] if device == "cpu" else SYNTHETIC_TEXTS)
    inputs = {"synthetic": True, "texts": texts}
    if device == "cuda":
        inputs["synthetic_inputs_sha256"] = _synthetic_inputs_sha256(texts)
    _write_json(output_dir / "inputs.json", inputs)

    def compute(name: str, batch_size: int) -> np.ndarray:
        provider.batch_size = batch_size
        _write_json(output_dir / f"{name}.identity.json", provider.identity)
        vectors = []
        for i in range(0, len(texts), batch_size):
            phase("inference")
            vectors.append(provider.encode(texts[i:i + batch_size]))
        result = np.concatenate(vectors, axis=0)
        _require_unit_vectors(result, norm_tolerance=tolerances["norm"])
        np.save(output_dir / f"{name}.npy", result, allow_pickle=False)
        return result

    base = compute("batch1", 1)
    repeat = compute("repeat_batch1", 1)
    batch2 = compute("batch2", 2)
    repeat_metrics = _comparison_metrics(base, repeat, norm_tolerance=tolerances["norm"])
    batch_metrics = _comparison_metrics(base, batch2, norm_tolerance=tolerances["norm"])
    _require_numerical_consistency(repeat_metrics, tolerance=tolerances["repeat"], label="repeat_batch1")
    _require_numerical_consistency(batch_metrics, tolerance=tolerances["batch"], label="batch2")
    if device == "mps":
        _require_numerical_consistency(_comparison_metrics(base[0:1], base[4:5]),
                                       tolerance=REPEAT_TOLERANCE, label="duplicate_input")
    if device == "cuda":
        for name, actual in (("duplicate_input_batch1", base), ("duplicate_input_batch2", batch2)):
            _require_numerical_consistency(
                _comparison_metrics(actual[0:1], actual[4:5], norm_tolerance=tolerances["norm"]),
                tolerance=tolerances["repeat"], label=name)
    provider.verify_assets()
    phase("validation")
    calls_before_rejection = provider.model_call_count
    try:
        provider.encode([texts[0], "海边散步" * 1000] if device == "cuda" else ["海边散步" * 1000])
    except EmbeddingError as exc:
        if str(exc) != "model_input_limit_exceeded":
            raise
    else:
        raise EmbeddingError("overlong_input_not_rejected")
    if calls_before_rejection != provider.model_call_count:
        raise EmbeddingError("overlong_input_reached_model")

    phase("inference")
    cache = EmbeddingCache(output_dir / "cache", provider.identity)
    cache_input = [texts[0], texts[1], texts[0]]
    first = cache.encode(cache_input, provider)
    cache_calls = provider.model_call_count
    hits_before = cache.cache_hit_count
    second = cache.encode([texts[1], texts[0], texts[1]], provider)
    hit_count = cache.cache_hit_count - hits_before
    if (not np.array_equal(first[[1, 0, 1]], second) or not np.array_equal(first[0], first[2])
            or provider.model_call_count != cache_calls or hit_count != len(cache_input)):
        raise EmbeddingError("cache_hit_did_not_reuse_exact_vectors")
    _require_numerical_consistency(_comparison_metrics(batch2[:2], first[:2], norm_tolerance=tolerances["norm"]),
                                   tolerance=tolerances["repeat"], label="cache_miss_against_actual_batch2")
    # 特意损坏隔离的 smoke 缓存，保留原记录与坏摘要作为拒绝命中的证据。
    entry = next(path for path in cache.root.iterdir() if path.is_dir())
    manifest_path = entry / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    (entry / "original-manifest.json").write_bytes(original_manifest)
    broken = json.loads(original_manifest)
    broken["vector_sha256"] = "0" * 64
    _write_json(manifest_path, broken)
    try:
        cache.encode(cache_input, provider)
    except EmbeddingError as exc:
        if "invalid_cache_entry" not in str(exc) or "cache_digest_mismatch" not in str(exc):
            raise
        if provider.model_call_count != cache_calls:
            raise EmbeddingError("corrupt_cache_reached_model") from exc
        corruption_rejected = True
    else:
        raise EmbeddingError("cache_corruption_not_rejected")

    result = {
        "status": "passed", "device": device, "identity": provider.identity,
        "identity_sha256": identity_sha256(provider.identity), "seed": 20261011,
        "input_count": len(texts), "shape": list(base.shape), "actual_model_calls": provider.model_call_count,
        "body_token_counts": [provider.count(text) for text in texts],
        "input_token_counts": [provider.input_count(text) for text in texts],
        "repeat_batch1": repeat_metrics, "batch2": batch_metrics,
        "numerical_consistency": "passed", "cache_reordered_hit_preserves_exact_vectors": True,
        "min_norm": float(np.linalg.norm(base, axis=1).min()),
        "max_norm": float(np.linalg.norm(base, axis=1).max()),
        "cache_hit_count_before_corruption": hit_count, "cache_miss_count": cache.cache_miss_count,
        "cache_corruption_rejected": corruption_rejected, "overlong_rejected_before_forward": True,
        "vector_files": {name: _sha_file(output_dir / name) for name in
                         ("batch1.npy", "repeat_batch1.npy", "batch2.npy")},
        "thread_count": torch.get_num_threads(),
        "process_peak_rss_bytes": process_peak_rss_bytes() if device == "cuda" else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "mps_recommended_max_memory": torch.mps.recommended_max_memory() if device == "mps" else None,
        "mps_driver_allocated_memory": torch.mps.driver_allocated_memory() if device == "mps" else None,
    }
    if device == "cuda":
        provider.verify_assets()
        result.update(cuda_worker_resources())
        result["numerical_tolerances"] = tolerances
        result["synthetic_inputs_sha256"] = _synthetic_inputs_sha256(texts)
        result["requested_embedding"] = asdict(configured)
        result["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        result["preflight_contract_sha256"] = contract_sha256
    _write_json(output_dir / "result.json", result)
    phase("complete")


def _worker(output_dir: Path, device: str, *, embedding: EmbeddingConfig | None = None,
            numerical_tolerances: dict[str, float] | None = None,
            contract_sha256: str | None = None) -> None:
    """执行合成预检；CUDA 失败时尽可能留存真实 allocator 峰值后重抛。

    仅当 CUDA 已初始化且读数可得时记录其峰值；无法观测时留下原因，
    不用零代替缺失读数。Mac worker 的原始失败流程不增加报告分支。
    """
    try:
        _execute_worker(output_dir, device, embedding=embedding,
                        numerical_tolerances=numerical_tolerances, contract_sha256=contract_sha256)
    except BaseException as exc:
        if device == "cuda":
            report = {"status": "failed", "device": "cuda", "error_type": type(exc).__name__,
                      "error": str(exc), "preflight_contract_sha256": contract_sha256}
            try:
                report["process_peak_rss_bytes"] = process_peak_rss_bytes()
                report.update(cuda_worker_resources())
            except Exception as observation_error:
                report["worker_resource_observation_error"] = str(observation_error)
            _write_json(output_dir / "result.json", report)
        raise


def _swap_bytes() -> int:
    """读取 macOS 当前 swap 已用字节；读数不可得时停止监督。"""
    value = subprocess.check_output(["/usr/sbin/sysctl", "-n", "vm.swapusage"], text=True)
    match = re.search(r"used\s*=\s*([0-9.]+)([KMG])", value)
    if match is None:
        raise RuntimeError("cannot_read_swap_usage")
    return int(float(match.group(1)) * {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)])


def _run_macos_smoke(output_dir: str | Path) -> dict[str, Any]:
    """按 CPU→MPS 顺序启动独立 worker，返回完整成功或失败摘要。

    仅支持本轮 macOS；RSS≤8GiB、新增 swap≤512MiB，加载≤300秒、
    每次推理≤120秒、整轮≤600秒。失败保留日志并停止后续设备。
    output_dir 必须不存在，防止覆盖先前实际实验。
    """
    if platform.system() != "Darwin":
        raise RuntimeError("local_smoke_requires_macos")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    started = time.monotonic()
    summary: dict[str, Any] = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "devices": {},
        "limits": {"rss_bytes": 8 * 1024**3, "additional_swap_bytes": 512 * 1024**2,
                   "load_seconds": 300, "inference_seconds": 120, "total_seconds": 600,
                   "mps_memory_fraction": 0.5, "sampling_seconds": 2},
        "numerical_tolerances": {"repeat": REPEAT_TOLERANCE, "batch_or_device": RUNTIME_TOLERANCE,
                                 "norm": NORM_TOLERANCE, "meaning": "FP32 engineering consistency, not semantic quality"},
    }
    _write_json(root / "summary.json", summary)
    process = None
    try:
        source_dir = root / "code-snapshot"
        source_dir.mkdir(mode=0o700)
        summary["code_sha256"] = {}
        for name in ("contracts.py", "embedding.py", "embedding_cache.py", "embedding_probe.py"):
            data = (Path(__file__).parent / name).read_bytes()
            copied = source_dir / name
            copied.write_bytes(data)
            copied.chmod(0o600)
            summary["code_sha256"][name] = hashlib.sha256(data).hexdigest()
        summary["assets"] = verify_local_assets()
        baseline_swap = _swap_bytes()
        summary["baseline_swap_bytes"] = baseline_swap
        summary["baseline_pressure_level"] = _pressure_level()
        if summary["baseline_pressure_level"] >= 4:
            raise RuntimeError("critical_memory_pressure_before_run")
        for device in ("cpu", "mps"):
            if any(_sha_file(Path(__file__).parent / name) != digest
                   for name, digest in summary["code_sha256"].items()):
                raise RuntimeError("probe_code_changed_during_attempt")
            device_dir = root / device
            device_dir.mkdir(mode=0o700)
            metrics = []
            with (device_dir / "worker.log").open("w", encoding="utf-8") as log:
                (device_dir / "worker.log").chmod(0o600)
                env = dict(os.environ, TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
                process = subprocess.Popen([sys.executable, "-m", __name__, "--worker", device,
                                            "--output", str(device_dir)], stdout=log, stderr=log, env=env)
                stage, stage_start = None, time.monotonic()
                while process.poll() is None:
                    now = time.monotonic()
                    state_path = device_dir / "status.json"
                    state = json.loads(state_path.read_text()) if state_path.exists() else {"phase": "loading", "step": 0}
                    marker = (state["phase"], state["step"])
                    if marker != stage:
                        stage, stage_start = marker, now
                    raw_rss = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(process.pid)],
                                             capture_output=True, text=True, check=False).stdout.strip()
                    if not raw_rss and process.poll() is not None:
                        break
                    rss = int(raw_rss) * 1024
                    swap = _swap_bytes()
                    pressure = _pressure_level()
                    metrics.append({"elapsed_seconds": round(now - started, 3), "rss_bytes": rss,
                                    "swap_bytes": swap, "memory_pressure_level": pressure, **state})
                    _write_json(device_dir / "resources.json", metrics)
                    if rss > 8 * 1024**3:
                        raise RuntimeError("rss_limit_exceeded")
                    if swap - baseline_swap > 512 * 1024**2:
                        raise RuntimeError("additional_swap_limit_exceeded")
                    if pressure >= 4:
                        raise RuntimeError("critical_memory_pressure_during_run")
                    if now - started > 600:
                        raise TimeoutError("smoke_total_timeout")
                    stage_limit = 300 if state["phase"] == "loading" else 120
                    if now - stage_start > stage_limit:
                        raise TimeoutError(f"smoke_{state['phase']}_timeout")
                    time.sleep(2)
                if process.wait() != 0:
                    raise RuntimeError(f"{device}_worker_failed:see_worker.log")
                # worker可能在两个采样点之间退出；设备切换前也必须核验整轮门槛。
                post_swap = _swap_bytes()
                if post_swap - baseline_swap > 512 * 1024**2:
                    raise RuntimeError("additional_swap_limit_exceeded")
                if time.monotonic() - started > 600:
                    raise TimeoutError("smoke_total_timeout")
                summary["devices"][device] = json.loads((device_dir / "result.json").read_text())
                if summary["devices"][device]["process_peak_rss_bytes"] > 8 * 1024**3:
                    raise RuntimeError("worker_peak_rss_limit_exceeded")
                summary["devices"][device]["peak_sampled_rss_bytes"] = max(item["rss_bytes"] for item in metrics)
                _write_json(root / "summary.json", summary)
        cpu = np.load(root / "cpu/batch1.npy", allow_pickle=False)
        mps = np.load(root / "mps/batch1.npy", allow_pickle=False)[:len(cpu)]
        cross = _comparison_metrics(cpu, mps)
        _require_numerical_consistency(cross, tolerance=RUNTIME_TOLERANCE, label="cpu_mps")
        summary["cross_device_max_absolute_difference"] = cross["max_absolute_vector_difference"]
        summary["cross_device_max_pairwise_cosine_difference"] = cross["max_pairwise_cosine_difference"]
        summary["numerical_consistency"] = "passed"
        summary["final_swap_bytes"] = _swap_bytes()
        summary["final_pressure_level"] = _pressure_level()
        if summary["final_swap_bytes"] - baseline_swap > 512 * 1024**2:
            raise RuntimeError("additional_swap_limit_exceeded")
        if summary["final_pressure_level"] >= 4:
            raise RuntimeError("critical_memory_pressure_after_run")
        if any(_sha_file(Path(__file__).parent / name) != digest
               for name, digest in summary["code_sha256"].items()):
            raise RuntimeError("probe_code_changed_during_attempt")
        summary["status"] = "passed"
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        summary.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        summary["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(root / "summary.json", summary)
    return summary


def _run_cuda_smoke(output_dir: str | Path, *, embedding: EmbeddingConfig,
                    resource_policy: LinuxResourcePolicy, numerical_tolerances: dict[str, float],
                    raw_tolerances: object | None) -> dict[str, Any]:
    """只监督显式声明的一张 CUDA 卡；不加载 CPU 对照或更换精度。

    以 GPU UUID 限定 worker 的逻辑 cuda:0，实际执行源码来自完整冻结包。
    失败保留私有日志与终态；全卡采样和 worker allocator 峰值分开记录。
    """
    from .semantic_provenance import freeze_code, verify_code_snapshot

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    started = time.monotonic()
    contract = {"embedding": asdict(embedding), "raw_numerical_tolerances": raw_tolerances,
                "numerical_tolerances": numerical_tolerances,
                "synthetic_inputs_sha256": _synthetic_inputs_sha256(list(SYNTHETIC_TEXTS))}
    summary: dict[str, Any] = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "devices": {},
        "preflight_contract": contract, "preflight_contract_sha256": identity_sha256(contract),
        "resource_policy": resource_policy.manifest(),
        "limits": {"load_seconds": 300, "inference_seconds": 120, "total_seconds": 600,
                   "sampling_seconds": 2},
        "numerical_tolerances": numerical_tolerances,
        "numerical_tolerances_meaning": "declared engineering consistency, not semantic quality",
        "cross_device_comparison": "not_required_cuda_only",
        "peak_sampled_rss_bytes": None,
        "peak_sampled_cuda_device_used_bytes": None,
        "sampled_cuda_device_peak_scope": "selected_device_global:nvidia-smi:sampled_baseline_live_final_not_continuous_peak",
    }
    _write_json(root / "summary.json", summary)
    process = None

    def retain_sample(sample: dict) -> None:
        """在每个资源门前保留采样峰值，超门失败也不能丢失触发读数。"""
        current = summary["peak_sampled_cuda_device_used_bytes"]
        summary["peak_sampled_cuda_device_used_bytes"] = max(current or 0, sample["cuda_device_used_bytes"])
        if sample["rss_bytes"] is not None:
            summary["peak_sampled_rss_bytes"] = max(summary["peak_sampled_rss_bytes"] or 0, sample["rss_bytes"])

    def retain_worker_report() -> None:
        """收集已落盘的真实 worker 报告，不把缺失报告拼成成功或零峰值。"""
        if "cuda" in summary["devices"]:
            return
        report_path = root / "cuda/result.json"
        if report_path.exists():
            try:
                summary["devices"]["cuda"] = json.loads(report_path.read_text())
            except (OSError, ValueError) as exc:
                summary["worker_resource_report_error"] = type(exc).__name__

    try:
        snapshot = freeze_code(root / "code-snapshot")
        snapshot["executed_from_snapshot"] = True
        summary["code_snapshot"] = snapshot
        baseline = sample_linux_resources(None, resource_policy)
        summary["baseline_resources"] = baseline
        retain_sample(baseline)
        require_linux_resources(baseline, baseline["swap_bytes"], resource_policy)
        summary["assets"] = verify_local_assets(embedding=embedding)
        device_dir = root / "cuda"
        device_dir.mkdir(mode=0o700)
        request = {"output": str(device_dir), "contract": contract,
                   "preflight_contract_sha256": summary["preflight_contract_sha256"]}
        _write_json(device_dir / "request.json", request)
        verify_code_snapshot(snapshot)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=resource_policy.cuda_device_uuid,
                   PYTHONPATH=str(root / "code-snapshot/src"), PYTHONDONTWRITEBYTECODE="1",
                   TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
        metrics = []
        with (device_dir / "worker.log").open("x", encoding="utf-8") as log:
            (device_dir / "worker.log").chmod(0o600)
            process = subprocess.Popen([sys.executable, "-m", "tourism_ugc_study.segmentation.embedding_probe",
                "--worker-request", str(device_dir / "request.json")], cwd=root, stdout=log, stderr=log, env=env)
            stage, stage_start = None, time.monotonic()
            while process.poll() is None:
                now = time.monotonic()
                state_path = device_dir / "status.json"
                state = json.loads(state_path.read_text()) if state_path.exists() else {"phase": "loading", "step": 0}
                marker = (state["phase"], state["step"])
                if marker != stage:
                    stage, stage_start = marker, now
                try:
                    observed = sample_linux_resources(process.pid, resource_policy)
                except SemanticResourceError as exc:
                    # 正常退出可能发生在 poll 与 /proc 读取之间；其他缺失读数不放行。
                    if str(exc) != "worker_rss_unavailable" or process.poll() is None:
                        raise
                    break
                metrics.append({"elapsed_seconds": round(now - started, 3), **observed, **state})
                _write_json(device_dir / "resources.json", metrics)
                summary["last_resources"] = observed
                retain_sample(observed)
                require_linux_resources(observed, baseline["swap_bytes"], resource_policy)
                if now - started > 600:
                    raise TimeoutError("smoke_total_timeout")
                stage_limit = 300 if state["phase"] == "loading" else 120
                if now - stage_start > stage_limit:
                    raise TimeoutError(f"smoke_{state['phase']}_timeout")
                time.sleep(2)
            summary["worker_return_code"] = process.wait()
        retain_worker_report()
        final = sample_linux_resources(None, resource_policy)
        summary["final_resources"] = final
        retain_sample(final)
        require_linux_resources(final, baseline["swap_bytes"], resource_policy)
        if summary["worker_return_code"] != 0:
            raise RuntimeError("cuda_worker_failed:see_worker.log")
        if time.monotonic() - started > 600:
            raise TimeoutError("smoke_total_timeout")
        result = json.loads((device_dir / "result.json").read_text())
        summary["devices"]["cuda"] = result
        if (result.get("status") != "passed" or result.get("device") != "cuda"
                or result.get("input_count") != len(SYNTHETIC_TEXTS)
                or result.get("requested_embedding") != asdict(embedding)
                or result.get("numerical_tolerances") != numerical_tolerances
                or result.get("synthetic_inputs_sha256") != contract["synthetic_inputs_sha256"]
                or result.get("preflight_contract_sha256") != summary["preflight_contract_sha256"]
                or result.get("cuda_visible_device_uuid") != resource_policy.cuda_device_uuid):
            raise EmbeddingError("cuda_smoke_worker_declaration_mismatch")
        observed_assets = {asset["file"]: asset["sha256"] for asset in summary["assets"]["assets"]}
        if result["identity"]["assets_sha256"] != observed_assets:
            raise EmbeddingError("cuda_smoke_asset_identity_mismatch")
        if result["process_peak_rss_bytes"] > resource_policy.rss_bytes:
            raise SemanticResourceError("worker_peak_rss_limit_exceeded")
        if result["cuda_peak_reserved_bytes"] > resource_policy.cuda_memory_bytes:
            raise SemanticResourceError("worker_peak_cuda_reserved_limit_exceeded")
        result["peak_sampled_rss_bytes"] = summary["peak_sampled_rss_bytes"]
        result["peak_sampled_cuda_device_used_bytes"] = summary["peak_sampled_cuda_device_used_bytes"]
        result["sampled_cuda_device_peak_scope"] = summary["sampled_cuda_device_peak_scope"]
        summary["numerical_consistency"] = "passed"
        verify_code_snapshot(snapshot)
        summary["status"] = "passed"
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        summary.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        retain_worker_report()
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        summary["ended_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(root / "summary.json", summary)
    return summary


def run_smoke(output_dir: str | Path, *, embedding: EmbeddingConfig | None = None,
              resource_policy: LinuxResourcePolicy | None = None,
              numerical_tolerances: object | None = None) -> dict[str, Any]:
    """保留默认 Mac smoke；显式模型声明启用单卡 Linux CUDA 预检。

    CUDA 参数错误在创建输出、核验资产和启动 worker 前拒绝。低精度不能
    略过数值门，FP32 门不能放宽；旧无参数入口仍按 CPU→MPS 执行。
    """
    if embedding is None:
        if resource_policy is not None or numerical_tolerances is not None:
            raise ValueError("cuda_smoke_algorithm_config_required")
        return _run_macos_smoke(output_dir)
    if embedding.device != "cuda":
        raise ValueError("cuda_smoke_requires_cuda_embedding")
    tolerances = parse_smoke_tolerances(embedding.dtype, numerical_tolerances)
    if resource_policy is None:
        raise ValueError("cuda_smoke_resource_policy_required")
    if platform.system() != "Linux":
        raise RuntimeError("cuda_smoke_requires_linux")
    return _run_cuda_smoke(output_dir, embedding=embedding, resource_policy=resource_policy,
                           numerical_tolerances=tolerances, raw_tolerances=numerical_tolerances)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen 本地独立 worker；常规调用请使用 probe CLI")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker", choices=("cpu", "mps"))
    mode.add_argument("--worker-request", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.worker_request:
        request = json.loads(arguments.worker_request.read_text(encoding="utf-8"))
        contract = request["contract"]
        digest = identity_sha256(contract)
        if (digest != request["preflight_contract_sha256"]
                or contract["synthetic_inputs_sha256"] != _synthetic_inputs_sha256(list(SYNTHETIC_TEXTS))):
            raise EmbeddingError("cuda_smoke_preflight_contract_mismatch")
        config = EmbeddingConfig(**contract["embedding"])
        if config.device != "cuda":
            raise EmbeddingError("cuda_smoke_requires_cuda_embedding")
        _worker(Path(request["output"]), "cuda", embedding=config,
                numerical_tolerances=contract["numerical_tolerances"], contract_sha256=digest)
    else:
        if arguments.output is None:
            parser.error("--worker需要--output")
        _worker(arguments.output, arguments.worker)
