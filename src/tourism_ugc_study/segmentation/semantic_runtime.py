"""本机语义批次的独立进程保护与实际源码留存。

仅监督一个同步批次；macOS保持固定资源门，Linux/CUDA使用显式独立政策。
worker从冻结的Python源码执行，完整结果先留在私有证据目录；父进程核验资源
及worker终态后才rename到请求输出。不提供远端调度或恢复系统，强制停止不删除缓存。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from typing import Callable

from .artifacts import canonical_json, write_json
from .config import ResearchInput
from .embedding_probe import _pressure_level, _swap_bytes
from .semantic_config import EmbeddingConfig, SemanticConfig, SemanticSettings
from .semantic_development_input import (
    FrozenDevelopmentInput, frozen_development_parameters, read_frozen_development_input,
)
from .semantic_provenance import freeze_code, verify_code_snapshot
from . import semantic_resources
from .semantic_resources import LinuxResourcePolicy
from .source import file_sha256


@dataclass(frozen=True)
class SemanticRuntimeLimits:
    """批次阶段时间预算与采样间隔；资源政策独立，不作为语义参数或吞吐目标。

    load为模型加载，preflight为合成预检，post为完整单帖（可能含很多forward），
    total为整轮；全部以秒计且必须有限正数。sampling仅决定观察间隔。
    """

    load_seconds: float = 300
    preflight_seconds: float = 120
    post_seconds: float = 1800
    total_seconds: float = 14400
    sampling_seconds: float = 2

    def __post_init__(self) -> None:
        """拒绝非正或非有限预算，避免NaN把超时比较永久关闭。"""
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("invalid_semantic_runtime_limits")

    def manifest(self, resource_policy: LinuxResourcePolicy | None = None) -> dict:
        """记录时间预算及对应资源门；Linux 不沿用 macOS 的硬件假设。"""
        if resource_policy is not None:
            return {**asdict(self), **resource_policy.manifest(),
                    "critical_pressure_level": None, "mps_memory_fraction": None}
        return {**asdict(self), "rss_bytes": 8 * 1024**3,
                "additional_swap_bytes": 512 * 1024**2, "critical_pressure_level": 4,
                "mps_memory_fraction": 0.5}


class SemanticRuntimeError(RuntimeError):
    """仅携带本模块确定的稳定原因码，不转述第三方异常或正文。"""


def _now() -> str:
    """返回UTC证据时间；阶段计时使用monotonic，不受系统时钟调整影响。"""
    return datetime.now(timezone.utc).isoformat()


def _replace_json(path: Path, value: dict) -> None:
    """原子替换本次独占证据文件，文件权限保持0600。"""
    pending = path.with_name(path.name + ".pending")
    write_json(pending, value)
    pending.replace(path)


def _resource_sample(pid: int | None) -> dict:
    """读取worker RSS和整机swap／压力；不可读或未知压力值必须失败。

    pid=None仅用于worker启动前及退出后复核系统指标，不伪造进程RSS读数。
    """
    try:
        rss = None
        if pid is not None:
            observation = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(pid)],
                                         capture_output=True, text=True, check=False)
            value = observation.stdout.strip()
            if not value and observation.returncode == 1:
                raise SemanticRuntimeError("worker_rss_unavailable")
            if observation.returncode != 0:
                raise ValueError("rss_command_failed")
            rss = int(value) * 1024
            if rss <= 0:
                raise ValueError("invalid_rss")
        swap, pressure = _swap_bytes(), _pressure_level()
        if swap < 0 or pressure not in {1, 2, 4}:
            raise ValueError("invalid_system_resource_observation")
        return {"rss_bytes": rss, "swap_bytes": swap, "memory_pressure_level": pressure}
    except SemanticRuntimeError:
        raise
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        raise SemanticRuntimeError("resource_observation_failed") from exc


def _require_resources(sample: dict, baseline_swap: int) -> None:
    """资源超门即中止；系统swap含其他应用，不据此归因模型独占。"""
    if sample["rss_bytes"] is not None and sample["rss_bytes"] > 8 * 1024**3:
        raise SemanticRuntimeError("rss_limit_exceeded")
    if sample["swap_bytes"] - baseline_swap > 512 * 1024**2:
        raise SemanticRuntimeError("additional_swap_limit_exceeded")
    if sample["memory_pressure_level"] >= 4:
        raise SemanticRuntimeError("critical_memory_pressure")


def _sample_resources(pid: int | None, policy: LinuxResourcePolicy | None) -> dict:
    """选择平台观测器；保留旧 macOS 注入点，Linux 错误统一为监督器原因码。"""
    if policy is None:
        return _resource_sample(pid)
    try:
        return semantic_resources.sample_linux_resources(pid, policy)
    except semantic_resources.SemanticResourceError as exc:
        raise SemanticRuntimeError(str(exc)) from exc


def _guard_resources(sample: dict, baseline_swap: int, policy: LinuxResourcePolicy | None) -> None:
    """选择政策，Linux 的 None 压力值不进入 macOS 压力门。"""
    if policy is None:
        _require_resources(sample, baseline_swap)
        return
    try:
        semantic_resources.require_linux_resources(sample, baseline_swap, policy)
    except semantic_resources.SemanticResourceError as exc:
        raise SemanticRuntimeError(str(exc)) from exc


def _remember_cuda_sample(summary: dict, sample: dict) -> None:
    """保留整卡实际采样峰值，包含启动及退出观察；不冒充连续监测峰值。"""
    if "cuda_device_used_bytes" in sample:
        summary["peak_sampled_cuda_device_used_bytes"] = max(
            summary.get("peak_sampled_cuda_device_used_bytes", 0), sample["cuda_device_used_bytes"])


def _record_worker_resources(summary: dict, result: dict, policy: LinuxResourcePolicy | None) -> None:
    """先留存 worker 生命周期峰值再复核，避免采样间隔遗漏瞬时超门。"""
    peak = result["process_peak_rss_bytes"]
    if type(peak) is not int or peak < 0:
        raise SemanticRuntimeError("resource_observation_failed")
    summary["worker_peak_rss_bytes"] = peak
    if policy is not None:
        summary["worker_cuda_resources"] = {key: value for key, value in result.items() if key.startswith("cuda_")}
        if result.get("cuda_visible_device_uuid") != policy.cuda_device_uuid:
            raise SemanticRuntimeError("cuda_device_identity_mismatch")
        reserved = result.get("cuda_peak_reserved_bytes")
        if type(reserved) is not int or reserved < 0:
            raise SemanticRuntimeError("resource_observation_failed")
        if reserved > policy.cuda_memory_bytes:
            raise SemanticRuntimeError("worker_peak_cuda_reserved_limit_exceeded")
    if peak > (policy.rss_bytes if policy is not None else 8 * 1024**3):
        raise SemanticRuntimeError("worker_peak_rss_limit_exceeded")


def _stop_worker(process: subprocess.Popen) -> None:
    """先TERM再KILL并wait回收，原生推理不响应TERM时仍能可靠停止。"""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_guarded_semantic_batch(
    config: ResearchInput | FrozenDevelopmentInput, output: Path, *, settings: SemanticSettings, code_version: str,
    cache_dir: Path | None = None, post_ids: tuple[int, ...] | None = None,
    seed: int | None = None, members_path: Path | None = None,
    limits: SemanticRuntimeLimits | None = None,
    resource_policy: LinuxResourcePolicy | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """在冻结副本中运行受监督批次，父进程复核后发布完整新目录。

    output及其.runtime证据目录和.attempt.json都必须不存在。失败／保护中断留存
    私有日志、资源JSONL、源码及attempt终态，清理本worker临时包，保留外部缓存。
    子进程异常不回显，调用者收到稳定SemanticRuntimeError；详细traceback仅在日志。
    Darwin 省略政策时保持原命令行为；Linux 仅支持显式政策绑定的单卡 CUDA，
    不提供默认硬件预算。Darwin 拒绝 Linux 政策，不能借此绕过原固定上限。
    离线开发输入只序列化包路径及固定SHA，不携带数据库路径；自动固定包内
    全部成员。父进程在worker退出后再次验证包，变动时不发布结果。
    """
    system = platform.system()
    if resource_policy is not None and not isinstance(resource_policy, LinuxResourcePolicy):
        raise SemanticRuntimeError("invalid_linux_resource_policy")
    if system == "Darwin":
        if resource_policy is not None:
            raise SemanticRuntimeError("linux_resource_policy_requires_linux")
    elif system == "Linux":
        if resource_policy is None:
            raise SemanticRuntimeError("semantic_linux_resource_policy_required")
        if settings.embedding.device != "cuda":
            raise SemanticRuntimeError("semantic_linux_requires_cuda")
    else:
        raise SemanticRuntimeError("unsupported_semantic_runtime_platform")
    offline = isinstance(config, FrozenDevelopmentInput)
    if offline:
        post_ids, seed, members_path = frozen_development_parameters(config, post_ids, seed, members_path)
    output = Path(output).resolve()
    cache_dir = Path(cache_dir).resolve() if cache_dir else output.parent / ".semantic-vector-cache"
    members_path = Path(members_path).resolve() if members_path is not None else None
    if cache_dir == output or cache_dir.is_relative_to(output):
        raise ValueError("cache_must_be_outside_run_output")
    evidence = output.with_name(output.name + ".runtime")
    attempt = output.with_name(output.name + ".attempt.json")
    if cache_dir == evidence or cache_dir.is_relative_to(evidence):
        raise ValueError("cache_must_be_outside_runtime_evidence")
    if output.exists() or evidence.exists() or attempt.exists():
        raise FileExistsError("semantic_output_or_attempt_exists")
    limits = limits or SemanticRuntimeLimits()
    evidence.mkdir(parents=True, mode=0o700)
    started = time.monotonic()
    summary = {"status": "running", "started_at_utc": _now(), "phase": "starting",
               "limits": limits.manifest(resource_policy), "code_version": code_version,
               "config_sha256": settings.fingerprint, "evidence_directory": str(evidence), "counts": {}}
    write_json(attempt, summary)
    process = None
    worker_output = evidence / "completed-run"
    try:
        snapshot = freeze_code(evidence / "code-snapshot")
        snapshot["executed_from_snapshot"] = True
        summary["code_snapshot"] = snapshot
        serialized_input = ({"manifest": str(Path(config.manifest).resolve()),
                             "manifest_sha256": config.manifest_sha256} if offline else
                            {k: str(v) if k == "candidate_manifest_sha256" else str(Path(v).resolve())
                             for k, v in asdict(config).items()})
        request = {"input": serialized_input, "settings": asdict(settings),
                   "output": str(worker_output), "cache_dir": str(cache_dir), "code_version": code_version,
                   "post_ids": post_ids, "seed": seed, "members_path": str(members_path) if members_path else None,
                   "code_snapshot": snapshot, "resource_supervision": {"limits": limits.manifest(resource_policy),
                       "attempt_path": str(attempt), "resources_path": str(evidence / "resources.jsonl"),
                       "worker_log_path": str(evidence / "worker.log")}}
        if offline:
            request["input_kind"] = "frozen_development"
        if resource_policy is not None:
            request["resource_policy"] = asdict(resource_policy)
            summary["resource_policy"] = resource_policy.manifest()
        write_json(evidence / "request.json", request)
        baseline = _sample_resources(None, resource_policy)
        summary["baseline_resources"] = baseline
        summary["baseline_swap_bytes"] = baseline["swap_bytes"]
        _remember_cuda_sample(summary, baseline)
        _guard_resources(baseline, baseline["swap_bytes"], resource_policy)
        env = dict(os.environ, PYTHONPATH=str(evidence / "code-snapshot/src"), PYTHONDONTWRITEBYTECODE="1",
                   TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
        if resource_policy is not None:
            # 完整 UUID 限定可见设备；provider 的 cuda 仍指向逻辑设备 0。
            env["CUDA_VISIBLE_DEVICES"] = resource_policy.cuda_device_uuid
        log_path, resources_path = evidence / "worker.log", evidence / "resources.jsonl"
        with log_path.open("x", encoding="utf-8") as log, resources_path.open("x", encoding="utf-8") as resources:
            log_path.chmod(0o600)
            resources_path.chmod(0o600)
            process = subprocess.Popen([sys.executable, "-m", __name__, "--worker-request",
                                        str(evidence / "request.json")], cwd=evidence,
                                       stdout=log, stderr=log, env=env)
            stage, stage_started, last_count = None, started, 0
            while process.poll() is None:
                now = time.monotonic()
                state_path = evidence / "worker-state.json"
                state = json.loads(state_path.read_text()) if state_path.exists() else {"phase": "model_loading"}
                marker = (state["phase"], state.get("active_source_post_id"))
                if marker != stage:
                    stage, stage_started = marker, now
                summary.update(phase=state["phase"], counts=state.get("counts", {}))
                summary.pop("active_source_post_id", None)
                if "active_source_post_id" in state:
                    summary["active_source_post_id"] = state["active_source_post_id"]
                # 单帖计时只在post变化时重置；同帖的计数／日志活动不掩盖停滞。
                phase_limit = (limits.load_seconds if state["phase"] == "model_loading"
                               else limits.preflight_seconds if state["phase"] == "model_preflight"
                               else limits.post_seconds)
                if now - started > limits.total_seconds:
                    raise SemanticRuntimeError("semantic_total_timeout")
                if now - stage_started > phase_limit:
                    raise SemanticRuntimeError("semantic_phase_timeout")
                try:
                    sample = _sample_resources(process.pid, resource_policy)
                except SemanticRuntimeError as exc:
                    if str(exc) != "worker_rss_unavailable" or process.poll() is None:
                        raise
                    # RSS观察可与正常退出竞争；退出仍须系统复核及worker峰值验证。
                    break
                resources.write(canonical_json({"elapsed_seconds": now - started, **sample, **state}) + "\n")
                resources.flush()
                # 先留存触发读数，使保护中断的finally摘要包含超门样本及真实采样峰值。
                summary["last_resources"] = sample
                summary["peak_sampled_rss_bytes"] = max(summary.get("peak_sampled_rss_bytes", 0), sample["rss_bytes"])
                _remember_cuda_sample(summary, sample)
                if resource_policy is not None and "cuda_resources" in state:
                    summary["last_worker_cuda_resources"] = state["cuda_resources"]
                _guard_resources(sample, baseline["swap_bytes"], resource_policy)
                if resource_policy is not None and "cuda_resources" in state:
                    if state["cuda_resources"]["cuda_peak_reserved_bytes"] > resource_policy.cuda_memory_bytes:
                        raise SemanticRuntimeError("worker_peak_cuda_reserved_limit_exceeded")
                _replace_json(attempt, summary)
                done = state.get("counts", {}).get("post_count", 0)
                if progress and done != last_count:
                    expected = len(post_ids) if post_ids is not None else state.get("expected_post_count")
                    if expected is not None:
                        progress(done, expected)
                    last_count = done
                time.sleep(limits.sampling_seconds)
            if process.wait() != 0:
                child_attempt = evidence / "completed-run.attempt.json"
                if child_attempt.exists():
                    child = json.loads(child_attempt.read_text())
                    summary.update(counts=child.get("counts", {}), worker_error_type=child.get("error_type"))
                failed_result = evidence / "worker-result.json"
                if resource_policy is not None and failed_result.exists():
                    _record_worker_resources(summary, json.loads(failed_result.read_text()), resource_policy)
                raise SemanticRuntimeError("semantic_worker_failed")
        final = _sample_resources(None, resource_policy)
        summary["final_resources"] = final
        _remember_cuda_sample(summary, final)
        _guard_resources(final, baseline["swap_bytes"], resource_policy)
        if time.monotonic() - started > limits.total_seconds:
            raise SemanticRuntimeError("semantic_total_timeout")
        result = json.loads((evidence / "worker-result.json").read_text())
        _record_worker_resources(summary, result, resource_policy)
        verify_code_snapshot(snapshot)
        manifest = json.loads((worker_output / "manifest.json").read_text())
        if manifest["code_snapshot"] != snapshot or manifest["config_sha256"] != settings.fingerprint:
            raise SemanticRuntimeError("worker_manifest_identity_mismatch")
        summary["run_manifest_sha256"] = file_sha256(worker_output / "manifest.json")
        if output.exists():
            raise FileExistsError("semantic_output_exists_before_publication")
        if progress and manifest["counts"]["post_count"] != last_count:
            progress(manifest["counts"]["post_count"], manifest["counts"]["post_count"])
        if offline:
            # 最后一个用户回调也可能改包；在回调之后、发布之前复核输入身份。
            read_frozen_development_input(config)
        worker_output.rename(output)
        summary.update(status=manifest["status"], phase="complete", counts=manifest["counts"])
        summary.pop("active_source_post_id", None)
        return manifest
    except BaseException as exc:
        if process is not None:
            _stop_worker(process)
        # 这些路径只可能由本次worker在独占evidence内创建，避免删其他运行或cache。
        if worker_output.exists():
            shutil.rmtree(worker_output)
        for staging in evidence.glob(".completed-run-*"):
            if staging.is_dir():
                shutil.rmtree(staging)
        protection = isinstance(exc, SemanticRuntimeError) and str(exc) != "semantic_worker_failed"
        summary.update(status="interrupted" if protection or isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                       error_type=type(exc).__name__)
        if isinstance(exc, SemanticRuntimeError):
            summary["error_code"] = str(exc)
        raise
    finally:
        summary.update(completed_at_utc=_now(), elapsed_seconds=time.monotonic() - started)
        _replace_json(attempt, summary)


def _worker(request_path: Path) -> int:
    """执行一份冻结请求；异常完整traceback仅进入父进程创建的私有日志。"""
    from .semantic_batch import run_semantic_batch

    request = json.loads(request_path.read_text())
    raw_input = request["input"]
    if request.get("input_kind", "research_corpus") == "frozen_development":
        source = FrozenDevelopmentInput(Path(raw_input["manifest"]), raw_input["manifest_sha256"])
    elif request.get("input_kind", "research_corpus") == "research_corpus":
        source = ResearchInput(**{k: Path(v) if k != "candidate_manifest_sha256" else v
                                  for k, v in raw_input.items()})
    else:
        raise ValueError("invalid_semantic_runtime_input_kind")
    raw_settings = request["settings"]
    settings = SemanticSettings(SemanticConfig(**raw_settings["algorithm"]), EmbeddingConfig(**raw_settings["embedding"]))
    policy = LinuxResourcePolicy(**request["resource_policy"]) if "resource_policy" in request else None

    def worker_state(state: dict) -> None:
        """模型加载后附带 allocator 累计峰值；父进程可保留中断前最后报告。"""
        if policy is not None and state["phase"] != "model_loading":
            state = {**state, "cuda_resources": semantic_resources.cuda_worker_resources()}
        _replace_json(request_path.parent / "worker-state.json", state)

    def worker_resources() -> dict:
        """RSS 统一为字节；CUDA 当前值与峰值维持独立 allocator 范围。"""
        return {"process_peak_rss_bytes": semantic_resources.process_peak_rss_bytes(),
                **(semantic_resources.cuda_worker_resources() if policy is not None else {})}

    try:
        run_semantic_batch(source, Path(request["output"]), settings=settings, code_version=request["code_version"],
            cache_dir=Path(request["cache_dir"]), post_ids=tuple(request["post_ids"]) if request["post_ids"] else None,
            seed=request["seed"], members_path=Path(request["members_path"]) if request["members_path"] else None,
            code_snapshot=request["code_snapshot"], resource_supervision=request["resource_supervision"],
            state=worker_state)
        write_json(request_path.parent / "worker-result.json", worker_resources())
        return 0
    except BaseException:
        traceback.print_exc()
        if policy is not None:
            # 失败后仅保留可实际读取的峰值；报告失败不能覆盖原模型失败或制造正常读数。
            try:
                write_json(request_path.parent / "worker-result.json", worker_resources())
            except (semantic_resources.SemanticResourceError, OSError):
                pass
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="语义批次私有worker；常规运行使用research CLI")
    parser.add_argument("--worker-request", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(_worker(args.worker_request))
