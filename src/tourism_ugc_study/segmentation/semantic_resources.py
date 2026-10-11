"""语义执行的 Linux/CUDA 资源政策、实际观测及 worker 峰值。

本模块不选择硬件、不加载模型，也不提供远端或容器调度。Linux 政策必须显式给定；
RSS 属于 worker，swap 属于 /proc/meminfo 可见系统，nvidia-smi 显存属于所选整卡。
CUDA allocator 的当前值及峰值单独报告，不能当作整卡或进程全部显存的峰值。
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import re
import resource
import subprocess


class SemanticResourceError(RuntimeError):
    """仅携带稳定资源原因码；底层命令输出及异常不进入公开摘要。"""


_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
_POLICY_FIELDS = {"rss_bytes", "additional_swap_bytes", "cuda_device_uuid", "cuda_memory_bytes"}


@dataclass(frozen=True)
class LinuxResourcePolicy:
    """显式 Linux/CUDA 资源门，单位均为字节且不提供硬件默认值。

    rss_bytes 限制 worker RSS，additional_swap_bytes 限制可见系统相对启动前的
    swap 增量（允许 0），cuda_memory_bytes 限制 UUID 所指整卡的实际已用显存。
    完整 GPU UUID 同时用于 CUDA_VISIBLE_DEVICES 与父进程观测，不支持序号或 MIG。
    """

    rss_bytes: int
    additional_swap_bytes: int
    cuda_device_uuid: str
    cuda_memory_bytes: int

    def __post_init__(self) -> None:
        """拒绝 bool、非整数、非法预算和不完整 UUID，避免默认政策悄然生效。"""
        if (type(self.rss_bytes) is not int or self.rss_bytes <= 0
                or type(self.additional_swap_bytes) is not int or self.additional_swap_bytes < 0
                or type(self.cuda_memory_bytes) is not int or self.cuda_memory_bytes <= 0
                or not isinstance(self.cuda_device_uuid, str)
                or _GPU_UUID.fullmatch(self.cuda_device_uuid) is None):
            raise SemanticResourceError("invalid_linux_resource_policy")

    def manifest(self) -> dict:
        """记录门值和精确适用范围；Linux 不伪造 macOS 内存压力等级。"""
        return {**asdict(self), "rss_scope": "worker_process:/proc/PID/status:VmRSS",
                "swap_scope": "system:/proc/meminfo:SwapTotal-SwapFree",
                "cuda_memory_scope": "selected_device_global:nvidia-smi",
                "memory_pressure_scope": "not_available_linux"}


def load_linux_resource_policy(path: Path) -> LinuxResourcePolicy:
    """从四字段 JSON 读取政策；纯解析不接触 GPU，未知或缺失字段均失败。"""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) != _POLICY_FIELDS:
            raise ValueError("invalid_policy_fields")
        return LinuxResourcePolicy(**raw)
    except SemanticResourceError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise SemanticResourceError("invalid_linux_resource_policy") from exc


def _proc_kib(text: str, key: str) -> int:
    """读取 Linux proc 文本中唯一的非负 kB 整数字段并转换为字节。"""
    matches = re.findall(rf"^{re.escape(key)}:\s+(\d+)\s+kB\s*$", text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError("invalid_proc_memory_field")
    return int(matches[0]) * 1024


def _cuda_device_resources(policy: LinuxResourcePolicy) -> dict:
    """查询完整 UUID 所指整卡；MiB 读数转换为字节，保留仪器精度说明。"""
    observation = subprocess.run(
        ["nvidia-smi", f"--id={policy.cuda_device_uuid}",
         "--query-gpu=uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False, timeout=5,
    )
    rows = list(csv.reader(observation.stdout.splitlines()))
    if observation.returncode != 0 or len(rows) != 1 or len(rows[0]) != 3:
        raise ValueError("invalid_cuda_observation")
    uuid, used, total = (item.strip() for item in rows[0])
    if uuid != policy.cuda_device_uuid:
        raise SemanticResourceError("cuda_device_identity_mismatch")
    if not used.isdecimal() or not total.isdecimal():
        raise ValueError("invalid_cuda_memory_value")
    used_bytes, total_bytes = int(used) * 1024**2, int(total) * 1024**2
    if total_bytes <= 0 or used_bytes > total_bytes:
        raise ValueError("invalid_cuda_memory_value")
    return {"cuda_device_uuid": uuid, "cuda_device_used_bytes": used_bytes,
            "cuda_device_total_bytes": total_bytes,
            "cuda_memory_scope": "selected_device_global:nvidia-smi",
            "cuda_memory_observation_unit": "MiB"}


def sample_linux_resources(pid: int | None, policy: LinuxResourcePolicy) -> dict:
    """读取 worker RSS、系统 swap 和所选整卡显存；不可读即失败。

    pid=None 用于启动前及退出后，仅进程 RSS 为 None。PID 状态文件不存在可与
    正常退出竞争，单独报 worker_rss_unavailable，由监督器核对进程终态。
    /proc/meminfo 读数不是 worker 独占量，本实现不声称支持 cgroup 内存范围。
    """
    try:
        rss = None
        if pid is not None:
            try:
                status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
            except FileNotFoundError as exc:
                raise SemanticResourceError("worker_rss_unavailable") from exc
            rss = _proc_kib(status, "VmRSS")
            if rss <= 0:
                raise ValueError("invalid_worker_rss")
        memory = Path("/proc/meminfo").read_text(encoding="utf-8")
        total_swap, free_swap = _proc_kib(memory, "SwapTotal"), _proc_kib(memory, "SwapFree")
        if free_swap > total_swap:
            raise ValueError("invalid_system_swap")
        return {"rss_bytes": rss, "swap_bytes": total_swap - free_swap,
                "memory_pressure_level": None,
                "rss_scope": "worker_process:/proc/PID/status:VmRSS",
                "swap_scope": "system:/proc/meminfo:SwapTotal-SwapFree",
                "memory_pressure_scope": "not_available_linux", **_cuda_device_resources(policy)}
    except SemanticResourceError:
        raise
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise SemanticResourceError("resource_observation_failed") from exc


def require_linux_resources(sample: dict, baseline_swap: int, policy: LinuxResourcePolicy) -> None:
    """复核 Linux 显式资源门；系统 swap 或整卡显存超门不归因模型独占。"""
    if sample["cuda_device_uuid"] != policy.cuda_device_uuid:
        raise SemanticResourceError("cuda_device_identity_mismatch")
    if policy.cuda_memory_bytes > sample["cuda_device_total_bytes"]:
        raise SemanticResourceError("cuda_memory_limit_exceeds_device_total")
    if sample["rss_bytes"] is not None and sample["rss_bytes"] > policy.rss_bytes:
        raise SemanticResourceError("rss_limit_exceeded")
    if sample["swap_bytes"] - baseline_swap > policy.additional_swap_bytes:
        raise SemanticResourceError("additional_swap_limit_exceeded")
    if sample["cuda_device_used_bytes"] > policy.cuda_memory_bytes:
        raise SemanticResourceError("cuda_memory_limit_exceeded")


def process_peak_rss_bytes() -> int:
    """将进程累计 ru_maxrss 转成字节；Darwin 原为字节，Linux 原为 KiB。"""
    system = platform.system()
    if system not in {"Darwin", "Linux"}:
        raise SemanticResourceError("unsupported_semantic_runtime_platform")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if system == "Darwin" else peak * 1024)


def cuda_worker_resources() -> dict:
    """读取已初始化 CUDA worker 的 allocator 当前值及累计峰值，不加载模型。

    调用方须在创建 worker 时用完整单卡 UUID 设置 CUDA_VISIBLE_DEVICES，逻辑
    设备固定为 0。未初始化、身份不符或读数不可得均失败；峰值未重置，覆盖本进程
    包括模型加载在内的累计 allocator 使用，不能替代整卡采样峰值。
    """
    try:
        import torch

        uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if _GPU_UUID.fullmatch(uuid) is None or not torch.cuda.is_initialized():
            raise SemanticResourceError("cuda_device_identity_mismatch")
        if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
            raise SemanticResourceError("cuda_device_identity_mismatch")
        values = {"cuda_allocated_bytes": torch.cuda.memory_allocated(0),
                  "cuda_reserved_bytes": torch.cuda.memory_reserved(0),
                  "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                  "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
        if any(type(value) is not int or value < 0 for value in values.values()):
            raise ValueError("invalid_cuda_allocator_observation")
        return {**values, "cuda_allocator_scope": "worker_torch_caching_allocator_lifetime",
                "cuda_logical_device": 0, "cuda_visible_device_uuid": uuid}
    except SemanticResourceError:
        raise
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        raise SemanticResourceError("resource_observation_failed") from exc
