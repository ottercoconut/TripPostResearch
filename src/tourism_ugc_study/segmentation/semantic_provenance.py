"""语义运行的精确源码副本与摘要复验，不依赖批处理或进程调度。

源码及依赖声明不含原始研究数据。监督者执行同一副本；直接API调用保存当前
工作区副本，并在结束时复验工作区未变化，明确区分两种执行方式。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .artifacts import canonical_json
from .source import file_sha256


def freeze_code(destination: Path) -> dict:
    """保存切分package、入口及依赖声明的精确字节，返回可核验源码清单。

    destination必须新建，0700目录内仅含源码与依赖声明，不读取研究正文。
    不把HEAD+dirty当实现证据；摘要绑定相对路径和每个文件的SHA。
    """
    root = Path(__file__).resolve().parents[3]
    paths = [root / "src/tourism_ugc_study/__init__.py", root / "pyproject.toml",
             root / "scripts/text_segment_research.py"]
    paths.extend(sorted((root / "src/tourism_ugc_study/segmentation").glob("*.py")))
    destination.mkdir(mode=0o700)
    hashes = {}
    for source in paths:
        relative = str(source.relative_to(root))
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        data = source.read_bytes()
        target.write_bytes(data)
        target.chmod(0o600)
        hashes[relative] = hashlib.sha256(data).hexdigest()
    return {"directory": str(destination), "files_sha256": hashes,
            "sha256": hashlib.sha256(canonical_json(hashes).encode()).hexdigest(),
            "executed_from_snapshot": False}


def verify_code_snapshot(snapshot: dict) -> None:
    """复验保存副本；非冻结执行的直接调用还复验原工作区，变化即拒绝发布。"""
    root = Path(__file__).resolve().parents[3]
    for relative, digest in snapshot["files_sha256"].items():
        if file_sha256(Path(snapshot["directory"]) / relative) != digest:
            raise ValueError("code_snapshot_changed_during_run")
        if not snapshot["executed_from_snapshot"] and file_sha256(root / relative) != digest:
            raise ValueError("code_changed_during_run")
