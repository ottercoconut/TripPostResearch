"""本地切分验收运行包的持久化；只写调用方指定的新目录。

JSONL是本阶段核验产物，不是管理端导入协议或片段数据库结构。私有原文只写
新运行目录，最终manifest由批处理层在来源复核后生成；失败不发布半份运行。
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def canonical_json(value: object) -> str:
    """输出稳定UTF-8 JSON的字符串形式，用于逐行落盘及内容摘要。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: object) -> None:
    """独占创建仅当前用户可读写的JSON文件；已有文件直接拒绝覆盖。"""
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


@contextmanager
def new_run_directory(output: Path) -> Iterator[Path]:
    """在私有临时目录构建运行包，正常退出后才移到新目标目录。

    既有输出不覆盖；异常只清理本函数创建的临时目录。调用方负责把所有输入
    校验和输出文件写入放在上下文内，防止失败产物被误认为完成结果。
    """
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"切分输出已存在：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        yield staging
        if output.exists():
            raise FileExistsError(f"切分输出已存在：{output}")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
