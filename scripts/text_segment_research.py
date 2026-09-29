#!/usr/bin/env python3
"""只读切分管理端研究语料的全部keep成员，结果写入新的本地验收运行包。"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT / "src"))

from tourism_ugc_study.segmentation.batch import run_research_segmentation
from tourism_ugc_study.segmentation.config import load_research_input


def main() -> int:
    """解析路径并打印无原文摘要；失败记录返回2，契约/运行异常返回1。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/text-segmentation-research.json")
    parser.add_argument("--output", type=Path, required=True, help="新的本地运行目录，建议放在results下")
    args = parser.parse_args()
    try:
        version = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
            version += "+dirty"
        result = run_research_segmentation(
            load_research_input(args.config), args.output, code_version=version,
            progress=lambda done, total: print(f"已切分 {done}/{total}", file=sys.stderr, flush=True),
        )
    except (OSError, ValueError, sqlite3.Error, subprocess.CalledProcessError) as exc:
        print(f"切分运行失败：{exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "manifest": str((args.output / "manifest.json").resolve())}, ensure_ascii=False, indent=2))
    return 2 if result["counts"]["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
