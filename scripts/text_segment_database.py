#!/usr/bin/env python3
"""片段单表导入、全量校验和随机抽查入口；不写采集或清洗数据库。"""

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

from tourism_ugc_study.segmentation.config import load_research_input
from tourism_ugc_study.segmentation.database_audit import sample_segment_bundle, verify_segment_bundle
from tourism_ugc_study.segmentation.importer import import_segmentation_run


def main() -> int:
    """解析命令并调用核心模块；打印数量/路径，错误返回1而不输出原文。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("import")
    ingest.add_argument("--run", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)
    ingest.add_argument("--config", type=Path, default=ROOT / "configs/text-segmentation-research.json")
    verify = commands.add_parser("verify")
    verify.add_argument("--bundle", type=Path, required=True)
    sample = commands.add_parser("sample")
    sample.add_argument("--bundle", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--count", type=int, default=50)
    sample.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    try:
        if args.command == "import":
            version = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
                version += "+dirty"
            result = import_segmentation_run(args.run, args.output, load_research_input(args.config), code_version=version)
        elif args.command == "verify":
            result = verify_segment_bundle(args.bundle)
        else:
            result = sample_segment_bundle(args.bundle, args.output, count=args.count, seed=args.seed)
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, subprocess.CalledProcessError) as exc:
        print(f"片段数据库操作失败：{exc}", file=sys.stderr)
        return 1
    fields = ("status", "batch_id", "post_count", "segment_count", "import_status", "verification",
              "sample_count", "seed", "review_status", "database_sha256")
    print(json.dumps({k: result[k] for k in fields if k in result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
