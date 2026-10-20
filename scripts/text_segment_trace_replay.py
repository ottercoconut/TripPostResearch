#!/usr/bin/env python3
"""开发用：以已完成正式运行的 trace 评分重放当前切分规则，与旧结果逐帖对比。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT / "src"))

from tourism_ugc_study.segmentation.semantic_config import load_semantic_settings
from tourism_ugc_study.segmentation.semantic_trace_replay import replay_run


def _code_version() -> str:
    """记录 Git 提交及是否有未提交改动；无 .git 时如实记 unknown。"""
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("+dirty" if dirty else "")


def main() -> int:
    """只解析参数并打印汇总；读取、重放和写出都在核心模块。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="已完成正式运行目录（含posts.jsonl与semantic-trace.jsonl）")
    parser.add_argument("--algorithm-config", type=Path, default=ROOT / "configs/text-segmentation-semantic-8b-coherent.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True, help="含同revision tokenizer的HF hub目录")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    try:
        settings = load_semantic_settings(args.algorithm_config)
        counter_kwargs = {key: value for key, value in asdict(settings.embedding).items()
                          if key not in {"device", "dtype", "batch_size"}}
        counter_kwargs.update(local_files_only=True, cache_dir=str(args.hf_cache))
        summary = replay_run(args.run, args.output, config=settings.algorithm, counter_kwargs=counter_kwargs,
                             code_version=_code_version(), config_sha256=settings.fingerprint, workers=args.workers)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"trace重放失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
