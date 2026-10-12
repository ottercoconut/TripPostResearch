#!/usr/bin/env python3
"""从冻结开发成员和真实语义运行的缓存重放最多九组窗口／阈值。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT / "src"))

from tourism_ugc_study.segmentation.config import load_research_input
from tourism_ugc_study.segmentation.semantic_comparison import compare_semantic_configs
from tourism_ugc_study.segmentation.semantic_config import load_semantic_settings, parse_semantic_settings
from tourism_ugc_study.segmentation.semantic_development_input import load_frozen_development_input


def main() -> int:
    """只解析参数与输出无原文摘要；所有核验和持久化留在核心模块。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/text-segmentation-research.json")
    parser.add_argument("--members", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--resource-report", type=Path)
    parser.add_argument("--frozen-input", type=Path, help="离线run原input-manifest；比较仍核对本地完整来源")
    parser.add_argument("--algorithm-config", type=Path, action="append", help="显式比较组，可重复最多九次；省略生成3×3组")
    args = parser.parse_args()
    try:
        configs = None
        if args.algorithm_config:
            baseline = parse_semantic_settings(json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))["config"])
            requested = tuple(load_semantic_settings(path) for path in args.algorithm_config)
            if any(settings.embedding != baseline.embedding for settings in requested):
                raise ValueError("comparison_embedding_config_changed")
            configs = tuple(settings.algorithm for settings in requested)
        result = compare_semantic_configs(load_research_input(args.config), args.members, args.run, args.output,
            configs=configs, cache_dir=args.cache_dir, resource_report=args.resource_report,
            frozen_input=load_frozen_development_input(args.frozen_input) if args.frozen_input else None)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"语义比较失败：{type(exc).__name__}", file=sys.stderr)
        return 1
    fields = ("status", "comparison_id", "post_count", "config_count", "baseline_reproduction",
              "actual_embedding_model_calls", "review_status", "total_comparison_seconds")
    print(json.dumps({key: result[key] for key in fields}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
