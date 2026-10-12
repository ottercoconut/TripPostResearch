#!/usr/bin/env python3
"""准备固定开发样本、导出离线输入或在本地核验回传运行，不加载embedding。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT / "src"))

from tourism_ugc_study.segmentation.config import load_research_input
from tourism_ugc_study.segmentation.semantic_config import load_semantic_settings
from tourism_ugc_study.segmentation.semantic_evaluation import (
    export_frozen_development_input, prepare_development_sample,
    verify_development_sample, verify_frozen_development_run,
)


def main() -> int:
    """只负责参数和摘要；抽样、持久化与单元规则均在核心模块。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/text-segmentation-research.json")
    parser.add_argument("--algorithm-config", type=Path, help="抽样/单位核验配置；省略沿用仓库语义配置")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--members", type=Path, help="已有冻结开发manifest；提供后仅核验同一候选，不重新抽样")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--export-frozen-input", action="store_true", help="核验本地来源并原字节复制固定开发输入")
    modes.add_argument("--verify-run", type=Path, help="回传离线运行；重新核对本地来源并保存核验记录")
    parser.add_argument("--frozen-input", type=Path, help="回传核验的原input-manifest，仅与--verify-run一起使用")
    args = parser.parse_args()
    if (args.export_frozen_input or args.verify_run) and not args.members:
        parser.error("离线导出和回传核验需要--members")
    if bool(args.verify_run) != bool(args.frozen_input):
        parser.error("--verify-run和--frozen-input必须配对提供")
    if (args.export_frozen_input or args.verify_run) and args.algorithm_config:
        parser.error("导出沿用冻结样本，回传核验沿用run配置，不接受--algorithm-config")
    try:
        source = load_research_input(args.config)
        if args.export_frozen_input:
            value = export_frozen_development_input(source, args.members, args.output)
        elif args.verify_run:
            value = verify_frozen_development_run(source, args.verify_run, args.output,
                frozen_input=args.frozen_input, members=args.members)
        else:
            settings = load_semantic_settings(args.algorithm_config or ROOT / "configs/text-segmentation-semantic-8b-coherent.json")
            value = (verify_development_sample(source, args.members, args.output, settings=settings) if args.members else
                     prepare_development_sample(source, args.output, settings=settings,
                        progress=lambda done, total: print(f"已核查候选单元 {done}/{total}", file=sys.stderr, flush=True)))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"开发单位核验失败：{type(exc).__name__}", file=sys.stderr)
        return 1
    fields = ("status", "counts", "bucket_counts", "max_unitization_seconds", "frozen_units_sha256", "validation",
              "actual_embedding_model_calls", "semantic_quality_evaluated")
    print(json.dumps({key: value[key] for key in fields if key in value}, ensure_ascii=False, indent=2))
    return 0 if value["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
