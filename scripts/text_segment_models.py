#!/usr/bin/env python3
"""汇总至少两个不同模型的同冻结样本参数比较包，保留技术统计和待审索引。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT / "src"))

from tourism_ugc_study.segmentation.semantic_model_comparison import ModelComparisonInput, compare_semantic_models


def main() -> int:
    """仅解析显式比较／运行配对及输出路径；校验和发布由核心模块负责。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, nargs=2, action="append", required=True,
                        metavar=("COMPARISON", "RUN"), help="比较目录及其原始运行目录，可重复；至少两个不同模型")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare_semantic_models(tuple(ModelComparisonInput(*pair) for pair in args.comparison), args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"跨模型比较失败：{type(exc).__name__}", file=sys.stderr)
        return 1
    fields = ("status", "comparison_id", "model_count", "config_count", "post_count",
              "actual_embedding_model_calls", "review_status", "semantic_quality_evaluated")
    print(json.dumps({key: result[key] for key in fields}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
