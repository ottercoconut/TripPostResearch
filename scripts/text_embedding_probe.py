#!/usr/bin/env python3
"""在资源监督下验证本地 Qwen：默认 Mac smoke 或显式 Linux/CUDA 合成预检。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT / "src"))

from tourism_ugc_study.segmentation.embedding_probe import run_smoke
from tourism_ugc_study.segmentation.semantic_config import load_semantic_settings
from tourism_ugc_study.segmentation.semantic_resources import load_linux_resource_policy


def main() -> int:
    """指定新私有输出目录；任何未完成测试均返回非零退出码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--algorithm-config", type=Path, help="CUDA预检的显式模型/精度/revision声明，复用语义配置")
    parser.add_argument("--resource-policy", type=Path, help="Linux/CUDA四字段资源门JSON，必须与显式语义配置一起使用")
    parser.add_argument("--numerical-tolerances", type=json.loads,
                        help='内联JSON：{"repeat":数值,"batch":数值,"norm":数值}；低精度必需，FP32固定门不可放宽')
    args = parser.parse_args()
    if not args.algorithm_config and (args.resource_policy or args.numerical_tolerances is not None):
        parser.error("--resource-policy和--numerical-tolerances需要--algorithm-config")
    try:
        if args.algorithm_config:
            config = load_semantic_settings(args.algorithm_config)
            policy = load_linux_resource_policy(args.resource_policy) if args.resource_policy else None
            result = run_smoke(args.output, embedding=config.embedding, resource_policy=policy,
                               numerical_tolerances=args.numerical_tolerances)
        else:
            result = run_smoke(args.output)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)},
                         ensure_ascii=False))
        return 1
    print(json.dumps({"status": result["status"], "error": result.get("error"),
                      "evidence": str(args.output.resolve() / "summary.json")}, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
