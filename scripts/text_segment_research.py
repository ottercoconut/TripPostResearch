#!/usr/bin/env python3
"""只读切分研究keep语料或显式冻结开发包，结果写入新的私有验收运行包。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT / "src"))

from tourism_ugc_study.segmentation.config import load_research_input


def main() -> int:
    """解析路径并打印无原文摘要；失败记录返回2，契约/运行异常返回1。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="配对来源配置；省略沿用仓库研究语料配置")
    parser.add_argument("--frozen-input", type=Path, help="离线开发input-manifest；与显式--config/--members互斥")
    parser.add_argument("--output", type=Path, required=True, help="新的本地运行目录，建议放在results下")
    parser.add_argument("--algorithm-config", type=Path, required=True,
                        help="当前连贯块算法配置；必须显式指定，不再回退到旧切分规则")
    parser.add_argument("--cache-dir", type=Path, help="私有向量缓存；默认输出旁.semantic-vector-cache")
    parser.add_argument("--members", type=Path, help="开发样本manifest；提供后不可作为全量发布包导入")
    parser.add_argument("--load-timeout", type=float, default=300, help="语义模型加载预算秒数，默认300")
    parser.add_argument("--post-timeout", type=float, default=1800, help="语义完整单帖预算秒数，默认1800")
    parser.add_argument("--total-timeout", type=float, default=14400, help="语义整轮预算秒数，默认14400")
    parser.add_argument("--resource-policy", type=Path, help="Linux/CUDA显式资源门JSON")
    parser.add_argument("--code-version", help="显式非空来源描述并跳过Git；实际执行身份始终由code_snapshot.sha256绑定")
    args = parser.parse_args()
    if args.frozen_input and (args.config is not None or args.members is not None):
        parser.error("--frozen-input与显式--config/--members互斥")
    if args.code_version is not None and not args.code_version.strip():
        parser.error("--code-version必须是非空来源描述")
    try:
        from tourism_ugc_study.segmentation.semantic_config import (
            CoherentSemanticConfig, load_semantic_settings, validate_current_frozen_settings,
        )
        settings = load_semantic_settings(args.algorithm_config)
        # 唯一当前配置须匹配完整冻结指纹；在读取正文或加载模型前拒绝漂移。
        if type(settings.algorithm) is not CoherentSemanticConfig:
            raise ValueError("retired_algorithm_for_new_run")
        validate_current_frozen_settings(settings)
        if args.code_version is not None:
            # 仅显式声明才允许无.git输入；该描述不是HEAD证明，实际字节由监督器留存。
            version = args.code_version
        else:
            version = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
                version += "+dirty"
        if args.frozen_input:
            from tourism_ugc_study.segmentation.semantic_development_input import load_frozen_development_input
            source = load_frozen_development_input(args.frozen_input)
        else:
            source = load_research_input(args.config or ROOT / "configs/text-segmentation-research.json")
        progress = lambda done, total: print(f"已切分 {done}/{total}", file=sys.stderr, flush=True)
        from tourism_ugc_study.segmentation.semantic_runtime import run_guarded_semantic_batch, SemanticRuntimeLimits
        from tourism_ugc_study.segmentation.semantic_evaluation import load_development_members
        from tourism_ugc_study.segmentation.semantic_resources import load_linux_resource_policy
        members = load_development_members(args.members, source) if args.members else None
        policy = load_linux_resource_policy(args.resource_policy) if args.resource_policy else None
        result = run_guarded_semantic_batch(source, args.output, code_version=version,
            settings=settings, cache_dir=args.cache_dir,
            post_ids=tuple(members["post_ids"]) if members else None,
            seed=members["seed"] if members else None, members_path=args.members.resolve() if args.members else None,
            limits=SemanticRuntimeLimits(load_seconds=args.load_timeout, post_seconds=args.post_timeout,
                                         total_seconds=args.total_timeout), resource_policy=policy, progress=progress)
    except KeyboardInterrupt:
        print("切分运行中断：KeyboardInterrupt", file=sys.stderr)
        return 130
    except Exception as exc:
        # 模型异常可能包含输入文本，控制台仅显示类型，具体证据在私有attempt中。
        print(f"切分运行失败：{type(exc).__name__}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "manifest": str((args.output / "manifest.json").resolve())}, ensure_ascii=False, indent=2))
    return 2 if result["counts"]["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
