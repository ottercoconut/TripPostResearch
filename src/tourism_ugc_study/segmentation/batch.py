"""研究语料切分调度：组合只读来源、纯函数引擎与独立运行包。

不回写源库，不修改管理端任务，不替换历史pilot的seg_id。异常输入逐条显式
记录；运行包带失败数，不把部分成功冒充全部通过。无随机抽样，固定ID顺序。
"""

from __future__ import annotations

import hashlib
import os
import platform
import unicodedata
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import regex

from .api import segment_body
from .artifacts import canonical_json, new_run_directory, write_json
from .config import ResearchInput
from .engine import IMPLEMENTATION_VERSION, RULE_VERSION
from .projection import PROJECTION_ID
from .source import file_sha256, open_research_corpus

FROZEN_ENGINE_SHA256 = "6a4c66ed1453e79c0f6af95db5504abcac8fdc50a14504223163aab643ce6f74"


def runtime_identity() -> dict[str, object]:
    """记录实际Python、Unicode、regex及独立模块字节身份，不加载机器学习库。"""
    root = Path(__file__).parent
    return {
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "unicode": unicodedata.unidata_version, "regex": regex.__version__,
        "module_sha256": {p.name: file_sha256(p) for p in sorted(root.glob("*.py"))},
        "frozen_engine_sha256": FROZEN_ENGINE_SHA256,
    }


def run_research_segmentation(
    config: ResearchInput,
    output: Path,
    *,
    code_version: str,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """将所有keep成员只读切分到新的本地验收运行包，返回无原文摘要。

    config显式绑定配对来源；output必须不存在。每条结果保存源身份、原body、
    可读投影、片段与诊断，manifest绑定输入、输出及代码哈希。输入无效保留为
    failed记录并返回completed_with_failures；库/契约错误抛异常且不发布产物。
    固定运行时避免同一规则在不同Unicode依赖下静默产生不同结果。
    """
    runtime = runtime_identity()
    if (runtime["implementation"] != "CPython" or runtime["python"] != "3.13.5"
            or runtime["regex"] != "2026.7.19"):
        raise ValueError("切分验收要求CPython 3.13.5与regex 2026.7.19")
    started = datetime.now(timezone.utc).isoformat()
    counts: Counter[str] = Counter()
    issues: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    inputs_digest = hashlib.sha256()
    outputs_digest = hashlib.sha256()
    with new_run_directory(output) as staging:
        with open_research_corpus(config) as corpus:
            identity = corpus.identity
            records = staging / "posts.jsonl"
            with records.open("x", encoding="utf-8") as stream:
                os.chmod(records, 0o600)
                for post in corpus.posts:
                    result = segment_body(post.body)
                    inputs_digest.update((canonical_json([
                        post.source_post_id, post.source_version, result.source_body_sha256,
                    ]) + "\n").encode("utf-8"))
                    row = {
                        "source_post_id": post.source_post_id, "source_version": post.source_version,
                        "source_snapshot_sha256": identity["source_snapshot_sha256"],
                        "original_body": post.body, **asdict(result),
                    }
                    line = canonical_json(row) + "\n"
                    stream.write(line)
                    outputs_digest.update(line.encode("utf-8"))
                    counts["post_count"] += 1
                    counts[result.status + "_count"] += 1
                    counts["segment_count"] += len(result.text_segments or ())
                    counts["empty_count"] += int(result.status == "completed" and not result.text_segments)
                    counts["null_body_count"] += int(result.body_was_null)
                    counts["structured_body_count"] += int(result.format_id == "quill_delta_json")
                    counts["posts_with_issues"] += bool(result.issues)
                    issues.update(item.code for item in result.issues)
                    if result.status == "failed":
                        failures[result.reason_code] += 1
                    if progress and counts["post_count"] % 250 == 0:
                        progress(counts["post_count"], identity["record_count"])
            if counts["post_count"] != identity["record_count"]:
                raise ValueError("切分输出成员数与输入不一致")
        if runtime_identity() != runtime:
            raise ValueError("切分模块或运行时在执行期间发生变化")
        manifest = {
            "artifact_kind": "objective-text-segmentation-run",
            "status": "completed_with_failures" if counts["failed_count"] else "completed",
            "rule_version": RULE_VERSION, "projection_id": PROJECTION_ID,
            "implementation_version": IMPLEMENTATION_VERSION,
            "code_version": code_version, "runtime": runtime, "input": identity,
            "input_paths": {k: str(v) for k, v in asdict(config).items()},
            "seed": None, "sampling": "all_keep_ordered_by_source_post_id",
            "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "counts": {key: counts[key] for key in (
                "post_count", "completed_count", "failed_count", "segment_count", "empty_count",
                "null_body_count", "structured_body_count", "posts_with_issues",
            )},
            "issue_counts": dict(sorted(issues.items())), "failure_counts": dict(sorted(failures.items())),
            "ordered_input_sha256": inputs_digest.hexdigest(),
            "posts_filename": "posts.jsonl", "posts_sha256": outputs_digest.hexdigest(),
            "source_database_write_count": 0, "membership_database_write_count": 0,
        }
        if file_sha256(records) != manifest["posts_sha256"]:
            raise ValueError("切分输出文件摘要不一致")
        write_json(staging / "manifest.json", manifest)
    return manifest
