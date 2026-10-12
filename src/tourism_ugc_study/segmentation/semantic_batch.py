"""带模型谱系和文件缓存的同步语义批处理，来源只读且按post排序。

开发样本与全量运行明确隔离；模型系统故障结束attempt，逐帖领域错误保留为
failed记录。失败诊断在临时运行包之外保存，成功向量可供显式重跑复用。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import time
from typing import Callable

from .artifacts import canonical_json, new_run_directory, write_json
from .artifacts import runtime_identity
from .config import ResearchInput
from .embedding import QwenEmbeddingProvider, identity_sha256, validate_vectors
from .embedding_cache import EmbeddingCache
from .projection import PROJECTION_ID
from .semantic_api import segment_body_semantic
from .semantic_config import (RUN_KIND, SemanticSettings, semantic_rule_identity,
                              validate_identity_settings)
from .semantic_development_input import (
    FrozenDevelopmentInput, development_units_sha256, frozen_development_parameters,
    open_frozen_development_corpus,
)
from .semantic_provenance import freeze_code, verify_code_snapshot
from .source import file_sha256, open_research_corpus


def _now() -> str:
    """返回带时区UTC时点，仅用于运行谱系，不影响边界计算。"""
    return datetime.now(timezone.utc).isoformat()


def run_semantic_batch(
    config: ResearchInput | FrozenDevelopmentInput, output: Path, *, settings: SemanticSettings, code_version: str,
    cache_dir: Path | None = None, provider=None, post_ids: tuple[int, ...] | None = None,
    seed: int | None = None, progress: Callable[[int, int], None] | None = None,
    state: Callable[[dict], None] | None = None, code_snapshot: dict | None = None,
    resource_supervision: dict | None = None,
    members_path: Path | None = None,
) -> dict:
    """生成全量或显式开发成员运行包，不切换Admin或写源库。

    output必须新建；post_ids为None才是全量，显式清单即使覆盖全部成员仍是
    development。provider用于测试注入，正式默认从本地已准备权重加载。
    异常只把类型写attempt（不打印可能含原文的异常），成功输出包含完整私有证据。
    state只接收阶段、当前post和计数，用于独立监督者观察整帖停滞，不传正文。
    code_snapshot由监督者绑定执行副本；直接调用则在运行包内保存当前源码并复验。
    FrozenDevelopmentInput自动固定包内全部成员，始终development；只读取案例，
    不访问源库。其逐帖source_snapshot_sha256明确继承parent来源引用。
    """
    rule_version, implementation_version = semantic_rule_identity(settings.algorithm)
    offline = isinstance(config, FrozenDevelopmentInput)
    if offline:
        post_ids, seed, members_path = frozen_development_parameters(config, post_ids, seed, members_path)
    output = Path(output).resolve()
    cache_dir = Path(cache_dir).resolve() if cache_dir else output.parent / ".semantic-vector-cache"
    if cache_dir == output or cache_dir.is_relative_to(output):
        raise ValueError("cache_must_be_outside_run_output")
    if output.exists():
        raise FileExistsError(f"切分输出已存在：{output}")
    if post_ids is not None and (not post_ids or len(set(post_ids)) != len(post_ids)
            or any(type(i) is not int or i < 1 for i in post_ids)):
        raise ValueError("invalid_development_members")
    runtime = runtime_identity()
    if (runtime["implementation"] != "CPython" or runtime["python"] != "3.13.5"
            or runtime["regex"] != "2026.7.19"):
        raise ValueError("切分验收要求CPython 3.13.5与regex 2026.7.19")
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt = output.with_name(output.name + ".attempt.json")
    started = _now()
    attempt_data = {"status": "running", "started_at_utc": started,
                    "config_sha256": settings.fingerprint, "code_version": code_version}
    write_json(attempt, attempt_data)
    counts: Counter = Counter()
    decisions: Counter = Counter()
    failures: Counter = Counter()
    issues: Counter = Counter()
    ordered_input = hashlib.sha256()
    output_hash = hashlib.sha256()
    def phase(name: str, *, post_id: int | None = None) -> None:
        """即时保存无原文阶段，外部监督者即使终止worker仍有最后可信状态。"""
        attempt_data["phase"] = name
        attempt_data["counts"] = dict(counts)
        if post_id is None:
            attempt_data.pop("active_source_post_id", None)
        else:
            attempt_data["active_source_post_id"] = post_id
        temporary = attempt.with_name(attempt.name + ".pending")
        write_json(temporary, attempt_data)
        os.replace(temporary, attempt)
        if state:
            state(dict(attempt_data))
    try:
        with new_run_directory(output) as staging:
            snapshot = code_snapshot or freeze_code(staging / "code-snapshot")
            phase("model_loading")
            if provider is None:
                provider = QwenEmbeddingProvider(**asdict(settings.embedding), local_files_only=True)
            # 实际模型预检绕过cache，与真实语料统计分开。
            identity = provider.identity
            validate_identity_settings(identity, settings.embedding)
            model_hash = identity_sha256(identity)
            phase("model_preflight")
            preflight_calls_before = provider.model_call_count
            probe = provider.encode(["海边散步，海风很轻。", "公交车站就在路口。"])
            preflight_model_call_count = provider.model_call_count - preflight_calls_before
            validate_vectors(probe, count=2, dimension=identity["dimension"])
            if members_path is not None:
                from .semantic_evaluation import validate_development_units
                frozen = validate_development_units(members_path, config, settings=settings, counter=provider)
                if post_ids is None or set(post_ids) != set(frozen.manifest["post_ids"]):
                    raise ValueError("development_members_identity_mismatch")
            cache = EmbeddingCache(cache_dir, identity)
            phase("corpus_processing")
            context = open_frozen_development_corpus(config) if offline else open_research_corpus(config)
            with context as corpus:
                source_identity = corpus.identity
                parent_identity = source_identity["parent_source_identity"] if offline else source_identity
                expected = len(post_ids) if post_ids is not None else source_identity["record_count"]
                attempt_data["expected_post_count"] = expected
                selected = set(post_ids) if post_ids is not None else None
                seen = []
                records = staging / "posts.jsonl"
                trace = staging / "semantic-trace.jsonl"
                with records.open("x", encoding="utf-8") as stream, trace.open("x", encoding="utf-8") as traces:
                    records.chmod(0o600)
                    trace.chmod(0o600)
                    for post in corpus.posts:
                        if selected is not None and post.source_post_id not in selected:
                            continue
                        phase("corpus_processing", post_id=post.source_post_id)
                        processing_started = time.monotonic()
                        result = segment_body_semantic(post.body, provider=provider, token_counter=provider,
                                                       config=settings.algorithm, cache=cache)
                        processing_seconds = time.monotonic() - processing_started
                        row = {"source_post_id": post.source_post_id, "source_version": post.source_version,
                               "source_snapshot_sha256": parent_identity["source_snapshot_sha256"],
                               "original_body": post.body, **asdict(result.result),
                               "semantic_diagnostics": result.summary()}
                        if offline:
                            row["source_snapshot_identity_scope"] = "parent_source_reference"
                        line = canonical_json(row) + "\n"
                        stream.write(line)
                        output_hash.update(line.encode())
                        trace_row = {"source_post_id": post.source_post_id,
                            "source_text_sha256": result.result.source_text_sha256,
                            "units": [asdict(u) for u in result.units],
                            "decision": asdict(result.decision), "summary": result.summary(),
                            "processing_seconds": processing_seconds}
                        # 完整解释保存在trace中，不新增数据库列。
                        if result.meaning_trace is not None:
                            trace_row["meaning"] = asdict(result.meaning_trace)
                        traces.write(canonical_json(trace_row) + "\n")
                        ordered_input.update((canonical_json([post.source_post_id, post.source_version,
                                                             result.result.source_body_sha256]) + "\n").encode())
                        seen.append(post.source_post_id)
                        counts["post_count"] += 1
                        counts[result.result.status + "_count"] += 1
                        counts["segment_count"] += len(result.result.text_segments or ())
                        counts["empty_count"] += int(result.result.status == "completed" and not result.result.text_segments)
                        counts["null_body_count"] += result.result.body_was_null
                        counts["structured_body_count"] += result.result.format_id == "quill_delta_json"
                        counts["posts_with_issues"] += bool(result.result.issues)
                        for key in ("embedded_unit_count", "cache_hit_count", "model_call_count"):
                            counts[key] += result.summary()[key]
                        decisions[result.decision.status] += 1
                        issues.update(i.code for i in result.result.issues)
                        if result.result.status == "failed":
                            failures.update(result.decision.reason_codes)
                        phase("corpus_processing", post_id=post.source_post_id)
                        if progress:
                            progress(counts["post_count"], expected)
                if counts["post_count"] != expected or (selected is not None and set(seen) != selected):
                    raise ValueError("development_members_not_found")
            if runtime_identity() != runtime or identity_sha256(provider.identity) != model_hash:
                raise ValueError("runtime_or_model_changed_during_run")
            if hasattr(provider, "verify_assets"):
                provider.verify_assets()
            phase("final_validation")
            verify_code_snapshot(snapshot)
            if code_snapshot is None:
                # 此目录随staging一起rename，manifest不能留下已消失的临时绝对路径。
                snapshot = dict(snapshot, directory=str(output / "code-snapshot"))
            manifest = {
                "artifact_kind": RUN_KIND, "scope": "development" if post_ids is not None else "full_corpus",
                "status": "completed_with_failures" if counts["failed_count"] else "completed",
                "rule_version": rule_version, "implementation_version": implementation_version,
                "projection_id": PROJECTION_ID, "code_version": code_version, "runtime": runtime,
                "code_snapshot": snapshot, "resource_supervision": resource_supervision,
                "input": source_identity, "input_paths": {k: str(v) for k, v in asdict(config).items()},
                "config": asdict(settings), "config_sha256": settings.fingerprint,
                "model_identity": identity, "model_identity_sha256": model_hash,
                "preflight_model_call_count": preflight_model_call_count,
                "selected_post_ids": seen if post_ids is not None else None,
                "seed": seed, "sampling": "explicit_development_members" if post_ids is not None else "all_keep_ordered_by_source_post_id",
                "started_at_utc": started, "completed_at_utc": _now(),
                "counts": {k: counts[k] for k in ("post_count", "completed_count", "failed_count", "segment_count",
                    "empty_count", "null_body_count", "structured_body_count", "posts_with_issues",
                    "embedded_unit_count", "cache_hit_count", "model_call_count")},
                "decision_counts": dict(sorted(decisions.items())), "failure_counts": dict(sorted(failures.items())),
                "issue_counts": dict(sorted(issues.items())), "ordered_input_sha256": ordered_input.hexdigest(),
                "posts_filename": "posts.jsonl", "posts_sha256": output_hash.hexdigest(),
                "trace_filename": trace.name, "trace_sha256": file_sha256(trace),
                "cache_directory": str(Path(cache_dir).resolve()) if cache_dir else None,
                "source_database_write_count": 0, "membership_database_write_count": 0,
            }
            if offline:
                manifest.update(input_verification_scope="frozen_files_cases_and_tokenizer_units",
                    frozen_units_sha256=development_units_sha256(source_identity, post_ids, frozen.cases))
            if file_sha256(records) != manifest["posts_sha256"]:
                raise ValueError("semantic_records_digest_mismatch")
            write_json(staging / "manifest.json", manifest)
            if members_path is not None:
                from .semantic_evaluation import validate_development_run
                validate_development_run(staging, config, members=members_path, counter=provider)
        attempt_data.update(status=manifest["status"], counts=manifest["counts"], completed_at_utc=_now())
        attempt_data["phase"] = "complete"
        attempt_data.pop("active_source_post_id", None)
        return manifest
    except BaseException as exc:
        attempt_data.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                            error_type=type(exc).__name__, counts=dict(counts), completed_at_utc=_now())
        # 仅已知的本层稳定原因可进入公开摘要，第三方异常可能夹带原文。
        safe_codes = {"embedding_identity_settings_mismatch", "development_members_not_found",
                      "runtime_or_model_changed_during_run", "semantic_records_digest_mismatch",
                      "code_snapshot_changed_during_run", "code_changed_during_run",
                      "development_members_identity_mismatch", "frozen_development_input_changed",
                      "frozen_development_input_binding_mismatch", "development_cases_digest_mismatch",
                      "development_cases_changed", "development_manifest_changed"}
        if str(exc) in safe_codes:
            attempt_data["error_code"] = str(exc)
        raise
    finally:
        # 只替换本次独占创建的attempt，绝不把失败摘要写入旧已验收包。
        temporary = attempt.with_name(attempt.name + ".pending")
        write_json(temporary, attempt_data)
        os.replace(temporary, attempt)
