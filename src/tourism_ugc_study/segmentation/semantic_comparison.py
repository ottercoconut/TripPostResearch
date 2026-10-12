"""冻结开发单元的缓存重放比较，不加载模型或改写来源。

入口先验证真实运行与冻结来源，再要求初始配置逐字段复现结果。最多九组
仅改变窗口和绝对阈值，缓存缺失／损坏或原始结果不一致直接拒绝发布半包。
私有对照包含全文；四层统计和待审模板不宣称语义质量已经通过。
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

from .artifacts import canonical_json, new_run_directory, write_json
from .batch import runtime_identity
from .config import ResearchInput
from .contracts import EmbeddingError, SemanticDecision, TextUnit
from .embedding import QwenTokenCounter, identity_sha256
from .embedding_cache import EmbeddingCache
from .integrity import WHITE, validate
from .semantic_boundaries import choose_boundaries
from .semantic_config import SemanticConfig, SemanticSettings, parse_semantic_settings
from .semantic_development_input import FrozenDevelopmentInput
from .semantic_evaluation import (validate_counter_settings, validate_development_run,
                                  validate_development_units)
from .semantic_reporting import make_review_template, summarize_comparison_cases
from .source import file_sha256

COMPARISON_KIND = "semantic-development-comparison"
_TUNABLE = {"window_tokens", "absolute_cohesion_threshold"}


def _read_jsonl(path: Path) -> list[dict]:
    """读取已校验摘要的私有逐帖文件，保留原顺序。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _select_configs(baseline: SemanticConfig, configs: tuple[SemanticConfig, ...] | None) -> tuple[SemanticConfig, ...]:
    """限制预先声明的最多九组预算，拒绝单位或其他固定算法参数漂移。"""
    result = configs if configs is not None else tuple(
        replace(baseline, window_tokens=window, absolute_cohesion_threshold=threshold)
        for window in (32, 64, 128) for threshold in (0.45, 0.55, 0.65))
    if not result or len(result) > 9 or any(not isinstance(item, SemanticConfig) for item in result):
        raise ValueError("invalid_comparison_config_budget")
    ids = []
    for item in result:
        if (item.window_tokens not in {32, 64, 128} or item.absolute_cohesion_threshold not in {0.45, 0.55, 0.65}
                or any(value != asdict(baseline)[key] for key, value in asdict(item).items() if key not in _TUNABLE)):
            raise ValueError("comparison_fixed_parameter_changed")
        ids.append(identity_sha256(asdict(item)))
    if len(ids) != len(set(ids)):
        raise ValueError("comparison_duplicate_config")
    return tuple(result)


def _boundary_result(case: dict, vectors, config: SemanticConfig, counter) -> tuple[list[dict] | None, dict, float]:
    """调用纯算法并保留明确逐帖失败，系统性 embedding 错误仍向上终止。"""
    started = time.monotonic()
    try:
        segments, decision = choose_boundaries(case["source_text"], tuple(TextUnit(**unit) for unit in case["units"]),
                                              vectors, config, counter)
        records = [asdict(segment) for segment in segments]
        validate(case["source_text"], records)
    except EmbeddingError:
        raise
    except ValueError as exc:
        records = None
        decision = SemanticDecision("failed", (str(exc),), (), (), ())
    else:
        decision = replace(decision, reason_codes=tuple(dict.fromkeys((*case["diagnostics"], *decision.reason_codes))))
    return records, json.loads(canonical_json(asdict(decision))), time.monotonic() - started


def _case_record(case: dict, segments: list[dict] | None, decision: dict, *, config_id: str,
                 model_hash: str, vector_hash: str | None, cache_hits: int, model_calls: int,
                 elapsed: float | None, counter, targeted: set[int]) -> dict:
    """组装私有新旧对照及可汇总量；计数器仅重计原文段，不请求向量。"""
    units = len(case["units"])
    return {"source_post_id": case["source_post_id"], "source_version": case["source_version"],
            "sample_role": "pressure" if case["source_post_id"] in targeted else "development",
            "config_id": config_id, "source_text": case["source_text"], "original_body": case["original_body"],
            "source_text_sha256": hashlib.sha256(case["source_text"].encode()).hexdigest(),
            "units": case["units"], "status": "completed" if segments is not None else "failed",
            "segments": segments or [], "old_status": case["old_status"], "old_segments": case["old_segments"],
            "old_boundaries": [segment["end"] for segment in case["old_segments"][:-1]],
            "new_boundaries": [segment["end"] for segment in (segments or [])[:-1]],
            "decision": decision, "unit_count": units,
            "embedded_unit_count": units if units >= 2 else 0, "cache_hit_count": cache_hits,
            "model_call_count": model_calls, "model_identity_sha256": model_hash,
            "vector_content_sha256": vector_hash, "boundary_seconds": elapsed,
            "semantic_boundary_count": len(decision["semantic_boundaries"]),
            "length_boundary_count": len(decision["length_boundaries"]),
            "segment_token_counts": [counter.count(segment["raw_text"]) for segment in segments or []],
            "segment_nonspace_counts": [sum(char not in WHITE for char in segment["raw_text"]) for segment in segments or []]}


def _runtime_resources() -> dict:
    """记录本进程 RSS 高水位；这是进程累计值，不能当成本次模型内存。"""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"process_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
            "rss_scope": "replay_process_lifetime_high_water", "model_loaded": False}


def compare_semantic_configs(source: ResearchInput, members: Path, run: Path, output: Path, *,
                             configs: tuple[SemanticConfig, ...] | None = None, cache_dir: Path | None = None,
                             counter=None, resource_report: Path | None = None,
                             frozen_input: FrozenDevelopmentInput | None = None) -> dict:
    """从一个已绑定语义运行与有效缓存重放最多九组开发配置。

    counter 默认只加载本地 tokenizer；测试注入仍核对其模型声明。窗口和阈值
    不影响缓存身份，其余算法参数固定。原配置先重放并逐字段比较 trace /
    segments / 向量摘要；缺缓存或不一致绝不 forward、补算或重抽成员。
    resource_report 可补真实运行的资源观察，必须绑定 run manifest SHA。
    离线run必须显式提供frozen_input，并在本机重新核对完整来源；输出input
    仍是原run实际包身份，候选摘要用此身份重绑，不借用旧全库摘要。
    返回无原文清单，完整正文和新旧对照仅写新建私有运行目录。
    """
    if not isinstance(source, ResearchInput):
        raise ValueError("development_local_research_source_required")
    started = time.monotonic()
    if Path(output).exists():
        raise FileExistsError("comparison_output_exists")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    settings = parse_semantic_settings(manifest.get("config"))
    chosen = _select_configs(settings.algorithm, configs)
    if counter is None:
        counter = QwenTokenCounter(**{key: value for key, value in asdict(settings.embedding).items()
                                   if key not in {"device", "dtype", "batch_size"}}, local_files_only=True)
    validate_counter_settings(counter, settings, model_identity=manifest["model_identity"])
    validation = validate_development_run(run, source, members=members, counter=counter, frozen_input=frozen_input)
    frozen = validate_development_units(members, source, settings=settings, counter=counter)
    records, traces = _read_jsonl(run / "posts.jsonl"), _read_jsonl(run / "semantic-trace.jsonl")
    targeted = set(frozen.manifest["targeted_post_ids"])
    actual_cache = cache_dir if cache_dir is not None else Path(manifest["cache_directory"])
    needs_vectors = any(len(case["units"]) >= 2 for case in frozen.cases)
    cache = EmbeddingCache(actual_cache, manifest["model_identity"], readonly=True) if needs_vectors else None
    vectors_by_post, baseline_rows = {}, []
    baseline_config_id = settings.fingerprint
    for case, record, trace in zip(frozen.cases, records, traces, strict=True):
        units = case["units"]
        vectors = None if len(units) < 2 else cache.read_cached([case["source_text"][unit["start"]:unit["end"]] for unit in units])
        vector_hash = None if vectors is None else hashlib.sha256(vectors.astype("<f4").tobytes(order="C")).hexdigest()
        if vector_hash != trace["summary"].get("vector_content_sha256"):
            raise ValueError("comparison_run_vector_digest_mismatch")
        segments, decision, _ = _boundary_result(case, vectors, settings.algorithm, counter)
        if segments != record["text_segments"] or decision != trace["decision"]:
            raise ValueError("comparison_baseline_reproduction_failed")
        original = record["semantic_diagnostics"]
        counts = {"unit_count": len(units), "candidate_gap_count": max(0, len(units) - 1),
                  "embedded_unit_count": len(units) if len(units) >= 2 else 0,
                  "semantic_boundary_count": len(decision["semantic_boundaries"]),
                  "length_boundary_count": len(decision["length_boundaries"]),
                  "selected_boundary_count": len(decision["semantic_boundaries"]) + len(decision["length_boundaries"])}
        if any(original.get(key) != value for key, value in counts.items()):
            raise ValueError("comparison_run_summary_counts_mismatch")
        vectors_by_post[case["source_post_id"]] = vectors
        baseline_rows.append(_case_record(case, segments, decision, config_id=baseline_config_id,
            model_hash=manifest["model_identity_sha256"], vector_hash=vector_hash,
            cache_hits=original["cache_hit_count"], model_calls=original["model_call_count"],
            elapsed=record.get("boundary_seconds"), counter=counter, targeted=targeted))
        # trace计时覆盖候选、向量与边界；不能挪作纯边界耗时。
        baseline_rows[-1]["processing_seconds"] = trace.get("processing_seconds")
    if any(sum(row[key] for row in baseline_rows) != manifest["counts"][key]
           for key in ("embedded_unit_count", "cache_hit_count", "model_call_count")):
        raise ValueError("comparison_run_aggregate_counts_mismatch")
    if manifest["counts"].get("post_count") != len(baseline_rows):
        raise ValueError("comparison_run_post_count_mismatch")
    run_sha = file_sha256(run / "manifest.json")
    resources = None
    if resource_report is not None:
        resources = json.loads(resource_report.read_text(encoding="utf-8"))
        if resources.get("run_manifest_sha256") != run_sha:
            raise ValueError("comparison_resource_run_mismatch")
    config_specs = [{"config_id": SemanticSettings(item, settings.embedding).fingerprint, "algorithm": asdict(item)} for item in chosen]
    binding = {"run_manifest_sha256": run_sha, "members_manifest_sha256": file_sha256(members),
               "frozen_units_sha256": validation["frozen_units_sha256"], "configs": config_specs,
               "runtime": runtime_identity()}
    comparison_id = identity_sha256(binding)
    with new_run_directory(output) as staging:
        summaries, files = [], {}
        for config, spec in zip(chosen, config_specs, strict=True):
            group_started = time.monotonic()
            rows = []
            for case in frozen.cases:
                vectors = vectors_by_post[case["source_post_id"]]
                segments, decision, elapsed = _boundary_result(case, vectors, config, counter)
                vector_hash = None if vectors is None else hashlib.sha256(vectors.astype("<f4").tobytes(order="C")).hexdigest()
                rows.append(_case_record(case, segments, decision, config_id=spec["config_id"],
                    model_hash=manifest["model_identity_sha256"], vector_hash=vector_hash,
                    cache_hits=len(case["units"]) if vectors is not None else 0, model_calls=0,
                    elapsed=elapsed, counter=counter, targeted=targeted))
            for row, baseline in zip(rows, baseline_rows, strict=True):
                row.update({"baseline_config_id": baseline_config_id, "baseline_segments": baseline["segments"],
                            "baseline_boundaries": baseline["new_boundaries"], "baseline_decision": baseline["decision"]})
            filename = f"cases-{spec['config_id']}.jsonl"
            with (staging / filename).open("x", encoding="utf-8") as stream:
                (staging / filename).chmod(0o600)
                for row in rows:
                    stream.write(canonical_json(row) + "\n")
            review_filename = f"review-{spec['config_id']}.json"
            write_json(staging / review_filename, make_review_template(rows, comparison_id=comparison_id, config_id=spec["config_id"]))
            files[filename], files[review_filename] = file_sha256(staging / filename), file_sha256(staging / review_filename)
            summaries.append({**spec, "statistics": summarize_comparison_cases(rows),
                              "total_replay_seconds": time.monotonic() - group_started,
                              "resources": _runtime_resources(), "review_status": "pending",
                              "review_counts": {"pending": len(rows), "usable": 0, "needs_change": 0, "uncertain": 0}})
        started_at = manifest.get("started_at_utc")
        completed_at = manifest.get("completed_at_utc")
        run_seconds = ((datetime.fromisoformat(completed_at) - datetime.fromisoformat(started_at)).total_seconds()
                       if started_at and completed_at else None)
        report = {"baseline": {"config_id": baseline_config_id, "statistics": summarize_comparison_cases(baseline_rows),
                               "run_elapsed_seconds": run_seconds, "resources": resources,
                               "resource_scope": "whole_original_run",
                               "actual_embedding_model_calls": manifest["counts"]["model_call_count"],
                               "corpus_model_call_count": manifest["counts"]["model_call_count"],
                               "preflight_model_call_count": manifest.get("preflight_model_call_count")},
                  "groups": summaries, "review_status": "pending", "semantic_quality_evaluated": False}
        write_json(staging / "summary.json", report)
        files["summary.json"] = file_sha256(staging / "summary.json")
        # 不给新配置借用旧结论；整个比较包始终以 pending 开始。
        result = {"artifact_kind": COMPARISON_KIND, "status": "completed", "comparison_id": comparison_id,
                  **binding, "input": manifest["input"], "post_ids": frozen.manifest["post_ids"],
                  "development_post_ids": frozen.manifest["development_post_ids"],
                  "targeted_post_ids": frozen.manifest["targeted_post_ids"],
                  "model_identity_sha256": manifest["model_identity_sha256"], "config_count": len(chosen),
                  "post_count": len(frozen.cases), "baseline_reproduction": "passed",
                  "files_sha256": files, "actual_embedding_model_calls": 0,
                  "replay_model_call_count": 0, "cache_read_unit_count": cache.cache_hit_count if cache else 0,
                  "total_comparison_seconds": time.monotonic() - started, "resources": _runtime_resources(),
                  "run_validation": validation, "review_status": "pending", "semantic_quality_evaluated": False,
                  "source_database_write_count": 0, "membership_database_write_count": 0}
        if hasattr(counter, "verify_assets"):
            counter.verify_assets()
        repeated = validate_development_run(run, source, members=members, counter=counter, frozen_input=frozen_input)
        if repeated != validation:
            raise ValueError("comparison_run_changed")
        if file_sha256(run / "manifest.json") != run_sha:
            raise ValueError("comparison_run_changed")
        write_json(staging / "manifest.json", result)
    return result
