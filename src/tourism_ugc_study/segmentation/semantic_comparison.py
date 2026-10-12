"""冻结开发单元的缓存重放比较，不加载模型或改写来源。

入口先验证真实运行与冻结来源，再要求初始配置复现决策、跨度与向量字节。
仅gap的两个float评分允许已声明的绝对末位差并记录实际漂移；其余字段严格
核对类型与值。最多九组仅改变窗口和绝对阈值，不一致直接拒绝发布半包。
私有对照包含全文；四层统计和待审模板不宣称语义质量已经通过。
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import math
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
from .semantic_dispatch import choose_semantic_boundaries
from .semantic_config import (CoherentSemanticConfig, SemanticConfig, SemanticSettings,
                              StructuredSemanticConfig, parse_semantic_settings)
from .semantic_development_input import FrozenDevelopmentInput
from .semantic_evaluation import (validate_counter_settings, validate_development_run,
                                  validate_development_units)
from .semantic_reporting import make_review_template, summarize_comparison_cases
from .source import file_sha256

COMPARISON_KIND = "semantic-development-comparison"
_TUNABLE = {"window_tokens", "absolute_cohesion_threshold"}
_REPLAY_ABSOLUTE_TOLERANCE = 1e-12


def _read_jsonl(path: Path) -> list[dict]:
    """读取已校验摘要的私有逐帖文件，保留原顺序。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _strict_json_equal(left, right) -> bool:
    """递归核对JSON的类型和值；拒绝Python默认把bool/int或int/float视为相等。

    dict键集合及list顺序须完全相同；本函数不提供任何数值容差，用于正文
    跨度和决策中不允许漂移的字段。非有限float也不能作为精确复现证据。
    """
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_strict_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_strict_json_equal(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, float) and not (math.isfinite(left) and math.isfinite(right)):
        return False
    return left == right


def _baseline_decision_replay(original: dict, replayed: dict) -> dict:
    """核对原trace决策并返回末位差证据；失败统一拒绝基线发布。

    只有gaps中cohesion/depth的有限float允许绝对差不超过1e-12，不用相对
    容差。键、类型、候选、选点、原因、边界及顺序仍须严格相同。本函数
    不改输入；通过后基线继续使用原trace，候选保留本机真实计算值。
    """
    failure = "comparison_baseline_reproduction_failed"
    if (type(original) is not dict or type(replayed) is not dict or original.keys() != replayed.keys()
            or "gaps" not in original or type(original["gaps"]) is not list
            or type(replayed["gaps"]) is not list or len(original["gaps"]) != len(replayed["gaps"])
            or not all(_strict_json_equal(original[key], replayed[key]) for key in original if key != "gaps")):
        raise ValueError(failure)
    maxima = {"cohesion": 0.0, "depth": 0.0}
    changed = 0
    for before, after in zip(original["gaps"], replayed["gaps"], strict=True):
        if (type(before) is not dict or type(after) is not dict or before.keys() != after.keys()
                or not maxima.keys() <= before.keys()
                or not all(_strict_json_equal(before[key], after[key]) for key in before if key not in maxima)):
            raise ValueError(failure)
        for field in maxima:
            left, right = before[field], after[field]
            if type(left) is not float or type(right) is not float or not (math.isfinite(left) and math.isfinite(right)):
                raise ValueError(failure)
            delta = abs(left - right)
            if delta > _REPLAY_ABSOLUTE_TOLERANCE:
                raise ValueError(failure)
            maxima[field] = max(maxima[field], delta)
            changed += left != right
    return {"decision_exact": changed == 0, "changed_score_count": changed,
            "cohesion_max_absolute_delta": maxima["cohesion"], "depth_max_absolute_delta": maxima["depth"],
            "max_absolute_delta": max(maxima.values())}


def _select_configs(baseline: SemanticConfig, configs: tuple[SemanticConfig, ...] | None) -> tuple[SemanticConfig, ...]:
    """限制预先声明的最多九组预算，拒绝单位或其他固定算法参数漂移。"""
    result = configs if configs is not None else tuple(
        replace(baseline, window_tokens=window, absolute_cohesion_threshold=threshold)
        for window in (32, 64, 128) for threshold in (0.45, 0.55, 0.65))
    if not result or len(result) > 9 or any(not isinstance(item, SemanticConfig) for item in result):
        raise ValueError("invalid_comparison_config_budget")
    ids = []
    for item in result:
        if (asdict(item).keys() != asdict(baseline).keys()
                or item.window_tokens not in {32, 64, 128} or item.absolute_cohesion_threshold not in {0.45, 0.55, 0.65}
                or any(value != asdict(baseline)[key] for key, value in asdict(item).items() if key not in _TUNABLE)):
            raise ValueError("comparison_fixed_parameter_changed")
        ids.append(identity_sha256(asdict(item)))
    if len(ids) != len(set(ids)):
        raise ValueError("comparison_duplicate_config")
    return tuple(result)


def _boundary_result(case: dict, vectors, config: SemanticConfig, counter) -> tuple[list[dict] | None, dict, float]:
    """保留旧重放的三项结果接口；新规则使用包含结构证据的完整计算结果。"""
    segments, decision, elapsed, _ = _boundary_calculation(case, vectors, config, counter)
    return segments, decision, elapsed


def _boundary_calculation(case: dict, vectors, config: SemanticConfig, counter) -> tuple[list[dict] | None, dict, float, dict | None]:
    """执行纯边界规则并保留独立来源证据；系统性embedding错误向上终止。

    领域失败无部分片段或结构trace，行为与运行API一致。decision并入冻结
    单元诊断，structure或meaning的initial_decision仍保持纯评分层原始身份。
    """
    started = time.monotonic()
    structure = None
    try:
        calculated = choose_semantic_boundaries(case["source_text"], tuple(TextUnit(**unit) for unit in case["units"]),
                                               vectors, config, counter)
        segments, decision = calculated.segments, calculated.decision
        records = [asdict(segment) for segment in segments]
        validate(case["source_text"], records)
        if calculated.structure_trace is not None:
            structure = json.loads(canonical_json(asdict(calculated.structure_trace)))
        if calculated.meaning_trace is not None:
            structure = json.loads(canonical_json(asdict(calculated.meaning_trace)))
    except EmbeddingError:
        raise
    except ValueError as exc:
        records = None
        structure = None
        decision = SemanticDecision("failed", (str(exc),), (), (), ())
    else:
        decision = replace(decision, reason_codes=tuple(dict.fromkeys((*case["diagnostics"], *decision.reason_codes))))
    return records, json.loads(canonical_json(asdict(decision))), time.monotonic() - started, structure


def _structure_replay(original: object, replayed: dict | None) -> dict:
    """复验0.2结构证据，不允许重算外层SHA掩盖缺失或伪造的解释。

    只有初始决策中的两种评分适用既有末位容差；结构跨度、原因、策略、
    超限退让及其token数均要求严格类型和值一致。失败路径两侧都须为None。
    """
    return _evidence_replay(original, replayed, kind="structure")


def _evidence_replay(original: object, replayed: dict | None, *, kind: str) -> dict:
    """复验独立解释的全部非评分字段，不能用外层SHA替代真实算法重放。

    历史structure和当前meaning分别使用独立键及错误身份；只有初始评分
    允许既定末位容差，边界来源、保护块和逐gap理由均要求严格相同。
    """
    if original is None and replayed is None:
        return {f"{kind}_exact": True}
    if (type(original) is not dict or type(replayed) is not dict
            or original.keys() != replayed.keys() or "initial_decision" not in replayed
            or not all(_strict_json_equal(original[key], replayed[key])
                       for key in replayed if key != "initial_decision")):
        raise ValueError(f"comparison_{kind}_reproduction_failed")
    initial = _baseline_decision_replay(original["initial_decision"], replayed["initial_decision"])
    return {f"{kind}_exact": initial["decision_exact"], "initial_decision_replay": initial}


def _evidence_key(config: SemanticConfig) -> str | None:
    """按显式配置选择独立trace键，不凭缺失字段猜测算法身份。"""
    if type(config) is CoherentSemanticConfig:
        return "meaning"
    if type(config) is StructuredSemanticConfig:
        return "structure"
    return None


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
    不影响缓存身份，其余算法参数固定。原配置先重放：向量摘要、片段和决策
    非评分字段严格相同，仅cohesion/depth允许有限float绝对末位差并记录。
    基线保留原trace决策；缺缓存或不一致绝不forward、补算或重抽成员。
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
    vectors_by_post, baseline_rows, replay_rows = {}, [], []
    baseline_config_id = settings.fingerprint
    for case, record, trace in zip(frozen.cases, records, traces, strict=True):
        units = case["units"]
        vectors = None if len(units) < 2 else cache.read_cached([case["source_text"][unit["start"]:unit["end"]] for unit in units])
        vector_hash = None if vectors is None else hashlib.sha256(vectors.astype("<f4").tobytes(order="C")).hexdigest()
        if vector_hash != trace["summary"].get("vector_content_sha256"):
            raise ValueError("comparison_run_vector_digest_mismatch")
        structure_evidence = {}
        evidence_key = _evidence_key(settings.algorithm)
        if evidence_key is not None:
            segments, decision, _, structure = _boundary_calculation(case, vectors, settings.algorithm, counter)
            structure_evidence = _evidence_replay(trace.get(evidence_key), structure, kind=evidence_key)
        else:
            segments, decision, _ = _boundary_result(case, vectors, settings.algorithm, counter)
        if not _strict_json_equal(segments, record["text_segments"]):
            raise ValueError("comparison_baseline_reproduction_failed")
        replay_rows.append({"source_post_id": case["source_post_id"],
                            **_baseline_decision_replay(trace["decision"], decision), **structure_evidence})
        original = record["semantic_diagnostics"]
        counts = {"unit_count": len(units), "candidate_gap_count": max(0, len(units) - 1),
                  "embedded_unit_count": len(units) if len(units) >= 2 else 0,
                  "semantic_boundary_count": len(decision["semantic_boundaries"]),
                  "length_boundary_count": len(decision["length_boundaries"]),
                  "selected_boundary_count": len(decision["semantic_boundaries"]) + len(decision["length_boundaries"])}
        if any(original.get(key) != value for key, value in counts.items()):
            raise ValueError("comparison_run_summary_counts_mismatch")
        vectors_by_post[case["source_post_id"]] = vectors
        baseline_rows.append(_case_record(case, segments, trace["decision"], config_id=baseline_config_id,
            model_hash=manifest["model_identity_sha256"], vector_hash=vector_hash,
            cache_hits=original["cache_hit_count"], model_calls=original["model_call_count"],
            elapsed=record.get("boundary_seconds"), counter=counter, targeted=targeted))
        # trace计时覆盖候选、向量与边界；不能挪作纯边界耗时。
        baseline_rows[-1]["processing_seconds"] = trace.get("processing_seconds")
        if evidence_key is not None and trace.get(evidence_key) is not None:
            baseline_rows[-1][evidence_key] = trace[evidence_key]
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
                structure = None
                if _evidence_key(config) is not None:
                    segments, decision, elapsed, structure = _boundary_calculation(case, vectors, config, counter)
                else:
                    segments, decision, elapsed = _boundary_result(case, vectors, config, counter)
                vector_hash = None if vectors is None else hashlib.sha256(vectors.astype("<f4").tobytes(order="C")).hexdigest()
                rows.append(_case_record(case, segments, decision, config_id=spec["config_id"],
                    model_hash=manifest["model_identity_sha256"], vector_hash=vector_hash,
                    cache_hits=len(case["units"]) if vectors is not None else 0, model_calls=0,
                    elapsed=elapsed, counter=counter, targeted=targeted))
                if structure is not None:
                    rows[-1][_evidence_key(config)] = structure
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
        baseline_replay = {"status": "passed", "absolute_tolerance": _REPLAY_ABSOLUTE_TOLERANCE,
                          "relative_tolerance": 0, "float_fields": ["gaps[].cohesion", "gaps[].depth"],
                          "post_count": len(replay_rows),
                          "exact_decision_post_count": sum(row["decision_exact"] for row in replay_rows),
                          "all_decisions_exact": all(row["decision_exact"] for row in replay_rows),
                          "max_absolute_delta": max((row["max_absolute_delta"] for row in replay_rows), default=0.0),
                          "vector_content_sha256_exact": True, "non_score_decision_fields_exact": True,
                          "segments_exact": True, "baseline_decision_source": "original_run_trace",
                          "posts": replay_rows}
        report = {"baseline_replay": baseline_replay,
                  "baseline": {"config_id": baseline_config_id, "statistics": summarize_comparison_cases(baseline_rows),
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
