"""合并同一冻结开发样本的跨模型技术比较，不推理或自动选型。

每个输入显式配对已完成的参数比较包及其原始开发运行。文件摘要、逐帖
候选和模型身份均须一致；不同模型按算法参数而非含模型的 config_id 对齐。
私有索引引用原比较文件，质量复核始终从 pending 开始，不复制旧判断。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path

from .artifacts import canonical_json, new_run_directory, write_json
from .embedding import identity_sha256
from .integrity import WHITE, validate
from .semantic_config import (RUN_KIND, SemanticSettings,
                              parse_algorithm_config, parse_semantic_settings, validate_identity_settings,
                              validate_semantic_rule_identity)
from .semantic_reporting import summarize_comparison_cases
from .source import file_sha256

MODEL_COMPARISON_KIND = "semantic-model-comparison"


@dataclass(frozen=True)
class ModelComparisonInput:
    """一个模型的已发布比较目录和原始运行目录，不能交换或借用另一运行。

    两条路径均指向包含 manifest.json 的目录。调用方无需提供 tokenizer、
    权重或向量缓存；原始运行的精确摘要及逐帖 trace 提供模型身份依据。
    """

    comparison: Path
    run: Path


@dataclass(frozen=True)
class _LoadedComparison:
    """保存核验后的只读输入和精确文件绑定，发布前再复验其字节未变化。"""

    source: ModelComparisonInput
    manifest: dict
    run_manifest: dict
    settings: SemanticSettings
    summary: dict
    groups: dict[str, dict]
    cases: dict[str, list[dict]]
    files: dict[Path, str]


def _read_json(path: Path) -> dict:
    """读取 JSON 对象；非法 JSON 或非对象输入不能作为已完成清单使用。"""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("model_comparison_invalid_json_object")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    """读取已核对文件摘要的逐帖记录，保持发布时的顺序。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _bind_file(path: Path, digest: str, files: dict[Path, str]) -> None:
    """将精确文件摘要加入输入绑定；拒绝损坏文件而不修补原包。"""
    if file_sha256(path) != digest:
        raise ValueError("model_comparison_file_digest_mismatch")
    files[path] = digest


def _validate_budget(manifest: dict, settings: SemanticSettings) -> dict[str, dict]:
    """恢复最多九组合法预算，以算法摘要对齐并拒绝固定参数或配置身份漂移。"""
    specs = manifest["configs"]
    if not specs or len(specs) > 9 or manifest["config_count"] != len(specs):
        raise ValueError("model_comparison_invalid_parameter_budget")
    fixed = asdict(settings.algorithm)
    result = {}
    for spec in specs:
        algorithm = parse_algorithm_config(spec["algorithm"], require_complete=True)
        values = asdict(algorithm)
        if (values != spec["algorithm"] or values.keys() != fixed.keys()
                or algorithm.window_tokens not in {32, 64, 128}
                or algorithm.absolute_cohesion_threshold not in {0.45, 0.55, 0.65}
                or any(value != fixed[key] for key, value in values.items()
                       if key not in {"window_tokens", "absolute_cohesion_threshold"})
                or spec["config_id"] != SemanticSettings(algorithm, settings.embedding).fingerprint):
            raise ValueError("model_comparison_parameter_binding_mismatch")
        algorithm_id = identity_sha256(values)
        if algorithm_id in result:
            raise ValueError("model_comparison_duplicate_parameter_group")
        result[algorithm_id] = spec
    return result


def _validate_members(manifest: dict) -> None:
    """核对成员和开发／压力角色，防止同 ID 集合掩盖角色变更或重复成员。"""
    ids = manifest["post_ids"]
    development, targeted = manifest["development_post_ids"], manifest["targeted_post_ids"]
    if (not ids or any(type(pid) is not int or pid < 1 for pid in ids)
            or ids != sorted(set(ids))
            or len(development) != len(set(development)) or len(targeted) != len(set(targeted))
            or set(development) & set(targeted) or sorted(development + targeted) != ids
            or manifest["post_count"] != len(ids)):
        raise ValueError("model_comparison_member_roles_mismatch")


def _case_basis(rows: list[dict]) -> list[dict]:
    """保留跨配置／模型必须完全相同的正文、来源及候选证据。"""
    keys = ("source_post_id", "source_version", "source_text", "source_text_sha256",
            "original_body", "units", "sample_role")
    return [{key: row[key] for key in keys} for row in rows]


def _validate_cases(rows: list[dict], manifest: dict, spec: dict, posts: list[dict],
                    traces: list[dict], settings: SemanticSettings) -> None:
    """核对候选和真实运行的模型／向量摘要；不打开缓存或重算语义边界。"""
    ids = manifest["post_ids"]
    if any([row["source_post_id"] for row in records] != ids for records in (rows, posts, traces)):
        raise ValueError("model_comparison_case_members_mismatch")
    targeted = set(manifest["targeted_post_ids"])
    for row, post, trace in zip(rows, posts, traces, strict=True):
        text, units = row["source_text"], row["units"]
        digest = hashlib.sha256(text.encode()).hexdigest()
        role = "pressure" if row["source_post_id"] in targeted else "development"
        if (row["config_id"] != spec["config_id"] or row["sample_role"] != role
                or row["source_text_sha256"] != digest or row["unit_count"] != len(units)
                or any(post.get(key) != row[key] for key in
                       ("source_text", "source_text_sha256", "source_version", "original_body"))
                or trace["source_text_sha256"] != digest or trace["units"] != units):
            raise ValueError("model_comparison_frozen_case_mismatch")
        if any(set(unit) != {"start", "end", "token_count"}
               or any(type(unit[key]) is not int for key in unit) or unit["token_count"] < 1
               for unit in units):
            raise ValueError("model_comparison_units_invalid")
        validate(text, [{"seg_id": index, "raw_text": text[unit["start"]:unit["end"]],
                         "start": unit["start"], "end": unit["end"]}
                        for index, unit in enumerate(units, 1)])
        trace_summary = trace["summary"]
        if (row["model_identity_sha256"] != manifest["model_identity_sha256"]
                or trace_summary["model_identity_sha256"] != manifest["model_identity_sha256"]
                or trace_summary != post["semantic_diagnostics"]
                or row["vector_content_sha256"] != trace_summary["vector_content_sha256"]
                or row["model_call_count"] != 0
                or row["embedded_unit_count"] != (len(units) if len(units) >= 2 else 0)
                or row["cache_hit_count"] != row["embedded_unit_count"]):
            raise ValueError("model_comparison_model_cache_binding_mismatch")
        baseline = post["text_segments"] or []
        if (row["baseline_config_id"] != settings.fingerprint or row["baseline_segments"] != baseline
                or row["baseline_decision"] != trace["decision"]
                or row["baseline_boundaries"] != [segment["end"] for segment in baseline[:-1]]):
            raise ValueError("model_comparison_baseline_binding_mismatch")
        if row["status"] == "completed":
            validate(text, row["segments"])
        elif row["status"] != "failed" or row["segments"]:
            raise ValueError("model_comparison_case_status_invalid")
        if (row["new_boundaries"] != [segment["end"] for segment in row["segments"][:-1]]
                or row["semantic_boundary_count"] != len(row["decision"]["semantic_boundaries"])
                or row["length_boundary_count"] != len(row["decision"]["length_boundaries"])):
            raise ValueError("model_comparison_boundary_binding_mismatch")
    binding = [{key: row[key] for key in ("source_post_id", "source_version", "source_text_sha256", "units")}
               for row in rows]
    if identity_sha256({"input": manifest["input"], "post_ids": ids, "cases": binding}) != manifest["frozen_units_sha256"]:
        raise ValueError("model_comparison_frozen_units_digest_mismatch")


def _validate_baseline(summary: dict, rows: list[dict], posts: list[dict], traces: list[dict],
                       run: dict, run_digest: str) -> None:
    """复核原运行可重建的统计、调用数和资源绑定，不重计需要 tokenizer 的段 token 数。

    原比较包里的段 token 数仍是其精确文件绑定的观测；其余分层统计从原
    run/trace 重新汇总。资源属于完整原运行，不能移作某参数组的内存结论。
    """
    baseline_rows = []
    for case, post, trace in zip(rows, posts, traces, strict=True):
        segments = post["text_segments"] or []
        if post["status"] == "completed":
            validate(case["source_text"], segments)
        elif post["status"] != "failed" or post["text_segments"] is not None:
            raise ValueError("model_comparison_baseline_status_invalid")
        row = dict(case, status=post["status"], segments=segments, decision=trace["decision"],
                   segment_token_counts=[], segment_nonspace_counts=[sum(char not in WHITE for char in segment["raw_text"])
                                                                        for segment in segments],
                   boundary_seconds=post.get("boundary_seconds"), processing_seconds=trace.get("processing_seconds"))
        row.update({key: trace["summary"][key] for key in
                    ("embedded_unit_count", "cache_hit_count", "model_call_count", "semantic_boundary_count", "length_boundary_count")})
        baseline_rows.append(row)
    rebuilt = summarize_comparison_cases(baseline_rows)
    reported = summary["baseline"]["statistics"]
    for actual, claimed in [(rebuilt["all"], reported["all"]),
                            *((value, reported["length_strata"][name]) for name, value in rebuilt["length_strata"].items())]:
        if any(claimed.get(key) != value for key, value in actual.items() if key != "segment_tokens"):
            raise ValueError("model_comparison_baseline_statistics_mismatch")
    if (any(run["counts"][key] != rebuilt["all"][key] for key in
            ("post_count", "completed_count", "failed_count", "embedded_unit_count", "cache_hit_count", "model_call_count"))
            or run["counts"]["segment_count"] != sum(len(row["segments"]) for row in baseline_rows)):
        raise ValueError("model_comparison_baseline_run_counts_mismatch")
    baseline = summary["baseline"]
    started, completed = run.get("started_at_utc"), run.get("completed_at_utc")
    elapsed = (datetime.fromisoformat(completed) - datetime.fromisoformat(started)).total_seconds() if started and completed else None
    if (baseline["actual_embedding_model_calls"] != run["counts"]["model_call_count"]
            or baseline["corpus_model_call_count"] != run["counts"]["model_call_count"]
            or baseline["preflight_model_call_count"] != run.get("preflight_model_call_count")
            or baseline["run_elapsed_seconds"] != elapsed or baseline["resource_scope"] != "whole_original_run"
            or baseline["resources"] is not None and baseline["resources"].get("run_manifest_sha256") != run_digest):
        raise ValueError("model_comparison_baseline_runtime_binding_mismatch")


def _load_comparison(source: ModelComparisonInput) -> _LoadedComparison:
    """核验一个显式配对的发布包；摘要有效后仍核对比较内容和运行身份。"""
    manifest_path, run_path = source.comparison / "manifest.json", source.run / "manifest.json"
    manifest, run = _read_json(manifest_path), _read_json(run_path)
    files = {manifest_path: file_sha256(manifest_path)}
    _bind_file(run_path, manifest["run_manifest_sha256"], files)
    if (manifest.get("artifact_kind") != "semantic-development-comparison"
            or manifest.get("status") != "completed" or manifest.get("baseline_reproduction") != "passed"
            or run.get("artifact_kind") != RUN_KIND or run.get("scope") != "development"
            or run.get("status") not in {"completed", "completed_with_failures"}):
        raise ValueError("model_comparison_incomplete_input")
    validate_semantic_rule_identity(run)
    settings = parse_semantic_settings(run["config"])
    validate_identity_settings(run["model_identity"], settings.embedding)
    if (run["config_sha256"] != settings.fingerprint
            or identity_sha256(run["model_identity"]) != run["model_identity_sha256"]
            or run["model_identity_sha256"] != manifest["model_identity_sha256"]):
        raise ValueError("model_comparison_model_identity_mismatch")
    _validate_members(manifest)
    if run["input"] != manifest["input"] or run["selected_post_ids"] != manifest["post_ids"]:
        raise ValueError("model_comparison_run_source_members_mismatch")
    validated = manifest["run_validation"]
    if (validated.get("validation") != "passed"
            or any(validated.get(key) != manifest[key] for key in
                   ("run_manifest_sha256", "members_manifest_sha256", "frozen_units_sha256"))
            or validated.get("post_count") != manifest["post_count"]):
        raise ValueError("model_comparison_run_validation_mismatch")
    binding = {key: manifest[key] for key in
               ("run_manifest_sha256", "members_manifest_sha256", "frozen_units_sha256", "configs", "runtime")}
    if identity_sha256(binding) != manifest["comparison_id"]:
        raise ValueError("model_comparison_comparison_identity_mismatch")
    groups = _validate_budget(manifest, settings)
    expected = {"summary.json"} | {f"{kind}-{spec['config_id']}.{suffix}"
        for spec in groups.values() for kind, suffix in (("cases", "jsonl"), ("review", "json"))}
    if set(manifest["files_sha256"]) != expected:
        raise ValueError("model_comparison_file_inventory_mismatch")
    for filename, digest in manifest["files_sha256"].items():
        _bind_file(source.comparison / filename, digest, files)
    for filename, key in (("posts.jsonl", "posts_sha256"), ("semantic-trace.jsonl", "trace_sha256")):
        if run.get("posts_filename" if filename == "posts.jsonl" else "trace_filename") != filename:
            raise ValueError("model_comparison_run_file_inventory_mismatch")
        _bind_file(source.run / filename, run[key], files)
    posts, traces = _read_jsonl(source.run / "posts.jsonl"), _read_jsonl(source.run / "semantic-trace.jsonl")
    summary = _read_json(source.comparison / "summary.json")
    if (summary["baseline"]["config_id"] != settings.fingerprint
            or [group["config_id"] for group in summary["groups"]] != [spec["config_id"] for spec in manifest["configs"]]):
        raise ValueError("model_comparison_summary_config_mismatch")
    cases = {}
    for group in summary["groups"]:
        algorithm_id = identity_sha256(group["algorithm"])
        if algorithm_id not in groups or group["algorithm"] != groups[algorithm_id]["algorithm"]:
            raise ValueError("model_comparison_summary_algorithm_mismatch")
        spec = groups[algorithm_id]
        rows = _read_jsonl(source.comparison / f"cases-{spec['config_id']}.jsonl")
        _validate_cases(rows, manifest, spec, posts, traces, settings)
        if group["statistics"] != summarize_comparison_cases(rows):
            raise ValueError("model_comparison_statistics_mismatch")
        review = _read_json(source.comparison / f"review-{spec['config_id']}.json")
        if (review["comparison_id"] != manifest["comparison_id"] or review["config_id"] != spec["config_id"]
                or [(sample["post_id"], sample["sample_role"], sample["source_text_sha256"], sample["config_id"], sample["technical_pass"])
                    for sample in review["samples"]] != [(row["source_post_id"], row["sample_role"], row["source_text_sha256"],
                        spec["config_id"], row["status"] == "completed") for row in rows]):
            raise ValueError("model_comparison_review_binding_mismatch")
        if cases and _case_basis(rows) != _case_basis(next(iter(cases.values()))):
            raise ValueError("model_comparison_group_candidates_mismatch")
        cases[algorithm_id] = rows
    _validate_baseline(summary, next(iter(cases.values())), posts, traces, run, manifest["run_manifest_sha256"])
    return _LoadedComparison(source, manifest, run, settings, summary, groups, cases, files)


def _validate_comparability(inputs: tuple[_LoadedComparison, ...]) -> None:
    """只让不同型号在同源、同角色、同候选及同参数预算下进入并排报告。"""
    first = inputs[0]
    if len({item.settings.embedding.model_id for item in inputs}) != len(inputs):
        raise ValueError("model_comparison_distinct_models_required")
    for item in inputs[1:]:
        if any(item.manifest[key] != first.manifest[key] for key in
               ("input", "members_manifest_sha256", "frozen_units_sha256", "post_ids", "development_post_ids", "targeted_post_ids")):
            raise ValueError("model_comparison_frozen_sample_mismatch")
        if (item.groups.keys() != first.groups.keys() or item.settings.algorithm != first.settings.algorithm
                or any(getattr(item.settings.embedding, key) != getattr(first.settings.embedding, key)
                       for key in ("max_input_tokens", "instruction"))):
            raise ValueError("model_comparison_parameter_budget_mismatch")
        if _case_basis(next(iter(item.cases.values()))) != _case_basis(next(iter(first.cases.values()))):
            raise ValueError("model_comparison_cross_model_candidates_mismatch")


def compare_semantic_models(inputs: tuple[ModelComparisonInput, ...], output: Path) -> dict:
    """发布至少两个不同型号的同冻结样本技术汇总及逐帖对照索引。

    输入只读取已发布 comparison/run 的 JSON 证据，不访问源库、tokenizer
    或 embedding 缓存。原始模型身份、参数和文件摘要不一致时抛 ValueError，
    缺文件或非法 JSON 保留相应读取异常；输出已存在时拒绝覆盖。成功只表示
    比较材料技术上可对照，不选择模型、不生成 usable，也不完成语义质量复核。
    """
    if len(inputs) < 2:
        raise ValueError("model_comparison_at_least_two_inputs_required")
    if Path(output).exists():
        raise FileExistsError("model_comparison_output_exists")
    paired = tuple(ModelComparisonInput(Path(item.comparison).resolve(), Path(item.run).resolve()) for item in inputs)
    loaded = tuple(_load_comparison(item) for item in paired)
    _validate_comparability(loaded)
    first = loaded[0]
    models = [{"model_id": item.settings.embedding.model_id,
               "model_identity_sha256": item.manifest["model_identity_sha256"],
               "model_identity": item.run_manifest["model_identity"],
               "comparison_id": item.manifest["comparison_id"],
               "comparison_directory": str(item.source.comparison), "run_directory": str(item.source.run),
               "comparison_manifest_sha256": item.files[item.source.comparison / "manifest.json"],
               "run_manifest_sha256": item.manifest["run_manifest_sha256"]} for item in loaded]
    budget = [{"algorithm_id": algorithm_id, "algorithm": spec["algorithm"]}
              for algorithm_id, spec in sorted(first.groups.items(), key=lambda pair:
                  (pair[1]["algorithm"]["window_tokens"], pair[1]["algorithm"]["absolute_cohesion_threshold"]))]
    binding = {key: first.manifest[key] for key in
               ("input", "members_manifest_sha256", "frozen_units_sha256", "post_ids", "development_post_ids", "targeted_post_ids")}
    binding.update({"models": models, "parameter_budget": budget})
    comparison_id = identity_sha256(binding)
    pending = {"review_status": "pending", "semantic_quality_evaluated": False}
    with new_run_directory(output) as staging:
        summary = {"models": [{**model, "baseline": item.summary["baseline"]} for model, item in zip(models, loaded, strict=True)],
            "groups": [{**spec, "models": [{"model_id": model["model_id"],
                "model_identity_sha256": model["model_identity_sha256"],
                "config_id": item.groups[spec["algorithm_id"]]["config_id"],
                "statistics": summarize_comparison_cases(item.cases[spec["algorithm_id"]]),
                "total_replay_seconds": next(group["total_replay_seconds"] for group in item.summary["groups"]
                    if group["config_id"] == item.groups[spec["algorithm_id"]]["config_id"]),
                **pending} for model, item in zip(models, loaded, strict=True)]} for spec in budget], **pending}
        write_json(staging / "summary.json", summary)
        index_path = staging / "post-index.jsonl"
        with index_path.open("x", encoding="utf-8") as stream:
            index_path.chmod(0o600)
            for row_number, case in enumerate(next(iter(first.cases.values())), 1):
                contrasts = []
                for spec in budget:
                    records = []
                    for model, item in zip(models, loaded, strict=True):
                        config_id = item.groups[spec["algorithm_id"]]["config_id"]
                        row = item.cases[spec["algorithm_id"]][row_number - 1]
                        records.append({"model_id": model["model_id"], "config_id": config_id,
                            "model_identity_sha256": model["model_identity_sha256"],
                            "cases_path": str(item.source.comparison / f"cases-{config_id}.jsonl"),
                            "cases_sha256": item.manifest["files_sha256"][f"cases-{config_id}.jsonl"],
                            "review_path": str(item.source.comparison / f"review-{config_id}.json"),
                            "review_sha256": item.manifest["files_sha256"][f"review-{config_id}.json"],
                            "line_number": row_number, "status": row["status"],
                            "new_boundaries": row["new_boundaries"], "segment_count": len(row["segments"])})
                    contrasts.append({**spec, "models": records})
                row = {"source_post_id": case["source_post_id"], "source_version": case["source_version"],
                    "sample_role": case["sample_role"], "source_text_sha256": case["source_text_sha256"],
                    "groups": contrasts, **pending}
                stream.write(canonical_json(row) + "\n")
        # 原比较包仍是逐帖正文及模型结果的唯一证据；索引发布前再次核对全部绑定文件。
        for item in loaded:
            for path, digest in item.files.items():
                if file_sha256(path) != digest:
                    raise ValueError("model_comparison_input_changed")
        result = {"artifact_kind": MODEL_COMPARISON_KIND, "status": "completed", "comparison_id": comparison_id,
            **binding, "model_count": len(loaded), "config_count": len(budget), "post_count": len(first.manifest["post_ids"]),
            "files_sha256": {filename: file_sha256(staging / filename) for filename in
                             ("summary.json", "post-index.jsonl")},
            "actual_embedding_model_calls": 0, "source_database_write_count": 0,
            "membership_database_write_count": 0, **pending}
        write_json(staging / "manifest.json", result)
    return result
