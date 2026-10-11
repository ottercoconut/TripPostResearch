"""语义比较的分层统计与待审模板，不执行切分或代填质量结论。

统计分母分别记录空帖、失败、单单元和有候选帖子；向量参与、缓存命中和
实际模型调用是不同量。原文只由调用方保存在私有逐帖对照，本模块输出摘要。
"""

from __future__ import annotations

from collections import Counter
from statistics import mean, median

from .semantic_evaluation import BUCKETS


def length_bucket(text: str) -> str:
    """按投影正文码点数分四层；零长度单列 empty，不混入 1–100 分母。"""
    for name, low, high in BUCKETS:
        if len(text) >= low and (high is None or len(text) <= high):
            return name
    return "empty"


def _distribution(values: list[int | float]) -> dict:
    """报告有实际观测值的分布；无片段／无计时不伪造零均值。"""
    return {"count": len(values), "min": min(values) if values else None,
            "max": max(values) if values else None, "mean": mean(values) if values else None,
            "median": median(values) if values else None}


def _ratio(numerator: int, denominator: int) -> dict:
    """保留明确分子分母；空分母用 null，避免误读为 0% 或 100%。"""
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def summarize_comparison_cases(cases: list[dict]) -> dict:
    """按长度层汇总一次配置，包含压力样本但同一帖子只计一次。

    输入是 comparison 的逐帖诊断结构，status/segments 表示技术结果，
    review 状态不参与算法统计。不能把整帖保留率称为语义准确率。
    """
    ids = [row["source_post_id"] for row in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("comparison_duplicate_post")
    groups = {name: [] for name, _, _ in BUCKETS}
    groups["empty"] = []
    for row in cases:
        groups[length_bucket(row["source_text"])].append(row)

    def summarize(rows: list[dict]) -> dict:
        counts: Counter = Counter()
        failures: Counter = Counter()
        roles: Counter = Counter()
        chars, tokens, nonspace, times, processing_times, segment_counts = [], [], [], [], [], []
        for row in rows:
            counts["post_count"] += 1
            counts["completed_count"] += row["status"] == "completed"
            counts["failed_count"] += row["status"] == "failed"
            counts["empty_completed_count"] += row["status"] == "completed" and not row["segments"]
            counts["single_unit_count"] += row["unit_count"] == 1
            counts["zero_unit_count"] += row["unit_count"] == 0
            counts["candidate_post_count"] += row["unit_count"] >= 2
            counts["candidate_unit_count"] += row["unit_count"]
            counts["candidate_gap_count"] += max(0, row["unit_count"] - 1)
            counts["vector_participating_post_count"] += row["embedded_unit_count"] > 0
            counts["cache_participating_post_count"] += row["cache_hit_count"] > 0
            counts["model_participating_post_count"] += row["model_call_count"] > 0
            for key in ("embedded_unit_count", "cache_hit_count", "model_call_count",
                        "semantic_boundary_count", "length_boundary_count"):
                counts[key] += row[key]
            retained = row["status"] == "completed" and len(row["segments"]) == 1
            counts["retained_post_count"] += retained
            counts["scored_retained_post_count"] += retained and row["embedded_unit_count"] > 0
            counts["scored_completed_post_count"] += row["status"] == "completed" and row["embedded_unit_count"] > 0
            roles[row["sample_role"]] += 1
            if row["status"] == "failed":
                failures.update(row["decision"]["reason_codes"])
            else:
                segment_counts.append(len(row["segments"]))
                chars.extend(len(segment["raw_text"]) for segment in row["segments"])
                tokens.extend(row["segment_token_counts"])
                nonspace.extend(row["segment_nonspace_counts"])
            if row.get("boundary_seconds") is not None:
                times.append(row["boundary_seconds"])
            if row.get("processing_seconds") is not None:
                processing_times.append(row["processing_seconds"])
        names = ("post_count", "completed_count", "failed_count", "empty_completed_count",
                 "zero_unit_count", "single_unit_count", "candidate_post_count", "candidate_unit_count",
                 "candidate_gap_count", "vector_participating_post_count", "cache_participating_post_count",
                 "model_participating_post_count", "embedded_unit_count", "cache_hit_count", "model_call_count",
                 "semantic_boundary_count", "length_boundary_count", "retained_post_count",
                 "scored_retained_post_count", "scored_completed_post_count")
        result = {key: counts[key] for key in names}
        nonempty = counts["completed_count"] - counts["empty_completed_count"]
        result.update({
            "role_counts": dict(roles), "failure_counts": dict(failures),
            "candidate_post_ratio": _ratio(counts["candidate_post_count"], counts["post_count"]),
            "vector_participation_ratio": _ratio(counts["vector_participating_post_count"], counts["candidate_post_count"]),
            "cache_participation_ratio": _ratio(counts["cache_participating_post_count"], counts["candidate_post_count"]),
            "model_participation_ratio": _ratio(counts["model_participating_post_count"], counts["candidate_post_count"]),
            "whole_post_retention_ratio": _ratio(counts["retained_post_count"], nonempty),
            "scored_whole_post_retention_ratio": _ratio(counts["scored_retained_post_count"], counts["scored_completed_post_count"]),
            "segments_per_completed_post": _distribution(segment_counts),
            "segment_chars": _distribution(chars), "segment_nonspace_chars": _distribution(nonspace),
            "segment_tokens": _distribution(tokens), "boundary_seconds": _distribution(times),
            "total_boundary_seconds": sum(times) if times else None,
            "processing_seconds": _distribution(processing_times),
            "total_processing_seconds": sum(processing_times) if processing_times else None,
        })
        return result

    return {"all": summarize(cases), "length_strata": {name: summarize(rows) for name, rows in groups.items()}}


def make_review_template(cases: list[dict], *, comparison_id: str, config_id: str) -> dict:
    """生成逐帖待审记录；技术通过不自动推出 usable 或完成审阅。

    affected_spans 始终相对同一投影正文，未来审阅提交必须绑定 comparison /
    config，不能把改参数之前的判断复制给新结果。
    """
    return {"review_status": "pending", "comparison_id": comparison_id, "config_id": config_id,
            "samples": [{"post_id": row["source_post_id"], "sample_role": row["sample_role"],
                         "config_id": config_id, "source_text_sha256": row["source_text_sha256"],
                         "technical_pass": row["status"] == "completed", "assistant_judgment": "pending",
                         "issue_codes": [], "affected_spans": [], "explanation": "", "suggested_change": "",
                         "reviewer_kind": "assistant", "review_time": None} for row in cases]}
