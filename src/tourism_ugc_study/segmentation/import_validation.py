"""切分JSONL到单表的输入验收；复核来源和无损契约，不重新执行边界算法。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .integrity import validate
from .projection import PROJECTION_ID, project_structured_text
from .semantic_config import (
    EmbeddingConfig, parse_algorithm_config, parse_semantic_settings, validate_identity_settings,
    validate_current_frozen_settings, validate_semantic_rule_identity, RUN_KIND as SEMANTIC_RUN_KIND,
)
from .source import ResearchPost, file_sha256


def _sha256(value: object) -> bool:
    """仅接受64位小写十六进制身份，不将任意等长字符串当作摘要。"""
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _json_sha256(value: object) -> str:
    """在不加载模型依赖的情况下核验有限JSON身份。"""
    try:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError("语义身份不是有限JSON") from None
    return hashlib.sha256(raw.encode()).hexdigest()


def _validate_semantic_identity(manifest: dict) -> None:
    """核验声明配置、实际模型身份与摘要；不把身份声明当作重新运行模型。"""
    config = manifest.get("config")
    if not isinstance(config, dict) or set(config) != {"algorithm", "embedding"}:
        raise ValueError("语义运行配置字段非法")
    raw_embedding = config["embedding"]
    if (not isinstance(raw_embedding, dict)
            or set(raw_embedding) != set(EmbeddingConfig.__dataclass_fields__)):
        raise ValueError("语义运行配置必须完整保存所有字段")
    try:
        parse_algorithm_config(config["algorithm"], require_complete=True)
        settings = parse_semantic_settings(config)
        validate_current_frozen_settings(settings)
    except (TypeError, ValueError):
        raise ValueError("语义运行配置值非法") from None
    if not _sha256(manifest.get("config_sha256")) or settings.fingerprint != manifest["config_sha256"]:
        raise ValueError("语义运行配置摘要不匹配")
    identity = manifest.get("model_identity")
    if not isinstance(identity, dict) or not _sha256(manifest.get("model_identity_sha256")):
        raise ValueError("缺少完整语义模型身份")
    if _json_sha256(identity) != manifest["model_identity_sha256"]:
        raise ValueError("语义模型身份摘要不匹配")
    validate_identity_settings(identity, settings.embedding)
    assets = identity.get("assets_sha256")
    required_assets = {"config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"}
    if (not isinstance(assets, dict) or not required_assets <= set(assets)
            or any(not isinstance(name, str) or Path(name).name != name or not _sha256(digest) for name, digest in assets.items())
            or not any(name.endswith(".safetensors") for name in assets)):
        raise ValueError("语义模型缺少本地资产摘要")
    dependencies = identity.get("dependencies")
    if (not isinstance(dependencies, dict)
            or not {"torch", "transformers", "tokenizers", "huggingface-hub", "numpy"} <= set(dependencies)
            or any(not isinstance(value, str) or not value for value in dependencies.values())):
        raise ValueError("语义模型缺少运行依赖身份")


def validate_run_manifest(manifest: dict) -> None:
    """公共发布门：只认可当前冻结连贯块规则的完整全量运行。

    原始run和五文件发布包共用此门，不能通过直接复验发布包绕开开发集隔离。
    本函数验证清单内身份及计数；原始trace文件摘要由require_records_hash核验。
    发布包只保留trace摘要，单凭五文件不能宣称已重放向量或语义边界。
    """
    if not isinstance(manifest, dict):
        raise ValueError("切分运行清单必须为对象")
    kind = manifest.get("artifact_kind")
    # 历史产物留在归档；线上导入与五文件复验均不再解释退役算法。
    if kind != SEMANTIC_RUN_KIND:
        raise ValueError("只允许导入当前冻结连贯块规则的完整成功切分运行")
    validate_semantic_rule_identity(manifest)
    counts = manifest.get("counts")
    if (not isinstance(counts, dict)
            or manifest.get("status") != "completed" or type(counts.get("failed_count")) is not int
            or counts["failed_count"] != 0 or manifest.get("projection_id") != PROJECTION_ID
            or manifest.get("posts_filename") != "posts.jsonl"):
        raise ValueError("只允许导入已登记契约的完整成功切分运行")
    if (manifest.get("scope") != "full_corpus" or "selected_post_ids" not in manifest
            or manifest["selected_post_ids"] is not None
            or manifest.get("sampling") != "all_keep_ordered_by_source_post_id"):
        raise ValueError("开发样本不能导入正式片段发布包")
    source = manifest.get("input")
    fields = ("post_count", "completed_count", "failed_count", "segment_count", "empty_count",
              "null_body_count", "structured_body_count", "posts_with_issues",
              "embedded_unit_count", "cache_hit_count", "model_call_count")
    if (not isinstance(source, dict) or type(source.get("record_count")) is not int or source["record_count"] < 0
            or any(type(counts.get(key)) is not int or counts[key] < 0 for key in fields)
            or counts["post_count"] != counts["completed_count"] or counts["post_count"] != source["record_count"]
            or any(counts[key] > counts["post_count"] for key in ("empty_count", "null_body_count", "structured_body_count", "posts_with_issues"))
            or counts["cache_hit_count"] > counts["embedded_unit_count"]
            or counts["model_call_count"] > counts["embedded_unit_count"]):
        raise ValueError("语义运行计数与完整来源不一致")
    if (manifest.get("trace_filename") != "semantic-trace.jsonl" or not _sha256(manifest.get("trace_sha256"))
            or not _sha256(manifest.get("posts_sha256")) or not _sha256(manifest.get("ordered_input_sha256"))):
        raise ValueError("语义运行缺少结果或trace摘要")
    decisions = manifest.get("decision_counts")
    if (not isinstance(decisions, dict)
            or set(decisions) - {"empty", "single_unit", "scored_keep", "scored_split", "length_split"}
            or any(type(count) is not int or count < 1 for count in decisions.values())
            or sum(decisions.values()) != counts["post_count"]
            or decisions.get("empty", 0) != counts["empty_count"]
            or manifest.get("failure_counts") != {}
            or any(type(manifest.get(key)) is not int or manifest[key] != 0 for key in
                   ("source_database_write_count", "membership_database_write_count"))):
        raise ValueError("语义决策总计或只读来源声明非法")
    _validate_semantic_identity(manifest)


def load_run_manifest(run: Path) -> tuple[dict, str]:
    """读取完整成功运行的清单并计算批次ID；失败或未知契约拒绝导入。"""
    path = run / "manifest.json"
    raw = path.read_bytes()
    manifest = json.loads(raw)
    validate_run_manifest(manifest)
    return manifest, hashlib.sha256(raw).hexdigest()


def require_records_hash(run: Path, manifest: dict) -> None:
    """核验完整JSONL字节；调用方在遍历前后执行，防止读入损坏或中途换文件。"""
    if file_sha256(run / "posts.jsonl") != manifest["posts_sha256"]:
        raise ValueError("切分JSONL摘要不匹配")
    trace = run / "semantic-trace.jsonl"
    if not trace.is_file() or file_sha256(trace) != manifest["trace_sha256"]:
        raise ValueError("语义trace缺失或摘要不匹配")


def _validate_semantic_summary(record: dict, manifest: dict, segment_count: int) -> None:
    """检查逐帖摘要的状态机、参与数量和模型绑定，不伪称语义效果已经复核。"""
    summary = record.get("semantic_diagnostics")
    numbers = ("unit_count", "candidate_gap_count", "embedded_unit_count", "cache_hit_count", "model_call_count",
               "semantic_boundary_count", "length_boundary_count", "selected_boundary_count")
    keys = {*numbers, "decision", "reason_codes", "model_identity_sha256", "vector_content_sha256"}
    if (not isinstance(summary, dict) or set(summary) != keys
            or any(type(summary[key]) is not int or summary[key] < 0 for key in numbers)
            or not isinstance(summary["reason_codes"], list) or not summary["reason_codes"]
            or any(not isinstance(code, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", code) for code in summary["reason_codes"])
            or summary["model_identity_sha256"] != manifest["model_identity_sha256"]):
        raise ValueError("语义逐帖摘要字段或模型身份非法")
    units = summary["unit_count"]
    semantic = summary["semantic_boundary_count"]
    length = summary["length_boundary_count"]
    if (summary["candidate_gap_count"] != max(0, units - 1)
            or summary["selected_boundary_count"] != semantic + length
            or semantic + length != max(0, segment_count - 1)
            or (units > 0 and segment_count == 0) or segment_count > units or segment_count > 2000
            or summary["cache_hit_count"] > summary["embedded_unit_count"]
            or summary["model_call_count"] > summary["embedded_unit_count"]):
        raise ValueError("语义逐帖单元、片段或边界计数不一致")
    if units <= 1:
        if (summary["decision"] != ("empty" if units == 0 else "single_unit")
                or segment_count != units or summary["embedded_unit_count"] != 0
                or summary["cache_hit_count"] != 0 or summary["model_call_count"] != 0
                or summary["vector_content_sha256"] is not None):
            raise ValueError("无语义评分路径的摘要非法")
    elif (summary["decision"] != ("length_split" if length else "scored_split" if semantic else "scored_keep")
            or summary["embedded_unit_count"] != units or not _sha256(summary["vector_content_sha256"])
            or (summary["cache_hit_count"] == units and summary["model_call_count"] != 0)
            or (summary["cache_hit_count"] < units and not 1 <= summary["model_call_count"] <= units - summary["cache_hit_count"])):
        raise ValueError("语义评分路径的摘要非法")


def validate_post_record(record: dict, source: ResearchPost, manifest: dict) -> None:
    """逐帖复核身份、源body、投影、字符覆盖和编号；失败不产生任何可用片段。

    片段端点必须为整数且落在Unicode字素边界；原文不能丢失或重叠。哈希正确
    不是内容正确的替代，因此还与当前绑定的只读源快照逐条对照。
    """
    if (record.get("source_post_id") != source.source_post_id
            or record.get("source_version") != source.source_version
            or record.get("original_body") != source.body
            or record.get("body_was_null") != (source.body is None)
            or record.get("status") != "completed"
            or record.get("source_snapshot_sha256") != manifest["input"]["source_snapshot_sha256"]
            or any(record.get(key) != manifest[key] for key in
                   ("rule_version", "implementation_version", "projection_id"))):
        raise ValueError("切分记录与源成员身份或规则不匹配")
    body_hash = None if source.body is None else hashlib.sha256(source.body.encode()).hexdigest()
    projection = project_structured_text("" if source.body is None else source.body)
    if (projection.status != "usable" or record.get("source_text") != projection.text
            or record.get("source_body_sha256") != body_hash
            or record.get("source_text_sha256") != hashlib.sha256(projection.text.encode()).hexdigest()
            or any(record.get(key) != getattr(projection, key) for key in
                   ("format_id", "text_insert_count", "ignored_embed_count"))):
        raise ValueError("切分记录的原文投影或摘要不一致")
    expected_reason = "source_body_null" if source.body is None else projection.reason_code
    if record.get("reason_code") != expected_reason:
        raise ValueError("切分投影状态原因不一致")
    segments = record.get("text_segments")
    if not isinstance(segments, list):
        raise ValueError("成功记录必须包含片段数组")
    for item in segments:
        if (not isinstance(item, dict) or set(item) != {"seg_id", "raw_text", "start", "end"}
                or any(type(item[k]) is not int for k in ("seg_id", "start", "end"))
                or not isinstance(item["raw_text"], str)):
            raise ValueError("片段字段或类型非法")
    validate(projection.text, segments)
    issues = record.get("issues")
    if not isinstance(issues, list):
        raise ValueError("缺少格式诊断列表")
    for item in issues:
        if (not isinstance(item, dict) or set(item) != {"code", "start", "end"}
                or not isinstance(item["code"], str)
                or type(item["start"]) is not int or type(item["end"]) is not int
                or not 0 <= item["start"] <= item["end"] <= len(projection.text)):
            raise ValueError("格式诊断范围或类型非法")
    _validate_semantic_summary(record, manifest, len(segments))
