"""切分JSONL到单表的输入验收；复核来源和无损契约，不重新执行边界算法。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .engine import IMPLEMENTATION_VERSION, RULE_VERSION, validate
from .projection import PROJECTION_ID, project_structured_text
from .source import ResearchPost, file_sha256


def load_run_manifest(run: Path) -> tuple[dict, str]:
    """读取完整成功运行的清单并计算批次ID；失败或未知契约拒绝导入。"""
    path = run / "manifest.json"
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if (manifest.get("artifact_kind") != "objective-text-segmentation-run"
            or manifest.get("status") != "completed"
            or manifest.get("counts", {}).get("failed_count") != 0
            or manifest.get("rule_version") != RULE_VERSION
            or manifest.get("implementation_version") != IMPLEMENTATION_VERSION
            or manifest.get("projection_id") != PROJECTION_ID
            or manifest.get("posts_filename") != "posts.jsonl"):
        raise ValueError("只允许导入当前契约的完整成功切分运行")
    return manifest, hashlib.sha256(raw).hexdigest()


def require_records_hash(run: Path, manifest: dict) -> None:
    """核验完整JSONL字节；调用方在遍历前后执行，防止读入损坏或中途换文件。"""
    if file_sha256(run / "posts.jsonl") != manifest["posts_sha256"]:
        raise ValueError("切分JSONL摘要不匹配")


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
