"""不依赖数据库、管理端、清洗流水线的纯函数切分接口。

只接受单份正文，不拼接标题，不加载模型，不写文件。已提取文本使用
segment_text；数据库原始正文使用segment_body以识别Delta结构。
"""

from __future__ import annotations

import hashlib

from .contracts import SegmentationIssue, SegmentationResult, TextSegment
from .engine import IMPLEMENTATION_VERSION, RULE_VERSION, segment
from .projection import PROJECTION_ID, StructuredTextProjection, project_structured_text


def _digest(text: str) -> str:
    """对未经改写的Unicode字符串计算UTF-8摘要；非法代理字符向调用方报错。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _apply(body: str | None, projection: StructuredTextProjection) -> SegmentationResult:
    """将成功投影交给边界引擎；结构或完整性失败返回显式失败且无部分片段。"""
    text = projection.text if projection.status == "usable" else None
    issues: tuple[SegmentationIssue, ...] = ()
    segments = None
    reason = "source_body_null" if body is None else projection.reason_code
    status = "failed"
    if text is not None:
        try:
            output = segment(text)
        except ValueError:
            reason = "segment_integrity_failed"
        else:
            segments = tuple(TextSegment(**item) for item in output["text_segments"])
            issues = tuple(SegmentationIssue(**item) for item in output["issues"])
            status = "completed"
    return SegmentationResult(
        rule_version=RULE_VERSION,
        implementation_version=IMPLEMENTATION_VERSION,
        projection_id=PROJECTION_ID,
        status=status,
        reason_code=reason,
        body_was_null=body is None,
        source_body_sha256=None if body is None else _digest(body),
        source_text=text,
        source_text_sha256=None if text is None else _digest(text),
        format_id=projection.format_id,
        text_insert_count=projection.text_insert_count,
        ignored_embed_count=projection.ignored_embed_count,
        text_segments=segments,
        issues=issues,
    )


def segment_text(text: str) -> SegmentationResult:
    """原样切分已投影正文；包括JSON外观的文字也不会再次解析。

    返回带位置、编号、诊断和哈希的不可变结果。非字符串抛TypeError；
    完整性失败返回failed及None片段。空白正文成功返回空元组。
    """
    if not isinstance(text, str):
        raise TypeError("text必须是字符串")
    return _apply(text, StructuredTextProjection(text, "plain_text", "usable", "plain_text", 0, 0))


def segment_body(body: str | None) -> SegmentationResult:
    """切分源数据库原始正文，自动处理冻结契约支持的Delta JSON。

    普通正文完全保留；NULL单独标记并按空正文成功检查。损坏Delta或未知
    嵌入返回failed及None片段，不把JSON当普通文字继续切。其他输入类型抛
    TypeError。结果位置相对source_text，绝不能用于截取原始Delta JSON。
    """
    if body is not None and not isinstance(body, str):
        raise TypeError("body必须是字符串或None")
    projection = project_structured_text("" if body is None else body)
    return _apply(body, projection)
