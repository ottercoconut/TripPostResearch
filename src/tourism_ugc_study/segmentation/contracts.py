"""切分的不可变内存结果；不定义数据库表或管理端存储结构。

片段编号仅在一份正文及一次规则执行内有效。持久化调用方必须同时携带来源、
正文哈希和规则身份，不能只凭post_id与seg_id跨运行关联。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TextSegment:
    """正文内的一段；seg_id从1连续，raw_text等于S[start:end]。

    start/end是Unicode码点零基左闭右开区间，不是字节或UTF-16位置。
    文本非空、片段不重叠、端点不切入组合字素，由引擎统一验证。
    """

    seg_id: int
    raw_text: str
    start: int
    end: int


@dataclass(frozen=True)
class SegmentationIssue:
    """相对可读正文的格式提示；提示本身不判定切分或人工编码失败。"""

    code: str
    start: int
    end: int


@dataclass(frozen=True)
class SegmentationResult:
    """一次正文切分结果及复核所需的输入、投影与算法身份。

    completed允许空元组，表示正文成功检查但无片段；failed的text_segments
    必须为None。source_text为可读投影S，source_body_sha256为原字符串哈希，
    源NULL以body_was_null区分。失败原因不包含原文；原文只由调用方保留。
    """

    rule_version: str
    implementation_version: str
    projection_id: str
    status: Literal["completed", "failed"]
    reason_code: str
    body_was_null: bool
    source_body_sha256: str | None
    source_text: str | None
    source_text_sha256: str | None
    format_id: str
    text_insert_count: int
    ignored_embed_count: int
    text_segments: tuple[TextSegment, ...] | None
    issues: tuple[SegmentationIssue, ...]
