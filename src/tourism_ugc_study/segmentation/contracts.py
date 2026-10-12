"""切分的不可变内存结果；不定义数据库表或管理端存储结构。

片段编号仅在一份正文及一次规则执行内有效。持久化调用方必须同时携带来源、
正文哈希和规则身份，不能只凭post_id与seg_id跨运行关联。
"""

from __future__ import annotations

from dataclasses import dataclass


class EmbeddingError(ValueError):
    """模型／tokenizer／缓存契约失败，必须终止attempt，不能降为逐帖失败。

    放在轻量契约层，单元构造可以识别系统故障而无需导入numpy或模型依赖。
    """
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


@dataclass(frozen=True)
class TextUnit:
    """参与语义比较的候选单元；文字直接取S[start:end]，不代表最终片段。

    token_count仅计正文，不含模型指令；端点为完整字素边界，空白间隙允许。
    构建与边界入口负责验证，结构自身不调用tokenizer或模型。
    """

    start: int
    end: int
    token_count: int


@dataclass(frozen=True)
class UnitizationResult:
    """候选单元及不含原文的诊断码；失败抛异常，不返回半份单元。"""

    units: tuple[TextUnit, ...]
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class GapScore:
    """两个相邻单元间的评分；index为零基gap序号，不是原文码点位置。

    candidate表示低谷候选，selected表示最终保留为边界；reason解释阈值、
    短段抑制或长度保护。无语义评分的路径不虚构gap数据。
    """

    index: int
    cohesion: float
    depth: float
    candidate: bool
    selected: bool
    reason: str


@dataclass(frozen=True)
class SemanticDecision:
    """单帖语义决策与可重放trace；两组boundaries均为零基gap序号。

    status区分empty/single_unit/scored_keep/scored_split/length_split/failed；
    本结构不改变SegmentationResult.reason_code的正文投影含义。
    当前semantic-coherent规则的semantic_boundaries沿用发布字段名，表示
    全部非长度切点（语义与结构来源的并集）；具体来源在MeaningTrace中
    分别保存。两类最终边界互斥，历史规则的字段和值不作追溯改写。
    """

    status: str
    reason_codes: tuple[str, ...]
    gaps: tuple[GapScore, ...]
    semantic_boundaries: tuple[int, ...]
    length_boundaries: tuple[int, ...]
