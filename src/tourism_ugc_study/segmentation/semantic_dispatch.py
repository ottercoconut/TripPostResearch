"""按显式算法配置分派纯边界计算，保持0.1重放与0.2开发规则隔离。

本层不读取模型、缓存、数据库或帖子ID。旧配置完整返回旧边界器结果；
0.2复用同一评分，再以结构分区重建最终决策。结构证据独立保存，不向旧
SemanticDecision或逐帖summary增加字段，也不将结构罚分称为质量金标。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Sequence

from .contracts import SemanticDecision, TextSegment, TextUnit
from .integrity import WHITE
from .semantic_boundaries import choose_boundaries
from .semantic_config import SemanticConfig, StructuredSemanticConfig, semantic_rule_identity
from .semantic_partition import BlockRelaxation, partition_with_structure
from .semantic_structure import GapStructure


class BoundaryTokenCounter(Protocol):
    """只对完整原文子串精确计数；无截断、模型调用或缓存操作。"""

    def count(self, text: str) -> int:
        """返回非负整数token数；正文非空时不能返回零。"""
        ...


@dataclass(frozen=True)
class StructureTrace:
    """0.2的独立结构证据；可asdict后写入trace的structure字段。

    initial_decision保留原评分与旧临时长度选择，不能作为最终切分摘要。
    gaps及各boundary使用零基gap序号，protected_blocks使用原文码点
    [start,end)区间。forced_structure_boundaries是最终长度边界的子集，
    表示为了满足硬长度而接受结构风险，不是第三种可重复计数的边界。
    block_relaxations记录超长保护块的原跨度、精确token数与缩减终点，
    以区分未发现结构和发现后因硬长度放弃保护。
    """

    structure_mode: str
    length_strategy: str
    initial_decision: SemanticDecision
    gaps: tuple[GapStructure, ...]
    suppressed_semantic_boundaries: tuple[int, ...]
    forced_structure_boundaries: tuple[int, ...]
    protected_blocks: tuple[tuple[int, int, str], ...]
    block_relaxations: tuple[BlockRelaxation, ...]


@dataclass(frozen=True)
class SemanticDispatchResult:
    """同一配置执行的完整片段、最终决策与可选结构证据。

    片段和decision必须描述同一组切点；0.1的structure_trace固定为None。
    失败沿用领域ValueError或系统EmbeddingError，不返回部分成功结果。
    """

    segments: tuple[TextSegment, ...]
    decision: SemanticDecision
    structure_trace: StructureTrace | None = None


def choose_semantic_boundaries(
    text: str, units: Sequence[TextUnit], vectors: object, config: SemanticConfig,
    counter: BoundaryTokenCounter,
) -> SemanticDispatchResult:
    """执行已登记配置的纯切分，并保证最终边界、状态及gap标记相符。

    输入和失败契约沿用choose_boundaries：候选单元须覆盖全部非空白正文、
    端点字素安全、计数精确；N≥2要求有效逐单元向量，N≤1不虚构评分。
    0.1不改其片段或决策。0.2仅重新分区既有gap，硬长度优先于结构偏好，
    被抑制语义点、被迫结构切断和初始决策在独立trace中留证。
    """
    semantic_rule_identity(config)
    segments, initial = choose_boundaries(text, units, vectors, config, counter)
    if type(config) is SemanticConfig:
        return SemanticDispatchResult(segments, initial)
    # 配置身份已限定确切登记类型；避免未知子类意外进入新算法。
    if type(config) is not StructuredSemanticConfig:
        raise ValueError("invalid_semantic_config_type")
    partition = partition_with_structure(
        text, units, initial, config, counter,
        mode=config.structure_mode, length_strategy=config.length_strategy,
    )
    semantic = set(partition.semantic_boundaries)
    lengths = set(partition.length_boundaries)
    suppressed = set(partition.suppressed_semantic_boundaries)
    # 初始decision含旧长度保护标记。先去除其临时前缀，再依最终选点重建，
    # 不能把已撤销的旧长度点继续标成selected或让其reason误称仍在切分。
    final_gaps = []
    for gap in initial.gaps:
        reason = gap.reason.removeprefix("length_limit_after_")
        if gap.index in lengths:
            reason = "length_limit_after_" + reason
        elif gap.index in suppressed:
            reason = "structure_suppressed_semantic"
        final_gaps.append(replace(gap, selected=gap.index in semantic | lengths, reason=reason))
    if not units:
        status, reasons = "empty", ["empty_text"]
    elif len(units) == 1:
        status, reasons = "single_unit", ["single_unit"]
    else:
        status = "scored_split" if semantic else "scored_keep"
        reasons = ["semantic_boundary_selected" if semantic else "no_semantic_boundary"]
    if any(gap.reason.removeprefix("length_limit_after_") == "min_segment_nonspace_chars" for gap in initial.gaps):
        reasons.append("min_segment_nonspace_chars")
    if lengths:
        status = "length_split"
        reasons.append("max_segment_tokens")
        if any(sum(c not in WHITE for c in segment.raw_text) < config.min_segment_nonspace_chars
               for segment in partition.segments):
            reasons.append("short_segment_due_to_limit")
    if suppressed:
        reasons.append("structure_semantic_boundary_suppressed")
    if partition.forced_structure_boundaries:
        reasons.append("forced_structure_boundary")
    decision = SemanticDecision(status, tuple(reasons), tuple(final_gaps),
                                partition.semantic_boundaries, partition.length_boundaries)
    trace = StructureTrace(config.structure_mode, config.length_strategy, initial, partition.structure,
                           partition.suppressed_semantic_boundaries, partition.forced_structure_boundaries,
                           partition.protected_blocks, partition.block_relaxations)
    return SemanticDispatchResult(partition.segments, decision, trace)
