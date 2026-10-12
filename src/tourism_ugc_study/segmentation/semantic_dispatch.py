"""按显式算法配置分派纯边界计算，隔离历史重放与当前连贯块规则。

本层不读取模型、缓存、数据库或帖子ID。历史配置保持原结果；当前规则
复用同一评分，再由已采纳v4分区器产生结果。来源证据独立保存，不向旧
SemanticDecision或逐帖summary增加字段，也不将结构罚分称为质量金标。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Sequence

from .contracts import SemanticDecision, TextSegment, TextUnit
from .integrity import WHITE
from .semantic_boundaries import choose_boundaries
from .semantic_config import (COHERENT_PARTITION_ID, CoherentSemanticConfig, SemanticConfig,
                              StructuredSemanticConfig, semantic_rule_identity)
from .semantic_meaning import MeaningGapDecision, MeaningPartition, partition_by_meaning
from .semantic_partition import BlockRelaxation, partition_with_structure
from .semantic_structure import GapStructure, StructureBlock


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
class MeaningTrace:
    """冻结v4的完整边界来源，独立于兼容发布摘要保存。

    semantic_boundaries和structural_boundaries为原算法来源，可重叠；最终
    SemanticDecision.semantic_boundaries则是二者并集，只计一次非长度切点。
    此处保留原实验身份、初始评分和全部保护证据，不修改原算法返回值。
    """

    experiment_id: str
    initial_decision: SemanticDecision
    semantic_boundaries: tuple[int, ...]
    structural_boundaries: tuple[int, ...]
    length_boundaries: tuple[int, ...]
    suppressed_semantic_boundaries: tuple[int, ...]
    forced_structure_boundaries: tuple[int, ...]
    gap_decisions: tuple[MeaningGapDecision, ...]
    protected_blocks: tuple[StructureBlock, ...]
    unavailable_structural_spans: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class SemanticDispatchResult:
    """同一配置执行的完整片段、最终决策与可选结构证据。

    片段和decision必须描述同一组切点；两种trace仅按各自配置提供，0.1均为空。
    失败沿用领域ValueError或系统EmbeddingError，不返回部分成功结果。
    """

    segments: tuple[TextSegment, ...]
    decision: SemanticDecision
    structure_trace: StructureTrace | None = None
    meaning_trace: MeaningTrace | None = None


def _coherent_result(initial: SemanticDecision, partition: MeaningPartition,
                     config: CoherentSemanticConfig, unit_count: int) -> SemanticDispatchResult:
    """适配冻结结果到既有发布契约，不重新选点或改动正文跨度。

    非长度来源取并集以避免重叠计数；真实来源全部保存在MeaningTrace。
    gap.selected、最终状态和摘要均跟随最终片段，不沿用临时长度选择。
    """
    if partition.experiment_id != COHERENT_PARTITION_ID:
        raise ValueError("coherent_partition_identity_mismatch")
    nonlength = tuple(sorted(set(partition.semantic_boundaries) | set(partition.structural_boundaries)))
    lengths = set(partition.length_boundaries)
    selected = set(nonlength) | lengths
    if set(nonlength) & lengths:
        raise ValueError("coherent_boundary_categories_overlap")
    gaps = []
    for gap in initial.gaps:
        reason = gap.reason.removeprefix("length_limit_after_")
        if gap.index in lengths:
            reason = "meaning_hard_token_limit"
        elif gap.index in selected:
            reason = "meaning_boundary_selected"
        elif gap.index in partition.suppressed_semantic_boundaries:
            reason = "meaning_boundary_suppressed"
        gaps.append(replace(gap, selected=gap.index in selected, reason=reason))
    if unit_count <= 1:
        status = "empty" if unit_count == 0 else "single_unit"
        reasons = ["empty_text" if unit_count == 0 else "single_unit"]
    else:
        status = "scored_split" if nonlength else "scored_keep"
        reasons = ["meaning_boundary_selected" if nonlength else "no_meaning_boundary"]
    if lengths:
        status = "length_split"
        reasons.append("max_segment_tokens")
        if any(sum(char not in WHITE for char in segment.raw_text) < config.min_segment_nonspace_chars
               for segment in partition.segments):
            reasons.append("short_segment_due_to_limit")
    if partition.suppressed_semantic_boundaries:
        reasons.append("meaning_semantic_boundary_suppressed")
    if partition.forced_structure_boundaries:
        reasons.append("forced_structure_boundary")
    if partition.unavailable_structural_spans:
        reasons.append("structural_boundary_unavailable")
    decision = SemanticDecision(status, tuple(reasons), tuple(gaps), nonlength, partition.length_boundaries)
    trace = MeaningTrace(partition.experiment_id, initial, partition.semantic_boundaries,
                         partition.structural_boundaries, partition.length_boundaries,
                         partition.suppressed_semantic_boundaries, partition.forced_structure_boundaries,
                         partition.gap_decisions, partition.protected_blocks, partition.unavailable_structural_spans)
    return SemanticDispatchResult(partition.segments, decision, meaning_trace=trace)


def choose_semantic_boundaries(
    text: str, units: Sequence[TextUnit], vectors: object, config: SemanticConfig,
    counter: BoundaryTokenCounter,
) -> SemanticDispatchResult:
    """执行已登记配置的纯切分，并保证最终边界、状态及gap标记相符。

    输入和失败契约沿用choose_boundaries：候选单元须覆盖全部非空白正文、
    端点字素安全、计数精确；N≥2要求有效逐单元向量，N≤1不虚构评分。
    0.1不改其片段或决策。0.2仅重新分区既有gap，硬长度优先于结构偏好，
    被抑制语义点、被迫结构切断和初始决策在独立trace中留证。
    当前连贯块规则原样调用v4分区器，只将语义／结构来源的并集适配为
    发布契约中的非长度边界；原始来源和初始决策另写meaning_trace。
    """
    semantic_rule_identity(config)
    segments, initial = choose_boundaries(text, units, vectors, config, counter)
    if type(config) is SemanticConfig:
        return SemanticDispatchResult(segments, initial)
    if type(config) is CoherentSemanticConfig:
        return _coherent_result(initial, partition_by_meaning(text, units, initial, config, counter),
                                config, len(units))
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
