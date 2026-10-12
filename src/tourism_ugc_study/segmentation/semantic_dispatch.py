"""组合语义评分与冻结连贯块分区；只处理原文、候选及向量。

本层不读取模型、缓存、数据库或帖子ID，完整保存边界来源及保护证据。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .contracts import SemanticDecision, TextSegment, TextUnit, TokenCounter
from .integrity import WHITE
from .semantic_boundaries import choose_boundaries
from .semantic_config import COHERENT_PARTITION_ID, CoherentSemanticConfig, semantic_rule_identity
from .semantic_meaning import MeaningGapDecision, MeaningPartition, partition_by_meaning
from .semantic_structure import StructureBlock


@dataclass(frozen=True)
class MeaningTrace:
    """冻结v4的完整边界来源，独立于发布摘要保存。

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
    """同一配置执行的完整片段、最终决策与完整解释证据。

    片段和decision必须描述同一组切点；成功计算必须提供meaning_trace。
    失败沿用领域ValueError或系统EmbeddingError，不返回部分成功结果。
    """

    segments: tuple[TextSegment, ...]
    decision: SemanticDecision
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
    text: str, units: Sequence[TextUnit], vectors: object, config: CoherentSemanticConfig,
    counter: TokenCounter,
) -> SemanticDispatchResult:
    """计算评分并原样执行冻结v4分区，不依赖帖子ID、模型或缓存。

    候选必须覆盖非空白正文，端点字素安全；N≥2要求有效逐单元向量。
    语义／结构来源并集对应最终非长度切点，原始证据另存meaning_trace。
    非法配置或候选抛ValueError，不返回部分结果。
    """
    semantic_rule_identity(config)
    _, initial = choose_boundaries(text, units, vectors, config, counter)
    return _coherent_result(initial, partition_by_meaning(text, units, initial, config, counter),
                            config, len(units))
