"""开发候选：在固定语义评分上保护结构，并全局选择长度切点。

与已发布的0.1算法隔离；本模块不读取帖子ID、模型、缓存或持久化状态。
调用者提供已验证的候选单元与评分。结构约束只影响既有gap，不增加或改写
原文；无法同时满足结构与硬长度时保留原文并显式记录被迫的结构切断。
本模块的结果不是正式运行包，必须经过开发复核及独立验证才可接入发布。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Sequence

from .contracts import SemanticDecision, TextSegment, TextUnit
from .integrity import WHITE, validate
from .semantic_config import SemanticConfig
from .semantic_structure import GapStructure, analyze_gap_structure


class TokenCounter(Protocol):
    """只依赖完整原文的精确token计数，不加载模型。"""
    def count(self, text: str) -> int:
        """计数必须为非负整数，非空原文不得计零。"""
        ...


@dataclass(frozen=True)
class BlockRelaxation:
    """结构块因硬长度退让的证据，位置为原文码点。

    token_count记录原块精确长度；retained_end为缩减后仍保护的终点，None
    表示题头至首句也不可保护或该块没有首句。发现失败与超限退让不可混同。
    """
    start: int
    end: int
    kind: str
    reason: str
    token_count: int
    retained_end: int | None


@dataclass(frozen=True)
class StructuredPartition:
    """候选切分及其解释；所有边界是零基gap序号，不是字符位置。"""
    segments: tuple[TextSegment, ...]
    semantic_boundaries: tuple[int, ...]
    length_boundaries: tuple[int, ...]
    suppressed_semantic_boundaries: tuple[int, ...]
    forced_structure_boundaries: tuple[int, ...]
    structure: tuple[GapStructure, ...]
    protected_blocks: tuple[tuple[int, int, str], ...] = ()
    block_relaxations: tuple[BlockRelaxation, ...] = ()


def partition_with_structure(
    text: str, units: Sequence[TextUnit], decision: SemanticDecision,
    config: SemanticConfig, counter: TokenCounter, *, mode: str = "structure",
    length_strategy: str = "global",
) -> StructuredPartition:
    """以自然边界过滤语义切点，再比较贪心与全局长度策略。

    输入须由现行choose_boundaries验证并评分，仍复核跨度、gap身份和输出覆盖。
    blocks模式额外保护可装入当前段长预算的完整重复条目，context还保护
    短题头链及尾部元数据与正文的附着；超长条目只尝试
    保护题头至首个实质句；两者都装不下时不强制整块保护。
    全局策略对各相邻语义边界间的全部可行连续区间动态规划，依序最小化结构
    罚分、短片段数、片段数和切点语义代价。不给单元token和作硬长度近似，
    也不因遇到一个超限前缀提前停止搜索。最坏为区间单元数平方的计数次数；
    计数仅在调用内缓存。等长单元下总分词输入量最坏为立方，不能把平方次
    调用当平方耗时。global与greedy同时改变搜索及排序目标，不能把比较收益
    单独归因于动态规划。无可表示路径则整帖失败，不返回半帖。
    """
    if length_strategy not in {"global", "greedy"}:
        raise ValueError("unknown_partition_length_strategy")
    SemanticConfig.__post_init__(config)
    if len(decision.gaps) != max(0, len(units)-1) or any(g.index != i for i,g in enumerate(decision.gaps)):
        raise ValueError("partition_gap_identity_mismatch")
    if any(type(i) is not int or not 0 <= i < len(decision.gaps) for i in decision.semantic_boundaries):
        raise ValueError("partition_boundary_identity_mismatch")
    unit_segments = [dict(seg_id=i+1,raw_text=text[u.start:u.end],start=u.start,end=u.end) for i,u in enumerate(units)]
    validate(text, unit_segments)
    features = analyze_gap_structure(text, units, mode="structure" if mode in {"blocks", "context"} else mode)
    protected_blocks = []
    relaxations = []
    if mode in {"blocks", "context"}:
        from .semantic_structure import discover_structure_blocks
        blocks = discover_structure_blocks(text, units)
        if mode == "context":
            from .semantic_structure import discover_context_blocks
            blocks = (*blocks, *discover_context_blocks(text, units))
        for block in blocks:
            span = (block.start, block.end)
            count = counter.count(text[span[0]:span[1]])
            if type(count) is not int or count <= 0:
                raise ValueError("invalid_partition_token_count")
            if count > config.max_segment_tokens:
                original_count = count
                first = [item for item in block.evidence if item[0] == "first_substantive_sentence"]
                if not first:
                    relaxations.append(BlockRelaxation(block.start, block.end, block.kind,
                        "whole_block_over_token_limit", original_count, None))
                    continue
                span = (block.start, first[0][2])
                count = counter.count(text[span[0]:span[1]])
                if type(count) is not int or count <= 0:
                    raise ValueError("invalid_partition_token_count")
                if count > config.max_segment_tokens:
                    relaxations.append(BlockRelaxation(block.start, block.end, block.kind,
                        "whole_block_over_token_limit", original_count, None))
                    continue
                relaxations.append(BlockRelaxation(block.start, block.end, block.kind,
                    "whole_block_over_token_limit", original_count, span[1]))
            protected_blocks.append((*span, block.kind))
        amended = []
        for feature in features:
            active = [block for block in protected_blocks if block[0] < units[feature.index].end < block[1]]
            if active:
                feature = replace(feature, penalty=feature.penalty + 12,
                    reasons=tuple(dict.fromkeys((*feature.reasons, *(f"inside_{kind}" for _,_,kind in active)))),
                    evidence_spans=(*feature.evidence_spans, *((f"inside_{kind}", a, b) for a,b,kind in active)))
            amended.append(feature)
        features = tuple(amended)
    semantic = tuple(i for i in decision.semantic_boundaries if features[i].penalty == 0)
    suppressed = tuple(i for i in decision.semantic_boundaries if features[i].penalty != 0)
    prefix = [0]
    for c in text:
        prefix.append(prefix[-1] + (c not in WHITE))
    counts: dict[tuple[int,int],int] = {}

    def tokens(first: int, last: int) -> int:
        """重计连续原文，而非简单加总子单元的BPE数量。"""
        key=first,last
        if key not in counts:
            value=counter.count(text[units[first].start:units[last].end])
            if type(value) is not int or value <= 0:
                raise ValueError("invalid_partition_token_count")
            counts[key]=value
        return counts[key]

    def short(first: int, last: int) -> bool:
        """只有长度拆分产生的短段计罚，整帖本身很短仍可原样保留。"""
        return prefix[units[last].end]-prefix[units[first].start] < config.min_segment_nonspace_chars

    lengths: list[int] = []
    start=0
    for end in (*semantic, len(units)-1) if units else ():
        if tokens(start,end) <= config.max_segment_tokens:
            start=end+1
            continue
        if length_strategy == "greedy":
            while tokens(start,end) > config.max_segment_tokens:
                pool=[i for i in range(start,end) if tokens(start,i) <= config.max_segment_tokens]
                if not pool:
                    raise ValueError("unsplittable_segment_limit")
                chosen=min(pool,key=lambda i:(features[i].penalty,short(start,i) or short(i+1,end),
                                             -decision.gaps[i].depth,decision.gaps[i].cohesion,-i))
                lengths.append(chosen)
                start=chosen+1
        else:
            # dp[j]表示[start,j)全部单元的最优完整分段；最终端点不收切点罚分。
            dp={start:((0,0,0,0.0), ())}
            for stop in range(start+1,end+2):
                choices=[]
                for first in range(start,stop):
                    if first not in dp or tokens(first,stop-1)>config.max_segment_tokens:
                        continue
                    previous,path=dp[first]
                    internal=stop<=end
                    gap=stop-1
                    penalty=features[gap].penalty if internal else 0
                    semantic_cost=decision.gaps[gap].cohesion-decision.gaps[gap].depth if internal else 0.0
                    cost=(previous[0]+penalty, previous[1]+int(short(first,stop-1)),
                          previous[2]+1, previous[3]+semantic_cost)
                    next_path=(*path,gap) if internal else path
                    choices.append((cost,next_path))
                if choices:
                    # 在当前状态同分时按路径字典序。浮点累计可使已剪枝的
                    # 前缀在更后状态舍入同分，故不声称全体完整路径同分最早。
                    dp[stop]=min(choices)
            if end+1 not in dp:
                raise ValueError("unsplittable_segment_limit")
            lengths.extend(dp[end+1][1])
        start=end+1
    cuts=sorted((*semantic,*lengths))
    segments=[]
    first=0
    for last in (*cuts,len(units)-1) if units else ():
        begin,end=units[first].start,units[last].end
        if tokens(first,last)>config.max_segment_tokens:
            raise ValueError("partition_length_invariant_failed")
        segments.append(TextSegment(len(segments)+1,text[begin:end],begin,end))
        first=last+1
    validate(text,[dict(seg_id=s.seg_id,raw_text=s.raw_text,start=s.start,end=s.end) for s in segments])
    return StructuredPartition(tuple(segments),semantic,tuple(lengths),suppressed,
                               tuple(i for i in cuts if features[i].penalty),features,
                               tuple(protected_blocks),tuple(relaxations))
