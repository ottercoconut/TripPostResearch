"""按固定向量选择语义边界，再施加精确token长度限制。

本模块依据项目方案独立实现TextTiling的窗口连贯度与谷值思想，不复制上游
实现。输入是完整可读正文、带码点span的候选单元及逐单元向量；仅返回原文
片段与决策证据，不访问模型、缓存、文件或数据库。向量计算固定在CPU
float64；同一输入及配置重复执行具有确定性。语义评分是启发式，不能保证
主题边界正确；非空白覆盖、字素端点、段长及编号属于可验证的不变量。
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from math import isfinite
from typing import Sequence

import numpy as np
import regex

from .contracts import GapScore, SemanticDecision, TextSegment, TextUnit, TokenCounter
from .integrity import WHITE
from .semantic_config import CoherentSemanticConfig


def _validate_config(config: CoherentSemanticConfig) -> None:
    """委托配置自身防御性复验，避免在计算层另维护一份参数范围规则。"""
    if not isinstance(config, CoherentSemanticConfig):
        raise ValueError("invalid_semantic_config_type")
    CoherentSemanticConfig.__post_init__(config)


def _checked_count(counter: TokenCounter, text: str) -> int:
    """拒绝布尔值、负数和非整数计数；失败不回退到字符长度。"""
    count = counter.count(text)
    if type(count) is not int or count < 0 or (text and count == 0):
        raise ValueError("invalid_token_count")
    return count


def _validate_units(
    text: str, units: Sequence[TextUnit], counter: TokenCounter
) -> list[int]:
    """检查候选单元无损且token数可信，返回正文非空白字符数前缀和。

    单元间及正文两端可以有WHITE空白，其他字符不得遗漏。端点必须是完整
    字素边界。验证包括N=0/1，因此空结果不能掩盖遗漏原文或超长单元。
    """
    if not isinstance(text, str):
        raise ValueError("invalid_source_text")
    prefix = [0]
    for char in text:
        prefix.append(prefix[-1] + (char not in WHITE))
    edges = {0, *(match.end() for match in regex.finditer(r"\X", text))}
    previous_end = 0
    for unit in units:
        if type(unit.start) is not int or type(unit.end) is not int:
            raise ValueError("invalid_unit_span")
        if not 0 <= previous_end <= unit.start < unit.end <= len(text):
            raise ValueError("invalid_unit_span")
        if unit.start not in edges or unit.end not in edges:
            raise ValueError("invalid_unit_grapheme_boundary")
        if prefix[unit.start] != prefix[previous_end]:
            raise ValueError("incomplete_unit_coverage")
        if prefix[unit.end] == prefix[unit.start]:
            raise ValueError("empty_text_unit")
        if type(unit.token_count) is not int or unit.token_count <= 0:
            raise ValueError("invalid_unit_token_count")
        if _checked_count(counter, text[unit.start:unit.end]) != unit.token_count:
            raise ValueError("unit_token_count_mismatch")
        previous_end = unit.end
    if prefix[-1] != prefix[previous_end]:
        raise ValueError("incomplete_unit_coverage")
    return prefix


def _normalized_vectors(vectors: object, count: int) -> np.ndarray:
    """转换为CPU float64并逐行归一化；拒绝无效shape、非有限及零向量。"""
    try:
        raw = np.asarray(vectors)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid_unit_vectors") from error
    if raw.ndim != 2 or raw.shape[0] != count or raw.shape[1] == 0:
        raise ValueError("invalid_unit_vector_shape")
    # 不把字符串、布尔值或复数的隐式转换视为合格embedding。
    if raw.dtype.kind not in "iuf":
        raise ValueError("invalid_unit_vectors")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(raw, dtype=np.float64)
        norms = np.linalg.norm(array, axis=1)
    if not np.isfinite(array).all() or not np.isfinite(norms).all():
        raise ValueError("invalid_unit_vectors")
    if (norms <= 1e-12).any():
        raise ValueError("invalid_unit_vectors")
    return array / norms[:, None]


def _cohesion(
    units: Sequence[TextUnit], vectors: np.ndarray, window_tokens: int
) -> tuple[float, ...]:
    """比较每个gap两侧的最近token窗口，不越过帖首尾且每侧至少一单元。

    token前缀和确定窗口，向量前缀和确定算术均值。均值范数接近零意味着
    方向不可定义，必须失败；用零余弦替代会伪造主题变化。
    """
    token_prefix = [0]
    for unit in units:
        token_prefix.append(token_prefix[-1] + unit.token_count)
    sums = np.concatenate((np.zeros((1, vectors.shape[1])), np.cumsum(vectors, axis=0)))
    scores: list[float] = []
    count = len(units)
    for gap in range(count - 1):
        middle = gap + 1
        left_start = max(0, bisect_right(token_prefix, token_prefix[middle] - window_tokens, 0, middle) - 1)
        right_end = min(count, bisect_left(token_prefix, token_prefix[middle] + window_tokens, middle + 1))
        left = (sums[middle] - sums[left_start]) / (middle - left_start)
        right = (sums[right_end] - sums[middle]) / (right_end - middle)
        left_norm, right_norm = float(np.linalg.norm(left)), float(np.linalg.norm(right))
        if not isfinite(left_norm) or not isfinite(right_norm) or min(left_norm, right_norm) <= 1e-12:
            raise ValueError("invalid_window_vector")
        cosine = float(np.dot(left / left_norm, right / right_norm))
        if not isfinite(cosine):
            raise ValueError("invalid_window_vector")
        scores.append(float(np.clip(cosine, -1.0, 1.0)))
    return tuple(scores)


def _depth_scores(cohesion: Sequence[float], epsilon: float) -> tuple[float, ...]:
    """线性扫描两侧近似单调上坡的最大值，返回非负谷值深度。

    epsilon只容忍逐步比较的浮点微扰；保存整个已访问区间的最大值，
    不让多步微小下滑将端点误当作峰顶。单gap的深度严格为0。
    """
    if not cohesion:
        return ()
    left = list(cohesion)
    right = list(cohesion)
    for index in range(1, len(cohesion)):
        if cohesion[index - 1] >= cohesion[index] - epsilon:
            left[index] = max(cohesion[index], left[index - 1])
    for index in range(len(cohesion) - 2, -1, -1):
        if cohesion[index + 1] >= cohesion[index] - epsilon:
            right[index] = max(cohesion[index], right[index + 1])
    return tuple(max(0.0, a - c) + max(0.0, b - c) for a, b, c in zip(left, right, cohesion, strict=True))


def _valley_candidates(cohesion: Sequence[float], epsilon: float) -> tuple[int, ...]:
    """按整组极差构建最大近等平台，仅返回低谷平台的中点代表gap。

    平台外邻值与平台最小值比较，避免连续微小漂移在分组边缘吞掉全部
    候选。帖子端点允许单侧低谷；整条曲线为同一平台时仍只有一个代表。
    """
    candidates: list[int] = []
    start = 0
    while start < len(cohesion):
        end = start
        low = high = cohesion[start]
        while end + 1 < len(cohesion):
            next_low, next_high = min(low, cohesion[end + 1]), max(high, cohesion[end + 1])
            if next_high - next_low > epsilon:
                break
            low, high = next_low, next_high
            end += 1
        left_ok = start == 0 or cohesion[start - 1] > low + epsilon
        right_ok = end == len(cohesion) - 1 or cohesion[end + 1] > low + epsilon
        if left_ok and right_ok:
            candidates.append((start + end) // 2)
        start = end + 1
    return tuple(candidates)


def _rank_candidates(
    candidates: Sequence[int], cohesion: Sequence[float], depths: Sequence[float],
    epsilon: float, *, prefer_later: bool = False,
) -> tuple[int, ...]:
    """深度降序、连贯度升序，容差并列再按位置决定，形成稳定的全序。

    每层排序后以组首数值为锚，组内极差不超过epsilon。不能直接编写
    abs(a-b)<=epsilon的两两比较器，否则A≈B、B≈C而A不≈C会破坏传递性。
    语义候选同分靠前；硬长度候选同分靠后以充分利用可用长度。
    """
    ordered = sorted(candidates, key=lambda index: (-depths[index], index))
    ranked: list[int] = []
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and depths[ordered[start]] - depths[ordered[end]] <= epsilon:
            end += 1
        depth_group = sorted(ordered[start:end], key=lambda index: (cohesion[index], index))
        subgroup_start = 0
        while subgroup_start < len(depth_group):
            subgroup_end = subgroup_start + 1
            while (subgroup_end < len(depth_group)
                   and cohesion[depth_group[subgroup_end]] - cohesion[depth_group[subgroup_start]] <= epsilon):
                subgroup_end += 1
            ranked.extend(sorted(depth_group[subgroup_start:subgroup_end], reverse=prefer_later))
            subgroup_start = subgroup_end
        start = end
    return tuple(ranked)


def _semantic_cuts(
    units: Sequence[TextUnit], nonspace_prefix: Sequence[int], cohesion: Sequence[float],
    depths: Sequence[float], config: CoherentSemanticConfig,
) -> tuple[tuple[int, ...], tuple[GapScore, ...]]:
    """按谷值、余弦、原文位置排序尝试边界，逐次检查新增两侧段的最小长度。"""
    candidates = set(_valley_candidates(cohesion, config.epsilon))
    depth_threshold = max(config.depth_floor, float(np.mean(depths)) + config.depth_std_factor * float(np.std(depths)))
    reasons = ["below_threshold" if index in candidates else "not_local_minimum" for index in range(len(cohesion))]
    selected: list[int] = []
    qualified = [index for index in candidates if depths[index] >= depth_threshold or cohesion[index] <= config.absolute_cohesion_threshold]
    for gap in _rank_candidates(qualified, cohesion, depths, config.epsilon):
        position = bisect_left(selected, gap)
        left_unit = selected[position - 1] + 1 if position else 0
        right_unit = selected[position] if position < len(selected) else len(units) - 1
        left_size = nonspace_prefix[units[gap].end] - nonspace_prefix[units[left_unit].start]
        right_size = nonspace_prefix[units[right_unit].end] - nonspace_prefix[units[gap + 1].start]
        if min(left_size, right_size) < config.min_segment_nonspace_chars:
            reasons[gap] = "min_segment_nonspace_chars"
            continue
        insort(selected, gap)
        reasons[gap] = "semantic_depth" if depths[gap] >= depth_threshold else "semantic_absolute_cohesion"
    selected_set = set(selected)
    gaps = tuple(GapScore(index=index, cohesion=float(cohesion[index]), depth=float(depths[index]), candidate=index in candidates,
                          selected=index in selected_set, reason=reasons[index]) for index in range(len(cohesion)))
    return tuple(selected), gaps


def _length_cuts(
    text: str, units: Sequence[TextUnit], semantic_cuts: Sequence[int], gaps: Sequence[GapScore],
    nonspace_prefix: Sequence[int], config: CoherentSemanticConfig, counter: TokenCounter,
) -> tuple[tuple[int, ...], bool]:
    """对完整原文子串精确重计，在硬上限内优先避免短尾段。

    BPE计数不保证随前缀长度单调，因此遍历全部单元边界，不在首个超限处
    提前停止。候选排序优先深度／低连贯度，全等时取靠后边界。每轮至少
    消耗一个单元，所以有限终止；不可表示时抛错，不截断或返回部分成功。
    """
    cuts: list[int] = []
    short_due_to_limit = False
    start_unit = 0
    counts: dict[tuple[int, int], int] = {}
    cohesion = tuple(gap.cohesion for gap in gaps)
    depths = tuple(gap.depth for gap in gaps)

    def count_span(first: int, last: int) -> int:
        """缓存完整span计数，不把各单元token数加总成段长。"""
        key = first, last
        if key not in counts:
            counts[key] = _checked_count(counter, text[units[first].start:units[last].end])
        return counts[key]

    for end_unit in (*semantic_cuts, len(units) - 1):
        while count_span(start_unit, end_unit) > config.max_segment_tokens:
            feasible = [gap for gap in range(start_unit, end_unit) if count_span(start_unit, gap) <= config.max_segment_tokens]
            if not feasible:
                raise ValueError("unsplittable_segment_limit")
            preferred = [gap for gap in feasible
                         if min(nonspace_prefix[units[gap].end] - nonspace_prefix[units[start_unit].start],
                                nonspace_prefix[units[end_unit].end] - nonspace_prefix[units[gap + 1].start]) >= config.min_segment_nonspace_chars]
            pool = preferred or feasible
            if not preferred:
                short_due_to_limit = True
            gap = _rank_candidates(pool, cohesion, depths, config.epsilon, prefer_later=True)[0]
            cuts.append(gap)
            start_unit = gap + 1
        start_unit = end_unit + 1
    return tuple(cuts), short_due_to_limit


def choose_boundaries(
    text: str, units: Sequence[TextUnit], vectors: object, config: CoherentSemanticConfig,
    token_counter: TokenCounter,
) -> tuple[tuple[TextSegment, ...], SemanticDecision]:
    """从逐单元向量选择无重叠原文片段，返回可重放的完整gap决策。

    N=0/1允许vectors为None且不进行语义评分；N≥2要求每单元一个有效向量。
    TextSegment的start/end为Unicode码点；decision中的两类boundary元组和
    GapScore.index均为零基gap索引，**不是码点位置**。selected表示该gap最终
    被语义或硬长度规则选中。参数、span、计数或向量异常均抛ValueError，
    其消息为稳定原因代码；调用层负责将整帖标为failed，禁止发布部分结果。
    """
    _validate_config(config)
    nonspace_prefix = _validate_units(text, units, token_counter)
    if not units:
        return (), SemanticDecision(status="empty", reason_codes=("empty_text",), gaps=(), semantic_boundaries=(), length_boundaries=())
    gaps: tuple[GapScore, ...] = ()
    semantic_cuts: tuple[int, ...] = ()
    if len(units) >= 2:
        normalized = _normalized_vectors(vectors, len(units))
        cohesion = _cohesion(units, normalized, config.window_tokens)
        depths = _depth_scores(cohesion, config.epsilon)
        semantic_cuts, gaps = _semantic_cuts(units, nonspace_prefix, cohesion, depths, config)
    length_cuts, short_due_to_limit = _length_cuts(text, units, semantic_cuts, gaps, nonspace_prefix, config, token_counter)
    length_set = set(length_cuts)
    gaps = tuple(GapScore(index=gap.index, cohesion=gap.cohesion, depth=gap.depth, candidate=gap.candidate,
                          selected=gap.selected or gap.index in length_set,
                          reason=f"length_limit_after_{gap.reason}" if gap.index in length_set else gap.reason) for gap in gaps)
    segments: list[TextSegment] = []
    start_unit = 0
    for last_unit in (*sorted((*semantic_cuts, *length_cuts)), len(units) - 1):
        start, end = units[start_unit].start, units[last_unit].end
        segments.append(TextSegment(seg_id=len(segments) + 1, raw_text=text[start:end], start=start, end=end))
        start_unit = last_unit + 1
    status = "single_unit" if len(units) == 1 else "scored_split" if semantic_cuts else "scored_keep"
    reasons = ["single_unit" if len(units) == 1 else "semantic_boundary_selected" if semantic_cuts else "no_semantic_boundary"]
    if any(gap.reason == "min_segment_nonspace_chars" for gap in gaps):
        reasons.append("min_segment_nonspace_chars")
    if length_cuts:
        status = "length_split"
        reasons.append("max_segment_tokens")
    if short_due_to_limit:
        reasons.append("short_segment_due_to_limit")
    return tuple(segments), SemanticDecision(status=status, reason_codes=tuple(reasons), gaps=gaps,
                                            semantic_boundaries=semantic_cuts, length_boundaries=length_cuts)
