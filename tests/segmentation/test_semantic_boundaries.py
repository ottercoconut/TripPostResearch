"""语义边界的手算、短帖退化、长度约束与无损测试；全部使用自造文本。"""

from dataclasses import asdict, replace
import math

import numpy as np
import pytest

from tourism_ugc_study.segmentation.contracts import TextUnit
from tourism_ugc_study.segmentation.integrity import validate
from tourism_ugc_study.segmentation.semantic_boundaries import (
    _depth_scores,
    _rank_candidates,
    _valley_candidates,
    choose_boundaries,
)
from tourism_ugc_study.segmentation.semantic_config import SemanticConfig


class CharacterCounter:
    """合成测试tokenizer：每个码点一个token，可直接手算段长。"""

    def count(self, text: str) -> int:
        """精确返回合成token数，不模拟真实模型语义。"""
        return len(text)


def make_units(parts, separator=""):
    """从依次给定的自造片段建立不同span，重复文字也不用find定位。"""
    text, units = separator.join(parts), []
    offset = 0
    for part in parts:
        units.append(TextUnit(start=offset, end=offset + len(part), token_count=len(part)))
        offset += len(part) + len(separator)
    return text, tuple(units)


def test_empty_and_single_unit_do_not_need_vectors():
    """短帖退化路径不伪造评分，同时保证原文与码点span正确。"""
    segments, decision = choose_boundaries(" \r\n", (), None, SemanticConfig(), CharacterCounter())
    assert segments == () and decision.status == "empty" and decision.gaps == ()
    text = "\t海边真好看。\r\n"
    unit = TextUnit(start=1, end=7, token_count=6)
    segments, decision = choose_boundaries(text, (unit,), None, SemanticConfig(), CharacterCounter())
    assert decision.status == "single_unit" and decision.gaps == ()
    assert [(segment.start, segment.end, segment.raw_text) for segment in segments] == [(1, 7, "海边真好看。")]
    validate(text, [asdict(segment) for segment in segments])


@pytest.mark.parametrize("cosine,expected", [(0.9, "scored_keep"), (0.2, "scored_split"), (0.55, "scored_split")])
def test_two_units_use_absolute_cohesion_even_without_a_depth_valley(cosine, expected):
    """一个gap的深度严格为0，短帖仍可根据已配置的绝对阈值切开。"""
    text, units = make_units(["甲" * 15 + "。", "乙" * 15 + "。"])
    vectors = np.array([[1, 0], [cosine, math.sqrt(1 - cosine ** 2)]])
    segments, decision = choose_boundaries(text, units, vectors, SemanticConfig(), CharacterCounter())
    assert decision.status == expected
    assert decision.gaps[0].depth == 0
    assert decision.gaps[0].cohesion == pytest.approx(cosine)
    assert len(segments) == (2 if expected == "scored_split" else 1)
    if expected == "scored_split":
        assert decision.semantic_boundaries == (0,)
        assert decision.gaps[0].reason == "semantic_absolute_cohesion"


def test_three_units_match_hand_calculated_endpoint_valley():
    """N=3不丢弃端点：c=(0,1/√2)，d=(1/√2,0)，选首gap。"""
    text, units = make_units(["甲" * 16, "乙" * 16, "丙" * 16])
    segments, decision = choose_boundaries(text, units, [[1, 0], [0, 1], [0, 1]], SemanticConfig(), CharacterCounter())
    assert tuple(gap.cohesion for gap in decision.gaps) == pytest.approx((0, 1 / math.sqrt(2)))
    assert tuple(gap.depth for gap in decision.gaps) == pytest.approx((1 / math.sqrt(2), 0))
    assert decision.semantic_boundaries == (0,)
    assert [segment.raw_text for segment in segments] == ["甲" * 16, "乙" * 16 + "丙" * 16]


def test_relative_depth_can_select_a_valley_above_absolute_threshold():
    """较高的绝对连贯度不妨碍显著相对谷值触发边界。"""
    text, units = make_units([char * 16 for char in "甲乙丙丁"])
    # 相邻余弦=(0.8,0.2,0.8)，window=1时每侧仅一个单元。
    angles = np.cumsum([0, math.acos(0.8), math.acos(0.2), math.acos(0.8)])
    vectors = np.column_stack((np.cos(angles), np.sin(angles)))
    config = replace(SemanticConfig(), window_tokens=1, absolute_cohesion_threshold=-1)
    _, decision = choose_boundaries(text, units, vectors, config, CharacterCounter())
    assert tuple(gap.cohesion for gap in decision.gaps) == pytest.approx((0.8, 0.2, 0.8))
    assert tuple(gap.depth for gap in decision.gaps) == pytest.approx((0, 1.2, 0))
    assert decision.semantic_boundaries == (1,)
    assert decision.gaps[1].reason == "semantic_depth"


def test_short_fragments_are_suppressed_without_removing_punctuation():
    """两个很短单元可以评分，但最小长度保护会保留整帖与全部标点。"""
    text, units = make_units(["好看！", "方便！"])
    segments, decision = choose_boundaries(text, units, [[1, 0], [0, 1]], SemanticConfig(), CharacterCounter())
    assert len(segments) == 1 and segments[0].raw_text == text
    assert decision.status == "scored_keep"
    assert decision.gaps[0].reason == "min_segment_nonspace_chars"
    assert "min_segment_nonspace_chars" in decision.reason_codes


def test_shortness_is_checked_against_already_accepted_neighbours():
    """两个强候选间只有6字时只接受一个，避免后续边界把现有段切碎。"""
    text, units = make_units(["甲" * 12, "乙" * 3, "丙" * 3, "丁" * 12])
    config = replace(SemanticConfig(), window_tokens=1)
    segments, decision = choose_boundaries(text, units, [[1, 0], [0, 1], [0, 1], [1, 0]], config, CharacterCounter())
    assert decision.semantic_boundaries == (0,)
    assert decision.gaps[2].reason == "min_segment_nonspace_chars"
    assert [len(segment.raw_text) for segment in segments] == [12, 18]


@pytest.mark.parametrize("curve,expected", [
    ([0.2], (0,)),
    ([0.2, 0.2, 0.2, 0.2], (1,)),
    ([0.9, 0.2, 0.2, 0.2, 0.9], (2,)),
    ([0.5, 0.5000006, 0.5000012], (0,)),
    ([0.9, 0.2, 0.9, 0.2, 0.9], (1, 3)),
])
def test_valley_platform_uses_one_representative_and_handles_drift(curve, expected):
    """手给曲线直接检验平台极差、端点低谷及中点规则。"""
    assert _valley_candidates(curve, 1e-6) == expected


def test_depth_handles_epsilon_drift_without_replacing_peak_by_endpoint():
    """逐步小漂移可以被扫描，但峰值必须仍是访问区间的最大值。"""
    assert _depth_scores((0.8, 0.2, 0.8), 1e-6) == pytest.approx((0, 1.2, 0))
    assert _depth_scores((0.5, 0.5000006, 0.5000012), 1e-6) == pytest.approx((0.0000012, 0.0000006, 0))


def test_epsilon_ties_use_transitive_anchored_groups_then_document_position():
    """微小浮点误差不应抢走位置并列规则，连续漂移也不能无限扩展同分组。"""
    cohesion = (0.5, 0.5000006, 0.5000012)
    assert _rank_candidates((2, 0, 1), cohesion, (1, 1, 1), 1e-6) == (0, 1, 2)
    assert _rank_candidates((2, 0, 1), cohesion, (1, 1, 1), 1e-6, prefer_later=True) == (1, 0, 2)
    assert _rank_candidates((2, 0, 1), (0.5, 0.1, 0.0), (1, 0.9999994, 0.9999988), 1e-6) == (1, 0, 2)


def test_identical_vectors_have_no_semantic_cuts_but_can_require_length_cuts():
    """平坦高相似度不会强造语义边界，硬上限另行标记并在同分时靠后切。"""
    text, units = make_units([char * 16 for char in "甲乙丙丁戊己"])
    config = replace(SemanticConfig(), unit_max_tokens=16, max_segment_tokens=32)
    segments, decision = choose_boundaries(text, units, np.ones((6, 3)), config, CharacterCounter())
    assert decision.semantic_boundaries == () and decision.length_boundaries == (1, 3)
    assert decision.status == "length_split"
    assert [len(segment.raw_text) for segment in segments] == [32, 32, 32]
    assert tuple(gap.cohesion for gap in decision.gaps) == pytest.approx((1,) * 5)
    assert tuple(gap.depth for gap in decision.gaps) == pytest.approx((0,) * 5)
    assert [gap.index for gap in decision.gaps if gap.selected] == [1, 3]
    validate(text, [asdict(segment) for segment in segments])


def test_exact_span_count_does_not_assume_additivity_or_monotonic_prefix_counts():
    """中间前缀超限而更长前缀合格时仍能选择后者，不能把单元数直接相加。"""
    class NonMonotonicCounter:
        """明确构造非单调tokenizer；只断言计数契约，不冒充真实BPE模型。"""

        def count(self, text):
            """三个单元组合可比两个单元组合更省token。"""
            return {12: 2, 24: 10, 36: 6, 48: 8}[len(text)]

    text, units = make_units([char * 12 for char in "甲乙丙丁"])
    units = tuple(replace(unit, token_count=2) for unit in units)
    config = replace(SemanticConfig(), unit_max_tokens=2, max_segment_tokens=6)
    segments, decision = choose_boundaries(text, units, np.ones((4, 2)), config, NonMonotonicCounter())
    assert decision.length_boundaries == (2,)
    assert [len(segment.raw_text) for segment in segments] == [36, 12]
    assert [NonMonotonicCounter().count(segment.raw_text) for segment in segments] == [6, 2]


def test_length_limit_can_allow_a_short_tail_and_records_it():
    """无法同时满足最小段长时优先硬上限，结果仍无损且明确记录短尾。"""
    text, units = make_units(["甲" * 12, "乙" * 3, "丙" * 3])
    config = replace(SemanticConfig(), unit_max_tokens=12, max_segment_tokens=12)
    segments, decision = choose_boundaries(text, units, np.ones((3, 2)), config, CharacterCounter())
    assert decision.length_boundaries == (0,)
    assert [len(segment.raw_text) for segment in segments] == [12, 6]
    assert "short_segment_due_to_limit" in decision.reason_codes


def test_single_unit_over_hard_limit_fails_without_truncating():
    """没有可行单元边界时明确失败，单单元路径不能绕过硬上限。"""
    text, units = make_units(["甲" * 65])
    config = replace(SemanticConfig(), max_segment_tokens=64)
    with pytest.raises(ValueError, match="^unsplittable_segment_limit$"):
        choose_boundaries(text, units, None, config, CharacterCounter())


@pytest.mark.parametrize("vectors,error", [
    (None, "invalid_unit_vector_shape"),
    ([[1, 0]], "invalid_unit_vector_shape"),
    ([[], []], "invalid_unit_vector_shape"),
    ([[1, 0], [0, 0]], "invalid_unit_vectors"),
    ([[1, 0], [0, float("nan")]], "invalid_unit_vectors"),
    ([[1, 0], [0, float("inf")]], "invalid_unit_vectors"),
    ([[1, 0], [0, 1e-14]], "invalid_unit_vectors"),
    ([["1", "0"], ["0", "1"]], "invalid_unit_vectors"),
    ([[1 + 1j, 0], [0, 1]], "invalid_unit_vectors"),
    ([[True, False], [False, True]], "invalid_unit_vectors"),
])
def test_invalid_vectors_fail_instead_of_silently_scoring(vectors, error):
    """行数、维度、有限性与范数均为算法入口契约。"""
    text, units = make_units(["甲" * 12, "乙" * 12])
    with pytest.raises(ValueError, match=f"^{error}$"):
        choose_boundaries(text, units, vectors, SemanticConfig(), CharacterCounter())


def test_cancelled_window_mean_fails_even_when_every_unit_vector_is_valid():
    """两个反向有效向量的均值为零，不能伪造低相似度切点。"""
    text, units = make_units([char * 16 for char in "甲乙丙"])
    with pytest.raises(ValueError, match="^invalid_window_vector$"):
        choose_boundaries(text, units, [[1, 0], [-1, 0], [0, 1]], SemanticConfig(), CharacterCounter())


def test_repeat_text_emoji_spaces_and_feff_preserve_offsets():
    """全相同句子仍使用独立span；合并保留内部空白，切开只允许空白间隙。"""
    parts = ["👨‍👩‍👧‍👦出游真开心真开心。", "👨‍👩‍👧‍👦出游真开心真开心。", "\ufeff换乘公交前往不同地点。"]
    text, units = make_units(parts, separator="\r\n \t")
    vectors = [[1, 0], [1, 0], [0, 1]]
    segments, decision = choose_boundaries(text, units, vectors, SemanticConfig(), CharacterCounter())
    validate(text, [asdict(segment) for segment in segments])
    assert "\r\n \t" in segments[0].raw_text
    assert segments[-1].raw_text.startswith("\ufeff")
    assert decision.semantic_boundaries == (1,)
    assert segments[1].start == units[2].start


@pytest.mark.parametrize("text,units,error", [
    ("甲", (), "incomplete_unit_coverage"),
    ("甲乙", (TextUnit(0, 1, 1),), "incomplete_unit_coverage"),
    ("甲乙", (TextUnit(1, 2, 1),), "incomplete_unit_coverage"),
    ("甲乙", (TextUnit(0, 2, 2), TextUnit(1, 2, 1)), "invalid_unit_span"),
    ("a\u0301", (TextUnit(0, 1, 1), TextUnit(1, 2, 1)), "invalid_unit_grapheme_boundary"),
    (" ", (TextUnit(0, 1, 1),), "empty_text_unit"),
    ("甲", (TextUnit(0, 1, 2),), "unit_token_count_mismatch"),
])
def test_invalid_units_fail_even_when_vectors_would_be_skipped(text, units, error):
    """任何缺字、重叠或字素拆坏在N=0/1时也不能成为成功结果。"""
    with pytest.raises(ValueError, match=f"^{error}$"):
        choose_boundaries(text, units, None, SemanticConfig(), CharacterCounter())


def test_invalid_window_budget_is_rejected_even_on_empty_input():
    """即使外部绕过冻结配置构造，纯函数仍拒绝零窗口预算。"""
    config = SemanticConfig()
    object.__setattr__(config, "window_tokens", 0)
    with pytest.raises(ValueError, match="^invalid_algorithm_parameter:window_tokens$"):
        choose_boundaries("", (), None, config, CharacterCounter())


def test_replay_is_exactly_deterministic_and_does_not_modify_vectors():
    """同向量同配置逐项一致，输入数组归一化不得原位改写。"""
    text, units = make_units([char * 15 + "。" for char in "甲乙丙丁戊己庚辛壬癸"])
    rng = np.random.default_rng(20261011)
    vectors = rng.normal(size=(10, 12))
    before = vectors.copy()
    config = replace(SemanticConfig(), max_segment_tokens=64)
    first = choose_boundaries(text, units, vectors, config, CharacterCounter())
    second = choose_boundaries(text, units, vectors, config, CharacterCounter())
    assert first == second
    np.testing.assert_array_equal(vectors, before)
    segments, decision = first
    validate(text, [asdict(segment) for segment in segments])
    assert all(len(segment.raw_text) <= 64 for segment in segments)
    assert [segment.seg_id for segment in segments] == list(range(1, len(segments) + 1))
    assert len(segments) == 1 + len(decision.semantic_boundaries) + len(decision.length_boundaries)
