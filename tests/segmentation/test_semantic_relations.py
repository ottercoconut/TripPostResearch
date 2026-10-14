"""显式语义关系禁止切规则的合成用例；不含来源帖子或真实UGC材料。"""

import pytest

from tourism_ugc_study.segmentation.contracts import GapScore, SemanticDecision, TextUnit
from tourism_ugc_study.segmentation.semantic_config import CoherentSemanticConfig
from tourism_ugc_study.segmentation.semantic_meaning import partition_by_meaning
from tourism_ugc_study.segmentation.semantic_relations import analyze_gap_relations, relation_reasons


class CharacterCounter:
    """每码点计一token的合成计数器。"""

    def count(self, text):
        """只为不变量构造可手算的长度。"""
        return len(text)


@pytest.mark.parametrize("right", ["否则因为距离太近，打车可能被拒载。", "不如晚上九点以后再来。",
                                   "相比之下，另一家店的服务慢很多。", "即便不进去，在外面拍照也很好看。",
                                   "所以建议提前一天预约。", "而且排队的人非常多。", "▪️因此不建议周末前往。"])
def test_strong_dependent_connective_blocks_cut_before_it(right):
    """强依赖词开头的后句离开前文会改变含义；项目符号不掩盖依赖词。"""
    assert "before_dependent_connective" in relation_reasons("建议提前询问机场是否接送。", right)


@pytest.mark.parametrize("right", ["但是爬山真的很累。", "不过我觉得这里没什么意思。", "然后我们去了海边。",
                                   "还有一家面馆也不错。", "如果不赶时间，坐公交也可以。"])
def test_weak_transitions_are_left_to_the_semantic_layer(right):
    """弱转折和自带结论的条件句常是真正的转折点，不在禁止切之列。"""
    assert relation_reasons("这里的景色很不错。", right) == ()


@pytest.mark.parametrize("left,expected", [
    ("但如果真的很喜欢人少和安静的感觉", True),
    ("假如时间比较充裕", True),
    ("如果距离不是很远，大家也可以走过去", False),
    ("如果下雨就改去博物馆。", False),
])
def test_open_condition_keeps_its_conclusion(left, expected):
    """条件句没有分句标点也未结束时，结论在后句，不能切开。"""
    assert ("after_open_condition" in relation_reasons(left, "晚上再去会更舒服。")) is expected


@pytest.mark.parametrize("right,expected", [("（下雨的时候拍的，没有图）", True), ("(毕竟中午才开门)", True),
                                            ("（一）交通方式", False), ("(2) 住宿推荐", False)])
def test_parenthetical_note_follows_previous_line_but_numbered_heading_does_not(right, expected):
    """独立成行的括号说明附着上一条；括号编号是题头，不受此规则约束。"""
    assert ("before_parenthetical_note" in relation_reasons("这家店的糖水很好喝。", right)) is expected


def test_relations_are_reported_per_existing_gap():
    """输出与gap一一对应，不新增或移动gap。"""
    text = "但如果喜欢人少\n不如晚上再去。\n（拍于夜里）"
    units = (TextUnit(0, 7, 7), TextUnit(8, 15, 7), TextUnit(16, 22, 6))
    assert analyze_gap_relations(text, units) == (
        ("before_dependent_connective", "after_open_condition"), ("before_parenthetical_note",))
    assert analyze_gap_relations("独立文字", (TextUnit(0, 4, 4),)) == ()


def test_model_selected_cut_is_vetoed_by_relation_and_length_cannot_override():
    """模型选中的切点遇到关系保护被否决；长度兜底也只能另找允许位置。"""
    parts = ["建议提前询问机场是否提供接送服务。", "否则因为距离太近，打车去酒店可能被拒载。", "第二件事是关于住宿的安排。"]
    text = "\n".join(parts)
    units, cursor = [], 0
    for part in parts:
        units.append(TextUnit(cursor, cursor + len(part), len(part)))
        cursor += len(part) + 1
    gaps = tuple(GapScore(i, 0.2, 0.3, True, True, "semantic_depth") for i in range(2))
    decision = SemanticDecision("scored_split", (), gaps, (0, 1), ())
    config = CoherentSemanticConfig(unit_max_tokens=32, max_segment_tokens=512, edge_segment_nonspace_chars=1)
    result = partition_by_meaning(text, units, decision, config, CharacterCounter())
    assert [s.raw_text for s in result.segments] == ["\n".join(parts[:2]), parts[2]]
    assert "rejected_semantic_dependency" in result.gap_decisions[0].reason_codes
    assert "before_dependent_connective" in result.gap_decisions[0].reason_codes
    tight = CoherentSemanticConfig(unit_max_tokens=30, max_segment_tokens=30, edge_segment_nonspace_chars=1)
    limited = partition_by_meaning(text, units, decision, tight, CharacterCounter())
    assert limited.segments[0].raw_text == "\n".join(parts[:2])
    assert limited.length_exceptions == (1,)
