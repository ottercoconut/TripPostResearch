"""结构候选的保守识别与误判反例；只用自造正文，不加载模型或研究库。"""

from dataclasses import FrozenInstanceError, asdict

import pytest

from tourism_ugc_study.segmentation.contracts import TextUnit
from tourism_ugc_study.segmentation.semantic_structure import (
    analyze_gap_structure, discover_context_blocks, discover_structure_blocks,
)


def _units(parts, separator=""):
    """按顺序定位自造单元，重复正文不使用 find。"""
    text, units, cursor = separator.join(parts), [], 0
    for part in parts:
        units.append(TextUnit(cursor, cursor + len(part), len(part)))
        cursor += len(part) + len(separator)
    return text, tuple(units)


def test_structure_distinguishes_sentence_line_and_artificial_gap():
    """逗号细分不是自然句末，换行与附闭引号的句末可独立消融。"""
    text, units = _units(["这里有海，", "那里有山。", "“公交方便！”", "继续行走"])
    result = analyze_gap_structure(text, units)
    assert [gap.boundary_kind for gap in result] == ["internal", "sentence", "sentence"]
    assert [gap.penalty for gap in result] == [8, 0, 0]
    text, units = _units(["海滨", "山地"], "\r\n\u3000")
    assert analyze_gap_structure(text, units)[0].boundary_kind == "line"


def test_matched_bracket_discourages_inner_sentence_boundary_only():
    """括号中的句号仍可生成单元，但与外部真实句末应有不同风险。"""
    text, units = _units(["公交方便（换乘一次。", "周末有加班车）。", "海边很近。"])
    first, second = analyze_gap_structure(text, units)
    assert "inside_matched_bracket" in first.reasons and first.penalty == 12
    assert "inside_matched_bracket" not in second.reasons and second.penalty == 0
    assert first.evidence_spans[-1] == ("inside_matched_bracket", 4, 17)


def test_unclosed_intro_bracket_does_not_protect_whole_post():
    """UGC 开头漏写闭括号很常见，后文不能因此全部变为括号内部。"""
    text, units = _units(["旅行感受（个人体验", "海边很好看。", "公交很方便。"], "\n")
    assert all("inside_matched_bracket" not in gap.reasons for gap in analyze_gap_structure(text, units))


def test_nested_matched_pair_survives_unclosed_outer_and_is_not_double_charged():
    """未闭外层不取消内层可靠配对，嵌套多个配对只加一次该类罚分。"""
    text, units = _units(["感受（公交【换乘一次。", "很方便】", "海边景色很好。"], "\n")
    gaps = analyze_gap_structure(text, units)
    assert gaps[0].penalty == 12 and gaps[0].reasons.count("inside_matched_bracket") == 1
    assert "inside_matched_bracket" not in gaps[1].reasons
    text, units = _units(["（【海很蓝。", "】）", "继续前行。"])
    assert analyze_gap_structure(text, units)[0].penalty == 12


@pytest.mark.parametrize("heading", [
    "Day2", "📍Day3", "第十二天", "6月3日，星期三，晴", "二、住宿", "2. 行程",
    "注意事项：", "🗺️ 3天路线，沿海走一圈", "📍海滨公园：推荐！", "⛺️山区-茶园",
])
def test_explicit_short_heading_prefers_its_following_description(heading):
    """通用序号、日期及正文题头向后附着，不依赖地点名或 post ID。"""
    text, units = _units(["上一节的完整说明。", heading, "这里保留完整的介绍和出行提示。"], "\n")
    before, after = analyze_gap_structure(text, units)
    assert "before_heading" in before.reasons
    assert "after_heading" in after.reasons and after.penalty == 6


def test_heading_split_by_sentence_punctuation_can_share_line_with_body():
    """正文行首“题头：短标签！”与紧随说明仍应关联。"""
    text, units = _units(["📍海滨公园：推荐！", "树荫很密，步行也方便。"])
    gap = analyze_gap_structure(text, units)[0]
    assert gap.boundary_kind == "sentence" and "after_heading" in gap.reasons


@pytest.mark.parametrize("sentence", [
    "🌊今天去了海边，心情很好。", "公交方便。", "🌊海边好漂亮！", "价格：很便宜。",
    "第一天去了海边，然后坐公交回旅馆。", "👋欢迎来到海边", "❤️喜欢这里",
])
def test_complete_short_sentence_is_not_automatically_a_heading(sentence):
    """短长度、emoji 和领域名词本身都不足以把完整句认成题头。"""
    text, units = _units([sentence, "随后又看到了新的风景。"], "\n")
    assert "after_heading" not in analyze_gap_structure(text, units)[0].reasons


@pytest.mark.parametrize("attribute", ["⌚️时间｜半天", "🎫无门票", "交通：地铁到达", "优点：离海近", "🚗地址 | 山海路八号"])
def test_attribute_line_is_softly_attached_to_prior_context(attribute):
    """明确字段形式得到向前关联证据，但仍允许下游按整体上下文决定。"""
    text, units = _units(["这里是海滨公园。", attribute, "下一节介绍山中的步道。"], "\n")
    first, second = analyze_gap_structure(text, units)
    assert "before_attribute_line" in first.reasons and first.penalty == 6
    assert "after_heading" not in second.reasons


def test_ordinary_field_words_and_midline_labels_do_not_trigger_line_rules():
    """交通等词的完整句及句中标签不能冒充独立的结构行。"""
    text, units = _units(["这里有海。", "交通方便，时间也充足。", "📍下一站：推荐！", "景色很美。"])
    gaps = analyze_gap_structure(text, units)
    assert all("before_attribute_line" not in gap.reasons for gap in gaps)
    assert all("after_heading" not in gap.reasons for gap in gaps)


def test_indentation_and_unicode_line_separators_keep_heading_evidence():
    """行首缩进、Unicode 行分隔符不会丢掉本来明确的题头。"""
    text, units = _units(["前一节完整说明。", "Day2", "第二天的完整路线介绍。"], "\u2028  ")
    assert "after_heading" in analyze_gap_structure(text, units)[1].reasons


def test_no_new_boundaries_text_changes_or_nondeterministic_evidence():
    """输出数量严格 N−1，坐标只来自输入，重复调用和输入对象保持不变。"""
    text, units = _units(["出发。", "📍海滨：推荐！", "这里风景很美。", "时间｜半天"], "\n")
    before = text, units
    result = analyze_gap_structure(text, units)
    assert len(result) == len(units) - 1
    assert [gap.index for gap in result] == list(range(len(units) - 1))
    assert result == analyze_gap_structure(text, units)
    assert (text, units) == before
    assert all(0 <= start <= end <= len(text) for gap in result for _, start, end in gap.evidence_spans)
    assert all("post_id" not in asdict(gap) for gap in result)


def test_empty_and_single_unit_require_no_boundary():
    """无 gap 时不伪造结构分数，但仍要求正文完整覆盖。"""
    assert analyze_gap_structure(" \n", ()) == ()
    text, units = _units(["海边很美。"])
    assert analyze_gap_structure(text, units) == ()


@pytest.mark.parametrize("text,units,error", [
    ("甲", (), "incomplete_structure_unit_coverage"),
    ("甲乙", (TextUnit(0, 1, 1),), "incomplete_structure_unit_coverage"),
    ("甲乙", (TextUnit(0, 2, 2), TextUnit(1, 2, 1)), "invalid_structure_unit_span"),
    ("a\u0301", (TextUnit(0, 1, 1), TextUnit(1, 2, 1)), "invalid_structure_unit_span"),
    (" ", (TextUnit(0, 1, 1),), "empty_structure_unit"),
])
def test_invalid_units_fail_with_reason_only(text, units, error):
    """坏坐标、漏字和字素拆坏不可借开发候选蒙混通过。"""
    with pytest.raises(ValueError, match=f"^{error}$"):
        analyze_gap_structure(text, units)


def test_repeated_checkmark_entries_keep_full_descriptions_and_stop_before_hashtags():
    """同类标记与说明共同证明条目，第二句及票务行仍归条目，尾标签不吞入。"""
    lines = ["前言是一段完整说明。", "✅山顶索道", "上站有宽阔的平台。下站可以乘公交。",
             "门票：现场购买", "✅山下步道", "这是一条沿溪的小路。全程设有休息点。",
             "#山中旅行# #周末出发#", "这段已经属于独立的附记。"]
    text, units = _units(lines, "\n")
    blocks = discover_structure_blocks(text, units)
    assert len(blocks) == 2
    assert text[blocks[0].start:blocks[0].end] == "\n".join(lines[1:4])
    assert text[blocks[1].start:blocks[1].end] == "\n".join(lines[4:6])
    for block, first_sentence in zip(blocks, ["上站有宽阔的平台。", "这是一条沿溪的小路。"], strict=True):
        evidence = {kind: (start, end) for kind, start, end in block.evidence}
        assert block.kind == "repeated_entry"
        assert text[slice(*evidence["first_substantive_sentence"])] == first_sentence
        assert text[slice(*evidence["repeated_marker"])] == "✅"
        assert evidence["matching_marker"] != evidence["repeated_marker"]


@pytest.mark.parametrize("headings", [
    ("🧪山间展馆", "🧪河畔展馆"), ("🗺山间展馆", "🗺️河畔展馆"),
    ("1. 山间展馆", "2. 河畔展馆"), ("一、山间展馆", "二、河畔展馆"),
    ("①山间展馆", "②河畔展馆"), ("1️⃣山间展馆", "2️⃣河畔展馆"),
    ("山间展馆：", "河畔展馆："),
])
def test_repeated_title_recognition_uses_layout_not_marker_or_place_allowlist(headings):
    """任意图形标记、数字变化及短冒号题头均可提供重复版式证据。"""
    text, units = _units([headings[0], "第一处有完整说明。", headings[1], "第二处也有完整说明。"], "\n")
    blocks = discover_structure_blocks(text, units)
    assert len(blocks) == 2
    assert text[blocks[0].start:blocks[0].end].startswith(headings[0])
    assert text[blocks[1].start:blocks[1].end].startswith(headings[1])


@pytest.mark.parametrize("headings", [
    ("✅海边真的很漂亮。", "✅公交确实很方便。"),
    ("✅海边非常漂亮", "✅今天心情不错"),
    ("✅我觉得不错", "✅喜欢这里"),
    ("✅海景很美！", "✅路途太远！"),
    ("✅价格：每晚一百元", "✅地址：山海路八号"),
    ("✅" + "山" * 33, "✅" + "河" * 33),
])
def test_repeated_checkmarks_do_not_turn_ordinary_evaluations_into_entry_titles(headings):
    """普通完整评价、程度句、属性字段和超长行不因重复勾号得到块保护。"""
    text, units = _units([headings[0], "后面仍有另一句感受。", headings[1], "这里也有另一句叙述。"], "\n")
    assert discover_structure_blocks(text, units) == ()


@pytest.mark.parametrize("parts", [
    ["✅山间展馆", "这里只出现一次这种题头。"],
    ["✅山间展馆", "🧪河畔展馆", "只有后者带说明。"],
    ["✅山间展馆", "✅河畔展馆", "✅林中展馆"],
    ["山间展馆", "这里有完整说明。", "河畔展馆", "这里也有完整说明。"],
    ["✅山间展馆", "🌊", "✅河畔展馆", "✨"],
])
def test_repetition_requires_two_matching_titles_with_substantive_descriptions(parts):
    """单次题头、纯列表、无标记短行及只有表情的正文均不足以发现条目。"""
    text, units = _units(parts, "\n")
    assert discover_structure_blocks(text, units) == ()


@pytest.mark.parametrize("trailer", ["#山间漫步 #周末", "https://example.test/travel", "@旅途记录", "附记："])
def test_final_entry_stops_at_metadata_or_unrelated_explicit_heading(trailer):
    """最后条目不能借没有下一同类标记而吞掉元数据或新章节。"""
    parts = ["✅山间展馆", "第一处介绍。", "✅河畔展馆", "第二处介绍。", trailer, "另一段独立说明。"]
    text, units = _units(parts, "\n")
    blocks = discover_structure_blocks(text, units)
    assert len(blocks) == 2
    assert text[blocks[-1].start:blocks[-1].end] == "✅河畔展馆\n第二处介绍。"


def test_block_recognition_uses_full_lines_and_first_sentence_ignores_token_splits():
    """细分单元不改变原文行或首句，且括号内句号不会制造短回退跨度。"""
    parts = ["✅山", "间展馆", "\n", "门内有平台（需要预约。", "周末有值班人员），", "出口设有指示牌。", "这里另有展厅。",
             "\n", "✅河畔展馆", "\n", "沿河可以步行。"]
    text = "".join(parts)
    units, cursor = [], 0
    for part in parts:
        if part != "\n":
            units.append(TextUnit(cursor, cursor + len(part), len(part)))
        cursor += len(part)
    blocks = discover_structure_blocks(text, tuple(units))
    assert len(blocks) == 2
    lead = next((start, end) for kind, start, end in blocks[0].evidence if kind == "first_substantive_sentence")
    assert text[slice(*lead)] == "门内有平台（需要预约。周末有值班人员），出口设有指示牌。"
    assert text[blocks[0].start:blocks[0].end].endswith("这里另有展厅。")


def test_structure_blocks_preserve_unicode_offsets_and_are_immutable():
    """缩进和 Unicode 换行保留坐标，纯发现过程无 token 调用或输入改写。"""
    text, units = _units(["✅山间展馆", "第一处介绍。", "✅河畔展馆", "第二处介绍。"], "\u2028  ")
    blocks = discover_structure_blocks(text, units)
    assert len(blocks) == 2 and blocks == discover_structure_blocks(text, units)
    assert all(0 <= block.start < block.end <= len(text) for block in blocks)
    assert all(0 <= start < end <= len(text) for block in blocks for _, start, end in block.evidence)
    with pytest.raises(FrozenInstanceError):
        blocks[0].end = len(text)
    assert discover_structure_blocks(" \r\n", ()) == ()
    with pytest.raises(ValueError, match="^incomplete_structure_unit_coverage$"):
        discover_structure_blocks("甲", ())


@pytest.mark.parametrize("detail", ["🎫可以在网上预约", "🚦需要提前取号", "🛎️应该保留凭证", "🧾建议带好证件"])
def test_modal_instruction_does_not_end_or_create_a_repeated_entry(detail):
    """重复情态说明也不是题头，短说明之后的地点属性必须留在同条目。"""
    parts = ["✅山顶车站", "上层有宽阔平台。", detail, "出口设有休息区。",
             "✅山下车站", "下层可以换乘公交。", detail, "站外有停车场。"]
    text, units = _units(parts, "\n")
    blocks = discover_structure_blocks(text, units)
    assert len(blocks) == 2
    assert text[blocks[0].start:blocks[0].end] == "\n".join(parts[:4])
    assert text[blocks[1].start:blocks[1].end] == "\n".join(parts[4:])


def test_single_emoji_line_without_repeated_layout_cannot_terminate_an_entry():
    """未出现通用谓词的短图标行也不能凭单次标记吞掉条目的后续说明。"""
    parts = ["✅山顶车站", "站台设有观景窗。", "🧾纸质凭证", "凭证在入口处检查。",
             "✅山下车站", "站外设有公交站。"]
    text, units = _units(parts, "\n")
    blocks = discover_structure_blocks(text, units)
    assert len(blocks) == 2
    assert text[blocks[0].start:blocks[0].end] == "\n".join(parts[:4])


def test_parenthetical_opinion_does_not_erase_the_repeated_heading_subject():
    """附注可以含观点或句号，题头主体仍能标出新条目而不被前条吞入。"""
    parts = ["✅山顶车站", "上层有宽阔平台。", "✅河畔步道（个人觉得很安静。）", "入口在桥的另一侧。"]
    text, units = _units(parts, "\n")
    blocks = discover_structure_blocks(text, units)
    assert len(blocks) == 2
    assert text[blocks[0].start:blocks[0].end] == "\n".join(parts[:2])
    assert text[blocks[1].start:blocks[1].end] == "\n".join(parts[2:])
    # 相反，普通评价主体不能借括号内出现名词而升级成题头。
    text, units = _units(["✅这里很美（河畔步道）", "这是后续说明。", "✅这里太远（山顶车站）", "这也是后续说明。"], "\n")
    assert discover_structure_blocks(text, units) == ()


def test_unmarked_heading_chain_attaches_only_through_first_substantive_sentence():
    """短题头链靠后继长说明得到证据，不将整段或后续独立内容自动保护。"""
    parts = ["产品观察", "材料与做工", "这件物品的外壳采用金属材质，接缝完整，边角也经过打磨处理。", "下一句是另一个完整观察。"]
    text, units = _units(parts, "\n")
    blocks = discover_context_blocks(text, units)
    assert len(blocks) == 1 and blocks[0].kind == "heading_context"
    assert text[blocks[0].start:blocks[0].end] == "\n".join(parts[:3])
    assert sum(kind == "heading" for kind, _, _ in blocks[0].evidence) == 2
    assert discover_structure_blocks(text, units) == ()


@pytest.mark.parametrize("heading", ["《山河之间》", "《我曾走过的路》", "6月3日，星期三，晴", "Day3", "✅山间站台"])
def test_explicit_work_date_and_marked_titles_can_bind_a_long_first_description(heading):
    """作品名、既有日期题头及标记题头不需要内容白名单或重复出现。"""
    body = "这段内容记录了沿途见到的建筑及街道，后面还介绍了观察的具体过程。"
    text, units = _units([heading, body], "\n")
    blocks = discover_context_blocks(text, units)
    assert len(blocks) == 1 and blocks[0].kind == "heading_context"
    assert text[blocks[0].start:blocks[0].end] == text


@pytest.mark.parametrize("heading", ["这里很美", "我觉得值得再来", "可以提前预约", "风景很好。", "沿河散步，随后返回", "时间：一小时"])
def test_ordinary_clause_is_not_a_context_heading_even_with_a_long_following_line(heading):
    """长后文不能把程度句、代词句、情态说明和属性字段倒推成题头。"""
    text, units = _units([heading, "这一段继续记录当天的实际经历，描述道路、建筑以及最后返程的完整过程。"], "\n")
    assert discover_context_blocks(text, units) == ()


def test_two_equally_short_narrative_lines_do_not_create_whole_post_protection():
    """即使都达到24字符，两个同长无标记短叙述也不能互相冒充题头和长说明。"""
    first = "沿着山间道路向前行走经过树林石桥草地后到达河畔平台"
    second = "从河畔平台转身返回穿过草地石桥树林后沿道路走到入口"
    assert len(first) == len(second) and 24 <= len(first) <= 32
    text, units = _units([first, second], "\n")
    assert discover_context_blocks(text, units) == ()
    text, units = _units(["岸边漫步", "雨后返回"], "\n")
    assert discover_context_blocks(text, units) == ()


def test_heading_length_and_following_description_lower_bound_are_explicit():
    """32字符题头上限和24字符说明下限分别生效，不因只有两行而强制粘合。"""
    text, units = _units(["山" * 33, "道路介绍。" * 10], "\n")
    assert discover_context_blocks(text, units) == ()
    text, units = _units(["景观概览", "沿路设有平台和休息区域。"], "\n")
    assert discover_context_blocks(text, units) == ()


def test_heading_context_never_uses_an_artificial_unit_split_as_first_sentence_end():
    """模型细分只提供已有坐标，首实质句必须保持完整原文句子。"""
    parts = ["建筑观察", "\n", "外墙采用浅色石材，", "窗边设有木制遮阳板，", "门厅保留通向庭院的完整通道。", "下一句另作说明。"]
    text, cursor, units = "".join(parts), 0, []
    for part in parts:
        if part != "\n":
            units.append(TextUnit(cursor, cursor + len(part), len(part)))
        cursor += len(part)
    block, = discover_context_blocks(text, tuple(units))
    assert text[block.start:block.end] == "".join(parts[:-1])


def test_trailing_mixed_hashtags_and_handles_bind_the_last_real_sentence():
    """标签和账号混排仍是元数据，前句起点跨token单元完整保留到文末。"""
    parts = ["前一句已经结束。", "最后一段包括建筑说明，", "以及出口方向和步行距离。", "\n",
             "#沿途记录# @行旅人", "\n", "@观察者 #建筑漫步#", "\n  "]
    text, cursor, units = "".join(parts), 0, []
    for part in parts:
        if part.strip():
            units.append(TextUnit(cursor, cursor + len(part), len(part)))
        cursor += len(part)
    block, = discover_context_blocks(text, tuple(units))
    assert block.kind == "trailing_metadata_context"
    assert block.start == len(parts[0]) and block.end == len(text)
    assert text[block.start:block.end] == "".join(parts[1:])
    span = next((a, b) for kind, a, b in block.evidence if kind == "preceding_substantive_sentence")
    assert text[slice(*span)] == "".join(parts[1:3])


def test_metadata_context_uses_last_sentence_after_a_bracketed_quote():
    """句中括号内句号不成为元数据块起点，尾部纯表情仍保留。"""
    text, units = _units(["第一句结束。最后的介绍（包括预约说明。和出口位置），需要完整保留。🌊", "#路线记录# @作者"], "\n")
    block, = discover_context_blocks(text, units)
    assert text[block.start:block.end].startswith("最后的介绍（包括预约说明。和出口位置），需要完整保留。")
    assert text[block.start:block.end].endswith("🌊\n#路线记录# @作者")


@pytest.mark.parametrize("parts", [
    ["正文已经写完。", "#标签# @账号", "后面还有正文。"],
    ["正文已经写完。", "今天用#标签#记录这段经历"],
    ["正文已经写完。", "感谢 @账号 提供路线"],
    ["#标签# @账号", "@另一个账号 #另一个标签#"],
])
def test_metadata_must_be_an_exclusive_tail_and_have_substantive_prior_context(parts):
    """正文提到标签、非尾部元数据或整帖只有元数据都不触发尾部绑定。"""
    text, units = _units(parts, "\n")
    assert discover_context_blocks(text, units) == ()


def test_context_discovery_is_separate_deterministic_and_validates_empty_input():
    """新候选不改变已有分析默认，空白和坏输入沿用明确校验语义。"""
    text, units = _units(["材料观察", "这段说明记录材料表面和边角的处理，并给出了相应的观察过程和判断依据。", "#观察记录#"], "\n")
    before = analyze_gap_structure(text, units), discover_structure_blocks(text, units)
    blocks = discover_context_blocks(text, units)
    assert {block.kind for block in blocks} == {"heading_context", "trailing_metadata_context"}
    assert blocks == discover_context_blocks(text, units)
    assert before == (analyze_gap_structure(text, units), discover_structure_blocks(text, units))
    assert discover_context_blocks(" \n", ()) == ()
    with pytest.raises(ValueError, match="^incomplete_structure_unit_coverage$"):
        discover_context_blocks("甲", ())


@pytest.mark.parametrize('heading', [
    '读《山间的故事》(文学剧本)林某著(女) 人物纪事。',
    '阅读《沿河行走》[散文]李某编著',
    '观看《春山》[电影]，(赵甲、钱乙 著 出处同前)。',
])
def test_work_bibliography_heading_chain_stays_with_first_explanation(heading):
    """作品、作者与体裁来自显式符号结构，日期链不在书目结束时误作已自足。"""
    description = '故事中的主人公在小城经营一家书店，同时承担了修缮老街建筑的实际工作。'
    text, units = _units(['8月2日，星期一，晴', heading, description], '\n')
    blocks = discover_context_blocks(text, units)
    assert len(blocks) == 1 and blocks[0].kind == 'heading_context'
    assert text[blocks[0].start:blocks[0].end] == text
    assert sum(kind == 'heading' for kind, _, _ in blocks[0].evidence) >= 2


def test_inline_work_bibliography_binds_its_following_description_without_rewriting():
    """同一原文行可含书目与说明，已有单元边界不能将两者误分离。"""
    heading = '读《山城旧事》[剧本]，(孙甲、周乙、吴丙、郑丁 著 出处同前)。'
    description = '老人坐在窗边回忆初到这座城市的经历，往事随着窗外的雨声逐渐展开。'
    text, units = _units([heading, description], ' ')
    assert len(heading) > 32  # 题名仍短，不能用整条作者元数据长度否定其结构。
    blocks = discover_context_blocks(text, units)
    assert len(blocks) == 1 and blocks[0].kind == 'heading_context'
    assert (blocks[0].start, blocks[0].end) == (0, len(text))
    assert text == heading + ' ' + description
    assert analyze_gap_structure(text, units)[0].penalty == 0


@pytest.mark.parametrize('opening', [
    '我读《沿河行走》之后写下了一些感想。',
    '读《沿河行走》觉得很有意思。',
    '读《沿河行走》让我想起从前的经历。',
    '为《沿河行走》寻找资料。',
])
def test_ordinary_book_mention_does_not_become_a_bibliography_heading(opening):
    """普通阅读叙述及书名引用没有书目结构，不因含《》就升级为标题。"""
    text, units = _units([opening, '后来我们继续沿着河岸行走，经过桥梁、码头和多处可以停下来休息的地方。'], '\n')
    assert discover_context_blocks(text, units) == ()


@pytest.mark.parametrize('separate_label', [False, True])
def test_very_short_label_is_not_the_complete_first_explanation(separate_label):
    """极短题材标签后须继续带入完整实质句，既支持同行也支持另起一行。"""
    heading, label = '读《工坊故事》[剧本]林某著', '职业题材。'
    body = '主人公来到一处旧工坊研究传统工具，随后与当地师傅一起完成了修复工作。'
    parts = [heading, label, body] if separate_label else [heading, label + body]
    text, units = _units(parts, '\n')
    block, = discover_context_blocks(text, units)
    assert text[block.start:block.end] == text
    start, end = next((a, b) for kind, a, b in block.evidence if kind == 'first_substantive_sentence')
    assert text[start:end].startswith(label) and text[start:end].endswith(body)


def test_short_lead_does_not_cross_another_heading_to_accumulate_length():
    """小节到此结束时，不为达到最少字数跨越下一显式作品题头。"""
    first, second = '读《山中的一天》', '读《海边的一天》'
    body = '第二部作品描写了海边的一处小镇，主人公每天穿过街道来到港口从事工作。'
    text, units = _units([first, '简短记录。', second, body], '\n')
    blocks = discover_context_blocks(text, units)
    assert len(blocks) == 1
    assert text[blocks[0].start:blocks[0].end] == second + '\n' + body


@pytest.mark.parametrize('punctuation', ['！', '。', '!?', '；'])
def test_adverbial_particle_across_anomalous_punctuation_protects_local_continuation(punctuation):
    """局部状语与同一行后续分句保持关联，旧structure罚分不被新候选改写。"""
    parts = ['大家停下来，认真地' + punctuation, '检查桥面的每一处接缝；', '随后再继续向前走。']
    text, units = _units(parts)
    previous = analyze_gap_structure(text, units)
    blocks = [block for block in discover_context_blocks(text, units) if block.kind == 'syntactic_continuation']
    assert len(blocks) == 1
    assert text[blocks[0].start:blocks[0].end] == '认真地' + punctuation + parts[1]
    assert previous == analyze_gap_structure(text, units)
    assert previous[0].penalty == 0
    particle = next((a, b) for kind, a, b in blocks[0].evidence if kind == 'adverbial_particle')
    assert text[slice(*particle)] == '地'


@pytest.mark.parametrize('left,right,separator', [
    ('旅游目的地！', '这里的交通很便利。', ''),
    ('露营地！', '这里可以搭帐篷。', ''),
    ('山间露营地！', '附近有水源。', ''),
    ('真的！', '这里的景色很漂亮。', ''),
    ('当地！', '这里是我们的下一站。', ''),
    ('认真地！', '后面开始另一段正文。', '\n'),
    ('“认真地！”', '他重复了这个词语。', ''),
])
def test_complete_exclamation_location_noun_and_line_break_do_not_get_adverb_protection(left, right, separator):
    """地理名词、的字感叹、跨行及完整引语不按末字统一判为状语断裂。"""
    text, units = _units([left, right], separator)
    assert not any(block.kind == 'syntactic_continuation' for block in discover_context_blocks(text, units))


def test_heading_only_document_does_not_repeatedly_read_all_heading_prefixes():
    """无说明的短标题表不产生块；读取工作随输入线性增长，不依赖墙钟阈值。"""
    class ObservedText(str):
        """统计向原文索取的切片字符量，捕获重复扫描而不窥探算法内部状态。"""
        def __new__(cls, value):
            instance = super().__new__(cls, value)
            instance.sliced_codepoints = 0
            return instance

        def __getitem__(self, key):
            value = super().__getitem__(key)
            if isinstance(key, slice):
                self.sliced_codepoints += len(value)
            return value

    measurements = []
    for count in (100, 200):
        text, units = _units(["山间风景"] * count, "\n")
        observed = ObservedText(text)
        assert discover_context_blocks(observed, units) == ()
        measurements.append(observed.sliced_codepoints)
    assert measurements[0] > 0
    assert measurements[1] <= 3 * measurements[0]
