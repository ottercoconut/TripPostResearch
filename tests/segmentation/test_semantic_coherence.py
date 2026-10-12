"""完整介绍／推荐父块的合成正反例，不写入真实UGC或以来源ID决定结果。"""

import pytest

from tourism_ugc_study.segmentation.contracts import TextUnit
from tourism_ugc_study.segmentation.semantic_coherence import _shared_span, discover_coherent_blocks


def discover(parts, separator="\n"):
    """按原文行构造既有单元；空行只保留在原文，输出坐标可独立核对。"""
    text, cursor, units = separator.join(parts), 0, []
    for part in parts:
        if part.strip():
            units.append(TextUnit(cursor, cursor + len(part), len(part)))
        cursor += len(part) + len(separator)
    blocks = discover_coherent_blocks(text, units)
    assert all(0 <= block.start < block.end <= len(text) for block in blocks)
    assert all(0 <= a < b <= len(text) for block in blocks for _, a, b in block.evidence)
    return text, blocks


def test_numbered_parts_keep_same_topic_introduction_and_stop_at_new_section():
    """组成部分受共同总述约束；新栏目仍位于父范围外，不能按编号机械拆散。"""
    parts = ["云河文化园参观记录？", "云河文化园沿河展开，园内保留了多年以前的旧建筑和树木。",
             "✔云河文化园主要分为3个部分：", "1、入口广场：广场两侧有长椅，可以从这里沿河走进园区。",
             "2、旧厂房：保留了当年的机器。里面还有记录厂区历史的老照片。",
             "3、花园：这里种植了不同季节开花的植物，沿小路可以返回入口。", "", "【实际体验】：",
             "当天抵达以后，我们发现部分道路正在整修，走回入口用了很长时间。"]
    text, blocks = discover(parts)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.kind == "coherent_enumeration"
    assert text[block.start:block.end] == "\n".join(parts[:6])
    assert len([span for span in block.evidence if span[0] == "coherent_item"]) == 3


@pytest.mark.parametrize("parts", [
    ["1、交通：镇里有公共小巴。", "2、婚礼：下个月将在家中举行。"],
    ["✅云边展台", "入口旁可以休息。", "✅电池维修", "请在断电以后检查。"],
    ["我推荐把自己的想法写下来。", "第一件事情已经结束。", "另一件事情还在等待。"],
])
def test_numbering_repeated_symbols_or_recommendation_words_alone_are_insufficient(parts):
    """没有父栏目关系时，并排无关列表和普通推荐文字不被整块保护。"""
    assert discover(parts)[1] == ()


def test_same_topic_evidence_is_required_to_extend_into_preceding_paragraph():
    """完全无关的相邻正文不成为列举总述；不是按相邻或短段门槛向前吞并。"""
    parts = ["下个月的会议改为在线举行，具体时间稍后通知。", "云河文化园主要分为2个部分：",
             "1、展厅：可以看到当年的照片。", "2、庭院：这里保留了几棵老树。"]
    text, blocks = discover(parts)
    assert blocks[0].start == text.index(parts[1])
    assert not any(e[0] == "preceding_introduction" for e in blocks[0].evidence)


def test_broken_number_order_does_not_create_a_complete_enumeration():
    """缺项不能靠出现两次数字补足父关系，避免把另一列表并入。"""
    assert discover(["云河文化园分为3个部分：", "1、展厅：可以看到照片。", "3、庭院：可以休息。"])[1] == ()


def test_numbered_title_and_following_paragraph_remain_one_constituent_item():
    """编号题头末尾冒号不冒充新父栏目；子项范围包括下一行完整说明。"""
    parts = ["云河文化园分为2个部分：", "1、旧物展厅：", "墙面展示当年留下的照片，还能看到旧式的生产工具。",
             "2、临河花园：", "沿着小路慢慢走，可以看到保留至今的老树和旧桥。"]
    text, blocks = discover(parts)
    assert text[blocks[0].start:blocks[0].end] == text
    item_spans = [span[1:] for span in blocks[0].evidence if span[0] == "coherent_item"]
    assert [text[a:b] for a, b in item_spans] == ["\n".join(parts[1:3]), "\n".join(parts[3:])]


def test_explicit_named_parts_can_be_complete_paragraphs_with_blank_lines():
    """先列出名称与数量，再逐段解释的构成关系不要求每段另有编号。"""
    parts = ["云河文化园位于河岸，保存了多座历史建筑和成片的老树。",
             "云河文化园共分为旧物历史、传统工艺、互动体验三个展区：",
             "在旧物历史区域，游客可以了解过去的生活方式和建筑变迁。", "",
             "传统工艺区域展示的，是当年留下的手工制作工具和工序。",
             "在互动体验区域，游客能够亲手操作模型，观察每个部件如何协作。",
             "另一个城市正在重新安排下个月举行的音乐节。"]
    text, blocks = discover(parts)
    assert text[blocks[0].start:blocks[0].end] == "\n".join(parts[:6])
    assert len([e for e in blocks[0].evidence if e[0] == "coherent_item"]) == 3


def test_named_part_count_does_not_swallow_unrelated_paragraph_without_reference():
    """即使段数相符，没有逐项名称回指的段落也不能冒充组成部分。"""
    parts = ["云河文化园分为历史典藏、互动体验两个展区：",
             "历史典藏区域展示以前的日常用品。", "明天的会议时间改为上午九点。"]
    assert discover(parts)[1] == ()


def test_directory_keeps_bare_titles_descriptions_and_referred_continuation():
    """没有编号的连续推荐依靠目录题头和题头说明配对，末尾全篇建议不被吞入。"""
    parts = ["这次沿河走过的几处地方↓", "云边茶屋🍵", "茶屋门前有一片宽敞的草地，靠近院门的位置还放着几张可以休息的长椅。",
             "河畔书屋📚", "书屋里可以看到许多旧书，走过书架以后就是能够安静坐下阅读的窗边座位。",
             "石桥展馆🏛️", "沿着河边的小路一直向前，可以看到保留着旧式门窗的小院和院中的老树。",
             "石桥展馆内部展示了当年的照片，我们在看完介绍后又沿着走廊缓慢走了一圈。",
             "周末外出以前应当查看天气，准备好饮水，并把返程时间告诉同行的人。"]
    text, blocks = discover(parts)
    assert len(blocks) == 1 and blocks[0].kind == "coherent_directory"
    assert text[blocks[0].start:blocks[0].end] == "\n".join(parts[:8])
    assert len([e for e in blocks[0].evidence if e[0] == "coherent_item"]) == 3


def test_pure_title_directory_without_substantive_descriptions_is_not_a_parent():
    """仅有一串标题不足以证明完整推荐，检测器不虚构正文或新单元。"""
    assert discover(["接下来看看这些地方↓", "云边茶屋", "河畔书屋", "石桥展馆"])[1] == ()


def test_multiday_route_overview_and_wrapped_routes_share_one_parent():
    """目录中的日序路线是并列概览，折行不变成独立题头，下一目录明确终止。"""
    parts = ["接下来三天的路线↓", "D1车站→河边→旧街", "D2展台→木桥→山间", "书屋",
             "D3小镇→公园→返程", "沿途吃过的几餐↓", "云边茶屋🍵",
             "店前可以看到一片开阔的草地，靠近院门的地方还摆着供客人休息的长椅。"]
    text, blocks = discover(parts)
    assert len(blocks) == 1 and blocks[0].kind == "coherent_directory"
    assert text[blocks[0].start:blocks[0].end] == "\n".join(parts[:5])
    assert len([span for span in blocks[0].evidence if span[0] == "parent_route"]) == 3


def test_multiday_route_overview_rejects_all_optional_semantic_cuts():
    """整份目录的保护不能依赖某组旧分数恰巧不提名日序之间的gap。"""
    from tourism_ugc_study.segmentation.contracts import GapScore, SemanticDecision
    from tourism_ugc_study.segmentation.semantic_config import CoherentSemanticConfig
    from tourism_ugc_study.segmentation.semantic_meaning import partition_by_meaning

    class Counter:
        """合成用逐码点计数；只验证保护优先级，不声明真实token长度。"""

        def count(self, value):
            """给出每个原文跨度的一致精确合成计数。"""
            return len(value)

    parts = ["接下来三天的路线↓", "D1车站→河边→旧街", "D2展台→木桥→山间", "书屋", "D3小镇→公园→返程"]
    text, units, cursor = "\n".join(parts), [], 0
    for part in parts:
        units.append(TextUnit(cursor, cursor + len(part), len(part)))
        cursor += len(part) + 1
    selected = tuple(range(len(units) - 1))
    gaps = tuple(GapScore(i, 0.1, 0.5, True, True, "synthetic_selected") for i in selected)
    decision = SemanticDecision("scored_split", (), gaps, selected, ())
    config = CoherentSemanticConfig(min_segment_nonspace_chars=1, max_segment_tokens=512)
    result = partition_by_meaning(text, units, decision, config, Counter())
    assert len(result.segments) == 1 and result.segments[0].raw_text == text
    assert result.semantic_boundaries == result.structural_boundaries == ()


def test_day_route_and_subitems_form_one_scope_and_end_before_next_day():
    """同日路线、景点介绍与内部评价连贯保留，下一天保持独立范围。"""
    parts = ["📅Day1 沿河路线", "云边展台→老街书屋→河畔集市", "✅云边展台", "展台可以看到河道的转弯。",
             "不过台阶比较窄，上下时需要慢一点。", "✅老街书屋", "书屋的窗边摆着可以坐下阅读的椅子。",
             "🌙晚上：河畔集市", "摊位沿河一字展开，可以边走边看。", "", "📅Day2 山间路线", "我们沿山路走进了树林。"]
    text, blocks = discover(parts)
    assert len(blocks) == 1 and blocks[0].kind == "coherent_schedule"
    assert text[blocks[0].start:blocks[0].end] == "\n".join(parts[:9])


def test_day_without_route_or_repeated_subheadings_does_not_absorb_independent_narration():
    """同一天这个时间关系本身不把完整叙述和独立主题全封进保护块。"""
    parts = ["📅Day1", "我们在河边坐了一会，随后回到住处。", "公司下午通知了新的会议安排。"]
    assert discover(parts)[1] == ()


def test_distinct_heading_styles_without_route_do_not_prove_same_day_recommendation():
    """同日的不同栏目样式不能仅凭两个装饰符推断成连续推荐。"""
    parts = ["📅Day1", "✅河边展台", "展台外可以看到宽阔的河面。", "🧰设备维修", "应在关闭电源后检查设备。"]
    assert discover(parts)[1] == ()


def test_independent_section_inside_day_ends_route_scope():
    """明确新栏目在同日内部也可结束推荐范围，功能词不参与判断。"""
    parts = ["📅Day1 河边", "展台→书屋", "✅展台", "站在这里可以看到宽阔的河面。",
             "【另一项记录】：", "我们下午收到了远方朋友发来的消息。"]
    text, blocks = discover(parts)
    assert blocks[0].end == text.index(parts[4]) - 1


def test_blank_paragraph_outside_item_style_ends_schedule_scope():
    """空行后的普通独立段不因此前出现Day题头和路线而被永久保护。"""
    parts = ["📅Day1 河边", "展台→书屋", "✅展台", "站在这里可以看到宽阔的河面。", "",
             "这周工作的安排已经改变，需要重新确定会议的时间。"]
    text, blocks = discover(parts)
    assert blocks[0].end == text.index(parts[3]) + len(parts[3])


@pytest.mark.parametrize("transition", ["第二天：我们坐车来到另一座城市，在山脚住了下来。",
                                       "📅Day2：我们乘车离开河岸，来到山间的住处。",
                                       "10月2日：我们来到另一座城市，在山脚住了下来。",
                                       "2026年10月2日，我们乘车离开河岸，来到山间的住处。"])
def test_full_sentence_next_day_ends_previous_schedule_parent(transition):
    """换天开头即使带完整叙述，也不能因不符合短题头规则而被前日吞入。"""
    parts = ["📅Day1 河岸漫游", "展馆→书屋", "✅展馆", "展厅介绍了河岸的历史。", transition]
    text, blocks = discover(parts)
    assert len(blocks) == 1
    assert blocks[0].end == text.index(parts[3]) + len(parts[3])


@pytest.mark.parametrize("transition", ["第二天：我们坐车来到另一座城市，在山脚住了下来。",
                                       "📅Day2：我们乘车离开河岸，来到山间的住处。",
                                       "10月2日：我们来到另一座城市，在山脚住了下来。",
                                       "2026年10月2日，我们乘车离开河岸，来到山间的住处。"])
def test_full_sentence_next_day_ends_numbered_enumeration_parent(transition):
    """列举扫描也使用换天终止，不能把之后的完整日程当作最后一项的续段。"""
    parts = ["园中布局分为2个部分：", "1、展馆：介绍河岸的历史。", "2、书屋：可以看到许多旧书。", transition]
    text, blocks = discover(parts)
    assert len(blocks) == 1
    assert blocks[0].end == text.index(parts[2]) + len(parts[2])


@pytest.mark.parametrize("continuation", ["展厅墙面写着第二天的开放安排，可以在离开时再确认。",
                                         "第二天候选名单列在展厅的墙面上，可以在离开时再确认。",
                                         "展厅墙面写着10月2日的开放安排，可以在离开时再确认。",
                                         "10月2日候选名单列在展厅的墙面上，可以在离开时再确认。"])
def test_day_mention_or_undelimited_noun_phrase_does_not_end_parent(continuation):
    """行内日序及无日程分隔版式的名词短语不冒充显式换天。"""
    parts = ["📅Day1 河岸漫游", "展馆→书屋", "✅展馆", "展厅介绍了河岸的历史。", continuation]
    text, blocks = discover(parts)
    assert len(blocks) == 1 and blocks[0].end == len(text)


@pytest.mark.parametrize("separator", ["\n", "\r\n", "\u2028"])
def test_unicode_numbered_items_and_metadata_keep_original_spans(separator):
    """Unicode编号、组合字素和换行均保留原文坐标，元数据交由既有尾附着处理。"""
    parts = ["园中布局分为2个部分：", "1️⃣展厅：墙面写着e\u0301的故事。", "2️⃣花园：树下可以安静休息。", "#游记# @旅客"]
    text, blocks = discover(parts, separator)
    assert len(blocks) == 1
    assert text[blocks[0].start:blocks[0].end] == separator.join(parts[:3])


def test_detector_never_moves_parent_start_to_an_existing_unit_gap():
    """父起点落在模型单元内部时仍返回真实坐标，交由消费方报告不可用起点。"""
    text = "别的事情已经结束。\n园中布局分为2个部分：\n1、展厅：可以阅读。\n2、庭院：可以休息。"
    units = (TextUnit(0, len(text), len(text)),)
    block, = discover_coherent_blocks(text, units)
    assert block.start == text.index("园中布局")


def test_empty_input_and_invalid_coverage_keep_explicit_failure_semantics():
    """空白正文合法；缺字及切坏字素不能作为结构检测的隐式修复输入。"""
    assert discover_coherent_blocks(" \n", ()) == ()
    with pytest.raises(ValueError):
        discover_coherent_blocks("正文", ())
    with pytest.raises(ValueError, match="字素"):
        discover_coherent_blocks("e\u0301", (TextUnit(0, 1, 1), TextUnit(1, 2, 1)))


def test_shared_topic_evidence_preserves_original_offsets_and_character_order():
    """短窗口证据保留原坐标，只有连续逐字复现达标，字序相似不等于回指。"""
    left, right = "🌊云—河文化园", "书屋旁的云河·文化园"
    span = _shared_span(left, right, 4)
    assert span == (1, 6, 4, 9)
    assert left[span[0]:span[1]] == "云—河文化"
    assert right[span[2]:span[3]] == "云河·文化"
    assert _shared_span("云河文化", "文化云河", 3) is None
    assert _shared_span("云河", "云河文化", 3) is None


def test_repeated_long_text_uses_bounded_exact_evidence_without_longest_match():
    """重复长文本只返回足够证明复现的短证据，不为求最长匹配建立二次搜索。"""
    left, right = "甲" * 20000, "乙" + "甲" * 20000
    assert _shared_span(left, right, 4) == (0, 4, 1, 5)
    assert _shared_span(left, "乙" * 20000, 4) is None


@pytest.mark.parametrize("minimum", [2, 3, 4])
def test_short_window_acceptance_matches_exhaustive_contiguous_evidence(minimum):
    """检查实际使用的三个证据门，不通过共同字符集合或重排词序制造回指。"""
    from itertools import product

    candidates = ["".join(chars) for length in range(5) for chars in product("甲乙", repeat=length)]
    for left in candidates:
        for right in candidates:
            expected = any(left[i:i + minimum] == right[j:j + minimum]
                           for i in range(len(left) - minimum + 1)
                           for j in range(len(right) - minimum + 1))
            span = _shared_span(left, right, minimum)
            assert (span is not None) == expected
            if span is not None:
                assert left[span[0]:span[1]] == right[span[2]:span[3]]
                assert span[1] - span[0] == span[3] - span[2] == minimum
