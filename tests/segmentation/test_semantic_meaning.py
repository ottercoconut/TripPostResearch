"""experimental意思分区的合成反例；不含来源帖子或真实UGC测试材料。"""

from dataclasses import asdict, replace

import pytest

from tourism_ugc_study.segmentation.contracts import GapScore, SemanticDecision, TextUnit
from tourism_ugc_study.segmentation.integrity import WHITE, validate
from tourism_ugc_study.segmentation.semantic_boundaries import choose_boundaries
from tourism_ugc_study.segmentation.semantic_config import SemanticConfig
from tourism_ugc_study.segmentation.semantic_meaning import EXPERIMENT_ID, partition_by_meaning
from tourism_ugc_study.segmentation.semantic_partition import partition_with_structure


class CharacterCounter:
    """每码点计一token的合成计数器；真实实验仍必须传原tokenizer。"""

    def count(self, text):
        """不模拟真实模型分词，只为不变量构造可手算的长度。"""
        return len(text)


def fixture(parts, *, selected=(), separator="\n", cohesion=0.95):
    """构造精确span和无深谷评分，验证结构边界无需评分过阈值。"""
    text = separator.join(parts)
    units, cursor = [], 0
    for part in parts:
        units.append(TextUnit(cursor, cursor + len(part), len(part)))
        cursor += len(part) + len(separator)
    gaps = tuple(GapScore(index, cohesion, 0.0, False, index in selected,
                          "synthetic_selected" if index in selected else "not_local_minimum")
                 for index in range(len(parts) - 1))
    return text, tuple(units), SemanticDecision("scored_keep", (), gaps, selected, ())


def run(parts, *, selected=(), separator="\n", cohesion=0.95, **config):
    """默认保留c32软门槛，明确短条目仍须靠结构自足性得到边界。"""
    text, units, decision = fixture(parts, selected=selected, separator=separator, cohesion=cohesion)
    settings = SemanticConfig(unit_max_tokens=32, max_segment_tokens=512,
                              min_segment_nonspace_chars=32, **config)
    return text, partition_by_meaning(text, units, decision, settings, CharacterCounter())


def assert_integrity(text, result, maximum=512):
    """从结果独立检验覆盖、连续原文、字素、真实合成token及完整trace。"""
    validate(text, [asdict(segment) for segment in result.segments])
    assert all(segment.raw_text == text[segment.start:segment.end] for segment in result.segments)
    assert all(len(segment.raw_text) <= maximum for segment in result.segments)
    assert "".join(c for s in result.segments for c in s.raw_text if c not in WHITE) == "".join(c for c in text if c not in WHITE)
    assert all(trace.reason_codes and trace.evidence_spans for trace in result.gap_decisions)
    assert all(0 <= start <= end <= len(text) for trace in result.gap_decisions for _, start, end in trace.evidence_spans)


def test_repeated_short_entries_split_before_heading_without_crossing_next_entry():
    """明确短说明已经完整，不能累计字符越过下一同级标题。"""
    parts = ["✅云间展台", "风吹着很舒服", "✅老城街角", "旧街通往河岸", "✅山间书屋", "窗边可以安静读书"]
    text, result = run(parts)
    assert [s.raw_text for s in result.segments] == ["\n".join(parts[i:i + 2]) for i in (0, 2, 4)]
    assert result.structural_boundaries == (1, 3)
    assert result.semantic_boundaries == ()
    assert all(block.end <= text.index(parts[2]) for block in result.protected_blocks if block.start == 0)
    assert_integrity(text, result)


def test_day_transition_does_not_bind_previous_evening_heading():
    """晚间独立标签后紧跟新日程时，不将两天合成标题链。"""
    parts = ["📅Day1 沿河漫步", "沿着河岸走过了木桥，返回时看到了点亮的路灯。", "🌙晚上：河湾灯景",
             "📅Day2 山林步道", "从山脚缓慢走到林间的平台，在长椅上休息后按原路返回。"]
    text, result = run(parts)
    assert 2 in result.structural_boundaries
    assert result.segments[-1].raw_text == "\n".join(parts[3:])
    assert not any(block.start <= text.index(parts[2]) < text.index(parts[3]) < block.end for block in result.protected_blocks)
    assert_integrity(text, result)


def test_inline_day_prefix_can_start_a_new_day_without_global_valley():
    """日程标签和当日描述同行也可提名，不要求整行是短标题。"""
    parts = ["第一天(7月2日)：我们先到河边走了一圈，下午返回了住宿的地方。",
             "第二天(7月3日)：我们来到山中的展馆，沿着走廊仔细观察了建筑。",
             "第三天(7月4日)：离开以前我们又去广场散步，最后乘车返回家中。"]
    text, result = run(parts)
    assert [s.raw_text for s in result.segments] == parts
    assert result.structural_boundaries == (0, 1)
    assert_integrity(text, result)


def test_meaning_shift_after_first_description_is_not_whole_entry_protected():
    """同一地点首说明之后的完整体验转换可分；不保护重复条目的全部正文。"""
    parts = ["✅山顶展台", "上午沿着宽阔步道来到入口，经过长廊以后终于登上视野开阔的平台。",
             "不过返程等候接驳车的时间实在太长，我们排队接近一小时才终于回到山脚。",
             "✅山下书屋", "走进书屋以后能够看到许多旧书，窗边设置了供旅客休息的座椅。"]
    text, result = run(parts, selected=(1,))
    assert result.semantic_boundaries == (1,)
    assert result.structural_boundaries == (2,)
    assert result.segments[0].raw_text == "\n".join(parts[:2])
    assert_integrity(text, result)


@pytest.mark.parametrize("separator,punctuation", [("", "，"), ("\n", "，"), ("\r\n", "、"), ("\u2028", "：")])
def test_comma_fragment_is_never_an_optional_semantic_boundary(separator, punctuation):
    """合成的‘登台，眼前…’必须完整，单元细分不等于意思转换。"""
    parts = ["经过长长的台阶登上山顶的观景亭" + punctuation, "眼前出现了绵延的山林和河流，我们站在栏杆旁看了很久。"]
    text, result = run(parts, selected=(0,), separator=separator, cohesion=0.2)
    assert [s.raw_text for s in result.segments] == [text]
    assert "rejected_semantic_dependency" in result.gap_decisions[0].reason_codes
    assert "punctuation_continuation" in result.gap_decisions[0].reason_codes
    assert_integrity(text, result)


def test_explicit_repeated_label_directory_binds_nested_title_and_description():
    """同一裸标签在下行标题重现，提供目录层级证据，避免住宿总题头孤立。"""
    parts = ["沿河三天慢游指南，写下这次旅行的安排，方便以后再来的时候参考。", "📌留宿",
             "🏨留宿首选：河边｜车站附近", "这里离车站比较近，步行到河岸也很方便，晚上返回时沿街还有灯光。"]
    text, result = run(parts, selected=(0, 1, 2), cohesion=0.80, absolute_cohesion_threshold=0.85)
    assert [s.raw_text for s in result.segments] == [parts[0], "\n".join(parts[1:])]
    assert "rejected_semantic_dependency" in result.gap_decisions[1].reason_codes
    assert_integrity(text, result)


def test_sentence_final_day_title_binds_following_route_without_new_gap():
    """文首导语末尾附带Day标题时，既有gap不能将该标题甩在上一段。"""
    parts = ["山河三天慢游笔记。📍Day1", "【车站→河岸→旧城→山顶步道→林间书屋→河畔展馆→住宿】",
             "这一天走过很多地方，最后沿着河岸返回，在入口附近休息了一会才回住处。"]
    text, units, decision = fixture(parts, selected=(0,), cohesion=0.80)
    config = SemanticConfig(min_segment_nonspace_chars=1, max_segment_tokens=512, absolute_cohesion_threshold=0.85)
    result = partition_by_meaning(text, units, decision, config, CharacterCounter())
    assert result.segments[0].raw_text.startswith("\n".join(parts[:2]))
    assert "rejected_semantic_dependency" in result.gap_decisions[0].reason_codes
    assert_integrity(text, result)


def test_shop_name_directory_lead_and_explanation_do_not_split_apart():
    """连续推荐的目录与多家店铺说明整体保留，不把上一交通陈述并入。"""
    parts = ["✅这里没有公共单车❌", "🙋这次喜欢的几家小店↓", "林间茶屋🍵",
             "茶屋院子里种着几株大树，树下放有桌椅，我们坐在那里喝茶聊天。",
             "河畔面馆🥣", "这家面馆的面条现点现煮，店主还准备了几种可以自行选择的小菜。"]
    text, result = run(parts, selected=(0, 1, 2, 3, 4))
    assert all(s.raw_text not in parts[1:3] + [parts[4]] for s in result.segments)
    assert any(s.raw_text.startswith(parts[1] + "\n" + parts[2]) for s in result.segments)
    assert not any(block.start == 0 for block in result.protected_blocks)
    assert [s.raw_text for s in result.segments] == [parts[0], "\n".join(parts[1:])]
    assert_integrity(text, result)


def test_complete_route_directory_does_not_force_a_cut_at_every_day_number():
    """箭头日程是完整路线清单，不能仅因D1/D2出现就逐项必切。"""
    parts = ["🙋三天路线如下↓", "D1车站→河岸→旧城", "D2山脚→山顶→小镇", "D3书屋→公园→返程"]
    text, result = run(parts)
    assert [s.raw_text for s in result.segments] == [text]
    assert result.structural_boundaries == ()
    assert_integrity(text, result)


def test_natural_gap_absolute_score_can_qualify_without_being_a_local_valley():
    """句内更低分不能遮蔽自然gap；绝对阈值可在无谷值时提名自然端点。"""
    parts = ["我们先沿着河岸慢慢散步，仔细观察沿途树木和桥梁，最后回到入口安静休息。",
             "晚上转到镇里的展馆观看演出，演员介绍了当地历史，散场之后我们乘车返回。"]
    text, result = run(parts, cohesion=0.80, absolute_cohesion_threshold=0.85)
    assert result.semantic_boundaries == (0,)
    assert "natural_absolute_cohesion" in result.gap_decisions[0].sources
    _, conservative = run(parts, cohesion=0.80, absolute_cohesion_threshold=0.55)
    assert len(conservative.segments) == 1
    assert_integrity(text, result)


def test_attribute_and_sentence_end_alone_do_not_generate_boundaries():
    """票价/交通字段与普通句号本身都不能成为主动切分理由。"""
    parts = ["沿着山坡可以走到河畔的平台。", "门票：十元", "交通：小巴可到", "回来时我们还在路边休息了一会。"]
    text, result = run(parts)
    assert result.semantic_boundaries == result.structural_boundaries == ()
    assert [s.raw_text for s in result.segments] == [text]
    assert_integrity(text, result)


def test_independent_transport_field_can_follow_a_complete_experience():
    """属性字段不一律否决独立说明；有语义信号且前一意思完整时可以切。"""
    parts = ["平台上视野开阔，可以看到河道转弯和远处山林，我们在这里停留很久才离开。",
             "交通：从镇里的车站乘坐小巴到山脚，下车以后沿指示牌走到步道入口即可。"]
    text, result = run(parts, selected=(0,))
    assert result.semantic_boundaries == (0,)
    assert "before_attribute_line" in result.gap_decisions[0].reason_codes
    assert result.forced_structure_boundaries == ()
    assert_integrity(text, result)


def test_pure_metadata_tail_attaches_to_the_last_complete_description():
    """尾部话题和账号不能单独成段，即使其gap被原评分选中。"""
    parts = ["在山间走过一条长长的步道，终于到达可以坐下来休息的观景台。",
             "离开之前我们走到了河边的入口，仔细查看返程方向以后才坐上回城的车。", "#山林记录# @旅客"]
    text, result = run(parts, selected=(0, 1))
    assert result.segments[-1].raw_text.endswith("\n".join(parts[1:]))
    assert "inside_trailing_metadata_context" in result.gap_decisions[1].reason_codes
    assert_integrity(text, result)


def test_missing_unit_gap_is_reported_and_no_embedding_or_new_gap_is_invented():
    """新标题落在单元内部时只报告坐标，不将最近逗号偷换成标题边界。"""
    parts = ["前一天已经结束。\n📅Day2 山林步道", "沿着树林之间的小路慢慢前行，随后从山腰走到了宽阔的平台。"]
    text, result = run(parts)
    assert result.unavailable_structural_spans
    assert len(result.gap_decisions) == 1
    assert [s.raw_text for s in result.segments] == [text]
    assert_integrity(text, result)


def test_nonmonotonic_exact_tokens_and_forced_internal_split_remain_visible():
    """长度搜索不能在第一个超限前缀停下，且不可避免句内切断有trace。"""
    text, units, decision = fixture(["甲乙", "丙丁", "戊己", "庚辛"], separator="")
    class Counter(CharacterCounter):
        """合成非单调分词器，长前缀反而能落入硬长度预算。"""
        def count(self, value):
            return 3 if value == "甲乙丙丁戊己" else len(value)
    config = SemanticConfig(unit_max_tokens=2, max_segment_tokens=3, min_segment_nonspace_chars=1)
    result = partition_by_meaning(text, units, decision, config, Counter())
    assert [s.raw_text for s in result.segments] == ["甲乙丙丁戊己", "庚辛"]
    assert result.forced_structure_boundaries == (2,)
    assert "forced_structure_dependency" in result.gap_decisions[2].reason_codes
    assert all(Counter().count(s.raw_text) <= 3 for s in result.segments)


def test_metadata_only_length_tail_is_rejected_if_no_feasible_attachment_exists():
    """硬上限与尾标签绑定真的无解时失败，不返回看似成功的标签孤段。"""
    text, units, decision = fixture(["正文。", "#标签#"])
    config = SemanticConfig(unit_max_tokens=4, max_segment_tokens=4, min_segment_nonspace_chars=1)
    with pytest.raises(ValueError, match="unsplittable_meaning_segment_limit"):
        partition_by_meaning(text, units, decision, config, CharacterCounter())


def test_unicode_exact_span_empty_single_and_invalid_inputs():
    """组合字素不被切开；空/单单元也验证输入，坏计数和缺字不能侥幸过关。"""
    text, units, decision = fixture(["✅林间书屋", "e\u0301与家人👨‍👩‍👧‍👦一起休息。", "✅河畔展馆", "沿河慢慢走到入口。"], separator="\r\n  ")
    config = SemanticConfig(min_segment_nonspace_chars=1)
    result = partition_by_meaning(text, units, decision, config, CharacterCounter())
    assert_integrity(text, result, 384)
    blank = SemanticDecision("empty", (), (), (), ())
    assert partition_by_meaning(" \n", (), blank, config, CharacterCounter()).segments == ()
    single = partition_by_meaning("独立文字", (TextUnit(0, 4, 4),), blank, config, CharacterCounter())
    assert single.experiment_id == EXPERIMENT_ID and len(single.segments) == 1
    with pytest.raises(ValueError):
        partition_by_meaning("缺字", (), blank, config, CharacterCounter())
    with pytest.raises(ValueError, match="meaning_unit_token_count_mismatch"):
        partition_by_meaning("甲", (TextUnit(0, 1, 2),), blank, config, CharacterCounter())
    with pytest.raises(ValueError, match="字素"):
        partition_by_meaning("e\u0301", (TextUnit(0, 1, 1), TextUnit(1, 2, 1)), replace(blank, gaps=(GapScore(0, 1.0, 0.0, False, False, "synthetic"),)), config, CharacterCounter())


def test_new_experimental_path_does_not_mutate_old_rules_or_inputs():
    """同一输入先后调用旧0.1/0.2结果一致，新规则也不借用正式身份。"""
    text, units, _ = fixture(["✅河畔展台", "栏杆外可以看到河水和小船。", "✅林间书屋", "屋内的窗边有安静的座椅。"])
    config, counter = SemanticConfig(min_segment_nonspace_chars=1), CharacterCounter()
    vectors = [[1.0, 0.0]] * len(units)
    old_segments, decision = choose_boundaries(text, units, vectors, config, counter)
    before = partition_with_structure(text, units, decision, config, counter, mode="context")
    experimental = partition_by_meaning(text, units, decision, config, counter)
    assert choose_boundaries(text, units, vectors, config, counter) == (old_segments, decision)
    assert partition_with_structure(text, units, decision, config, counter, mode="context") == before
    assert len(experimental.segments) == 2
    assert experimental.experiment_id.startswith("experimental-")


@pytest.mark.parametrize("title_has_gap", [True, False])
def test_sentence_final_schedule_uses_its_own_start_not_the_prior_narration(title_has_gap):
    """同行句后天序只能用自己的真实起点，不把上一句开头替代为结构切点。"""
    previous = "上一天已经结束。\n我们回到了住处。"
    title, body = "📍Day2", "我们沿着山路慢慢走到展台。"
    text = previous + title + "\n" + body
    title_start = len(previous)
    spans = [(0, 8), (9, title_start), (title_start, title_start + len(title)),
             (title_start + len(title) + 1, len(text))]
    if not title_has_gap:
        spans[1:3] = [(spans[1][0], spans[2][1])]
    units = tuple(TextUnit(a, b, b - a) for a, b in spans)
    decision = SemanticDecision("scored_keep", (), tuple(
        GapScore(i, 0.95, 0.0, False, False, "synthetic") for i in range(len(units) - 1)), (), ())
    config = SemanticConfig(min_segment_nonspace_chars=1, max_segment_tokens=512)
    result = partition_by_meaning(text, units, decision, config, CharacterCounter())
    block = next(b for b in result.protected_blocks if b.kind == "trailing_schedule_heading")
    assert block.start == title_start
    assert ("heading", title_start, title_start + len(title)) in block.evidence
    if title_has_gap:
        assert result.structural_boundaries == (1,)
        assert [s.raw_text for s in result.segments] == [previous, title + "\n" + body]
        assert result.unavailable_structural_spans == ()
    else:
        assert result.structural_boundaries == ()
        assert [s.raw_text for s in result.segments] == [text]
        assert result.unavailable_structural_spans == ((block.kind, title_start, len(text)),)
    assert_integrity(text, result)


@pytest.mark.parametrize("tail", ["https://example.invalid/trip", "#山路# 🎈",
                                    "#山路# @旅客 https://example.invalid/trip 🎈", "✨"])
def test_pure_metadata_forms_share_the_same_tail_attachment(tail):
    """纯链接、标签账号混排及纯装饰尾行都依赖正文，不单独输出。"""
    parts = ["我们沿着山路慢慢走到展台。", tail]
    text, units, decision = fixture(parts, selected=(0,))
    config = SemanticConfig(min_segment_nonspace_chars=1, max_segment_tokens=512)
    result = partition_by_meaning(text, units, decision, config, CharacterCounter())
    assert [s.raw_text for s in result.segments] == [text]
    assert "inside_trailing_metadata_context" in result.gap_decisions[0].reason_codes
    assert_integrity(text, result)


@pytest.mark.parametrize("tail", ["https://example.invalid/trip", "#山路# 🎈"])
def test_pure_metadata_tail_fails_when_hard_limit_prevents_attachment(tail):
    """每个单元可装下但正文与纯尾行合并无解时，不用元数据孤段回避上限。"""
    text, units, decision = fixture(["正文。", tail])
    maximum = len(tail)
    config = SemanticConfig(unit_max_tokens=maximum, max_segment_tokens=maximum,
                            min_segment_nonspace_chars=1)
    with pytest.raises(ValueError, match="unsplittable_meaning_segment_limit"):
        partition_by_meaning(text, units, decision, config, CharacterCounter())


@pytest.mark.parametrize("body", ["我们看到了#山路#，后来在河岸休息。",
                                    "我们参考 https://example.invalid/trip 选择了返程路线。"])
def test_substantive_tail_mentioning_metadata_remains_independent(body):
    """含标签或链接的普通陈述仍是正文，不能被纯尾部规则强制合并。"""
    parts = ["我们沿着山路慢慢走到展台。", body]
    text, units, decision = fixture(parts, selected=(0,))
    config = SemanticConfig(min_segment_nonspace_chars=1, max_segment_tokens=512)
    result = partition_by_meaning(text, units, decision, config, CharacterCounter())
    assert [s.raw_text for s in result.segments] == parts
    assert not any(block.kind == "trailing_metadata_context" for block in result.protected_blocks)
    assert_integrity(text, result)


def test_component_introduction_keeps_its_numbered_parts_before_experience_section():
    """同一地点简介与组成清单保持整体，语义低分不能逐项拆开父介绍块。"""
    parts = ["云岸展馆是一座沿河建设的公共展览空间。", "云岸展馆主要分为三个部分：",
             "1、入口广场：这里可以休息，也能欣赏河边的树木。",
             "2、历史展厅：这里介绍城镇的发展过程，可以了解当地历史。",
             "3、河边平台：平台设有长椅，适合休息和观察来往的小船。",
             "【实际游览体验】：", "我们中午来到展馆，发现入口广场没有遮阴，排队时十分闷热。"]
    text, units, decision = fixture(parts, selected=tuple(range(len(parts) - 1)), cohesion=0.2)
    result = partition_by_meaning(text, units, decision,
                                 SemanticConfig(min_segment_nonspace_chars=1, max_segment_tokens=512),
                                 CharacterCounter())
    assert [s.raw_text for s in result.segments] == ["\n".join(parts[:5]), "\n".join(parts[5:])]
    assert any("inside_coherent_enumeration" in trace.reason_codes for trace in result.gap_decisions)
    assert_integrity(text, result)


def test_same_day_recommendations_stay_together_but_next_day_can_start_a_segment():
    """日程中的连续推荐不逐景点拆，下一日程仍是可定位的父层边界。"""
    parts = ["📅Day1 河岸慢游", "展馆→书屋→河湾", "✅展馆", "展厅里的陈列很丰富。",
             "✅书屋", "靠窗的座位安静，可以在这里读书和休息。",
             "📅Day2 山间步道", "我们坐车来到山脚，沿着步道前行，在平台停留之后原路返回。"]
    text, units, decision = fixture(parts, selected=tuple(range(len(parts) - 1)), cohesion=0.2)
    result = partition_by_meaning(text, units, decision,
                                 SemanticConfig(min_segment_nonspace_chars=1, max_segment_tokens=512),
                                 CharacterCounter())
    assert [s.raw_text for s in result.segments] == ["\n".join(parts[:6]), "\n".join(parts[6:])]
    assert any("inside_coherent_schedule" in trace.reason_codes for trace in result.gap_decisions)
    assert_integrity(text, result)


def test_oversized_coherent_group_uses_whole_items_when_they_fit_the_hard_limit():
    """完整块超硬上限时可退让，但有完整条目方案时不切开条目说明。"""
    parts = ["本次路线主要包括三个部分：",
             "1、河边展馆：沿河慢慢散步可以来到展馆，屋里陈列了当地的历史资料。",
             "2、山间平台：沿着石阶可以来到平台，在这里能够观察河流和周围的树林。",
             "3、镇中书屋：屋里有供游客阅读和休息的座位，窗外是一条安静的街道。"]
    text, units, decision = fixture(parts, selected=(0, 1, 2), cohesion=0.2)
    config = SemanticConfig(unit_max_tokens=64, max_segment_tokens=75, min_segment_nonspace_chars=1)
    result = partition_by_meaning(text, units, decision, config, CharacterCounter())
    assert len(result.segments) > 1
    items = [block for block in result.protected_blocks if block.kind == "coherent_item"]
    assert len(items) == 3
    assert all(not (item.start < segment.end < item.end)
               for segment in result.segments[:-1] for item in items)
    assert result.length_boundaries and result.forced_structure_boundaries
    assert_integrity(text, result, 75)


def test_hard_limit_prefers_two_between_item_cuts_to_one_inside_item_cut():
    """不能用少切一次父块抵消完整条目被切断；各项含多个已有计算单元。"""
    lines = ["游园路线分为三个部分："] + [
        f"{n}、展区：" + "甲" * 26 + "。" + "乙" * 26 + "。" for n in range(1, 4)]
    text = "\n".join(lines)
    units = [TextUnit(0, len(lines[0]), len(lines[0]))]
    cursor = len(lines[0]) + 1
    for line in lines[1:]:
        middle = line.index("。") + 1
        units.extend((TextUnit(cursor, cursor + middle, middle),
                      TextUnit(cursor + middle, cursor + len(line), len(line) - middle)))
        cursor += len(line) + 1
    decision = SemanticDecision("scored_keep", (), tuple(
        GapScore(i, .95, 0.0, False, False, "synthetic") for i in range(len(units) - 1)), (), ())
    config = SemanticConfig(unit_max_tokens=64, max_segment_tokens=105,
                            min_segment_nonspace_chars=1)
    result = partition_by_meaning(text, units, decision, config, CharacterCounter())
    assert [s.raw_text for s in result.segments] == ["\n".join(lines[:2]), *lines[2:]]
    assert result.length_boundaries == (2, 4)
    assert all("inside_coherent_item" not in result.gap_decisions[i].reason_codes
               for i in result.length_boundaries)
    assert_integrity(text, result, 105)
