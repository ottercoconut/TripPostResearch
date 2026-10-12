"""结构候选的全局长度、不可避免切断和非单调BPE反例。"""
from dataclasses import asdict

import pytest

from tourism_ugc_study.segmentation.contracts import GapScore, SemanticDecision, TextUnit
from tourism_ugc_study.segmentation.integrity import WHITE, validate
from tourism_ugc_study.segmentation.semantic_config import SemanticConfig
from tourism_ugc_study.segmentation.semantic_partition import partition_with_structure


class CharacterCounter:
    """每码点一token的合成计数器，不冒称真实Qwen。"""
    def count(self, text):
        return len(text)


def fixture(parts, *, semantic=(), depths=None, separator=""):
    """手工构造无损单元及评分，便于区分语义与长度选择。"""
    text=separator.join(parts)
    units=[]
    cursor=0
    for part in parts:
        units.append(TextUnit(cursor,cursor+len(part),len(part)))
        cursor+=len(part)+len(separator)
    gaps=tuple(GapScore(i,0.8,float(depths[i] if depths else 0),True,i in semantic,'synthetic') for i in range(len(parts)-1))
    decision=SemanticDecision('scored_split' if semantic else 'scored_keep',(),gaps,semantic,())
    return text,units,decision


def test_global_length_avoids_greedy_extra_segments():
    """先取局部深谷会产生4段，但全局可在同一硬上限内只需3段。"""
    text,units,decision=fixture(['甲。','乙。','丙。','丁。','戊。'],depths=[10,9,1,8])
    config=SemanticConfig(unit_max_tokens=2,max_segment_tokens=4,min_segment_nonspace_chars=1)
    args=(text,units,decision,config,CharacterCounter())
    greedy=partition_with_structure(*args,mode='sentence_only',length_strategy='greedy')
    optimal=partition_with_structure(*args,mode='sentence_only',length_strategy='global')
    assert len(greedy.segments)==4
    assert len(optimal.segments)==3
    assert optimal.forced_structure_boundaries==()
    validate(text,[asdict(s) for s in optimal.segments])


def test_token_fragment_semantic_cut_is_suppressed_without_text_loss():
    """同一句内为模型细分的gap不能自动成为最终主题边界。"""
    text,units,decision=fixture(['海边有风，','但夕阳很好。'],semantic=(0,))
    result=partition_with_structure(text,units,decision,SemanticConfig(),CharacterCounter(),mode='sentence_only')
    assert result.suppressed_semantic_boundaries==(0,)
    assert result.semantic_boundaries==()
    assert [s.raw_text for s in result.segments]==[text]


def test_unavoidable_sentence_cut_is_explicit():
    """没有自然边界的超长文本只能有损于上下文，但仍无损于原文字节与覆盖。"""
    text,units,decision=fixture(['甲乙','丙丁','戊己'])
    config=SemanticConfig(unit_max_tokens=2,max_segment_tokens=4,min_segment_nonspace_chars=1)
    result=partition_with_structure(text,units,decision,config,CharacterCounter(),mode='sentence_only')
    assert len(result.segments)==2
    assert len(result.forced_structure_boundaries)==1
    assert ''.join(s.raw_text for s in result.segments)==text


def test_nonmonotonic_counter_does_not_stop_at_first_overflow():
    """中间前缀超限、较长前缀又合规时，仍能找到两段路径。"""
    text,units,decision=fixture(['甲。','乙。','丙。','丁。'])
    class Counter(CharacterCounter):
        def count(self, value):
            return 3 if value=='甲。乙。丙。' else len(value)
    config=SemanticConfig(unit_max_tokens=2,max_segment_tokens=3,min_segment_nonspace_chars=1)
    result=partition_with_structure(text,units,decision,config,Counter(),mode='sentence_only')
    assert [s.raw_text for s in result.segments]==['甲。乙。丙。','丁。']


def test_unknown_strategy_and_mismatched_gaps_fail():
    """非法策略和不同单元的trace不能混用。"""
    text,units,decision=fixture(['甲。','乙。'])
    with pytest.raises(ValueError,match='unknown_partition_length_strategy'):
        partition_with_structure(text,units,decision,SemanticConfig(),CharacterCounter(),length_strategy='guess')
    with pytest.raises(ValueError,match='partition_gap_identity_mismatch'):
        partition_with_structure(text,units[:1],decision,SemanticConfig(),CharacterCounter())


def test_repeated_entry_protects_heading_and_body_without_new_boundaries():
    """重复条目可装入段长时，题头后的语义切点应抑制并保留条目完整。"""
    parts=['✅山间书屋','沿着步道走到木屋可以安静读书。','✅滨水展馆','展厅展示地方历史，窗外可以看见河岸。']
    text='\n'.join(parts)
    units=[]
    cursor=0
    for part in parts:
        units.append(TextUnit(cursor,cursor+len(part),len(part)))
        cursor+=len(part)+1
    gaps=tuple(GapScore(i,0.8,0.0,True,i==0,'synthetic') for i in range(3))
    decision=SemanticDecision('scored_split',(),gaps,(0,),())
    config=SemanticConfig(unit_max_tokens=24,max_segment_tokens=30,min_segment_nonspace_chars=3)
    result=partition_with_structure(text,units,decision,config,CharacterCounter(),mode='blocks')
    assert result.suppressed_semantic_boundaries==(0,)
    assert len(result.protected_blocks)==2
    assert result.length_boundaries==(1,)
    assert result.forced_structure_boundaries==()
    assert len(result.segments)==2


def assert_complete_and_bounded(text, result, maximum):
    """独立核对原文内容顺序和精确合成长度，再调用公共完整性验收。"""
    expected = ''.join(char for char in text if char not in WHITE)
    actual = ''.join(char for segment in result.segments for char in segment.raw_text if char not in WHITE)
    assert actual == expected
    assert all(segment.raw_text == text[segment.start:segment.end] for segment in result.segments)
    assert all(len(segment.raw_text) <= maximum for segment in result.segments)
    validate(text, [asdict(segment) for segment in result.segments])


@pytest.mark.parametrize('headings', [('建筑观察',), ('建筑观察', '材料与开口')])
@pytest.mark.parametrize('strategy', ['global', 'greedy'])
def test_context_keeps_plain_heading_chain_with_its_first_description(headings, strategy):
    """普通章节名不会悬在前段，已有上下文外的主题切点仍保留。"""
    description = '外墙使用浅色石材，窗边设有木制遮阳板，门厅还保留通向庭院的完整通道。'
    parts = ['前一节已经结束。', *headings, description, '下一节另作完整说明。']
    text, units, decision = fixture(parts, semantic=tuple(range(len(parts) - 1)), separator='\n')
    config = SemanticConfig(unit_max_tokens=64, max_segment_tokens=128, min_segment_nonspace_chars=1)
    result = partition_with_structure(text, units, decision, config, CharacterCounter(),
                                      mode='context', length_strategy=strategy)
    assert [segment.raw_text for segment in result.segments] == [
        parts[0], '\n'.join([*headings, description]), parts[-1],
    ]
    assert result.forced_structure_boundaries == ()
    assert result.block_relaxations == ()
    assert any(kind == 'heading_context' for _, _, kind in result.protected_blocks)
    assert_complete_and_bounded(text, result, config.max_segment_tokens)


@pytest.mark.parametrize('strategy', ['global', 'greedy'])
def test_context_attaches_trailing_mixed_metadata_without_merging_prior_topic(strategy):
    """标签与账号仍保留原文并归最后说明，不能单独悬空或吞掉前一主题。"""
    parts = ['前一段已经完整结束。', '最后一句说明出口的位置。', '#沿途记录# @行旅人', '@同行者 #建筑记录#']
    text, units, decision = fixture(parts, semantic=(0, 1, 2), separator='\n')
    config = SemanticConfig(unit_max_tokens=24, max_segment_tokens=64, min_segment_nonspace_chars=1)
    result = partition_with_structure(text, units, decision, config, CharacterCounter(),
                                      mode='context', length_strategy=strategy)
    assert [segment.raw_text for segment in result.segments] == [parts[0], '\n'.join(parts[1:])]
    assert result.suppressed_semantic_boundaries == (1, 2)
    assert result.semantic_boundaries == (0,)
    assert result.forced_structure_boundaries == ()
    assert result.block_relaxations == ()
    assert any(kind == 'trailing_metadata_context' for _, _, kind in result.protected_blocks)
    assert_complete_and_bounded(text, result, config.max_segment_tokens)


@pytest.mark.parametrize('strategy', ['global', 'greedy'])
def test_oversized_repeated_entry_falls_back_to_heading_and_first_sentence(strategy):
    """整条目超预算时仍保护可装下的题头首句，后续说明可在自然边界分段。"""
    parts = ['✅山间书屋', '书屋面对一处庭院。', '窗边设有供读者使用的长桌，后门通向林中的另一条步道。',
             '✅河畔展馆', '展厅展示本地的历史。']
    text, units, decision = fixture(parts, semantic=(0,), separator='\n')
    config = SemanticConfig(unit_max_tokens=30, max_segment_tokens=32, min_segment_nonspace_chars=1)
    assert len('\n'.join(parts[:3])) > config.max_segment_tokens
    assert len('\n'.join(parts[:2])) <= config.max_segment_tokens
    result = partition_with_structure(text, units, decision, config, CharacterCounter(),
                                      mode='context', length_strategy=strategy)
    assert [segment.raw_text for segment in result.segments] == [
        '\n'.join(parts[:2]), parts[2], '\n'.join(parts[3:]),
    ]
    assert result.suppressed_semantic_boundaries == (0,)
    assert result.forced_structure_boundaries == ()
    relaxation, = (item for item in result.block_relaxations if item.kind == 'repeated_entry' and item.start == 0)
    assert relaxation.reason == 'whole_block_over_token_limit'
    assert text[relaxation.start:relaxation.end] == '\n'.join(parts[:3])
    assert relaxation.token_count == len('\n'.join(parts[:3])) > config.max_segment_tokens
    assert relaxation.retained_end == len('\n'.join(parts[:2]))
    assert_complete_and_bounded(text, result, config.max_segment_tokens)


@pytest.mark.parametrize('strategy', ['global', 'greedy'])
def test_oversized_metadata_context_yields_to_hard_limit_without_losing_tags(strategy):
    """尾部依赖本身超限且无更短回退时，允许拆开但不突破预算或删元数据。"""
    parts = ['最后一句说明出口。', '#山间建筑观察# @行旅者', '#沿途步行记录# @同行者']
    text, units, decision = fixture(parts, separator='\n')
    config = SemanticConfig(unit_max_tokens=16, max_segment_tokens=18, min_segment_nonspace_chars=1)
    assert all(len(part) <= config.max_segment_tokens for part in parts)
    result = partition_with_structure(text, units, decision, config, CharacterCounter(),
                                      mode='context', length_strategy=strategy)
    assert len(result.segments) >= 2
    assert result.length_boundaries
    assert not any(kind == 'trailing_metadata_context' for _, _, kind in result.protected_blocks)
    relaxation, = result.block_relaxations
    assert relaxation.kind == 'trailing_metadata_context'
    assert relaxation.reason == 'whole_block_over_token_limit'
    assert (relaxation.start, relaxation.end) == (0, len(text))
    assert relaxation.token_count == len(text) > config.max_segment_tokens
    assert relaxation.retained_end is None
    assert_complete_and_bounded(text, result, config.max_segment_tokens)


@pytest.mark.parametrize('strategy', ['global', 'greedy'])
def test_oversized_heading_first_sentence_records_complete_relaxation(strategy):
    """题头首句本身也超限时明确放弃该保护，不能伪报成功保留或拆坏原文。"""
    heading = '建筑观察'
    body_parts = ['外墙采用浅色石材，', '窗边设有木制遮阳板，', '门厅保留通向庭院的完整通道。']
    body, body_units, _ = fixture(body_parts)
    text = heading + '\n' + body
    offset = len(heading) + 1
    units = [TextUnit(0, len(heading), len(heading)),
             *(TextUnit(unit.start + offset, unit.end + offset, unit.token_count) for unit in body_units)]
    gaps = tuple(GapScore(i, 0.8, 0.0, True, i == 0, 'synthetic') for i in range(len(units) - 1))
    decision = SemanticDecision('scored_split', (), gaps, (0,), ())
    config = SemanticConfig(unit_max_tokens=20, max_segment_tokens=22, min_segment_nonspace_chars=1)
    result = partition_with_structure(text, units, decision, config, CharacterCounter(),
                                      mode='context', length_strategy=strategy)
    relaxation, = result.block_relaxations
    assert relaxation.kind == 'heading_context' and relaxation.reason == 'whole_block_over_token_limit'
    assert (relaxation.start, relaxation.end, relaxation.token_count) == (0, len(text), len(text))
    assert relaxation.retained_end is None
    assert not any(kind == 'heading_context' for _, _, kind in result.protected_blocks)
    assert result.forced_structure_boundaries
    assert_complete_and_bounded(text, result, config.max_segment_tokens)


@pytest.mark.parametrize('strategy', ['global', 'greedy'])
def test_overlapping_valid_contexts_report_unavoidable_protected_cut(strategy):
    """题头首句和尾标签依赖分别可装下，合起来超限时所有切口均有结构代价。"""
    parts = ['建筑观察', '展厅沿着山坡逐级展开，各层以连廊相连，窗边保留宽阔的休息空间。', '#建筑记录# @观察者']
    text, units, decision = fixture(parts, semantic=(0, 1), separator='\n')
    maximum = max(len('\n'.join(parts[:2])), len('\n'.join(parts[1:])))
    config = SemanticConfig(unit_max_tokens=max(map(len, parts)), max_segment_tokens=maximum,
                            min_segment_nonspace_chars=1)
    assert len(text) > maximum
    result = partition_with_structure(text, units, decision, config, CharacterCounter(),
                                      mode='context', length_strategy=strategy)
    assert {kind for _, _, kind in result.protected_blocks} == {'heading_context', 'trailing_metadata_context'}
    assert all(gap.penalty > 0 for gap in result.structure)
    assert result.semantic_boundaries == () and result.suppressed_semantic_boundaries == (0, 1)
    assert len(result.segments) == 2
    assert result.length_boundaries and result.forced_structure_boundaries == result.length_boundaries
    assert result.block_relaxations == ()
    assert_complete_and_bounded(text, result, maximum)


@pytest.mark.parametrize('text', ['', ' \r\n'])
def test_context_empty_input_has_no_fabricated_blocks_or_cuts(text):
    """空正文仍是成功空结果，不虚构题头、标签依赖或强制切点。"""
    decision = SemanticDecision('empty', (), (), (), ())
    config = SemanticConfig()
    result = partition_with_structure(text, (), decision, config, CharacterCounter(), mode='context')
    assert result.segments == () and result.structure == () and result.protected_blocks == ()
    assert result.block_relaxations == ()
    assert result.semantic_boundaries == result.length_boundaries == result.forced_structure_boundaries == ()
    assert_complete_and_bounded(text, result, config.max_segment_tokens)


def test_context_single_unit_preserves_body_and_rejects_unrepresentable_limit():
    """没有既有gap时保留完整单元，硬上限容不下则明确失败而非新增切点。"""
    text, units, decision = fixture(['完整的一句话。'])
    config = SemanticConfig(unit_max_tokens=len(text), max_segment_tokens=len(text), min_segment_nonspace_chars=1)
    result = partition_with_structure(text, units, decision, config, CharacterCounter(), mode='context')
    assert [segment.raw_text for segment in result.segments] == [text]
    assert result.structure == () and result.protected_blocks == ()
    assert result.block_relaxations == ()
    assert result.semantic_boundaries == result.length_boundaries == result.forced_structure_boundaries == ()
    assert_complete_and_bounded(text, result, config.max_segment_tokens)
    too_small = SemanticConfig(unit_max_tokens=1, max_segment_tokens=len(text) - 1, min_segment_nonspace_chars=1)
    with pytest.raises(ValueError, match='^unsplittable_segment_limit$'):
        partition_with_structure(text, units, decision, too_small, CharacterCounter(), mode='context')
