"""候选单元的真实反例、无损覆盖和计数失败测试；不下载模型或读科研正文。"""

import subprocess
import sys

import pytest
import regex

from tourism_ugc_study.segmentation.integrity import WHITE
from tourism_ugc_study.segmentation.semantic_config import SemanticConfig
from tourism_ugc_study.segmentation.semantic_units import build_units


class CodepointCounter:
    """可手算的测试 tokenizer；生产路径不得用该字符计数替代真实模型。"""

    max_input_tokens = 256

    def count(self, text):
        """每个码点视为一个模拟 token。"""
        return len(text)

    def input_count(self, text):
        """固定十个模板 token，验证正文预算与实际输入预算分离。"""
        return self.count(text) + 10

    def offsets(self, text):
        """一个模拟 token 指向一个原文码点。"""
        return tuple((i, i + 1) for i in range(len(text)))


def _parts(text, *, counter=None, config=None):
    """每个场景同时校验独立的非空白覆盖与完整字素端点。"""
    result = build_units(text, counter or CodepointCounter(), config or SemanticConfig())
    boundaries = {0, *(match.end() for match in regex.finditer(r"\X", text))}
    covered = [0] * len(text)
    previous_end = 0
    for unit in result.units:
        assert previous_end <= unit.start < unit.end
        assert unit.start in boundaries and unit.end in boundaries
        for index in range(unit.start, unit.end):
            covered[index] += 1
        previous_end = unit.end
    assert all(covered[i] == 1 for i, char in enumerate(text) if char not in WHITE)
    assert all(count <= 1 for count in covered)
    return [text[unit.start:unit.end] for unit in result.units]


@pytest.mark.parametrize("text,expected", [
    ("", []),
    (" \r\n\t\u3000", []),
    ("海边真好看。", ["海边真好看。"]),
    ("好看！方便！", ["好看！", "方便！"]),
    ("相同的句子。相同的句子。", ["相同的句子。", "相同的句子。"]),
    ("甲\r\n\r\n乙\r丙\n丁\u2028戊\u2029己", ["甲", "乙", "丙", "丁", "戊", "己"]),
    ("“海边真美？！”公交方便。", ["“海边真美？！”", "公交方便。"]),
    ("（风景好看。）[坐车方便！]", ["（风景好看。）", "[坐车方便！]"]),
    ("票价3.5元，网址https://a.example/x。", ["票价3.5元，网址https://a.example/x。"]),
    ("网址https://a.example/?x=1;y=2。公交方便。", ["网址https://a.example/?x=1;y=2。", "公交方便。"]),
    ("https://a.example/!x?y;z 海边！", ["https://a.example/!x?y;z 海边！"]),
    ("https://a.example/x'后面！下一句", ["https://a.example/x'后面！", "下一句"]),
    ("甲。\u0301 \u0301乙。", ["甲。\u0301", " \u0301乙。"]),
    (" \u0301\n乙。", [" \u0301", "乙。"]),
    ("甲?\ufe0f乙。", ["甲?\ufe0f", "乙。"]),
    ("\ufeff甲。", ["\ufeff甲。"]),
    ("👨‍👩‍👧‍👦。🌊！", ["👨‍👩‍👧‍👦。", "🌊！"]),
    ("！？\n甲乙。\n；\n下一句", ["！？\n甲乙。\n；", "下一句"]),
    ("！？\n；", ["！？\n；"]),
    ("🌊\n甲乙。", ["🌊", "甲乙。"]),
])
def test_sentence_and_grapheme_contract(text, expected):
    """短候选保持分立、URL轻保护、标点归邻及字素 trim 均有具体反例。"""
    assert _parts(text) == expected


def test_repeated_sentences_have_distinct_positions():
    """重复句必须由扫描获得各自位置，不能 str.find 回到第一次。"""
    result = build_units("重复。重复。", CodepointCounter(), SemanticConfig())
    assert [(unit.start, unit.end) for unit in result.units] == [(0, 3), (3, 6)]


def test_weak_boundary_precedes_token_offset_fallback():
    """超限时优先选上限内最靠后的弱边界，余下无标点再按 offset。"""
    config = SemanticConfig(unit_max_tokens=4)
    assert _parts("甲乙，丙丁戊己庚", config=config) == ["甲乙，", "丙丁戊己", "庚"]
    assert _parts("甲 乙 丙丁戊己", config=config) == ["甲 乙", "丙丁戊己"]


def test_nonmonotone_prefix_counts_are_not_binary_searched():
    """更长的 BPE 前缀可以更少 token，必须找到靠后仍可行的弱边界。"""
    class NonmonotoneCounter(CodepointCounter):
        def count(self, text):
            return {"甲，乙，": 5, "甲，乙，丙，": 2}.get(text, len(text))

    assert _parts("甲，乙，丙，丁", counter=NonmonotoneCounter(), config=SemanticConfig(unit_max_tokens=3)) == ["甲，乙，丙，", "丁"]


def test_offset_suggestion_is_recounted_then_backed_off():
    """offset 只提议位置；切出的子串重新 tokenize 后超限需逐字素后退。"""
    class PrefixCounter(CodepointCounter):
        def count(self, text):
            return 3 if text == "ab" else len(text)

    assert _parts("abcdef", counter=PrefixCounter(), config=SemanticConfig(unit_max_tokens=2)) == ["a", "bc", "de", "f"]


def test_symbol_and_url_runs_obey_unit_limit():
    """纯符号和超长URL不享受结构保护，也不能拆后再次合回超长块。"""
    config = SemanticConfig(unit_max_tokens=4)
    assert _parts("++++++++++", config=config) == ["++++", "++++", "++"]
    parts = _parts("https://example.test/?x=1234567;y=7654321", config=config)
    assert all(len(part) <= 4 for part in parts)


def test_oversized_grapheme_is_one_unit_and_records_diagnostic():
    """一个字素可超64正文token，但不可截断，且必须满足两项硬上限。"""
    glyph = "a" + "\u0301" * 70
    text = glyph + "尾声"
    result = build_units(text, CodepointCounter(), SemanticConfig())
    assert [text[unit.start:unit.end] for unit in result.units] == [glyph, "尾声"]
    assert result.diagnostics == ("unit_oversized_grapheme",)
    _parts(text)


def test_offsets_may_overlap_for_multiple_tokens_of_one_character():
    """Unicode 字节 BPE 的重复 offset 合法；端点最终由原文字素决定。"""
    class OverlapCounter(CodepointCounter):
        def offsets(self, text):
            return ((0, 1), (0, 1), (1, 2), (2, 3)) if text == "甲乙丙" else super().offsets(text)

        def count(self, text):
            return 4 if text == "甲乙丙" else len(text)

    assert _parts("甲乙丙", counter=OverlapCounter(), config=SemanticConfig(unit_max_tokens=2)) == ["甲", "乙丙"]


@pytest.mark.parametrize("glyph_tokens,max_input,max_segment,reason", [
    (71, 80, 384, "unrepresentable_grapheme"),
    (401, 256, 384, "unrepresentable_grapheme"),
    (401, 500, 384, "unsplittable_segment_limit"),
])
def test_grapheme_limit_failure_order(glyph_tokens, max_input, max_segment, reason):
    """同时违反模型和段上限时优先报告模型输入不可表示。"""
    counter = CodepointCounter()
    counter.max_input_tokens = max_input
    with pytest.raises(ValueError, match=f"^{reason}$"):
        build_units("a" + "\u0301" * (glyph_tokens - 1), counter, SemanticConfig(max_segment_tokens=max_segment))


def test_every_ordinary_unit_checks_complete_model_input():
    """即使正文 token 合格，模板导致超限也不能静默送给模型截断。"""
    counter = CodepointCounter()
    counter.max_input_tokens = 11
    with pytest.raises(ValueError, match="^model_input_limit_exceeded$"):
        build_units("甲乙", counter, SemanticConfig())


@pytest.mark.parametrize("bad", [
    ((0, 1),),
    ((0, 1), (1, 2), (2, 3), (3, 9)),
    ((0, 1), (1, 3), (1, 2), (3, 4)),
    ((0, 1), (2, 3), (2, 3), (3, 4)),
    ((0, 1), (1, 2), (2, 3), (True, 4)),
    ((0, 1), (1, 2), (2, 3), (3, 3)),
])
def test_invalid_token_offsets_fail_closed(bad):
    """数量、范围、单调和非空白覆盖损坏不能退到字符估算。"""
    class BadCounter(CodepointCounter):
        def offsets(self, text):
            return bad

    with pytest.raises(ValueError, match="^invalid_token_offsets$"):
        build_units("甲乙丙丁", BadCounter(), SemanticConfig(unit_max_tokens=2))


def test_missing_offsets_and_tokenizer_exception_do_not_leak_text():
    """后端异常可能含用户正文，对外仅传播稳定的失败码。"""
    class MissingCounter(CodepointCounter):
        def offsets(self, text):
            raise RuntimeError("sensitive:" + text)

    class FailingCounter(CodepointCounter):
        def count(self, text):
            raise RuntimeError("sensitive:" + text)

    with pytest.raises(ValueError, match="^token_offsets_unavailable$"):
        build_units("甲乙丙丁", MissingCounter(), SemanticConfig(unit_max_tokens=2))
    with pytest.raises(ValueError, match="^token_count_failed$"):
        build_units("甲乙丙丁", FailingCounter(), SemanticConfig())


def test_pure_units_import_does_not_load_embedding_runtime():
    """构造候选单元的导入不应加载 torch、numpy 或 transformers。"""
    subprocess.run([sys.executable, "-c", """
import sys
from tourism_ugc_study.segmentation.semantic_units import build_units
if any(name in sys.modules for name in ('numpy', 'torch', 'transformers')):
    raise RuntimeError('候选单元模块加载了模型依赖')
"""], check=True)
