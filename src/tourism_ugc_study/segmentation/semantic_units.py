"""从完整可读正文构造语义评分候选单元，保留 Unicode 码点跨度。

本模块只决定可供后续评分的边界，不判断主题、加载模型或改写原文。句末与
换行产生的短单元不会被预先装箱；长单元仅依赖注入的真实 tokenizer 计数
细分。每个输出都保护完整字素、模型输入上限和全部非空白字符的覆盖。
任何无法可靠表示的输入抛出仅含 reason code 的 ValueError，不返回部分结果。
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
import re
from typing import TYPE_CHECKING, Protocol
import unicodedata

import regex

from .contracts import EmbeddingError, TextUnit, UnitizationResult
from .integrity import WHITE

if TYPE_CHECKING:
    from .semantic_config import SemanticConfig


class UnitTokenCounter(Protocol):
    """单元构造所需的最小计数协议；实现不能静默截断或用字符数估算。

    count/offsets 使用同一份无特殊 token 的正文编码；input_count 包括完整
    模型输入模板与特殊 token。offsets 是原文字串的码点区间，允许多个 token
    对应同一字素或码点，但不得遗漏非空白文字。上限是本轮配置值。
    """

    max_input_tokens: int

    def count(self, text: str) -> int:
        """返回完整正文的实际 token 数。"""
        ...

    def input_count(self, text: str) -> int:
        """返回带模板的实际输入 token 数。"""
        ...

    def offsets(self, text: str) -> tuple[tuple[int, int], ...]:
        """返回与 count 同序同数量的正文 token 原文区间。"""
        ...


_STRONG = frozenset("。！？；!?;")
_CLOSERS = frozenset("”’」』)）】]》〉\"'")
_WEAK = frozenset("，,、：:")
_LINES = frozenset(("\r\n", "\r", "\n", "\u2028", "\u2029"))
_URL = re.compile(r'''https?://[^\s<>"'“”‘’「」『』。！？；]+''', re.IGNORECASE)


def _checked_count(counter: UnitTokenCounter, value: str, *, model_input: bool = False) -> int:
    """验证外部计数的基本契约，且不在异常文本中带出原文。"""
    try:
        count = counter.input_count(value) if model_input else counter.count(value)
    except (MemoryError, OSError, EmbeddingError):
        raise
    except Exception:
        raise ValueError("token_count_failed") from None
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("invalid_token_count")
    return count


def _trim(text: str, start: int, end: int, boundaries: tuple[int, ...]) -> tuple[int, int]:
    """仅移除整簇全部属于 WHITE 的字素；空格加重音等组合完整保留。"""
    left = bisect_left(boundaries, start)
    right = bisect_left(boundaries, end)
    while left < right and all(c in WHITE for c in text[boundaries[left]:boundaries[left + 1]]):
        left += 1
    while right > left and all(c in WHITE for c in text[boundaries[right - 1]:boundaries[right]]):
        right -= 1
    return boundaries[left], boundaries[right]


def _initial_spans(text: str, boundaries: tuple[int, ...]) -> list[tuple[int, int]]:
    """扫描字素定位句末和换行，URL 内 ASCII 句末符号不形成边界。"""
    urls = [match.span() for match in _URL.finditer(text)]
    url_cursor = 0
    strong: set[int] = set()
    for index, start in enumerate(boundaries[:-1]):
        while url_cursor < len(urls) and urls[url_cursor][1] <= start:
            url_cursor += 1
        inside_url = url_cursor < len(urls) and urls[url_cursor][0] <= start < urls[url_cursor][1]
        if text[start] in _STRONG and not inside_url:
            strong.add(index)

    spans: list[tuple[int, int]] = []
    cursor = 0
    index = 0
    while index < len(boundaries) - 1:
        cluster = text[boundaries[index]:boundaries[index + 1]]
        if cluster in _LINES:
            cut = boundaries[index + 1]
        elif index in strong:
            # 标点附加的 variation selector／组合符已包含在当前字素中。
            while index + 1 in strong:
                index += 1
            while index + 1 < len(boundaries) - 1 and text[boundaries[index + 1]] in _CLOSERS:
                index += 1
            cut = boundaries[index + 1]
        else:
            index += 1
            continue
        span = _trim(text, cursor, cut, boundaries)
        if span[0] < span[1]:
            spans.append(span)
        cursor = cut
        index += 1
    final = _trim(text, cursor, len(text), boundaries)
    if final[0] < final[1]:
        spans.append(final)
    return spans


def _join_punctuation(text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """纯 P 类候选优先归前邻，无前邻归后邻；内部空白随连续 span 保留。"""
    joined = list(spans)
    index = 0
    while index < len(joined):
        start, end = joined[index]
        punctuation = all(c in WHITE or unicodedata.category(c).startswith("P") for c in text[start:end])
        if not punctuation or len(joined) == 1:
            index += 1
        elif index:
            joined[index - 1] = joined[index - 1][0], end
            del joined[index]
        else:
            joined[1] = start, joined[1][1]
            del joined[0]
    return joined


def _checked_offsets(counter: UnitTokenCounter, value: str, token_count: int) -> tuple[tuple[int, int], ...]:
    """拒绝不完整、反向或非码点 offset；允许字节 BPE 产生的重叠区间。"""
    try:
        offsets = counter.offsets(value)
    except (MemoryError, OSError, EmbeddingError):
        raise
    except Exception:
        raise ValueError("token_offsets_unavailable") from None
    if not isinstance(offsets, (tuple, list)) or len(offsets) != token_count:
        raise ValueError("invalid_token_offsets")
    previous_start = previous_end = covered_end = 0
    clean: list[tuple[int, int]] = []
    for item in offsets:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("invalid_token_offsets")
        start, end = item
        if any(isinstance(v, bool) or not isinstance(v, int) for v in item):
            raise ValueError("invalid_token_offsets")
        if not (0 <= start < end <= len(value)) or start < previous_start or end < previous_end:
            raise ValueError("invalid_token_offsets")
        if start > covered_end and any(c not in WHITE for c in value[covered_end:start]):
            raise ValueError("invalid_token_offsets")
        previous_start, previous_end, covered_end = start, end, max(end, covered_end)
        clean.append((start, end))
    if any(c not in WHITE for c in value[covered_end:]):
        raise ValueError("invalid_token_offsets")
    return tuple(clean)


def _unit(
    text: str, start: int, end: int, count: int, counter: UnitTokenCounter,
    config: SemanticConfig, diagnostics: list[str], *, single_grapheme: bool,
) -> TextUnit:
    """发布单个已定位单元前重检输入；单字素例外不得越过硬性限制。"""
    input_count = _checked_count(counter, text[start:end], model_input=True)
    if input_count > counter.max_input_tokens:
        raise ValueError("unrepresentable_grapheme" if single_grapheme else "model_input_limit_exceeded")
    if count > config.max_segment_tokens:
        raise ValueError("unsplittable_segment_limit")
    if count > config.unit_max_tokens:
        if not single_grapheme:
            raise ValueError("unit_limit_exceeded")
        diagnostics.append("unit_oversized_grapheme")
    return TextUnit(start=start, end=end, token_count=count)


def _split_span(
    text: str, span: tuple[int, int], boundaries: tuple[int, ...], counter: UnitTokenCounter,
    config: SemanticConfig, diagnostics: list[str],
) -> list[TextUnit]:
    """按弱边界、token offset、逐字素回退的顺序拆长单元，cursor 严格前进。"""
    cursor, end = span
    units: list[TextUnit] = []
    while cursor < end:
        cursor, end = _trim(text, cursor, end, boundaries)
        if cursor == end:
            break
        first_index = bisect_left(boundaries, cursor)
        first_end = boundaries[first_index + 1]
        count = _checked_count(counter, text[cursor:end])
        if count <= config.unit_max_tokens or first_end == end:
            units.append(_unit(text, cursor, end, count, counter, config, diagnostics, single_grapheme=first_end == end))
            break

        cut = None
        cut_count = None
        # 从后向前逐一实测，找到的第一个便是最靠后可行点。不能因某一前缀
        # 超限就停止向后搜索：BPE 的前缀 token 数没有严格单调的保证。
        for boundary_index in range(bisect_left(boundaries, end) - 1, first_index, -1):
            candidate = boundaries[boundary_index]
            cluster = text[boundaries[boundary_index - 1]:candidate]
            if cluster[0] not in _WEAK and not all(c in WHITE for c in cluster):
                continue
            _, trimmed_end = _trim(text, cursor, candidate, boundaries)
            if trimmed_end == cursor:
                continue
            candidate_count = _checked_count(counter, text[cursor:trimmed_end])
            if candidate_count <= config.unit_max_tokens:
                cut, cut_count = trimmed_end, candidate_count
                break

        if cut is None:
            offsets = _checked_offsets(counter, text[cursor:end], count)
            proposed_end = cursor + offsets[min(config.unit_max_tokens, len(offsets)) - 1][1]
            # 一个 tokenizer token 可以跨多个码点；反之，一个字素也可跨
            # 多个 token。只把 offset 当作切点建议，最终回切原文并真实重计。
            candidate_index = bisect_right(boundaries, proposed_end) - 1
            candidate_index = min(candidate_index, bisect_left(boundaries, end) - 1)
            while candidate_index > first_index:
                candidate = boundaries[candidate_index]
                _, candidate = _trim(text, cursor, candidate, boundaries)
                if candidate == cursor:
                    candidate_index -= 1
                    continue
                candidate_count = _checked_count(counter, text[cursor:candidate])
                if candidate_count <= config.unit_max_tokens:
                    cut, cut_count = candidate, candidate_count
                    break
                candidate_index -= 1
            if cut is None:
                cut = first_end
                cut_count = _checked_count(counter, text[cursor:cut])

        units.append(_unit(text, cursor, cut, cut_count, counter, config, diagnostics, single_grapheme=cut == first_end))
        cursor = cut
    return units


def build_units(text: str, counter: UnitTokenCounter, config: SemanticConfig) -> UnitizationResult:
    """构造有界、字素安全且保持原文的候选单元，不作语义合并。

    输入必须为已投影的可读字符串；空白返回空 units，NULL 应由投影层处理。
    counter 必须提供实际正文／完整输入计数，以及必要时的原文 offset。
    输出跨度以 Unicode 码点表达，按原文顺序且不重叠，全部非 WHITE 字符
    恰好覆盖一次。单字素超出单元预算但符合两项硬限制时记录例外诊断；
    计数、offset 或表示约束失败均抛出无原文的 ValueError reason code。
    """
    if not isinstance(text, str):
        raise ValueError("invalid_source_text")
    if isinstance(counter.max_input_tokens, bool) or not isinstance(counter.max_input_tokens, int) or counter.max_input_tokens <= 0:
        raise ValueError("invalid_model_input_limit")
    boundaries = (0, *(match.end() for match in regex.finditer(r"\X", text)))
    diagnostics: list[str] = []
    units: list[TextUnit] = []
    for span in _join_punctuation(text, _initial_spans(text, boundaries)):
        units.extend(_split_span(text, span, boundaries, counter, config, diagnostics))

    # 核心路径自身验证覆盖，而不依赖调用方或可被 -O 关闭的 assert。
    previous_end = 0
    valid_boundaries = set(boundaries)
    for unit in units:
        if not (previous_end <= unit.start < unit.end <= len(text)) or unit.start not in valid_boundaries or unit.end not in valid_boundaries:
            raise ValueError("unit_integrity_failed")
        if any(c not in WHITE for c in text[previous_end:unit.start]):
            raise ValueError("unit_integrity_failed")
        previous_end = unit.end
    if any(c not in WHITE for c in text[previous_end:]):
        raise ValueError("unit_integrity_failed")
    return UnitizationResult(units=tuple(units), diagnostics=tuple(dict.fromkeys(diagnostics)))
