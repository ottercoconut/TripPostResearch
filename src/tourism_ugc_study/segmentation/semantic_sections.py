"""独立小节题头的确认：把“新小节在题头前切开”落实为可复核的行级检查。

对应编码表分段规范`section-heading-20261018`及适配方案§5.2.3第1条。本模块只读
原文行、调用方识别的题头形态和同形条目序列，不读取向量、模型、帖子身份或编码
标签，也不选点、不改写原文。每个候选题头都返回确认结果与原因码：确认的题头由
分区器作为必切依据，未确认的只记录原因，仍按普通边界规则处理。

检查顺序固定为：像不像标题 → 组成题头组 → 前文完成 → 后文展开 → 非同栏目清单。
“像不像标题”排除规范列明不触发的清单条目、描述性长行和叙述短句：阿拉伯数字、
keycap或圈号编号的条目按清单处理；标题主体不超过12个非空白字符（与本项目短标签
上限一致）；纯文字标题不带叙述句特征，前一行须以句末标点结束；emoji或括号若在
同帖也用作正文行的项目符号，则不是标题标记；标题之前至少已有两行内容（或首行
不少于40个非空白字符），帖子标题行不单独成段；前一行是带标记的标题式短行时本行
是其从属内容；下一行以“也、还、就”等接着说时本行只是半句话。表情代码与纯装饰行
分别按行首标记和分隔线处理。其余阈值是规范文字的直接落实：题头组至多再带2行短行；展开至少
两个完整句（强句末与行末各计一句），到下一同级题头为止——带标记的题头以同类标记
为同级，纯文字题头以任何题头形态的行为同级。这些是一次性固定的识别规则，不按
个案调参；识别依据是版式与句末标点，不做中文句法分析，无法确认时保守地不建立
必切边界。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence
import unicodedata

import regex

from .integrity import WHITE

_STRONG = frozenset("。！？!?；;")
_CLOSERS = frozenset("”’」』)）】]》〉\"'")
_PAIRS = {"(": ")", "（": "）", "[": "]", "【": "】", "{": "}", "「": "」", "『": "』", "“": "”", "‘": "’", "《": "》"}
# 前一行以这些符号结束，说明叙述尚未完成，后面的短行是它的延续而不是新小节。
_CONTINUATION = frozenset("，,、：:")
# 引导行结尾：冒号或向下指示符号，表示后面是它统领的清单。
_LEAD_ENDINGS = ("：", ":", "👇", "↓", "⬇")
# 可以开启独立小节的题头形态；日程题头另由换天规则处理，不在此列。
CANDIDATE_KINDS = frozenset({"explicit_heading", "unmarked_heading", "directory_lead"})
# 可以作为副题／栏目短行并入题头组的形态；目录引导和日程题头本身另起结构。
_GROUP_KINDS = frozenset({"explicit_heading", "unmarked_heading"})
# 日程题头总是另起结构，任何小节的展开范围都到此为止。
_SCHEDULE_KINDS = frozenset({"schedule_heading", "trailing_schedule_heading"})
GROUP_EXTRA_LINES = 2
MIN_EXPANSION_SENTENCES = 2
MAX_TITLE_CHARS = 12
# 带emoji或括号标记的标题常附副标题或说明，主体上限放宽到20个非空白字符。
MAX_MARKED_TITLE_CHARS = 20
# 首行达到这一长度即视为标题已连着正文，可作为小节之前的已完成内容。
MIN_TITLE_LINE_BODY_CHARS = 40
# 小红书正文中的表情代码，如“[三R]”“[打卡R]”“[蹲后续H]”，整体是一个行首标记。
XHS_EMOJI_CODE = re.compile(r"\[[^\[\]\s]{1,8}[RH]\]")
# 下一行以这些副词开头，说明题头行只是被换行截断的前半句。
_FRAGMENT_CONTINUATION = re.compile(r"^(?:也|还|就|都|才|又|便|却|而|并|且|则)")
# 抓取正文中替代媒体的占位行，不是标题也不是句子。
_PLACEHOLDERS = frozenset({"图片", "视频", "图"})
# 前一行以这些符号结束才算纯文字标题之前的叙述已完成（尾部闭符和装饰可忽略）。
_FINAL_MARKS = frozenset("。！？!?…～~")
# 中文数字加顿号是常见的一级小节编号；其他编号（1.、1️⃣、②、（一））多为清单条目。
_CHINESE_SECTION_NUMBER = re.compile(r"^[一二三四五六七八九十百]+、")
# 纯文字短行带这些特征时多是叙述句而非标题：叙述性开头，或判断／体貌／情态／语气成分。
_NARRATIVE_START = re.compile(r"^(?:在|从|当|每|一切|然后|接着|后来|最后|终于|预计|今天|昨天|明天|今年|去年|周[一二三四五六日天末]|"
                              r"这里|那里|这次|那次|刚|就|还|又|也|但|而且|所以|因为|如果|只要|"
                              r"我|你|他|她|它|咱|大家)")
_NARRATIVE_MARK = re.compile(r"是|了|过|着|真的|都|就|也|还|又|会|能|想|吧|呢|吗|啊|呀|哦|啦")
CONFIRMED = "selected_section_heading"
AT_POST_START = "section_heading_at_post_start"


@dataclass(frozen=True)
class SectionLine:
    """一条非空原文行的坐标及其题头形态。

    start/end为去除两端WHITE后的原文码点区间；heading是调用方题头识别给出的
    形态名，None表示普通正文行。本结构不保存正文副本。
    """

    start: int
    end: int
    heading: str | None


@dataclass(frozen=True)
class SectionCandidate:
    """一个题头候选的检查结果；坐标均为原文码点，左闭右开。

    start是题头组起点（必切边界所在位置），group_end是题头组最后一行终点，
    first_end是题头组之后首个完整说明的终点（没有正文时等于group_end），
    expansion_end是展开范围终点，sentences为展开范围内的完整句数。reason为
    CONFIRMED、AT_POST_START（帖首题头，通过其余检查但无切点）或一个
    `section_heading_*`未确认原因；evidence列出可核对的原文区间。
    """

    line_index: int
    start: int
    group_end: int
    first_end: int
    expansion_end: int
    sentences: int
    reason: str
    evidence: tuple[tuple[str, int, int], ...]

    @property
    def confirmed(self) -> bool:
        """是否通过全部检查，可作为独立小节必切依据。"""
        return self.reason == CONFIRMED

    @property
    def protected(self) -> bool:
        """题头组与首说明是否应受禁止切保护：已确认的小节及帖首合格题头。"""
        return self.reason in {CONFIRMED, AT_POST_START}


def first_complete_end(text: str, start: int, end: int) -> int:
    """取区间内首个括号外完整句的终点；没有强句末时返回end。

    闭符和组合符附着在句末，返回的端点不会落入组合字素内部；不按逗号拆，
    极短的完整句也算一句。坐标始终是原文码点。
    """
    stack: list[str] = []
    cursor = start
    while cursor < end:
        char = text[cursor]
        if char in _PAIRS:
            stack.append(_PAIRS[char])
        elif stack and char == stack[-1]:
            stack.pop()
        elif char in _STRONG and not stack:
            cursor += 1
            while cursor < end and (text[cursor] in _CLOSERS or unicodedata.category(text[cursor]).startswith("M")):
                cursor += 1
            return cursor
        cursor += 1
    return end


def metadata_text(value: str) -> bool:
    """整段只有话题标签、账号、链接或装饰，没有任何正文文字或数字。"""
    stripped = re.sub(r"https?://\S+|#[^\s#]+#?|@\S+", "", value)
    return bool(value.strip()) and not any(unicodedata.category(c)[0] in {"L", "N"} for c in stripped)


_LINE_BREAK = re.compile(r"\r\n|[\r\n\u2028\u2029]")


def count_sentences(text: str, start: int, end: int) -> int:
    """统计区间内含文字的完整句数：强句末与行末各结束一句。

    没有句末标点的整行计一句，所以多行无标点清单按行数计，不会被当成一句。
    """
    count = 0
    for line in _LINE_BREAK.split(text[start:end]):
        cursor = 0
        while cursor < len(line):
            while cursor < len(line) and line[cursor] in WHITE:
                cursor += 1
            if cursor >= len(line):
                break
            stop = first_complete_end(line, cursor, len(line))
            if any(unicodedata.category(c)[0] in {"L", "N"} for c in line[cursor:stop]):
                count += 1
            cursor = stop
    return count


_NUMBERED = re.compile(r"^(?:\d+[、．.）)]|[一二三四五六七八九十百]+[、．.]|[（(][一二三四五六七八九十百\d]+[）)]|\d\ufe0f?\u20e3)")
_OPEN_BRACKETS = frozenset("【[〖「『〔")


def marker_class(value: str) -> str:
    """题头行的标记类别：symbol（emoji或符号开头）、bracket、numbered或plain。

    用于判断小节展开到哪里结束：只有同类标记的题头才视为同级小节，形态不同的
    短行（如emoji小节下的纯文字清单项）属于该小节正文。
    """
    body = value.lstrip("".join(WHITE))
    if not body:
        return "plain"
    if _NUMBERED.match(body):
        return "numbered"
    if body[0] in _OPEN_BRACKETS:
        return "bracket"
    return "symbol" if unicodedata.category(body[0]) in {"So", "Sk"} else "plain"


def placeholder_text(value: str) -> bool:
    """整行只是“图片”“视频”等媒体占位。"""
    return value.strip("".join(WHITE)) in _PLACEHOLDERS


def _decoration(char: str) -> bool:
    """标题两端可忽略的装饰：空白、符号、标点、组合符及变体选择符。"""
    return char in WHITE or char == "\ufe0f" or unicodedata.category(char)[0] in {"S", "P", "M"}


def title_core(value: str) -> str:
    """标题主体：去掉行首编号、表情代码与装饰、括号内附注和行尾装饰（含冒号）后的文字。"""
    body = value.strip("".join(WHITE))
    while match := XHS_EMOJI_CODE.match(body):
        body = body[match.end():].lstrip("".join(WHITE))
    if match := (_CHINESE_SECTION_NUMBER.match(body) or _NUMBERED.match(body)):
        body = body[match.end():]
    start = 0
    while start < len(body) and _decoration(body[start]):
        start += 1
    body = re.sub(r"[（(][^()（）]*[）)]", "", body[start:])
    end = len(body)
    while end and _decoration(body[end - 1]):
        end -= 1
    return body[:end]


def _leading_mark(value: str) -> str | None:
    """行首的非文字标记：表情代码整体，或一个emoji、符号、括号字素（去除变体选择符）。

    文字或数字开头返回None。“[三R]”与“[打卡R]”是不同标记，不因同为方括号而混同。
    """
    body = value.lstrip("".join(WHITE))
    if code := XHS_EMOJI_CODE.match(body):
        return code.group()
    match = regex.match(r"\X", body)
    if not match:
        return None
    mark = match.group().replace("\ufe0f", "")
    return None if not mark or unicodedata.category(mark[0])[0] in {"L", "N"} else mark


def separator_text(value: str) -> bool:
    """整行只有分隔或装饰符号（如“-”“---”“·”“~”“❤❤❤”“▪▪▪”），没有任何文字或数字。"""
    stripped = value.strip("".join(WHITE))
    return bool(stripped) and not any(unicodedata.category(c)[0] in {"L", "N"} for c in stripped)


def _ends_final(value: str) -> bool:
    """去掉尾部空白、闭符和装饰后以句末标点结束。"""
    end = len(value)
    while end and (value[end - 1] in WHITE or value[end - 1] in _CLOSERS or value[end - 1] == "\ufe0f"
                   or (unicodedata.category(value[end - 1])[0] in {"S", "M"})):
        end -= 1
    return bool(end) and value[end - 1] in _FINAL_MARKS


def _title_issue(text: str, lines: Sequence[SectionLine], index: int, marker: str,
                 bullets: frozenset[str]) -> str | None:
    """题头行是否不像独立小节标题；像标题时返回None，否则返回未确认原因码。"""
    value = text[lines[index].start:lines[index].end]
    body = value.strip("".join(WHITE))
    if marker == "numbered" and not _CHINESE_SECTION_NUMBER.match(body):
        return "section_heading_list_number"
    core = title_core(value)
    limit = MAX_TITLE_CHARS if marker == "plain" else MAX_MARKED_TITLE_CHARS
    if marker != "numbered" and sum(c not in WHITE for c in core) > limit:
        return "section_heading_too_long"
    if marker == "plain" and (_NARRATIVE_START.match(core) or _NARRATIVE_MARK.search(core)):
        return "section_heading_narrative"
    if marker in {"symbol", "bracket"} and _leading_mark(value) in bullets:
        return "section_heading_bullet_marker"
    # 下一行以同一标记开头，说明本行是并列清单的第一项（如连续的“□ ……”），不是标题。
    if (marker in {"symbol", "bracket"} and index + 1 < len(lines)
            and _leading_mark(text[lines[index + 1].start:lines[index + 1].end]) == _leading_mark(value)):
        return "section_heading_list_sequence"
    # 帖子首行多是标题；标题之前至少要有两行实质内容（分隔线不算），或首行长到
    # 超出一般标题（不少于40个非空白字符，说明首行已连着正文），帖子标题不单独成段。
    before = [text[line.start:line.end] for line in lines[:index] if not separator_text(text[line.start:line.end])]
    if len(before) < 2 and sum(c not in WHITE for value in before for c in value) < MIN_TITLE_LINE_BODY_CHARS:
        return "section_heading_after_title_only"
    previous = text[lines[index - 1].start:lines[index - 1].end]
    if separator_text(previous):
        return None
    # 前一行是带标记、像标题的短行（如“📍石老人→雕塑园”：无句内标点、无句末标点），
    # 本行是它的从属内容；带逗号或冒号的标记行是上一节的清单项，不在此列。
    if (_leading_mark(previous) and not _ends_final(previous)
            and not any(c in previous for c in "，,。！？!?；;：:")
            and sum(c not in WHITE for c in previous) <= 32):
        return "section_heading_previous_incomplete"
    if marker == "plain" and not _ends_final(previous):
        return "section_heading_previous_incomplete"
    following = next((text[line.start:line.end] for line in lines[index + 1:]
                      if not separator_text(text[line.start:line.end])), "")
    if _FRAGMENT_CONTINUATION.match(following.lstrip("".join(WHITE))):
        return "section_heading_sentence_fragment"
    return None


def _label_value(value: str) -> bool:
    """“标签：内容”字段本身就是内容，不作为小节题头或副题。"""
    return bool(re.search(r"[:：]\s*\S", value))


def _ends_with_continuation(value: str) -> bool:
    """去掉尾部空白后以逗号、顿号或冒号结束，说明前文还没说完。"""
    stripped = value.rstrip("".join(WHITE))
    return bool(stripped) and stripped[-1] in _CONTINUATION


def _is_lead(text: str, line: SectionLine) -> bool:
    """清单引导行：上级题头，或以冒号、向下指示符号结束的说明行。"""
    if line.heading is not None:
        return True
    value = text[line.start:line.end].rstrip("".join(WHITE) + "️")
    return value.endswith(_LEAD_ENDINGS)


def _same_level(text: str, line: SectionLine, marker: str) -> bool:
    """该行是否与标记类别为marker的题头同级，从而结束其展开范围或使其前文不完整。

    带标记（emoji、括号、编号）的题头只以同类标记的题头为同级，其下的纯文字
    短行是清单项等正文；纯文字题头没有可区分的标记，任何题头形态的行都视为
    同级或上级。日程题头总是另起结构。
    """
    if line.heading is None:
        return False
    return (line.heading in _SCHEDULE_KINDS or marker == "plain"
            or marker_class(text[line.start:line.end]) == marker)


def _expansion(text: str, lines: Sequence[SectionLine], after: int, marker: str) -> tuple[int, int, int, int]:
    """返回题头组之后的展开范围：(起点, 首说明终点, 范围终点, 完整句数)。

    展开范围从下一行开始，到下一条同级题头、日程题头或帖尾为止；纯元数据行不计
    句数。没有正文时起点与终点都取题头组终点，句数为0。
    """
    group_end = lines[after].end
    body = []
    for line in lines[after + 1:]:
        if _same_level(text, line, marker):
            break
        value = text[line.start:line.end]
        if not (metadata_text(value) or placeholder_text(value)):
            body.append(line)
    if not body:
        return group_end, group_end, group_end, 0
    first = first_complete_end(text, body[0].start, body[0].end)
    return (body[0].start, first, body[-1].end,
            sum(count_sentences(text, line.start, line.end) for line in body))


def _group_last(text: str, lines: Sequence[SectionLine], index: int) -> int:
    """题头组最后一行：题头后至多再带两行无句末标点的副题／栏目短行。"""
    last = index
    while (last + 1 < len(lines) and last - index < GROUP_EXTRA_LINES
           and lines[last + 1].heading in _GROUP_KINDS
           and not _label_value(text[lines[last + 1].start:lines[last + 1].end])):
        last += 1
    return last


def discover_sections(text: str, lines: Sequence[SectionLine],
                      sibling_groups: Sequence[frozenset[int]]) -> tuple[SectionCandidate, ...]:
    """逐个检查题头候选，返回按原文顺序排列的确认／未确认结果。

    lines为全部非空原文行（按顺序）；sibling_groups是同形条目序列，元素为lines
    中的行号集合。帖首题头直接作为首段开头，没有可切的位置；它通过其余检查时
    以AT_POST_START返回，供调用方保护题头组与首说明。题头组内部的行不再单独
    作为候选。本函数不访问token、向量或单元，单元上是否
    存在对应gap、是否被禁止切规则否决，由调用方判断并记录。
    """
    sibling_of = {member: group for group in sibling_groups for member in group}
    # 同帖正文行（非题头形态）行首用过的标记，说明它在本帖是项目符号而非标题标记。
    bullets = frozenset(mark for line in lines if line.heading is None
                        for mark in (_leading_mark(text[line.start:line.end]),) if mark)
    output: list[SectionCandidate] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        value = text[line.start:line.end]
        if (line.heading not in CANDIDATE_KINDS or _label_value(value.rstrip(":："))
                or placeholder_text(value)):
            index += 1
            continue
        last = _group_last(text, lines, index)
        marker = marker_class(value)
        body_start, first_end, expansion_end, sentences = _expansion(text, lines, last, marker)
        evidence = [("section_heading_group", line.start, lines[last].end)]
        if index:
            previous = lines[index - 1]
            evidence.append(("previous_line", previous.start, previous.end))
        if body_start < expansion_end:
            evidence.append(("first_substantive_sentence", body_start, first_end))
            evidence.append(("section_expansion", body_start, expansion_end))
        # 紧接在同类题头之后说明是连续题头（目录或题头组），不是前文已完成；
        # 标记类别不同的短行（如上一节的纯文字清单项）属于上一节正文。
        issue = _title_issue(text, lines, index, marker, bullets) if index else None
        if issue is not None:
            reason = issue
        elif index and (_same_level(text, previous, marker)
                        or _ends_with_continuation(text[previous.start:previous.end])):
            reason = "section_heading_previous_incomplete"
        elif sentences < MIN_EXPANSION_SENTENCES:
            reason = "section_heading_no_expansion"
        elif index in sibling_of and _list_sequence(text, lines, sibling_of[index]):
            reason = "section_heading_list_sequence"
            evidence.extend(("sibling_entry", lines[i].start, lines[i].end) for i in sorted(sibling_of[index]) if i != index)
        else:
            # 帖首题头通过其余检查时只作为首段开头：没有可切位置，但题头组仍随首说明。
            reason = CONFIRMED if index else AT_POST_START
        output.append(SectionCandidate(index, line.start, lines[last].end, first_end, expansion_end,
                                       sentences, reason, tuple(evidence)))
        index = last + 1
    return tuple(output)


def _list_sequence(text: str, lines: Sequence[SectionLine], group: frozenset[int]) -> bool:
    """同形条目序列是否属于同一栏目清单，而非全帖的一级小节。

    序列首项前有引导行（上级题头、冒号或向下指示符号结尾）时属于清单；没有
    引导行时，只要有一项展开不足两个完整句，也按清单处理。
    """
    first = min(group)
    if first > 0 and _is_lead(text, lines[first - 1]):
        return True
    return any(_expansion(text, lines, _group_last(text, lines, member),
                          marker_class(text[lines[member].start:lines[member].end]))[3] < MIN_EXPANSION_SENTENCES
               for member in group)
