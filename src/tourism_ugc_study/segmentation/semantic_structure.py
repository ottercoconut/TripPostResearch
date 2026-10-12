"""为既有候选 gap 提取可消融的正文结构证据，不决定最终切点。

本模块是独立开发候选，不接入现行语义规则的默认路径。输入仍为原文与
候选单元；输出与相邻单元 gap 一一对应，不改写原文、不拆合单元、不访问
帖子 ID、模型或数据库。罚分只表达结构风险的经验优先级，不是概率或金标。
下游可比较 sentence_only 与 structure，或按 reason 单独消融；硬长度不可
满足时须由下游显式记录退让，不能把本模块的软证据当作无限制禁止切分。
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import re
from typing import Literal, Sequence
import unicodedata

import regex

from .contracts import TextUnit
from .integrity import WHITE

StructureMode = Literal["sentence_only", "structure"]

_STRONG = frozenset("。！？；!?;")
_CLOSERS = frozenset("”’」』)）】]》〉\"'")
_PAIRS = {"(": ")", "（": "）", "[": "]", "【": "】", "{": "}",
          "「": "」", "『": "』", "“": "”", "‘": "’", "《": "》", "〈": "〉"}
_OPENERS = {closing: opening for opening, closing in _PAIRS.items()}
_LINES = re.compile(r"\r\n|[\r\n\u2028\u2029]")
_ATTRIBUTE = re.compile(
    r"^(?:交通|地址|位置|导航|时间|时长|开放时间|营业时间|门票|票价|价格|费用|"
    r"预约|停车|电话|优点|缺点|亮点)\s*[:：|｜]"
)
_DAY = re.compile(r"^(?:day\s*\d+|第[0-9一二三四五六七八九十百]+天)(?:\s*[:：、.．-]?\s*[^，,。！？!?；;]*)$", re.I)
_DATE = re.compile(r"^(?:\d{2,4}年)?\d{1,2}月\d{1,2}日(?:\s|[，,]|$)")
_NUMBERED = re.compile(r"^(?:[一二三四五六七八九十百]+[、．.]|\d+[、．.]|[（(][一二三四五六七八九十百\d]+[）)])\s*\S")
_ROUTE_TITLE = re.compile(r"^(?:\d+|[一二三四五六七八九十]+)\s*[天日].{0,8}(?:路线|行程)")


@dataclass(frozen=True)
class GapStructure:
    """一个既有 gap 的结构证据及可重放经验罚分。

    index 是零基 gap 序号；boundary_kind 为 line、sentence 或 internal。
    natural_boundary 仅表示句／行自然端点，不能证明语义独立。reasons
    列出观察到的证据；evidence_spans 是 (reason, start, end) 码点区间，
    只定位来源、不复制正文。before_heading 是偏好信号，不自行减为负罚分。
    """

    index: int
    boundary_kind: str
    natural_boundary: bool
    penalty: int
    reasons: tuple[str, ...]
    evidence_spans: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class StructureBlock:
    """原文中一个有结构依据的连续区间；本结构不声明必须合并。

    start/end 为原文码点左闭右开区间。repeated_entry 的 evidence 包含
    heading、first_substantive_sentence、repeated_marker、matching_marker；
    后者可以位于另一条目中。heading_context 包含题头链和首实质句，
    trailing_metadata_context 包含前一实质句和尾元数据，syntactic_continuation
    包含状语助词、异常标点和后继分句。调用方须精确
    计数整个块并显式处理超限；发现结构不能绕过 token 上限。对象及证据
    均不可变、不存正文副本，各发现函数的规则保持独立以便消融。
    """

    start: int
    end: int
    kind: str
    evidence: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class _StructureLine:
    """跳过行两端 WHITE 后的原文跨度；空行仍保留其顺序。"""

    start: int
    end: int


@dataclass(frozen=True)
class _EntryHeading:
    """仅供重复版式匹配的候选题头，不将单次标记自动视为条目。"""

    line_index: int
    family: str
    marker_start: int
    marker_end: int


def _structure_lines(text: str) -> tuple[_StructureLine, ...]:
    """按真实换行建立坐标，保留 CRLF 和 Unicode 行分隔符语义。"""
    lines: list[_StructureLine] = []
    start = 0
    for end, next_start in [*((m.start(), m.end()) for m in _LINES.finditer(text)),
                            (len(text), len(text))]:
        left, right = start, end
        while left < right and text[left] in WHITE:
            left += 1
        while right > left and text[right - 1] in WHITE:
            right -= 1
        lines.append(_StructureLine(left, right))
        start = next_start
    return tuple(lines)


def _metadata_line(value: str) -> bool:
    """独立话题标签／链接／账号行终止条目，不据其内容猜测章节主题。"""
    return bool(re.fullmatch(r"(?:#[^\s#]+#?\s*)+", value)
                or re.fullmatch(r"https?://\S+", value, re.I)
                or re.fullmatch(r"@\S+", value))


def _title_fragment(value: str) -> bool:
    """保守排除完整叙述和属性字段，不用地名或旅游对象词表识别名词。

    重复符号本身不能区分题头与评价。这里另要求短、无完整句标点，
    并排除明显的代词／时点起句、程度副词及体标记；这是一般句法线索，
    不是完整中文词性分析。模糊的名词式短句仍须靠开发复核判断。
    """
    value = value.strip()
    if (not value or sum(c not in WHITE for c in value) > 32
            or not any(unicodedata.category(c)[0] in {"L", "N"} for c in value)
            or _attribute_line(value)):
        return False
    # “题头：短标签！”可作为短题头；其余感叹句不以标记重复为充分条件。
    label = re.fullmatch(r"([^：:，,。！？!?；;]+)[：:][^：:，,。！？!?；;]{1,8}[！!]", value)
    plain = label.group(1) if label else value.rstrip(":：")
    # 题头可以附带作者的括号式取舍；检查其主体，而不把附注中的“觉得”
    # 当作整行是叙述句的证据。只跳过真实闭合区间，未闭括号不能吃掉正文。
    pairs = _matched_brackets(plain)
    plain = "".join(char for index, char in enumerate(plain)
                    if not any(start <= index < end for start, end in pairs)).strip()
    if not plain or not any(unicodedata.category(c)[0] in {"L", "N"} for c in plain):
        return False
    if any(c in "，,。！？!?；;" for c in plain) or (not label and any(c in "！!" for c in value)):
        return False
    if re.match(r"^(?:我(?:们)?|你(?:们)?|他(?:们)?|她(?:们)?|它(?:们)?|大家|今天|昨天|明天|这次|刚刚)", plain):
        return False
    if re.search(r"(?:很|非常|十分|太|真是|挺|已经|正在|觉得|喜欢|值得)|[了着](?:$|[的呢呀吧])", plain):
        return False
    if re.match(r"^(?:可以|需要|应该|建议|能够|必须|应当|不必|不能|无需|不需要)", plain):
        return False
    return True


def _entry_heading(value: str, line_index: int, offset: int) -> _EntryHeading | None:
    """提取一般编号、任意 emoji 标记或短冒号题头的重复身份。

    相同 emoji 的展示变体归为同族；数字内容不影响编号族。单独普通
    emoji 短句不经本函数直接成为保护块，还须同帖重复和后续说明成立。
    """
    if not value or _metadata_line(value):
        return None
    numbered = re.match(r"(?:\d+[、．.）)]|[一二三四五六七八九十百]+[、．.]|[（(][一二三四五六七八九十百\d]+[）)])\s*", value)
    if numbered and _title_fragment(value[numbered.end():]):
        return _EntryHeading(line_index, "numbered", offset, offset + numbered.end())
    cluster = regex.match(r"\X", value)
    if cluster:
        marker = cluster.group()
        keycap = bool(regex.fullmatch(r"[0-9#*]\ufe0f?\u20e3", marker))
        enclosed_number = unicodedata.category(marker[0]) == "No"
        emoji = bool(regex.match(r"(?:\p{Extended_Pictographic}|\p{Emoji_Presentation}|\p{Regional_Indicator})", marker))
        if (keycap or enclosed_number or emoji) and _title_fragment(value[cluster.end():]):
            family = "numbered" if keycap or enclosed_number else "emoji:" + marker.replace("\ufe0f", "")
            return _EntryHeading(line_index, family, offset, offset + cluster.end())
    if value.endswith((":", "：")) and _title_fragment(value):
        return _EntryHeading(line_index, "colon_title", offset + len(value) - 1, offset + len(value))
    return None


def _first_sentence_span(text: str, line: _StructureLine) -> tuple[int, int]:
    """首实质行中取首个括号外句末；无句末则保留该行，不按 token 细分。

    配对括号内的句号不截断这份结构证据。返回的端点不新增模型单元或
    输出切点，消费方只用区间判断既有 gap 是否破坏题头的首段说明。
    """
    value = text[line.start:line.end]
    pairs = _matched_brackets(value)
    for index, char in enumerate(value):
        if char not in _STRONG or any(start < index < end for start, end in pairs):
            continue
        end = index + 1
        while end < len(value) and (value[end] in _CLOSERS
                                   or unicodedata.category(value[end]).startswith("M")):
            end += 1
        return line.start, line.start + end
    return line.start, line.end


def discover_structure_blocks(text: str, units: Sequence[TextUnit]) -> tuple[StructureBlock, ...]:
    """发现同帖至少两次出现、且各有后续说明的重复题头条目。

    题头按完整原文行识别，不把 token 细分片段当完整题头。条目延伸至
    下一重复族题头、短冒号章节或独立元数据行之前；单次 emoji 短行没有
    足够证据终止上层条目。文末条目也在独立 hashtag／URL／账号行处停止。
    只有题头或连续列表没有实质说明时不输出保护块。
    不推断地名、标签、章节层级或跨帖模式，不读取模型／tokenizer。
    输入非法时与 gap 分析使用相同失败语义，空白正文返回空元组。
    """
    _validate_spans(text, units)
    lines = _structure_lines(text)
    headings = {index: heading for index, line in enumerate(lines)
                if (heading := _entry_heading(text[line.start:line.end], index, line.start)) is not None}
    observed_families: dict[str, int] = {}
    for heading in headings.values():
        observed_families[heading.family] = observed_families.get(heading.family, 0) + 1
    # 发现阶段先要求版式至少出现两次；稍后仍要求其中至少两条分别有
    # 实质说明。短冒号章节只作为明确终止信号，单次出现不会产生保护块。
    terminators = {index for index, heading in headings.items()
                   if observed_families[heading.family] >= 2 or heading.family == "colon_title"}
    eligible: dict[int, tuple[_EntryHeading, int, tuple[int, int]]] = {}
    for index, heading in headings.items():
        end_index = len(lines)
        first_sentence: tuple[int, int] | None = None
        last_content = index
        for cursor in range(index + 1, len(lines)):
            line = lines[cursor]
            value = text[line.start:line.end]
            if cursor in terminators or _metadata_line(value):
                end_index = cursor
                break
            if value:
                last_content = cursor
                if first_sentence is None and any(unicodedata.category(c)[0] in {"L", "N"} for c in value):
                    first_sentence = _first_sentence_span(text, line)
        if first_sentence is not None and last_content < end_index:
            eligible[index] = heading, lines[last_content].end, first_sentence
    families: dict[str, list[int]] = {}
    for index, (heading, _, _) in eligible.items():
        families.setdefault(heading.family, []).append(index)
    output: list[StructureBlock] = []
    for index, (heading, end, first_sentence) in eligible.items():
        siblings = families[heading.family]
        if len(siblings) < 2:
            continue
        other = eligible[next(i for i in siblings if i != index)][0]
        line = lines[index]
        output.append(StructureBlock(
            line.start, end, "repeated_entry",
            (("heading", line.start, line.end),
             ("first_substantive_sentence", *first_sentence),
             ("repeated_marker", heading.marker_start, heading.marker_end),
             ("matching_marker", other.marker_start, other.marker_end)),
        ))
    return tuple(output)


def _book_title(value: str) -> bool:
    """独立书名号本身提供作品题头证据，不按书名内容维护白名单。"""
    return bool(re.fullmatch(r"《[^《》\r\n]+》", value)
                and any(unicodedata.category(c)[0] in {"L", "N"} for c in value))


def _work_heading_end(value: str) -> int | None:
    """识别行首作品题名及显式书目信息，返回题头末端而非整行末端。

    只允许固定阅读／观看动作前缀、书名号及括号信息或带署名角色的短串。
    普通“我读……”和作品名后的自由叙述不据此成为题头；无任何书目
    结构时，只接受单独题名（可带句号）。每项书目信息限32码点，不依赖
    作者、作品或体裁名称名单。同一行的首解释留给调用方继续识别。
    """
    title = re.match(r"(?:阅读|观看|读|观)?《(?P<title>[^《》\r\n]{1,32})》", value)
    if title is None or not any(unicodedata.category(c)[0] in {"L", "N"} for c in title.group('title')):
        return None
    pairs = {start: end for start, end in _matched_brackets(value)}
    cursor, metadata = title.end(), False
    while cursor < len(value):
        while cursor < len(value) and value[cursor] in WHITE:
            cursor += 1
        if cursor >= len(value):
            break
        end = pairs.get(cursor)
        if value[cursor] in "([（【" and end is not None:
            inside = value[cursor + 1:end - 1]
            if _title_fragment(inside):
                metadata = True
                cursor = end
                continue
        author = re.match(r"[^\[\]()（）【】《》，,。！？!?；;:：]{1,32}?(?:编著|编译|著|译)", value[cursor:])
        if author and not re.match(r"^(?:我|你|他|她|它|今天|觉得|可以|需要)", author.group()):
            metadata = True
            cursor += author.end()
            continue
        if metadata and value[cursor] in "，,、:：":
            cursor += 1
            continue
        if value[cursor] == "。":
            return cursor + 1
        break
    if cursor == len(value) or metadata:
        while cursor > title.end() and value[cursor - 1] in WHITE:
            cursor -= 1
        return cursor
    return None


def _context_lines(text: str) -> tuple[_StructureLine, ...]:
    """仅对结构识别拆开同一行的书目题头与说明，不改变原文或模型单元。"""
    output: list[_StructureLine] = []
    for line in _structure_lines(text):
        value = text[line.start:line.end]
        end = _work_heading_end(value)
        if end is None or end == len(value):
            output.append(line)
            continue
        output.append(_StructureLine(line.start, line.start + end))
        start = line.start + end
        while start < line.end and text[start] in WHITE:
            start += 1
        if start < line.end:
            output.append(_StructureLine(start, line.end))
    return tuple(output)


def _context_lead_span(text: str, start: int, end: int) -> tuple[int, int]:
    """题头后累计完整句到至少12个非空白码点，避免只附着极短提示标签。

    调用方先限定同一说明区间及足够长的后继材料；这里不越过区间另找
    内容，不按字符阈值截断句子，也不改旧重复条目的首句识别规则。
    """
    cursor = start
    while cursor < end:
        _, stop = _first_sentence_span(text, _StructureLine(cursor, end))
        if sum(char not in WHITE for char in text[start:stop]) >= 12 or stop == end:
            return start, stop
        cursor = stop
        while cursor < end and text[cursor] in WHITE:
            cursor += 1
    return start, end


def _syntactic_continuation_blocks(text: str, units: Sequence[TextUnit]) -> tuple[StructureBlock, ...]:
    """对“状语地＋异常强标点＋同一行分句”保留局部连续结构证据。

    只看既有gap；前置短语须在逗号等分句端点后、以2—8个汉字加“地”
    收尾，右侧须直接继续汉字正文。含“的”的短串（如“目的地”）、营地
    复合词、单字词干、跨行和普通“真的！”均不触发。该窄语法启发式
    不承担通用词性消歧，不把所有地字或指代关系变为禁止切点。
    """
    output: list[StructureBlock] = []
    for left, right in zip(units, units[1:]):
        if _LINES.search(text[left.end:right.start]) or not regex.match(r"\p{Han}", text[right.start:right.end]):
            continue
        # 短短语只需局部后缀，不能为每个gap复制、扫描全部前文。
        offset = max(0, left.end - 32)
        prefix = text[offset:left.end]
        match = regex.search(r"(?:^|[，,。！？!?；;：:\r\n\u2028\u2029])(?P<phrase>[\p{Han}]{2,8}地)(?P<punct>[。！？!?；;]+)$", prefix)
        if (match is None or "的" in match.group("phrase") or match.group("phrase").endswith("营地")
                or (offset and match.start("phrase") == 0)):
            continue
        line_end = _LINES.search(text, right.start)
        end = line_end.start() if line_end else len(text)
        _, end = _first_sentence_span(text, _StructureLine(right.start, end))
        start = offset + match.start("phrase")
        output.append(StructureBlock(
            start, end, "syntactic_continuation",
            (("adverbial_particle", offset + match.end("phrase") - 1, offset + match.end("phrase")),
             ("intervening_punctuation", offset + match.start("punct"), offset + match.end("punct")),
             ("following_clause", right.start, end)),
        ))
    return tuple(output)


def _context_heading(value: str) -> tuple[bool, bool]:
    """返回是否为短题头候选、是否有显式题头符号。

    无标记短行只有接上更长说明后才构成证据。日期沿用既有题头规则，
    可含日期后的逗号；独立《作品名》不能因配对符移除后为空而漏掉。
    返回值只是候选身份，不保护两条相邻且长度相似的普通短叙述。
    """
    value = value.strip()
    if not value:
        return False, False
    if _work_heading_end(value) == len(value):
        return True, True
    if (sum(c not in WHITE for c in value) > 32 or _attribute_line(value)
            or _metadata_line(value) or _tail_metadata_line(value)):
        return False, False
    plain = _plain_label(value)
    explicit = bool(_book_title(value) or _DAY.fullmatch(plain) or _DATE.match(plain))
    if explicit:
        return True, True
    if _entry_heading(value, 0, 0) is not None:
        return True, True
    if any(c in "，,。！？!?；;" for c in value):
        return False, False
    return _title_fragment(value), False


def _tail_metadata_line(value: str) -> bool:
    """整行只能由标签和账号组成，允许混排；正文提到标签不命中。"""
    return bool(re.fullmatch(r"(?:(?:#[^\s#@]+#?|@[^\s#@]+)\s*)+", value))


def _last_substantive_sentence(text: str, line: _StructureLine) -> tuple[int, int]:
    """取得实质行中最后一句的跨度，不借 token 细分截掉句首。

    配对括号内标点不结束外层句子；尾部只有闭符／emoji 时沿用前句。
    行末没有强句末符号时，以该行剩余的实质表达作为最后一句。
    """
    value = text[line.start:line.end]
    pairs = _matched_brackets(value)
    spans: list[tuple[int, int]] = []
    start, cursor = 0, 0
    while cursor < len(value):
        if value[cursor] not in _STRONG or any(a < cursor < b for a, b in pairs):
            cursor += 1
            continue
        end = cursor + 1
        while end < len(value) and (value[end] in _STRONG or value[end] in _CLOSERS
                                   or unicodedata.category(value[end]).startswith("M")):
            end += 1
        if any(unicodedata.category(c)[0] in {"L", "N"} for c in value[start:end]):
            spans.append((start, end))
        start = cursor = end
    if any(unicodedata.category(c)[0] in {"L", "N"} for c in value[start:]):
        spans.append((start, len(value)))
    # 调用方只传入含 L/N 的实质行，故至少存在一段；不用静默的空跨度兜底。
    if not spans:
        raise ValueError("missing_substantive_metadata_context")
    start, end = spans[-1]
    while start < end and value[start] in WHITE:
        start += 1
    return line.start + start, line.start + end


def discover_context_blocks(text: str, units: Sequence[TextUnit]) -> tuple[StructureBlock, ...]:
    """独立发现短题头链及尾元数据依赖，不改变既有 gap 或条目默认规则。

    普通题头最多32个非空白码点；显式作品书目允许逐项短元数据。连续题头链
    须后继至少24码点的实质说明，且说明长于每个无标记题头。首说明累计完整
    句至至少12码点，避免只护极短标签；不越过新题头拼接，也不自动合并整篇。
    尾部连续标签／账号行另绑定前一实质行的最后一句，跨模型单元细分保持
    原句起点，末端保留至原文末尾。两类块可以相交，由调用方统一施加精确
    token 限制并记录退让。输入校验同 discover_structure_blocks，不计 token、
    另给异常强标点两侧的短状语连续结构提供局部证据。不改原文、不按普通
    指代词建立依赖，也不把输出视为已验收的语义金标。
    """
    _validate_spans(text, units)
    lines = _context_lines(text)
    nonempty = [index for index, line in enumerate(lines) if line.start < line.end]
    headings = {index: _context_heading(text[lines[index].start:lines[index].end]) for index in nonempty}
    output: list[StructureBlock] = []
    position = 0
    while position < len(nonempty):
        index = nonempty[position]
        if not headings[index][0]:
            position += 1
            continue
        chain = [index]
        cursor = position + 1
        body_index: int | None = None
        lead_start: int | None = None
        while cursor < len(nonempty):
            candidate = nonempty[cursor]
            line = lines[candidate]
            value = text[line.start:line.end]
            if _metadata_line(value) or _tail_metadata_line(value):
                break
            size = sum(c not in WHITE for c in value)
            unmarked_sizes = [sum(c not in WHITE for c in text[lines[i].start:lines[i].end])
                              for i in chain if not headings[i][1]]
            if (size >= 24 and not headings[candidate][1]
                    and any(unicodedata.category(c)[0] in {"L", "N"} for c in value)
                    and all(size > prior_size for prior_size in unmarked_sizes)):
                body_index = candidate
                break
            if (not headings[candidate][0] and size < 12 and any(c in _STRONG for c in value)
                    and any(unicodedata.category(c)[0] in {"L", "N"} for c in value)):
                if lead_start is None:
                    lead_start = line.start
                cursor += 1
                continue
            if lead_start is not None:
                break
            if not headings[candidate][0]:
                break
            chain.append(candidate)
            cursor += 1
        if body_index is not None:
            first_sentence = _context_lead_span(
                text, lead_start if lead_start is not None else lines[body_index].start, lines[body_index].end,
            )
            output.append(StructureBlock(
                lines[chain[0]].start, first_sentence[1], "heading_context",
                (*( ("heading", lines[i].start, lines[i].end) for i in chain),
                 ("first_substantive_sentence", *first_sentence)),
            ))
            position = cursor + 1
        else:
            position += 1

    tail_start: int | None = None
    before_tail = len(nonempty) - 1
    while before_tail >= 0:
        line = lines[nonempty[before_tail]]
        if not _tail_metadata_line(text[line.start:line.end]):
            break
        tail_start = line.start
        before_tail -= 1
    if tail_start is not None:
        while before_tail >= 0:
            line = lines[nonempty[before_tail]]
            if any(unicodedata.category(c)[0] in {"L", "N"} for c in text[line.start:line.end]):
                prior = _last_substantive_sentence(text, line)
                output.append(StructureBlock(
                    prior[0], len(text), "trailing_metadata_context",
                    (("preceding_substantive_sentence", *prior),
                     ("trailing_metadata", tail_start, len(text))),
                ))
                break
            before_tail -= 1
    output.extend(_syntactic_continuation_blocks(text, units))
    return tuple(sorted(output, key=lambda block: (block.start, block.end, block.kind)))


def _validate_spans(text: str, units: Sequence[TextUnit]) -> None:
    """拒绝越界、重叠、漏字及拆坏字素的单元；此处不重新核验 token 数。"""
    if not isinstance(text, str):
        raise ValueError("invalid_source_text")
    boundaries = {0, *(match.end() for match in regex.finditer(r"\X", text))}
    previous = 0
    for unit in units:
        if (not isinstance(unit, TextUnit) or type(unit.start) is not int or type(unit.end) is not int
                or not previous <= unit.start < unit.end <= len(text)
                or unit.start not in boundaries or unit.end not in boundaries):
            raise ValueError("invalid_structure_unit_span")
        if any(char not in WHITE for char in text[previous:unit.start]):
            raise ValueError("incomplete_structure_unit_coverage")
        if all(char in WHITE for char in text[unit.start:unit.end]):
            raise ValueError("empty_structure_unit")
        previous = unit.end
    if any(char not in WHITE for char in text[previous:]):
        raise ValueError("incomplete_structure_unit_coverage")


def _matched_brackets(text: str) -> tuple[tuple[int, int], ...]:
    """仅返回实际成对的括号／中文引号区间，不把未闭开符扩展到文末。

    遇到错嵌闭符时，只闭合栈内最近的同类开符，并丢弃中间未配对开符。
    英文单双引号可能是缩写或撇号，故不作配对。跨行配对仍是软证据，
    不能以此取消硬段长；完全位于长引用内的正文仍允许下游必要切分。
    """
    stack: list[tuple[str, int]] = []
    spans: list[tuple[int, int]] = []
    for position, char in enumerate(text):
        if char in _PAIRS:
            stack.append((char, position))
        elif char in _OPENERS:
            expected = _OPENERS[char]
            for index in range(len(stack) - 1, -1, -1):
                if stack[index][0] == expected:
                    spans.append((stack[index][1], position + 1))
                    del stack[index:]
                    break
    return tuple(sorted(spans))


def _plain_label(value: str) -> str:
    """仅为模式识别跳过行首装饰符及组合符；不产生改写后的输出正文。"""
    cursor = 0
    while cursor < len(value):
        char = value[cursor]
        if char in WHITE or unicodedata.category(char)[0] in {"S", "P", "M"}:
            cursor += 1
        else:
            break
    return value[cursor:]


def _attribute_line(value: str) -> bool:
    """识别有字段分隔符的属性行及明确的免票字段，普通完整句不命中。

    这些字段可以在原文中独立成节，因此只增加绑定前文的软罚分。不能
    将出现“交通”“时间”等词的任意句子都视为依赖前段的属性。
    """
    plain = _plain_label(value)
    return bool(_ATTRIBUTE.match(plain) or re.fullmatch(r"(?:无门票|免门票|免费入场|门票免费)[。！!]?", plain.strip()))


def _heading(value: str) -> bool:
    """识别短的显式题头，排除属性字段和普通短句。

    32 个非空白码点仅是保守开发上限。日期／天序／编号、末尾冒号、
    地点／地图装饰符开头的名词式短行提供证据；并不尝试命名实体识别或推断缺失
    的标题层级。带完整叙述标点的装饰短句不因含 emoji 自动成为题头。
    """
    value = value.strip()
    if not value or sum(char not in WHITE for char in value) > 32 or _attribute_line(value):
        return False
    plain = _plain_label(value)
    if not plain:
        return False
    if _DAY.fullmatch(plain) or _DATE.match(plain):
        return True
    if value.endswith((":", "：")) and not any(char in _STRONG for char in value[:-1]):
        return True
    if _NUMBERED.match(value) and not any(char in "，,。！？!?；;：:" for char in value):
        return True
    decorated = plain != value and any(unicodedata.category(char).startswith("S") for char in value[:len(value) - len(plain)])
    if not decorated:
        return False
    if _ROUTE_TITLE.match(plain):
        return True
    # 问候、情绪和天气 emoji 常出现在普通短句中，不能把所有装饰行都
    # 当题头。这里只接纳常见地点／露营／地图标签；不包含具体地点词表。
    if not any(char in "📍⛺🗺" for char in value[:len(value) - len(plain)]):
        return False
    # 具有“题头：短标签”的行允许末尾感叹号；逗号、句号或问号意味着
    # 完整叙述的可能性较大，保守不识别。标签不靠景点名称词表判断。
    if any(char in "，,。？?；;" for char in plain):
        return False
    if any(char in "！!" for char in plain):
        return bool(re.fullmatch(r"[^：:！!]+[：:][^：:！!]{1,8}[！!]", plain))
    return True


def _sentence_end(value: str) -> bool:
    """忽略尾部闭符及附着组合符，识别已存在单元的句末端点。"""
    cursor = len(value)
    while cursor:
        char = value[cursor - 1]
        if char in WHITE or char in _CLOSERS or unicodedata.category(char).startswith("M"):
            cursor -= 1
        else:
            return char in _STRONG
    return False


def analyze_gap_structure(
    text: str, units: Sequence[TextUnit], *, mode: StructureMode = "structure",
) -> tuple[GapStructure, ...]:
    """返回每个既有 gap 的结构证据，供边界候选独立比较。

    sentence_only 仅区分句／行端点和单元内细分点；structure 另观察已配对
    括号、左侧短题头、右侧属性行和右侧题头。基础句内罚分为 8；配对括号
    内另加 12，题头后另加 6，属性行前另加 6。罚分可累加但不是概率，
    不蕴含任何默认正式规则或段数目标。输入非法时抛稳定 reason code，
    空白输入配空单元返回空元组；N=1 仍校验覆盖后返回空元组。
    """
    if not isinstance(mode, str) or mode not in {"sentence_only", "structure"}:
        raise ValueError("invalid_structure_mode")
    _validate_spans(text, units)
    line_starts = (0, *(match.end() for match in _LINES.finditer(text)))
    pairs = _matched_brackets(text) if mode == "structure" else ()

    def starts_line(unit: TextUnit) -> bool:
        """缩进空白不妨碍行首；句中 emoji 不冒充一行题头。"""
        line_start = line_starts[bisect_right(line_starts, unit.start) - 1]
        return all(char in WHITE for char in text[line_start:unit.start])

    headings = tuple(starts_line(unit) and _heading(text[unit.start:unit.end]) for unit in units)
    attributes = tuple(starts_line(unit) and _attribute_line(text[unit.start:unit.end]) for unit in units)
    output: list[GapStructure] = []
    for index, (left, right) in enumerate(zip(units, units[1:])):
        line = bool(_LINES.search(text[left.end:right.start]))
        sentence = _sentence_end(text[left.start:left.end])
        kind = "line" if line else "sentence" if sentence else "internal"
        reasons = ["line_end" if line else "sentence_end" if sentence else "unit_internal"]
        evidence = [(reasons[0], left.end, right.start)]
        penalty = 0 if line or sentence else 8
        if mode == "structure":
            for start, end in pairs:
                if start < left.end < end:
                    reasons.append("inside_matched_bracket")
                    evidence.append(("inside_matched_bracket", start, end))
            if "inside_matched_bracket" in reasons:
                penalty += 12
            if headings[index]:
                reasons.append("after_heading")
                evidence.append(("after_heading", left.start, left.end))
                penalty += 6
            if attributes[index + 1]:
                reasons.append("before_attribute_line")
                evidence.append(("before_attribute_line", right.start, right.end))
                penalty += 6
            if headings[index + 1]:
                reasons.append("before_heading")
                evidence.append(("before_heading", right.start, right.end))
        output.append(GapStructure(index=index, boundary_kind=kind, natural_boundary=line or sentence,
                                   penalty=penalty, reasons=tuple(dict.fromkeys(reasons)),
                                   evidence_spans=tuple(evidence)))
    return tuple(output)
