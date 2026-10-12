"""从作者显式栏目关系发现完整介绍／推荐块，仅提供可审计的原文跨度。

本模块不计算向量、选择切点、访问帖子身份或计数 token。日程＋路线／条目、
列举引导＋编号、目录引导＋题头说明，以及明确逐项列出的组成部分，提供父块
证据；编号、地点变更或普通“推荐”字样本身均不足以建立父块。正文中的一般
转折不终止父块，新日程、明确栏目或父结构结束才终止。未识别的结构保持无
证据，不把启发式输出当作语义金标。调用者须按真实 tokenizer 执行硬上限。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence
import unicodedata

import regex

from .contracts import TextUnit
from .integrity import WHITE, validate
from .semantic_structure import StructureBlock

_LINES = re.compile(r"\r\n|[\r\n\u2028\u2029]")
_DAY = re.compile(r"^(?:day\s*\d+|d\s*\d+|第[0-9一二三四五六七八九十百]+天)(?![0-9])", re.I)
_DATE = re.compile(r"^(?:\d{2,4}年)?\d{1,2}月\d{1,2}日")
_NUMBER = re.compile(r"^(?P<number>\d+|[一二三四五六七八九十]+)[、．.）)]\s*")
_PAREN_NUMBER = re.compile(r"^[（(](?P<number>\d+|[一二三四五六七八九十]+)[）)]\s*")
_RELATION = re.compile(r"(?:主要|共)?(?:分为|包括|包含)")
_STRONG = frozenset("。！？!?；;")
_ARROWS = frozenset("→➡➜")


@dataclass(frozen=True)
class _Line:
    """保留空行与原文坐标，避免移除段落间隔后错误跨栏目合并。"""

    start: int
    end: int


def _lines(text: str) -> tuple[_Line, ...]:
    """按真实换行建行；仅坐标忽略缩进，输出始终使用原文。"""
    result: list[_Line] = []
    start = 0
    for end, following in [*((m.start(), m.end()) for m in _LINES.finditer(text)), (len(text), len(text))]:
        left, right = start, end
        while left < right and text[left] in WHITE:
            left += 1
        while right > left and text[right - 1] in WHITE:
            right -= 1
        result.append(_Line(left, right))
        start = following
    return tuple(result)


def _plain(value: str) -> str:
    """只跳过用于识别的前置装饰，不改写原文或证据坐标。"""
    cursor = 0
    while cursor < len(value) and (value[cursor] in WHITE or unicodedata.category(value[cursor])[0] in {"S", "P", "M"}):
        cursor += 1
    return value[cursor:]


def _metadata(value: str) -> bool:
    """纯尾标签、链接、账号或装饰终止实质块，正文中的这些内容不终止。"""
    remainder = re.sub(r"https?://\S+|#[^\s#]+#?|@\S+", "", value)
    return bool(value) and not any(unicodedata.category(c)[0] in {"L", "N"} for c in remainder)


def _integer(value: str) -> int | None:
    """仅转换现有列表格式的一至九十九或十进制数，不猜测损坏编号。"""
    if value.isdecimal():
        return int(value)
    digits = dict(zip("一二三四五六七八九", range(1, 10), strict=True))
    if value in digits:
        return digits[value]
    if value == "十":
        return 10
    if value.count("十") == 1:
        left, right = value.split("十")
        if (not left or left in digits) and (not right or right in digits):
            return 10 * digits.get(left, 1) + digits.get(right, 0)
    return None


def _number(value: str) -> int | None:
    """编号独立于条目长度，完整长段落也可作为一条组成部分。"""
    match = _NUMBER.match(value) or _PAREN_NUMBER.match(value)
    if match:
        return _integer(match.group("number"))
    cluster = regex.match(r"\X", value)
    if cluster:
        marker = cluster.group()
        if regex.fullmatch(r"[0-9]\ufe0f?\u20e3", marker):
            return int(marker[0])
        if unicodedata.category(marker[0]) == "No":
            numeric = unicodedata.numeric(marker[0], None)
            if numeric is not None and numeric.is_integer():
                return int(numeric)
    return None


def _title(value: str) -> bool:
    """保守识别短题头版式；完整陈述、属性说明与单纯短字数不是充分证据。"""
    core = re.sub(r"[（(][^()（）]*[）)]", "", _plain(value)).strip()
    if (not core or len(core) > 32 or _metadata(value)
            or any(c in core for c in "，,。！？!?；;：:")):
        return False
    if re.match(r"^(?:我(?:们)?|你(?:们)?|他(?:们)?|她(?:们)?|大家|今天|昨天|明天|这次|可以|需要|应该|建议|不能|不用|无需|一定)", core):
        return False
    return not bool(re.search(r"没有|很|非常|十分|太|已经|正在|觉得|喜欢|值得|[了着](?:$|[的呢呀吧])", core))


def _is_day_transition(value: str) -> bool:
    """显式行首日序／日期终止前一父块，允许之后紧接完整叙述。

    终止条件与创建父栏目不同：句末标点不能掩盖换天。中文日序后须有分隔
    版式，避免把“第二天候选名单”这类名词短语误判为新日程；日期沿用
    相同分隔要求。D1／Day1允许紧接路线名称。行中提及日期／日序不命中，
    不靠叙述动作词猜测，也不将日期用于推断事件先后或计算旅行天数。
    """
    plain = _plain(value)
    prefix = _DAY.match(plain) or _DATE.match(plain)
    if prefix is None:
        return False
    remainder = plain[prefix.end():]
    return (not remainder or remainder[0] in WHITE or remainder[0] in ":：、.．-（(【[，,"
            or prefix.group().lower().startswith("d"))


def _day(value: str) -> bool:
    """只有日序题头可建立父栏目；整行路线及完整日记句仍只用于终止。"""
    return (bool(_DAY.match(_plain(value))) and _is_day_transition(value)
            and not any(c in value for c in _STRONG | _ARROWS))


def _directory(value: str) -> bool:
    """显式向下引导符才提供目录证据，不通过推荐、店铺或景点词表猜测。"""
    return len(value) <= 64 and value.rstrip().endswith(("↓", "👇"))


def _section(value: str) -> bool:
    """封闭括号栏目或独立冒号短题头终止父层，不把行内冒号字段视为栏目。"""
    bare = value.strip().rstrip(":：").strip()
    if re.fullmatch(r"[【\[][^\[\]【】\r\n]+[】\]]", bare):
        return not any(c in bare for c in _STRONG | _ARROWS) and len(bare) <= 40
    return value.endswith((":", "：")) and _title(value[:-1])


def _letters(value: str) -> tuple[str, tuple[int, ...]]:
    """只为重复主题证据生成字符与原偏移映射；不做实体识别或分词。"""
    positions = tuple(i for i, c in enumerate(value) if unicodedata.category(c)[0] in {"L", "N"})
    return "".join(value[i] for i in positions), positions


def _shared_span(left: str, right: str, minimum: int) -> tuple[int, int, int, int] | None:
    """返回达到既定证据门的连续短复现及两侧原偏移，不搜索最长公共子串。

    调用方只需判定是否存在至少2／3／4字复现；存在长度不小于门槛的公共
    子串，与存在一个门槛长度的公共窗口等价。固定短窗口索引因此保持原
    接纳条件，并避免重复长正文触发最长匹配的二次耗时。按左侧最早、再右侧
    最早窗口稳定选证据；归一化仍仅略过非文字／数字，未引入模糊语义匹配。
    门槛固定时耗时与输入长度线性相关；原坐标可含被略过的标点或装饰。
    """
    a, offsets_a = _letters(left)
    b, offsets_b = _letters(right)
    if len(a) < minimum or len(b) < minimum:
        return None
    right_windows: dict[str, int] = {}
    for start in range(len(b) - minimum + 1):
        right_windows.setdefault(b[start:start + minimum], start)
    for left_start in range(len(a) - minimum + 1):
        right_start = right_windows.get(a[left_start:left_start + minimum])
        if right_start is not None:
            return (offsets_a[left_start], offsets_a[left_start + minimum - 1] + 1,
                    offsets_b[right_start], offsets_b[right_start + minimum - 1] + 1)
    return None


def _intro_start(text: str, lines: Sequence[_Line], index: int, lead_prefix: str) -> tuple[int, tuple[tuple[str, int, int], ...]]:
    """仅附着相邻同主题简介及其短总题头，不凭相邻关系吞并前一独立段。

    引导句在列举关系词之前与前一实质行须逐字复现至少四个文字／数字字符。
    可再附着同样复现主题的短总题头；不能跨空行、编号、目录或新栏目。
    这只是同主题原文证据，无法证明未标记文字的叙述功能，调用方仍须复核。
    """
    start, evidence = lines[index].start, []
    for previous in range(index - 1, max(-1, index - 3), -1):
        line = lines[previous]
        value = text[line.start:line.end]
        if (not value or _number(value) is not None or _is_day_transition(value) or _directory(value)
                or _section(value) or _metadata(value)):
            break
        if previous == index - 2 and len(value) > 48:
            break
        shared = _shared_span(value, lead_prefix, 4)
        if shared is None:
            break
        start = line.start
        evidence.append(("shared_parent_topic", line.start + shared[0], line.start + shared[1]))
        evidence.append(("preceding_introduction" if previous == index - 1 else "parent_title", line.start, line.end))
    return start, tuple(evidence)


def _numbered_items(text: str, lines: Sequence[_Line], index: int) -> tuple[tuple[int, int], ...]:
    """提取列举引导之后从一开始的连续编号，组内续句和转折随条目保留。"""
    items: list[tuple[int, int]] = []
    blank = False
    for line in lines[index + 1:]:
        value = text[line.start:line.end]
        if not value:
            blank = True
            continue
        number = _number(value)
        if _is_day_transition(value) or _directory(value) or _metadata(value) or (number is None and _section(value)):
            break
        if number is not None:
            if number != len(items) + 1:
                break
            items.append((line.start, line.end))
        elif not items or blank:
            break
        else:
            items[-1] = items[-1][0], line.end
        blank = False
    return tuple(items) if len(items) >= 2 else ()


def _described_items(text: str, lines: Sequence[_Line], index: int, suffix: str) -> tuple[tuple[int, int], ...]:
    """只接纳先列出N个名称、再按次序解释N个原文段落的显式组成关系。

    每段开头须与对应名称复现至少两个文字字符；名称可改变词序，但完全没有
    复现证据时宁可不建块。空行允许分隔组成部分，不靠段数机械吞入下文。
    """
    match = re.fullmatch(r"(?P<labels>.+?)(?P<count>\d+|[一二三四五六七八九十]+)个[^，,。:：]+[：:]", suffix)
    if match is None:
        return ()
    labels = re.split(r"[、，,]", match.group("labels"))
    count = _integer(match.group("count"))
    if count is None or count < 2 or len(labels) != count:
        return ()
    body = [line for line in lines[index + 1:] if line.start < line.end]
    if len(body) < count:
        return ()
    items = []
    for label, line in zip(labels, body[:count], strict=True):
        value = text[line.start:line.end]
        opening = re.split(r"[，,。:：；;]", value, maxsplit=1)[0]
        if (_is_day_transition(value) or _directory(value) or _section(value) or _metadata(value)
                or (len(value) < 24 and not any(c in value for c in _STRONG))
                or _shared_span(label, opening, 2) is None):
            return ()
        items.append((line.start, line.end))
    return tuple(items)


def _directory_items(text: str, lines: Sequence[_Line], index: int) -> tuple[tuple[int, int], ...]:
    """目录后以完整题头＋说明形成连续子项，末项延伸须有标题回指证据。

    首说明要求比无标记短题头更长且至少24字，与既有题头识别证据一致。
    后续独立行若不逐字回指该标题，停止目录范围，避免将末尾全篇建议吞入。
    这会保守遗漏没有任何版式／回指证据的续段，不能伪称理解了所有推荐。
    """
    items: list[tuple[int, int]] = []
    cursor = index + 1
    while cursor + 1 < len(lines):
        title_line = lines[cursor]
        value = text[title_line.start:title_line.end]
        if not value:
            cursor += 1
            continue
        if _is_day_transition(value) or _directory(value) or _section(value) or not _title(value):
            break
        body_index = cursor + 1
        while body_index < len(lines) and lines[body_index].start == lines[body_index].end:
            body_index += 1
        if body_index == len(lines):
            break
        body = lines[body_index]
        description = text[body.start:body.end]
        if (_title(description) or _section(description) or _directory(description)
                or _is_day_transition(description) or _metadata(description)
                or len(description) < max(24, len(value) + 1)):
            break
        end, cursor = body.end, body_index + 1
        while cursor < len(lines):
            following = lines[cursor]
            continuation = text[following.start:following.end]
            if (not continuation or _is_day_transition(continuation) or _title(continuation) or _section(continuation)
                    or _directory(continuation) or _metadata(continuation)
                    or _shared_span(value, re.split(r"[，,。:：；;]", continuation, maxsplit=1)[0], 3) is None):
                break
            end, cursor = following.end, cursor + 1
        items.append((title_line.start, end))
    return tuple(items) if len(items) >= 2 else ()


def _route_items(text: str, lines: Sequence[_Line], index: int) -> tuple[tuple[int, int], ...]:
    """向下目录中的多日箭头路线属于一份概览，不按天序建立逐日必切点。

    每条路线须同时含日序前缀与箭头；后续无句末、无装饰的短行只作为排版
    折行续接。下一目录、完整日程题头、独立栏目或实质叙述结束概览，不能把
    一次偶然箭头或无目录的多行正文升级为全篇保护块。
    """
    items: list[tuple[int, int]] = []
    for line in lines[index + 1:]:
        value = text[line.start:line.end]
        if not value:
            continue
        route = bool(_DAY.match(_plain(value))) and any(c in value for c in _ARROWS)
        if ((_is_day_transition(value) and not route) or _directory(value) or _section(value) or _metadata(value)):
            break
        if route:
            items.append((line.start, line.end))
        elif (items and len(value) <= 32 and _plain(value) == value
              and not any(c in value for c in "，,。！？!?；;：:")):
            items[-1] = items[-1][0], line.end
        else:
            break
    return tuple(items) if len(items) >= 2 else ()


def _schedule_block(text: str, lines: Sequence[_Line], index: int) -> StructureBlock | None:
    """仅对有路线或重复子题头支撑的日程建立范围，不整天保护普通日记。"""
    heading = lines[index]
    entries: list[tuple[int, int]] = []
    families: dict[str, int] = {}
    route: tuple[int, int] | None = None
    end, blank = heading.end, False
    for line in lines[index + 1:]:
        value = text[line.start:line.end]
        if not value:
            blank = True
            continue
        if _is_day_transition(value) or _directory(value) or _section(value) or _metadata(value):
            break
        marked_title = _title(value) and (_plain(value) != value or _number(value) is not None)
        if blank and not marked_title:
            break
        if any(c in value for c in _ARROWS):
            # 只有紧接日程题头的路线才支撑整栏；正文中的偶然箭头不算。
            if end == heading.end:
                route = line.start, line.end
        elif marked_title:
            entries.append((line.start, line.end))
            cluster = regex.match(r"\X", value)
            family = "numbered" if _number(value) is not None else cluster.group().replace("\ufe0f", "")
            families[family] = families.get(family, 0) + 1
        elif entries:
            entries[-1] = entries[-1][0], line.end
        end, blank = line.end, False
    if end == heading.end or (route is None and max(families.values(), default=0) < 2):
        return None
    evidence = [("parent_heading", heading.start, heading.end)]
    if route is not None:
        evidence.append(("parent_route", *route))
    evidence.extend(("coherent_item", *span) for span in entries)
    return StructureBlock(heading.start, end, "coherent_schedule", tuple(evidence))


def discover_coherent_blocks(text: str, units: Sequence[TextUnit]) -> tuple[StructureBlock, ...]:
    """返回有明确父结构证据的完整介绍／推荐跨度，不决定它们是否成为片段。

    输入为完整原文和既有TextUnit；无损覆盖、边界字素与单元类型须合法，
    非法时抛ValueError。空白输入配空单元返回空元组。输出跨度可以位于模型
    单元内部，但调用方只能在已有gap选点；没有对应gap时应报告而非挪动坐标。
    evidence中的coherent_item提供完整子项，供硬长度退让与复核；父块发现不
    自动证明上下文语义相同，更不允许免除512等配置中的真实token硬限制。
    """
    if (not isinstance(text, str) or any(not isinstance(unit, TextUnit)
            or type(unit.start) is not int or type(unit.end) is not int for unit in units)):
        raise ValueError("invalid_coherent_input")
    validate(text, [dict(seg_id=i + 1, raw_text=text[u.start:u.end], start=u.start, end=u.end)
                    for i, u in enumerate(units)])
    if any(not any(c not in WHITE for c in text[u.start:u.end]) for u in units):
        raise ValueError("empty_coherent_unit")
    lines, output = _lines(text), []
    for index, line in enumerate(lines):
        value = text[line.start:line.end]
        if _day(value):
            block = _schedule_block(text, lines, index)
            if block is not None:
                output.append(block)
        if _directory(value):
            routes = _route_items(text, lines, index)
            items = routes or _directory_items(text, lines, index)
            if items:
                output.append(StructureBlock(line.start, items[-1][1], "coherent_directory",
                              (("parent_heading", line.start, line.end),
                               *(("parent_route", *item) for item in routes),
                               *(("coherent_item", *item) for item in items))))
        relation = _RELATION.search(value)
        if relation is None or not value.endswith((":", "：")):
            continue
        items = _numbered_items(text, lines, index)
        if not items:
            items = _described_items(text, lines, index, value[relation.end():])
        if not items:
            continue
        start, introduction = _intro_start(text, lines, index, value[:relation.start()])
        output.append(StructureBlock(start, items[-1][1], "coherent_enumeration",
                      (("parent_heading", line.start, line.end), *introduction,
                       *(("coherent_item", *item) for item in items))))
    return tuple(sorted(set(output), key=lambda block: (block.start, block.end, block.kind)))
