"""分层意思分区核心：规则决定禁止切与必须切，语义模型只在普通边界上选择。

本模块仅做纯计算；不访问文件、数据库、帖子身份、模型或向量缓存，不新增
单元或伪造 embedding。调用者先用原有评分函数生成 SemanticDecision；窗口
和阈值由调用者明确保存。1.3（适配方案§5.2.3）的优先级为：禁止切＞换天与
独立小节必切＞长度兜底＞普通边界。普通边界只来自过深度门槛的语义谷值，
且须前文已完成、不在同栏目清单内部；句号、换行、编号、题头形态或属性变化
本身都不触发切分。1.3.1（开发试验）另把放错位置的普通切点移到所属边界：
条目参数行留在前项、挂在前段末尾的标题随后文、标题前单独一句收束语留在前节，
落点须是可确认的题头或下一条目首行，否则保留原位；换天逐行识别，不依赖结构提名。最后只撤销切点：无文字碎段、过短首末段、只有
题头的片段及独立小节前单独剩下的收束句并入相邻段。运行结果仍须抽查，不能
自动证明所有片段均符合人的切分直觉。无法在既有 gap 上表达的意思转换保留为
限制，不能移动到句内相邻位置代替。
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from dataclasses import asdict, dataclass
from math import isfinite
import re
from typing import Sequence
import unicodedata

from .contracts import SemanticDecision, TextSegment, TextUnit, TokenCounter
from .integrity import WHITE, validate
from .semantic_config import CoherentSemanticConfig
from .semantic_coherence import discover_coherent_blocks
from .semantic_relations import analyze_gap_continuations, analyze_gap_relations
from .semantic_sections import (
    XHS_EMOJI_CODE, SectionCandidate, SectionLine, count_sentences, discover_sections,
    first_complete_end, introduces_schedule, label_field, leading_mark, marker_class, metadata_text,
    parameter_text, separator_text,
)
from .semantic_structure import (
    StructureBlock, analyze_gap_structure, discover_context_blocks,
    discover_structure_blocks,
)

EXPERIMENT_ID = "meaning-layers-4"
_LINES = re.compile(r"\r\n|[\r\n\u2028\u2029]")
_DAY = re.compile(r"(?:day\s*\d+|d\s*\d+|第[0-9一二三四五六七八九十百]+天)(?![0-9])", re.I)
_DATE = re.compile(r"(?:\d{2,4}年)?\d{1,2}月\d{1,2}日")
_ATTRIBUTE = re.compile(r"^(?:交通|地址|位置|导航|时间|时长|开放时间|营业时间|门票|票价|价格|费用|预约|停车|电话|优点|缺点|亮点)\s*[:：|｜]")
_NUMBER = re.compile(r"^(?:\d+[、．.）)]|[一二三四五六七八九十百]+[、．.]|[（(][一二三四五六七八九十百\d]+[）)])")
_STRONG = frozenset("。！？!?；;")
_PICTURE_NOTE = re.compile(r"[，,]\s*(?:图|[pP])\s*\d+(?:\s*[-~—–]\s*\d+)?\s*$")
# 天序数字：Day3、D3、第三天；用于识别同一天的重复标题（如“Day3”下一行“第三天去了……”）。
_DAY_NUMBER = re.compile(r"(?:day\s*(\d+)|d\s*(\d+)|第([0-9一二三四五六七八九十百]+)天)", re.I)
_CHINESE_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


@dataclass(frozen=True)
class MeaningGapDecision:
    """一个已有 gap 的最终选择及原文证据；index 是零基单元间索引。

    sources 记录被提名的来源，不等同被选中；reason_codes 同时解释过滤、
    碎段合并及长度兜底。evidence_spans 使用原文码点左闭右开坐标，
    包含评分所对应的左右单元，避免仅给不可核验的抽象原因。
    """

    index: int
    selected: bool
    sources: tuple[str, ...]
    reason_codes: tuple[str, ...]
    evidence_spans: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class MeaningPartition:
    """完整分区结果；边界元组均为 gap 索引，结构块则为原文码点。

    protected_blocks 同时含禁止切依据（题头附着、尾标签、句法续接）和只作
    证据的连贯块（coherent_*），后者不阻止切分。
    结构与语义来源可在同一个切点重叠，length_boundaries 只记录额外长度切点。
    length_exceptions 是超过长度触发线却无法在允许位置补切的片段编号（从1
    开始）；这些片段保留原文，等待人工查看。所有字段不可变，
    dataclasses.asdict 可用于调用方的实验记录。
    """

    segments: tuple[TextSegment, ...]
    semantic_boundaries: tuple[int, ...]
    structural_boundaries: tuple[int, ...]
    length_boundaries: tuple[int, ...]
    suppressed_semantic_boundaries: tuple[int, ...]
    length_exceptions: tuple[int, ...]
    gap_decisions: tuple[MeaningGapDecision, ...]
    protected_blocks: tuple[StructureBlock, ...]
    unavailable_structural_spans: tuple[tuple[str, int, int], ...]
    experiment_id: str = EXPERIMENT_ID


@dataclass(frozen=True)
class _Line:
    """完整原文行去除两端 WHITE 后的坐标，不以模型单元伪装原文行。"""

    start: int
    end: int


@dataclass(frozen=True)
class _Heading:
    """题头来源及原文行内起点；未标记短行还需要后文支持。

    start_offset 保留同行句后标题的位置，不能用整行起点替代已有 gap。
    """

    kind: str
    explicit: bool
    start_offset: int = 0


def _lines(text: str) -> tuple[_Line, ...]:
    """保留真实行位置；跳过空行，但不删除输出中的换行或缩进。"""
    output, start = [], 0
    for end, following in [*((m.start(), m.end()) for m in _LINES.finditer(text)), (len(text), len(text))]:
        left, right = start, end
        while left < right and text[left] in WHITE:
            left += 1
        while right > left and text[right - 1] in WHITE:
            right -= 1
        if left < right:
            output.append(_Line(left, right))
        start = following
    return tuple(output)


def _plain(value: str) -> str:
    """只在识别副本上去行首装饰，绝不将此副本用作输出正文。

    1.3起小红书表情代码（如“[三R]”）整体视为行首装饰，否则“[三R]第三天”
    这类日程题头会因代码中的文字而认不出。
    """
    cursor = 0
    while cursor < len(value):
        if code := XHS_EMOJI_CODE.match(value, cursor):
            cursor = code.end()
        elif value[cursor] in WHITE or unicodedata.category(value[cursor])[0] in {"S", "P", "M"}:
            cursor += 1
        else:
            break
    return value[cursor:]


def _trailing_schedule_start(value: str) -> int | None:
    """只识别强句末之后独立附带的天序标题，不从普通正文猜测时间主题。

    比如‘出行笔记。📍Day1’末尾含显式日程，而‘出行笔记。晚上出发’
    只是普通叙述，不在此建立题头。返回原字符串偏移，不能创造模型gap。
    """
    for index in range(len(value) - 1, -1, -1):
        if value[index] in _STRONG:
            start = index + 1
            while start < len(value) and value[start] in WHITE:
                start += 1
            return start if _DAY.fullmatch(_plain(value[start:])) else None
    return None


def _heading(value: str) -> _Heading | None:
    """用一般版式识别日程、目录引导及短题头，排除明显的完整陈述。

    日程前缀可与当日正文同处一行；带箭头的 D1 路线行只是目录说明，
    不以每个天序强拆完整路线目录。无标记题头的后文证据在发现函数检查。
    本启发式不读取对象名词表，也不声称解决完整中文句法或标题层级。
    """
    plain = _plain(value)
    # 人均、价格、地址等条目参数行属于所在条目，不是题头（1.3.1）。
    if not plain or metadata_text(value) or _ATTRIBUTE.match(plain) or parameter_text(value):
        return None
    trailing_start = _trailing_schedule_start(value)
    if trailing_start is not None:
        return _Heading("trailing_schedule_heading", True, trailing_start)
    if any(c in value for c in "→➡➜"):
        return None
    if _DAY.match(plain) or _DATE.match(plain):
        return _Heading("schedule_heading", True)
    if len(value) <= 48 and value.rstrip().endswith(("↓", "👇")):
        return _Heading("directory_lead", True)
    if sum(c not in WHITE for c in value) > 32:
        return None
    # 括号附注中的评价不否定名词式题头主体；正文的强标点仍排除。
    core = re.sub(r"[（(][^()（）]*[）)]", "", plain).strip()
    # 行尾“，图4”“，p2-3”只是配图指引，不使标题变成句子（1.3.1）。
    core = _PICTURE_NOTE.sub("", core)
    if any(c in core for c in "，,。！？!?；;") or not core:
        return None
    # “我的路线”“我们的行程”这类所有格开头是名词式标题，不是第一人称叙述（1.3.1）。
    if re.match(r"^(?:我(?:们)?(?!的)|你(?:们)?(?!的)|他(?:们)?|她(?:们)?|大家|今天|昨天|明天|可以|需要|应该|建议|不能|不用|无需|一定)", core):
        return None
    if re.search(r"没有|很|非常|十分|太|已经|正在|觉得|喜欢|值得|[了着](?:$|[的呢呀吧])", core):
        return None
    explicit = plain != value or bool(_NUMBER.match(value)) or core.endswith(("：", ":"))
    # ‘晚上：地点’等短冒号标签是显式小题头，普通价格/交通字段已排除。
    explicit = explicit or bool(re.fullmatch(r"[^:：]{1,12}[:：][^:：]{1,20}", core))
    return _Heading("explicit_heading" if explicit else "unmarked_heading", explicit)


def _trailing_metadata_block(text: str, lines: Sequence[_Line]) -> StructureBlock | None:
    """将连续纯元数据尾行绑定到前一实质行的最后一句，坐标始终取原文。

    标签、账号、链接和装饰均沿用本模块的整行识别，不吞含普通陈述的行。
    原句可以横跨多个模型单元；后续硬长度图禁止只含这些尾行的末段。
    全文只有元数据时没有可依附的正文，返回 None，不虚构实质句证据。
    """
    before = len(lines) - 1
    while before >= 0 and metadata_text(text[lines[before].start:lines[before].end]):
        before -= 1
    if before < 0 or before == len(lines) - 1:
        return None
    line = lines[before]
    cursor, prior = line.start, None
    while cursor < line.end:
        stop = first_complete_end(text, cursor, line.end)
        if any(unicodedata.category(c)[0] in {"L", "N"} for c in text[cursor:stop]):
            prior = cursor, stop
        cursor = stop
        while cursor < line.end and text[cursor] in WHITE:
            cursor += 1
    if prior is None:
        raise ValueError("missing_substantive_metadata_context")
    return StructureBlock(prior[0], len(text), "trailing_metadata_context",
                          (("preceding_substantive_sentence", *prior),
                           ("trailing_metadata", lines[before + 1].start, len(text))))


def _meaning_blocks(text: str, units: Sequence[TextUnit]) -> tuple[StructureBlock, ...]:
    """只绑定题头与首说明，并保留尾标签及窄句法依赖，不保护整条目。

    旧 heading_context 会跨过短说明及下一显式条目，故不直接复用。显式
    题头遇到下一显式题头立即停止；无标记短行须接到更长、至少24字的
    说明才有足够证据，明确题头则无需此长度。目录引导允许店名/路线标题
    作为中间一层；裸标题与下行重复标签前缀提供有限的嵌套层级证据。
    不据未知层级越过新日程或不同条目。不能确认的题头不造保护块。
    """
    output: list[StructureBlock] = []
    # 仅复用已独立定义的窄句法依赖；尾部须与本模块元数据识别保持一致。
    output.extend(block for block in discover_context_blocks(text, units)
                  if block.kind == "syntactic_continuation")
    lines = _lines(text)
    if (tail := _trailing_metadata_block(text, lines)) is not None:
        output.append(tail)
    headings = tuple(_heading(text[line.start:line.end]) for line in lines)
    repeated = {block.start: block for block in discover_structure_blocks(text, units)}
    for index, (line, heading) in enumerate(zip(lines, headings, strict=True)):
        if heading is None:
            continue
        value = text[line.start:line.end]
        heading_start = line.start + heading.start_offset
        kind = "repeated_entry_lead" if line.start in repeated else heading.kind
        evidence: list[tuple[str, int, int]] = [("heading", heading_start, line.end)]
        if line.start in repeated:
            evidence.extend(item for item in repeated[line.start].evidence if item[0] in {"repeated_marker", "matching_marker"})
        # 同行日程本身带完整叙述时，首句已经足够，不再绑定下一原文行。
        if heading.kind == "schedule_heading" and any(c in _STRONG for c in value):
            stop = first_complete_end(text, line.start, line.end)
            output.append(StructureBlock(line.start, stop, kind, (*evidence, ("first_substantive_sentence", line.start, stop))))
            continue
        if index + 1 == len(lines):
            continue
        body_index = index + 1
        following = headings[body_index]
        # “信号山：官方预约”这类字段行本身已带内容；下一行是题头形态时它是另一项或新
        # 标题，不是字段的说明，不绑定（1.3.1）。
        if following is not None and label_field(value):
            continue
        if following is not None and following.explicit:
            bare_label = _plain(value)
            following_value = _plain(text[lines[body_index].start:lines[body_index].end])
            nested_label = (heading.kind == "explicit_heading" and len(bare_label) <= 12
                            and not any(c in bare_label for c in ":：，,。！？!?；;")
                            and following.kind == "explicit_heading"
                            and following_value.startswith(bare_label)
                            and len(following_value) > len(bare_label))
            if not nested_label or body_index + 1 == len(lines):
                continue
            evidence.append(("repeated_directory_label", line.start, line.end))
            evidence.append(("nested_heading", lines[body_index].start, lines[body_index].end))
            body_index += 1
            following = headings[body_index]
            if following is not None and following.explicit:
                continue
        # ‘几餐↓ / 店名 / 店铺说明’中的店名依赖后继解释，目录不能单飞。
        if heading.kind == "directory_lead" and following is not None and body_index + 1 < len(lines):
            next_heading = headings[body_index + 1]
            if next_heading is None or not next_heading.explicit:
                body_index += 1
        body = lines[body_index]
        body_value = text[body.start:body.end]
        if metadata_text(body_value):
            continue
        if not heading.explicit and (len(body_value) < 24 or len(body_value) <= len(value)):
            continue
        stop = first_complete_end(text, body.start, body.end)
        output.append(StructureBlock(heading_start, stop, kind, (*evidence, ("first_substantive_sentence", body.start, stop))))
    return tuple(sorted(set(output), key=lambda block: (block.start, block.end, block.kind)))


def _checked_count(counter: TokenCounter, value: str) -> int:
    """精确计数失败时立即拒绝；不拿字符数或单元 token 和冒充段长。"""
    count = counter.count(value)
    if type(count) is not int or count <= 0:
        raise ValueError("invalid_meaning_token_count")
    return count


def _validate_input(text: str, units: Sequence[TextUnit], decision: SemanticDecision,
                    config: CoherentSemanticConfig, counter: TokenCounter) -> None:
    """沿用公共原文与字素验收，并拒绝坏评分身份、非有限数和虚假计数。"""
    if not isinstance(text, str) or not isinstance(config, CoherentSemanticConfig):
        raise ValueError("invalid_meaning_input")
    CoherentSemanticConfig.__post_init__(config)
    validate(text, [dict(seg_id=i + 1, raw_text=text[u.start:u.end], start=u.start, end=u.end) for i, u in enumerate(units)])
    for unit in units:
        if not any(c not in WHITE for c in text[unit.start:unit.end]):
            raise ValueError("empty_meaning_unit")
        if type(unit.token_count) is not int or _checked_count(counter, text[unit.start:unit.end]) != unit.token_count:
            raise ValueError("meaning_unit_token_count_mismatch")
    if len(decision.gaps) != max(0, len(units) - 1):
        raise ValueError("meaning_gap_identity_mismatch")
    for index, gap in enumerate(decision.gaps):
        if (type(gap.index) is not int or gap.index != index or not isfinite(gap.depth)
                or not isfinite(gap.cohesion) or gap.depth < 0 or not -1 <= gap.cohesion <= 1):
            raise ValueError("invalid_meaning_gap_score")
    for boundaries in (decision.semantic_boundaries, decision.length_boundaries):
        if (any(type(i) is not int or not 0 <= i < len(decision.gaps) for i in boundaries)
                or tuple(sorted(set(boundaries))) != tuple(boundaries)):
            raise ValueError("meaning_boundary_identity_mismatch")


def _day_heading(text: str, start: int) -> bool:
    """判断结构提名起点所在行是否为叙述换天题头（Day2、第二天、日期等）。

    只认 _heading 判定的日程题头；带箭头的路线目录行不算换天，交由
    灰色地带处理，避免把多日目录逐行拆开。
    """
    match = _LINES.search(text, start)
    heading = _heading(text[start:match.start() if match else len(text)])
    return heading is not None and heading.kind in {"schedule_heading", "trailing_schedule_heading"}


def _chinese_number(value: str) -> int | None:
    """一至九十九的中文数字；更大或认不出时返回 None。"""
    if "百" in value:
        return None
    if "十" in value:
        tens, _, ones = value.partition("十")
        if (tens and tens not in _CHINESE_DIGITS) or (ones and ones not in _CHINESE_DIGITS):
            return None
        return (_CHINESE_DIGITS[tens] if tens else 1) * 10 + (_CHINESE_DIGITS[ones] if ones else 0)
    return _CHINESE_DIGITS.get(value)


def _day_number(value: str) -> int | None:
    """行首天序（Day3、D3、第三天）的数值；日期或认不出时返回 None。"""
    match = _DAY_NUMBER.match(_plain(value))
    if not match:
        return None
    digits = next(group for group in match.groups() if group)
    return int(digits) if digits.isdigit() else _chinese_number(digits)


def _day_transition_lines(text: str, lines: Sequence[_Line]) -> tuple[frozenset[int], frozenset[int]]:
    """返回（全部行首日程题头的行起点，换天切点应落在的行起点）。

    1.3.1起换天不依赖结构提名，逐行检查。以下日程行不算换天：帖首行（没有切点）；
    紧跟在无句末标点的帖子标题行之后的日程行（标题随第一天）；
    与上一个日程题头天序相同的行（“Day3”之下又写“第三天去了……”，同一天的标题与
    说明）；行程总览、目录中的光秃日程行（本行不带句子，且前一行也是光秃日程行，或
    后一行是不同天序的光秃日程行；下一行是“7月25日，星期四”这类带句子的日期时
    不算总览）；紧凑行程表中的日程行（前一行也是日程行且没有句末标点、天序递增，
    即一行列完一天，如“D2：栈桥-八大关”“7月1日：12:00退潮”；一句话写一天的叙述、
    总览之后重新从第一天写起的详细行程仍换天）。日程行紧跟在引出它的
    标题之后时（“📝行程安排”“出行攻略：”之下的第一天），换天切点上移到该标题之前，
    标题随这一天；标题在帖首时不切。引出标题须以冒号结束或带行程、路线、安排等日程
    词，上一天末尾的条目短行（“台东夜市”）不算；清单项、“标签：内容”字段和参数行
    也不算。行尾附带的日程（“……。📍Day2”）
    不在此列，仍由结构提名处理。
    """
    values = [text[line.start:line.end] for line in lines]
    headings = [_heading(value) for value in values]
    is_day = [heading is not None and heading.kind == "schedule_heading" for heading in headings]
    bare = [day and not any(c in value for c in "，,。！？!?；;") for day, value in zip(is_day, values)]
    numbers = [_day_number(value) if day else None for value, day in zip(values, is_day)]
    all_days, transitions = set(), set()
    last_number: int | None = None
    for index, line in enumerate(lines):
        if not is_day[index]:
            continue
        all_days.add(line.start)
        number = numbers[index]
        same_day = number is not None and number == last_number
        if number is not None:
            last_number = number
        if index == 0 or same_day:
            continue
        previous = values[index - 1]
        # 帖子首行是没有句末标点的标题时，紧接的第一天随标题，不把标题单独切出。
        if index == 1 and not any(c in previous for c in "。！？!?"):
            continue
        # 天序不递增（总览之后又从第一天写起）说明进入详细行程，不按总览或紧凑行程表排除。
        rising = number is None or numbers[index - 1] is None or number > numbers[index - 1]
        if bare[index] and ((bare[index - 1] and rising) or (index + 1 < len(lines) and bare[index + 1]
                                                             and numbers[index + 1] != number)):
            continue
        if is_day[index - 1] and not any(c in previous for c in "。！？!?") and rising:
            continue
        body = previous.lstrip("".join(WHITE))
        listed = body[:1] in "-·•*▪▫◦‧・" or bool(_NUMBER.match(body) and not _SECTION_NUMBER.match(body))
        introduces = introduces_schedule(previous)
        if (headings[index - 1] is not None and not is_day[index - 1] and not listed and not label_field(previous)
                and introduces and not parameter_text(previous) and not any(c in previous for c in "，,。！？!?；;")):
            if index - 1 > 0:
                transitions.add(lines[index - 1].start)
            continue
        transitions.add(line.start)
    return frozenset(all_days), frozenset(transitions)


def _ends_sentence(value: str) -> bool:
    """去掉尾部空白、闭符和装饰后以句末标点结束。"""
    end = len(value)
    while end and (value[end - 1] in WHITE or value[end - 1] in "”’」』)）】]》〉\"'️"
                   or unicodedata.category(value[end - 1])[0] in {"S", "M"}):
        end -= 1
    return bool(end) and value[end - 1] in "。！？!?…～~"


def _list_item_line(text: str, start: int, end: int) -> bool:
    """[start, end) 所在行以项目符号（-·•等）开头，且相邻的上一或下一非空行以同一符号开头。

    这样的行是清单项，不是纯标签小标题（1.3.1）。
    """
    body = text[start:end].lstrip("".join(WHITE))
    if not body or body[0] not in "-·•*▪▫◦‧・":
        return False
    mark = body[0]
    previous = text[:start].rstrip("".join(WHITE))
    previous_line = previous[previous.rfind("\n") + 1:].lstrip("".join(WHITE)) if previous else ""
    following = text[end:].lstrip("".join(WHITE))
    following_line = _LINES.split(following, 1)[0] if following else ""
    return previous_line.startswith(mark) or following_line.startswith(mark)


def _last_list_item_line(text: str, start: int, end: int) -> bool:
    """清单的最后一项：上一行以同一项目符号开头，下一非空行不是。

    只有末项之后的切点不受“标签之后禁止切”否决（清单结束、进入下一部分）；清单
    内部两项之间仍按标签行否决，不在清单中间切开（1.3.1）。
    """
    if not _list_item_line(text, start, end):
        return False
    mark = text[start:end].lstrip("".join(WHITE))[0]
    following = text[end:].lstrip("".join(WHITE))
    return not (_LINES.split(following, 1)[0] if following else "").startswith(mark)


def _label_heading_line(text: str, end: int) -> bool:
    """判断以 end 结束的整行是否为明确的纯标签小标题，其后正文不能被切走。

    只认 _heading 判定的显式题头、日程题头或行尾带装饰的短题头（如“交通🅿️”“▪️游玩路线”
    “🌷Day2 金沙滩对比实测”）；“晚上：河湾灯景”这类“标签：内容”行本身就是
    内容，不受此限制，后面遇到换天仍可切开。无标记短行不在此列，避免把
    普通短句误当题头而少切。
    """
    starts = [match.end() for match in _LINES.finditer(text, 0, end)]
    line_start = starts[-1] if starts else 0
    line = text[line_start:end]
    heading = _heading(line)
    # 清单末项之后是清单的结束，不按标签行否决；清单中间各项之间仍否决（1.3.1）。
    if heading is None or _last_list_item_line(text, line_start, end):
        return False
    # 日程识别只看行首“第X天／DayN”，同行接着写叙述（带逗号或句末标点）时整行
    # 是当天正文，不是纯标签；否则“第一天我们……。”之后的换天切点会被误否决（1.3修正）。
    if heading.kind == "schedule_heading" and any(c in "，,。！？!?；;" for c in line):
        return False
    # “交通🅿️”“美食🍽️”以行尾装饰标示题头，与行首装饰同等视为明确标签。
    core = line.rstrip()
    while core and (core[-1] in WHITE or core[-1] == "\ufe0f" or unicodedata.category(core[-1])[0] in {"S", "M"}):
        core = core[:-1]
    decorated = bool(core) and core != line.rstrip()
    if not (heading.explicit or heading.kind == "trailing_schedule_heading" or decorated):
        return False
    return not re.search(r"[:：]\s*\S", _plain(line))


def _label_above_separator(text: str, end: int) -> bool:
    """以 end 结束的行是分隔线（如“━━━━”），且其上一条非空行是纯标签小标题。

    这种“题头＋分隔线”之后的普通切点应上移到题头之前（1.3），使题头不挂在上一节
    末尾；它不是禁止切证据，不否决独立小节或换天必切。
    """
    starts = [match.end() for match in _LINES.finditer(text, 0, end)]
    start = starts[-1] if starts else 0
    if not separator_text(text[start:end]):
        return False
    previous = start - 1
    while previous > 0 and text[previous - 1] in WHITE:
        previous -= 1
    return previous > 0 and _label_heading_line(text, previous)


def _length_partition(text: str, units: Sequence[TextUnit], decision: SemanticDecision,
                      config: CoherentSemanticConfig, counter: TokenCounter, cuts: Sequence[int],
                      ranks: Sequence[int | None],
                      ) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """长度兜底：只对超过 max_segment_tokens 的意思段补切，返回补切点与超长单元区间。

    ranks 给出每个 gap 作为长度切点的等级：2 换行、1 句末、0 无标点长文中
    以空白分隔的位置，None 表示被禁止切规则否决。先在上限之内的允许位置
    中挑等级最高者，同级取语义变化（depth）最大、再取连贯度更低、更靠后
    者——人被迫分段时也先找段落边界。上限内没有允许位置时，才向
    后取第一个允许位置，并把这一段记为超长异常；直到段尾都没有允许位置
    时，整段保留并记为异常。长度从不推翻禁止切规则，也不截断原文。BPE
    计数不必随前缀单调，因此逐一精确计数而不二分。
    """
    counts: dict[tuple[int, int], int] = {}

    def tokens(first: int, last: int) -> int:
        """调用内缓存完整 span 的真实计数，缓存不跨正文或调用复用。"""
        key = first, last
        if key not in counts:
            counts[key] = _checked_count(counter, text[units[first].start:units[last].end])
        return counts[key]

    output: list[int] = []
    exceptions: list[tuple[int, int]] = []
    start = 0
    for end in (*cuts, len(units) - 1) if units else ():
        while tokens(start, end) > config.max_segment_tokens:
            allowed = [gap for gap in range(start, end) if ranks[gap] is not None]
            within = [gap for gap in allowed if tokens(start, gap) <= config.max_segment_tokens]
            if within:
                gap = max(within, key=lambda i: (ranks[i], decision.gaps[i].depth, -decision.gaps[i].cohesion, i))
            elif allowed:
                gap = allowed[0]
                exceptions.append((start, gap))
            else:
                exceptions.append((start, end))
                break
            output.append(gap)
            start = gap + 1
        start = end + 1
    return tuple(output), tuple(exceptions)


# 韩文填充符属字母类却不显示任何内容，排版时常被当作空行占位。
_FILLERS = frozenset("ᅟᅠㅤﾠ")


def _wordless(value: str) -> bool:
    """片段没有任何文字或数字，只有 emoji、符号、零宽字符或占位符。"""
    return not any(char.isalnum() and char not in _FILLERS for char in value)


def _rows(value: str) -> list[str]:
    """片段内去除两端空白后的非空行，仅供版式判断，不作输出正文。"""
    return [row for row in (part.strip("".join(WHITE)) for part in _LINES.split(value)) if row]


def _title_row(row: str) -> bool:
    """该行是题头形态且不含句子标点；“第二天：早上……。”这类带叙述的日程行不算。"""
    return _heading(row) is not None and not any(c in row for c in "，,。！？!?；;")


def _heading_only(value: str) -> bool:
    """片段像一个题头组：至多三行，均为题头形态或纯元数据，且至少一行题头。

    行数上限与题头组定义一致（题头加至多两行副题／栏目短行），避免把短行
    密集的整段正文当成题头。
    """
    rows = _rows(value)
    headings = [_title_row(row) for row in rows]
    return (0 < len(rows) <= 3 and any(headings)
            and all(heading or metadata_text(row) for heading, row in zip(headings, rows)))


def _closing_sentence(value: str) -> bool:
    """片段是一句普通叙述：不以符号、括号或编号开头，不是题头或“标签：内容”字段。

    收束语留在前节的前提是它本身只是叙述；单独成行的“✅路线：……”之类是另一个
    条目，不当作上一节的收束语合并。
    """
    rows = _rows(value)
    if not rows:
        return False
    first = rows[0]
    if _heading(first) is not None or _NUMBER.match(first) or re.match(r"^[^，,。！？!?；;]{1,12}[:：]", first):
        return False
    return unicodedata.category(first[0])[0] in {"L", "N"} and first[0] not in "【[〖"


def _merge_fragments(text: str, units: Sequence[TextUnit], cuts: Sequence[int], locked: set[int],
                     sections: set[int], config: CoherentSemanticConfig, counter: TokenCounter,
                     ) -> tuple[tuple[int, ...], dict[int, str]]:
    """撤销会留下不像人切的碎段的切点，返回保留切点与“撤销切点→原因”。

    四条规则，按顺序执行：①没有任何文字的片段（单个 emoji、箭头、分隔线、
    零宽字符）并入前段，位于开头时并入后段；②首段、末段非空白字符不足
    edge_segment_nonspace_chars 时分别并入后段、前段；③（1.3）只有题头、没有
    任何说明的片段并入后段，题头随后文；④（1.3）普通切点与随后的独立小节
    必切之间只剩一个完整句时，撤销该普通切点，收束句留在前节。正文中间有
    说明的短段不处理。locked 中的换天与独立小节切点从不撤销；合并后超过
    max_segment_tokens 的也不撤销，因此长度兜底切点与长度异常片段保持不变。
    sections 是已选独立小节切点。只删切点、不新增或移动切点。
    """
    kept = sorted(cuts)
    merged: dict[int, str] = {}

    def segment(position: int) -> str:
        """第 position 个片段的原文；片段由相邻保留切点围成。"""
        bounds = (-1, *kept, len(units) - 1)
        return text[units[bounds[position] + 1].start:units[bounds[position + 1]].end]

    def remove(position: int, reason: str) -> bool:
        """尝试撤销 kept[position]，即合并其两侧片段；不允许时返回 False。"""
        cut = kept[position]
        if cut in locked:
            return False
        bounds = (-1, *kept, len(units) - 1)
        first, last = bounds[position] + 1, bounds[position + 2]
        if _checked_count(counter, text[units[first].start:units[last].end]) > config.max_segment_tokens:
            return False
        merged[cut] = reason
        kept.pop(position)
        return True

    position = 0
    while kept and position <= len(kept):
        if _wordless(segment(position)):
            # 并入前段后当前位置变为下一个片段；并入后段后当前片段已有文字。
            if position and remove(position - 1, "merged_wordless_segment"):
                continue
            if position < len(kept) and remove(position, "merged_wordless_segment"):
                continue
        position += 1
    limit = config.edge_segment_nonspace_chars
    if kept and sum(char not in WHITE for char in segment(0)) < limit:
        remove(0, "merged_short_first_segment")
    if kept and sum(char not in WHITE for char in segment(len(kept))) < limit:
        remove(len(kept) - 1, "merged_short_last_segment")
    position = 0
    while position < len(kept):
        # 只在后段以正文开头时并入，题头才是在引出后文；并入后当前片段已含说明。
        following = _rows(segment(position + 1))
        if not (_heading_only(segment(position)) and following and not _title_row(following[0])
                and remove(position, "merged_heading_only_segment")):
            position += 1
    # 每个独立小节最多撤销其前一个普通切点一次；合并后的较长片段不再连锁并入。
    handled: set[int] = set()
    for position in range(len(kept) - 2, -1, -1):
        if kept[position + 1] not in sections or kept[position + 1] in handled:
            continue
        handled.add(kept[position + 1])
        bounds = (-1, *kept, len(units) - 1)
        start, end = units[bounds[position + 1] + 1].start, units[bounds[position + 2]].end
        if count_sentences(text, start, end) <= 1 and _closing_sentence(text[start:end]):
            remove(position, "merged_section_closure")
    return tuple(kept), merged


# 列表符号与话题标签不是题头标记；以它们开头的“标签行”多是清单条目或尾部标签。
_LIST_MARKS = frozenset("-·•*▪▫◦‧・#＃")
_OPEN_MARKS = frozenset("【[〖「『〔")
_KEYCAP = re.compile(r"^\d\ufe0f?\u20e3")


def _marked_heading(line: str) -> bool:
    """该行以题头标记开头：emoji 或符号、开方括号、编号、键帽数字或日程词。

    1.1 的标签行判断较宽，会把以 emoji 结尾的普通句子、清单条目也当作标签；
    这种宽判断用于否决切分是保守的，但用于上移切点会切进句子中间。上移只
    针对人一眼能认出的题头，例如“⭐️景点必去打卡”“四、行程安排”“[二R]渔家”。
    """
    body = line.lstrip("".join(WHITE))
    if not body or body[0] in _LIST_MARKS:
        return False
    if body[0] in _OPEN_MARKS or _NUMBER.match(body) or _KEYCAP.match(body) or _DAY.match(body):
        return True
    return unicodedata.category(body[0]) in ("So", "Sk")


def _before_label_lines(features: Sequence, labels: Sequence[bool], index: int) -> int | None:
    """返回标签行之前的换行 gap；index 是标签行之后、被否决的 gap。

    向上跨过紧邻的连续标签行（如“⭐️景点必去打卡”上面还有一行标签），
    找到第一处不是“标签行之后”的换行 gap。标签行位于帖首时没有可用位置，
    返回 None。只在既有 gap 上移动，不新增 gap；目标处是否允许切由调用者判断。
    """
    gap = index
    while True:
        unit = gap
        while unit > 0 and features[unit - 1].boundary_kind != "line":
            unit -= 1
        if unit == 0:
            return None
        if not labels[unit - 1]:
            return unit - 1
        gap = unit - 1


_COHERENT_KINDS = frozenset({"coherent_enumeration", "coherent_schedule", "coherent_directory", "coherent_item"})
# 以题头行自身为起点的“题头与首说明”保护块；与标签否决同源，不阻止切点上移到题头前。
_OWN_HEADING_KINDS = frozenset({"explicit_heading", "unmarked_heading", "directory_lead", "schedule_heading",
                                "repeated_entry_lead", "section_heading"})


def _sibling_groups(lines: Sequence[_Line], repeated: Sequence[StructureBlock]) -> tuple[frozenset[int], ...]:
    """把重复条目块按标记互指关系连成同形条目序列，元素为行号。

    discover_structure_blocks 的每个 repeated_entry 证据含自身标记及同族另一条目
    的标记；按互指关系合并即得同族全部条目。只以码点区间定位所在行，不重新
    识别版式；找不到所在行的标记忽略。
    """
    starts = [line.start for line in lines]
    parent: dict[int, int] = {}

    def line_of(position: int) -> int | None:
        """标记起点所在的非空行号。"""
        index = bisect_right(starts, position) - 1
        return index if index >= 0 and lines[index].start <= position < lines[index].end else None

    def root(item: int) -> int:
        """并查集根节点，带路径压缩。"""
        while parent.setdefault(item, item) != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for block in repeated:
        marks = {kind: line_of(start) for kind, start, _ in block.evidence if kind in {"repeated_marker", "matching_marker"}}
        own, other = marks.get("repeated_marker"), marks.get("matching_marker")
        if own is not None and other is not None:
            parent[root(own)] = root(other)
    groups: dict[int, set[int]] = {}
    for item in list(parent):
        groups.setdefault(root(item), set()).add(item)
    return tuple(frozenset(group) for group in groups.values() if len(group) >= 2)


# 可作为移动目标的题头形态；纯文字短行（unmarked）还须是没有被判为“不像标题”的小节候选。
_MOVE_KINDS = frozenset({"explicit_heading", "directory_lead", "schedule_heading"})
_NOT_TITLE = frozenset({"section_heading_narrative", "section_heading_too_long", "section_heading_sentence_fragment",
                        "section_heading_list_number", "section_heading_bullet_marker",
                        "section_heading_after_title_only", "section_heading_item_line"})
# 小节检查判为清单条目的原因码；这样的行不是标签小标题，也不触发标签上移。
_LIST_REASONS = frozenset({"section_heading_list_number", "section_heading_list_sequence",
                           "section_heading_item_line"})
# 纯文字短行只有在小节检查中确认为小节或子标题时，才能作为移动切点的目标。
_PLAIN_MOVE_REASONS = frozenset({"selected_section_heading", "section_heading_subsection"})
_SECTION_NUMBER = re.compile(r"^(?:[一二三四五六七八九十百]+、|[（(][一二三四五六七八九十百]+[）)])")


def _movable_heading(text: str, lines: Sequence[_Line], index: int, siblings: frozenset[int],
                     candidate_reasons: dict[int, str]) -> bool:
    """该行能否作为移动切点的标题：人一眼能认出的题头，而不是清单项、字段或参数。

    须为题头形态且不含句子标点（行尾配图指引除外）；同形条目、项目符号或阿拉伯
    数字编号开头的清单项，“标签：内容”字段和参数行不算；小节检查判为不像标题或
    清单序列的行不算；纯文字短行须已确认为独立小节或子标题，或是清单引导行——没有
    标记的短行常是清单项或换行截断的半句，不能凭形态把切点挪到它前面。
    """
    line = lines[index]
    value = text[line.start:line.end]
    heading = _heading(value)
    if heading is None or index in siblings or label_field(value) or parameter_text(value):
        return False
    body = value.lstrip("".join(WHITE))
    if body[:1] in "-·•*▪▫◦‧・" or (_NUMBER.match(body) and not _SECTION_NUMBER.match(body)):
        return False
    if any(c in _PICTURE_NOTE.sub("", body.rstrip("".join(WHITE))) for c in "，,。！？!?；;"):
        return False
    # 清单引导行（“……分别是：”）即使带判断词，也与其后的清单一体。
    if _list_lead(text, lines, index):
        return True
    reason = candidate_reasons.get(line.start)
    if reason in _NOT_TITLE or reason == "section_heading_list_sequence":
        return False
    if heading.kind in _MOVE_KINDS or marker_class(value) == "ordinal":
        return True
    return heading.kind == "unmarked_heading" and reason in _PLAIN_MOVE_REASONS


def _list_lead(text: str, lines: Sequence[_Line], index: int) -> bool:
    """以冒号结束、其后至少两行以同一标记开头的行是清单引导行（“图3到图5的建筑分别是：”）。

    引导行与它引出的清单是一体，切点可以移到引导行之前。
    """
    value = text[lines[index].start:lines[index].end].rstrip("".join(WHITE))
    if not value.endswith((":", "：")) or index + 2 >= len(lines):
        return False
    marks = [leading_mark(text[lines[i].start:lines[i].end]) for i in (index + 1, index + 2)]
    return marks[0] is not None and marks[0] == marks[1]


def _entry_start(text: str, lines: Sequence[_Line], index: int, siblings: frozenset[int],
                 candidate_reasons: dict[int, str]) -> bool:
    """该行开始一个新条目：可移动的标题，或同形条目、编号条目的首行（“2️⃣ 面馆”）。"""
    body = text[lines[index].start:lines[index].end].lstrip("".join(WHITE))
    return (_movable_heading(text, lines, index, siblings, candidate_reasons) or index in siblings
            or bool(_NUMBER.match(body)) or marker_class(body) == "numbered")


def _settle_boundaries(text: str, units: Sequence[TextUnit], lines: Sequence[_Line], siblings: frozenset[int],
                       candidate_reasons: dict[int, str], cuts: list[int], mandatory: set[int],
                       blocked: Sequence[bool], contrasts: Sequence[bool], reasons: list[list[str]],
                       evidence: list[list[tuple[str, int, int]]], line_gap) -> list[int]:
    """把放错位置的普通切点移到所属边界（1.3.1），就地修改 cuts，返回移动后的目标 gap。

    只处理落在行首的普通切点，按顺序检查三种情形，命中一种即止：
    ①前段末尾是标题，或标题加一行（至多一句）首说明，而后段不以标题开头：切点移到
    标题之前，标题随后文；后段已以标题开头时原切点正确，不动；
    ②后段开头是前一条目的参数行（人均、地址、价格……，连同紧随的“标签：内容”字段
    组成参数块）：参数块之后是标题或下一条目的首行时，切点移到该行之前；后段只有参数块时撤销该切点，
    参数留在前项；参数块之后不是标题时保留原位，不把参数块拆开；
    ③后段第一行只是一句已完成的普通叙述、第二行是标题：切点移到标题之前，收束句
    留在前节。
    ①先于②：前段以标题结束时，随后的参数属于该标题，应一起随后文。目标须是既有
    gap、不被禁止切否决、后句不以对比／解释词开头，且不越过相邻切点；不满足时保留
    原位。换天与独立小节必切（mandatory）不移动；本函数不新增切点。
    """
    line_starts = [line.start for line in lines]
    starts_at_line = set(line_starts)
    moved: list[int] = []

    def value(index: int) -> str:
        return text[lines[index].start:lines[index].end]

    for cut in [index for index in cuts if index not in mandatory]:
        position = cuts.index(cut)
        low = cuts[position - 1] if position else -1
        high = cuts[position + 1] if position + 1 < len(cuts) else len(units) - 1
        here = units[cut + 1].start
        if here not in starts_at_line:
            continue
        begin, finish = units[low + 1].start, units[high].end
        before = [i for i in range(bisect_left(line_starts, begin), bisect_left(line_starts, here))]
        after = [i for i in range(bisect_left(line_starts, here), bisect_left(line_starts, finish))]

        def allowed(target: int | None) -> bool:
            return target is not None and low < target < high and not blocked[target] and not contrasts[target]

        def move(target: int, why: str) -> None:
            reasons[cut].append(why)
            reasons[target].append("selected_moved_boundary")
            evidence[target].append(("moved_from_gap", units[cut].end, units[cut + 1].start))
            cuts.remove(cut)
            if target not in cuts:
                insort(cuts, target)
            moved.append(target)

        def heading(index: int) -> bool:
            return _movable_heading(text, lines, index, siblings, candidate_reasons)

        if not (after and heading(after[0])):
            done = False
            for back in (2, 1):
                if len(before) <= back or not heading(before[-back]):
                    continue
                if back == 2 and count_sentences(text, lines[before[-1]].start, lines[before[-1]].end) > 1:
                    continue
                target = line_gap(lines[before[-back]].start)
                if allowed(target):
                    move(target, "moved_before_heading")
                    done = True
                    break
            if done:
                continue
        count = 0
        if after and parameter_text(value(after[0])):
            while count < len(after) and (parameter_text(value(after[count])) or label_field(value(after[count]))):
                count += 1
        if count and not (before and heading(before[-1])):
            if count == len(after):
                reasons[cut].append("merged_trailing_parameters")
                cuts.remove(cut)
                continue
            if _entry_start(text, lines, after[count], siblings, candidate_reasons):
                target = line_gap(lines[after[count]].start)
                if allowed(target):
                    move(target, "moved_after_item_parameters")
                    continue
        if len(after) < 2:
            continue
        first = value(after[0])
        if (_closing_sentence(first) and _ends_sentence(first)
                and count_sentences(text, lines[after[0]].start, lines[after[0]].end) == 1
                and heading(after[1])):
            target = line_gap(lines[after[1]].start)
            if allowed(target):
                move(target, "moved_after_closing_sentence")
    return moved


def _section_candidates(text: str, units: Sequence[TextUnit]) -> tuple[SectionCandidate, ...]:
    """以本模块的题头识别和重复条目证据，交给 semantic_sections 做小节确认。"""
    lines = _lines(text)
    section_lines = tuple(SectionLine(line.start, line.end, heading.kind if heading else None)
                          for line in lines for heading in (_heading(text[line.start:line.end]),))
    repeated = [block for block in discover_structure_blocks(text, units) if block.kind == "repeated_entry"]
    return discover_sections(text, section_lines, _sibling_groups(lines, repeated))


def partition_by_meaning(text: str, units: Sequence[TextUnit], decision: SemanticDecision,
                         config: CoherentSemanticConfig, counter: TokenCounter) -> MeaningPartition:
    """按“禁止切＞换天与独立小节必切＞长度兜底＞普通边界”选择最终切点（1.3）。

    禁止切（一票否决，长度也不能推翻）：句内续接、同一行内的配对括号、题头与
    首说明、独立小节题头组及其首说明、纯标签小标题之后、尾部标签（含句末同行
    的标签），以及 semantic_relations 识别的强依赖词、未完结条件和括号说明。
    必切：叙述换天题头，以及 semantic_sections 确认的独立小节题头组之前；二者
    不要求语义曲线下降，碎段合并也不撤销。普通边界只来自过深度门槛（或绝对
    连贯度门槛）的语义谷值，并须满足：未被禁止切否决、后句不以对比／解释词
    开头、不在同栏目清单等连贯块内部。小标题、编号条目等结构形态本身不再
    放宽门槛。只被“纯标签小标题之后”否决的普通边界，若标签行以题头标记开头，
    切点上移到标签行之前（目标处同样须通过上述检查）。1.3.1另有两处变化：换天
    逐行识别（_day_transition_lines），不依赖结构提名；普通切点选定后由
    _settle_boundaries 把放错位置的切点移到所属边界（标题随后文、参数留前项、
    收束句留前节）。长度兜底只处理超限；最后由 _merge_fragments 撤销碎段、题头
    孤段和小节前收束句的切点。
    结构跨度没有准确已有 gap 时记录 unavailable，不创造单元。N=0/1仍检验原文和
    实际token；异常抛 ValueError，不返回半帖。
    """
    _validate_input(text, units, decision, config, counter)
    features = analyze_gap_structure(text, units)
    relations = analyze_gap_relations(text, units)
    contrasts = analyze_gap_continuations(text, units)
    coherent = discover_coherent_blocks(text, units)
    starts = {unit.start: index - 1 for index, unit in enumerate(units) if index}
    unit_starts = [unit.start for unit in units]
    lines = _lines(text)
    line_starts = [line.start for line in lines]
    repeated = [block for block in discover_structure_blocks(text, units) if block.kind == "repeated_entry"]
    sibling_lines = frozenset(member for group in _sibling_groups(lines, repeated) for member in group)
    day_starts, day_transitions = _day_transition_lines(text, lines)

    def line_gap(start: int) -> int | None:
        """行起点所在的 gap：右侧单元从该行开始（单元前导空白不计）；行在帖首或行中时为 None。"""
        index = bisect_right(unit_starts, start) - 1
        if index <= 0 or any(char not in WHITE for char in text[units[index].start:start]):
            return None
        return index - 1

    unavailable: list[tuple[str, int, int]] = []
    section_gaps: dict[int, SectionCandidate] = {}
    candidate_gaps: dict[int, SectionCandidate] = {}
    candidate_reasons: dict[int, str] = {}
    section_blocks: list[StructureBlock] = []
    for candidate in _section_candidates(text, units):
        candidate_reasons[candidate.start] = candidate.reason
        gap = starts.get(candidate.start)
        if candidate.protected:
            # 题头组与首个完整说明之间禁止切，题头不能挂在上一节末尾或单独成段。
            section_blocks.append(StructureBlock(candidate.start, max(candidate.first_end, candidate.group_end),
                                                 "section_heading", candidate.evidence))
        if not candidate.confirmed:
            if gap is not None:
                candidate_gaps[gap] = candidate
        elif gap is None:
            unavailable.append(("section_heading", candidate.start, candidate.group_end))
        else:
            section_gaps[gap] = candidate
    blocks = tuple(sorted(set((*_meaning_blocks(text, units), *coherent, *section_blocks)),
                          key=lambda block: (block.start, block.end, block.kind)))
    structural: dict[int, list[StructureBlock]] = {}
    for block in blocks:
        if (block.kind in {"trailing_metadata_context", "syntactic_continuation", "section_heading"}
                or block.start == (units[0].start if units else 0)):
            continue
        gap = starts.get(block.start)
        if gap is None:
            unavailable.append((block.kind, block.start, block.end))
        else:
            structural.setdefault(gap, []).append(block)
    # 句末同行的尾部标签（如“……。 #话题#”）不成整行，行级尾标签识别看不到；
    # 末尾连续的纯标签／账号／链接单元一律附着前文，与 A 层尾标签规则一致。
    meta_tail = len(units)
    while meta_tail > 1 and metadata_text(text[units[meta_tail - 1].start:units[meta_tail - 1].end]):
        meta_tail -= 1
    blocked, ranks, reasons, evidence, sources, labels, label_only, in_list = [], [], [], [], [], [], [], []
    after_separator: list[bool] = []
    original = set(decision.semantic_boundaries)
    absolute = {index for index, feature in enumerate(features)
                if feature.natural_boundary and decision.gaps[index].cohesion <= config.absolute_cohesion_threshold}
    nominated = original | absolute
    for index, feature in enumerate(features):
        active = [block for block in blocks if block.start < units[index].end < block.end]
        # 连贯块不否决切分，只让普通边界不选；真正否决的是题头附着、尾标签和句法续接。
        protecting = [block for block in active if block.kind not in _COHERENT_KINDS]
        in_list.append(len(protecting) < len(active))
        local = list(feature.reasons)
        local.extend(f"inside_{block.kind}" for block in active)
        local.extend(relations[index])
        internal = feature.boundary_kind == "internal"
        # 跨行配对的括号多半是引号错配，只保护同一行内的括号说明，避免一个
        # 未闭合引号锁住数千字。
        bracket = any(kind == "inside_matched_bracket" and not _LINES.search(text, start, end)
                      for kind, start, end in feature.evidence_spans)
        if "inside_matched_bracket" in feature.reasons and not bracket:
            local.append("cross_line_bracket_ignored")
        label = feature.boundary_kind == "line" and _label_heading_line(text, units[index].end)
        if label:
            local.append("after_label_heading")
        separated = feature.boundary_kind == "line" and not label and _label_above_separator(text, units[index].end)
        if separated:
            local.append("after_label_heading_separator")
        tail = index + 1 >= meta_tail
        if tail:
            local.append("before_trailing_metadata")
        veto = bracket or label or tail or bool(protecting) or bool(relations[index])
        # 换行不能将句内逗号/顿号/冒号变成独立意思；原文断行和模型断句
        # 都服从同一续接门，长度兜底同样不能越过。
        suffix = units[index].end
        while suffix > units[index].start and text[suffix - 1] in WHITE:
            suffix -= 1
        continuation = suffix > units[index].start and text[suffix - 1] in "，,、：:"
        if continuation:
            local.append("punctuation_continuation")
            veto = True
        # 无标点长文常以空白代替标点；这类句内位置不供语义或小标题选择，
        # 只在长度兜底找不到换行和句末时作为最后的允许位置。逗号续接仍否决。
        spaced = internal and any(c in WHITE for c in text[units[index].end:units[index + 1].start])
        if spaced and not veto:
            local.append("space_separated_fragment")
        blocked.append(veto or internal)
        labels.append(label or separated)
        after_separator.append(separated and not veto)
        # 只被“纯标签小标题之后”否决的提名，切点可以上移到标签行之前；
        # 同时触发其他禁止切规则的位置仍直接否决。1.3.1起，标签行自身的“题头与首说明”
        # 保护与标签否决同源，也允许上移，不再因此丢掉这个分界。
        own_start = 0
        if label or separated:
            breaks = [match.end() for match in _LINES.finditer(text, 0, units[index].end)]
            own_start = breaks[-1] if breaks else 0
            while own_start < units[index].end and text[own_start] in WHITE:
                own_start += 1
        foreign = [block for block in protecting
                   if not (block.kind in _OWN_HEADING_KINDS and block.start == own_start)]
        label_only.append((label or separated)
                          and not (bracket or tail or foreign or relations[index] or continuation))
        ranks.append(None if veto or (internal and not spaced) else
                     2 if feature.boundary_kind == "line" else 1 if feature.boundary_kind == "sentence" else 0)
        reasons.append(local)
        evidence.append([("left_scored_unit", units[index].start, units[index].end),
                         ("right_scored_unit", units[index + 1].start, units[index + 1].end),
                         *feature.evidence_spans, *(item for block in active for item in block.evidence)])
        if continuation:
            evidence[-1].append(("punctuation_continuation", suffix - 1, suffix))
        sources.append([*( ["local_semantic"] if index in original else []),
                        *( ["natural_absolute_cohesion"] if index in absolute else []),
                        *(f"before_{block.kind}" for block in structural.get(index, [])),
                        *( ["before_section_heading"] if index in section_gaps else [])])
        evidence[-1].extend(item for block in structural.get(index, []) for item in block.evidence)
        if index in section_gaps or index in candidate_gaps:
            evidence[-1].extend((section_gaps.get(index) or candidate_gaps[index]).evidence)
    cuts: list[int] = []
    structural_targets: set[int] = set()

    def ordinary(index: int) -> bool:
        """普通边界的附加条件：后句不以对比／解释词开头，且不在连贯块内部。"""
        if contrasts[index]:
            reasons[index].append("rejected_open_contrast_or_reason")
            return False
        if in_list[index]:
            reasons[index].append("rejected_inside_coherent_block")
            return False
        return True

    def shift(index: int, day: bool) -> int | None:
        """把只因标签行被否决的提名上移到标签行之前；人会让标签随后文，而不是丢掉这个分界。"""
        if not label_only[index]:
            return None
        target = _before_label_lines(features, labels, index)
        if target is None or blocked[target]:
            reasons[index].append("shift_before_label_heading_unavailable")
            return None
        first_line = _LINES.split(text[units[target + 1].start:], 1)[0]
        if not _marked_heading(first_line):
            reasons[index].append("shift_before_label_heading_unmarked")
            return None
        # 同形条目、成串项目符号或小节检查判为清单条目的一项是清单内容，不是标签
        # 小标题；不因它把普通切点上移进清单（如打分清单“9.0分××｜……”），原否决
        # 照常生效。换天必切不受此限，仍上移到标签行之前（1.3.1）。
        head = units[target + 1].start
        while head < len(text) and text[head] in WHITE:
            head += 1
        row = bisect_right(line_starts, head) - 1
        if not day and row >= 0 and (row in sibling_lines or _list_item_line(text, lines[row].start, lines[row].end)
                         or candidate_reasons.get(lines[row].start) in _LIST_REASONS):
            reasons[index].append("shift_before_label_heading_list_item")
            return None
        if not day and target not in cuts and not ordinary(target):
            reasons[index].append("shift_before_label_heading_rejected")
            return None
        reasons[index].append("shifted_before_label_heading")
        reasons[target].append("selected_before_label_heading")
        if day:
            reasons[target].append("selected_day_transition")
        sources[target].append("shifted_from_label_heading")
        evidence[target].append(("shifted_from_gap", units[index].end, units[index + 1].start))
        if target not in cuts:
            insort(cuts, target)
        return target

    for index, candidate in candidate_gaps.items():
        reasons[index].append(candidate.reason)
    for index in sorted(section_gaps):
        if blocked[index]:
            reasons[index].append("rejected_section_dependency")
            continue
        insort(cuts, index)
        structural_targets.add(index)
        reasons[index].append("selected_section_heading")
    def day(start: int) -> bool:
        """结构提名起点是否为真实换天：行首日程按逐行检查结果，行尾附带日程沿用题头识别。"""
        return start in day_transitions if start in day_starts else _day_heading(text, start)

    day_handled: set[int] = set()
    for index in sorted(structural):
        if blocked[index]:
            reasons[index].append("rejected_structure_dependency")
            if any(day(block.start) for block in structural[index]):
                day_handled.add(index)
                target = shift(index, True)
                if target is not None:
                    structural_targets.add(target)
        elif any(day(block.start) for block in structural[index]):
            day_handled.add(index)
            if index not in cuts:
                insort(cuts, index)
            reasons[index].append("selected_day_transition")
        elif index not in nominated and index not in section_gaps:
            # 1.3取消“小标题处语义略降即切”：结构形态须同时是过门槛的语义谷值。
            reasons[index].append("rejected_structure_below_valley")
    # 1.3.1：换天不依赖结构提名。日程题头下一行是编号条目、【】标题，或本行较长时，
    # 结构分析可能不提名该处，此前这些换天会漏切；这里逐行补齐，同样服从禁止切。
    for start in sorted(day_transitions):
        index = line_gap(start)
        if index is None:
            unavailable.append(("day_transition", start, start))
            continue
        if index in day_handled or "selected_day_transition" in reasons[index]:
            continue
        sources[index].append("day_transition_scan")
        if blocked[index]:
            reasons[index].append("rejected_day_dependency")
            target = shift(index, True)
            if target is not None:
                structural_targets.add(target)
            continue
        if index not in cuts:
            insort(cuts, index)
        structural_targets.add(index)
        reasons[index].append("selected_day_transition")
    semantic: list[int] = []
    mandatory = set(cuts)
    for index in sorted(nominated):
        if blocked[index]:
            reasons[index].append("rejected_semantic_dependency")
            target = shift(index, False)
            if target is not None and target not in semantic:
                semantic.append(target)
            continue
        if index not in mandatory and not ordinary(index):
            continue
        # “题头＋分隔线”之后的普通切点上移到题头之前；上移不成时保留原位。
        if index not in mandatory and after_separator[index]:
            target = shift(index, False)
            if target is not None:
                if target not in semantic:
                    semantic.append(target)
                continue
        semantic.append(index)
        reasons[index].append("selected_local_semantic")
        if index not in cuts:
            insort(cuts, index)
    semantic.extend(_settle_boundaries(text, units, lines, sibling_lines, candidate_reasons, cuts, mandatory,
                                       blocked, contrasts, reasons, evidence, line_gap))
    lengths, exception_spans = _length_partition(text, units, decision, config, counter, cuts, ranks)
    sections = {index for index in section_gaps if "selected_section_heading" in reasons[index]}
    locked = sections | {index for index in cuts if "selected_day_transition" in reasons[index]}
    final, merged = _merge_fragments(text, units, (*cuts, *lengths), locked, sections, config, counter)
    for index, reason in merged.items():
        reasons[index].append(reason)
    selected = set(final)
    semantic = sorted(index for index in set(semantic) if index in selected)
    structural_selected = tuple(index for index in cuts
                                if (index in structural or index in structural_targets) and index in selected)
    lengths = tuple(index for index in lengths if index in selected)
    segments, first = [], 0
    exception_ids: list[int] = []
    for last in (*sorted(selected), len(units) - 1) if units else ():
        start, end = units[first].start, units[last].end
        value = text[start:end]
        segments.append(TextSegment(len(segments) + 1, value, start, end))
        if _checked_count(counter, value) > config.max_segment_tokens:
            if (first, last) not in exception_spans:
                raise ValueError("meaning_length_invariant_failed")
            exception_ids.append(len(segments))
        first = last + 1
    validate(text, [asdict(segment) for segment in segments])
    for index in range(len(features)):
        if index in lengths:
            sources[index].append("length_fallback")
            reasons[index].append("selected_length_fallback")
        elif index not in selected and not sources[index]:
            reasons[index].append("not_nominated")
            reasons[index].append("upstream_" + decision.gaps[index].reason)
    traces = tuple(MeaningGapDecision(index, index in selected, tuple(dict.fromkeys(sources[index])),
                  tuple(dict.fromkeys(reasons[index])), tuple(dict.fromkeys(evidence[index]))) for index in range(len(features)))
    return MeaningPartition(tuple(segments), tuple(sorted(semantic)), structural_selected, lengths,
                            tuple(sorted(nominated - set(semantic))), tuple(exception_ids),
                            traces, blocks, tuple(unavailable))
