"""分层意思分区核心：规则决定禁止切与必须切，语义模型只在灰色地带选择。

本模块仅做纯计算；不访问文件、数据库、帖子身份、模型或向量缓存，不新增
单元或伪造 embedding。调用者先用原有评分函数生成 SemanticDecision；窗口
和阈值由调用者明确保存。优先级为：禁止切＞换天必切＞长度兜底＞灰色地带。
句号、属性变化本身不会触发切分；小标题只在语义曲线有下降时切开。最后一步
只合并碎段：无文字的片段和过短的首末段并入相邻段。运行结果仍须抽查，不能
自动证明所有片段均符合人的切分直觉。无法在既有 gap 上表达的意思转换保留
为限制，不能移动到句内相邻位置代替。
"""

from __future__ import annotations

from bisect import insort
from dataclasses import asdict, dataclass
from math import isfinite
import re
from typing import Sequence
import unicodedata

from .contracts import SemanticDecision, TextSegment, TextUnit, TokenCounter
from .integrity import WHITE, validate
from .semantic_config import CoherentSemanticConfig
from .semantic_coherence import discover_coherent_blocks
from .semantic_relations import analyze_gap_relations
from .semantic_structure import (
    StructureBlock, analyze_gap_structure, discover_context_blocks,
    discover_structure_blocks,
)

EXPERIMENT_ID = "meaning-layers-1"
_LINES = re.compile(r"\r\n|[\r\n\u2028\u2029]")
_DAY = re.compile(r"(?:day\s*\d+|d\s*\d+|第[0-9一二三四五六七八九十百]+天)(?![0-9])", re.I)
_DATE = re.compile(r"(?:\d{2,4}年)?\d{1,2}月\d{1,2}日")
_ATTRIBUTE = re.compile(r"^(?:交通|地址|位置|导航|时间|时长|开放时间|营业时间|门票|票价|价格|费用|预约|停车|电话|优点|缺点|亮点)\s*[:：|｜]")
_NUMBER = re.compile(r"^(?:\d+[、．.）)]|[一二三四五六七八九十百]+[、．.]|[（(][一二三四五六七八九十百\d]+[）)])")
_STRONG = frozenset("。！？!?；;")
_PAIRS = {"(": ")", "（": "）", "[": "]", "【": "】", "{": "}", "「": "」", "『": "』", "“": "”", "‘": "’", "《": "》"}
_CLOSERS = frozenset("”’」』)）】]》〉\"'")


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
    """只在识别副本上去行首装饰，绝不将此副本用作输出正文。"""
    cursor = 0
    while cursor < len(value) and (value[cursor] in WHITE or unicodedata.category(value[cursor])[0] in {"S", "P", "M"}):
        cursor += 1
    return value[cursor:]


def _metadata(value: str) -> bool:
    """整段纯标签、账号、链接或装饰才是元数据，不吞正文里的话题词。"""
    stripped = re.sub(r"https?://\S+|#[^\s#]+#?|@\S+", "", value)
    return bool(value.strip()) and not any(unicodedata.category(c)[0] in {"L", "N"} for c in stripped)


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
    if not plain or _metadata(value) or _ATTRIBUTE.match(plain):
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
    if any(c in core for c in "，,。！？!?；;") or not core:
        return None
    if re.match(r"^(?:我(?:们)?|你(?:们)?|他(?:们)?|她(?:们)?|大家|今天|昨天|明天|可以|需要|应该|建议|不能|不用|无需|一定)", core):
        return None
    if re.search(r"没有|很|非常|十分|太|已经|正在|觉得|喜欢|值得|[了着](?:$|[的呢呀吧])", core):
        return None
    explicit = plain != value or bool(_NUMBER.match(value)) or core.endswith(("：", ":"))
    # ‘晚上：地点’等短冒号标签是显式小题头，普通价格/交通字段已排除。
    explicit = explicit or bool(re.fullmatch(r"[^:：]{1,12}[:：][^:：]{1,20}", core))
    return _Heading("explicit_heading" if explicit else "unmarked_heading", explicit)


def _first_complete_end(text: str, start: int, end: int) -> int:
    """取首个括号外完整句；没有强句末时完整保留该原文行，不按逗号拆。

    极短但完整说明可成为首说明，不为了累计字符跨下一明确题头。
    闭符和组合符附着在句末，保护区间端点不会落入其组合字素内部。
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


def _trailing_metadata_block(text: str, lines: Sequence[_Line]) -> StructureBlock | None:
    """将连续纯元数据尾行绑定到前一实质行的最后一句，坐标始终取原文。

    标签、账号、链接和装饰均沿用本模块的整行识别，不吞含普通陈述的行。
    原句可以横跨多个模型单元；后续硬长度图禁止只含这些尾行的末段。
    全文只有元数据时没有可依附的正文，返回 None，不虚构实质句证据。
    """
    before = len(lines) - 1
    while before >= 0 and _metadata(text[lines[before].start:lines[before].end]):
        before -= 1
    if before < 0 or before == len(lines) - 1:
        return None
    line = lines[before]
    cursor, prior = line.start, None
    while cursor < line.end:
        stop = _first_complete_end(text, cursor, line.end)
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
            stop = _first_complete_end(text, line.start, line.end)
            output.append(StructureBlock(line.start, stop, kind, (*evidence, ("first_substantive_sentence", line.start, stop))))
            continue
        if index + 1 == len(lines):
            continue
        body_index = index + 1
        following = headings[body_index]
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
        if _metadata(body_value):
            continue
        if not heading.explicit and (len(body_value) < 24 or len(body_value) <= len(value)):
            continue
        stop = _first_complete_end(text, body.start, body.end)
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


def _label_heading_line(text: str, end: int) -> bool:
    """判断以 end 结束的整行是否为明确的纯标签小标题，其后正文不能被切走。

    只认 _heading 判定的显式题头、日程题头或行尾带装饰的短题头（如“交通🅿️”“▪️游玩路线”
    “🌷Day2 金沙滩对比实测”）；“晚上：河湾灯景”这类“标签：内容”行本身就是
    内容，不受此限制，后面遇到换天仍可切开。无标记短行不在此列，避免把
    普通短句误当题头而少切。
    """
    starts = [match.end() for match in _LINES.finditer(text, 0, end)]
    line = text[starts[-1] if starts else 0:end]
    heading = _heading(line)
    if heading is None:
        return False
    # “交通🅿️”“美食🍽️”以行尾装饰标示题头，与行首装饰同等视为明确标签。
    core = line.rstrip()
    while core and (core[-1] in WHITE or core[-1] == "\ufe0f" or unicodedata.category(core[-1])[0] in {"S", "M"}):
        core = core[:-1]
    decorated = bool(core) and core != line.rstrip()
    if not (heading.explicit or heading.kind == "trailing_schedule_heading" or decorated):
        return False
    return not re.search(r"[:：]\s*\S", _plain(line))


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


def _merge_fragments(text: str, units: Sequence[TextUnit], cuts: Sequence[int], locked: set[int],
                     config: CoherentSemanticConfig, counter: TokenCounter,
                     ) -> tuple[tuple[int, ...], dict[int, str]]:
    """撤销会留下不像人切的碎段的切点，返回保留切点与“撤销切点→原因”。

    两条规则，按顺序执行：没有任何文字的片段（单个 emoji、箭头、分隔线、
    零宽字符）并入前段，位于开头时并入后段；首段、末段非空白字符不足
    edge_segment_nonspace_chars 时分别并入后段、前段。正文中间的短段不处理，
    因为小标题和单句感受常是真正的意思单位。locked 中的换天切点从不撤销；
    合并后超过 max_segment_tokens 的也不撤销，因此长度兜底切点与长度异常
    片段保持不变。只删切点、不新增或移动切点。
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
    return tuple(kept), merged


_COHERENT_KINDS = frozenset({"coherent_enumeration", "coherent_schedule", "coherent_directory", "coherent_item"})


def partition_by_meaning(text: str, units: Sequence[TextUnit], decision: SemanticDecision,
                         config: CoherentSemanticConfig, counter: TokenCounter) -> MeaningPartition:
    """按“禁止切＞换天必切＞长度兜底＞灰色地带”的优先级选择最终切点。

    禁止切（一票否决，长度也不能推翻）：句内续接、同一行内的配对括号、题头与
    首说明、纯标签小标题之后、尾部标签（含句末同行的标签），以及
    semantic_relations 识别的强依赖词、未完结条件和括号说明。
    句内位置不供语义或小标题选择；无标点长文中以空白分隔的位置只作为长度
    兜底的最后手段。
    换天必切：叙述中的日程题头只要未被禁止即切开。灰色地带：其余结构提名
    （小标题、连贯块起点）只在该处语义曲线有下降（depth>epsilon）时切开；
    语义谷值按深度门槛提名后直接采纳，不设全局最短段门槛。连贯块（完整介绍、
    组成列表、同日推荐）只作证据保存，不再阻止切分。所有切点定下后，再由
    _merge_fragments 撤销留下无文字碎段或过短首末段的切点（换天切点除外）。结构跨度没有准确已有
    gap 时记录 unavailable，不创造单元。N=0/1仍检验原文和实际token；异常
    抛 ValueError，不返回半帖。
    """
    _validate_input(text, units, decision, config, counter)
    features = analyze_gap_structure(text, units)
    relations = analyze_gap_relations(text, units)
    coherent = discover_coherent_blocks(text, units)
    blocks = tuple(sorted(set((*_meaning_blocks(text, units), *coherent)),
                          key=lambda block: (block.start, block.end, block.kind)))
    starts = {unit.start: index - 1 for index, unit in enumerate(units) if index}
    structural: dict[int, list[StructureBlock]] = {}
    unavailable: list[tuple[str, int, int]] = []
    for block in blocks:
        if block.kind in {"trailing_metadata_context", "syntactic_continuation"} or block.start == (units[0].start if units else 0):
            continue
        gap = starts.get(block.start)
        if gap is None:
            unavailable.append((block.kind, block.start, block.end))
        else:
            structural.setdefault(gap, []).append(block)
    # 句末同行的尾部标签（如“……。 #话题#”）不成整行，行级尾标签识别看不到；
    # 末尾连续的纯标签／账号／链接单元一律附着前文，与 A 层尾标签规则一致。
    meta_tail = len(units)
    while meta_tail > 1 and _metadata(text[units[meta_tail - 1].start:units[meta_tail - 1].end]):
        meta_tail -= 1
    blocked, ranks, reasons, evidence, sources = [], [], [], [], []
    original = set(decision.semantic_boundaries)
    absolute = {index for index, feature in enumerate(features)
                if feature.natural_boundary and decision.gaps[index].cohesion <= config.absolute_cohesion_threshold}
    nominated = original | absolute
    for index, feature in enumerate(features):
        active = [block for block in blocks if block.start < units[index].end < block.end]
        # 连贯块只留作证据；真正否决切分的是题头附着、尾标签和句法续接。
        protecting = [block for block in active if block.kind not in _COHERENT_KINDS]
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
                        *(f"before_{block.kind}" for block in structural.get(index, []))])
        evidence[-1].extend(item for block in structural.get(index, []) for item in block.evidence)
    cuts: list[int] = []
    for index in sorted(structural):
        if blocked[index]:
            reasons[index].append("rejected_structure_dependency")
        elif any(_day_heading(text, block.start) for block in structural[index]):
            cuts.append(index)
            reasons[index].append("selected_day_transition")
        elif decision.gaps[index].depth > config.epsilon:
            cuts.append(index)
            reasons[index].append("selected_structure_with_semantic_dip")
        else:
            reasons[index].append("rejected_structure_without_semantic_dip")
    semantic: list[int] = []
    for index in sorted(nominated):
        if blocked[index]:
            reasons[index].append("rejected_semantic_dependency")
            continue
        semantic.append(index)
        reasons[index].append("selected_local_semantic")
        if index not in cuts:
            insort(cuts, index)
    lengths, exception_spans = _length_partition(text, units, decision, config, counter, cuts, ranks)
    locked = {index for index in cuts if "selected_day_transition" in reasons[index]}
    final, merged = _merge_fragments(text, units, (*cuts, *lengths), locked, config, counter)
    for index, reason in merged.items():
        reasons[index].append(reason)
    selected = set(final)
    semantic = [index for index in semantic if index in selected]
    structural_selected = tuple(index for index in cuts if index in structural and index in selected)
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
