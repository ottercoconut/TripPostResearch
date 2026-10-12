"""experimental：在既有单元及评分上尝试独立意思边界，不继承正式规则身份。

本模块仅做纯计算；不访问文件、数据库、帖子身份、模型或向量缓存，不新增
单元或伪造 embedding。调用者先用原有评分函数生成 SemanticDecision；窗口
和阈值由调用者明确保存。结构题头可以主动提出边界，但句号、属性变化本身
不会触发切分。结果仍是待人工复核的开发候选，不是独立可编码性的自动证明。
无法在既有 gap 上表达的意思转换保留为限制，不能移动到句内相邻位置代替。
"""

from __future__ import annotations

from bisect import bisect_left, insort
from dataclasses import asdict, dataclass
from math import isfinite
import re
from typing import Sequence
import unicodedata

from .contracts import SemanticDecision, TextSegment, TextUnit
from .integrity import WHITE, validate
from .semantic_config import SemanticConfig
from .semantic_coherence import discover_coherent_blocks
from .semantic_partition import TokenCounter
from .semantic_structure import (
    StructureBlock, analyze_gap_structure, discover_context_blocks,
    discover_structure_blocks,
)

EXPERIMENT_ID = "experimental-meaning-units-4"
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
    最小长度及硬长度覆盖。evidence_spans 使用原文码点左闭右开坐标，
    包含评分所对应的左右单元，避免仅给不可核验的抽象原因。
    """

    index: int
    selected: bool
    sources: tuple[str, ...]
    reason_codes: tuple[str, ...]
    evidence_spans: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class MeaningPartition:
    """完整实验结果；边界元组均为 gap 索引，保护块则为原文码点。

    结构与语义来源可在同一个切点重叠，length_boundaries 只记录额外长度切点。
    forced_structure_boundaries 明示硬上限导致的依赖破坏；不会默默弱化规则。
    所有字段不可变，dataclasses.asdict 可用于调用方的实验记录。
    """

    segments: tuple[TextSegment, ...]
    semantic_boundaries: tuple[int, ...]
    structural_boundaries: tuple[int, ...]
    length_boundaries: tuple[int, ...]
    suppressed_semantic_boundaries: tuple[int, ...]
    forced_structure_boundaries: tuple[int, ...]
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
                    config: SemanticConfig, counter: TokenCounter) -> None:
    """沿用公共原文与字素验收，并拒绝坏评分身份、非有限数和虚假计数。"""
    if not isinstance(text, str) or not isinstance(config, SemanticConfig):
        raise ValueError("invalid_meaning_input")
    SemanticConfig.__post_init__(config)
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


def _length_partition(text: str, units: Sequence[TextUnit], decision: SemanticDecision,
                      config: SemanticConfig, counter: TokenCounter, cuts: Sequence[int],
                      penalties: Sequence[tuple[int, int, int]], metadata_start: int | None) -> tuple[int, ...]:
    """对固定意思段执行全局硬长度分区，精确重计每一个可行原文区间。

    排序目标依次为一般依赖破坏罚分、完整条目破坏数、连贯父块破坏数、
    短段数、总段数、已有语义代价。三个依赖层级分别累计，不能用少切一次
    父块抵消切开完整条目的损失，括号和句内续接又优先于条目完整性。BPE计数
    不必随前缀单调，因此首个超限不停止扫描。纯元数据末段不进入可行图；
    无路径时整帖抛错，既不截断也不通过超长输出回避硬上限。
    """
    counts: dict[tuple[int, int], int] = {}
    prefix = [0]
    for char in text:
        prefix.append(prefix[-1] + (char not in WHITE))

    def tokens(first: int, last: int) -> int:
        """调用内缓存完整 span 的真实计数，缓存不跨正文或调用复用。"""
        key = first, last
        if key not in counts:
            counts[key] = _checked_count(counter, text[units[first].start:units[last].end])
        return counts[key]

    output: list[int] = []
    start = 0
    for end in (*cuts, len(units) - 1) if units else ():
        if tokens(start, end) <= config.max_segment_tokens:
            start = end + 1
            continue
        dp = {start: ((0, 0, 0, 0, 0, 0.0), ())}
        for stop in range(start + 1, end + 2):
            candidates = []
            for first in range(start, stop):
                if first not in dp or tokens(first, stop - 1) > config.max_segment_tokens:
                    continue
                if metadata_start is not None and units[first].start >= metadata_start:
                    continue
                previous, path = dp[first]
                internal, gap = stop <= end, stop - 1
                short = prefix[units[stop - 1].end] - prefix[units[first].start] < config.min_segment_nonspace_chars
                score = decision.gaps[gap].cohesion - decision.gaps[gap].depth if internal else 0.0
                damage = penalties[gap] if internal else (0, 0, 0)
                cost = (previous[0] + damage[0], previous[1] + damage[1],
                        previous[2] + damage[2], previous[3] + int(short),
                        previous[4] + 1, previous[5] + score)
                candidates.append((cost, (*path, gap) if internal else path))
            if candidates:
                dp[stop] = min(candidates)
        if end + 1 not in dp:
            raise ValueError("unsplittable_meaning_segment_limit")
        output.extend(dp[end + 1][1])
        start = end + 1
    return tuple(output)


def partition_by_meaning(text: str, units: Sequence[TextUnit], decision: SemanticDecision,
                         config: SemanticConfig, counter: TokenCounter) -> MeaningPartition:
    """以已评分 gap 构造实验意思片段，返回无损原文与逐 gap 选择理由。

    连贯父块优先保留完整介绍／推荐，子条目可独立编码不代表必须拆开。
    结构边界先选，局部语义边界随后按既有深度排序检查最小段长；除了上游
    已选点，所有自然 gap 在 cohesion 不高于配置绝对阈值时也获得提名，
    不要求它恰好成为局部谷值。阈值仍是开发参数，不代表语义质量保证。明确题头
    及其完整说明允许短于软字符门槛，不因长度短就跨对象合并。句内逗号、
    配对括号、题头首说明和尾标签附着抑制软切点；硬上限只在没有更好路径
    时显式退让。结构跨度没有准确已有 gap 时记录 unavailable，不创造单元。
    N=0/1仍检验原文和实际token；异常抛 ValueError，不返回半帖。
    """
    _validate_input(text, units, decision, config, counter)
    features = analyze_gap_structure(text, units)
    coherent = discover_coherent_blocks(text, units)
    # 父块阻止组内结构与语义提名；完整子条目只增加硬长度退让时的代价，
    # 使可行的条目间切点优先于条目内部。超长单项仍可退让且保留trace。
    items = tuple(StructureBlock(start, end, "coherent_item",
                  (("coherent_parent", block.start, block.end), (kind, start, end)))
                  for block in coherent for kind, start, end in block.evidence
                  if kind == "coherent_item")
    blocks = tuple(sorted(set((*_meaning_blocks(text, units), *coherent, *items)),
                          key=lambda block: (block.start, block.end, block.kind)))
    starts = {unit.start: index - 1 for index, unit in enumerate(units) if index}
    structural: dict[int, list[StructureBlock]] = {}
    unavailable: list[tuple[str, int, int]] = []
    for block in blocks:
        if block.kind in {"trailing_metadata_context", "syntactic_continuation", "coherent_item"} or block.start == (units[0].start if units else 0):
            continue
        gap = starts.get(block.start)
        if gap is None:
            unavailable.append((block.kind, block.start, block.end))
        else:
            structural.setdefault(gap, []).append(block)
    metadata = [span[1] for block in blocks if block.kind == "trailing_metadata_context"
                for span in block.evidence if span[0] == "trailing_metadata"]
    metadata_start = min(metadata) if metadata else None
    length_penalties, blocked, reasons, evidence, sources = [], [], [], [], []
    original = set(decision.semantic_boundaries)
    absolute = {index for index, feature in enumerate(features)
                if feature.natural_boundary and decision.gaps[index].cohesion <= config.absolute_cohesion_threshold}
    nominated = original | absolute
    for index, feature in enumerate(features):
        active = [block for block in blocks if block.start < units[index].end < block.end]
        # 旧版 after_heading 来自单个模型单元，可能只是完整句被短切后的
        # 假题头；本候选改用完整原文行与首说明块。属性字段仅留证据，不
        # 一律否决交通/价格等独立说明，也不凭属性词主动选点。
        local = list(feature.reasons)
        local.extend(f"inside_{block.kind}" for block in active)
        penalty = (8 if feature.boundary_kind == "internal" else 0)
        penalty += 12 * int("inside_matched_bracket" in feature.reasons) + 12 * len(active)
        # 换行不能将句内逗号/顿号/冒号变成独立意思；原文断行和模型断句
        # 都服从同一续接门。硬长度仍可退让，但必须在最终trace显式标记。
        suffix = units[index].end
        while suffix > units[index].start and text[suffix - 1] in WHITE:
            suffix -= 1
        continuation = suffix > units[index].start and text[suffix - 1] in "，,、：:"
        if continuation:
            local.append("punctuation_continuation")
            penalty += 8
        coherent_items = sum(block.kind == "coherent_item" for block in active)
        coherent_parents = sum(block.kind in {"coherent_enumeration", "coherent_schedule",
                                              "coherent_directory"} for block in active)
        dependency = penalty - 12 * (coherent_items + coherent_parents)
        length_penalties.append((dependency, coherent_items, coherent_parents))
        blocked.append(penalty > 0)
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
        else:
            cuts.append(index)
            reasons[index].append("selected_explicit_structure")
    semantic: list[int] = []
    for index in sorted(nominated, key=lambda i: (-decision.gaps[i].depth, decision.gaps[i].cohesion, i)):
        if blocked[index]:
            reasons[index].append("rejected_semantic_dependency")
            continue
        if index in cuts:
            semantic.append(index)
            reasons[index].append("selected_local_semantic")
            continue
        position = bisect_left(cuts, index)
        first = cuts[position - 1] + 1 if position else 0
        last = cuts[position] if position < len(cuts) else len(units) - 1
        sizes = [sum(c not in WHITE for c in text[a:b]) for a, b in
                 ((units[first].start, units[index].end), (units[index + 1].start, units[last].end))]
        if min(sizes) < config.min_segment_nonspace_chars:
            reasons[index].append("rejected_min_segment_nonspace_chars")
        else:
            insort(cuts, index)
            semantic.append(index)
            reasons[index].append("selected_local_semantic")
    structural_selected = tuple(index for index in cuts if index in structural)
    lengths = _length_partition(text, units, decision, config, counter, cuts, length_penalties, metadata_start)
    selected = set((*cuts, *lengths))
    segments, first = [], 0
    for last in (*sorted(selected), len(units) - 1) if units else ():
        start, end = units[first].start, units[last].end
        value = text[start:end]
        if _checked_count(counter, value) > config.max_segment_tokens:
            raise ValueError("meaning_length_invariant_failed")
        segments.append(TextSegment(len(segments) + 1, value, start, end))
        if last in structural_selected and sum(c not in WHITE for c in value) < config.min_segment_nonspace_chars:
            reasons[last].append("short_structural_segment")
        first = last + 1
    validate(text, [asdict(segment) for segment in segments])
    for index in range(len(features)):
        if index in lengths:
            sources[index].append("hard_token_limit")
            reasons[index].append("selected_hard_token_limit")
            if blocked[index]:
                reasons[index].append("forced_structure_dependency")
        elif index not in selected and not sources[index]:
            reasons[index].append("not_nominated")
            reasons[index].append("upstream_" + decision.gaps[index].reason)
    traces = tuple(MeaningGapDecision(index, index in selected, tuple(dict.fromkeys(sources[index])),
                  tuple(dict.fromkeys(reasons[index])), tuple(dict.fromkeys(evidence[index]))) for index in range(len(features)))
    return MeaningPartition(tuple(segments), tuple(sorted(semantic)), structural_selected, lengths,
                            tuple(sorted(nominated - set(semantic))), tuple(i for i in sorted(selected) if blocked[i]),
                            traces, blocks, tuple(unavailable))
