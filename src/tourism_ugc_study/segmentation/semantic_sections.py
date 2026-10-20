"""独立小节题头的确认：把“新小节在题头前切开”落实为可复核的行级检查。

对应编码表分段规范`section-heading-20261018`及适配方案§5.2.3第1条。本模块只读
原文行、调用方识别的题头形态和同形条目序列，不读取向量、模型、帖子身份或编码
标签，也不选点、不改写原文。每个候选题头都返回确认结果与原因码：确认的题头由
分区器作为必切依据，未确认的只记录原因，仍按普通边界规则处理。

检查顺序固定为：像不像标题 → 组成题头组 → 前文完成 → 后文展开 → 非同栏目清单
→ 层级（1.3.1）。“像不像标题”排除规范列明不触发的清单条目、描述性长行和叙述短句：
阿拉伯数字、keycap或圈号编号的条目及“-”“·”等项目符号开头的行按清单处理；条目
参数行（人均、价格、地址、营业时间、推荐指数等，或以金额为主体的短行）、带金额的
条目行、图片索引行（“P3 栈桥”）和行尾带箭头的图注不是标题；标题主体纯
文字不超过12个（下一行是成段正文时放宽到20个，即文章式小标题）、带标记不超过32个
非空白字符（与题头识别上限一致）；纯文字标题不以叙述性词开头，不带判断、体貌或语气
成分，“都、会、能”这类常在词内出现的字（都市、会场、功能）单独出现一个不否决；emoji或
括号若在同帖也用作正文行的项目符号，则不是标题标记；标题之前至少已有两行内容（或
首行不少于40个非空白字符），帖子标题行不单独成段；前一行是带标记的标题式短行时本行
是其从属内容；下一行以“也、还、就”等接着说时本行只是半句话。表情代码与纯装饰行
分别按行首标记和分隔线处理。

“前文完成”：前一行以逗号、顿号或冒号结束即未完成；纯文字标题之前的普通叙述行还
须以句末标点结束，但前一行是清单项、成串的字段行或参数行，或两行之间隔着空行时，
视为前项已经结束；本身像题头的编号行、日程行、带标记或单独一行的“名称：说明”行不算
清单项。“第X站：名称”“【A：B】”这类序号或括号标题不按“标签：内容”字段排除。
题头组至多再带2行短行；展开到下一同级题头为止，至少两个完整句（强句末与行末各计
一句）或40个非空白字符，参数行和图片索引行不计——带标记的题头以同类标记为同级，纯文字题头以纯文字题头形态
的行为同级，序号标题（第X站、第X部分）只以序号标题为同级，任何题头的展开都止于
序号标题和日程题头；引出日程的题头（以冒号结束或带行程、路线等词）下一行就是日程
题头时，这些日程是它的展开。层级：位于已
确认序号小节展开范围之内的其他题头是子标题，不另作必切。这些是固定的识别规则，
不按个案调参；识别依据是版式与标点，不做中文句法分析，无法确认时保守地不建立
必切边界。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
# 展开不足两句时，展开范围的非空白字符达到这一数量也算实质展开（无标点长说明）。
MIN_EXPANSION_CHARS = 40
# 纯文字标题主体上限：短于此的无标点短行才可能是标题，再长多是没打标点的叙述句。
MAX_TITLE_CHARS = 12
# 新闻稿、长文的纯文字小标题常超过12字；后接成段正文（不少于40个非空白字符、带句末
# 标点的一行）时，放宽到这一上限（1.3.1）。
MAX_ARTICLE_TITLE_CHARS = 20
# 文章式小标题之后那一行至少这么长才算成段正文。
MIN_PARAGRAPH_CHARS = 40
# 带emoji或括号标记的标题常附副标题或说明，上限与题头识别的32个非空白字符一致。
MAX_MARKED_TITLE_CHARS = 32
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
                              r"我(?:们)?(?!的)|你(?:们)?(?!的)|他|她|它|咱|大家)")
# 判断、体貌、情态与语气成分，出现即像叙述句。
_NARRATIVE_MARK = re.compile(r"是|了|过|着|就|也|还|又|真的|想|吧|呢|吗|啊|呀|哦|啦")
# 这些字常在词内出现（都市、会场、功能……），单独一个不否决标题，两个以上才算叙述（1.3.1）。
_WEAK_NARRATIVE_MARK = re.compile(r"都|会|能")
# 引出日程的标题常带这些词（“行程安排”“三日松弛路线”“出行攻略”）。
_SCHEDULE_CUE = re.compile(r"行程|路线|线路|安排|攻略|计划|日程|规划|总览|概览|玩法")
# 行尾的图片来源或摄影署名（“(来源：信网)”“（记者 王晨 摄）”）：图注之后前文已经结束。
_SOURCE_NOTE = re.compile(r"[（(][^（()）]*(?:来源|摄|图源|供图)[^（()）]*[）)]\s*$")
# 去掉行首标记后以“第X天／DayN／DN”开头的行是日程行（行首带emoji时题头识别可能不归为日程）。
_DAY_START = re.compile(r"^(?:第[0-9一二三四五六七八九十]+天|day\s*\d|d\d)", re.I)
# 图片署名行（“图片来自大众点评网友@……”“图源：……”）是图注，不是标题。
_CREDIT = re.compile(r"^(?:(?:图片|照片|图)(?:来自|来源|出自)|图源|摄影[:：])")
# 行尾指向配图的箭头说明本行是图注。
_CAPTION_ARROWS = ("↑", "↓", "⬆", "⬇", "⬆️", "⬇️")
# 图片索引行（“图1-2：第三海水浴场”“P3 栈桥”）是配图说明，不算题头之后的实质展开。
_PICTURE_INDEX = re.compile(r"^(?:图|[pP])\s*\d+(?:\s*[-~—–]\s*\d+)?(?![\d.])")
# 序号标题：第X站、第X部分等。冒号后的名称是标题的一部分，不是“标签：内容”字段。
_ORDINAL_TITLE = re.compile(r"^(?:第[0-9一二三四五六七八九十百]+(?:站|部分|章|节|篇|part)|(?:part|stop)\s*\d+)", re.I)
# 条目参数行：以参数名（后接冒号、分隔符、金额符号或数字）开头，或以金额为主体的短行。
_PARAMETER_PREFIX = re.compile(r"^(?:人均|均价|客单价|价格|票价|门票|费用|花费|预算|地址|位置|电话|营业时间|开放时间|时长|停车|"
                               r"推荐指数|推荐度|评分|打分|星级|难度|强度)[^\w:：]{0,3}(?:[:：|｜💰￥¥$]|\d)")
# 评分行：评分名之后只有星号、数字或符号（“推荐指数⭐️⭐️⭐️⭐️⭐️”）。
_RATING = re.compile(r"^(?:推荐指数|推荐度|评分|打分|星级|难度|强度)\W*$")
_MONEY = re.compile(r"(?:[￥¥$]|RM|rmb|RMB)\s*\d|\d+(?:\.\d+)?\s*(?:元|块|￥|¥|rmb|RMB)")
# 以金额为主体的参数行去掉金额后至多剩这么多文字。
MAX_MONEY_LINE_WORDS = 4
# 清单项目符号：以这些符号开头的行是清单项，不是标题标记。
_LIST_MARKS = frozenset("-·•*▪▫◦‧・")
CONFIRMED = "selected_section_heading"
AT_POST_START = "section_heading_at_post_start"
SUBSECTION = "section_heading_subsection"


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


# 编号条目：“1、”“2.”“（三）”、keycap，或一两位数字后接空格（“3 各大景区门口的桂花糕”，1.3.1）。
_NUMBERED = re.compile(r"^(?:\d+[、．.）)]|[一二三四五六七八九十百]+[、．.]|[（(][一二三四五六七八九十百\d]+[）)]|\d\ufe0f?\u20e3"
                       r"|\d{1,2}[ \u3000]+(?=[^\d\s]))")
_OPEN_BRACKETS = frozenset("【[〖「『〔")


def strip_marks(value: str) -> str:
    """去掉行首空白、表情代码、emoji、符号、标点、组合符和格式符（emoji内的零宽连接符），只供识别，不作输出正文。"""
    cursor = 0
    while cursor < len(value):
        if code := XHS_EMOJI_CODE.match(value, cursor):
            cursor = code.end()
        elif (value[cursor] in WHITE or unicodedata.category(value[cursor])[0] in {"S", "P", "M"}
              or unicodedata.category(value[cursor]) == "Cf"):
            cursor += 1
        else:
            break
    return value[cursor:]


def parameter_text(value: str) -> bool:
    """整行是条目参数：人均、价格、地址、营业时间等字段，或以金额为主体的短行。

    参数属于所在条目（店铺、景点、交通段），既不是标题，也不能当作分隔线上方的
    题头；行内有句末标点的完整叙述不算参数行。以金额为主体指去掉金额后至多剩
    4个文字（“¥98/位”“约50元一位”）；“8、竹青胶东摔面 井冈山路373号 22元”这类
    带价格的条目标题不是参数行。
    """
    plain = strip_marks(value)
    if not plain:
        return False
    if _PARAMETER_PREFIX.match(plain) or _RATING.match(plain):
        return True
    rest = _MONEY.sub("", plain)
    return (bool(_MONEY.search(value)) and not any(c in value for c in "。！？!?")
            and sum(unicodedata.category(c)[0] == "L" for c in rest) <= MAX_MONEY_LINE_WORDS)


def marker_class(value: str) -> str:
    """题头行的标记类别：ordinal、list、numbered、bracket、symbol或plain。

    用于判断小节展开到哪里结束：只有同类标记的题头才视为同级小节，形态不同的
    短行（如emoji小节下的纯文字清单项）属于该小节正文。ordinal是“第X站”“第X部分”
    等序号标题（行首可带emoji），层级高于其他标记；list是以“-·•”等项目符号开头
    的行（1.3.1起单列，不再混入plain）。
    """
    body = value.lstrip("".join(WHITE))
    if not body:
        return "plain"
    if _ORDINAL_TITLE.match(strip_marks(body)):
        return "ordinal"
    if body[0] in _LIST_MARKS:
        return "list"
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


def introduces_schedule(value: str) -> bool:
    """该行像引出日程的标题：以冒号结束，或带行程、路线、安排等日程词（1.3.1）。

    上一天末尾的条目短行（“台东夜市”“晚上回酒店点外卖吃”）不算，紧跟的日程不是它的展开。
    """
    return value.rstrip("".join(WHITE)).endswith((":", "：")) or bool(_SCHEDULE_CUE.search(value))


def leading_mark(value: str) -> str | None:
    """公开给分区器的行首标记识别，语义同 _leading_mark。"""
    return _leading_mark(value)


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


def _blank_everywhere(text: str, lines: Sequence[SectionLine]) -> bool:
    """全帖多数相邻非空行之间都隔着空行：空行是排版习惯，不是段落间隔。"""
    pairs = len(lines) - 1
    return pairs > 0 and sum(_blank_before(text, lines, i) for i in range(1, len(lines))) * 2 > pairs


def _blank_before(text: str, lines: Sequence[SectionLine], index: int) -> bool:
    """该行与前一非空行之间隔着至少一个空行（段落间隔）。"""
    return index > 0 and len(_LINE_BREAK.findall(text[lines[index - 1].end:lines[index].start])) >= 2


def _item_like(value: str) -> bool:
    """带金额的条目行、图片索引行、图片署名行或行尾带箭头的图注：本身是条目或说明，不是标题。"""
    body = value.strip("".join(WHITE))
    plain = strip_marks(value)
    return bool(_MONEY.search(value) or _PICTURE_INDEX.match(plain) or _CREDIT.match(plain)
                or body.endswith(_CAPTION_ARROWS))


def _item_line(text: str, lines: Sequence[SectionLine], index: int, siblings: frozenset[int]) -> bool:
    """该行是清单项、字段或参数：项目符号开头、参数行、带标点的同形条目，或不像题头的编号行与字段行。

    这样的行结束时前一项已经完整，不要求句末标点；它也不是题头，不构成连续题头。
    编号行带句内标点才是条目（“1、坐火车来的选青岛站，……”），否则是序号标题；
    “标签：内容”只有纯文字、非数字开头、非日程行，且与相邻的字段或参数行成串时才是
    字段（“崂山：官方公众号预约”“信号山：官方公众号预约”）；单独一行的“核心产品测评：
    ……”和“04 八大关风景区：……”“Day 2：……”“🔵⑧ 琴屿路：……”仍是题头，其下的短行是
    从属内容。
    """
    value = text[lines[index].start:lines[index].end]
    body = value.lstrip("".join(WHITE))
    if not body:
        return False
    if body[0] in _LIST_MARKS or parameter_text(value):
        return True
    # 同形条目只有带句内标点或本身是字段时才是清单内容；短而无标点的同形条目
    # （“🔵⑤ 小鱼山下山 → 大学路”“1.大珠山📍山川路｜40元”）是各站标题，其下一行是正文。
    if index in siblings and (any(c in body for c in "，,。！？!?；;") or _field_line(text, lines, index)):
        return True
    if lines[index].heading in _SCHEDULE_KINDS:
        return False
    if _NUMBERED.match(body):
        return any(c in body for c in "，,。！？!?；;")
    return _field_line(text, lines, index) and any(
        0 <= other < len(lines) and (_field_line(text, lines, other)
                                     or parameter_text(text[lines[other].start:lines[other].end]))
        for other in (index - 1, index + 1))


def _field_line(text: str, lines: Sequence[SectionLine], index: int) -> bool:
    """纯文字“标签：内容”字段行：非数字开头、非日程行、不含网址（网址里的冒号不是字段分隔）。"""
    value = text[lines[index].start:lines[index].end]
    body = value.lstrip("".join(WHITE))
    return (bool(body) and lines[index].heading not in _SCHEDULE_KINDS and "://" not in value
            and _label_value(value) and marker_class(value) == "plain" and not body[0].isdigit())


def _title_issue(text: str, lines: Sequence[SectionLine], index: int, marker: str,
                 bullets: frozenset[str], siblings: frozenset[int]) -> str | None:
    """题头行是否不像独立小节标题；像标题时返回None，否则返回未确认原因码。

    siblings是全部同形条目所在行号，用于判断前一行是否为已结束的清单项。
    """
    value = text[lines[index].start:lines[index].end]
    body = value.strip("".join(WHITE))
    if marker == "numbered" and not _CHINESE_SECTION_NUMBER.match(body):
        return "section_heading_list_number"
    # 以“-”“·”等项目符号开头的行是清单项，不作独立小节（1.3.1）。
    if marker == "list":
        return "section_heading_bullet_marker"
    # 带金额的条目行、图片索引行和指向配图的图注（“……↑”）是条目或说明，不是标题（1.3.1）。
    if _item_like(value):
        return "section_heading_item_line"
    core = title_core(value)
    following = next((text[line.start:line.end] for line in lines[index + 1:]
                      if not separator_text(text[line.start:line.end])), "")
    if marker == "plain":
        paragraph = (sum(c not in WHITE for c in following) >= MIN_PARAGRAPH_CHARS
                     and any(c in following for c in _FINAL_MARKS) and not _ends_final(value))
        limit = MAX_ARTICLE_TITLE_CHARS if paragraph else MAX_TITLE_CHARS
    else:
        limit = MAX_MARKED_TITLE_CHARS
    if marker != "numbered" and sum(c not in WHITE for c in core) > limit:
        return "section_heading_too_long"
    if marker == "plain" and (_NARRATIVE_START.match(core) or _NARRATIVE_MARK.search(core)
                              or len(set(_WEAK_NARRATIVE_MARK.findall(core))) >= 2):
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
    # 前一项是清单项、字段或参数，以图片来源结束，或两行之间隔着空行时，前文已经结束（1.3.1）；
    # 全帖多数行之间都隔空行时，空行只是排版习惯，不作段落结束的证据。
    finished = ((_blank_before(text, lines, index) and not _blank_everywhere(text, lines))
                or _item_line(text, lines, index - 1, siblings) or bool(_SOURCE_NOTE.search(previous)))
    # 前一行是带标记、像标题的短行（如“📍石老人→雕塑园”：无句内标点、无句末标点），
    # 本行是它的从属内容；带逗号或冒号的标记行是上一节的清单项，不在此列。
    if (not finished and _leading_mark(previous) and not _ends_final(previous)
            and not any(c in previous for c in "，,。！？!?；;：:")
            and sum(c not in WHITE for c in previous) <= 32):
        return "section_heading_previous_incomplete"
    if marker == "plain" and not finished and not _ends_final(previous):
        return "section_heading_previous_incomplete"
    if _FRAGMENT_CONTINUATION.match(following.lstrip("".join(WHITE))):
        return "section_heading_sentence_fragment"
    return None


def label_field(value: str) -> bool:
    """整行是“标签：内容”字段（序号标题与括号内冒号除外），供分区器判断移动目标。"""
    return _label_value(value)


def _label_value(value: str) -> bool:
    """“标签：内容”字段本身就是内容，不作为小节题头或副题。

    序号标题（“第二站：小麦岛”）与括号内的冒号（“【市区点位：雕塑园】”）不算字段（1.3.1）。
    """
    if _ORDINAL_TITLE.match(strip_marks(value)):
        return False
    outside = re.sub(r"【[^】]*】|〖[^〗]*〗|\[[^\]]*\]", "", value)
    return bool(re.search(r"[:：]\s*\S", outside))


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
    """该行是否与标记类别为marker的题头同级或更高，从而结束其展开范围。

    日程题头与序号标题（第X站、第X部分）总是另起结构；序号标题只以这两者为界。
    带标记（emoji、括号、编号、项目符号）的题头只以同类标记的题头为同级，其下的
    纯文字短行是清单项等正文；纯文字题头以纯文字题头形态的行为同级，冒号结尾的
    栏目短行和“标签：内容”字段是它的正文（1.3.1，此前任何题头形态都算同级）。
    """
    if line.heading is None:
        return False
    value = text[line.start:line.end]
    cls = marker_class(value)
    if line.heading in _SCHEDULE_KINDS or cls == "ordinal":
        return True
    if marker == "ordinal":
        return False
    if marker == "plain":
        return cls == "plain" and not value.rstrip("".join(WHITE)).endswith((":", "：")) and not _label_value(value)
    return cls == marker


def _consecutive(text: str, lines: Sequence[SectionLine], index: int, marker: str, siblings: frozenset[int]) -> bool:
    """前一行是与本行同级的题头，说明两行是连续题头（目录或题头组），前文并未完成。

    清单项、字段和参数行不是题头；中文编号、序号标题之前的日程行属于上一节正文。
    """
    previous = lines[index - 1]
    if previous.heading is None or _item_line(text, lines, index - 1, siblings):
        return False
    if marker in {"numbered", "ordinal"}:
        return marker_class(text[previous.start:previous.end]) == marker
    return _same_level(text, previous, marker)


def _expansion(text: str, lines: Sequence[SectionLine], after: int, marker: str) -> tuple[int, int, int, int]:
    """返回题头组之后的展开范围：(起点, 首说明终点, 范围终点, 完整句数)。

    展开范围从下一行开始，到下一条同级题头、日程题头或帖尾为止；纯元数据行、占位行、
    参数行和图片索引行不计句数——名称之下只有门票、开放时间的是清单条目，不是小节。题头下一行就是日程题头时（如“行程安排”下接Day1、Day2），这些日程是它的
    展开，范围到下一条同级的非日程题头为止。没有正文时起点与终点都取题头组终点，
    句数为0。题头须像引出日程的标题（以冒号结束或带日程词），否则下一行的日程
    仍结束展开。
    """
    group_end = lines[after].end
    body = []
    def day(line: SectionLine) -> bool:
        return line.heading in _SCHEDULE_KINDS or bool(_DAY_START.match(strip_marks(text[line.start:line.end])))

    days = (after + 1 < len(lines) and day(lines[after + 1])
            and introduces_schedule(text[lines[after].start:lines[after].end]))
    for line in lines[after + 1:]:
        if _same_level(text, line, marker) and not (days and day(line)):
            break
        value = text[line.start:line.end]
        if not (metadata_text(value) or placeholder_text(value) or parameter_text(value)
                 or _PICTURE_INDEX.match(strip_marks(value))):
            body.append(line)
    if not body:
        return group_end, group_end, group_end, 0
    first = first_complete_end(text, body[0].start, body[0].end)
    return (body[0].start, first, body[-1].end,
            sum(count_sentences(text, line.start, line.end) for line in body))


def _group_last(text: str, lines: Sequence[SectionLine], index: int) -> int:
    """题头组最后一行：题头后至多再带两行无句末标点的副题／栏目短行。

    “标签：内容”字段、参数行和以项目符号开头的清单项不是副题（后两者1.3.1起）；
    清单项、图注和带金额的条目行本身也不带副题，纯文字短行（常是清单末项）也不跨
    空行带副题，否则会把下一节题头吞进自己的组。
    """
    last = index
    value = text[lines[index].start:lines[index].end]
    if marker_class(value) == "list" or _item_like(value):
        return last
    plain = marker_class(value) == "plain"
    while (last + 1 < len(lines) and last - index < GROUP_EXTRA_LINES
           and lines[last + 1].heading in _GROUP_KINDS and not (plain and _blank_before(text, lines, last + 1))
           and not _label_value(value := text[lines[last + 1].start:lines[last + 1].end])
           and not parameter_text(value) and marker_class(value) != "list"):
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
    siblings = frozenset(sibling_of)
    # 同帖正文行（非题头形态）行首用过的标记，说明它在本帖是项目符号而非标题标记。
    bullets = frozenset(mark for line in lines if line.heading is None
                        for mark in (_leading_mark(text[line.start:line.end]),) if mark)
    output: list[SectionCandidate] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        value = text[line.start:line.end]
        if (line.heading not in CANDIDATE_KINDS or _label_value(value.rstrip(":："))
                or placeholder_text(value) or parameter_text(value)):
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
        expanded = (sentences >= MIN_EXPANSION_SENTENCES
                    or sum(c not in WHITE for c in text[body_start:expansion_end]) >= MIN_EXPANSION_CHARS)
        # 紧接在同级题头之后说明是连续题头（目录或题头组），不是前文已完成；
        # 标记类别不同的短行（如上一节的纯文字清单项）属于上一节正文。
        issue = _title_issue(text, lines, index, marker, bullets, siblings) if index else None
        if issue is not None:
            reason = issue
        elif index and (_consecutive(text, lines, index, marker, siblings)
                        or _ends_with_continuation(text[previous.start:previous.end])):
            reason = "section_heading_previous_incomplete"
        elif not expanded:
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
    return tuple(_demote_subsections(text, lines, output))


def _demote_subsections(text: str, lines: Sequence[SectionLine],
                        candidates: list[SectionCandidate]) -> list[SectionCandidate]:
    """位于已确认序号小节（第X站等）展开范围之内的其他已确认题头降为子标题。

    “第二站：小麦岛”之下的“📌 自驾贴士”是该站的组成部分，不是与站并列的新小节；
    子标题不作必切，仍可由普通边界规则处理（1.3.1）。
    """
    parents = [c for c in candidates if c.protected
               and marker_class(text[lines[c.line_index].start:lines[c.line_index].end]) == "ordinal"]
    output = []
    for candidate in candidates:
        if (candidate.confirmed
                and marker_class(text[lines[candidate.line_index].start:lines[candidate.line_index].end]) != "ordinal"
                and any(parent.group_end < candidate.start < parent.expansion_end for parent in parents)):
            candidate = replace(candidate, reason=SUBSECTION,
                                evidence=(*candidate.evidence, *(("parent_section", p.start, p.group_end) for p in parents
                                                                 if p.group_end < candidate.start < p.expansion_end)))
        output.append(candidate)
    return output


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
