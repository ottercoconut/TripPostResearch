"""识别切开后会损害原意的显式语义关系，为每个既有 gap 给出禁止切证据。

本模块只看原文与候选单元，不读取向量、模型、帖子身份或编码标签。三类
证据对应规则清单的禁止切项：后句以强依赖词开头（A2）、前句是尚未给出
结论的条件（A3）、后句是附着上一条的括号说明（A5）。识别只依赖显式
词面与标点，隐含比较或跨多句的条件仍无法覆盖，属于已知限制。输出与
gap 一一对应；无关系的 gap 返回空元组，不表示该处应当切开。

另有一类较弱的证据（1.3起）：后句以对比或解释前文的词开头，说明前一叙述
尚未完成。它不进入禁止切，只供分区器在普通边界上不选该处；换天、独立小节
必切和长度兜底不受影响。
"""

from __future__ import annotations

import re
from typing import Sequence
import unicodedata

from .contracts import TextUnit
from .integrity import WHITE

# 强依赖词：后句离开前文就无法成立或会改变含义；弱转折（但是、不过、
# 然后等）不进入禁止切，其中对比／解释开头词只作下方普通边界证据。
_DEPENDENT = re.compile(
    r"^(?:否则|不然|不如|相比之下|相比|相较|比起|即便|即使|哪怕|"
    r"因此|所以|因而|而且|并且|况且|但如果|可如果)"
)
# 对比或解释前文的开头词（1.3，适配方案§5.2.3第3条）：后句接着前句讲，
# 前一叙述还没完成。只作用于普通边界；“另外、此外、然后”等引出新内容，不在此列。
_OPEN_CONTRAST_OR_REASON = re.compile(r"^(?:但是|但|可是|然而|不过|只是|因为)")
# 条件引导词；可带“但／可／而”前缀，如“但如果……”。
_CONDITION = re.compile(r"^(?:但|可|而)?(?:如果|假如|要是|只要|除非|若是|倘若)")
# “（一）”“(2)”等是编号题头，不是附着说明。
_NUMBERED_PAREN = re.compile(r"^[（(][一二三四五六七八九十百\d]+[）)]")
_STRONG = frozenset("。！？!?；;")
_CLAUSE = frozenset("，,；;")
_BULLETS = frozenset("-•·*")


def _strip_lead(value: str) -> str:
    """去掉行首空白、项目符号和 emoji 等符号，保留括号及正文首字。"""
    index = 0
    while index < len(value):
        char = value[index]
        if char in WHITE or char in _BULLETS or unicodedata.category(char).startswith(("S", "M")) or char == "️":
            index += 1
        else:
            break
    return value[index:]


def _ends_strong(value: str) -> bool:
    """去掉尾部空白与闭引号／括号后，判断是否以句末强标点结束。"""
    stripped = value.rstrip("".join(WHITE) + "”’」』)）】]》〉\"'")
    return bool(stripped) and stripped[-1] in _STRONG


def relation_reasons(left: str, right: str) -> tuple[str, ...]:
    """返回切在 left 与 right 之间会破坏的显式关系代码，可能为空。

    left/right 分别是 gap 两侧的完整候选单元原文。代码含义：
    before_dependent_connective——right 以强依赖词开头；
    after_open_condition——left 以条件词开头，且既无分句标点也未以强标点
    结束，说明结论还在后文；before_parenthetical_note——right 以开括号
    开头且不是编号题头，属于上一条的补充说明。
    """
    reasons: list[str] = []
    right_body = _strip_lead(right)
    left_body = _strip_lead(left)
    if _DEPENDENT.match(right_body):
        reasons.append("before_dependent_connective")
    if (_CONDITION.match(left_body) and not _ends_strong(left)
            and not any(char in _CLAUSE for char in left_body)):
        reasons.append("after_open_condition")
    right_text = right.lstrip("".join(WHITE))
    if right_text[:1] in ("（", "(") and not _NUMBERED_PAREN.match(right_text):
        reasons.append("before_parenthetical_note")
    return tuple(reasons)


def analyze_gap_relations(text: str, units: Sequence[TextUnit]) -> tuple[tuple[str, ...], ...]:
    """对每个既有 gap 返回显式关系代码；长度为单元数减一，N≤1 时为空。

    调用者负责先校验单元覆盖与顺序；本函数只切取原文，不新增或移动 gap。
    """
    return tuple(relation_reasons(text[left.start:left.end], text[right.start:right.end])
                 for left, right in zip(units, units[1:]))


def opens_with_contrast_or_reason(right: str) -> bool:
    """后句（gap 右侧完整候选单元）是否以对比或解释前文的词开头。

    去掉行首空白、项目符号和 emoji 后再判断。返回 True 表示前一叙述尚未完成，
    普通边界不应选在此处；这不是禁止切证据，不能否决必切或长度兜底。
    """
    return bool(_OPEN_CONTRAST_OR_REASON.match(_strip_lead(right)))


def analyze_gap_continuations(text: str, units: Sequence[TextUnit]) -> tuple[bool, ...]:
    """对每个既有 gap 判断后句是否以对比或解释词开头；长度为单元数减一。"""
    return tuple(opens_with_contrast_or_reason(text[right.start:right.end]) for right in units[1:])
