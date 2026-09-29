"""已验收的 objective-structure-0.5 客观结构边界引擎。

所有边界均用Unicode码点位置表达；token、配对符号、列表和标签先定位，
再统一裁决边界，最后编号。返回JSON兼容片段和问题，不写数据库。
从2026-09-08冻结实现迁入；只依赖标准库和regex，不访问文件、数据库或模型。
输入必须是已投影的字符串，空白正文输出空列表；完整性失败抛出ValueError，
不返回部分结果。外部调用应使用本包的segment_text或segment_body接口。
"""

import re
from bisect import bisect_left, bisect_right
import regex

RULE_VERSION = "objective-structure-0.5"
IMPLEMENTATION_VERSION = "objective-structure-0.5-grapheme-safe-1"
WHITE = frozenset("\t\n\v\f\r \x85\xa0\u1680\u2028\u2029\u202f\u205f\u3000" + "".join(chr(i) for i in range(0x2000, 0x200B)))
LINES = re.compile(r"\r\n|[\r\n\u2028\u2029]")
STRONG = "。！？；!?;"
PAIRS = {"“": "”", "‘": "’", "「": "」", "『": "』", "(": ")", "（": "）", "【": "】", "[": "]", "《": "》", "〈": "〉", '"': '"', "'": "'"}
QUOTES = set("“‘「『\"'")
CLOSERS = set(PAIRS.values())
ENTITY = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);")
URL = re.compile(r'''(?:https?://|www\.)[^\s<>"“”‘’「」『』。！？；]+''', re.I)
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
PLATFORM_TOKEN = re.compile(r"\[[^\]\[\r\n]{1,20}(?:R)\]|\[doge\]|\[微笑\]|\[笑脸\]")
ABBR = re.compile(r"(?<![A-Za-z])(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e)\.|(?<![A-Za-z])(?:[A-Za-z]\.){2,}|\b[A-Z]\.(?=\s+[A-Z][a-z])", re.I)
MARKER = re.compile(r"(?P<hierarchy>(?<![A-Za-z\d.])(?P<hparent>\d{1,2})\.(?P<hchild>\d{1,2})\s+(?=[\u3400-\u9fff][^。！？；!?;\r\n：:]{0,20}[:：]))|(?P<day>(?<![A-Za-z])(?:Day\s*|D)(?P<dnum>\d{1,2})(?:\ufe0f?\u20e3)?(?:天)?\s*[:：]?)|(?P<zhday>第(?P<znum>\d{1,2}|[一二三四五六七八九十]+)天\s*[:：]?)|(?P<number>(?<![A-Za-z\d.])(?P<nnum>\d{1,2})(?P<nmark>[.、)）：:])(?!\d))|(?P<circled>[①-⑳])|(?P<keycap>[0-9]\ufe0f?\u20e3)|(?P<chinese>[一二三四五六七八九十]+、)", re.I)
KEYCAP_NUMBER = r"(?:[0-9]\ufe0f?\u20e3)+|🔟"
NUMBER_RANGE = re.compile(rf"(?:{KEYCAP_NUMBER}|[①-⑳])\s*[-—–~～至]\s*(?:{KEYCAP_NUMBER}|[①-⑳])")
# 组合数字必须一次识别，例如1️⃣2️⃣是12，不是1和2两个列表标记。
MARKER = re.compile(MARKER.pattern.replace(r"(?P<keycap>[0-9]\ufe0f?\u20e3)", rf"(?P<keycap>{KEYCAP_NUMBER})"), re.I)


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """仅修剪明确空白集合，绝不修改输入或删除内部空白。"""
    while start < end and text[start] in WHITE:
        start += 1
    while start < end and text[end - 1] in WHITE:
        end -= 1
    return start, end


def _inside(position: int, spans: list[tuple[int, int]]) -> bool:
    """位置是否严格位于受保护区间内部；端点仍可作为边界。"""
    return any(a < position < b for a, b in spans)


def _covered(position: int, spans: list[tuple[int, int]]) -> bool:
    """字符是否属于完整token或配对范围。"""
    return any(a <= position < b for a, b in spans)


def _line_ranges(text: str) -> list[tuple[int, int]]:
    """返回不含换行事件的原文物理行区间，兼容CRLF与Unicode分隔符。"""
    result, start = [], 0
    for match in LINES.finditer(text):
        result.append((start, match.start()))
        start = match.end()
    return result + [(start, len(text))]


def _tokens(text: str) -> tuple[list[tuple[int, int]], list[dict]]:
    """定位HTML、链接、邮箱、缩写与平台token，歧义仅提示，不解码。"""
    spans, issues = [], []
    for pattern in (ENTITY, EMAIL, PLATFORM_TOKEN, ABBR, NUMBER_RANGE):
        for match in pattern.finditer(text):
            spans.append(match.span())
            if pattern is ABBR:
                issues.append({"code": "ASCII_PERIOD_AMBIGUOUS", "start": match.start(), "end": match.end()})
    for match in URL.finditer(text):
        start, end = match.span()
        # 链接外层的闭合括号不属于链接，链接自身平衡的括号仍受保护。
        for opening, closing in (("(", ")"), ("（", "）"), ("[", "]"), ("【", "】")):
            while end > start and text[end - 1] == closing and text[start:end].count(closing) > text[start:end].count(opening):
                end -= 1
        if end > start and (text[end - 1] in "!?;" or re.search(r"[\u3400-\u9fff]", text[start:end])):
            issues.append({"code": "URL_BOUNDARY_AMBIGUOUS", "start": start, "end": end})
        spans.append((start, end))
    return sorted(set(spans)), issues


def _marker_info(match: re.Match) -> tuple[str, int]:
    """将列表形态投影到系列与序号，仅用于递增序列判断。"""
    if match.group("hierarchy"):
        return "hierarchy" + match.group("hparent"), int(match.group("hchild"))
    if match.group("day"):
        return "day", int(match.group("dnum"))
    if match.group("zhday"):
        return "zhday", _number_value(match.group("znum"))
    if match.group("number"):
        return "number" + match.group("nmark"), int(match.group("nnum"))
    if match.group("circled"):
        return "circle", ord(match.group()) - ord("①") + 1
    if match.group("keycap"):
        return "keycap", 10 if match.group() == "🔟" else int("".join(c for c in match.group() if c.isdigit()))
    return "chinese", _number_value(match.group().rstrip("、"))


def _number_value(raw: str) -> int:
    """解析本轮支持的1—99阿拉伯/汉字列表数值，其余返回0并不作为递增证据。"""
    if raw.isdigit():
        return int(raw)
    mapping = {c: i for i, c in enumerate("一二三四五六七八九", 1)}
    if "十" in raw:
        left, right = raw.split("十", 1)
        return mapping.get(left, 1) * 10 + mapping.get(right, 0)
    return mapping.get(raw, 0)


def _lists(text: str, tokens: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], set[int], list[dict]]:
    """识别行首与同系列递增的行内列表，返回标记保护区间和条目边界。"""
    markers, cuts, issues = [], set(), []
    for a, b in _line_ranges(text):
        start, end = _trim(text, a, b)
        found = []
        for match in MARKER.finditer(text, start, end):
            if _covered(match.start(), tokens):
                continue
            if match.group("zhday") and not match.group().rstrip().endswith((":", "：")):
                # 普通叙事中的第二天不是客观列表标记；仅接受独立行首。
                if _decorated_start(text, start, match.start()) != start:
                    continue
            found.append(match)
        accepted = set()
        previous_by_family = {}
        for index, match in enumerate(found):
            prefix_start = _decorated_start(text, start, match.start())
            if prefix_start == start:
                accepted.add(index)
            family, number = _marker_info(match)
            if family in previous_by_family:
                prev_index, prev_number = previous_by_family[family]
                previous = found[prev_index]
                if number > prev_number and text[previous.end():match.start()].strip():
                    accepted.update((prev_index, index))
            previous_by_family[family] = (index, number)
        for index, match in enumerate(found):
            if index in accepted:
                markers.append(match.span())
                cuts.add(_decorated_start(text, start, match.start()))
            else:
                issues.append({"code": "LIST_BOUNDARY_AMBIGUOUS", "start": match.start(), "end": match.end()})
    return markers, cuts, issues


def _decorated_start(text: str, line_start: int, marker_start: int) -> int:
    """将紧接编号前的emoji装饰归入条目；普通正文与单纯空格不向前吞并。"""
    prefix = text[line_start:marker_start]
    begin, has_emoji = marker_start, False
    for match in reversed(list(regex.finditer(r"\X", prefix))):
        cluster = match.group()
        if all(c in WHITE for c in cluster):
            begin = line_start + match.start()
        elif regex.search(r"\p{Extended_Pictographic}|\p{Regional_Indicator}", cluster):
            has_emoji = True
            begin = line_start + match.start()
        else:
            break
    return begin if has_emoji else marker_start


def _pairs(text: str, tokens: list[tuple[int, int]]) -> tuple[list[tuple[int, int, str]], list[dict]]:
    """按栈匹配嵌套符号；未配对开符号不保护后续整帖，错误不补写原文。"""
    stack, pairs, issues = [], [], []
    for index, char in enumerate(text):
        if _covered(index, tokens):
            continue
        if char == "'" and 0 < index < len(text) - 1 and text[index - 1].isalnum() and text[index + 1].isalnum():
            continue
        if stack and PAIRS[stack[-1][1]] == char:
            begin, opening = stack.pop()
            pairs.append((begin, index + 1, "quote" if opening in QUOTES else "bracket"))
        elif char in PAIRS:
            stack.append((index, char))
        elif char in CLOSERS:
            issues.append({"code": "UNBALANCED_DELIMITER", "start": index, "end": index + 1})
    issues.extend({"code": "UNBALANCED_DELIMITER", "start": index, "end": index + 1} for index, _ in stack)
    return pairs, issues


def _tags(text: str, protected: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], set[int], list[dict]]:
    """保护完整话题，并只为明确的行前后缀标签块生成边界，不逐标签切分。"""
    all_tokens, cuts, issues = [], set(), []
    for a, b in _line_ranges(text):
        start, end = _trim(text, a, b)
        # 共用井号或单双端混合、缺少分隔时，不把未知标签名强行改称正文。
        # 只有行内首次#之后没有强句末，且可见“#甲#乙#”形态时整体回退。
        tail_start = text.find("#", start, end)
        if tail_start >= 0 and not _covered(tail_start, protected):
            tail = text[tail_start:end]
            if (not any(c in STRONG for c in tail)
                    and (re.search(r"#[^#\s]+#[^#\s,，、。！？;；!?:：@]+#", tail)
                         or re.search(r"#[^#\r\n]*\s#(?=\s*#)", tail))):
                all_tokens.append((tail_start, end))
                cuts.update((tail_start, end))
                issues.append({"code": "TAG_SYNTAX_AMBIGUOUS", "start": tail_start, "end": end})
                continue
        matches = []
        position = start
        while position < end:
            if text[position] != "#" or _covered(position, protected):
                position += 1
                continue
            # 单端标签之间的“空白+#”是下一标签开头，不是前一标签的闭合。
            paired = re.match(r"#[^#\r\n]*[^#\s]#", text[position:end])
            if paired and not text[position + 1].isspace():
                stop = position + paired.end()
            else:
                opened = re.match(r"#[^#\s]+", text[position:end])
                if not opened:
                    position += 1
                    continue
                stop = position + opened.end()
            matches.append((position, stop))
            position = stop
        all_tokens.extend(matches)
        if not matches:
            continue
        groups = []
        for t_start, t_end in matches:
            if groups and not text[groups[-1][1]:t_start].strip("".join(WHITE) + "\ufeff\u200b，,、"):
                groups[-1] = (groups[-1][0], t_end)
            else:
                groups.append((t_start, t_end))
        for t_start, t_end in groups:
            while t_end < end and text[t_end] in "\ufeff\u200b":
                t_end += 1
            while t_start > start and text[t_start - 1] in "\ufeff\u200b":
                t_start -= 1
            members = [(x, y) for x, y in matches if t_start <= x and y <= t_end]
            attached_open = t_start > start and text[t_start - 1] not in WHITE and not text[members[0][0]:members[0][1]].endswith("#")
            if attached_open and len(members) < 2:
                issues.append({"code": "TAG_BOUNDARY_AMBIGUOUS", "start": t_start, "end": t_end})
                continue
            if t_start == start or t_end == end:
                cuts.update((t_start, t_end))
    return all_tokens, cuts, issues


def _emoji_only(text: str) -> bool:
    """按字素簇检查纯emoji/平台表情；不能把变体或连接符分离为字。"""
    cleaned = PLATFORM_TOKEN.sub("", text)
    clusters = regex.findall(r"\X", cleaned)
    visible = [cluster for cluster in clusters if not all(c in WHITE for c in cluster)]
    return bool(text.strip("".join(WHITE))) and all(regex.search(r"\p{Extended_Pictographic}|\p{Regional_Indicator}|\u20e3", c) for c in visible)


def _terminal_at(text: str, index: int, tokens: list[tuple[int, int]]) -> bool:
    """判断单字符是否有句末功能；省略号、小数和已保护token不触发。"""
    if _covered(index, tokens):
        return False
    char = text[index]
    if char in STRONG:
        return True
    if char != ".":
        return False
    if (index and text[index - 1] == ".") or (index + 1 < len(text) and text[index + 1] == "."):
        return False
    if index and index + 1 < len(text) and text[index - 1].isdigit() and text[index + 1].isdigit():
        return False
    return (index + 1 == len(text) or text[index + 1] in WHITE or text[index + 1] in CLOSERS
            or (index > 0 and index + 1 < len(text)
                and re.fullmatch(r"[\u3400-\u9fff]", text[index - 1]) is not None
                and re.fullmatch(r"[\u3400-\u9fff]", text[index + 1]) is not None))


def _make_segments(text: str, cuts: set[int]) -> list[dict]:
    """将边界对齐完整字素，再产生原文区间并连续编号。

    R09要求组合字素不可截断。标点后的ZWNJ/ZWJ/变体选择符仍属于该字素，
    因此候选切点右移至字素末尾；空白与附加符组成的字素也不可只裁掉空白。
    此处只修正原本会在validate失败的输入，不引入语义边界或改写原文。
    """
    edges = [0] + [match.end() for match in regex.finditer(r"\X", text)]
    points = sorted({edges[bisect_left(edges, point)] for point in cuts | {0, len(text)}})
    segments = []
    for left, right in zip(points, points[1:]):
        start, end = _trim(text, left, right)
        start = edges[bisect_right(edges, start) - 1]
        end = edges[bisect_left(edges, end)]
        if start < end:
            segments.append({"seg_id": len(segments) + 1, "raw_text": text[start:end], "start": start, "end": end})
    return segments


def validate(text: str, segments: list[dict]) -> None:
    """验证全部非空白字符恰好覆盖一次及字素边界；任一不变量破坏立即失败。"""
    cluster_edges = {0, len(text)} | {m.start() for m in regex.finditer(r"\X", text)} | {m.end() for m in regex.finditer(r"\X", text)}
    previous = 0
    for index, segment in enumerate(segments, 1):
        a, b = segment["start"], segment["end"]
        if not (segment["seg_id"] == index and previous <= a < b <= len(text)):
            raise ValueError("片段编号或区间非法")
        if not (a in cluster_edges and b in cluster_edges):
            raise ValueError("边界切入Unicode字素簇")
        if not all(c in WHITE for c in text[previous:a]):
            raise ValueError("丢失非空白字符")
        if segment["raw_text"] != text[a:b]:
            raise ValueError("片段不是原文连续子串")
        previous = b
    if not all(c in WHITE for c in text[previous:]):
        raise ValueError("尾部非空白字符未覆盖")


def segment(text: str) -> dict:
    """按冻结规则切分已投影正文，返回片段、位置和有序问题列表。

    输入为原样字符串；所有坐标使用Unicode码点左闭右开区间。空白间隙允许
    不产段，其余字符恰好覆盖一次，组合字素不可截断。非字符串抛TypeError，
    完整性失败抛ValueError。保留实验返回字段便于与冻结快照逐字段对照；
    公开接口仅暴露片段及诊断，不按主题、属性或情绪增加边界。
    """
    if not isinstance(text, str):
        raise TypeError("text必须为原始字符串")
    tokens, issues = _tokens(text)
    markers, list_cuts, list_issues = _lists(text, tokens)
    issues.extend(list_issues)
    tokens += markers
    pairs, pair_issues = _pairs(text, tokens)
    issues.extend(pair_issues)
    pair_spans = [(a, b) for a, b, _ in pairs]
    tags, tag_cuts, tag_issues = _tags(text, tokens + pair_spans)
    issues.extend(tag_issues)
    token_spans = tokens + tags
    protected = token_spans + pair_spans
    cuts = {p for p in list_cuts | tag_cuts if not _inside(p, protected)}
    for match in LINES.finditer(text):
        if not _inside(match.start(), protected):
            cuts.update(match.span())
    for index in range(len(text)):
        if _terminal_at(text, index, token_spans) and not _covered(index, pair_spans):
            if index + 1 < len(text) and text[index + 1] in STRONG:
                continue
            cuts.add(index + 1)
    for a, b, kind in pairs:
        if kind != "quote" or _inside(b, pair_spans):
            continue
        inner_end = b - 2
        while inner_end > a and text[inner_end] in WHITE:
            inner_end -= 1
        if inner_end <= a or not _terminal_at(text, inner_end, token_spans):
            continue
        end = b
        while end < len(text) and text[end] in STRONG:
            end += 1
        next_index = end
        while next_index < len(text) and text[next_index] in WHITE:
            next_index += 1
        if next_index == len(text) or text[next_index] not in "，,、：:":
            cuts.difference_update(range(b, end + 1))
            cuts.add(end)
    # 同行句末后的纯表情尾缀归前句，但不跨越作者显式换行。
    for line_a, line_b in _line_ranges(text):
        for cut in sorted(cuts):
            if line_a < cut < line_b and text[cut - 1] in STRONG and _emoji_only(text[cut:line_b]):
                cuts.discard(cut)
                cuts.add(line_b)
    base = _make_segments(text, cuts)
    segments = base
    for part in segments:
        raw = part["raw_text"]
        if len(raw) > 200:
            issues.append({"code": "LONG_SEGMENT", "start": part["start"], "end": part["end"]})
        if not regex.search(r"[\p{L}\p{N}]", raw):
            issues.append({"code": "SYMBOL_ONLY", "start": part["start"], "end": part["end"]})
    if not segments:
        issues.append({"code": "EMPTY_BODY", "start": 0, "end": len(text)})
    for match in re.finditer("[\u200b\ufeff]", text):
        issues.append({"code": "FORMAT_CHARACTER", "start": match.start(), "end": match.end()})
    validate(text, base)
    validate(text, segments)
    return {"text_segments": segments, "base_segments": base,
            "evaluation_added_boundaries": [],
            "issues": sorted(issues, key=lambda item: (item["start"], item["end"], item["code"])),
            "r10_coverage": "NOT_APPLICABLE_OBJECTIVE_STRUCTURE_ONLY"}
