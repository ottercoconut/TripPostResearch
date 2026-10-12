"""算法无关的原文片段完整性校验，仅依赖标准库与固定版本regex。

空白集合沿用已归档客观引擎；规则替换不能改变已发布数据的覆盖含义。
输入为可读正文及seg_id/raw_text/start/end字典；失败抛ValueError，不修复原文。
"""

from __future__ import annotations

import regex

WHITE = frozenset("\t\n\v\f\r \x85\xa0\u1680\u2028\u2029\u202f\u205f\u3000" + "".join(chr(i) for i in range(0x2000, 0x200B)))


def validate(text: str, segments: list[dict]) -> None:
    """校验整数编号、完整字素端点及全部非空白字符恰好覆盖一次。

    允许片段间仅空白的间隙；空正文可无片段。使用显式异常而非assert，
    避免Python优化模式关闭数据库和纯函数的完整性守卫。
    """
    cluster_edges = {0, len(text)} | {m.start() for m in regex.finditer(r"\X", text)} | {m.end() for m in regex.finditer(r"\X", text)}
    previous = 0
    for index, segment in enumerate(segments, 1):
        if (not isinstance(segment, dict)
                or set(segment) != {"seg_id", "raw_text", "start", "end"}
                or any(type(segment[k]) is not int for k in ("seg_id", "start", "end"))
                or not isinstance(segment["raw_text"], str)):
            raise ValueError("片段字段或类型非法")
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
