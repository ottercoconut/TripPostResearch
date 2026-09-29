"""独立正文投影：保留普通正文，按冻结Delta契约拼接文本insert。

从2026-09-08验收使用的project_structured_text抽出，不导入清洗、模型或配置
加载器。非文本嵌入仅计数；损坏结构失败关闭。这里不做NFKC、空白折叠、
emoji替换或标题拼接；投影坐标只相对可读正文，不代表源JSON位置。
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping

PROJECTION_ID = "readable-body-structure-only-v1"
IGNORED_EMBED_TYPES = frozenset({"article-card", "cut-off", "image", "native-image", "video-card"})


@dataclass(frozen=True)
class StructuredTextProjection:
    """内部投影结果；invalid的text不可用，计数保留失败前结构诊断。"""

    text: object
    format_id: str
    status: str
    reason_code: str
    text_insert_count: int
    ignored_embed_count: int


def project_structured_text(
    value: object,
) -> StructuredTextProjection:
    """把受支持的 Delta JSON 字段确定性投影为纯文本。

    Args:
        value: 源标题或正文；普通值保持原样。

    Returns:
        纯文本或原始普通值，以及不含正文的结构证据。

    Notes:
        仅当对象声明 ``ops`` 时才视为结构化正文。文本 ``insert`` 按原顺序
        拼接；图片和截断节点不提供文本信息，因此只计数而不生成模型 token。
        疑似 Delta 但无法解析、操作结构非法或包含未知嵌入时失败关闭。
    """

    if not isinstance(value, str):
        return StructuredTextProjection(value, "plain_text", "usable", "plain_text", 0, 0)
    # 采集字段可能在 JSON 前带 BOM 或零宽格式符；这些字符不是正文，也不能
    # 让结构文档降级为普通文本。这里只移动探测起点，不改写 JSON 内部字符。
    structured_start = 0
    while structured_start < len(value):
        character = value[structured_start]
        if character.isspace() or unicodedata.category(character) in {"Cf", "Cc"}:
            structured_start += 1
            continue
        break
    stripped = value[structured_start:].strip()
    operations_key = "ops"
    marker_pattern = re.compile(rf'["\']{re.escape(operations_key)}["\']\s*:')
    if not stripped.startswith("{"):
        return StructuredTextProjection(value, "plain_text", "usable", "plain_text", 0, 0)
    try:
        parsed: Any = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        if marker_pattern.search(stripped):
            return StructuredTextProjection(
                "", "quill_delta_json", "invalid",
                "structured_text_malformed", 0, 0,
            )
        return StructuredTextProjection(value, "plain_text", "usable", "plain_text", 0, 0)
    if not isinstance(parsed, Mapping) or operations_key not in parsed:
        return StructuredTextProjection(value, "plain_text", "usable", "plain_text", 0, 0)
    operations = parsed.get(operations_key)
    if not isinstance(operations, list):
        return StructuredTextProjection(
            "", "quill_delta_json", "invalid",
            "structured_text_operations_invalid", 0, 0,
        )
    pieces: list[str] = []
    text_count = 0
    embed_count = 0
    for operation in operations:
        if (
            not isinstance(operation, Mapping)
            or "insert" not in operation
            or set(operation) - {"insert", "attributes"}
            or (
                "attributes" in operation
                and not isinstance(operation["attributes"], Mapping)
            )
        ):
            return StructuredTextProjection(
                "", "quill_delta_json", "invalid",
                "structured_text_operation_invalid", text_count, embed_count,
            )
        insert = operation["insert"]
        if isinstance(insert, str):
            pieces.append(insert)
            text_count += 1
            continue
        if isinstance(insert, Mapping) and len(insert) == 1:
            embed_type = next(iter(insert))
            if str(embed_type) in IGNORED_EMBED_TYPES:
                embed_count += 1
                continue
        return StructuredTextProjection(
            "", "quill_delta_json", "invalid",
            "structured_text_embed_unsupported", text_count, embed_count,
        )
    projected_text = "".join(pieces)
    # 与冻结清洗投影的空正文判定等价；仅用于检查，绝不改写返回文本。
    has_text = any(
        not character.isspace()
        and unicodedata.category(character) not in {"Cf", "Cc"}
        and character not in {"\ufe0e", "\ufe0f"}
        for character in projected_text
    )
    if not has_text:
        return StructuredTextProjection(
            "", "quill_delta_json", "invalid",
            "structured_text_no_text", text_count, embed_count,
        )
    return StructuredTextProjection(
        projected_text, "quill_delta_json", "usable",
        "structured_text_extracted", text_count, embed_count,
    )

