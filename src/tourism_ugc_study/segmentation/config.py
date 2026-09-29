"""切分输入配置解析；仅定位已授权的研究数据，不推断或发布keep成员。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResearchInput:
    """明确绑定成员库、正文快照和配对manifest的只读输入路径。"""

    membership_db: Path
    source_db: Path
    candidate_manifest: Path
    candidate_manifest_sha256: str


def load_research_input(path: Path) -> ResearchInput:
    """读取JSON配置，路径相对配置文件，拒绝未知字段和未绑定的manifest。

    文件不存在、JSON非法或字段错误均抛异常；本函数不打开数据库。具体数据库
    内容及摘要由source适配层验证，不能用配置中的路径代替完整性验收。
    """
    path = path.resolve(strict=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    fields = {"membership_db", "source_db", "candidate_manifest", "candidate_manifest_sha256"}
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError("切分配置字段不匹配")
    if any(not isinstance(v, str) or not v.strip() for v in raw.values()):
        raise ValueError("切分配置字段必须为非空字符串")
    if not re.fullmatch(r"[0-9a-f]{64}", raw["candidate_manifest_sha256"]):
        raise ValueError("candidate_manifest_sha256无效")
    return ResearchInput(
        **{key: (path.parent / raw[key]).resolve(strict=True)
           for key in fields - {"candidate_manifest_sha256"}},
        candidate_manifest_sha256=raw["candidate_manifest_sha256"],
    )
