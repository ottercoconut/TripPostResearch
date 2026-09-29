"""管理端研究语料的只读适配；只消费keep成员及配对正文快照。

输入为封存SQLite及其manifest，不访问Admin账号、任务或模型。先验摘要与
无活动journal保证immutable读取不会遗漏WAL；结束后再次校验防止运行中换源。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .config import ResearchInput


def file_sha256(path: Path) -> str:
    """流式读取文件摘要；文件不可读直接失败，不返回不完整摘要。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _require_sealed(path: Path, expected: str) -> None:
    """仅接收完整封存文件；存在有效WAL/journal或摘要变化立即拒绝。"""
    for suffix in ("-wal", "-journal"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise ValueError(f"数据库存在活动日志：{path.name}")
    if file_sha256(path) != expected:
        raise ValueError(f"数据库摘要不匹配：{path.name}")


@dataclass(frozen=True)
class ResearchPost:
    """单个keep成员及原始正文；身份属于同一来源快照，标题不进入切分。"""

    source_post_id: int
    source_version: int
    body: str | None


@dataclass(frozen=True)
class ResearchCorpus:
    """只读上下文内有效的单次迭代器和已核实的来源身份，不泄露连接对象。"""

    identity: dict[str, object]
    posts: Iterator[ResearchPost]


def _validate(connection: sqlite3.Connection, manifest: dict) -> int:
    """核验成员计数、元数据、版本及原文一致性；返回keep数量。

    使用LEFT JOIN主动检查缺失正文记录，不能让内连接静默丢失成员。SQL固定，
    不接受调用方拼入表名或任意筛选条件。任何矛盾抛ValueError或sqlite错误。
    """
    count = connection.execute("SELECT count(*) FROM cleaning_candidates").fetchone()[0]
    counts = dict(connection.execute("SELECT decision,count(*) FROM cleaning_candidates GROUP BY decision"))
    expected = manifest.get("decision_counts")
    if (type(manifest.get("record_count")) is not int or manifest["record_count"] != count
            or not isinstance(expected, dict) or set(expected) != {"keep", "exclude", "manual_review"}
            or any(type(v) is not int or v < 0 for v in expected.values())
            or counts != {k: v for k, v in expected.items() if v}):
        raise ValueError("候选成员计数与manifest不一致")
    duplicates = connection.execute(
        "SELECT count(*) - count(DISTINCT source_post_id) FROM cleaning_candidates"
    ).fetchone()[0]
    mismatches = connection.execute("""
        SELECT count(*) FROM cleaning_candidates c
        LEFT JOIN source.web_posts p ON p.id=c.source_post_id
        WHERE p.id IS NULL OR typeof(c.source_post_id) != 'integer'
           OR typeof(c.source_version) != 'integer' OR c.source_version < 1
           OR c.platform_key IS NOT p.platform_key
           OR COALESCE(c.title, '') != COALESCE(p.title, '')
           OR COALESCE(c.body, '') != COALESCE(p.content_text, '')
           OR (p.content_text IS NOT NULL AND typeof(p.content_text) != 'text')
    """).fetchone()[0]
    if duplicates or mismatches:
        raise ValueError("候选成员身份或原始正文与来源快照不一致")
    # 与管理端契约一致：候选构建把空标题投影成空串，源快照仍可能保存NULL。
    # 此等价仅用于配对核验；实际切分始终读取p.content_text，保留源NULL身份。
    metadata = dict(connection.execute("SELECT key,value_json FROM round_metadata"))
    for key in ("status", "round_name", "source_snapshot_sha256", "round_manifest_sha256"):
        if json.loads(metadata.get(key, "null")) != manifest[key]:
            raise ValueError(f"候选库元数据不匹配：{key}")
    return counts.get("keep", 0)


@contextmanager
def open_research_corpus(config: ResearchInput) -> Iterator[ResearchCorpus]:
    """在只读事务中产生与管理端相同的decision=keep语料。

    manifest绑定数据库和源快照SHA，读取前后均核验。只接受已封存文件，
    不创建缺失数据库、不执行迁移、不修改清洗决定。发生漂移或来源不一致
    抛异常；调用方须在上下文正常退出后才发布运行产物。
    """
    raw = config.candidate_manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config.candidate_manifest_sha256:
        raise ValueError("候选manifest摘要与配置不一致")
    manifest = json.loads(raw)
    if (not isinstance(manifest, dict)
            or manifest.get("artifact_kind") != "full-research-cleaning-candidates"
            or manifest.get("status") != "CANDIDATES_READY_AWAITING_HUMAN_REVIEW"
            or manifest.get("database_filename") != config.membership_db.name):
        raise ValueError("不支持的候选manifest或数据库名称")
    for path, expected in ((config.membership_db, manifest["database_sha256"]),
                           (config.source_db, manifest["source_snapshot_sha256"])):
        _require_sealed(path, expected)
    connection = sqlite3.connect(config.membership_db.as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("ATTACH DATABASE ? AS source", (config.source_db.as_uri() + "?mode=ro&immutable=1",))
        connection.execute("BEGIN")
        keep_count = _validate(connection, manifest)
        cursor = connection.execute("""
            SELECT c.source_post_id,c.source_version,p.content_text
            FROM cleaning_candidates c JOIN source.web_posts p ON p.id=c.source_post_id
            WHERE c.decision='keep' ORDER BY c.source_post_id
        """)
        yield ResearchCorpus(
            identity={
                "dataset_kind": "cleaning_candidates",
                "selection": "decision=keep",
                "human_final_review_status": "pending",
                "round_name": manifest["round_name"],
                "record_count": keep_count,
                "candidate_manifest_sha256": config.candidate_manifest_sha256,
                "membership_db_sha256": manifest["database_sha256"],
                "source_snapshot_sha256": manifest["source_snapshot_sha256"],
            },
            posts=(ResearchPost(*row) for row in cursor),
        )
    finally:
        connection.close()
    # 检查发生在调用方写manifest之前；不能把中途换源的结果标为完整运行。
    if file_sha256(config.candidate_manifest) != config.candidate_manifest_sha256:
        raise ValueError("候选manifest在运行中发生变化")
    _require_sealed(config.membership_db, manifest["database_sha256"])
    _require_sealed(config.source_db, manifest["source_snapshot_sha256"])
