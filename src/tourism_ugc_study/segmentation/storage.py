"""片段单表的SQLite持久化与只读查询，不读取正文源库或执行切分。

一行只表示一个片段；批次公共元数据由配对manifest保存。源post是外部库身份，
不能伪造SQLite跨库外键，其对应关系由导入与验收层逐条核验。
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Iterable

SCHEMA = """
CREATE TABLE text_segments (
    segment_id TEXT PRIMARY KEY NOT NULL CHECK(length(segment_id)=64),
    source_post_id INTEGER NOT NULL CHECK(source_post_id > 0),
    seg_no INTEGER NOT NULL CHECK(seg_no > 0),
    text TEXT NOT NULL CHECK(text != ''),
    start INTEGER NOT NULL CHECK(start >= 0),
    end INTEGER NOT NULL CHECK(end > start),
    batch_id TEXT NOT NULL CHECK(length(batch_id)=64),
    UNIQUE(batch_id, source_post_id, seg_no)
) STRICT;
"""


def segment_identifier(batch_id: str, post_id: int, seg_no: int) -> str:
    """由批次、源post和序号生成稳定身份；重复导入或重建不会重新编号。"""
    return hashlib.sha256(f"{batch_id}:{post_id}:{seg_no}".encode("ascii")).hexdigest()


def create_segment_database(path: Path) -> sqlite3.Connection:
    """在新路径创建唯一业务表并返回连接；路径已存在即拒绝。

    调用方负责关闭连接及包级发布事务。数据库使用DELETE日志，不更改来源库，
    不创建批次表、帖子表或sqlite_sequence。批量写入由调用方控制一个事务。
    """
    with path.open("xb"):
        path.chmod(0o600)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.executescript(SCHEMA)
        connection.execute("BEGIN IMMEDIATE")
        return connection
    except BaseException:
        connection.close()
        raise


def insert_segments(
    connection: sqlite3.Connection, batch_id: str, post_id: int, segments: Iterable[dict],
) -> None:
    """写入一帖的已校验片段，不自行提交；重复身份或序号由约束拒绝。

    输入沿用切分器seg_id/raw_text/start/end字段，在表中对应seg_no/text/start/end。
    不使用INSERT OR REPLACE，避免覆盖已存在的片段及其下游引用。
    """
    connection.executemany(
        "INSERT INTO text_segments VALUES (?,?,?,?,?,?,?)",
        ((segment_identifier(batch_id, post_id, s["seg_id"]), post_id, s["seg_id"],
          s["raw_text"], s["start"], s["end"], batch_id) for s in segments),
    )


def open_segment_database(path: Path) -> sqlite3.Connection:
    """只读打开已封存的片段文件；拒绝缺文件及活动日志，连接由调用方关闭。"""
    path = path.resolve(strict=True)
    for suffix in ("-wal", "-journal"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise ValueError("片段数据库存在活动日志")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def get_post_segments(connection: sqlite3.Connection, batch_id: str, post_id: int) -> list[dict]:
    """按联合索引读取指定批次、指定帖子的有序片段；缺成员返回空列表。

    空列表不能独自判断帖子不存在还是正文为空，成员资格由来源契约提供。
    此接口不解析源正文，也不允许调用方拼入任意SQL条件。
    """
    return [dict(row) for row in connection.execute(
        "SELECT * FROM text_segments WHERE batch_id=? AND source_post_id=? ORDER BY seg_no",
        (batch_id, post_id),
    )]
