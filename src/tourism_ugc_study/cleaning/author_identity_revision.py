"""从只读冻结库派生微博、抖音、知乎作者身份修订副本。

冻结文件不能原位改写。本模块只读打开来源和身份库，用不继承权限与文件标志的
方式复制来源，再在副本的一个事务里替换仍处于旧哈希和脱敏昵称的作者字段。
校验失败、哈希对不上或 JSON 不能按对象解析时删除暂存，不留下输出，也不改
来源、身份库、冻结登记或 ``current-*.json``。

身份恢复值只取指定列和指定 JSON 键。``updated_at``、``author_profile_url``
以及其他列、其他表保持复制进来的来源原值，避免把身份库里不同的正文或
``content_detail_source`` 等字段带进研究副本。

``drop_unresolved`` 为真时，同一事务在替换之后删除仍无法恢复的帖子，以及
``web_post_images``、``post_detail_repair_waivers`` 中指向这些帖的行。提交后
执行 ``VACUUM``，被删正文不得留在空闲页。为假时不删除、也不做这次重建。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TARGET_PLATFORMS = ("weibo", "douyin", "zhihu")
_MUTABLE_COLUMNS = (
    "author_display_name",
    "author_platform_id",
    "author_json",
    "raw_sample_json",
)
_REQUIRED_COLUMNS = (
    "id",
    "platform_key",
    "platform_post_id",
    *_MUTABLE_COLUMNS,
)
_RELATED_TABLES = (
    ("web_post_images", "web_post_id"),
    ("post_detail_repair_waivers", "web_post_id"),
)
_HASH_RE = re.compile(r"^[0-9a-f]{16}$")
_AUTHOR_JSON_KEYS = frozenset({"creator_hash", "nickname"})
_RAW_KEYS = {
    "weibo": frozenset({"creator_hash", "nickname"}),
    "douyin": frozenset({"creator_hash", "nickname"}),
    "zhihu": frozenset({"creator_hash", "user_nickname"}),
}
_STATUS = "DERIVED_NOT_FROZEN"


class _IdentityPost:
    """身份库中一条帖子的作者字段，不保存正文或其他列。"""

    __slots__ = (
        "author_display_name",
        "author_json",
        "author_platform_id",
        "raw_sample_json",
    )

    def __init__(
        self,
        author_platform_id: Any,
        author_display_name: Any,
        author_json: Any,
        raw_sample_json: Any,
    ) -> None:
        self.author_platform_id = author_platform_id
        self.author_display_name = author_display_name
        self.author_json = author_json
        self.raw_sample_json = raw_sample_json


def derive_author_identity_revision(
    source: Path,
    identity_db: Path,
    output: Path,
    manifest: Path,
    *,
    revision_id: str,
    reason: str,
    code_commit: str,
    drop_unresolved: bool = False,
) -> dict[str, Any]:
    """复制来源库并在副本上恢复可核对的作者身份，成功后写出 manifest。

    Args:
        source: 冻结或研究来源 SQLite。必须以只读方式可读，且不能有 WAL/SHM 旁文件。
        identity_db: 已恢复平台原值的身份库。只读打开，不要求与来源逐列相同。
        output: 派生库最终路径。必须不存在；发布前写入同目录临时文件。
        manifest: 派生清单路径。必须不存在。同目录另写 ``unresolved-posts.json``。
        revision_id: 本次派生标识，写入清单，不复用原冻结 ID。
        reason: 修订原因，原样写入清单。
        code_commit: 执行时的代码 commit。调用方负责从版本库读取。
        drop_unresolved: 为真时从副本删除 ``absent`` 与 ``still_masked`` 的帖子
            及其图片、详情豁免，提交后 ``VACUUM``。为假时保留这些行。

    Returns:
        已写入磁盘的 manifest 对象。状态固定为 ``DERIVED_NOT_FROZEN``。
        其中没有昵称或平台用户 ID。未替换帖的平台、帖子 ID 和 ``web_posts.id``
        在旁路 JSON 中。开启删除时清单另有删除计数和 ``drop_unresolved: true``。

    Raises:
        FileExistsError: 输出、manifest、未替换清单或对应临时文件已经存在。
        ValueError: 路径重叠、来源有旁文件、哈希不一致、JSON 非法、校验失败，
            或来源/身份库的权限、标志或字节在运行后发生变化。
        OSError: 复制或发布无法完成。
        sqlite3.Error: 副本或只读库无法按 SQLite 读取。

    失败时删除本次创建的临时文件和尚未完整发布的输出，不修改来源与身份库的
    字节、权限或标志。成功时输出版本权限为 ``0400``，不设置 ``uchg``。
    不写冻结登记或 ``current-*.json``。保留行的 ``id`` 不重新编号。
    """

    if not revision_id or not revision_id.strip():
        raise ValueError("author_identity_revision_id_missing")
    if not reason or not reason.strip():
        raise ValueError("author_identity_reason_missing")
    if not code_commit or not code_commit.strip():
        raise ValueError("author_identity_code_commit_missing")

    source_path = source.expanduser().resolve(strict=True)
    identity_path = identity_db.expanduser().resolve(strict=True)
    output_path = output.expanduser().resolve()
    manifest_path = manifest.expanduser().resolve()
    unresolved_path = manifest_path.parent / "unresolved-posts.json"
    temporary = Path(str(output_path) + ".tmp")
    _reject_paths(
        source_path,
        identity_path,
        output_path,
        manifest_path,
        unresolved_path,
        temporary,
    )
    _reject_source_sidecars(source_path)

    source_guard = _input_guard(source_path)
    identity_guard = _input_guard(identity_path)
    identity_index = _read_identity_index(identity_path)
    _assert_input_guard(identity_path, identity_guard, "identity_db")
    _assert_input_guard(source_path, source_guard, "source")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    published = False
    output_moved = False
    manifest_moved = False
    unresolved_moved = False
    manifest_partial: Path | None = None
    unresolved_partial: Path | None = None
    try:
        # copyfile 只复制字节。copy2/copystat 会连 0400 和 uchg 一起继承，
        # 副本将无法改写，也不能在发布时单独去掉用户不可变标志。
        shutil.copyfile(source_path, temporary)
        os.chmod(temporary, 0o600)
        copied_digest = _file_sha256(temporary)
        if copied_digest != source_guard[2]:
            raise ValueError("author_identity_copy_hash_mismatch")

        stats, unresolved, validation, dropped = _revise_copy(
            temporary,
            source_path,
            identity_index,
            drop_unresolved=drop_unresolved,
        )
        _assert_input_guard(source_path, source_guard, "source")
        _assert_input_guard(identity_path, identity_guard, "identity_db")
        if _sidecars(source_path):
            raise ValueError("author_identity_source_sidecar")
        _assert_no_sidecars(temporary)

        output_digest = _file_sha256(temporary)
        created_at = datetime.now(timezone.utc).isoformat()
        document = _manifest_document(
            revision_id=revision_id,
            reason=reason,
            code_commit=code_commit,
            created_at=created_at,
            source_path=source_path,
            source_digest=source_guard[2],
            source_size=source_guard[3],
            identity_path=identity_path,
            identity_digest=identity_guard[2],
            identity_size=identity_guard[3],
            output_path=output_path,
            output_digest=output_digest,
            output_size=temporary.stat().st_size,
            stats=stats,
            validation=validation,
        )
        if drop_unresolved:
            document["drop_unresolved"] = True
            document["dropped_posts"] = dropped["posts"]
            document["dropped_images"] = dropped["images"]
            document["dropped_waivers"] = dropped["waivers"]
        manifest_partial = _write_private_json(manifest_path.parent, document)
        unresolved_partial = _write_private_json(
            manifest_path.parent,
            {
                "revision_id": revision_id,
                "absent": unresolved["absent"],
                "still_masked": unresolved["still_masked"],
            },
        )
        _fsync(temporary)
        os.replace(temporary, output_path)
        output_moved = True
        os.chmod(output_path, 0o400)
        if (output_path.stat().st_mode & 0o777) != 0o400:
            raise ValueError("author_identity_output_mode_failed")
        if _file_sha256(output_path) != output_digest:
            raise ValueError("author_identity_output_hash_changed")
        os.replace(manifest_partial, manifest_path)
        manifest_moved = True
        os.replace(unresolved_partial, unresolved_path)
        unresolved_moved = True
        _assert_input_guard(source_path, source_guard, "source")
        _assert_input_guard(identity_path, identity_guard, "identity_db")
        published = True
    finally:
        if not published:
            if unresolved_moved:
                _unlink(unresolved_path)
            if manifest_moved:
                _unlink(manifest_path)
            if output_moved:
                _unlink(output_path)
            _unlink(temporary)
            for sidecar in _sidecars(temporary):
                _unlink(sidecar)
            _unlink(manifest_partial)
            _unlink(unresolved_partial)
    return document


def _reject_paths(
    source_path: Path,
    identity_path: Path,
    output_path: Path,
    manifest_path: Path,
    unresolved_path: Path,
    temporary: Path,
) -> None:
    """拒绝会覆盖冻结库、身份库或已有产物的路径。尚未复制任何文件。"""

    protected = {source_path, identity_path}
    products = (output_path, manifest_path, unresolved_path, temporary)
    if any(path in protected for path in products) or len(set(products)) != len(products):
        raise ValueError("author_identity_paths_overlap")
    if output_path.exists():
        raise FileExistsError("author_identity_output_exists")
    if manifest_path.exists():
        raise FileExistsError("author_identity_manifest_exists")
    if unresolved_path.exists():
        raise FileExistsError("author_identity_unresolved_exists")
    if temporary.exists():
        raise FileExistsError("author_identity_temporary_exists")


def _reject_source_sidecars(source_path: Path) -> None:
    """来源带 WAL/SHM 时，主文件字节不是完整已提交视图，不能当作冻结副本。"""

    if _sidecars(source_path):
        raise ValueError("author_identity_source_sidecar")


def _sidecars(path: Path) -> list[Path]:
    """返回 SQLite 可能遗留的旁文件；调用方决定是拒绝来源还是删除自己的暂存。"""

    return [candidate for suffix in ("-wal", "-shm", "-journal") if (candidate := Path(str(path) + suffix)).exists()]


def _input_guard(path: Path) -> tuple[int, int, str, int]:
    """记录模式、标志、SHA-256 和大小，供运行结束后确认只读输入没被碰到。"""

    info = path.stat()
    return info.st_mode, getattr(info, "st_flags", 0), _file_sha256(path), info.st_size


def _assert_input_guard(path: Path, guard: tuple[int, int, str, int], label: str) -> None:
    """只读库的权限、标志和字节都必须与开跑时一致，否则整次失败。"""

    mode, flags, digest, size = guard
    info = path.stat()
    if info.st_mode != mode or getattr(info, "st_flags", 0) != flags or info.st_size != size:
        raise ValueError("author_identity_input_metadata_changed:" + label)
    if _file_sha256(path) != digest:
        raise ValueError("author_identity_input_bytes_changed:" + label)


def _read_identity_index(identity_db: Path) -> dict[tuple[str, str], _IdentityPost]:
    """只读加载三平台帖子键。重复键无法判断该信哪一行，在复制前失败。"""

    connection = _connect_readonly(identity_db, immutable=False)
    try:
        _require_columns(connection, "identity_db")
        rows = connection.execute(
            """
            SELECT platform_key, platform_post_id, author_platform_id,
                   author_display_name, author_json, raw_sample_json
            FROM web_posts
            WHERE platform_key IN ('weibo', 'douyin', 'zhihu')
            """
        ).fetchall()
    finally:
        connection.close()
    index: dict[tuple[str, str], _IdentityPost] = {}
    for row in rows:
        key = _post_key(row["platform_key"], row["platform_post_id"])
        if key is None:
            continue
        if key in index:
            raise ValueError("author_identity_duplicate_post:" + key[0] + ":" + key[1])
        index[key] = _IdentityPost(
            row["author_platform_id"],
            row["author_display_name"],
            row["author_json"],
            row["raw_sample_json"],
        )
    return index


def _revise_copy(
    temporary: Path,
    source: Path,
    identity_index: Mapping[tuple[str, str], _IdentityPost],
    *,
    drop_unresolved: bool,
) -> tuple[dict[str, dict[str, int]], dict[str, list[dict[str, Any]]], dict[str, str], dict[str, int]]:
    """在副本事务中替换作者身份，按需删除未恢复帖，提交后再校验。

    ``VACUUM`` 放在提交之后：SQLite 拒绝在事务里重建数据库，而 ``DELETE``
    留下的空闲页仍可能含有被删正文。不删除时跳过重建，避免改动既有副本字节。
    """

    connection = _connect_copy(temporary)
    dropped = {"posts": 0, "images": 0, "waivers": 0}
    try:
        _require_columns(connection, "source_copy")
        mode = connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
        if str(mode).lower() != "delete":
            raise ValueError("author_identity_journal_mode_failed")
        connection.execute("PRAGMA foreign_keys=ON").fetchall()
        stats = _empty_stats()
        unresolved: dict[str, list[dict[str, Any]]] = {"absent": [], "still_masked": []}
        connection.execute("BEGIN IMMEDIATE")
        dropped_ids: list[int] = []
        try:
            _replace_authors(connection, identity_index, stats, unresolved)
            if drop_unresolved:
                dropped_ids = [
                    int(item["source_post_id"])
                    for item in (*unresolved["absent"], *unresolved["still_masked"])
                ]
                dropped = _drop_unresolved_posts(connection, dropped_ids)
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        if drop_unresolved:
            connection.execute("VACUUM")
            _assert_no_sidecars(temporary)
        validation = _validate_copy(
            connection,
            source,
            drop_unresolved=drop_unresolved,
            dropped_ids=dropped_ids if drop_unresolved else None,
        )
    finally:
        connection.close()
    for item in stats.values():
        if item["renamed"] > item["replaced"] or item["rows"] != (
            item["replaced"] + item["still_masked"] + item["absent"]
        ):
            raise ValueError("author_identity_count_inconsistent")
    if drop_unresolved and dropped["posts"] != (
        sum(item["still_masked"] + item["absent"] for item in stats.values())
    ):
        raise ValueError("author_identity_drop_count_mismatch")
    return stats, unresolved, validation, dropped


def _replace_authors(
    connection: sqlite3.Connection,
    identity_index: Mapping[tuple[str, str], _IdentityPost],
    stats: dict[str, dict[str, int]],
    unresolved: dict[str, list[dict[str, Any]]],
) -> None:
    """按帖子键决定缺席、仍脱敏或替换。哈希不符时让整个事务回滚。"""

    # sqlite3.Row 不能用键名 rowid 取值，所以查询里改成 stored_rowid。
    rows = connection.execute(
        """
        SELECT id, rowid AS stored_rowid, platform_key, platform_post_id, author_platform_id,
               author_display_name, author_json, raw_sample_json
        FROM web_posts
        WHERE platform_key IN ('weibo', 'douyin', 'zhihu')
        ORDER BY rowid
        """
    ).fetchall()
    for row in rows:
        if not _is_candidate(row["platform_key"], row["author_platform_id"], row["author_display_name"]):
            continue
        key = _post_key(row["platform_key"], row["platform_post_id"])
        if key is None:
            raise ValueError("author_identity_post_key_invalid")
        platform, post_id = key
        stats[platform]["rows"] += 1
        identity = identity_index.get(key)
        record = {
            "platform_key": platform,
            "platform_post_id": post_id,
            "source_post_id": _source_post_id(row["id"]),
        }
        if identity is None:
            stats[platform]["absent"] += 1
            unresolved["absent"].append(record)
            continue
        # 先识别“身份库也还没恢复”的行。旧哈希再拿去 sha256 不会等于自身，
        # 如果倒过来先做哈希核对，未恢复行会把整次运行打成失败。
        if _is_still_masked(identity, row["author_platform_id"]):
            stats[platform]["still_masked"] += 1
            unresolved["still_masked"].append(record)
            continue
        new_id = identity.author_platform_id
        if _author_id_digest(new_id) != row["author_platform_id"]:
            raise ValueError("author_identity_hash_mismatch:" + platform + ":" + post_id)
        new_name = identity.author_display_name
        if not isinstance(new_name, str):
            raise ValueError("author_identity_display_name_not_text:" + platform + ":" + post_id)
        author_json = _merge_author_json(row["author_json"], identity.author_json)
        raw_json = _merge_raw_json(platform, row["raw_sample_json"], identity.raw_sample_json)
        connection.execute(
            """
            UPDATE web_posts
            SET author_display_name = ?, author_platform_id = ?,
                author_json = ?, raw_sample_json = ?
            WHERE rowid = ?
            """,
            (new_name, new_id, author_json, raw_json, row["stored_rowid"]),
        )
        stats[platform]["replaced"] += 1
        if _mask_nickname(new_name) != row["author_display_name"]:
            stats[platform]["renamed"] += 1
    unresolved["absent"].sort(key=_post_sort_key)
    unresolved["still_masked"].sort(key=_post_sort_key)


def _drop_unresolved_posts(connection: sqlite3.Connection, post_ids: list[int]) -> dict[str, int]:
    """删除未恢复帖及指向它们的图片和详情豁免。

    外键已经打开。先删子表，否则删除 ``web_posts`` 会在语句结束时因仍被引用而回滚。
    表不存在时记 0；帖子删除数必须等于待删 ID 数，避免漏删后仍去发布。
    """

    if len(post_ids) != len(set(post_ids)):
        raise ValueError("author_identity_drop_id_duplicate")
    images = _delete_ids(connection, "web_post_images", "web_post_id", post_ids)
    waivers = _delete_ids(connection, "post_detail_repair_waivers", "web_post_id", post_ids)
    posts = _delete_ids(connection, "web_posts", "id", post_ids)
    if posts != len(post_ids):
        raise ValueError("author_identity_drop_count_mismatch")
    return {"posts": posts, "images": images, "waivers": waivers}


def _delete_ids(connection: sqlite3.Connection, table: str, column: str, ids: list[int]) -> int:
    """按列删除命中行。表不存在则视为没有关联行，不创建表。"""

    if not ids or not _table_exists(connection, table):
        return 0
    if column not in set(_column_names(connection, "main", table)):
        raise ValueError("author_identity_related_column_missing:" + table + ":" + column)
    quoted = _quote(table)
    before = connection.execute(f"SELECT count(*) FROM {quoted}").fetchone()[0]
    placeholders = ",".join("?" * len(ids))
    connection.execute(
        f"DELETE FROM {quoted} WHERE {_quote(column)} IN ({placeholders})",
        ids,
    )
    after = connection.execute(f"SELECT count(*) FROM {quoted}").fetchone()[0]
    return int(before - after)


def _validate_copy(
    connection: sqlite3.Connection,
    source: Path,
    *,
    drop_unresolved: bool,
    dropped_ids: list[int] | None,
) -> dict[str, str]:
    """对照 immutable 来源核对副本。任一差异都拒绝发布。"""

    quick = [row[0] for row in connection.execute("PRAGMA quick_check").fetchall()]
    if quick != ["ok"]:
        raise ValueError("author_identity_quick_check_failed")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise ValueError("author_identity_foreign_key_check_failed")

    source_uri = source.resolve().as_uri() + "?mode=ro&immutable=1"
    connection.execute("ATTACH DATABASE ? AS src", (source_uri,))
    try:
        if drop_unresolved:
            _assert_other_tables_identical(
                connection,
                skip=frozenset(table for table, _column in _RELATED_TABLES),
            )
            _assert_web_posts_preserved(connection, dropped_ids=dropped_ids or [])
            _assert_related_tables_without_dropped_posts(connection, dropped_ids or [])
            _assert_no_unresolved_masked_rows(connection)
        else:
            _assert_other_tables_identical(connection, skip=frozenset())
            _assert_web_posts_preserved(connection, dropped_ids=None)
    finally:
        connection.execute("DETACH DATABASE src")
    mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    if str(mode).lower() != "delete":
        raise ValueError("author_identity_journal_mode_failed")
    if not drop_unresolved:
        return {
            "quick_check": "ok",
            "foreign_key_check": "passed",
            "other_tables_bidirectional_except": "empty",
            "other_tables_row_counts": "equal",
            "web_posts_row_count": "equal",
            "web_posts_non_author_columns": "equal",
            "json_diff_keys": "allowed",
            "changed_row_rehash": "passed",
            "journal_mode": "delete",
        }
    return {
        "quick_check": "ok",
        "foreign_key_check": "passed",
        "other_tables_bidirectional_except": "empty",
        "other_tables_row_counts": "equal",
        "web_posts_row_count": "source_minus_dropped",
        "web_posts_non_author_columns": "kept_rows_equal",
        "related_tables_without_dropped_posts": "equal",
        "unresolved_masked_rows": "none",
        "json_diff_keys": "allowed",
        "changed_row_rehash": "passed",
        "journal_mode": "delete",
        "vacuum": "completed",
    }


def _assert_other_tables_identical(connection: sqlite3.Connection, *, skip: frozenset[str]) -> None:
    """除 web_posts 和调用方另行核对的表外，用户表必须双向差集为空且行数相同。"""

    main_tables = _user_tables(connection, "main")
    source_tables = _user_tables(connection, "src")
    if main_tables != source_tables:
        raise ValueError("author_identity_table_set_mismatch")
    for name in main_tables:
        if name == "web_posts" or name in skip:
            continue
        quoted = _quote(name)
        main_count = connection.execute(f"SELECT count(*) FROM main.{quoted}").fetchone()[0]
        source_count = connection.execute(f"SELECT count(*) FROM src.{quoted}").fetchone()[0]
        if main_count != source_count:
            raise ValueError("author_identity_table_count_mismatch:" + name)
        for statement in (
            f"SELECT * FROM main.{quoted} EXCEPT SELECT * FROM src.{quoted}",
            f"SELECT * FROM src.{quoted} EXCEPT SELECT * FROM main.{quoted}",
        ):
            if connection.execute(statement).fetchall():
                raise ValueError("author_identity_table_diff:" + name)


def _assert_web_posts_preserved(connection: sqlite3.Connection, *, dropped_ids: list[int] | None) -> None:
    """web_posts 只允许四列变化；JSON 变化不能波及作者键以外的内容。

    ``dropped_ids`` 为 None 表示未开启删除，行数必须与来源相同。给出列表时，
    行数等于来源减去该列表长度，且这些 ``id`` 不得留在副本中。非作者列只比较保留行。
    """

    main_count = connection.execute("SELECT count(*) FROM main.web_posts").fetchone()[0]
    source_count = connection.execute("SELECT count(*) FROM src.web_posts").fetchone()[0]
    if dropped_ids is None:
        if main_count != source_count:
            raise ValueError("author_identity_web_posts_count_mismatch")
        source_stable = "SELECT {fields} FROM src.web_posts"
    else:
        if main_count != source_count - len(dropped_ids):
            raise ValueError("author_identity_web_posts_count_mismatch")
        if dropped_ids:
            placeholders = ",".join("?" * len(dropped_ids))
            leftover = connection.execute(
                f"SELECT count(*) FROM main.web_posts WHERE id IN ({placeholders})",
                dropped_ids,
            ).fetchone()[0]
            if leftover:
                raise ValueError("author_identity_dropped_post_remains")
        source_stable = (
            "SELECT {fields} FROM src.web_posts WHERE id IN (SELECT id FROM main.web_posts)"
        )
    columns = _column_names(connection, "main", "web_posts")
    stable = [name for name in columns if name not in _MUTABLE_COLUMNS]
    if stable:
        fields = ", ".join(_quote(name) for name in stable)
        main_stable = f"SELECT {fields} FROM main.web_posts"
        compared = source_stable.format(fields=fields)
        for statement in (
            f"{main_stable} EXCEPT {compared}",
            f"{compared} EXCEPT {main_stable}",
        ):
            if connection.execute(statement).fetchall():
                raise ValueError("author_identity_non_author_column_diff")
    paired = connection.execute(
        """
        SELECT m.rowid, m.platform_key, s.platform_key, m.platform_post_id, s.platform_post_id,
               m.author_display_name, s.author_display_name, m.author_platform_id, s.author_platform_id,
               m.author_json, s.author_json, m.raw_sample_json, s.raw_sample_json
        FROM main.web_posts AS m
        JOIN src.web_posts AS s ON m.rowid = s.rowid
        """
    ).fetchall()
    if len(paired) != main_count:
        raise ValueError("author_identity_web_posts_row_alignment_failed")
    for row in paired:
        if row[1] != row[2] or row[3] != row[4]:
            raise ValueError("author_identity_post_key_moved")
        platform = row[1]
        new_name, old_name = row[5], row[6]
        new_id, old_id = row[7], row[8]
        new_author, old_author = row[9], row[10]
        new_raw, old_raw = row[11], row[12]
        changed = not (new_name == old_name and new_id == old_id and new_author == old_author and new_raw == old_raw)
        if not changed:
            continue
        if platform not in _TARGET_PLATFORMS or _author_id_digest(new_id) != old_id:
            raise ValueError("author_identity_rehash_failed")
        if not _json_text_diff_allowed(old_author, new_author, value_keys=_AUTHOR_JSON_KEYS, removable_keys=frozenset()):
            raise ValueError("author_identity_json_diff_forbidden")
        raw_keys = _RAW_KEYS[platform]
        removable = frozenset({"nickname"}) if platform == "zhihu" else frozenset()
        if not _json_text_diff_allowed(old_raw, new_raw, value_keys=raw_keys, removable_keys=removable):
            raise ValueError("author_identity_json_diff_forbidden")


def _assert_related_tables_without_dropped_posts(connection: sqlite3.Connection, dropped_ids: list[int]) -> None:
    """图片和豁免只少掉被删帖的关联行，保留行的每一列与来源相同。"""

    for table, column in _RELATED_TABLES:
        if not _table_exists(connection, table):
            continue
        if column not in set(_column_names(connection, "main", table)):
            raise ValueError("author_identity_related_column_missing:" + table + ":" + column)
        quoted = _quote(table)
        quoted_column = _quote(column)
        params: list[int] = []
        where = "1"
        if dropped_ids:
            placeholders = ",".join("?" * len(dropped_ids))
            where = f"({quoted_column} IS NULL OR {quoted_column} NOT IN ({placeholders}))"
            params = list(dropped_ids)
        main_count = connection.execute(f"SELECT count(*) FROM main.{quoted}").fetchone()[0]
        kept_count = connection.execute(
            f"SELECT count(*) FROM src.{quoted} WHERE {where}",
            params,
        ).fetchone()[0]
        if main_count != kept_count:
            raise ValueError("author_identity_related_table_count:" + table)
        for statement in (
            f"SELECT * FROM main.{quoted} EXCEPT SELECT * FROM src.{quoted} WHERE {where}",
            f"SELECT * FROM src.{quoted} WHERE {where} EXCEPT SELECT * FROM main.{quoted}",
        ):
            if connection.execute(statement, params).fetchall():
                raise ValueError("author_identity_related_table_diff:" + table)


def _assert_no_unresolved_masked_rows(connection: sqlite3.Connection) -> None:
    """删除之后，三平台不得再留下旧哈希加脱敏昵称的行。"""

    rows = connection.execute(
        """
        SELECT platform_key, author_platform_id, author_display_name
        FROM web_posts
        WHERE platform_key IN ('weibo', 'douyin', 'zhihu')
        """
    )
    for row in rows:
        if _is_candidate(row["platform_key"], row["author_platform_id"], row["author_display_name"]):
            raise ValueError("author_identity_unresolved_row_remains")


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM main.sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _source_post_id(value: Any) -> int:
    """未恢复清单只用 ``web_posts.id``。布尔值在 Python 里是 int 的子类，不能当成帖子 ID。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("author_identity_source_post_id_invalid")
    return value


def _manifest_document(
    *,
    revision_id: str,
    reason: str,
    code_commit: str,
    created_at: str,
    source_path: Path,
    source_digest: str,
    source_size: int,
    identity_path: Path,
    identity_digest: str,
    identity_size: int,
    output_path: Path,
    output_digest: str,
    output_size: int,
    stats: Mapping[str, Mapping[str, int]],
    validation: Mapping[str, str],
) -> dict[str, Any]:
    """组装不含昵称和平台用户 ID 的派生清单。"""

    return {
        "revision_id": revision_id,
        "reason": reason,
        "status": _STATUS,
        "created_at_utc": created_at,
        "code_commit": code_commit,
        "source": _file_record(source_path, source_digest, source_size),
        "identity_db": _file_record(identity_path, identity_digest, identity_size),
        "output": _file_record(output_path, output_digest, output_size),
        "platforms": {platform: dict(stats[platform]) for platform in _TARGET_PLATFORMS},
        "validation": dict(validation),
    }


def _file_record(path: Path, digest: str, size: int) -> dict[str, Any]:
    return {"path": str(path), "sha256": digest, "size_bytes": size}


def _connect_readonly(path: Path, *, immutable: bool) -> sqlite3.Connection:
    """按调用方要求的只读 URI 打开。immutable 用于不再接受旁文件的冻结来源。"""

    suffix = "?mode=ro&immutable=1" if immutable else "?mode=ro"
    connection = sqlite3.connect(path.resolve().as_uri() + suffix, uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON").fetchall()
    return connection


def _connect_copy(path: Path) -> sqlite3.Connection:
    """打开可写副本。显式 autocommit，避免隐式事务包住 journal_mode。"""

    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.isolation_level = None
    return connection


def _require_columns(connection: sqlite3.Connection, label: str) -> None:
    columns = set(_column_names(connection, "main", "web_posts"))
    missing = [name for name in _REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise ValueError("author_identity_columns_missing:" + label + ":" + ",".join(missing))


def _column_names(connection: sqlite3.Connection, schema: str, table: str) -> list[str]:
    rows = connection.execute(f"PRAGMA {schema}.table_info({_quote(table)})").fetchall()
    if not rows:
        raise ValueError("author_identity_table_missing:" + table)
    return [str(row[1]) for row in rows]


def _user_tables(connection: sqlite3.Connection, schema: str) -> list[str]:
    rows = connection.execute(
        f"SELECT name FROM {schema}.sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [str(row[0]) for row in rows]


def _is_candidate(platform: Any, author_id: Any, display_name: Any) -> bool:
    return (
        platform in _TARGET_PLATFORMS
        and isinstance(author_id, str)
        and _HASH_RE.fullmatch(author_id) is not None
        and _is_masked_nickname(display_name)
    )


def _is_still_masked(identity: _IdentityPost, old_author_id: str) -> bool:
    """身份库仍保存本行旧哈希，且昵称仍是脱敏形态时，不把该行当成已恢复。"""

    return identity.author_platform_id == old_author_id and _is_masked_nickname(identity.author_display_name)


def _is_masked_nickname(value: Any) -> bool:
    """识别旧脱敏函数的三种输出，不尝试还原原文。"""

    if not isinstance(value, str):
        return False
    if value == "*":
        return True
    if len(value) == 2 and value[1] == "*":
        return True
    return len(value) == 5 and value[1:4] == "***"


def _mask_nickname(nickname: str) -> str:
    """复现旧脱敏：不超过 1 字为 *，2 字为首字加 *，更长为首字、*** 与尾字。"""

    if len(nickname) <= 1:
        return "*"
    if len(nickname) == 2:
        return nickname[0] + "*"
    return nickname[0] + "***" + nickname[-1]


def _author_id_digest(platform_user_id: Any) -> str:
    """计算 sha256(str(id).strip()) 的十六进制前 16 位，用于核对恢复后的 ID。"""

    if not isinstance(platform_user_id, str):
        raise ValueError("author_identity_id_not_text")
    return hashlib.sha256(platform_user_id.strip().encode("utf-8")).hexdigest()[:16]


def _merge_author_json(source_text: Any, identity_text: Any) -> str:
    """只替换 creator_hash 与 nickname，其余键保留来源对象。"""

    source_obj = _load_object(source_text, "author_json")
    identity_obj = _load_object(identity_text, "identity_author_json")
    merged = dict(source_obj)
    merged["creator_hash"] = _required_json_value(identity_obj, "creator_hash", "author_json")
    merged["nickname"] = _required_json_value(identity_obj, "nickname", "author_json")
    return _dump_object(merged, "author_json")


def _merge_raw_json(platform: str, source_text: Any, identity_text: Any) -> str:
    """按平台替换 raw_sample 的作者键。知乎删除历史残留的 nickname，不从身份库抄回。"""

    source_obj = _load_object(source_text, "raw_sample_json")
    identity_obj = _load_object(identity_text, "identity_raw_sample_json")
    merged = dict(source_obj)
    merged["creator_hash"] = _required_json_value(identity_obj, "creator_hash", "raw_sample_json")
    if platform == "zhihu":
        merged["user_nickname"] = _required_json_value(identity_obj, "user_nickname", "raw_sample_json")
        merged.pop("nickname", None)
    else:
        merged["nickname"] = _required_json_value(identity_obj, "nickname", "raw_sample_json")
    return _dump_object(merged, "raw_sample_json")


def _load_object(value: Any, field: str) -> dict[str, Any]:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("author_identity_json_invalid:" + field) from exc
    if not isinstance(value, str):
        raise ValueError("author_identity_json_invalid:" + field)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("author_identity_json_invalid:" + field) from exc
    if not isinstance(parsed, dict):
        raise ValueError("author_identity_json_not_object:" + field)
    return parsed


def _required_json_value(obj: Mapping[str, Any], key: str, field: str) -> Any:
    if key not in obj:
        raise ValueError("author_identity_json_key_missing:" + field + ":" + key)
    return obj[key]


def _dump_object(value: Mapping[str, Any], field: str) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("author_identity_json_invalid:" + field) from exc


def _json_text_diff_allowed(
    before: Any,
    after: Any,
    *,
    value_keys: frozenset[str],
    removable_keys: frozenset[str],
) -> bool:
    """解析后比较。允许指定键改值；知乎 raw_sample 还允许删除残留 nickname。"""

    if before == after:
        return True
    if not isinstance(before, str) or not isinstance(after, str):
        return False
    try:
        old = json.loads(before)
        new = json.loads(after)
    except json.JSONDecodeError:
        return False
    if not isinstance(old, dict) or not isinstance(new, dict):
        return False
    removed = set(old) - set(new)
    added = set(new) - set(old)
    if not removed <= removable_keys or not added <= value_keys:
        return False
    for key in set(old) & set(new):
        if old[key] != new[key] and key not in value_keys:
            return False
    return True


def _post_key(platform: Any, post_id: Any) -> tuple[str, str] | None:
    if platform not in _TARGET_PLATFORMS:
        return None
    if isinstance(post_id, str) and post_id:
        return platform, post_id
    if isinstance(post_id, int) and not isinstance(post_id, bool):
        return platform, str(post_id)
    return None


def _post_sort_key(item: Mapping[str, Any]) -> tuple[str, str]:
    return str(item["platform_key"]), str(item["platform_post_id"])


def _empty_stats() -> dict[str, dict[str, int]]:
    return {
        platform: {"rows": 0, "replaced": 0, "renamed": 0, "still_masked": 0, "absent": 0}
        for platform in _TARGET_PLATFORMS
    }


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _fsync(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _write_private_json(directory: Path, payload: Mapping[str, Any]) -> Path:
    """先写同目录临时文件。调用方在副本就绪后用 replace 发布，避免半份清单。"""

    descriptor, name = tempfile.mkstemp(prefix=".author-identity-", suffix=".json", dir=directory)
    path = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o600)
    except Exception:
        _unlink(path)
        raise
    return path


def _assert_no_sidecars(path: Path) -> None:
    if _sidecars(path):
        raise ValueError("author_identity_sidecar_present")


def _unlink(path: Path | None) -> None:
    if path is not None and path.exists():
        path.unlink()
