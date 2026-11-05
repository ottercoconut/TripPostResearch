"""按未恢复帖子清单从只读库派生排除副本。

清洗候选和切分库不在作者身份修订的同一文件里。本模块只读打开来源，用
不继承权限和文件标志的方式复制，再按调用方给出的表列删除命中行。其他表
和保留行逐列保持原值；指向被删帖的引用只计数，不改写。失败时不留输出，
也不修改来源、冻结登记或 ``current-*.json``。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_STATUS = "DERIVED_NOT_FROZEN"
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REFERENCE_COLUMN = re.compile(r"^(exact_representative_id|web_post_id|post_id|.+_post_id|.+_representative_id)$")
_UNCHANGED_TABLES_NOTE = (
    "未列入 table-column 的表（包括 round_metadata）保持原样，其中的计数可能是删除前的值。"
)


def derive_post_exclusion(
    source: Path,
    exclusions: Sequence[Path],
    table_columns: Sequence[str],
    output: Path,
    manifest: Path,
    *,
    revision_id: str,
    reason: str,
    code_commit: str,
) -> dict[str, Any]:
    """复制来源并删除排除清单命中的行，成功后写出不含昵称或平台用户 ID 的 manifest。

    Args:
        source: 只读 SQLite。不能有 WAL、SHM 或 journal 旁文件。
        exclusions: 一个或多个 ``unresolved-posts.json``。只读取其中
            ``absent`` 与 ``still_masked`` 的 ``source_post_id``。
        table_columns: ``表:列`` 列表，例如 ``cleaning_candidates:source_post_id``。
            只删除这些列命中排除 ID 的行。
        output: 派生库最终路径。必须不存在；发布前写入同目录临时文件。
        manifest: 派生清单路径。必须不存在。
        revision_id: 本次派生标识，写入清单，不复用原冻结 ID。
        reason: 修订原因，原样写入清单。
        code_commit: 执行时的代码 commit。调用方负责从版本库读取。

    Returns:
        已写入磁盘的 manifest。状态为 ``DERIVED_NOT_FROZEN``。
        ``dangling_references`` 统计保留行里仍指向被删帖的引用列。
        ``unchanged_tables_note`` 说明未指定表中的计数可能是删除前的值。

    Raises:
        FileExistsError: 输出、manifest 或临时文件已经存在。
        ValueError: 路径重叠、来源有旁文件、排除清单无法解析、表列不存在、
            校验失败，或来源的权限、标志或字节在运行后发生变化。
        OSError: 复制或发布无法完成。
        sqlite3.Error: 副本或只读库无法按 SQLite 读取。

    保留行的 ``decision``、簇字段和其他列不修改。失败时删除本次临时文件和
    尚未完整发布的输出。成功时输出权限为 ``0400``，不设置 ``uchg``。
    """

    if not revision_id or not revision_id.strip():
        raise ValueError("post_exclusion_revision_id_missing")
    if not reason or not reason.strip():
        raise ValueError("post_exclusion_reason_missing")
    if not code_commit or not code_commit.strip():
        raise ValueError("post_exclusion_code_commit_missing")
    pairs = _parse_table_columns(table_columns)
    if not exclusions:
        raise ValueError("post_exclusion_exclusions_missing")

    source_path = source.expanduser().resolve(strict=True)
    exclusion_paths = [item.expanduser().resolve(strict=True) for item in exclusions]
    output_path = output.expanduser().resolve()
    manifest_path = manifest.expanduser().resolve()
    temporary = Path(str(output_path) + ".tmp")
    _reject_paths(source_path, exclusion_paths, output_path, manifest_path, temporary)
    _reject_source_sidecars(source_path)

    source_guard = _input_guard(source_path)
    excluded_ids, exclusion_records, exclusion_guards = _read_exclusions(exclusion_paths)
    _assert_targets(source_path, pairs)
    _assert_input_guard(source_path, source_guard, "source")
    _assert_exclusion_guards(exclusion_paths, exclusion_guards)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    published = False
    output_moved = False
    manifest_moved = False
    manifest_partial: Path | None = None
    try:
        # copyfile 只复制字节。copy2 会连来源的 0400 和 uchg 一起继承，副本将无法改写。
        shutil.copyfile(source_path, temporary)
        os.chmod(temporary, 0o600)
        if _file_sha256(temporary) != source_guard[2]:
            raise ValueError("post_exclusion_copy_hash_mismatch")
        dropped_rows, dangling, validation = _exclude_copy(temporary, source_path, pairs, excluded_ids)
        _assert_input_guard(source_path, source_guard, "source")
        _assert_exclusion_guards(exclusion_paths, exclusion_guards)
        if _sidecars(source_path) or _sidecars(temporary):
            raise ValueError("post_exclusion_sidecar_present")

        output_digest = _file_sha256(temporary)
        document = _manifest_document(
            revision_id=revision_id,
            reason=reason,
            code_commit=code_commit,
            created_at=datetime.now(timezone.utc).isoformat(),
            source_path=source_path,
            source_guard=source_guard,
            exclusion_records=exclusion_records,
            excluded_ids=excluded_ids,
            pairs=pairs,
            dropped_rows=dropped_rows,
            dangling=dangling,
            output_path=output_path,
            output_digest=output_digest,
            output_size=temporary.stat().st_size,
            validation=validation,
        )
        manifest_partial = _write_private_json(manifest_path.parent, document)
        _fsync(temporary)
        os.replace(temporary, output_path)
        output_moved = True
        os.chmod(output_path, 0o400)
        if (output_path.stat().st_mode & 0o777) != 0o400:
            raise ValueError("post_exclusion_output_mode_failed")
        if _file_sha256(output_path) != output_digest:
            raise ValueError("post_exclusion_output_hash_changed")
        os.replace(manifest_partial, manifest_path)
        manifest_moved = True
        _assert_input_guard(source_path, source_guard, "source")
        _assert_exclusion_guards(exclusion_paths, exclusion_guards)
        published = True
    finally:
        if not published:
            if manifest_moved:
                _unlink(manifest_path)
            if output_moved:
                _unlink(output_path)
            _unlink(temporary)
            for sidecar in _sidecars(temporary):
                _unlink(sidecar)
            _unlink(manifest_partial)
    return document


def _parse_table_columns(values: Sequence[str]) -> list[tuple[str, str]]:
    """解析 ``表:列``。标识符只允许字母、数字和下划线，避免把参数拼进 SQL。"""

    if not values:
        raise ValueError("post_exclusion_table_column_missing")
    pairs: list[tuple[str, str]] = []
    for value in values:
        table, separator, column = value.partition(":")
        if separator != ":" or _IDENT.fullmatch(table) is None or _IDENT.fullmatch(column) is None:
            raise ValueError("post_exclusion_table_column_invalid")
        pairs.append((table, column))
    if len({table for table, _column in pairs}) != len(pairs):
        raise ValueError("post_exclusion_table_column_duplicate")
    return pairs


def _reject_paths(
    source_path: Path,
    exclusion_paths: Sequence[Path],
    output_path: Path,
    manifest_path: Path,
    temporary: Path,
) -> None:
    """拒绝覆盖来源、排除清单或已有产物。此时还没有复制文件。"""

    protected = {source_path, *exclusion_paths}
    products = (output_path, manifest_path, temporary)
    if len(exclusion_paths) != len(set(exclusion_paths)):
        raise ValueError("post_exclusion_paths_overlap")
    if any(path in protected for path in products) or len(set(products)) != len(products):
        raise ValueError("post_exclusion_paths_overlap")
    if output_path.exists():
        raise FileExistsError("post_exclusion_output_exists")
    if manifest_path.exists():
        raise FileExistsError("post_exclusion_manifest_exists")
    if temporary.exists():
        raise FileExistsError("post_exclusion_temporary_exists")


def _reject_source_sidecars(source_path: Path) -> None:
    """来源带日志旁文件时，主文件不是完整已提交视图。"""

    if _sidecars(source_path):
        raise ValueError("post_exclusion_source_sidecar")


def _sidecars(path: Path) -> list[Path]:
    return [candidate for suffix in ("-wal", "-shm", "-journal") if (candidate := Path(str(path) + suffix)).exists()]


def _input_guard(path: Path) -> tuple[int, int, str, int]:
    """记录模式、标志、SHA-256 和大小，供结束后确认只读输入没被碰到。"""

    info = path.stat()
    return info.st_mode, getattr(info, "st_flags", 0), _file_sha256(path), info.st_size


def _assert_input_guard(path: Path, guard: tuple[int, int, str, int], label: str) -> None:
    mode, flags, digest, size = guard
    info = path.stat()
    if info.st_mode != mode or getattr(info, "st_flags", 0) != flags or info.st_size != size:
        raise ValueError("post_exclusion_input_metadata_changed:" + label)
    if _file_sha256(path) != digest:
        raise ValueError("post_exclusion_input_bytes_changed:" + label)


def _read_exclusions(
    paths: Sequence[Path],
) -> tuple[list[int], list[dict[str, Any]], list[tuple[int, int, str, int]]]:
    """合并排除清单中的 ``source_post_id``。不把昵称或平台用户 ID 读进返回的记录。"""

    found: set[int] = set()
    records: list[dict[str, Any]] = []
    guards: list[tuple[int, int, str, int]] = []
    for path in paths:
        guard = _input_guard(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("post_exclusion_file_not_object")
        entries = 0
        for key in ("absent", "still_masked"):
            rows = payload.get(key)
            if not isinstance(rows, list):
                raise ValueError("post_exclusion_entries_missing:" + key)
            for row in rows:
                if not isinstance(row, dict) or "source_post_id" not in row:
                    raise ValueError("post_exclusion_source_post_id_missing")
                found.add(_require_post_id(row["source_post_id"]))
                entries += 1
        records.append(
            {"path": str(path), "sha256": guard[2], "size_bytes": guard[3], "entry_count": entries}
        )
        guards.append(guard)
    return sorted(found), records, guards


def _assert_exclusion_guards(paths: Sequence[Path], guards: Sequence[tuple[int, int, str, int]]) -> None:
    for path, guard in zip(paths, guards, strict=True):
        _assert_input_guard(path, guard, "exclusion")


def _require_post_id(value: Any) -> int:
    """排除键只接受整数帖子 ID。JSON 布尔值不能因继承 ``int`` 而被当成 ID。"""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("post_exclusion_source_post_id_invalid")
    return value


def _assert_targets(source: Path, pairs: Sequence[tuple[str, str]]) -> None:
    """复制前确认要删的表和列存在。只读打开，不改来源。"""

    connection = _connect_readonly(source)
    try:
        tables = set(_user_tables(connection, "main"))
        for table, column in pairs:
            if table not in tables:
                raise ValueError("post_exclusion_table_missing:" + table)
            if column not in set(_column_names(connection, "main", table)):
                raise ValueError("post_exclusion_column_missing:" + table + ":" + column)
    finally:
        connection.close()


def _exclude_copy(
    temporary: Path,
    source: Path,
    pairs: Sequence[tuple[str, str]],
    excluded_ids: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """删除命中行，提交后重建文件，再对照只读来源校验保留行。"""

    connection = _connect_copy(temporary)
    try:
        mode = connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
        if str(mode).lower() != "delete":
            raise ValueError("post_exclusion_journal_mode_failed")
        connection.execute("PRAGMA foreign_keys=ON").fetchall()
        connection.execute("BEGIN IMMEDIATE")
        try:
            dropped_rows = [_delete_matching(connection, table, column, excluded_ids) for table, column in pairs]
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        # 与作者身份修订相同：VACUUM 必须在事务外，才能清掉 DELETE 留下的空闲页。
        connection.execute("VACUUM")
        _assert_no_sidecars(temporary)
        dangling = _dangling_references(connection, set(excluded_ids))
        validation = _validate_copy(connection, source, pairs, excluded_ids)
    finally:
        connection.close()
    return dropped_rows, dangling, validation


def _delete_matching(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    excluded_ids: Sequence[int],
) -> dict[str, Any]:
    """只删除指定列命中排除 ID 的行，返回实际删除数。没有命中时删除数为 0。"""

    quoted = _quote(table)
    deleted = 0
    if excluded_ids:
        before = connection.execute(f"SELECT count(*) FROM {quoted}").fetchone()[0]
        placeholders = ",".join("?" * len(excluded_ids))
        connection.execute(
            f"DELETE FROM {quoted} WHERE {_quote(column)} IN ({placeholders})",
            list(excluded_ids),
        )
        after = connection.execute(f"SELECT count(*) FROM {quoted}").fetchone()[0]
        deleted = int(before - after)
    return {"table": table, "column": column, "count": deleted}


def _dangling_references(connection: sqlite3.Connection, excluded_ids: set[int]) -> list[dict[str, Any]]:
    """统计保留行中仍指向被删帖的引用列。发现后不更新这些列。"""

    if not excluded_ids:
        return []
    grouped: dict[tuple[str, str], int] = {}
    referenced: dict[tuple[str, str], set[int]] = {}
    for table in _user_tables(connection, "main"):
        for column in _column_names(connection, "main", table):
            if _REFERENCE_COLUMN.fullmatch(column) is None:
                continue
            rows = connection.execute(
                f"SELECT {_quote(column)} AS value, count(*) AS n FROM {_quote(table)} GROUP BY {_quote(column)}"
            )
            for row in rows:
                matched = _matched_post_id(row["value"], excluded_ids)
                if matched is None:
                    continue
                key = (table, column)
                grouped[key] = grouped.get(key, 0) + int(row["n"])
                referenced.setdefault(key, set()).add(matched)
    return [
        {
            "table": table,
            "column": column,
            "count": grouped[(table, column)],
            "referenced_source_post_ids": sorted(referenced[(table, column)]),
        }
        for table, column in sorted(grouped)
    ]


def _matched_post_id(value: Any, excluded_ids: set[int]) -> int | None:
    """整数或与排除 ID 十进制写法相同的文本才算引用。不把普通数字文本当成帖子。"""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int) and value in excluded_ids:
        return value
    if isinstance(value, str):
        for post_id in excluded_ids:
            if value == str(post_id):
                return post_id
    return None


def _validate_copy(
    connection: sqlite3.Connection,
    source: Path,
    pairs: Sequence[tuple[str, str]],
    excluded_ids: Sequence[int],
) -> dict[str, str]:
    """完整性、外键，以及保留行相对来源逐列不变。差异拒绝发布。"""

    quick = [row[0] for row in connection.execute("PRAGMA quick_check").fetchall()]
    if quick != ["ok"]:
        raise ValueError("post_exclusion_quick_check_failed")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise ValueError("post_exclusion_foreign_key_check_failed")
    source_uri = source.resolve().as_uri() + "?mode=ro&immutable=1"
    connection.execute("ATTACH DATABASE ? AS src", (source_uri,))
    try:
        targets = {table for table, _column in pairs}
        main_tables = _user_tables(connection, "main")
        if main_tables != _user_tables(connection, "src"):
            raise ValueError("post_exclusion_table_set_mismatch")
        for name in main_tables:
            if name in targets:
                continue
            _assert_table_equal(connection, name, where="1", params=[])
        for table, column in pairs:
            where, params = _kept_predicate(column, excluded_ids)
            _assert_table_equal(connection, table, where=where, params=params)
    finally:
        connection.execute("DETACH DATABASE src")
    mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    if str(mode).lower() != "delete":
        raise ValueError("post_exclusion_journal_mode_failed")
    return {
        "quick_check": "ok",
        "foreign_key_check": "passed",
        "kept_rows": "unchanged",
        "other_tables_bidirectional_except": "empty",
        "other_tables_row_counts": "equal",
        "journal_mode": "delete",
        "vacuum": "completed",
    }


def _kept_predicate(column: str, excluded_ids: Sequence[int]) -> tuple[str, list[int]]:
    """保留列为空或不是排除 ID 的行。``NOT IN`` 对 NULL 的结果是未知，必须单独保留。"""

    if not excluded_ids:
        return "1", []
    placeholders = ",".join("?" * len(excluded_ids))
    return f"({_quote(column)} IS NULL OR {_quote(column)} NOT IN ({placeholders}))", list(excluded_ids)


def _assert_table_equal(connection: sqlite3.Connection, table: str, *, where: str, params: list[int]) -> None:
    quoted = _quote(table)
    main_count = connection.execute(f"SELECT count(*) FROM main.{quoted}").fetchone()[0]
    kept_count = connection.execute(f"SELECT count(*) FROM src.{quoted} WHERE {where}", params).fetchone()[0]
    if main_count != kept_count:
        raise ValueError("post_exclusion_table_count_mismatch:" + table)
    for statement in (
        f"SELECT * FROM main.{quoted} EXCEPT SELECT * FROM src.{quoted} WHERE {where}",
        f"SELECT * FROM src.{quoted} WHERE {where} EXCEPT SELECT * FROM main.{quoted}",
    ):
        if connection.execute(statement, params).fetchall():
            raise ValueError("post_exclusion_table_diff:" + table)


def _manifest_document(
    *,
    revision_id: str,
    reason: str,
    code_commit: str,
    created_at: str,
    source_path: Path,
    source_guard: tuple[int, int, str, int],
    exclusion_records: Sequence[Mapping[str, Any]],
    excluded_ids: Sequence[int],
    pairs: Sequence[tuple[str, str]],
    dropped_rows: Sequence[Mapping[str, Any]],
    dangling: Sequence[Mapping[str, Any]],
    output_path: Path,
    output_digest: str,
    output_size: int,
    validation: Mapping[str, str],
) -> dict[str, Any]:
    """组装清单。只记录路径、摘要、整数帖子 ID 和计数，不记录昵称或平台用户 ID。"""

    return {
        "revision_id": revision_id,
        "reason": reason,
        "status": _STATUS,
        "created_at_utc": created_at,
        "code_commit": code_commit,
        "source": _file_record(source_path, source_guard[2], source_guard[3]),
        "exclusion_files": [dict(item) for item in exclusion_records],
        "excluded_source_post_ids": list(excluded_ids),
        "table_columns": [table + ":" + column for table, column in pairs],
        "dropped_rows": [dict(item) for item in dropped_rows],
        "dangling_references": [dict(item) for item in dangling],
        "unchanged_tables_note": _UNCHANGED_TABLES_NOTE,
        "output": _file_record(output_path, output_digest, output_size),
        "validation": dict(validation),
    }


def _file_record(path: Path, digest: str, size: int) -> dict[str, Any]:
    return {"path": str(path), "sha256": digest, "size_bytes": size}


def _connect_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON").fetchall()
    return connection


def _connect_copy(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.isolation_level = None
    return connection


def _column_names(connection: sqlite3.Connection, schema: str, table: str) -> list[str]:
    rows = connection.execute(f"PRAGMA {schema}.table_info({_quote(table)})").fetchall()
    if not rows:
        raise ValueError("post_exclusion_table_missing:" + table)
    return [str(row[1]) for row in rows]


def _user_tables(connection: sqlite3.Connection, schema: str) -> list[str]:
    rows = connection.execute(
        f"SELECT name FROM {schema}.sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [str(row[0]) for row in rows]


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _fsync(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _write_private_json(directory: Path, payload: Mapping[str, Any]) -> Path:
    """先写同目录临时文件。调用方在副本就绪后用 replace 发布。"""

    descriptor, name = tempfile.mkstemp(prefix=".post-exclusion-", suffix=".json", dir=directory)
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
        raise ValueError("post_exclusion_sidecar_present")


def _unlink(path: Path | None) -> None:
    if path is not None and path.exists():
        path.unlink()
