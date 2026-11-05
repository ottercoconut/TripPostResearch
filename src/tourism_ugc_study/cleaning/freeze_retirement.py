"""解除已封存的研究数据冻结，并按帖子 ID 从冻结根内删除指定帖。

派生库已经去掉这些帖。本模块只在调用方给出的冻结根里原位替换三份数据库，
并按文件类型删掉其余副本。写操作前先把将被改写或删除的文件备份到冻结根之外；
备份校验通过后才清 ``uchg``、改权限和动原件。失败后可以重跑：已备份的字节
不会被后来的内容盖掉，已经做完的删除和过滤再做一次仍然是同一结果。

摘要、清单和标准输出只保留路径、计数和哈希，不写昵称、平台用户 ID 或正文。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .research_freeze_files import owned_path
from .source_snapshot import file_sha256

_ROLES = ("raw", "topic", "candidates")
_ROLE_FILENAME = {
    "raw": "raw-crawl.sqlite",
    "topic": "topic-relevant-20260911.sqlite",
    "candidates": "cleaning-candidates.sqlite",
}
_DERIVED_STATUS = "DERIVED_NOT_FROZEN"
_ID_KEY_RE = re.compile(
    r"^(source_post_id|left_source_post_id|right_source_post_id|web_post_id|post_id|"
    r"exact_representative_id|.+_post_id|.+_representative_id)$"
)
_SQLITE_ID_COLUMN = re.compile(
    r"^(source_post_id|web_post_id|post_id|exact_representative_id|left_source_post_id|"
    r"right_source_post_id|.+_post_id|.+_representative_id)$"
)
_ROW_ORDER = {
    "schema_migrations": "version",
    "source_platforms": "platform_key",
    "ctf_captures": "id",
    "web_posts": "id",
    "web_post_images": "id",
    "post_detail_repair_waivers": "id",
}
_SNAPSHOT_STALE = (
    "builder_sha256",
    "code_version",
    "completed_at_utc",
    "excluded_by_topic_flag",
    "previous_current_pointer",
    "source_consistency",
    "source_digest_scope",
    "source_main_sha256_after",
    "source_main_sha256_before",
    "source_path",
    "source_total_posts",
    "started_at_utc",
)
_CANDIDATE_STALE = (
    "assembly_code_version",
    "batch_receipts",
    "execution_state_sha256",
    "inference_code_version",
    "policy_manifest_sha256",
    "round_manifest_sha256",
)
_FREEZE_STALE = (
    "candidate_database_sha256",
    "cleaning",
    "media",
    "raw_database_sha256",
    "raw_platform_counts",
    "raw_post_count",
    "raw_topic_counts",
    "source_snapshot_sha256",
    "topic_subset_full_rows_check",
)
_REVIEW_PAIRS = (
    ("keep-review-private-map.json", "keep-review.csv"),
    ("manual-review-review-private-map.json", "manual-review-review.csv"),
    ("manual-review-private-map.json", "manual-review-tourism-relevance-annotation.csv"),
)
_STALE_NAMES = {
    "checkpoint-state.json": "checkpoint-state",
    "completion-state.json": "completion-state",
    "execution-state.json": "execution-state",
    "final-audit.json": "final-audit",
    "incremental-inference-manifest.json": "incremental-inference-manifest",
    "round-manifest.json": "round-manifest",
    "transfer-manifest.json": "transfer-manifest",
}
_POINTER_FREEZE = "data/processed/current-freeze.json"
_POINTER_SOURCE = "data/processed/current-source.json"
_POINTER_CLEANING = "data/processed/current-cleaning.json"
_REGISTRY = "governance/research-data-freezes.json"
_MEDIA_SUFFIXES = {".npy", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp4", ".mov", ".bin"}


def retire_freeze(
    workspace: Path,
    manifest: Path,
    expected_sha256: str,
    post_ids: Path | str,
    replacements: Mapping[str, Path],
    replacement_manifests: Mapping[str, Path],
    media_db: Path,
    backup_root: Path,
    reason: str,
    *,
    confirm_retire: str | None = None,
    dry_run: bool = False,
    chflags: Callable[[str | os.PathLike[str], int], object] | None = None,
) -> dict[str, Any]:
    """校验冻结身份后删除指定帖，或在 dry-run 时只返回计划和计数。

    Args:
        workspace: 仓库根。冻结根、登记和指针都相对它解析。
        manifest: ``freeze-manifest.json``。首次运行其 SHA-256 必须等于
            ``expected_sha256``。重跑时若备份里的原件仍等于该 SHA，允许清单
            已经被写过 ``retired`` 块。
        expected_sha256: 退役前 manifest 的 SHA-256，保留为历史。
        post_ids: ``{"ids": [整数, ...]}`` 的 JSON 文件或 JSON 文本。
            整数是 ``source_post_id`` / ``web_posts.id``。
        replacements: 角色到 r2 库的映射。角色只接受 ``raw``、``topic``、
            ``candidates``。
        replacement_manifests: 同角色的 ``revision-manifest.json``。输出 SHA
            必须与替换文件一致，状态必须是 ``DERIVED_NOT_FROZEN``。
        media_db: 仍含目标帖的只读库。用它的 ``web_post_images.local_path``
            定位媒体，用 ``content_text`` 做终检片段。
        backup_root: 备份目录，不得位于 manifest ``roots`` 之内。
        reason: 退役原因，写入登记和 manifest 的 ``retired`` 块。
        confirm_retire: 写操作时必须等于 manifest 的 ``freeze_id``。
        dry_run: 为真时不创建备份、不改权限、不写文件。
        chflags: 清除不可变标志的函数。缺省用 ``os.chflags``；没有该能力时
            由测试注入替身。

    Returns:
        dry-run 返回计划计数和相对路径。正式运行返回写入备份目录的
        ``run_summary.json`` 内容，只含路径、计数和哈希。

    Raises:
        ValueError: 校验失败、确认值不符、路径越界、终检仍有目标帖，
            或 SQLite 完整性 / 外键检查失败。
        NotImplementedError: 需要清 ``uchg`` 但没有 ``chflags`` 也没有替身。
        OSError: 备份或原子替换无法完成。
        sqlite3.Error: 只读库或替换库无法读取。
    """

    context = _preflight(
        workspace=workspace,
        manifest=manifest,
        expected_sha256=expected_sha256,
        post_ids=post_ids,
        replacements=replacements,
        replacement_manifests=replacement_manifests,
        media_db=media_db,
        backup_root=backup_root,
        reason=reason,
        confirm_retire=confirm_retire,
        dry_run=dry_run,
        chflags=chflags,
    )
    plan = _discover(context)
    if dry_run:
        return _dry_plan(context, plan)
    _ensure_backup(context, plan)
    _unprotect(context)
    _replay_deletes(context)
    _replace_databases(context)
    _rewrite_content(context)
    stale = _refresh_known_manifests(context)
    measurements = _measure(context)
    record = _sync_registry_and_pointers(context, measurements)
    _sync_inventory(context, record, measurements)
    scan = _final_scan(context)
    summary = _write_summary(context, measurements, stale, scan)
    return summary


def _preflight(
    *,
    workspace: Path,
    manifest: Path,
    expected_sha256: str,
    post_ids: Path | str,
    replacements: Mapping[str, Path],
    replacement_manifests: Mapping[str, Path],
    media_db: Path,
    backup_root: Path,
    reason: str,
    confirm_retire: str | None,
    dry_run: bool,
    chflags: Callable[[str | os.PathLike[str], int], object] | None,
) -> dict[str, Any]:
    """解析并核对全部输入。此阶段不创建、不修改任何文件。"""

    if not reason or not reason.strip():
        raise ValueError("freeze_retire_reason_missing")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("freeze_retire_expected_sha256_invalid")
    workspace = workspace.expanduser().resolve()
    manifest_path = manifest.expanduser().resolve()
    try:
        relative_manifest = manifest_path.relative_to(workspace).as_posix()
    except ValueError as exc:
        raise ValueError("freeze_retire_manifest_outside_workspace") from exc
    owned_path(workspace, relative_manifest)
    if manifest_path.is_symlink():
        raise ValueError("freeze_retire_symlink")
    spec = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec.get("contract") != "research-data-freeze-v1":
        raise ValueError("freeze_contract_invalid")
    freeze_id = spec.get("freeze_id")
    if not isinstance(freeze_id, str) or not freeze_id:
        raise ValueError("freeze_retire_freeze_id_missing")
    if confirm_retire is not None and confirm_retire != freeze_id:
        raise ValueError("freeze_retire_confirm_mismatch")
    if not dry_run and confirm_retire != freeze_id:
        raise ValueError("freeze_retire_confirm_mismatch")
    roots = [_resolve_root(workspace, value) for value in spec["roots"]]
    _reject_overlapping(roots)
    files = _enumerate_roots(roots)
    _reject_sidecars_in(files)
    ids = _load_post_ids(post_ids)
    role_dbs = _load_replacements(replacements, replacement_manifests, roots)
    media_path = media_db.expanduser().resolve(strict=True)
    _reject_symlink_and_sidecars(media_path)
    if any(media_path == item for item in role_dbs.values()) or any(media_path == path for path in files):
        raise ValueError("freeze_retire_media_db_overlaps_target")
    backup = backup_root.expanduser().resolve()
    _reject_backup_location(backup, workspace, roots, manifest_path.parent)
    texts, media_rels = _read_media_db(media_path, ids)
    _assert_replacements_exclude(role_dbs, ids)
    _accept_historical_manifest(
        manifest_path, expected_sha256, backup, relative_manifest, freeze_id, dry_run=dry_run
    )
    kept_texts = _kept_texts(media_path, role_dbs["raw"], ids)
    return {
        "workspace": workspace,
        "manifest_path": manifest_path,
        "relative_manifest": relative_manifest,
        "expected_sha256": expected_sha256,
        "spec": spec,
        "freeze_id": freeze_id,
        "roots": roots,
        "files": files,
        "ids": ids,
        "role_dbs": role_dbs,
        "media_db": media_path,
        "backup": backup,
        "reason": reason.strip(),
        "texts": texts,
        "kept_texts": kept_texts,
        "media_rels": media_rels,
        "chflags": chflags if chflags is not None else getattr(os, "chflags", None),
        "metadata_root": manifest_path.parent,
    }


def _resolve_root(workspace: Path, relative: str) -> Path:
    path = owned_path(workspace, relative)
    if not path.exists():
        raise FileNotFoundError("freeze_retire_root_missing:" + relative)
    return path


def _reject_overlapping(roots: Sequence[Path]) -> None:
    if len(set(roots)) != len(roots):
        raise ValueError("freeze_overlapping_roots")
    for left in roots:
        for right in roots:
            if left != right and left in right.parents:
                raise ValueError("freeze_overlapping_roots")


def _parent_is_symlink(path: Path, root: Path) -> bool:
    """根到该路径之间的目录不能是符号链接，避免解析后逃出冻结根。"""

    for parent in path.parents:
        if parent == root:
            return parent.is_symlink()
        if root in parent.parents and parent.is_symlink():
            return True
        if root not in parent.parents and parent != root:
            break
    return False


def _enumerate_roots(roots: Sequence[Path]) -> list[Path]:
    """列出根内普通文件。符号链接、特殊文件、硬链接和越界解析都拒绝。"""

    found: list[Path] = []
    for root in roots:
        if root.is_symlink():
            raise ValueError("freeze_retire_symlink")
        if root.is_file():
            _assert_regular_file(root)
            found.append(root)
            continue
        if not root.is_dir():
            raise ValueError("freeze_retire_nonregular")
        for path in sorted(root.rglob("*")):
            if path.is_symlink() or _parent_is_symlink(path, root):
                raise ValueError("freeze_retire_symlink")
            try:
                path.resolve().relative_to(root.resolve())
            except ValueError as exc:
                raise ValueError("freeze_retire_path_escapes_root") from exc
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise ValueError("freeze_retire_nonregular")
            _assert_regular_file(path)
            found.append(path)
    return found


def _assert_regular_file(path: Path) -> None:
    info = path.stat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ValueError("freeze_retire_nonregular")
    if info.st_nlink != 1:
        raise ValueError("freeze_retire_hardlink")


def _reject_sidecars_in(files: Sequence[Path]) -> None:
    for path in files:
        if path.name.endswith(("-wal", "-shm", "-journal")):
            raise ValueError("freeze_retire_sqlite_sidecar")
        if path.suffix == ".sqlite":
            _reject_symlink_and_sidecars(path)


def _reject_symlink_and_sidecars(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("freeze_retire_symlink")
    if _sidecars(path):
        raise ValueError("freeze_retire_sqlite_sidecar")


def _sidecars(path: Path) -> list[Path]:
    return [candidate for suffix in ("-wal", "-shm", "-journal") if (candidate := Path(str(path) + suffix)).exists()]


def _load_post_ids(value: Path | str) -> list[int]:
    if isinstance(value, Path) or (isinstance(value, str) and Path(value).expanduser().is_file()):
        text = Path(value).expanduser().read_text(encoding="utf-8")
    else:
        text = str(value)
    payload = json.loads(text)
    if not isinstance(payload, dict) or not isinstance(payload.get("ids"), list):
        raise ValueError("freeze_retire_post_ids_invalid")
    found: list[int] = []
    for item in payload["ids"]:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError("freeze_retire_post_ids_invalid")
        found.append(item)
    if not found or len(found) != len(set(found)):
        raise ValueError("freeze_retire_post_ids_invalid")
    return sorted(found)


def _load_replacements(
    replacements: Mapping[str, Path],
    manifests: Mapping[str, Path],
    roots: Sequence[Path],
) -> dict[str, Path]:
    if set(replacements) != set(_ROLES) or set(manifests) != set(_ROLES):
        raise ValueError("freeze_retire_replacement_role_invalid")
    resolved: dict[str, Path] = {}
    for role in _ROLES:
        path = replacements[role].expanduser().resolve(strict=True)
        manifest_path = manifests[role].expanduser().resolve(strict=True)
        _reject_symlink_and_sidecars(path)
        if path == manifest_path or _inside_any(path, roots) or _inside_any(manifest_path, roots):
            raise ValueError("freeze_retire_replacement_inside_roots")
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        output = document.get("output") if isinstance(document, dict) else None
        if document.get("status") != _DERIVED_STATUS or not isinstance(output, dict):
            raise ValueError("freeze_retire_replacement_not_derived")
        digest = output.get("sha256")
        if digest != file_sha256(path):
            raise ValueError("freeze_retire_replacement_hash_mismatch")
        if "size_bytes" in output and output["size_bytes"] != path.stat().st_size:
            raise ValueError("freeze_retire_replacement_size_mismatch")
        resolved[role] = path
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("freeze_retire_replacement_role_invalid")
    return resolved


def _inside_any(path: Path, roots: Sequence[Path]) -> bool:
    resolved = path.resolve()
    for root in roots:
        root_resolved = root.resolve()
        if resolved == root_resolved or root_resolved in resolved.parents:
            return True
    return False


def _reject_backup_location(backup: Path, workspace: Path, roots: Sequence[Path], metadata: Path) -> None:
    # 备份可以放在仓库里、冻结根之外。不能盖住仓库本身，否则相对路径会写回原件。
    if backup == workspace or backup in workspace.parents:
        raise ValueError("freeze_retire_backup_overlaps_workspace")
    for root in (*roots, metadata):
        resolved = root.resolve()
        if backup == resolved or resolved in backup.parents or backup in resolved.parents:
            raise ValueError("freeze_retire_backup_inside_roots")
    if backup.exists() and not backup.is_dir():
        raise ValueError("freeze_retire_backup_not_directory")


def _read_media_db(path: Path, ids: Sequence[int]) -> tuple[dict[int, str], list[str]]:
    """只读取出目标帖正文和图片相对路径，随即关闭连接。"""

    connection = _connect_readonly(path)
    try:
        present = {
            int(row["id"]): row["content_text"]
            for row in connection.execute("SELECT id, content_text FROM web_posts")
            if not isinstance(row["id"], bool)
        }
        missing = [post_id for post_id in ids if post_id not in present]
        if missing:
            raise ValueError("freeze_retire_post_missing")
        texts = {post_id: present[post_id] if isinstance(present[post_id], str) else "" for post_id in ids}
        placeholders = ",".join("?" * len(ids))
        rows = connection.execute(
            f"SELECT local_path FROM web_post_images WHERE web_post_id IN ({placeholders})",
            list(ids),
        ).fetchall()
    finally:
        connection.close()
    relatives: list[str] = []
    for row in rows:
        relative = row["local_path"]
        item = Path(relative) if isinstance(relative, str) else None
        if item is None or item.is_absolute() or ".." in item.parts or not item.parts:
            raise ValueError("freeze_retire_media_path_invalid")
        relatives.append(item.as_posix())
    return texts, sorted(set(relatives))


def _kept_texts(media_db: Path, raw_replacement: Path, ids: Sequence[int]) -> list[str]:
    """保留帖的正文只用于区分共享文本，不进入摘要。"""

    kept: list[str] = []
    excluded = set(ids)
    for path in (media_db, raw_replacement):
        connection = _connect_readonly(path)
        try:
            rows = connection.execute("SELECT id, content_text FROM web_posts")
            for row in rows:
                if isinstance(row["id"], bool) or not isinstance(row["id"], int):
                    continue
                if row["id"] in excluded or not isinstance(row["content_text"], str):
                    continue
                kept.append(row["content_text"])
        finally:
            connection.close()
    return kept


def _assert_replacements_exclude(role_dbs: Mapping[str, Path], ids: Sequence[int]) -> None:
    checks = (
        ("raw", "web_posts", "id"),
        ("topic", "web_posts", "id"),
        ("candidates", "cleaning_candidates", "source_post_id"),
    )
    for role, table, column in checks:
        connection = _connect_readonly(role_dbs[role])
        try:
            names = {row[1] for row in connection.execute(f"PRAGMA table_info({_quote(table)})")}
            if column not in names:
                raise ValueError("freeze_retire_replacement_column_missing")
            placeholders = ",".join("?" * len(ids))
            hit = connection.execute(
                f"SELECT 1 FROM {_quote(table)} WHERE {_quote(column)} IN ({placeholders}) LIMIT 1",
                list(ids),
            ).fetchone()
            if hit is not None:
                raise ValueError("freeze_retire_replacement_still_contains_post")
        finally:
            connection.close()


def _accept_historical_manifest(
    manifest_path: Path,
    expected_sha256: str,
    backup: Path,
    relative_manifest: str,
    freeze_id: str,
    *,
    dry_run: bool,
) -> None:
    current = file_sha256(manifest_path)
    if current == expected_sha256:
        return
    index_path = backup / "backup-index.json"
    if dry_run or not index_path.is_file():
        raise ValueError("freeze_retire_manifest_hash_mismatch")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    archived = backup / relative_manifest
    if (
        index.get("expected_sha256") != expected_sha256
        or index.get("freeze_id") != freeze_id
        or not archived.is_file()
        or file_sha256(archived) != expected_sha256
    ):
        raise ValueError("freeze_retire_manifest_hash_mismatch")


def _discover(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """算出将改写、删除或替换的文件。只读，不写盘。"""

    workspace: Path = context["workspace"]
    ids: set[int] = set(context["ids"])
    role_targets = {role: _unique_named(context["files"], filename) for role, filename in _ROLE_FILENAME.items()}
    operations: list[dict[str, Any]] = []
    for role, target in role_targets.items():
        operations.append(
            {
                "path": _relative(workspace, target),
                "action": "replace",
                "kind": role,
            }
        )
    deleted_keys, kept_keys = _embedding_keys(context["files"], ids)
    context_keys = {"deleted": deleted_keys, "kept": kept_keys}
    operations.extend(_content_operations(context["files"], workspace, ids, context_keys))
    operations.extend(_media_operations(context, workspace))
    operations.extend(_metadata_operations(context, role_targets))
    _reject_duplicate_ops(operations)
    context["embedding_shared"] = len(deleted_keys & kept_keys)  # type: ignore[index]
    return operations


def _unique_named(files: Sequence[Path], name: str) -> Path:
    matches = [path for path in files if path.name == name]
    if len(matches) != 1:
        raise ValueError("freeze_retire_required_file_missing:" + name)
    return matches[0]


def _content_operations(
    files: Sequence[Path],
    workspace: Path,
    ids: set[int],
    keys: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    paired_csv = {name for _map_name, name in _REVIEW_PAIRS}
    paired_maps = {name for name, _csv_name in _REVIEW_PAIRS}
    for path in files:
        relative = path.relative_to(workspace).as_posix()
        category = _stale_category(Path(relative))
        if category is not None:
            continue
        kind = _editable_kind(path)
        if kind is None:
            continue
        if path.name in paired_csv:
            continue
        if kind == "checkpoint":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if _record_hits(payload, ids):
                operations.append({"path": relative, "action": "delete", "kind": kind})
            continue
        if kind == "embedding":
            cache_key = _cache_key_from_path(path)
            if cache_key in keys["deleted"] and cache_key not in keys["kept"]:
                operations.append({"path": relative, "action": "delete", "kind": kind})
            continue
        if path.name in paired_maps or kind in {"records", "near", "incremental", "batch_csv", "predictions", "inventory"}:
            operations.append({"path": relative, "action": "rewrite", "kind": kind})
    for map_name, csv_name in _REVIEW_PAIRS:
        for folder in {path.parent for path in files if path.name == map_name}:
            csv_path = folder / csv_name
            if not csv_path.is_file():
                raise ValueError("freeze_retire_review_csv_missing")
            operations.append(
                {
                    "path": csv_path.relative_to(workspace).as_posix(),
                    "action": "rewrite",
                    "kind": "review_csv",
                    "map_path": (folder / map_name).relative_to(workspace).as_posix(),
                }
            )
    return operations


def _editable_kind(path: Path) -> str | None:
    parts = path.parts
    name = path.name
    if "embedding-cache" in parts and name in {"embedding.npy", "receipt.json"}:
        return "embedding"
    if "checkpoint-records" in parts and name.endswith(".json"):
        return "checkpoint"
    if name == "records.json":
        return "records"
    if name == "near-duplicate-candidates.json":
        return "near"
    if name == "incremental-scored-records.json":
        return "incremental"
    if name == "media-inventory.json":
        return "inventory"
    if "inference-batches" in parts and name.startswith("batch-") and name.endswith(".csv"):
        return "batch_csv"
    if name == "tourism-relevance-predictions.csv":
        return "predictions"
    if name in {item for pair in _REVIEW_PAIRS for item in pair}:
        return "review"
    return None


def _stale_category(relative: Path) -> str | None:
    name = relative.name
    parts = relative.parts
    if name in _STALE_NAMES:
        return _STALE_NAMES[name]
    if "logs" in parts or name.endswith(".log"):
        return "logs"
    if "embedding-cache" in parts:
        return None
    if "receipts" in parts or name == "input-receipt.json" or name.endswith("receipt.json"):
        return "receipts"
    return None


def _embedding_keys(files: Sequence[Path], ids: set[int]) -> tuple[set[str], set[str]]:
    deleted: set[str] = set()
    kept: set[str] = set()
    for path in files:
        if path.name not in {"incremental-scored-records.json"} and "checkpoint-records" not in path.parts:
            continue
        if path.suffix != ".json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            if not isinstance(record, dict):
                continue
            cache_key = record.get("embedding_cache_key")
            if not isinstance(cache_key, str) or ":" not in cache_key:
                continue
            if _record_hits(record, ids):
                deleted.add(cache_key)
            else:
                kept.add(cache_key)
    return deleted, kept


def _cache_key_from_path(path: Path) -> str:
    """``embedding-cache/<namespace>/<xx>/<key>/file`` 还原缓存键。"""

    parts = path.parts
    index = parts.index("embedding-cache")
    namespace = parts[index + 1]
    key = parts[index + 3]
    return namespace + ":" + key


def _media_operations(context: Mapping[str, Any], workspace: Path) -> list[dict[str, Any]]:
    """定位要删的图片。重跑时文件已不在，只要备份索引里有这条删除记录就跳过。"""

    media_root = _media_root(context["roots"])
    already_deleted = _indexed_deletes(context["backup"])
    operations: list[dict[str, Any]] = []
    for relative in context["media_rels"]:
        path = media_root / relative
        if path.is_symlink() or ".." in Path(relative).parts:
            raise ValueError("freeze_retire_media_path_invalid")
        try:
            path.relative_to(media_root)
        except ValueError as exc:
            raise ValueError("freeze_retire_path_escapes_root") from exc
        listed = path.relative_to(workspace).as_posix()
        if not path.is_file():
            if listed not in already_deleted:
                raise ValueError("freeze_retire_media_missing")
            continue
        operations.append({"path": listed, "action": "delete", "kind": "media"})
    return operations


def _indexed_deletes(backup: Path) -> set[str]:
    index_path = backup / "backup-index.json"
    if not index_path.is_file():
        return set()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("verified") is not True or not isinstance(index.get("files"), list):
        return set()
    return {item["path"] for item in index["files"] if item.get("action") == "delete"}


def _metadata_operations(context: Mapping[str, Any], role_targets: Mapping[str, Path]) -> list[dict[str, Any]]:
    workspace: Path = context["workspace"]
    operations = [
        {"path": context["relative_manifest"], "action": "rewrite", "kind": "freeze_manifest"},
        {
            "path": _relative(workspace, context["metadata_root"] / "frozen-files.json"),
            "action": "rewrite",
            "kind": "frozen_files",
        },
        {"path": _REGISTRY, "action": "rewrite", "kind": "registry"},
        {"path": _POINTER_FREEZE, "action": "rewrite", "kind": "pointer_freeze"},
        {"path": _POINTER_SOURCE, "action": "rewrite", "kind": "pointer_source"},
        {"path": _POINTER_CLEANING, "action": "rewrite", "kind": "pointer_cleaning"},
    ]
    manifests = {
        "raw": role_targets["raw"].with_suffix(".manifest.json"),
        "topic": role_targets["topic"].with_suffix(".manifest.json"),
        "candidates": role_targets["candidates"].parent / "candidate-manifest.json",
    }
    for role, path in manifests.items():
        if not path.is_file():
            raise ValueError("freeze_retire_manifest_missing:" + role)
        operations.append({"path": _relative(workspace, path), "action": "rewrite", "kind": "data_manifest", "role": role})
    for relative in (_REGISTRY, _POINTER_FREEZE, _POINTER_SOURCE, _POINTER_CLEANING):
        if not (workspace / relative).is_file():
            raise ValueError("freeze_retire_pointer_missing")
    return operations


def _reject_duplicate_ops(operations: Sequence[Mapping[str, Any]]) -> None:
    paths = [item["path"] for item in operations]
    if len(paths) != len(set(paths)):
        raise ValueError("freeze_retire_plan_duplicate")


def _dry_plan(context: Mapping[str, Any], operations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "dry_run": True,
        "deleted_post_count": len(context["ids"]),
        "rewrite_count": sum(item["action"] == "rewrite" for item in operations),
        "delete_count": sum(item["action"] == "delete" for item in operations),
        "replace_count": sum(item["action"] == "replace" for item in operations),
        "shared_embedding_kept": context["embedding_shared"],
        "paths": [item["path"] for item in operations],
    }


def _ensure_backup(context: Mapping[str, Any], operations: Sequence[Mapping[str, Any]]) -> None:
    """把计划中的文件按仓库相对路径备份。已校验的备份不再覆盖。"""

    backup: Path = context["backup"]
    workspace: Path = context["workspace"]
    backup.mkdir(parents=True, exist_ok=True)
    os.chmod(backup, 0o700)
    index_path = backup / "backup-index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        _verify_backup_index(backup, index)
        if isinstance(index.get("shared_embedding_kept"), int):
            context["embedding_shared"] = index["shared_embedding_kept"]
        known = {item["path"] for item in index["files"]}
        extra = [item for item in operations if item["path"] not in known]
        if extra:
            index["files"].extend(_copy_entries(workspace, backup, extra, repair=False))
            _publish_json(index_path, index)
            _verify_backup_index(backup, index)
        context["index"] = index
        return
    entries = _copy_entries(workspace, backup, operations, repair=True)
    index = {
        "freeze_id": context["freeze_id"],
        "expected_sha256": context["expected_sha256"],
        "verified": True,
        "shared_embedding_kept": context["embedding_shared"],
        "files": entries,
    }
    _publish_json(index_path, index)
    _verify_backup_index(backup, index)
    for item in entries:
        source = workspace / item["path"]
        if file_sha256(source) != item["sha256"]:
            raise ValueError("freeze_retire_backup_source_changed")
    context["index"] = index


def _copy_entries(
    workspace: Path,
    backup: Path,
    operations: Sequence[Mapping[str, Any]],
    *,
    repair: bool,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in operations:
        source = workspace / item["path"]
        if not source.is_file() or source.is_symlink():
            raise ValueError("freeze_retire_backup_source_missing")
        destination = backup / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and file_sha256(destination) != file_sha256(source):
            # 还没有校验过的索引时，半份拷贝只能来自尚未改动的原件，可以补齐。
            if not repair:
                raise ValueError("freeze_retire_backup_conflict")
            destination.unlink()
        if not destination.exists():
            shutil.copyfile(source, destination)
            os.chmod(destination, 0o600)
            _fsync(destination)
        info = destination.stat()
        entries.append(
            {
                "path": item["path"],
                "size_bytes": info.st_size,
                "sha256": file_sha256(destination),
                "action": item["action"],
            }
        )
    return entries


def _verify_backup_index(backup: Path, index: Mapping[str, Any]) -> None:
    if index.get("verified") is not True or not isinstance(index.get("files"), list):
        raise ValueError("freeze_retire_backup_index_invalid")
    for item in index["files"]:
        path = backup / item["path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError("freeze_retire_backup_missing")
        if path.stat().st_size != item["size_bytes"] or file_sha256(path) != item["sha256"]:
            raise ValueError("freeze_retire_backup_hash_mismatch")


def _unprotect(context: Mapping[str, Any]) -> None:
    """只清冻结根和 manifest 所在目录的不可变标志，并改成所有者可写。"""

    flag_fn = context["chflags"]
    if flag_fn is None:
        raise NotImplementedError("freeze_retire_requires_chflags")
    directories: list[Path] = []
    files: list[Path] = []
    for root in [*context["roots"], context["metadata_root"]]:
        if root.is_dir():
            directories.append(root)
            for path in root.rglob("*"):
                if path.is_symlink():
                    raise ValueError("freeze_retire_symlink")
                if path.is_dir():
                    directories.append(path)
                elif path.is_file():
                    files.append(path)
        elif root.is_file():
            files.append(root)
    immutable = getattr(stat, "UF_IMMUTABLE", 0)
    for path in sorted(set(directories), key=lambda item: len(item.parts)):
        _clear_flags(path, flag_fn, immutable, directory=True)
    for path in sorted(set(files)):
        _clear_flags(path, flag_fn, immutable, directory=False)


def _clear_flags(path: Path, flag_fn: Callable[[str | os.PathLike[str], int], object], immutable: int, *, directory: bool) -> None:
    current = getattr(path.stat(), "st_flags", 0)
    flag_fn(path, current & ~immutable)
    os.chmod(path, 0o700 if directory else 0o600)


def _replay_deletes(context: Mapping[str, Any]) -> None:
    workspace: Path = context["workspace"]
    for item in context["index"]["files"]:
        if item["action"] != "delete":
            continue
        path = workspace / item["path"]
        if path.is_symlink():
            raise ValueError("freeze_retire_symlink")
        if not path.exists():
            _prune_empty(path)
            continue
        if file_sha256(path) != item["sha256"]:
            raise ValueError("freeze_retire_delete_hash_mismatch")
        path.unlink()
        _prune_empty(path)


def _prune_empty(path: Path) -> None:
    parts = path.parts
    if "media-root" in parts:
        stop = "media-root"
    elif "embedding-cache" in parts:
        stop = "embedding-cache"
    else:
        return
    parent = path.parent
    while parent.name != stop:
        if not parent.exists():
            parent = parent.parent
            continue
        try:
            next(parent.iterdir())
        except StopIteration:
            parent.rmdir()
            parent = parent.parent
            continue
        break


def _replace_databases(context: Mapping[str, Any]) -> None:
    workspace: Path = context["workspace"]
    for item in context["index"]["files"]:
        if item["action"] != "replace":
            continue
        destination = workspace / item["path"]
        source = context["role_dbs"][_role_of(destination.name)]
        if destination.is_file() and file_sha256(destination) == file_sha256(source):
            continue
        _replace_file(source, destination)


def _role_of(filename: str) -> str:
    for role, name in _ROLE_FILENAME.items():
        if name == filename:
            return role
    raise ValueError("freeze_retire_replacement_role_invalid")


def _replace_file(source: Path, destination: Path) -> None:
    temporary = _temporary_path(destination)
    try:
        shutil.copyfile(source, temporary)
        os.chmod(temporary, 0o600)
        if file_sha256(temporary) != file_sha256(source):
            raise ValueError("freeze_retire_replace_hash_mismatch")
        _fsync(temporary)
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if _sidecars(destination):
        raise ValueError("freeze_retire_sqlite_sidecar")


def _rewrite_content(context: Mapping[str, Any]) -> None:
    """先读私有映射得到 task_id，再改 CSV 和映射本身。"""

    workspace: Path = context["workspace"]
    ids = set(context["ids"])
    backup: Path = context["backup"]
    handled: set[str] = set()
    for item in context["index"]["files"]:
        if item.get("action") != "rewrite":
            continue
        path = workspace / item["path"]
        if path.name not in {name for _map, name in _REVIEW_PAIRS}:
            continue
        map_relative = _map_relative_for(item["path"])
        if map_relative is None:
            continue
        task_ids = _task_ids_for(workspace / map_relative, backup / map_relative, ids)
        _rewrite_csv_ids(path, drop_tasks=task_ids, drop_posts=set())
        _rewrite_json_records(workspace / map_relative, ids)
        handled.add(item["path"])
        handled.add(map_relative)
    for item in context["index"]["files"]:
        if item["action"] != "rewrite" or item["path"] in handled:
            continue
        path = workspace / item["path"]
        kind = _editable_kind(path)
        if kind == "records":
            _rewrite_json_records(path, ids)
        elif kind == "near":
            _rewrite_near(path, ids)
        elif kind == "incremental":
            _rewrite_json_records(path, ids)
        elif kind in {"batch_csv", "predictions"}:
            _rewrite_csv_ids(path, drop_tasks=set(), drop_posts=ids)
        elif kind == "inventory":
            _rewrite_inventory(path, context["media_rels"])
        elif path.name in {name for name, _csv in _REVIEW_PAIRS}:
            _rewrite_json_records(path, ids)


def _map_relative_for(csv_relative: str) -> str | None:
    name = Path(csv_relative).name
    for map_name, csv_name in _REVIEW_PAIRS:
        if csv_name == name:
            return str(Path(csv_relative).with_name(map_name).as_posix())
    return None


def _task_ids_for(live: Path, backup: Path, ids: set[int]) -> set[str]:
    """合并仍在盘上的映射和备份中的原映射，避免重跑时任务号已经删掉。"""

    found: set[str] = set()
    for path in (backup, live):
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("freeze_retire_private_map_invalid")
        for item in payload:
            if isinstance(item, dict) and _as_post_id(item.get("source_post_id")) in ids:
                task_id = item.get("task_id")
                if not isinstance(task_id, str) or not task_id:
                    raise ValueError("freeze_retire_task_id_missing")
                found.add(task_id)
    return found


def _rewrite_json_records(path: Path, ids: set[int]) -> None:
    original = path.read_text(encoding="utf-8")
    payload = json.loads(original)
    if not isinstance(payload, list):
        raise ValueError("freeze_retire_json_list_invalid")
    kept = [item for item in payload if not _record_hits(item, ids)]
    _publish_styled_json(path, original, kept)


def _rewrite_near(path: Path, ids: set[int]) -> None:
    original = path.read_text(encoding="utf-8")
    payload = json.loads(original)
    if not isinstance(payload, list):
        raise ValueError("freeze_retire_json_list_invalid")
    kept = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("freeze_retire_near_duplicate_invalid")
        left = _as_post_id(item.get("left_source_post_id"))
        right = _as_post_id(item.get("right_source_post_id"))
        if left in ids or right in ids:
            continue
        kept.append(item)
    _publish_styled_json(path, original, kept)


def _rewrite_csv_ids(path: Path, *, drop_tasks: set[str], drop_posts: set[int]) -> None:
    original = path.read_bytes()
    rows, fields, bom, newline = _read_csv(original)
    if "source_post_id" not in fields and "task_id" not in fields:
        raise ValueError("freeze_retire_csv_column_missing")
    kept = []
    for row in rows:
        post_id = _as_post_id(row.get("source_post_id")) if "source_post_id" in fields else None
        task_id = row.get("task_id")
        if post_id in drop_posts or (isinstance(task_id, str) and task_id in drop_tasks):
            continue
        kept.append(row)
    updated = _csv_bytes(kept, fields, bom=bom, newline=newline)
    if updated != original:
        _publish_bytes(path, updated)


def _rewrite_inventory(path: Path, media_rels: Sequence[str]) -> None:
    original = path.read_text(encoding="utf-8")
    payload = json.loads(original)
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
        raise ValueError("freeze_retire_inventory_invalid")
    dropping = set(media_rels)
    entries = []
    for item in payload["entries"]:
        if not isinstance(item, dict) or not isinstance(item.get("local_path"), str):
            raise ValueError("freeze_retire_inventory_invalid")
        if item["local_path"] in dropping:
            continue
        entries.append(item)
    payload["entries"] = entries
    if "file_count" in payload:
        payload["file_count"] = len(entries)
    if "logical_bytes" in payload:
        payload["logical_bytes"] = sum(int(item["size_bytes"]) for item in entries if "size_bytes" in item)
    if "relationship_count" in payload:
        payload["relationship_count"] = len(entries)
    _publish_styled_json(path, original, payload)


def _refresh_known_manifests(context: Mapping[str, Any]) -> list[str]:
    workspace: Path = context["workspace"]
    stale: list[str] = []
    for item in context["index"]["files"]:
        if item["action"] != "rewrite":
            continue
        path = workspace / item["path"]
        if path.name == "candidate-manifest.json":
            stale.extend(_refresh_candidate_manifest(path, context))
        elif path.name in {"raw-crawl.manifest.json", "topic-relevant-20260911.manifest.json"}:
            database = path.with_name(path.name.replace(".manifest.json", ".sqlite"))
            stale.extend(_refresh_snapshot_manifest(path, database))
    return sorted(set(stale))


def _refresh_snapshot_manifest(path: Path, database: Path) -> list[str]:
    original = path.read_text(encoding="utf-8")
    payload = json.loads(original)
    if not isinstance(payload, dict):
        raise ValueError("freeze_retire_manifest_invalid")
    connection = _connect_readonly(database)
    try:
        post_count = int(connection.execute("SELECT count(*) FROM web_posts").fetchone()[0])
        if "selected_posts" in payload:
            payload["selected_posts"] = post_count
        if "platform_counts" in payload:
            payload["platform_counts"] = {
                str(key): count
                for key, count in connection.execute(
                    "SELECT platform_key, count(*) FROM web_posts GROUP BY platform_key ORDER BY platform_key"
                )
            }
        if "topic_counts" in payload:
            payload["topic_counts"] = {
                str(key): count
                for key, count in connection.execute(
                    "SELECT topic_relevant, count(*) FROM web_posts GROUP BY topic_relevant ORDER BY topic_relevant"
                )
            }
        evidence = payload.get("table_evidence")
        if isinstance(evidence, dict):
            for table, order in _ROW_ORDER.items():
                if table not in evidence or not isinstance(evidence[table], dict):
                    continue
                if not _table_exists(connection, table):
                    continue
                columns = _column_names(connection, table)
                if order not in columns:
                    continue
                fields = ",".join(_quote(name) for name in columns)
                rows = connection.execute(
                    f"SELECT {fields} FROM {_quote(table)} ORDER BY {_quote(order)}"
                ).fetchall()
                evidence[table]["count"] = len(rows)
                if "rows_sha256" in evidence[table]:
                    evidence[table]["rows_sha256"] = _rows_digest(rows)
                if "columns" in evidence[table]:
                    evidence[table]["columns"] = columns
    finally:
        connection.close()
    if isinstance(payload.get("database"), dict):
        payload["database"]["sha256"] = file_sha256(database)
        if "size_bytes" in payload["database"]:
            payload["database"]["size_bytes"] = database.stat().st_size
    _publish_styled_json(path, original, payload)
    return [path.name + ":" + key for key in _SNAPSHOT_STALE if key in payload]


def _refresh_candidate_manifest(path: Path, context: Mapping[str, Any]) -> list[str]:
    original = path.read_text(encoding="utf-8")
    payload = json.loads(original)
    if not isinstance(payload, dict):
        raise ValueError("freeze_retire_manifest_invalid")
    database = path.parent / "cleaning-candidates.sqlite"
    connection = _connect_readonly(database)
    try:
        count = int(connection.execute("SELECT count(*) FROM cleaning_candidates").fetchone()[0])
        decisions = {
            str(key): count_value
            for key, count_value in connection.execute(
                "SELECT decision, count(*) FROM cleaning_candidates GROUP BY decision ORDER BY decision"
            )
        }
        if "record_count" in payload:
            payload["record_count"] = count
        if "decision_counts" in payload:
            payload["decision_counts"] = decisions
        if "platform_decision_counts" in payload and _column_exists(connection, "cleaning_candidates", "platform_key"):
            grouped: dict[str, dict[str, int]] = {}
            for platform, decision, amount in connection.execute(
                "SELECT platform_key, decision, count(*) FROM cleaning_candidates "
                "GROUP BY platform_key, decision ORDER BY platform_key, decision"
            ):
                grouped.setdefault(str(platform), {})[str(decision)] = amount
            payload["platform_decision_counts"] = grouped
        if "database_sha256" in payload:
            payload["database_sha256"] = file_sha256(database)
        if "source_snapshot_sha256" in payload:
            payload["source_snapshot_sha256"] = file_sha256(context["role_dbs"]["topic"])
    finally:
        connection.close()
    near = path.parent / "near-duplicate-candidates.json"
    if near.is_file():
        near_payload = json.loads(near.read_text(encoding="utf-8"))
        if "near_candidate_count" in payload and isinstance(near_payload, list):
            payload["near_candidate_count"] = len(near_payload)
        if "near_candidates_sha256" in payload:
            payload["near_candidates_sha256"] = file_sha256(near)
    reviews = payload.get("review_tasks")
    if isinstance(reviews, dict):
        for scope, spec in reviews.items():
            if not isinstance(spec, dict):
                continue
            filename = spec.get("filename")
            if isinstance(filename, str) and (path.parent / filename).is_file():
                spec["sha256"] = file_sha256(path.parent / filename)
                spec["count"] = _csv_data_rows(path.parent / filename)
            map_name = spec.get("private_map_filename")
            if isinstance(map_name, str) and (path.parent / map_name).is_file():
                spec["private_map_sha256"] = file_sha256(path.parent / map_name)
    _publish_styled_json(path, original, payload)
    return [path.name + ":" + key for key in _CANDIDATE_STALE if key in payload]


def _measure(context: Mapping[str, Any]) -> dict[str, Any]:
    raw = context["role_dbs"]["raw"]
    topic = context["role_dbs"]["topic"]
    candidates = context["role_dbs"]["candidates"]
    raw_posts, raw_images = _table_counts(raw, ("web_posts", "web_post_images"))
    topic_posts, _ignored = _table_counts(topic, ("web_posts", "web_posts"))
    connection = _connect_readonly(candidates)
    try:
        record_count = int(connection.execute("SELECT count(*) FROM cleaning_candidates").fetchone()[0])
        decisions = {
            str(key): value
            for key, value in connection.execute(
                "SELECT decision, count(*) FROM cleaning_candidates GROUP BY decision ORDER BY decision"
            )
        }
    finally:
        connection.close()
    media_root = _media_root(context["roots"])
    inventory = json.loads((media_root.parent / "media-inventory.json").read_text(encoding="utf-8"))
    image_files = len(inventory["entries"]) if isinstance(inventory.get("entries"), list) else raw_images
    return {
        "raw_sha256": file_sha256(context["workspace"] / _relative_of_role(context, "raw")),
        "raw_size_bytes": (context["workspace"] / _relative_of_role(context, "raw")).stat().st_size,
        "raw_post_count": raw_posts,
        "raw_image_count": image_files,
        "topic_sha256": file_sha256(context["workspace"] / _relative_of_role(context, "topic")),
        "topic_size_bytes": (context["workspace"] / _relative_of_role(context, "topic")).stat().st_size,
        "topic_manifest_sha256": file_sha256(
            (context["workspace"] / _relative_of_role(context, "topic")).with_suffix(".manifest.json")
        ),
        "selected_posts": topic_posts,
        "candidate_sha256": file_sha256(context["workspace"] / _relative_of_role(context, "candidates")),
        "candidate_size_bytes": (context["workspace"] / _relative_of_role(context, "candidates")).stat().st_size,
        "candidate_manifest_sha256": file_sha256(
            (context["workspace"] / _relative_of_role(context, "candidates")).parent / "candidate-manifest.json"
        ),
        "record_count": record_count,
        "decision_counts": decisions,
        "source_record_count": topic_posts,
    }


def _relative_of_role(context: Mapping[str, Any], role: str) -> str:
    filename = _ROLE_FILENAME[role]
    for item in context["index"]["files"]:
        if item["action"] == "replace" and item["path"].endswith(filename):
            return item["path"]
    raise ValueError("freeze_retire_required_file_missing:" + filename)


def _table_counts(database: Path, tables: tuple[str, str]) -> tuple[int, int]:
    connection = _connect_readonly(database)
    try:
        first = int(connection.execute(f"SELECT count(*) FROM {_quote(tables[0])}").fetchone()[0])
        second = int(connection.execute(f"SELECT count(*) FROM {_quote(tables[1])}").fetchone()[0])
    finally:
        connection.close()
    return first, second


def _sync_registry_and_pointers(context: Mapping[str, Any], measurements: Mapping[str, Any]) -> dict[str, Any]:
    workspace: Path = context["workspace"]
    registry_path = workspace / _REGISTRY
    original = registry_path.read_text(encoding="utf-8")
    registry = json.loads(original)
    entries = registry.get("freezes")
    if not isinstance(entries, list):
        raise ValueError("freeze_retire_registry_invalid")
    current = next((item for item in entries if isinstance(item, dict) and item.get("freeze_id") == context["freeze_id"]), None)
    if current is None:
        raise ValueError("freeze_retire_registry_missing")
    historical_sha = current.get("manifest_sha256")
    historical_inventory = current.get("inventory_sha256")
    record = _retired_record(current, context, measurements)
    if historical_sha is not None:
        record["manifest_sha256"] = historical_sha
    if historical_inventory is not None:
        record["inventory_sha256"] = historical_inventory
    index = entries.index(current)
    entries[index] = record
    _publish_styled_json(registry_path, original, registry)
    _sync_pointer(workspace / _POINTER_FREEZE, record, measurements, kind="freeze")
    _sync_pointer(workspace / _POINTER_SOURCE, record, measurements, kind="source")
    _sync_pointer(workspace / _POINTER_CLEANING, record, measurements, kind="cleaning")
    return record


def _retired_record(current: Mapping[str, Any], context: Mapping[str, Any], measurements: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(current)
    stamp = record.get("retired_at_utc")
    if record.get("status") != "RETIRED" or not isinstance(stamp, str):
        stamp = datetime.now(timezone.utc).isoformat()
    record["status"] = "RETIRED"
    record["retired_at_utc"] = stamp
    record["retirement_reason"] = context["reason"]
    record["deleted_post_count"] = len(context["ids"])
    countable = {
        "raw_post_count": measurements["raw_post_count"],
        "raw_image_count": measurements["raw_image_count"],
        "source_record_count": measurements["source_record_count"],
        "decision_counts": measurements["decision_counts"],
    }
    for key, value in countable.items():
        if key in record:
            record[key] = value
    return record


def _sync_pointer(path: Path, record: Mapping[str, Any], measurements: Mapping[str, Any], *, kind: str) -> None:
    original = path.read_text(encoding="utf-8")
    payload = json.loads(original)
    if not isinstance(payload, dict):
        raise ValueError("freeze_retire_pointer_invalid")
    if kind == "freeze":
        preserved = payload.get("manifest_sha256")
        inventory = payload.get("inventory_sha256")
        payload.update(record)
        if isinstance(preserved, str):
            payload["manifest_sha256"] = preserved
        if isinstance(inventory, str):
            payload["inventory_sha256"] = inventory
    elif kind == "source":
        payload["data_freeze"] = dict(record)
        if "snapshot_sha256" in payload:
            payload["snapshot_sha256"] = measurements["topic_sha256"]
        if "manifest_sha256" in payload:
            payload["manifest_sha256"] = measurements["topic_manifest_sha256"]
        if "selected_posts" in payload:
            payload["selected_posts"] = measurements["selected_posts"]
    else:
        payload["data_freeze"] = dict(record)
        if "candidate_database_sha256" in payload:
            payload["candidate_database_sha256"] = measurements["candidate_sha256"]
        if "candidate_manifest_sha256" in payload:
            payload["candidate_manifest_sha256"] = measurements["candidate_manifest_sha256"]
        if "decision_counts" in payload:
            payload["decision_counts"] = measurements["decision_counts"]
        if "source_snapshot_sha256" in payload:
            payload["source_snapshot_sha256"] = measurements["topic_sha256"]
        if "record_count" in payload:
            payload["record_count"] = measurements["record_count"]
    _publish_styled_json(path, original, payload)


def _sync_inventory(context: Mapping[str, Any], record: Mapping[str, Any], measurements: Mapping[str, Any]) -> None:
    workspace: Path = context["workspace"]
    inventory_path = context["metadata_root"] / "frozen-files.json"
    original = inventory_path.read_text(encoding="utf-8")
    entries = json.loads(original)
    if not isinstance(entries, list):
        raise ValueError("freeze_retire_frozen_files_invalid")
    updated = []
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("freeze_retire_frozen_files_invalid")
        path = workspace / item["path"]
        if path.is_symlink():
            raise ValueError("freeze_retire_symlink")
        if not path.is_file():
            continue
        refreshed = dict(item)
        refreshed["size_bytes"] = path.stat().st_size
        refreshed["sha256"] = file_sha256(path)
        updated.append(refreshed)
    _publish_styled_json(inventory_path, original, updated)
    manifest_path: Path = context["manifest_path"]
    manifest_text = manifest_path.read_text(encoding="utf-8")
    spec = json.loads(manifest_text)
    previous = spec.get("retired") if isinstance(spec.get("retired"), dict) else {}
    stamp = previous.get("retired_at_utc")
    if not isinstance(stamp, str):
        stamp = record.get("retired_at_utc")
    file_count = len(updated)
    logical_bytes = sum(item["size_bytes"] for item in updated)
    spec["inventory_sha256"] = file_sha256(inventory_path)
    spec["file_count"] = file_count
    spec["logical_bytes"] = logical_bytes
    spec["retired"] = {
        "status": "RETIRED",
        "retired_at_utc": stamp,
        "retirement_reason": context["reason"],
        "deleted_post_count": len(context["ids"]),
        "prior_manifest_sha256": context["expected_sha256"],
        "raw_database_sha256": measurements["raw_sha256"],
        "raw_post_count": measurements["raw_post_count"],
        "raw_image_file_count": measurements["raw_image_count"],
        "source_snapshot_sha256": measurements["topic_sha256"],
        "source_record_count": measurements["selected_posts"],
        "candidate_database_sha256": measurements["candidate_sha256"],
        "candidate_record_count": measurements["record_count"],
        "decision_counts": measurements["decision_counts"],
    }
    _publish_styled_json(manifest_path, manifest_text, spec)
    _patch_inventory_counts(workspace, context["freeze_id"], file_count, logical_bytes)
    context["freeze_stale"] = [context["relative_manifest"] + ":" + key for key in _FREEZE_STALE if key in spec]
    context["inventory_entries"] = updated


def _patch_inventory_counts(workspace: Path, freeze_id: str, file_count: int, logical_bytes: int) -> None:
    """文件数在清单重写后才知道。原 SHA 不动，只回写已经存在的计数字段。"""

    for relative in (_REGISTRY, _POINTER_FREEZE, _POINTER_SOURCE, _POINTER_CLEANING):
        path = workspace / relative
        original = path.read_text(encoding="utf-8")
        payload = json.loads(original)
        if _apply_inventory_counts(payload, freeze_id, file_count, logical_bytes):
            _publish_styled_json(path, original, payload)


def _apply_inventory_counts(payload: Any, freeze_id: str, file_count: int, logical_bytes: int) -> bool:
    changed = False
    if isinstance(payload, dict):
        if payload.get("freeze_id") == freeze_id:
            changed = _set_present(payload, file_count, logical_bytes) or changed
        freeze = payload.get("data_freeze")
        if isinstance(freeze, dict) and freeze.get("freeze_id") == freeze_id:
            changed = _set_present(freeze, file_count, logical_bytes) or changed
        for item in payload.get("freezes", []):
            if isinstance(item, dict) and item.get("freeze_id") == freeze_id:
                changed = _set_present(item, file_count, logical_bytes) or changed
    return changed


def _set_present(record: dict[str, Any], file_count: int, logical_bytes: int) -> bool:
    changed = False
    if "file_count" in record and record["file_count"] != file_count:
        record["file_count"] = file_count
        changed = True
    if "logical_bytes" in record and record["logical_bytes"] != logical_bytes:
        record["logical_bytes"] = logical_bytes
        changed = True
    return changed


def _final_scan(context: Mapping[str, Any]) -> dict[str, int]:
    """结构化字段不得再含目标 ID；独特正文命中则失败，共享正文只计数。"""

    ids = set(context["ids"])
    shared_hits = 0
    fragments = _text_fragments(context["texts"], context["kept_texts"])
    for path in _enumerate_roots(context["roots"]):
        if _is_media_file(path):
            continue
        if path.suffix == ".sqlite":
            _check_sqlite(path, ids)
        elif path.suffix == ".json":
            _scan_json_ids(path, ids)
        elif path.suffix == ".csv":
            _scan_csv_ids(path, ids)
        else:
            _scan_loose_ids(path, ids)
        shared_hits += _scan_text(path, fragments)
    for path in _enumerate_roots(context["roots"]):
        if path.suffix == ".sqlite":
            _reject_symlink_and_sidecars(path)
    return {"shared_text_hits": shared_hits}


def _text_fragments(texts: Mapping[int, str], kept_texts: Sequence[str]) -> list[tuple[bytes, bool]]:
    fragments: list[tuple[bytes, bool]] = []
    kept = [item for item in kept_texts if isinstance(item, str) and item]
    for text in texts.values():
        if not isinstance(text, str):
            continue
        piece, shared = _classify_piece(text, kept)
        if piece is None:
            continue
        for encoded in _encoded_forms(piece):
            fragments.append((encoded, shared))
    return fragments


def _classify_piece(text: str, kept: Sequence[str]) -> tuple[str | None, bool]:
    stripped = text.strip()
    if len(stripped) < 8:
        return None, False
    if any(stripped == item for item in kept):
        return stripped[:400], True
    if len(stripped) <= 400:
        if any(stripped in item for item in kept):
            return stripped, True
        return stripped, False
    for start in range(0, len(stripped) - 80, 40):
        piece = stripped[start : start + 80]
        if not any(piece in item for item in kept):
            return piece, False
    return stripped[:80], True


def _encoded_forms(text: str) -> list[bytes]:
    forms = [text.encode("utf-8")]
    escaped = json.dumps(text, ensure_ascii=False)[1:-1].encode("utf-8")
    ascii_escaped = json.dumps(text, ensure_ascii=True)[1:-1].encode("utf-8")
    for item in (escaped, ascii_escaped):
        if item not in forms:
            forms.append(item)
    return forms


def _is_media_file(path: Path) -> bool:
    return "media-root" in path.parts or path.suffix.lower() in _MEDIA_SUFFIXES


def _scan_json_ids(path: Path, ids: set[int]) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _scan_loose_ids(path, ids)
        return
    if _json_has_id(payload, ids):
        raise ValueError("freeze_retire_residue:" + path.name)


def _json_has_id(value: Any, ids: set[int]) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and _ID_KEY_RE.fullmatch(key) and _as_post_id(item) in ids:
                return True
            if _json_has_id(item, ids):
                return True
    elif isinstance(value, list):
        return any(_json_has_id(item, ids) for item in value)
    return False


def _scan_csv_ids(path: Path, ids: set[int]) -> None:
    rows, fields, _bom, _newline = _read_csv(path.read_bytes())
    watched = [name for name in fields if _ID_KEY_RE.fullmatch(name)]
    for row in rows:
        for name in watched:
            if _as_post_id(row.get(name)) in ids:
                raise ValueError("freeze_retire_residue:" + path.name)


def _scan_loose_ids(path: Path, ids: set[int]) -> None:
    text = path.read_bytes()
    for match in re.finditer(
        rb'"(?:source_post_id|left_source_post_id|right_source_post_id|web_post_id|post_id|exact_representative_id)"\s*:\s*"?(\d+)"?',
        text,
    ):
        if int(match.group(1)) in ids:
            raise ValueError("freeze_retire_residue:" + path.name)


def _scan_text(path: Path, fragments: Sequence[tuple[bytes, bool]]) -> int:
    if not fragments:
        return 0
    blob = path.read_bytes()
    shared = False
    for needle, is_shared in fragments:
        if needle in blob:
            if is_shared:
                shared = True
            else:
                raise ValueError("freeze_retire_residue:" + path.name)
    return 1 if shared else 0


def _check_sqlite(path: Path, ids: set[int]) -> None:
    connection = _connect_readonly(path)
    try:
        quick = [row[0] for row in connection.execute("PRAGMA quick_check")]
        if quick != ["ok"]:
            raise ValueError("freeze_retire_quick_check_failed")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("freeze_retire_foreign_key_check_failed")
        for table in _user_tables(connection):
            for column in _column_names(connection, table):
                if not _SQLITE_ID_COLUMN.fullmatch(column) and not (table == "web_posts" and column == "id"):
                    continue
                rows = connection.execute(f"SELECT {_quote(column)} FROM {_quote(table)}")
                for row in rows:
                    if _as_post_id(row[0]) in ids:
                        raise ValueError("freeze_retire_residue:" + path.name)
    finally:
        connection.close()


def _write_summary(
    context: Mapping[str, Any],
    measurements: Mapping[str, Any],
    stale_fields: Sequence[str],
    scan: Mapping[str, int],
) -> dict[str, Any]:
    workspace: Path = context["workspace"]
    categories = sorted(
        {
            category
            for path in context["files"]
            if (category := _stale_category(path.relative_to(workspace))) is not None
        }
    )
    databases = []
    for role, label in (
        ("raw", "raw_database"),
        ("topic", "source_snapshot"),
        ("candidates", "candidate_database"),
    ):
        relative = _relative_of_role(context, role)
        path = workspace / relative
        databases.append({"role": label, "path": relative, "sha256": file_sha256(path), "size_bytes": path.stat().st_size})
    manifest_path = context["manifest_path"]
    summary = {
        "paths": {
            "backup_root": str(context["backup"]),
            "backup_index": str(context["backup"] / "backup-index.json"),
            "manifest": context["relative_manifest"],
        },
        "counts": {
            "deleted_post_count": len(context["ids"]),
            "raw_post_count": measurements["raw_post_count"],
            "raw_image_file_count": measurements["raw_image_count"],
            "source_record_count": measurements["source_record_count"],
            "candidate_record_count": measurements["record_count"],
            "decision_counts": measurements["decision_counts"],
            "shared_text_hits": scan["shared_text_hits"],
            "shared_embedding_kept": context["embedding_shared"],
            "rewritten_file_count": sum(item["action"] == "rewrite" for item in context["index"]["files"]),
            "deleted_file_count": sum(item["action"] == "delete" for item in context["index"]["files"]),
        },
        "hashes": {
            "prior_manifest_sha256": context["expected_sha256"],
            "manifest_sha256": file_sha256(manifest_path),
            "databases": databases,
        },
        "stale_fields": sorted(set(stale_fields) | set(context.get("freeze_stale", []))),
        "stale_hash_evidence": categories,
    }
    _publish_json(context["backup"] / "run_summary.json", summary)
    return summary


def _publish_styled_json(path: Path, original: str, payload: Any) -> None:
    style = _json_style(original)
    rendered = json.dumps(payload, ensure_ascii=style["ensure_ascii"], indent=style["indent"], separators=style["separators"])
    if style["trailing_newline"]:
        rendered += "\n"
    data = rendered.encode("utf-8")
    if data != original.encode("utf-8"):
        _publish_bytes(path, data)


def _json_style(text: str) -> dict[str, Any]:
    indent: int | str | None = None
    for line in text.splitlines()[1:]:
        if not line.strip():
            continue
        if line.startswith("\t"):
            indent = "\t"
            break
        width = len(line) - len(line.lstrip(" "))
        if width:
            indent = width
            break
    raw_non_ascii = any(ord(char) > 127 for char in text)
    ensure_ascii = (not raw_non_ascii) and "\\u" in text
    separators = (", ", ": ")
    if indent is None and ", " not in text and "," in text:
        separators = (",", ":")
    return {
        "indent": indent,
        "ensure_ascii": ensure_ascii,
        "separators": separators,
        "trailing_newline": text.endswith("\n"),
    }


def _publish_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _publish_bytes(path, data)


def _publish_bytes(path: Path, data: bytes) -> None:
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _temporary_path(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=".freeze-retire-", suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    return Path(name)


def _read_csv(raw: bytes) -> tuple[list[dict[str, str]], list[str], bool, str]:
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    newline = "\r\n" if "\r\n" in text else "\n"
    previous = csv.field_size_limit()
    csv.field_size_limit(max(previous, len(raw), 1))
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if reader.fieldnames is None:
            raise ValueError("freeze_retire_csv_header_missing")
        fields = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    finally:
        csv.field_size_limit(previous)
    return rows, fields, bom, newline


def _csv_bytes(rows: Sequence[Mapping[str, str]], fields: Sequence[str], *, bom: bool, newline: str) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator=newline, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fields})
    data = buffer.getvalue().encode("utf-8")
    if bom:
        return b"\xef\xbb\xbf" + data
    return data


def _csv_data_rows(path: Path) -> int:
    rows, _fields, _bom, _newline = _read_csv(path.read_bytes())
    return len(rows)


def _record_hits(record: Any, ids: set[int]) -> bool:
    if not isinstance(record, dict):
        return False
    for key, value in record.items():
        if isinstance(key, str) and _ID_KEY_RE.fullmatch(key) and _as_post_id(value) in ids:
            return True
    return False


def _as_post_id(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return None


def _connect_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _column_exists(connection: sqlite3.Connection, table: str, column: str) -> bool:
    return column in _column_names(connection, table)


def _column_names(connection: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in connection.execute(f"PRAGMA table_info({_quote(table)})")]


def _user_tables(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    return [str(row[0]) for row in rows]


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _rows_digest(rows: Sequence[Sequence[Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(list(row), ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _media_root(roots: Sequence[Path]) -> Path:
    found = [path for root in roots for path in ([root] if root.is_dir() else []) if (path / "media-root").is_dir()]
    nested = [path / "media-root" for path in found]
    extra = [path for root in roots if root.is_dir() for path in root.rglob("media-root") if path.is_dir()]
    candidates = list(dict.fromkeys([*nested, *extra]))
    if len(candidates) != 1:
        raise ValueError("freeze_retire_media_root_missing")
    return candidates[0]


def _relative(workspace: Path, path: Path) -> str:
    return path.resolve().relative_to(workspace).as_posix()


def _fsync(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
