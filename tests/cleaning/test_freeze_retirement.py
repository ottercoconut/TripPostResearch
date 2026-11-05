"""用合成冻结目录验证退役：不读取本仓库以外的研究数据。"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from tourism_ugc_study.cleaning.freeze_retirement import _json_style, _publish_styled_json, retire_freeze
from tourism_ugc_study.cleaning.research_freeze import verify_freeze
from tourism_ugc_study.cleaning.source_snapshot import file_sha256

_NS = "a" * 32
_DEAD = "b" * 64
_SHARED = "c" * 64
_H32 = "0123456789abcdef0123456789abcdef"
_FREEZE = "research-data-test"
_REASON = "无法恢复实名，从各层删除这些帖"
_ROOTS = [
    "data/processed/raw-crawl-20260912",
    "data/processed/source-snapshots/topic-relevant-20260911.sqlite",
    "data/processed/source-snapshots/topic-relevant-20260911.manifest.json",
    "data/processed/research-cleaning-20260911-autodl-r2",
    "results/research-cleaning-20260911-autodl-r2",
]


class _World:
    """一次合成退役所需的路径。所有字节都在测试临时目录里。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.workspace = root / "repo"
        self.backup = root / "backup"
        self.inputs = root / "inputs"
        self.manifest = self.workspace / "data/processed/research-freezes/mini/freeze-manifest.json"
        self.post_ids = self.inputs / "post-ids.json"
        self.media_db = self.inputs / "media.sqlite"
        self.records = self.workspace / "results/research-cleaning-20260911-autodl-r2/records.json"
        self.batch_csv = self.workspace / "results/research-cleaning-20260911-autodl-r2/inference-batches/batch-001.csv"
        self.private_map = (
            self.workspace
            / "results/research-cleaning-20260911-autodl-r2/inference/batch-001"
            / _H32
            / "manual-review-private-map.json"
        )
        self.annotation = self.private_map.with_name("manual-review-tourism-relevance-annotation.csv")
        self.keep_csv = self.workspace / "data/processed/research-cleaning-20260911-autodl-r2/keep-review.csv"
        self.keep_map = self.keep_csv.with_name("keep-review-private-map.json")
        self.manual_csv = self.workspace / "data/processed/research-cleaning-20260911-autodl-r2/manual-review-review.csv"
        self.near = self.workspace / "results/research-cleaning-20260911-autodl-r2/near-duplicate-candidates.json"
        self.round_manifest = self.workspace / "results/research-cleaning-20260911-autodl-r2/round-manifest.json"
        self.raw_db = self.workspace / "data/processed/raw-crawl-20260912/raw-crawl.sqlite"
        self.topic_db = self.workspace / "data/processed/source-snapshots/topic-relevant-20260911.sqlite"
        self.candidate_db = self.workspace / "data/processed/research-cleaning-20260911-autodl-r2/cleaning-candidates.sqlite"
        self.media_root = self.workspace / "data/processed/raw-crawl-20260912/media-root"
        self.cache = (
            self.workspace
            / "results/research-cleaning-20260911-autodl-r2/inference/batch-001/embedding-cache"
            / _NS
        )
        self.replacements = {
            "raw": self.inputs / "raw-r2.sqlite",
            "topic": self.inputs / "topic-r2.sqlite",
            "candidates": self.inputs / "candidates-r2.sqlite",
        }
        self.replacement_manifests = {role: self.inputs / f"{role}-revision-manifest.json" for role in self.replacements}
        self.digest = ""


def _sha(path: Path) -> str:
    return file_sha256(path)


def _write_json(path: Path, payload: object, *, indent: int = 2, ensure_ascii: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent) + "\n", encoding="utf-8")


def _posts_db(path: Path, posts: list[tuple[int, str, int, str, int]], images: list[tuple[int, int, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            """CREATE TABLE web_posts (
                id INTEGER PRIMARY KEY,
                content_text TEXT NOT NULL,
                topic_relevant INTEGER NOT NULL,
                platform_key TEXT NOT NULL,
                post_images_count INTEGER NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE web_post_images (
                id INTEGER PRIMARY KEY,
                web_post_id INTEGER NOT NULL REFERENCES web_posts(id),
                local_path TEXT NOT NULL,
                sha256 TEXT NOT NULL
            )"""
        )
        conn.executemany("INSERT INTO web_posts VALUES (?,?,?,?,?)", posts)
        conn.executemany("INSERT INTO web_post_images VALUES (?,?,?,?)", images)


def _candidate_db(path: Path, rows: list[tuple[int, str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """CREATE TABLE cleaning_candidates (
                source_post_id INTEGER PRIMARY KEY,
                decision TEXT NOT NULL,
                platform_key TEXT NOT NULL,
                exact_representative_id INTEGER NOT NULL
            )"""
        )
        conn.executemany("INSERT INTO cleaning_candidates VALUES (?,?,?,?)", rows)


def _revision(path: Path, database: Path) -> None:
    _write_json(
        path,
        {
            "status": "DERIVED_NOT_FROZEN",
            "output": {"path": str(database), "sha256": _sha(database), "size_bytes": database.stat().st_size},
        },
    )


def _inventory(workspace: Path) -> list[dict[str, object]]:
    entries = []
    for relative in _ROOTS:
        root = workspace / relative
        paths = [root] if root.is_file() else sorted(item for item in root.rglob("*") if item.is_file())
        for path in paths:
            entries.append(
                {"path": path.relative_to(workspace).as_posix(), "sha256": _sha(path), "size_bytes": path.stat().st_size}
            )
    return entries


def _build(tmp_path: Path) -> _World:
    """摆一套迷你冻结。7、8 要删；3 与 8 共用正文；3 与 7 的向量键有一个被保留帖引用。"""

    world = _World(tmp_path.resolve())
    world.inputs.mkdir()
    posts = [
        (3, "SHARED_BODY", 1, "weibo", 1),
        (4, "KEPT_ONLY_4", 1, "weibo", 0),
        (7, "UNIQUE_GONE_7", 1, "weibo", 1),
        (8, "SHARED_BODY", 1, "weibo", 0),
    ]
    images = [
        (1, 3, "data/media/keep.jpg", "ab" * 32),
        (2, 7, "data/media/gone/drop.jpg", "cd" * 32),
    ]
    kept_posts = [item for item in posts if item[0] in {3, 4}]
    kept_images = [item for item in images if item[1] == 3]
    _posts_db(world.raw_db, posts, images)
    _posts_db(world.topic_db, posts, images)
    _posts_db(world.media_db, posts, images)
    _posts_db(world.replacements["raw"], kept_posts, kept_images)
    _posts_db(world.replacements["topic"], kept_posts, kept_images)
    all_rows = [(3, "keep", "weibo", 3), (4, "exclude", "weibo", 4), (7, "manual_review", "weibo", 7), (8, "keep", "weibo", 8)]
    _candidate_db(world.candidate_db, all_rows)
    _candidate_db(world.replacements["candidates"], all_rows[:2])
    for role, database in world.replacements.items():
        _revision(world.replacement_manifests[role], database)
    _write_json(world.post_ids, {"ids": [8, 7]})

    keep_jpg = world.media_root / "data/media/keep.jpg"
    drop_jpg = world.media_root / "data/media/gone/drop.jpg"
    keep_jpg.parent.mkdir(parents=True)
    drop_jpg.parent.mkdir(parents=True)
    keep_jpg.write_bytes(b"keep-image")
    drop_jpg.write_bytes(b"drop-image")
    _write_json(
        world.media_root.parent / "media-inventory.json",
        {
            "artifact_kind": "raw-media-byte-inventory",
            "file_count": 2,
            "logical_bytes": keep_jpg.stat().st_size + drop_jpg.stat().st_size,
            "entries": [
                {"local_path": "data/media/keep.jpg", "sha256": "ab" * 32, "size_bytes": keep_jpg.stat().st_size},
                {"local_path": "data/media/gone/drop.jpg", "sha256": "cd" * 32, "size_bytes": drop_jpg.stat().st_size},
            ],
        },
    )
    _write_json(
        world.raw_db.with_suffix(".manifest.json"),
        {
            "database": {"path": "raw-crawl.sqlite", "sha256": "old", "size_bytes": 1},
            "selected_posts": 4,
            "topic_counts": {"1": 4},
            "platform_counts": {"weibo": 4},
            "table_evidence": {
                "web_posts": {"count": 4, "columns": ["id"], "rows_sha256": "old"},
                "web_post_images": {"count": 2, "columns": ["id"], "rows_sha256": "old"},
            },
            "source_total_posts": 9,
            "source_main_sha256_before": "abc",
        },
    )
    _write_json(
        world.topic_db.with_suffix(".manifest.json"),
        {
            "database": {"path": "topic-relevant-20260911.sqlite", "sha256": "old", "size_bytes": 1},
            "selected_posts": 4,
            "source_total_posts": 9,
        },
    )
    records = [
        {"source_post_id": 3, "body": "SHARED_BODY", "mark": "α"},
        {"source_post_id": 4, "body": "KEPT_ONLY_4"},
        {"source_post_id": 7, "body": "UNIQUE_GONE_7"},
        {"source_post_id": 8, "body": "SHARED_BODY"},
    ]
    _write_json(world.records, records, indent=4, ensure_ascii=False)
    near = [
        {"left_source_post_id": 7, "right_source_post_id": 4},
        {"left_source_post_id": 3, "right_source_post_id": 4},
    ]
    _write_json(world.near, near)
    candidate_near = world.candidate_db.parent / "near-duplicate-candidates.json"
    _write_json(candidate_near, near)
    batch = "\r\n".join(
        [
            "source_post_id,source_version,component_id,title,body,source_status",
            "7,1,c7,t7,UNIQUE_GONE_7,ok",
            "3,1,c3,t3,SHARED_BODY,ok",
            "4,1,c4,t4,KEPT_ONLY_4,ok",
            "8,1,c8,t8,SHARED_BODY,ok",
        ]
    ) + "\r\n"
    world.batch_csv.parent.mkdir(parents=True, exist_ok=True)
    world.batch_csv.write_bytes(b"\xef\xbb\xbf" + batch.encode("utf-8"))
    scored = world.private_map.parent
    scored.mkdir(parents=True)
    predictions = "\n".join(
        [
            "source_post_id,source_version,component_id,normalized_model_text,p_unrelated,routing_action,provisional_tourism_label",
            "7,1,c7,x,0.2,exclude,unrelated",
            "3,1,c3,y,0.1,keep,related",
        ]
    ) + "\n"
    (scored / "tourism-relevance-predictions.csv").write_text(predictions, encoding="utf-8")
    _write_json(
        scored / "incremental-scored-records.json",
        [
            {"source_post_id": 7, "embedding_cache_key": _NS + ":" + _DEAD},
            {"source_post_id": 3, "embedding_cache_key": _NS + ":" + _SHARED},
            {"source_post_id": 8, "embedding_cache_key": _NS + ":" + _SHARED},
        ],
    )
    _write_json(
        world.private_map,
        [{"source_post_id": 7, "task_id": "inf-7"}, {"source_post_id": 3, "task_id": "inf-3", "mark": "α"}],
        ensure_ascii=True,
    )
    world.annotation.write_text(
        "task_id,sample_run_id,normalized_model_text,tourism_label\ninf-7,run,text-7,\ninf-3,run,text-3,\n",
        encoding="utf-8",
    )
    (scored / "checkpoint-records").mkdir()
    _write_json(scored / "checkpoint-records/0000007.json", {"source_post_id": 7, "embedding_cache_key": _NS + ":" + _DEAD})
    _write_json(scored / "checkpoint-records/0000003.json", {"source_post_id": 3, "embedding_cache_key": _NS + ":" + _SHARED})
    for key, prefix in ((_DEAD, "bb"), (_SHARED, "cc")):
        folder = world.cache / prefix / key
        folder.mkdir(parents=True)
        (folder / "embedding.npy").write_bytes(b"vector")
        _write_json(folder / "receipt.json", {"sha256": "ef" * 32})
    _write_json(world.keep_map, [{"source_post_id": 3, "task_id": "keep-3"}, {"source_post_id": 8, "task_id": "keep-8"}])
    world.keep_csv.write_text(
        "task_id,sample_run_id,normalized_model_text,tourism_label\nkeep-3,run,kept,\nkeep-8,run,shared,\n",
        encoding="utf-8",
    )
    _write_json(
        world.manual_csv.with_name("manual-review-review-private-map.json"),
        [{"source_post_id": 7, "task_id": "man-7"}, {"source_post_id": 4, "task_id": "man-4"}],
    )
    world.manual_csv.write_text(
        "task_id,sample_run_id,normalized_model_text,tourism_label\nman-7,run,gone,\nman-4,run,stay,\n",
        encoding="utf-8",
    )
    _write_json(
        world.candidate_db.parent / "candidate-manifest.json",
        {
            "database_filename": "cleaning-candidates.sqlite",
            "database_sha256": "old",
            "record_count": 4,
            "decision_counts": {"exclude": 1, "keep": 2, "manual_review": 1},
            "source_snapshot_sha256": "old",
            "near_candidate_count": 2,
            "near_candidates_sha256": "old",
            "round_manifest_sha256": "old-round",
            "batch_receipts": {"batch-001": {"manifest_sha256": "deadbeef"}},
            "review_tasks": {
                "keep": {
                    "filename": "keep-review.csv",
                    "sha256": "old",
                    "count": 2,
                    "private_map_filename": "keep-review-private-map.json",
                    "private_map_sha256": "old",
                }
            },
        },
    )
    _write_json(world.round_manifest, {"record_count": 4, "records_sha256": "abc"})
    _write_json(world.round_manifest.with_name("final-audit.json"), {"status": "ok", "file_count": 4})
    log = world.round_manifest.parent / "logs/batch.log"
    log.parent.mkdir()
    log.write_text("count=4\n", encoding="utf-8")
    receipt = world.round_manifest.parent / "receipts/input-receipt.json"
    receipt.parent.mkdir()
    _write_json(receipt, {"count": 4, "sha256": "abc"})

    entries = _inventory(world.workspace)
    frozen = world.manifest.parent / "frozen-files.json"
    _write_json(frozen, entries)
    logical = sum(int(item["size_bytes"]) for item in entries)
    _write_json(
        world.manifest,
        {
            "artifact_kind": "research-data-freeze-specification",
            "contract": "research-data-freeze-v1",
            "freeze_id": _FREEZE,
            "inventory_filename": "frozen-files.json",
            "inventory_sha256": _sha(frozen),
            "roots": _ROOTS,
            "file_count": len(entries),
            "logical_bytes": logical,
            "raw_database_sha256": "old",
            "raw_post_count": 4,
            "source_snapshot_sha256": "old",
            "candidate_database_sha256": "old",
            "media": {"file_count": 2},
            "cleaning": {"record_count": 4, "decision_counts": {"keep": 2}},
        },
    )
    world.digest = _sha(world.manifest)
    record = {
        "freeze_id": _FREEZE,
        "status": "FROZEN",
        "manifest": world.manifest.relative_to(world.workspace).as_posix(),
        "manifest_sha256": world.digest,
        "inventory_sha256": _sha(frozen),
        "sealed_at_utc": "2026-09-12T00:00:00+00:00",
        "raw_post_count": 4,
        "raw_image_count": 2,
        "source_record_count": 4,
        "decision_counts": {"exclude": 1, "keep": 2, "manual_review": 1},
        "file_count": len(entries),
        "logical_bytes": logical,
    }
    registry = world.workspace / "governance"
    registry.mkdir()
    _write_json(registry / "research-data-freezes.json", {"contract": "research-data-freeze-registry-v1", "freezes": [record]})
    processed = world.workspace / "data/processed"
    _write_json(processed / "current-freeze.json", record)
    _write_json(
        processed / "current-source.json",
        {
            "snapshot_sha256": "old",
            "manifest_sha256": "old",
            "selected_posts": 4,
            "data_freeze": record,
        },
    )
    _write_json(
        processed / "current-cleaning.json",
        {
            "candidate_database_sha256": "old",
            "candidate_manifest_sha256": "old",
            "source_snapshot_sha256": "old",
            "decision_counts": {"exclude": 1, "keep": 2, "manual_review": 1},
            "record_count": 4,
            "data_freeze": record,
        },
    )
    return world


def _run(world: _World, **kwargs: object) -> dict[str, object]:
    confirm = kwargs.pop("confirm_retire", _FREEZE)
    return retire_freeze(
        world.workspace,
        world.manifest,
        world.digest,
        world.post_ids,
        world.replacements,
        world.replacement_manifests,
        world.media_db,
        kwargs.pop("backup_root", world.backup),
        _REASON,
        confirm_retire=confirm if isinstance(confirm, str) or confirm is None else _FREEZE,
        dry_run=bool(kwargs.pop("dry_run", False)),
        chflags=kwargs.pop("chflags", None),  # type: ignore[arg-type]
    )


def _unlock(root: Path) -> None:
    if not root.exists():
        return
    paths = [root, *root.rglob("*")]
    for path in sorted(paths, key=lambda item: len(item.parts)):
        if hasattr(os, "chflags"):
            try:
                os.chflags(path, 0)
            except OSError:
                pass
        if path.exists():
            os.chmod(path, 0o700 if path.is_dir() else 0o600)


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    world = _build(tmp_path)
    before = _sha(world.records)
    result = _run(world, dry_run=True, confirm_retire=None)
    assert result["dry_run"] is True
    assert result["deleted_post_count"] == 2
    assert result["delete_count"] > 0
    assert result["rewrite_count"] > 0
    assert _sha(world.records) == before
    assert not world.backup.exists()


def test_confirm_mismatch_rejects_before_backup(tmp_path: Path) -> None:
    world = _build(tmp_path)
    before = _sha(world.records)
    with pytest.raises(ValueError, match="confirm_mismatch"):
        _run(world, confirm_retire="other-freeze")
    assert _sha(world.records) == before
    assert not world.backup.exists()


def test_backup_precedes_mutation_and_flags_are_cleared(tmp_path: Path) -> None:
    world = _build(tmp_path)
    original = world.records.read_bytes()

    def explode(path: Path, flags: int) -> None:
        raise RuntimeError("stop-after-backup")

    with pytest.raises(RuntimeError, match="stop-after-backup"):
        _run(world, chflags=explode)
    archived = world.backup / world.records.relative_to(world.workspace)
    assert archived.read_bytes() == original
    assert world.records.read_bytes() == original
    assert json.loads((world.backup / "backup-index.json").read_text())["verified"] is True


def test_retire_purges_every_file_kind_and_keeps_shared_embedding(tmp_path: Path) -> None:
    world = _build(tmp_path)
    original_records = _sha(world.records)
    original_round = world.round_manifest.read_bytes()
    original_log = (world.round_manifest.parent / "logs/batch.log").read_bytes()
    if hasattr(os, "chflags"):
        os.chflags(world.records.parent, stat.UF_IMMUTABLE)
        os.chflags(world.records, stat.UF_IMMUTABLE)
    try:
        summary = _run(world)
    finally:
        _unlock(world.workspace)

    archived = world.backup / world.records.relative_to(world.workspace)
    assert _sha(archived) == original_records
    records_text = world.records.read_text(encoding="utf-8")
    assert records_text.splitlines()[1].startswith("    ")
    assert "α" in records_text and "\\u03b1" not in records_text
    assert "UNIQUE_GONE_7" not in records_text and "SHARED_BODY" in records_text
    assert {item["source_post_id"] for item in json.loads(records_text)} == {3, 4}
    private = world.private_map.read_text(encoding="utf-8")
    assert "\\u03b1" in private and "α" not in private and "inf-7" not in private
    annotation = world.annotation.read_text(encoding="utf-8")
    assert "inf-7" not in annotation and "inf-3" in annotation
    assert "keep-8" not in world.keep_csv.read_text(encoding="utf-8") and "keep-3" in world.keep_csv.read_text(encoding="utf-8")
    assert "man-7" not in world.manual_csv.read_text(encoding="utf-8") and "man-4" in world.manual_csv.read_text(encoding="utf-8")
    batch = world.batch_csv.read_bytes()
    assert batch.startswith(b"\xef\xbb\xbf") and b"\r\n" in batch
    assert batch.decode("utf-8-sig").splitlines()[0] == "source_post_id,source_version,component_id,title,body,source_status"
    assert b"UNIQUE_GONE_7" not in batch and b"SHARED_BODY" in batch
    near = json.loads(world.near.read_text(encoding="utf-8"))
    assert near == [{"left_source_post_id": 3, "right_source_post_id": 4}]
    assert world.near.read_text(encoding="utf-8").splitlines()[1].startswith("  ")
    scored = world.private_map.parent
    assert not (scored / "checkpoint-records/0000007.json").exists()
    assert (scored / "checkpoint-records/0000003.json").is_file()
    predictions = (scored / "tourism-relevance-predictions.csv").read_text(encoding="utf-8")
    assert "\n7," not in predictions and "\n3," in predictions
    incremental = json.loads((scored / "incremental-scored-records.json").read_text(encoding="utf-8"))
    assert [item["source_post_id"] for item in incremental] == [3]
    assert not (world.cache / "bb").exists()
    assert (world.cache / "cc" / _SHARED / "embedding.npy").is_file()
    assert not (world.media_root / "data/media/gone").exists()
    assert (world.media_root / "data/media/keep.jpg").read_bytes() == b"keep-image"
    inventory = json.loads((world.media_root.parent / "media-inventory.json").read_text(encoding="utf-8"))
    assert [item["local_path"] for item in inventory["entries"]] == ["data/media/keep.jpg"]
    assert inventory["file_count"] == 1
    assert world.round_manifest.read_bytes() == original_round
    assert (world.round_manifest.parent / "logs/batch.log").read_bytes() == original_log
    with sqlite3.connect(world.raw_db) as conn:
        assert [row[0] for row in conn.execute("SELECT id FROM web_posts ORDER BY id")] == [3, 4]
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    raw_manifest = json.loads(world.raw_db.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert raw_manifest["selected_posts"] == 2
    assert raw_manifest["source_total_posts"] == 9
    assert raw_manifest["database"]["sha256"] == _sha(world.raw_db)
    assert raw_manifest["database"]["size_bytes"] == world.raw_db.stat().st_size
    assert raw_manifest["table_evidence"]["web_posts"]["count"] == 2
    assert raw_manifest["table_evidence"]["web_posts"]["rows_sha256"] != "old"
    topic_manifest = json.loads(world.topic_db.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert topic_manifest["selected_posts"] == 2
    candidate_manifest = json.loads((world.candidate_db.parent / "candidate-manifest.json").read_text(encoding="utf-8"))
    assert candidate_manifest["record_count"] == 2
    assert candidate_manifest["decision_counts"] == {"exclude": 1, "keep": 1}
    assert candidate_manifest["database_sha256"] == _sha(world.candidate_db)
    assert candidate_manifest["near_candidate_count"] == 1
    assert candidate_manifest["near_candidates_sha256"] == _sha(world.candidate_db.parent / "near-duplicate-candidates.json")
    assert candidate_manifest["review_tasks"]["keep"]["count"] == 1
    assert candidate_manifest["batch_receipts"]["batch-001"]["manifest_sha256"] == "deadbeef"
    spec = json.loads(world.manifest.read_text(encoding="utf-8"))
    assert spec["retired"]["status"] == "RETIRED"
    assert spec["retired"]["deleted_post_count"] == 2
    assert spec["retired"]["prior_manifest_sha256"] == world.digest
    assert spec["raw_database_sha256"] == "old"
    assert spec["retired"]["raw_database_sha256"] == _sha(world.raw_db)
    registry = json.loads((world.workspace / "governance/research-data-freezes.json").read_text(encoding="utf-8"))
    entry = registry["freezes"][0]
    assert entry["status"] == "RETIRED"
    assert entry["manifest_sha256"] == world.digest
    assert entry["deleted_post_count"] == 2
    assert entry["decision_counts"] == {"exclude": 1, "keep": 1}
    assert entry["retirement_reason"] == _REASON
    source = json.loads((world.workspace / "data/processed/current-source.json").read_text(encoding="utf-8"))
    assert source["snapshot_sha256"] == _sha(world.topic_db)
    assert source["selected_posts"] == 2
    assert source["data_freeze"]["status"] == "RETIRED"
    cleaning = json.loads((world.workspace / "data/processed/current-cleaning.json").read_text(encoding="utf-8"))
    assert cleaning["candidate_database_sha256"] == _sha(world.candidate_db)
    assert cleaning["decision_counts"] == {"exclude": 1, "keep": 1}
    assert cleaning["data_freeze"]["manifest_sha256"] == world.digest
    current = json.loads((world.workspace / "data/processed/current-freeze.json").read_text(encoding="utf-8"))
    assert current["status"] == "RETIRED" and current["manifest_sha256"] == world.digest
    frozen = json.loads((world.manifest.parent / "frozen-files.json").read_text(encoding="utf-8"))
    frozen_paths = {item["path"] for item in frozen}
    assert "data/processed/raw-crawl-20260912/media-root/data/media/gone/drop.jpg" not in frozen_paths
    raw_entry = next(item for item in frozen if item["path"].endswith("raw-crawl.sqlite"))
    assert raw_entry["sha256"] == _sha(world.raw_db) and raw_entry["size_bytes"] == world.raw_db.stat().st_size
    rendered = json.dumps(summary, ensure_ascii=False)
    assert "UNIQUE_GONE_7" not in rendered and "SHARED_BODY" not in rendered and "α" not in rendered
    assert summary["counts"]["shared_text_hits"] >= 1
    assert summary["counts"]["shared_embedding_kept"] == 1
    assert summary["counts"]["deleted_post_count"] == 2
    assert {"final-audit", "logs", "receipts", "round-manifest"} <= set(summary["stale_hash_evidence"])
    assert any(item.endswith("source_total_posts") for item in summary["stale_fields"])
    assert any(item.endswith("batch_receipts") for item in summary["stale_fields"])
    assert stat.S_IMODE(world.records.stat().st_mode) == 0o600
    assert stat.S_IMODE(world.records.parent.stat().st_mode) == 0o700
    if hasattr(os, "chflags"):
        assert world.records.stat().st_flags & stat.UF_IMMUTABLE == 0
    with pytest.raises(ValueError, match="已退役"):
        verify_freeze(world.workspace, world.manifest, world.digest)


def test_rerun_does_not_overwrite_backup_and_uses_saved_task_ids(tmp_path: Path) -> None:
    world = _build(tmp_path)
    first = _run(world)
    stamp = json.loads(world.manifest.read_text(encoding="utf-8"))["retired"]["retired_at_utc"]
    archived = world.backup / world.keep_csv.relative_to(world.workspace)
    backup_hash = _sha(archived)
    records_backup = _sha(world.backup / world.records.relative_to(world.workspace))
    world.keep_csv.write_bytes(archived.read_bytes())
    second = _run(world)
    assert json.loads(world.manifest.read_text(encoding="utf-8"))["retired"]["retired_at_utc"] == stamp
    assert _sha(archived) == backup_hash
    assert _sha(world.backup / world.records.relative_to(world.workspace)) == records_backup
    assert "keep-8" not in world.keep_csv.read_text(encoding="utf-8")
    assert second["counts"]["shared_embedding_kept"] == first["counts"]["shared_embedding_kept"]
    assert not (world.media_root / "data/media/gone").exists()


def test_residue_fails_and_shared_text_alone_does_not(tmp_path: Path) -> None:
    world = _build(tmp_path)
    extra = world.round_manifest.parent / "extra-residue.json"
    _write_json(extra, {"source_post_id": 7, "body": "UNIQUE_GONE_7"})
    with pytest.raises(ValueError, match="freeze_retire_residue"):
        _run(world)
    assert extra.is_file()
    assert (world.backup / "backup-index.json").is_file()
    assert "UNIQUE_GONE_7" not in world.records.read_text(encoding="utf-8")


def test_rejects_sidecar_symlink_escape_and_bad_replacement(tmp_path: Path) -> None:
    world = _build(tmp_path)
    wal = Path(str(world.raw_db) + "-wal")
    wal.write_bytes(b"unmerged")
    with pytest.raises(ValueError, match="sidecar"):
        _run(world)
    assert not world.backup.exists()
    wal.unlink()

    link = world.round_manifest.parent / "linked.json"
    link.symlink_to(world.round_manifest)
    with pytest.raises(ValueError, match="symlink"):
        _run(world)
    link.unlink()

    with pytest.raises(ValueError, match="backup_inside_roots"):
        _run(world, backup_root=world.workspace / "results/research-cleaning-20260911-autodl-r2/inside")
    assert not (world.workspace / "results/research-cleaning-20260911-autodl-r2/inside").exists()

    bad_media = world.inputs / "bad-media.sqlite"
    _posts_db(bad_media, [(7, "UNIQUE_GONE_7", 1, "weibo", 1), (3, "SHARED_BODY", 1, "weibo", 0), (4, "KEPT_ONLY_4", 1, "weibo", 0), (8, "SHARED_BODY", 1, "weibo", 0)], [(1, 7, "../secret.jpg", "ab" * 32)])
    world.media_db = bad_media
    with pytest.raises(ValueError, match="media_path_invalid"):
        _run(world)
    world.media_db = world.inputs / "media.sqlite"

    manifest = json.loads(world.replacement_manifests["raw"].read_text(encoding="utf-8"))
    manifest["output"]["sha256"] = "0" * 64
    world.replacement_manifests["raw"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="replacement_hash_mismatch"):
        _run(world)
    manifest["status"] = "FROZEN"
    manifest["output"]["sha256"] = _sha(world.replacements["raw"])
    world.replacement_manifests["raw"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="replacement_not_derived"):
        _run(world)
    assert not world.backup.exists()


def test_verify_reports_retired_without_freeze_checks(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    manifest = workspace / "data/processed/freeze-mini/freeze-manifest.json"
    manifest.parent.mkdir(parents=True)
    payload = {
        "contract": "research-data-freeze-v1",
        "freeze_id": _FREEZE,
        "inventory_filename": "frozen-files.json",
        "roots": ["results/synthetic-run"],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash_mismatch"):
        verify_freeze(workspace, manifest, "0" * 64)
    payload["retired"] = {"status": "RETIRED"}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="已退役"):
        verify_freeze(workspace, manifest, "0" * 64)
    payload.pop("retired")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    registry = workspace / "governance"
    registry.mkdir()
    (registry / "research-data-freezes.json").write_text(
        json.dumps(
            {
                "freezes": [
                    {
                        "freeze_id": _FREEZE,
                        "status": "RETIRED",
                        "manifest": manifest.relative_to(workspace).as_posix(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    asset = workspace / "results/synthetic-run/file.txt"
    asset.parent.mkdir(parents=True)
    asset.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="已退役"):
        verify_freeze(workspace, manifest, hashlib.sha256(b"wrong").hexdigest())


def test_cli_dry_run_and_rejected_confirm(tmp_path: Path) -> None:
    world = _build(tmp_path)
    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts/cleaning_retire_freeze.py"
    common = [
        sys.executable,
        str(script),
        "--workspace",
        str(world.workspace),
        "--manifest",
        str(world.manifest),
        "--expected-sha256",
        world.digest,
        "--post-ids",
        str(world.post_ids),
        "--media-db",
        str(world.media_db),
        "--backup-root",
        str(world.backup),
        "--reason",
        _REASON,
    ]
    for role, path in world.replacements.items():
        common.extend(["--replacement", f"{role}={path}"])
    for role, path in world.replacement_manifests.items():
        common.extend(["--replacement-manifest", f"{role}={path}"])
    before = _sha(world.records)
    dry = subprocess.run([*common, "--dry-run"], cwd=repo, check=False, capture_output=True, text=True)
    assert dry.returncode == 0, dry.stderr
    assert json.loads(dry.stdout)["dry_run"] is True
    assert not world.backup.exists()
    rejected = subprocess.run([*common, "--confirm-retire", "nope"], cwd=repo, check=False, capture_output=True, text=True)
    assert rejected.returncode == 1
    assert "confirm_mismatch" in rejected.stderr
    assert _sha(world.records) == before


def _assert_json_roundtrip(path: Path, text: str, *, indent: int | None, separators: tuple[str, str]) -> None:
    """写回未改动的 JSON。字节必须相同，且不得替换文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    before = path.stat()
    style = _json_style(text)
    assert style["indent"] == indent
    assert style["separators"] == separators
    assert style["trailing_newline"] is text.endswith("\n")
    _publish_styled_json(path, text, json.loads(text))
    after = path.stat()
    assert path.read_text(encoding="utf-8") == text
    assert after.st_ino == before.st_ino
    assert after.st_mtime_ns == before.st_mtime_ns


def test_indented_json_roundtrip_does_not_rewrite(tmp_path: Path) -> None:
    """缩进 JSON 经样式识别写回时字节不变，也不落盘。"""

    list_text = [{"地名": "桂林", "source_post_id": 3}, {"地名": "阳朔", "source_post_id": 4}]
    dict_text = {"批次": "甲", "帖子": [{"source_post_id": 3, "说明": "无目标"}]}
    samples = [
        (list_text, 2, False, True),
        (dict_text, 2, False, False),
        (list_text, 2, True, False),
        ({"说明": "阳朔", "条目": [3, 4], "状态": "保留"}, 4, True, True),
    ]
    for index, (payload, indent, ensure_ascii, trailing) in enumerate(samples):
        text = json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent)
        if trailing:
            text += "\n"
        assert _json_style(text)["ensure_ascii"] is ensure_ascii
        _assert_json_roundtrip(
            tmp_path / f"styled-{index}.json",
            text,
            indent=indent,
            separators=(",", ": "),
        )
    _assert_json_roundtrip(tmp_path / "compact.json", '{"left":1,"right":2}', indent=None, separators=(",", ":"))
    _assert_json_roundtrip(
        tmp_path / "spaced.json",
        '{"left": 1, "right": 2}',
        indent=None,
        separators=(", ", ": "),
    )


def test_untargeted_incremental_and_private_maps_keep_sha(tmp_path: Path) -> None:
    """没有目标帖的计分记录和私有映射，完整退役后 SHA 不变。"""

    world = _build(tmp_path)
    scored = world.private_map.parent / "incremental-scored-records.json"
    manual_map = world.manual_csv.with_name("manual-review-review-private-map.json")
    _write_json(
        scored,
        [
            {"source_post_id": 3, "embedding_cache_key": _NS + ":" + _SHARED, "备注": "保留"},
            {"source_post_id": 4, "embedding_cache_key": _NS + ":" + _SHARED, "备注": "同样保留"},
        ],
    )
    _write_json(
        world.private_map,
        [{"source_post_id": 3, "task_id": "inf-3", "mark": "桂林"}],
        ensure_ascii=True,
    )
    _write_json(world.keep_map, [{"source_post_id": 3, "task_id": "keep-3", "地名": "阳朔"}])
    _write_json(manual_map, [{"source_post_id": 4, "task_id": "man-4", "说明": "留下"}])
    watched = [scored, world.private_map, world.keep_map, manual_map]
    for path in watched:
        posts = {item["source_post_id"] for item in json.loads(path.read_text(encoding="utf-8"))}
        assert posts.isdisjoint({7, 8})
    before = {path: _sha(path) for path in watched}
    _run(world)
    for path, digest in before.items():
        assert _sha(path) == digest
