"""次级库排除的合成库测试。不读取 data/、results/ 或真实采集库。"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import subprocess
from pathlib import Path

import pytest

from tourism_ugc_study.cleaning.post_exclusion import derive_post_exclusion

SECRET_NICK = "LEAKED_NICK_TOKEN"
SECRET_UID = "LEAKED_UID_TOKEN"
PYTHON = Path("/Users/kawauso/Documents/Projects/TripPostResearch/.venv/bin/python")
DROPPED_BODY = "DROPBODY-7c1e9a4b-UNRESOLVED"
KEPT_BODY = "KEEPBODY-should-remain"


def _write_exclusion(path: Path, *, absent: list[dict[str, object]], still_masked: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"revision_id": "rev-unresolved", "absent": absent, "still_masked": still_masked}, ensure_ascii=False),
        encoding="utf-8",
    )


def _candidate_database(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE cleaning_candidates(
                source_post_id INTEGER PRIMARY KEY,
                decision TEXT NOT NULL,
                exact_cluster_id TEXT NOT NULL,
                exact_representative_id INTEGER NOT NULL,
                body TEXT NOT NULL,
                linked_post_id INTEGER
            );
            CREATE TABLE text_segments(
                segment_id TEXT PRIMARY KEY,
                source_post_id INTEGER NOT NULL,
                text TEXT NOT NULL
            );
            CREATE TABLE round_metadata(key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
            """
        )
        conn.executemany(
            "INSERT INTO cleaning_candidates VALUES(?, ?, ?, ?, ?, ?)",
            [
                (1, "keep", "cluster-kept", 1, KEPT_BODY, None),
                (2, "exclude", "cluster-drop", 2, DROPPED_BODY, None),
                (3, "keep", "cluster-dangling", 2, "kept-but-points-at-2", 2),
                (4, "manual_review", "cluster-other", 4, "other-dropped-body", None),
            ],
        )
        conn.executemany(
            "INSERT INTO text_segments VALUES(?, ?, ?)",
            [
                ("seg-kept", 1, "fragment-kept"),
                ("seg-drop", 2, DROPPED_BODY),
            ],
        )
        conn.execute(
            "INSERT INTO round_metadata VALUES('candidate_count', ?)",
            (json.dumps({"count": 4}, sort_keys=True),),
        )


def _exclude(directory: Path, source: Path, exclusions: list[Path], table_columns: list[str]) -> dict[str, object]:
    return derive_post_exclusion(
        source,
        exclusions,
        table_columns,
        directory / "excluded.sqlite",
        directory / "manifest.json",
        revision_id="rev-exclude-1",
        reason="synthetic-post-exclusion",
        code_commit="test-commit",
    )


def test_excludes_matching_rows_and_counts_dangling_references(tmp_path: Path) -> None:
    """只删除指定表列的命中行；悬挂引用被计数，decision 和元数据表保持原值。"""

    source = tmp_path / "candidates.sqlite"
    _candidate_database(source)
    os.chmod(source, 0o400)
    source_bytes = source.read_bytes()
    source_mode = source.stat().st_mode
    source_flags = getattr(source.stat(), "st_flags", 0)
    first = tmp_path / "absent.json"
    second = tmp_path / "masked.json"
    _write_exclusion(
        first,
        absent=[{
            "platform_key": "douyin",
            "platform_post_id": SECRET_UID,
            "source_post_id": 2,
            "nickname": SECRET_NICK,
        }],
        still_masked=[],
    )
    _write_exclusion(
        second,
        absent=[],
        still_masked=[{"platform_key": "weibo", "platform_post_id": "post-4", "source_post_id": 4}],
    )
    registry = Path(__file__).resolve().parents[2] / "governance" / "research-data-freezes.json"
    registry_before = registry.read_bytes()

    result = _exclude(
        tmp_path,
        source,
        [first, second],
        ["cleaning_candidates:source_post_id", "text_segments:source_post_id"],
    )
    output = tmp_path / "excluded.sqlite"
    manifest_text = (tmp_path / "manifest.json").read_text()

    assert source.read_bytes() == source_bytes
    assert source.stat().st_mode == source_mode
    assert getattr(source.stat(), "st_flags", 0) == source_flags
    assert registry.read_bytes() == registry_before
    assert not list(tmp_path.glob("current-*.json"))
    assert (output.stat().st_mode & 0o777) == 0o400
    assert getattr(output.stat(), "st_flags", 0) & stat.UF_IMMUTABLE == 0
    assert result["status"] == "DERIVED_NOT_FROZEN"
    assert result["excluded_source_post_ids"] == [2, 4]
    assert result["dropped_rows"] == [
        {"table": "cleaning_candidates", "column": "source_post_id", "count": 2},
        {"table": "text_segments", "column": "source_post_id", "count": 1},
    ]
    assert result["dangling_references"] == [
        {
            "table": "cleaning_candidates",
            "column": "exact_representative_id",
            "count": 1,
            "referenced_source_post_ids": [2],
        },
        {
            "table": "cleaning_candidates",
            "column": "linked_post_id",
            "count": 1,
            "referenced_source_post_ids": [2],
        },
    ]
    assert "删除前" in str(result["unchanged_tables_note"])
    assert SECRET_NICK not in manifest_text
    assert SECRET_UID not in manifest_text
    blob = output.read_bytes()
    assert DROPPED_BODY.encode() not in blob
    assert KEPT_BODY.encode() in blob

    with sqlite3.connect(output.as_uri() + "?mode=ro", uri=True) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        rows = conn.execute(
            "SELECT source_post_id, decision, exact_cluster_id, exact_representative_id, body, linked_post_id "
            "FROM cleaning_candidates ORDER BY source_post_id"
        ).fetchall()
        assert rows == [
            (1, "keep", "cluster-kept", 1, KEPT_BODY, None),
            (3, "keep", "cluster-dangling", 2, "kept-but-points-at-2", 2),
        ]
        assert conn.execute("SELECT segment_id, source_post_id, text FROM text_segments").fetchall() == [
            ("seg-kept", 1, "fragment-kept"),
        ]
        assert conn.execute("SELECT key, value_json FROM round_metadata").fetchall() == [
            ("candidate_count", json.dumps({"count": 4}, sort_keys=True)),
        ]


def test_existing_output_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "candidates.sqlite"
    _candidate_database(source)
    os.chmod(source, 0o400)
    source_bytes = source.read_bytes()
    exclusion = tmp_path / "absent.json"
    _write_exclusion(
        exclusion,
        absent=[{"source_post_id": 2}],
        still_masked=[],
    )
    output = tmp_path / "excluded.sqlite"
    output.write_bytes(b"keep-existing-output")
    with pytest.raises(FileExistsError, match="post_exclusion_output_exists"):
        _exclude(tmp_path, source, [exclusion], ["cleaning_candidates:source_post_id"])
    assert output.read_bytes() == b"keep-existing-output"
    assert source.read_bytes() == source_bytes
    assert not Path(str(output) + ".tmp").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_missing_table_leaves_no_output(tmp_path: Path) -> None:
    source = tmp_path / "candidates.sqlite"
    _candidate_database(source)
    source_bytes = source.read_bytes()
    exclusion = tmp_path / "absent.json"
    _write_exclusion(exclusion, absent=[{"source_post_id": 2}], still_masked=[])
    output = tmp_path / "excluded.sqlite"
    with pytest.raises(ValueError, match="post_exclusion_table_missing"):
        _exclude(tmp_path, source, [exclusion], ["missing_table:source_post_id"])
    assert source.read_bytes() == source_bytes
    assert not output.exists()
    assert not Path(str(output) + ".tmp").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_script_excludes_posts_without_leaking_identifiers(tmp_path: Path) -> None:
    source = tmp_path / "segments.sqlite"
    with sqlite3.connect(source) as conn:
        conn.execute("CREATE TABLE text_segments(source_post_id INTEGER, text TEXT)")
        conn.executemany(
            "INSERT INTO text_segments VALUES(?, ?)",
            [(1, KEPT_BODY), (2, DROPPED_BODY)],
        )
    exclusion = tmp_path / "unresolved-posts.json"
    _write_exclusion(
        exclusion,
        absent=[{"source_post_id": 2, "platform_post_id": SECRET_UID, "nickname": SECRET_NICK}],
        still_masked=[],
    )
    repo = Path(__file__).resolve().parents[2]
    output = tmp_path / "excluded.sqlite"
    manifest = tmp_path / "manifest.json"
    completed = subprocess.run(
        [
            str(PYTHON),
            str(repo / "scripts/cleaning_derive_post_exclusion.py"),
            "--source", str(source),
            "--exclusions", str(exclusion),
            "--table-column", "text_segments:source_post_id",
            "--output", str(output),
            "--manifest", str(manifest),
            "--revision-id", "rev-cli-exclude",
            "--reason", "synthetic-cli-exclusion",
        ],
        check=True,
        cwd=repo,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    printed = json.loads(completed.stdout)
    assert printed["code_commit"] == head
    assert printed["dropped_rows"] == [{"table": "text_segments", "column": "source_post_id", "count": 1}]
    assert SECRET_NICK not in completed.stdout
    assert SECRET_UID not in manifest.read_text()
    assert DROPPED_BODY.encode() not in output.read_bytes()
