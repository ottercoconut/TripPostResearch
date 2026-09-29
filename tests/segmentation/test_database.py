"""单表入库、稳定身份、来源校验、随机抽查及失败回滚的合成集成测试。"""

from contextlib import closing
import json
from pathlib import Path
import sqlite3

import pytest

from tourism_ugc_study.segmentation.batch import run_research_segmentation
from tourism_ugc_study.segmentation.config import ResearchInput
from tourism_ugc_study.segmentation.database_audit import sample_segment_bundle, verify_segment_bundle
from tourism_ugc_study.segmentation.importer import import_segmentation_run
from tourism_ugc_study.segmentation.source import file_sha256
from tourism_ugc_study.segmentation.storage import get_post_segments, open_segment_database


@pytest.fixture
def prepared(tmp_path):
    """产生含普通文、Delta、空白和排除项的配对库及真实切分运行包。"""
    source, members = tmp_path / "source.sqlite", tmp_path / "members.sqlite"
    rows = [(1, "甲。乙！", "keep"),
            (2, '{"ops":[{"insert":"丙。丁。"}]}', "keep"),
            (3, " \n", "keep"), (4, "不保留。", "exclude")]
    with sqlite3.connect(source) as con:
        con.execute("CREATE TABLE web_posts(id INTEGER PRIMARY KEY,platform_key TEXT,title TEXT,content_text TEXT)")
        con.executemany("INSERT INTO web_posts VALUES (?,'xhs',NULL,?)", [(i, body) for i, body, _ in rows])
    manifest = {
        "artifact_kind": "full-research-cleaning-candidates", "status": "CANDIDATES_READY_AWAITING_HUMAN_REVIEW",
        "database_filename": members.name, "source_snapshot_sha256": file_sha256(source),
        "round_manifest_sha256": "b" * 64, "round_name": "synthetic-import",
        "record_count": 4, "decision_counts": {"keep": 3, "exclude": 1, "manual_review": 0},
    }
    with sqlite3.connect(members) as con:
        con.executescript("""
            CREATE TABLE cleaning_candidates(source_post_id INTEGER,source_version INTEGER,
                platform_key TEXT,title TEXT,body TEXT,decision TEXT);
            CREATE TABLE round_metadata(key TEXT PRIMARY KEY,value_json TEXT);
        """)
        con.executemany("INSERT INTO cleaning_candidates VALUES (?,1,'xhs','',?,?)", rows)
        con.executemany("INSERT INTO round_metadata VALUES (?,?)", [(key, json.dumps(manifest[key])) for key in
            ("status", "round_name", "source_snapshot_sha256", "round_manifest_sha256")])
    manifest["database_sha256"] = file_sha256(members)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(manifest))
    config = ResearchInput(members, source, path, file_sha256(path))
    run = tmp_path / "run"
    run_research_segmentation(config, run, code_version="synthetic")
    return config, run, tmp_path / "database"


def test_single_table_round_trip_and_idempotency(prepared):
    """完整重建JSONL哈希、只存在一张表，重复入库及删JSONL后复核均成立。"""
    config, run, output = prepared
    before = (file_sha256(config.source_db), file_sha256(config.membership_db))
    result = import_segmentation_run(run, output, config, code_version="synthetic")
    assert result["post_count"] == 3 and result["segment_count"] == 4
    assert result["empty_post_ids"] == [3]
    with closing(open_segment_database(output / "text-segments.sqlite")) as con:
        assert [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")] == ["text_segments"]
        parts = get_post_segments(con, result["batch_id"], 1)
        assert [p["text"] for p in parts] == ["甲。", "乙！"]
        assert len({p["segment_id"] for p in parts}) == 2
        with pytest.raises(sqlite3.OperationalError):
            con.execute("DELETE FROM text_segments")
    assert import_segmentation_run(run, output, config, code_version="synthetic")["import_status"] == "already_imported"
    (run / "posts.jsonl").unlink()
    verified = verify_segment_bundle(output)
    assert verified["reconstructed_jsonl_sha256"] == result["source_records_sha256"]
    assert import_segmentation_run(run, output, config, code_version="synthetic")["import_status"] == "already_imported"
    assert before == (file_sha256(config.source_db), file_sha256(config.membership_db))


def test_modified_record_rejected_even_with_matching_file_hash(prepared):
    """上层摘要被一并修改，也不能将错误原文关系写入数据库。"""
    config, run, output = prepared
    records = run / "posts.jsonl"
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    rows[0]["text_segments"][0]["raw_text"] = "错。"
    records.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["posts_sha256"] = file_sha256(records)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        import_segmentation_run(run, output, config, code_version="synthetic")
    assert not output.exists() and not list(output.parent.glob(".database-*"))


def test_database_tamper_detected_beyond_digest(prepared):
    """篡改表行且重写外层摘要，仍被原文覆盖和原始JSONL重建验收发现。"""
    config, run, output = prepared
    import_segmentation_run(run, output, config, code_version="synthetic")
    db = output / "text-segments.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("UPDATE text_segments SET text='错。' WHERE source_post_id=1 AND seg_no=1")
    path = output / "manifest.json"
    manifest = json.loads(path.read_text()); manifest["database_sha256"] = file_sha256(db)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify_segment_bundle(output)


def test_other_batch_cannot_overwrite(prepared):
    """相同目标目录不能被另一运行覆盖，即使它引用相同正文。"""
    config, run, output = prepared
    result = import_segmentation_run(run, output, config, code_version="synthetic")
    path = run / "manifest.json"
    manifest = json.loads(path.read_text()); manifest["code_version"] = "different-run"
    path.write_text(json.dumps(manifest))
    with pytest.raises(FileExistsError):
        import_segmentation_run(run, output, config, code_version="synthetic")
    assert file_sha256(output / "text-segments.sqlite") == result["database_sha256"]


def test_random_sample_is_reproducible_and_review_is_not_fabricated(prepared):
    """同种子抽查同成员，无重复，程序不自动宣称人工审阅通过。"""
    config, run, output = prepared
    import_segmentation_run(run, output, config, code_version="synthetic")
    first = sample_segment_bundle(output, output.parent / "sample-a", count=2, seed=20260928)
    second = sample_segment_bundle(output, output.parent / "sample-b", count=2, seed=20260928)
    assert first["source_post_ids"] == second["source_post_ids"]
    assert len(set(first["source_post_ids"])) == 2
    assert first["review_status"] == "pending"
    with pytest.raises(ValueError):
        sample_segment_bundle(output, output.parent / "too-many", count=4, seed=1)


def test_unique_constraints_reject_duplicate_segments(prepared):
    """主键之外还禁止同批次同帖出现两个相同序号。"""
    config, run, output = prepared
    result = import_segmentation_run(run, output, config, code_version="synthetic")
    with sqlite3.connect(output / "text-segments.sqlite") as con:
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("INSERT INTO text_segments VALUES (?,?,?,?,?,?,?)",
                        ("a" * 64, 1, 1, "甲。", 0, 2, result["batch_id"]))
