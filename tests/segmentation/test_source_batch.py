"""研究来源只读选择、配对拒绝、失败记录及运行包验收测试。"""

from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from tourism_ugc_study.segmentation.batch import run_research_segmentation
from tourism_ugc_study.segmentation.config import ResearchInput, load_research_input
from tourism_ugc_study.segmentation.source import file_sha256, open_research_corpus


def make_input(tmp_path: Path, *, mismatch=False, missing=False) -> ResearchInput:
    """构造四条keep和两条不应切分的合成成员，并绑定真实文件摘要。"""
    source = tmp_path / "source.sqlite"
    members = tmp_path / "members.sqlite"
    manifest_path = tmp_path / "candidate-manifest.json"
    rows = [(1, "甲。乙。", "keep"), (2, '{"ops":broken}', "keep"),
            (3, None, "keep"), (4, " \n", "keep"),
            (5, "排除文本。", "exclude"), (6, "待审核。", "manual_review")]
    with sqlite3.connect(source) as con:
        con.execute("CREATE TABLE web_posts(id INTEGER PRIMARY KEY,platform_key TEXT,title TEXT,content_text TEXT)")
        con.executemany("INSERT INTO web_posts VALUES (?,'xhs',NULL,?)",
                        [(i, body) for i, body, _ in rows if not (missing and i == 1)])
    manifest = {
        "artifact_kind": "full-research-cleaning-candidates",
        "status": "CANDIDATES_READY_AWAITING_HUMAN_REVIEW",
        "database_filename": members.name, "source_snapshot_sha256": file_sha256(source),
        "round_manifest_sha256": "b" * 64, "round_name": "synthetic-round",
        "record_count": 6, "decision_counts": {"keep": 4, "exclude": 1, "manual_review": 1},
    }
    with sqlite3.connect(members) as con:
        con.executescript("""
            CREATE TABLE cleaning_candidates(source_post_id INTEGER,source_version INTEGER,
                platform_key TEXT,title TEXT,body TEXT,decision TEXT);
            CREATE TABLE round_metadata(key TEXT PRIMARY KEY,value_json TEXT);
        """)
        con.executemany("INSERT INTO cleaning_candidates VALUES (?,1,'xhs','',?,?)",
                        [(i, "错误正文" if mismatch and i == 1 else body, decision) for i, body, decision in rows])
        con.executemany("INSERT INTO round_metadata VALUES (?,?)", [
            (key, json.dumps(manifest[key])) for key in
            ("status", "round_name", "source_snapshot_sha256", "round_manifest_sha256")
        ])
    manifest["database_sha256"] = file_sha256(members)
    manifest_path.write_text(json.dumps(manifest))
    return ResearchInput(members, source, manifest_path, file_sha256(manifest_path))


def test_selects_only_keep_and_never_writes_sources(tmp_path):
    """只有keep进入切分，原始数据库和manifest字节全程保持不变。"""
    config = make_input(tmp_path)
    before = [file_sha256(p) for p in (config.membership_db, config.source_db, config.candidate_manifest)]
    with open_research_corpus(config) as corpus:
        assert corpus.identity["record_count"] == 4
        assert [p.source_post_id for p in corpus.posts] == [1, 2, 3, 4]
    output = tmp_path / "run"
    manifest = run_research_segmentation(config, output, code_version="synthetic")
    assert manifest["status"] == "completed_with_failures"
    assert manifest["counts"]["post_count"] == 4
    assert manifest["counts"]["completed_count"] == 3
    assert manifest["counts"]["failed_count"] == 1
    assert manifest["counts"]["empty_count"] == 2
    assert manifest["counts"]["segment_count"] == 2
    rows = [json.loads(line) for line in (output / "posts.jsonl").read_text().splitlines()]
    assert rows[1]["text_segments"] is None and rows[2]["text_segments"] == []
    assert [r["source_post_id"] for r in rows] == [1, 2, 3, 4]
    assert file_sha256(output / "posts.jsonl") == manifest["posts_sha256"]
    assert [file_sha256(p) for p in (config.membership_db, config.source_db, config.candidate_manifest)] == before
    assert not list(tmp_path.glob("*.sqlite-*"))
    with pytest.raises(FileExistsError):
        run_research_segmentation(config, output, code_version="synthetic")


@pytest.mark.parametrize("option", ["mismatch", "missing"])
def test_bad_pairing_rejects_whole_run(tmp_path, option):
    """正文不一致或缺成员都拒绝整轮，不能靠inner join掉过异常成员。"""
    config = make_input(tmp_path, **{option: True})
    output = tmp_path / "bad-run"
    with pytest.raises(ValueError, match="不一致"):
        run_research_segmentation(config, output, code_version="synthetic")
    assert not output.exists() and not list(tmp_path.glob(".bad-run-*"))


def test_manifest_drift_and_active_wal_are_rejected(tmp_path):
    """manifest漂移或活动WAL不得绕过封存验证。"""
    config = make_input(tmp_path)
    with pytest.raises(ValueError, match="摘要"):
        with open_research_corpus(replace(config, candidate_manifest_sha256="0" * 64)):
            pass
    Path(str(config.source_db) + "-wal").write_bytes(b"synthetic WAL")
    with pytest.raises(ValueError, match="活动日志"):
        with open_research_corpus(config):
            pass


def test_change_during_read_is_detected(tmp_path):
    """运行末尾再次校验来源，阻止产物绑定过期摘要。"""
    config = make_input(tmp_path)
    with pytest.raises(ValueError, match="运行中"):
        with open_research_corpus(config) as corpus:
            list(corpus.posts)
            with config.candidate_manifest.open("a") as stream:
                stream.write(" ")


def test_config_and_cli_are_location_independent(tmp_path):
    """配置路径相对配置文件，CLI在别的工作目录仍正确绑定配对输入。"""
    config = make_input(tmp_path)
    settings = tmp_path / "config.json"
    settings.write_text(json.dumps({"membership_db": config.membership_db.name,
        "source_db": config.source_db.name, "candidate_manifest": config.candidate_manifest.name,
        "candidate_manifest_sha256": config.candidate_manifest_sha256}))
    assert load_research_input(settings) == config
    script = Path(__file__).resolve().parents[2] / "scripts/text_segment_research.py"
    completed = subprocess.run([sys.executable, str(script), "--config", str(settings),
        "--output", str(tmp_path / "cli-run")], cwd=tmp_path, text=True, capture_output=True)
    assert completed.returncode == 2, completed.stderr
    assert json.loads(completed.stdout)["counts"]["post_count"] == 4
