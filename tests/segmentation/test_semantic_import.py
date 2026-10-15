"""当前冻结语义发布契约与单表约束测试；不运行或下载真实模型。

当前算法处理合成正文，向量与模型资产身份是测试替身，不是实际Qwen结果。
复用数据库测试的只读配对输入fixture，避免复制来源与运行构造逻辑。
"""

from contextlib import closing
import hashlib
import json
import sqlite3

import pytest

from test_database import prepared
from tourism_ugc_study.segmentation.artifacts import canonical_json
from tourism_ugc_study.segmentation.database_audit import _verify_table_schema, verify_segment_bundle
from tourism_ugc_study.segmentation.import_validation import load_run_manifest, validate_post_record, validate_run_manifest
from tourism_ugc_study.segmentation.importer import import_segmentation_run
from tourism_ugc_study.segmentation.source import ResearchPost, file_sha256
from tourism_ugc_study.segmentation.storage import SCHEMA


def _digest(value):
    """按正式身份规范编码合成清单的摘要。"""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _write(path, value):
    """测试中显式改写独占临时文件，不作用于科研产物。"""
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def test_semantic_single_table_round_trip_and_five_file_verification(prepared):
    """同七字段单表保留语义摘要；复验五文件只验证保存结果，不依赖模型trace。"""
    config, run, output = prepared
    before = file_sha256(config.source_db), file_sha256(config.membership_db)
    result = import_segmentation_run(run, output, config, code_version="synthetic")
    assert result["post_count"] == 3 and result["segment_count"] == 4
    assert {path.name for path in output.iterdir()} == {
        "text-segments.sqlite", "manifest.json", "source-run-manifest.json", "input-config.json", "post-evidence.jsonl"}
    evidence = [json.loads(line) for line in (output / "post-evidence.jsonl").read_text().splitlines()]
    assert evidence[0]["semantic_diagnostics"]["decision"] == "scored_split"
    assert import_segmentation_run(run, output, config, code_version="synthetic")["import_status"] == "already_imported"
    (run / "posts.jsonl").unlink()
    (run / "semantic-trace.jsonl").unlink()
    assert verify_segment_bundle(output)["verification"] == "passed"
    assert import_segmentation_run(run, output, config, code_version="synthetic")["import_status"] == "already_imported"
    assert before == (file_sha256(config.source_db), file_sha256(config.membership_db))


@pytest.mark.parametrize("field,value", [
    ("scope", "development"), ("selected_post_ids", [1, 2, 3]),
    ("sampling", "explicit_development_members"), ("status", "completed_with_failures"),
    ("artifact_kind", "unknown-semantic-run"), ("artifact_kind", "objective-text-segmentation-run"),
    ("rule_version", "semantic-depth-ugc-0.1"), ("rule_version", "semantic-depth-ugc-0.2"),
    ("config_sha256", "0" * 64),
    ("model_identity_sha256", "0" * 64), ("trace_filename", "../private.jsonl"),
    ("trace_sha256", "x" * 64), ("vector_source", None), ("vector_source", {"mode": "cpu"}),
    ("vector_source", {"mode": "cache_only", "identity_source_manifest": "run/manifest.json",
                       "identity_source_manifest_sha256": "a" * 64}),
])
def test_manifest_rejects_development_unknown_and_unbound_identity(prepared, field, value):
    """即使开发清单恰好列出全体成员，也不能冒充完整科研发布。"""
    config, run, output = prepared
    manifest = json.loads((run / "manifest.json").read_text())
    manifest[field] = value
    _write(run / "manifest.json", manifest)
    with pytest.raises(ValueError):
        import_segmentation_run(run, output, config, code_version="synthetic")
    assert not output.exists()


@pytest.mark.parametrize("edit", [
    lambda m: m["counts"].update(post_count=2),
    lambda m: m["counts"].update(failed_count=False),
    lambda m: m["model_identity"].update(device="mps"),
    lambda m: m["model_identity"].update(dimension=2560),
    lambda m: m["model_identity"].update(truncation=0),
    lambda m: m["model_identity"].update(batch_size=2),
    lambda m: m["model_identity"]["assets_sha256"].pop("model.safetensors"),
    lambda m: m["model_identity"]["assets_sha256"].update({"tokenizer.json": "bad"}),
    lambda m: m["config"]["algorithm"].pop("window_tokens"),
    lambda m: m["config"]["algorithm"].update(window_tokens=64),
    lambda m: m["config"]["embedding"].update(batch_size=2),
])
def test_identity_and_counts_still_rejected_after_rehash(prepared, edit):
    """外层摘要重算不能掩盖声明／实际身份冲突、缺资产或不完整配置。"""
    _, run, _ = prepared
    manifest = json.loads((run / "manifest.json").read_text())
    edit(manifest)
    manifest["model_identity_sha256"] = _digest(manifest["model_identity"])
    manifest["config_sha256"] = _digest(manifest["config"])
    with pytest.raises(ValueError):
        validate_run_manifest(manifest)


@pytest.mark.parametrize("missing", [False, True])
def test_original_trace_must_exist_and_match(prepared, missing):
    """原始run导入前后都验trace；五文件复验不会虚构trace仍在场。"""
    config, run, output = prepared
    trace = run / "semantic-trace.jsonl"
    trace.unlink() if missing else trace.write_text("{}\n")
    with pytest.raises(ValueError, match="trace"):
        import_segmentation_run(run, output, config, code_version="synthetic")
    assert not output.exists()


@pytest.mark.parametrize("field,value", [
    ("decision", "single_unit"), ("unit_count", True), ("unit_count", 0),
    ("embedded_unit_count", 0), ("candidate_gap_count", 0),
    ("selected_boundary_count", 0), ("model_identity_sha256", "b" * 64),
    ("vector_content_sha256", None), ("model_call_count", 0),
    ("cache_hit_count", 2), ("reason_codes", ["包含原文不合法"]),
])
def test_semantic_summary_state_machine_is_checked(prepared, field, value):
    """逐帖已覆盖原文仍需语义状态与模型参与证据一致。"""
    _, run, _ = prepared
    manifest, _ = load_run_manifest(run)
    row = json.loads((run / "posts.jsonl").read_text().splitlines()[0])
    row["semantic_diagnostics"][field] = value
    with pytest.raises(ValueError):
        validate_post_record(row, ResearchPost(1, 1, row["original_body"]), manifest)


def test_admin_per_post_segment_limit_is_enforced(prepared):
    """所有跨度都无损仍不能发布Admin无法选材的2,001段帖子。"""
    _, run, _ = prepared
    manifest, _ = load_run_manifest(run)
    row = json.loads((run / "posts.jsonl").read_text().splitlines()[0])
    text = "甲" * 2001
    row.update(original_body=text, source_text=text, source_body_sha256=hashlib.sha256(text.encode()).hexdigest(),
               source_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
               text_segments=[{"seg_id": i + 1, "raw_text": "甲", "start": i, "end": i + 1} for i in range(2001)])
    row["semantic_diagnostics"].update(unit_count=2001, embedded_unit_count=2001, model_call_count=2001,
                                        candidate_gap_count=2000, selected_boundary_count=2000, semantic_boundary_count=2000)
    with pytest.raises(ValueError, match="计数"):
        validate_post_record(row, ResearchPost(1, 1, text), manifest)


def test_bundle_manifest_gate_cannot_be_bypassed(prepared):
    """修改发布包中源清单并更新batch摘要，直接verify也先拒绝开发契约。"""
    config, run, output = prepared
    import_segmentation_run(run, output, config, code_version="synthetic")
    source_path = output / "source-run-manifest.json"
    source = json.loads(source_path.read_text())
    source["scope"] = "development"
    _write(source_path, source)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["batch_id"] = file_sha256(source_path)
    _write(path, manifest)
    with pytest.raises(ValueError, match="开发"):
        verify_segment_bundle(output)


def test_semantic_aggregate_counts_are_recomputed(prepared):
    """逐帖合法且外层计数非负也不足够；实际参与总数必须与清单一致。"""
    config, run, output = prepared
    path = run / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["counts"]["embedded_unit_count"] = 5
    _write(path, manifest)
    with pytest.raises(ValueError, match="汇总"):
        import_segmentation_run(run, output, config, code_version="synthetic")
    assert not output.exists()


@pytest.mark.parametrize("schema", [
    SCHEMA.replace(") STRICT", ")"),
    SCHEMA.replace("CHECK(length(segment_id)=64)", ""),
    SCHEMA.replace("CHECK(source_post_id > 0)", ""),
    SCHEMA.replace("CHECK(seg_no > 0)", ""),
    SCHEMA.replace("CHECK(text != '')", ""),
    SCHEMA.replace("CHECK(start >= 0)", ""),
    SCHEMA.replace("CHECK(end > start)", ""),
    SCHEMA.replace("CHECK(length(batch_id)=64)", ""),
    SCHEMA.replace("UNIQUE(batch_id, source_post_id, seg_no)", "UNIQUE(source_post_id, seg_no)"),
    SCHEMA.replace("segment_id TEXT PRIMARY KEY NOT NULL", "segment_id TEXT NOT NULL"),
    SCHEMA.replace("text TEXT NOT NULL", "text TEXT"),
    SCHEMA.replace("UNIQUE(batch_id, source_post_id, seg_no)", "extra TEXT, UNIQUE(batch_id, source_post_id, seg_no)"),
])
def test_actual_schema_is_rejected_even_with_valid_current_rows(schema):
    """表中当前行合法不代表结构合格；缺CHECK等必须由实际约束试写发现。"""
    with closing(sqlite3.connect(":memory:")) as con:
        con.executescript(schema)
        with pytest.raises(ValueError):
            _verify_table_schema(con)


def test_expected_schema_passes_without_writing_source_database(tmp_path):
    """强约束验收也适用于query_only连接，试写只发生在内存副本。"""
    path = tmp_path / "segments.sqlite"
    with sqlite3.connect(path) as con:
        con.executescript(SCHEMA)
    before = file_sha256(path)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)) as con:
        con.execute("PRAGMA query_only=ON")
        _verify_table_schema(con)
    assert file_sha256(path) == before
