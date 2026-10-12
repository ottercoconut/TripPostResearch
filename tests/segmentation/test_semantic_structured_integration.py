"""新结构规则跨批处理、缓存、worker和同表发布的合成验收。

全部来源与向量是隔离的测试数据，检验身份、最终边界及持久化，不代替
真实8B语义评审。旧冻结样本可供新规则复用，开发包仍不能导入正式单表。
"""

from dataclasses import asdict, replace
import json
import sqlite3

import pytest

from test_semantic_acceptance import replay_witness
from test_semantic_api import SyntheticProvider
from test_semantic_batch import make_input, read_rows
from tourism_ugc_study.segmentation import embedding, semantic_batch, semantic_runtime
from tourism_ugc_study.segmentation.database_audit import verify_segment_bundle
from tourism_ugc_study.segmentation.import_validation import validate_run_manifest
from tourism_ugc_study.segmentation.importer import import_segmentation_run
from tourism_ugc_study.segmentation.semantic_comparison import compare_semantic_configs
from tourism_ugc_study.segmentation.semantic_config import (
    EmbeddingConfig, SemanticSettings, StructuredSemanticConfig, semantic_rule_identity,
)
from tourism_ugc_study.segmentation.source import file_sha256


def _settings():
    """使用显式新规则和CPU合成身份；不依赖目标机器是否有GPU。"""
    return SemanticSettings(StructuredSemanticConfig(max_segment_tokens=64), EmbeddingConfig(device="cpu"))


def _provider(settings):
    """补齐发布门所需的合成资产声明，test_fixture身份不伪装真实权重。"""
    provider = SyntheticProvider(settings.embedding)
    provider.identity["assets_sha256"] = {name: "a" * 64 for name in (
        "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors")}
    provider.identity["dependencies"] = {name: "synthetic" for name in (
        "torch", "transformers", "tokenizers", "huggingface-hub", "numpy")}
    return provider


def test_structured_batch_actual_decision_imports_into_same_seven_field_table(tmp_path):
    """完整运行实际产生新切点；trace、摘要和单表片段数一致且来源不变。"""
    source = make_input(tmp_path, rows=[
        (1, "景点介绍\n" + "甲" * 30 + "。\n" + "乙" * 40 + "。", "keep"),
        (2, "海边真好看。", "keep"), (3, None, "keep"),
    ])
    configured = _settings()
    before = file_sha256(source.source_db), file_sha256(source.membership_db)
    run = tmp_path / "structured-run"
    manifest = semantic_batch.run_semantic_batch(source, run, settings=configured,
        code_version="synthetic-structured", provider=_provider(configured))
    assert (manifest["rule_version"], manifest["implementation_version"]) == semantic_rule_identity(configured.algorithm)
    assert manifest["config_sha256"] == configured.fingerprint
    validate_run_manifest(manifest)
    for row, trace in zip(read_rows(run / "posts.jsonl"), read_rows(run / "semantic-trace.jsonl"), strict=True):
        segments = row["text_segments"]
        cuts = sorted(trace["decision"]["semantic_boundaries"] + trace["decision"]["length_boundaries"])
        assert [trace["units"][i]["end"] for i in cuts] == [s["end"] for s in segments[:-1]]
        assert [g["index"] for g in trace["decision"]["gaps"] if g["selected"]] == cuts
        assert trace["summary"] == row["semantic_diagnostics"]
        assert trace["summary"]["selected_boundary_count"] == max(0, len(segments) - 1)
    output = tmp_path / "published"
    imported = import_segmentation_run(run, output, source, code_version="synthetic-structured")
    assert imported["segment_count"] == manifest["counts"]["segment_count"]
    with sqlite3.connect(f"file:{output / 'text-segments.sqlite'}?mode=ro", uri=True) as connection:
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        assert tables == [("text_segments",)]
        assert [r[1] for r in connection.execute("PRAGMA table_info(text_segments)")] == [
            "segment_id", "source_post_id", "seg_no", "text", "start", "end", "batch_id"]
    assert import_segmentation_run(run, output, source, code_version="synthetic-structured")["import_status"] == "already_imported"
    assert verify_segment_bundle(output)["verification"] == "passed"
    assert before == (file_sha256(source.source_db), file_sha256(source.membership_db))


def test_structured_worker_restores_same_config_and_rule(tmp_path, monkeypatch):
    """真实worker入口JSON往返新配置，不能因恢复旧类而丢失结构规则。"""
    source = make_input(tmp_path, rows=[(1, "海边真好看。", "keep")])
    configured = _settings()
    monkeypatch.setattr(semantic_batch, "QwenEmbeddingProvider", lambda **kwargs: _provider(configured))
    output, request_path = tmp_path / "worker-run", tmp_path / "request.json"
    request = {"input": {k: str(v) for k, v in asdict(source).items()}, "settings": asdict(configured),
        "output": str(output), "cache_dir": str(tmp_path / "cache"), "code_version": "synthetic-worker",
        "post_ids": None, "seed": None, "members_path": None, "code_snapshot": None, "resource_supervision": None}
    request_path.write_text(json.dumps(request))
    assert semantic_runtime._worker(request_path) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["config"] == asdict(configured)
    assert manifest["config_sha256"] == configured.fingerprint
    assert (manifest["rule_version"], manifest["implementation_version"]) == semantic_rule_identity(configured.algorithm)


def test_structured_run_reuses_old_frozen_units_and_replays_without_forward(replay_witness, tmp_path, monkeypatch):
    """新规则与旧单位兼容；重放必须读取同向量，不能靠补算伪造可重复性。"""
    source, members, _, cache, legacy, _ = replay_witness
    configured = replace(legacy, algorithm=StructuredSemanticConfig())
    provider = SyntheticProvider(configured.embedding)
    frozen = json.loads(members.read_text())
    run = tmp_path / "structured-development"
    semantic_batch.run_semantic_batch(source, run, settings=configured, code_version="synthetic-new",
        provider=provider, post_ids=tuple(frozen["post_ids"]), seed=frozen["seed"], members_path=members)
    baseline_calls = provider.model_call_count
    def forbidden(*args, **kwargs):
        """任何模型补算都会使只读重放验收失败。"""
        pytest.fail("结构候选重放不得forward")
    monkeypatch.setattr(provider, "encode", forbidden)
    monkeypatch.setattr(embedding.QwenEmbeddingProvider, "__init__", forbidden)
    manifest = compare_semantic_configs(source, members, run, tmp_path / "replay",
        configs=(configured.algorithm,), counter=provider)
    assert manifest["baseline_reproduction"] == "passed"
    assert manifest["actual_embedding_model_calls"] == 0 and provider.model_call_count == baseline_calls
    with pytest.raises(ValueError, match="开发"):
        import_segmentation_run(run, tmp_path / "forbidden-bundle", source, code_version="synthetic-new")


@pytest.mark.parametrize("field", ["rule_version", "implementation_version"])
def test_structured_manifest_cannot_borrow_legacy_identity(tmp_path, field):
    """仅改清单身份而保留新配置仍必须拒绝，不能绕过版本配对核验。"""
    source = make_input(tmp_path, rows=[(1, "海边真好看。", "keep")])
    configured = _settings()
    manifest = semantic_batch.run_semantic_batch(source, tmp_path / "run", settings=configured,
        code_version="synthetic", provider=_provider(configured))
    manifest[field] = "semantic-depth-ugc-0.1" if field == "rule_version" else "semantic-depth-ugc-0.1-original-1"
    with pytest.raises(ValueError):
        validate_run_manifest(manifest)
