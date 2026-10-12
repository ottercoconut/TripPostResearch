"""冻结连贯块算法的正式身份、原样适配、同表发布和只读重放验收。

只用合成正文和向量，不代表真实81帖独立质量通过。跨层反例验证来源
重叠不重复计数、历史配置不升级、重算外层SHA不能掩盖解释证据篡改。
"""

from dataclasses import asdict, replace
import json
from pathlib import Path
import sqlite3

import pytest

from test_semantic_api import SyntheticProvider
from test_semantic_batch import make_input, read_rows
from test_semantic_dispatch import CharacterCounter, assert_final_consistency, make_units
from test_semantic_structured_integration import _provider
from tourism_ugc_study.segmentation import embedding, semantic_batch, semantic_runtime
from tourism_ugc_study.segmentation.database_audit import verify_segment_bundle
from tourism_ugc_study.segmentation.import_validation import validate_run_manifest
from tourism_ugc_study.segmentation.importer import import_segmentation_run
from tourism_ugc_study.segmentation.semantic_api import segment_body_semantic
from tourism_ugc_study.segmentation.semantic_boundaries import choose_boundaries
from tourism_ugc_study.segmentation.semantic_comparison import compare_semantic_configs
from tourism_ugc_study.segmentation.semantic_config import (
    COHERENT_IMPLEMENTATION_VERSION, COHERENT_RULE_VERSION, CoherentSemanticConfig,
    EmbeddingConfig, SemanticConfig, SemanticSettings, load_semantic_settings,
    parse_algorithm_config, parse_semantic_settings, semantic_rule_identity,
)
from tourism_ugc_study.segmentation.semantic_dispatch import choose_semantic_boundaries
from tourism_ugc_study.segmentation.semantic_evaluation import prepare_development_sample
from tourism_ugc_study.segmentation.semantic_meaning import partition_by_meaning
from tourism_ugc_study.segmentation.source import file_sha256


PARTS = ["📅Day1 河岸漫游", "展馆→书屋", "✅展馆", "展厅介绍了河岸历史和往来船只的故事。",
         "✅书屋", "可以在这里读书看河景，傍晚离开时沿岸散步很舒服。",
         "📅Day2 山野漫游", "林间步道→山顶", "✅林间步道", "松林间鸟鸣清楚，沿途有清晰的路牌。",
         "✅山顶", "登高能看到山脚村落和远处的群山，返程沿着原路走即可。"]


def _settings():
    """使用当前算法默认值与CPU合成provider身份，不加载冻结8B权重。"""
    return SemanticSettings(CoherentSemanticConfig(), EmbeddingConfig(device="cpu"))


def test_frozen_config_has_explicit_identity_and_exact_adopted_parameters():
    """新配置独立登记，不更改旧类字段或从模型大小猜测规则。"""
    path = Path(__file__).resolve().parents[2] / "configs/text-segmentation-semantic-8b-coherent.json"
    settings = load_semantic_settings(path)
    assert type(settings.algorithm) is CoherentSemanticConfig
    assert settings.algorithm == CoherentSemanticConfig()
    assert semantic_rule_identity(settings.algorithm) == (COHERENT_RULE_VERSION, COHERENT_IMPLEMENTATION_VERSION)
    assert parse_semantic_settings(asdict(settings)) == settings
    assert parse_algorithm_config(asdict(settings.algorithm), require_complete=True) == settings.algorithm
    assert type(parse_algorithm_config(asdict(SemanticConfig()))) is SemanticConfig
    assert settings.embedding.model_id == "Qwen/Qwen3-Embedding-8B"
    assert settings.embedding.revision == "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
    assert (settings.embedding.device, settings.embedding.dtype, settings.embedding.batch_size) == ("cuda", "bfloat16", 1)


@pytest.mark.parametrize("raw", [
    {"rule_version": COHERENT_RULE_VERSION, "structure_mode": "context"},
    {"rule_version": COHERENT_RULE_VERSION, "length_strategy": "greedy"},
    {"rule_version": COHERENT_RULE_VERSION, "window_tokens": True},
    {"rule_version": []},
])
def test_frozen_config_rejects_legacy_strategy_switches_and_invalid_types(raw):
    """冻结规则不接受已淘汰策略开关，坏身份不产生隐式默认结果。"""
    with pytest.raises(ValueError):
        parse_algorithm_config(raw)


@pytest.mark.parametrize("overlapping", [False, True])
def test_dispatch_keeps_original_partition_and_deduplicates_structural_semantic_sources(overlapping):
    """同一天保留；换天纯结构点或语义结构重叠点都只形成一个最终切点。"""
    text, units = make_units(PARTS, "\n")
    config = CoherentSemanticConfig(window_tokens=1)
    vectors = [[0, 1] if overlapping and index >= 6 else [1, 0] for index in range(len(units))]
    _, initial = choose_boundaries(text, units, vectors, config, CharacterCounter())
    expected = partition_by_meaning(text, units, initial, config, CharacterCounter())
    actual = choose_semantic_boundaries(text, units, vectors, config, CharacterCounter())
    assert actual.segments == expected.segments
    assert len(actual.segments) == 2
    assert actual.segments[0].raw_text == "\n".join(PARTS[:6])
    assert actual.segments[1].raw_text == "\n".join(PARTS[6:])
    assert actual.decision.semantic_boundaries == (5,)
    assert actual.meaning_trace.structural_boundaries == (5,)
    assert actual.meaning_trace.semantic_boundaries == ((5,) if overlapping else ())
    assert actual.meaning_trace.initial_decision == initial
    assert actual.meaning_trace.gap_decisions == expected.gap_decisions
    assert actual.structure_trace is None
    assert_final_consistency(text, units, actual)


@pytest.mark.parametrize("body,status", [(None, "empty"), ("", "empty"), ("海边真好看。", "single_unit")])
def test_coherent_api_short_paths_skip_forward_and_keep_meaning_evidence(body, status):
    """零或一个单元不为结构规则额外生成向量，兼容摘要仍精确一致。"""
    provider = SyntheticProvider(fail_on_call=1)
    output = segment_body_semantic(body, provider=provider, token_counter=provider, config=CoherentSemanticConfig())
    assert output.result.status == "completed" and output.decision.status == status
    assert output.model_call_count == 0 and output.meaning_trace is not None
    assert output.meaning_trace.gap_decisions == () and output.structure_trace is None
    assert output.summary()["selected_boundary_count"] == 0


def test_coherent_api_failed_projection_has_no_partial_meaning_result():
    """损坏投影不产生部分片段或解释，也不会调用模型。"""
    provider = SyntheticProvider(fail_on_call=1)
    output = segment_body_semantic('{"ops":broken}', provider=provider, token_counter=provider,
                                    config=CoherentSemanticConfig())
    assert output.result.status == "failed" and output.result.text_segments is None
    assert output.meaning_trace is None and output.structure_trace is None
    assert provider.model_call_count == 0


def test_coherent_hard_limit_has_disjoint_final_counts_and_forced_evidence():
    """父块超限时只将额外切点算作长度边界，被迫退让证据仍保留。"""
    parts = ["展馆包含三个部分：", "1、历史厅：" + "甲" * 22 + "。",
             "2、生活厅：" + "乙" * 22 + "。", "3、工艺厅：" + "丙" * 22 + "。"]
    text, units = make_units(parts, "\n")
    config = CoherentSemanticConfig(unit_max_tokens=32, max_segment_tokens=48)
    vectors = [[1, 0]] * len(units)
    actual = choose_semantic_boundaries(text, units, vectors, config, CharacterCounter())
    assert actual.decision.status == "length_split"
    assert actual.decision.semantic_boundaries == () and actual.decision.length_boundaries
    assert actual.meaning_trace.forced_structure_boundaries == actual.decision.length_boundaries
    assert "forced_structure_boundary" in actual.decision.reason_codes
    assert all(len(segment.raw_text) <= 48 for segment in actual.segments)
    assert_final_consistency(text, units, actual)


def test_formal_identity_cannot_silently_follow_a_new_experiment(monkeypatch):
    """将来更换核心实验身份须另行登记，不能自动借已冻结的正式标签。"""
    from tourism_ugc_study.segmentation import semantic_dispatch
    original = semantic_dispatch.partition_by_meaning
    def changed_identity(*args, **kwargs):
        """模拟新实验核心返回另一身份而适配器未同步更新。"""
        return replace(original(*args, **kwargs), experiment_id="experimental-meaning-units-future")
    monkeypatch.setattr(semantic_dispatch, "partition_by_meaning", changed_identity)
    text, units = make_units(["海边真好看。"])
    with pytest.raises(ValueError, match="coherent_partition_identity_mismatch"):
        choose_semantic_boundaries(text, units, None, CoherentSemanticConfig(), CharacterCounter())


def test_coherent_batch_imports_same_table_without_source_writes(tmp_path):
    """实际完整运行可走既有单表发布门；这里只写隔离测试库。"""
    source = make_input(tmp_path, rows=[(1, "\n".join(PARTS), "keep"), (2, None, "keep")])
    before = (file_sha256(source.source_db), file_sha256(source.membership_db))
    settings, run = _settings(), tmp_path / "run"
    manifest = semantic_batch.run_semantic_batch(source, run, settings=settings, code_version="synthetic-coherent",
                                                 provider=_provider(settings))
    validate_run_manifest(manifest)
    rows, traces = read_rows(run / "posts.jsonl"), read_rows(run / "semantic-trace.jsonl")
    assert rows[0]["semantic_diagnostics"]["semantic_boundary_count"] > 0
    assert traces[0]["meaning"]["semantic_boundaries"] == []
    for row, trace in zip(rows, traces, strict=True):
        assert "meaning" in trace and "structure" not in trace
        assert row["semantic_diagnostics"] == trace["summary"]
        assert trace["summary"]["selected_boundary_count"] == max(0, len(row["text_segments"]) - 1)
    output = tmp_path / "bundle"
    imported = import_segmentation_run(run, output, source, code_version="synthetic-coherent")
    assert imported["segment_count"] == manifest["counts"]["segment_count"]
    with sqlite3.connect(f"file:{output / 'text-segments.sqlite'}?mode=ro", uri=True) as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [("text_segments",)]
        assert [row[1] for row in connection.execute("PRAGMA table_info(text_segments)")] == [
            "segment_id", "source_post_id", "seg_no", "text", "start", "end", "batch_id"]
    assert verify_segment_bundle(output)["verification"] == "passed"
    assert before == (file_sha256(source.source_db), file_sha256(source.membership_db))
    manifest["implementation_version"] = "semantic-depth-ugc-0.2-structure-global-1"
    with pytest.raises(ValueError, match="semantic_rule_identity_mismatch"):
        validate_run_manifest(manifest)


def test_coherent_worker_preserves_explicit_config_and_meaning_trace(tmp_path, monkeypatch):
    """worker JSON恢复新类及正式身份，不回落到旧边界实现。"""
    source = make_input(tmp_path, rows=[(1, "\n".join(PARTS), "keep")])
    settings, output = _settings(), tmp_path / "worker"
    monkeypatch.setattr(semantic_batch, "QwenEmbeddingProvider", lambda **kwargs: _provider(settings))
    request = {"input": {key: str(value) for key, value in asdict(source).items()}, "settings": asdict(settings),
               "output": str(output), "cache_dir": str(tmp_path / "cache"), "code_version": "synthetic-worker",
               "post_ids": None, "seed": None, "members_path": None, "code_snapshot": None, "resource_supervision": None}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    assert semantic_runtime._worker(path) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["config"] == asdict(settings) and manifest["config_sha256"] == settings.fingerprint
    assert manifest["rule_version"] == COHERENT_RULE_VERSION
    assert "meaning" in read_rows(output / "semantic-trace.jsonl")[0]


@pytest.fixture
def coherent_replay(tmp_path, monkeypatch):
    """复用旧冻结单位但执行当前规则，后续禁止任何模型加载或补算。"""
    source = make_input(tmp_path, rows=[(1, "海边真好看。", "keep"), (2, "\n".join(PARTS), "keep"),
                                       (3, "山" * 500, "keep"), (4, "路" * 1100, "keep")])
    settings, members_dir, run = _settings(), tmp_path / "members", tmp_path / "run"
    provider = _provider(settings)
    members = prepare_development_sample(source, members_dir,
        settings=replace(settings, algorithm=SemanticConfig()), per_bucket=1, counter=provider)
    semantic_batch.run_semantic_batch(source, run, settings=settings, code_version="synthetic-replay",
        provider=provider, post_ids=tuple(members["post_ids"]), seed=members["seed"], members_path=members_dir / "manifest.json")
    cache = tmp_path / ".semantic-vector-cache"
    before = {str(path): file_sha256(path) for path in cache.rglob("*") if path.is_file()}
    def forbidden(*args, **kwargs):
        """重放只能读缓存，不允许暗中补算。"""
        pytest.fail("连贯块重放不得加载模型或forward")
    monkeypatch.setattr(provider, "encode", forbidden)
    monkeypatch.setattr(embedding.QwenEmbeddingProvider, "__init__", forbidden)
    return source, members_dir / "manifest.json", run, settings, provider, before


@pytest.mark.parametrize("change", [None, "missing", "extra", "block", "source", "reason", "score", "roundoff"])
def test_coherent_replay_verifies_all_meaning_evidence_even_after_rehash(coherent_replay, tmp_path, change):
    """解释字段篡改必须拒绝，只有评分末位舍入可按既定容差接受。"""
    source, members, run, settings, provider, cache_before = coherent_replay
    original = read_rows(run / "semantic-trace.jsonl")
    rows = json.loads(json.dumps(original))
    target = next(row for row in rows if row["meaning"]["structural_boundaries"])
    if change == "missing":
        target.pop("meaning")
    elif change == "extra":
        target["meaning"]["invented"] = True
    elif change == "block":
        target["meaning"]["protected_blocks"][0]["end"] -= 1
    elif change == "source":
        target["meaning"]["structural_boundaries"] = []
    elif change == "reason":
        target["meaning"]["gap_decisions"][0]["reason_codes"] = ["invented"]
    elif change in {"score", "roundoff"}:
        target["meaning"]["initial_decision"]["gaps"][0]["cohesion"] += 1e-6 if change == "score" else 1e-13
    if change is not None:
        path = run / "semantic-trace.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        manifest = json.loads((run / "manifest.json").read_text())
        manifest["trace_sha256"] = file_sha256(path)
        (run / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "comparison"
    def replay():
        """使用唯一当前配置重算全部证据，不改变固定单位或缓存。"""
        return compare_semantic_configs(source, members, run, output, configs=(settings.algorithm,), counter=provider)
    if change in {None, "roundoff"}:
        manifest = replay()
        assert manifest["baseline_reproduction"] == "passed" and manifest["actual_embedding_model_calls"] == 0
        cases = read_rows(output / f"cases-{settings.fingerprint}.jsonl")
        assert [case["meaning"] for case in cases] == [row["meaning"] for row in original]
        with pytest.raises(ValueError, match="开发"):
            import_segmentation_run(run, tmp_path / "rejected-bundle", source, code_version="synthetic")
    else:
        with pytest.raises(ValueError):
            replay()
        assert not output.exists()
    assert cache_before == {path: file_sha256(Path(path)) for path in cache_before}
