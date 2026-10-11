"""由统筹方独立编写的语义开发验收反例，不以实现内部状态作为正确性依据。

测试只使用隔离来源和合成计数器。外层摘要一致不代表可比性成立：原文、
候选跨度以及独立抽查人口仍须分别核实；边界参数变化则不应误伤可复用单元。
"""

from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from test_database import prepared
from test_semantic_api import SyntheticProvider
from test_semantic_batch import make_input
from test_semantic_evaluation import CharacterCounter, make_source, settings
from tourism_ugc_study.segmentation import embedding
from tourism_ugc_study.segmentation.database_audit import sample_segment_bundle
from tourism_ugc_study.segmentation.importer import import_segmentation_run
from tourism_ugc_study.segmentation.semantic_batch import run_semantic_batch
from tourism_ugc_study.segmentation.semantic_comparison import compare_semantic_configs
from tourism_ugc_study.segmentation.semantic_config import SemanticSettings
from tourism_ugc_study.segmentation.semantic_evaluation import (
    prepare_development_sample,
    validate_development_units,
)
from tourism_ugc_study.segmentation.source import file_sha256


@pytest.fixture
def frozen_development(tmp_path):
    """保留四个真实长度层与压力成员角色，但不加载模型或访问正式语料。"""
    source = make_source(tmp_path)
    output = tmp_path / "frozen-development"
    manifest = prepare_development_sample(
        source, output, settings=settings(), per_bucket=1, counter=CharacterCounter(),
    )
    return source, output, manifest


def _rewrite_cases(output: Path, manifest: dict, rows: list[dict]) -> None:
    """同步外层摘要，确保测试击中内容契约而不是简单的损坏文件检测。"""
    cases = output / "cases.jsonl"
    cases.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest["cases_sha256"] = file_sha256(cases)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))


def test_rehashed_unit_merge_cannot_silently_change_comparison_basis(frozen_development):
    """即使单元仍完整覆盖正文，冻结后合并候选也会改变比较机会，必须拒绝。"""
    source, output, manifest = frozen_development
    rows = [json.loads(line) for line in (output / "cases.jsonl").read_text().splitlines()]
    case = next(row for row in rows if len(row["units"]) >= 2)
    first, second = case["units"][:2]
    case["units"][:2] = [{"start": first["start"], "end": second["end"],
                          "token_count": second["end"] - first["start"]}]
    _rewrite_cases(output, manifest, rows)
    with pytest.raises(ValueError):
        validate_development_units(output / "manifest.json", source,
                                   settings=settings(), counter=CharacterCounter())


@pytest.mark.parametrize("field", ["source_text", "original_body", "source_version"])
def test_rehashed_cases_cannot_change_source_evidence(frozen_development, field):
    """只保留成员ID和有效JSON摘要，仍不足以声称样本来自所绑定的源帖子。"""
    source, output, manifest = frozen_development
    rows = [json.loads(line) for line in (output / "cases.jsonl").read_text().splitlines()]
    rows[0][field] = rows[0][field] + 1 if field == "source_version" else "另一篇完全不同的正文。"
    _rewrite_cases(output, manifest, rows)
    with pytest.raises(ValueError):
        validate_development_units(output / "manifest.json", source,
                                   settings=settings(), counter=CharacterCounter())


def test_boundary_only_tuning_preserves_units_but_unit_budget_change_does_not(frozen_development):
    """允许重放真正只改边界的配置；不能以同成员掩盖token单元预算变化。"""
    source, output, _ = frozen_development
    boundary_settings = replace(settings(), algorithm=replace(
        settings().algorithm, window_tokens=32, absolute_cohesion_threshold=0.65,
    ))
    original = validate_development_units(output / "manifest.json", source,
                                          settings=settings(), counter=CharacterCounter())
    tuned = validate_development_units(output / "manifest.json", source,
                                       settings=boundary_settings, counter=CharacterCounter())
    assert original.cases == tuned.cases
    changed_units = replace(settings(), algorithm=replace(settings().algorithm, unit_max_tokens=32))
    with pytest.raises(ValueError):
        validate_development_units(output / "manifest.json", source,
                                   settings=changed_units, counter=CharacterCounter())


class ThresholdWitnessProvider(SyntheticProvider):
    """两句余弦固定为0.60，能区分0.45与0.65阈值，不预设真实主题答案。"""

    def encode(self, texts):
        """沿用调用证据，仅把乙向量改成可手算的(0.6, 0.8)。"""
        vectors = super().encode(texts)
        for index, text in enumerate(texts):
            if text.startswith("乙"):
                vectors[index, :] = 0
                vectors[index, :2] = (0.6, 0.8)
        return vectors


@pytest.fixture
def replay_witness(tmp_path):
    """四层各一帖；首帖检验阈值敏感性，其余覆盖硬长度切点。"""
    source = make_input(tmp_path, rows=[
        (1, "甲" * 15 + "。" + "乙" * 15 + "。", "keep"),
        (2, "海" * 150, "keep"), (3, "山" * 500, "keep"), (4, "路" * 1100, "keep"),
    ])
    configured = replace(settings(), embedding=replace(settings().embedding, device="cpu"))
    provider = ThresholdWitnessProvider(configured.embedding)
    members, run, cache = tmp_path / "members", tmp_path / "run", tmp_path / "vectors"
    sampled = prepare_development_sample(source, members, settings=configured,
                                         per_bucket=1, counter=provider)
    run_semantic_batch(source, run, settings=configured, code_version="independent-test",
                       cache_dir=cache, provider=provider, post_ids=tuple(sampled["post_ids"]),
                       seed=sampled["seed"])
    return source, members / "manifest.json", run, cache, configured, provider


def _forbid_model(*args, **kwargs):
    """若比较为了修补证据而请求模型，直接使验收失败。"""
    pytest.fail("缓存比较不能加载模型或执行任何embedding forward")


def test_cached_threshold_change_changes_boundaries_without_changing_evidence(
    replay_witness, tmp_path, monkeypatch,
):
    """反证“参数有变但结果沿用旧文件”和“重放实为重推理”两类伪比较。"""
    source, members, run, cache, configured, provider = replay_witness
    baseline_calls = provider.model_call_count
    monkeypatch.setattr(ThresholdWitnessProvider, "encode", _forbid_model)
    monkeypatch.setattr(embedding.QwenEmbeddingProvider, "__init__", _forbid_model)
    cache_before = {str(path.relative_to(cache)): file_sha256(path) for path in cache.rglob("*") if path.is_file()}
    low = replace(configured.algorithm, absolute_cohesion_threshold=0.45)
    high = replace(configured.algorithm, absolute_cohesion_threshold=0.65)
    output = tmp_path / "comparison"
    manifest = compare_semantic_configs(source, members, run, output,
                                        configs=(low, high), counter=provider)
    groups = []
    for algorithm in (low, high):
        config_id = SemanticSettings(algorithm, configured.embedding).fingerprint
        rows = [json.loads(line) for line in (output / f"cases-{config_id}.jsonl").read_text().splitlines()]
        groups.append(rows)
        review = json.loads((output / f"review-{config_id}.json").read_text())
        assert all(row["assistant_judgment"] == "pending" for row in review["samples"])
        assert all(row["model_call_count"] == 0 for row in rows)
    assert groups[0][0]["new_boundaries"] == []
    assert groups[1][0]["new_boundaries"] == [16]
    assert [row["source_text_sha256"] for row in groups[0]] == [row["source_text_sha256"] for row in groups[1]]
    assert [row["units"] for row in groups[0]] == [row["units"] for row in groups[1]]
    assert provider.model_call_count == baseline_calls
    assert manifest["actual_embedding_model_calls"] == 0 and manifest["baseline_reproduction"] == "passed"
    assert cache_before == {str(path.relative_to(cache)): file_sha256(path) for path in cache.rglob("*") if path.is_file()}


@pytest.mark.parametrize("damage", ["missing", "rehash_replacement"])
def test_replay_refuses_missing_or_rehashed_replacement_vector(
    replay_witness, tmp_path, monkeypatch, damage,
):
    """缓存条目自身有效也不代表是本次run使用的向量，不能补算后伪称重放。"""
    source, members, run, cache, _, provider = replay_witness
    monkeypatch.setattr(ThresholdWitnessProvider, "encode", _forbid_model)
    monkeypatch.setattr(embedding.QwenEmbeddingProvider, "__init__", _forbid_model)
    vector_path = next(cache.rglob("vector.npy"))
    if damage == "missing":
        shutil.rmtree(vector_path.parent)
    else:
        vectors = np.load(vector_path, allow_pickle=False)
        vectors[:] = 0
        vectors[:, 2] = 1
        np.save(vector_path, vectors, allow_pickle=False)
        metadata = vector_path.parent / "manifest.json"
        value = json.loads(metadata.read_text())
        value["vector_sha256"] = file_sha256(vector_path)
        metadata.write_text(json.dumps(value))
    output = tmp_path / "invalid-comparison"
    with pytest.raises(ValueError, match="cache_entry_missing|comparison_run_vector_digest_mismatch"):
        compare_semantic_configs(source, members, run, output, counter=provider)
    assert not output.exists()


def test_independent_sampling_cannot_fill_shortfall_with_development_posts(prepared, tmp_path):
    """人口三帖排除一帖后只能抽两帖；不能缩小count或把开发帖放回凑数。"""
    source, run, bundle = prepared
    manifest = import_segmentation_run(run, bundle, source, code_version="independent-test")
    excluded = tmp_path / "development-exclusions.json"
    excluded.write_text(json.dumps({"artifact_kind": "semantic-development-exclusions",
                                    "input": manifest["input"], "post_ids": [1]}))
    with pytest.raises(ValueError, match="人口不足"):
        sample_segment_bundle(bundle, tmp_path / "shortfall", count=3, seed=20261012,
                              exclusion_manifest=excluded, independent=True)
    assert not (tmp_path / "shortfall").exists()
    sampled = sample_segment_bundle(bundle, tmp_path / "independent", count=2, seed=20261012,
                                    exclusion_manifest=excluded, independent=True)
    assert set(sampled["source_post_ids"]) == {2, 3}
    assert sampled["excluded_post_ids"] == [1] and sampled["candidate_post_ids"] == [2, 3]
    # 第三帖是空白帖，也不得因没有段而偷偷改变抽样人口。
    records = json.loads((tmp_path / "independent/samples.json").read_text())
    assert next(row for row in records if row["source_post_id"] == 3)["segments"] == []
    review = json.loads((tmp_path / "independent/review.json").read_text())
    assert all(row["assistant_judgment"] == "pending" for row in review["samples"])


def test_independent_sampling_includes_pressure_exclusions_and_ignores_list_order(prepared, tmp_path):
    """随机开发与最长压力两类都参与排除；清单列举顺序不能影响候选人口。"""
    source, run, bundle = prepared
    manifest = import_segmentation_run(run, bundle, source, code_version="independent-test")
    selected = []
    for index, ids in enumerate(([1, 2], [2, 1])):
        excluded = tmp_path / f"comparison-{index}.json"
        excluded.write_text(json.dumps({"artifact_kind": "semantic-development-comparison",
            "input": manifest["input"], "post_ids": ids,
            "development_post_ids": [1], "targeted_post_ids": [2]}))
        sampled = sample_segment_bundle(bundle, tmp_path / f"audit-{index}", count=1, seed=20261012,
                                        exclude_manifests=(excluded,), independent=True)
        selected.append(sampled)
    assert selected[0]["source_post_ids"] == selected[1]["source_post_ids"] == [3]
    assert selected[0]["candidate_population_sha256"] == selected[1]["candidate_population_sha256"]
    assert selected[0]["exclusion_post_ids_sha256"] == selected[1]["exclusion_post_ids_sha256"]


def test_matching_post_ids_do_not_make_cross_source_exclusions_valid(prepared, tmp_path):
    """同ID可在另一快照指向不同帖，排除关系必须绑定当前来源而非裸整数。"""
    source, run, bundle = prepared
    manifest = import_segmentation_run(run, bundle, source, code_version="independent-test")
    excluded = tmp_path / "other-source.json"
    identity = dict(manifest["input"], source_snapshot_sha256="0" * 64)
    excluded.write_text(json.dumps({"artifact_kind": "semantic-development-exclusions",
                                    "input": identity, "post_ids": [1]}))
    with pytest.raises(ValueError, match="来源不一致"):
        sample_segment_bundle(bundle, tmp_path / "cross-source", count=1, seed=20261012,
                              exclusion_manifest=excluded, independent=True)
    assert not (tmp_path / "cross-source").exists()
