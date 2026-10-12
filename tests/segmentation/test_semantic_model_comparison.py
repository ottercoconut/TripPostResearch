"""跨模型材料可比性的合成反例；不加载权重，不把技术成功当作质量通过。"""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from test_semantic_api import SyntheticProvider
from test_semantic_batch import make_input
from tourism_ugc_study.segmentation import embedding
from tourism_ugc_study.segmentation.embedding import identity_sha256
from tourism_ugc_study.segmentation.semantic_batch import run_semantic_batch
from tourism_ugc_study.segmentation.semantic_comparison import compare_semantic_configs
from tourism_ugc_study.segmentation.semantic_config import EmbeddingConfig, SemanticConfig, SemanticSettings
from tourism_ugc_study.segmentation.semantic_evaluation import prepare_development_sample
from tourism_ugc_study.segmentation.semantic_model_comparison import ModelComparisonInput, compare_semantic_models
from tourism_ugc_study.segmentation.source import file_sha256


@pytest.fixture
def model_comparisons(tmp_path):
    """两种合成型号共用四层冻结成员和两组预算，原包仍走真实编排接口。"""
    source = make_input(tmp_path, rows=[
        (1, "甲" * 15 + "。" + "乙" * 15 + "。", "keep"),
        (2, "海" * 150, "keep"), (3, "山" * 500, "keep"), (4, "路" * 1100, "keep"),
    ])
    inputs, providers = [], []
    members = tmp_path / "members"
    specs = (("0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"),
             ("4B", "5cf2132abc99cad020ac570b19d031efec650f2b"))
    for index, (size, revision) in enumerate(specs):
        settings = SemanticSettings(SemanticConfig(), EmbeddingConfig(
            model_id=f"Qwen/Qwen3-Embedding-{size}", revision=revision, device="cpu"))
        provider = SyntheticProvider(settings.embedding)
        if not index:
            sample = prepare_development_sample(source, members, settings=settings,
                                                per_bucket=1, counter=provider)
        run, comparison = tmp_path / f"run-{index}", tmp_path / f"comparison-{index}"
        cache = tmp_path / f"cache-{index}"
        run_semantic_batch(source, run, settings=settings, code_version="synthetic-test",
            cache_dir=cache, provider=provider, post_ids=tuple(sample["post_ids"]),
            members_path=members / "manifest.json")
        compare_semantic_configs(source, members / "manifest.json", run, comparison,
            configs=tuple(replace(settings.algorithm, absolute_cohesion_threshold=threshold)
                          for threshold in (0.45, 0.65)), counter=provider)
        inputs.append(ModelComparisonInput(comparison, run))
        providers.append(provider)
    return tuple(inputs), providers


def _rewrite_json(path, value):
    """写回被篡改内容，使外层摘要同步而非依赖简单坏文件检测。"""
    path.write_text(json.dumps(value, ensure_ascii=False))


@pytest.mark.parametrize("field", ["units", "source_text", "model_identity_sha256"])
def test_rehashed_cases_cannot_change_candidates_or_borrow_model_identity(model_comparisons, tmp_path, field):
    """完整 JSON 和新的文件 SHA 不能替代原运行、冻结候选及缓存模型身份。"""
    inputs, _ = model_comparisons
    path = inputs[1].comparison / "manifest.json"
    manifest = json.loads(path.read_text())
    config_id = manifest["configs"][0]["config_id"]
    cases_path = inputs[1].comparison / f"cases-{config_id}.jsonl"
    rows = [json.loads(line) for line in cases_path.read_text().splitlines()]
    if field == "units":
        rows[0][field][0]["end"] -= 1
        rows[0][field][0]["token_count"] -= 1
        rows[0][field][1]["start"] -= 1
        rows[0][field][1]["token_count"] += 1
    elif field == "source_text":
        rows[0][field] = "外" + rows[0][field][1:]
    else:
        rows[0][field] = json.loads((inputs[0].comparison / "manifest.json").read_text())[field]
    cases_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest["files_sha256"][cases_path.name] = file_sha256(cases_path)
    _rewrite_json(path, manifest)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="frozen_case|model_cache_binding"):
        compare_semantic_models(inputs, output)
    assert not output.exists()


def test_rehashed_smaller_budget_is_not_a_matching_model_comparison(model_comparisons, tmp_path):
    """单模型子预算自身有效，仍不能当作另一模型完整预算的公平比较。"""
    inputs, _ = model_comparisons
    directory = inputs[1].comparison
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    removed = manifest["configs"].pop()
    manifest["config_count"] = 1
    summary["groups"].pop()
    _rewrite_json(directory / "summary.json", summary)
    manifest["files_sha256"]["summary.json"] = file_sha256(directory / "summary.json")
    for name in (f"cases-{removed['config_id']}.jsonl", f"review-{removed['config_id']}.json"):
        manifest["files_sha256"].pop(name)
    manifest["comparison_id"] = identity_sha256({key: manifest[key] for key in
        ("run_manifest_sha256", "members_manifest_sha256", "frozen_units_sha256", "configs", "runtime")})
    review_path = directory / f"review-{manifest['configs'][0]['config_id']}.json"
    review = json.loads(review_path.read_text())
    review["comparison_id"] = manifest["comparison_id"]
    _rewrite_json(review_path, review)
    manifest["files_sha256"][review_path.name] = file_sha256(review_path)
    _rewrite_json(directory / "manifest.json", manifest)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="parameter_budget_mismatch"):
        compare_semantic_models(inputs, output)
    assert not output.exists()


def test_comparison_bytes_must_match_the_published_inventory(model_comparisons, tmp_path):
    """只追加合法 JSON 空白也改变精确文件身份，发布后不能静默接纳。"""
    inputs, _ = model_comparisons
    with (inputs[1].comparison / "summary.json").open("a") as stream:
        stream.write("\n")
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="file_digest_mismatch"):
        compare_semantic_models(inputs, output)
    assert not output.exists()


def test_same_model_twice_cannot_supply_two_models(model_comparisons, tmp_path):
    """不同路径不是不同型号，重复提交同一已完成比较也须拒绝。"""
    inputs, _ = model_comparisons
    with pytest.raises(ValueError, match="distinct_models_required"):
        compare_semantic_models((inputs[0], inputs[0]), tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_pending_inputs_never_become_quality_conclusions_or_trigger_inference(model_comparisons, tmp_path, monkeypatch):
    """核验逐帖索引，同时反证推理和自动质量提升，保留唯一原审阅入口。"""
    inputs, providers = model_comparisons
    original_calls = [provider.model_call_count for provider in providers]

    def forbidden(*args, **kwargs):
        pytest.fail("跨模型汇总不应加载模型、tokenizer或请求embedding")

    monkeypatch.setattr(SyntheticProvider, "encode", forbidden)
    monkeypatch.setattr(embedding.QwenEmbeddingProvider, "__init__", forbidden)
    monkeypatch.setattr(embedding.QwenTokenCounter, "__init__", forbidden)
    output = tmp_path / "models"
    manifest = compare_semantic_models(inputs, output)
    assert manifest["review_status"] == "pending"
    assert manifest["semantic_quality_evaluated"] is False
    assert manifest["actual_embedding_model_calls"] == 0
    assert [provider.model_call_count for provider in providers] == original_calls
    summary = json.loads((output / "summary.json").read_text())
    assert all(model["review_status"] == "pending" and model["semantic_quality_evaluated"] is False
               for group in summary["groups"] for model in group["models"])
    rows = [json.loads(line) for line in (output / "post-index.jsonl").read_text().splitlines()]
    assert [row["source_post_id"] for row in rows] == [1, 2, 3, 4]
    assert len(rows[0]["groups"]) == 2
    assert len(rows[0]["groups"][0]["models"]) == 2
    assert not (output / "review.json").exists()
    for model in rows[0]["groups"][0]["models"]:
        review = json.loads(Path(model["review_path"]).read_text())
        assert all(row["assistant_judgment"] == "pending" for row in review["samples"])
