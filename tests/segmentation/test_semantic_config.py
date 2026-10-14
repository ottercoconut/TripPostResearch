"""算法和模型配置的严格解析及可复现指纹；不下载或初始化模型。"""

from dataclasses import asdict, replace
import json

import pytest

from tourism_ugc_study.segmentation.semantic_config import (
    EmbeddingConfig, CoherentSemanticConfig, SemanticSettings, load_semantic_settings,
)


@pytest.mark.parametrize("field,value", [
    ("unit_max_tokens", 0), ("window_tokens", -1), ("window_tokens", True),
    ("max_segment_tokens", 32),
    ("absolute_cohesion_threshold", float("nan")), ("absolute_cohesion_threshold", 1.1),
    ("depth_floor", 0), ("depth_std_factor", -1), ("epsilon", 0),
    ("epsilon", float("inf")), ("smoothing", 1), ("smoothing", False),
])
def test_algorithm_rejects_parameters_that_break_boundary_contracts(field, value):
    """长度关系、有限数、正预算与禁用平滑均在任何模型工作前拒绝。"""
    with pytest.raises(ValueError):
        CoherentSemanticConfig(**{field: value})


@pytest.mark.parametrize("field,value", [
    ("model_id", "unexpected/model"), ("revision", "main"), ("revision", "abc"),
    ("revision", "A" * 40), ("device", "auto"), ("dtype", "int8"),
    ("batch_size", True), ("batch_size", 0), ("max_input_tokens", 32769),
    ("instruction", "  "), ("instruction", None),
])
def test_embedding_requires_explicit_frozen_supported_runtime(field, value):
    """不接受浮动revision、自动设备或未声明的量化精度。"""
    with pytest.raises(ValueError):
        EmbeddingConfig(**{field: value})


def test_configuration_load_is_order_independent_but_fingerprint_tracks_each_domain(tmp_path):
    """JSON键顺序不影响谱系；模型或算法参数变化都必须改变配置指纹。"""
    config = SemanticSettings(CoherentSemanticConfig(), EmbeddingConfig(device="cpu"))
    file = tmp_path / "semantic.json"
    file.write_text(json.dumps(asdict(config), sort_keys=True), encoding="utf-8")
    first = load_semantic_settings(file)
    raw = asdict(config)
    file.write_text(json.dumps({"embedding": raw["embedding"], "algorithm": raw["algorithm"]}), encoding="utf-8")
    second = load_semantic_settings(file)
    assert first == second == config and first.fingerprint == second.fingerprint
    assert replace(config, algorithm=replace(config.algorithm, window_tokens=64)).fingerprint != config.fingerprint
    assert replace(config, embedding=replace(config.embedding, device="mps")).fingerprint != config.fingerprint


@pytest.mark.parametrize("raw", [
    [], {}, {"algorithm": {}}, {"embedding": {}},
    {"algorithm": {}, "embedding": {}, "unknown": 1},
    {"algorithm": {"window": 64}, "embedding": {}},
    {"algorithm": {}, "embedding": {"revision_name": "main"}},
    {"algorithm": None, "embedding": {}}, {"algorithm": {}, "embedding": []},
])
def test_unknown_or_incomplete_configuration_fields_are_rejected(tmp_path, raw):
    """拼错字段、缺领域或错误层级不能被静默忽略而退到默认配置。"""
    file = tmp_path / "invalid.json"
    file.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic_config_fields_invalid"):
        load_semantic_settings(file)
