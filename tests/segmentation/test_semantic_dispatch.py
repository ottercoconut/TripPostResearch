"""两版配置和边界分派的兼容证据；只用合成文本、计数器及向量。"""

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from test_semantic_api import SyntheticProvider
from tourism_ugc_study.segmentation.contracts import TextUnit
from tourism_ugc_study.segmentation.integrity import validate
from tourism_ugc_study.segmentation.semantic_api import segment_body_semantic
from tourism_ugc_study.segmentation.semantic_boundaries import choose_boundaries
from tourism_ugc_study.segmentation.semantic_config import (
    EmbeddingConfig, IMPLEMENTATION_VERSION, RULE_VERSION, SemanticConfig, SemanticSettings,
    STRUCTURED_IMPLEMENTATION_VERSION, STRUCTURED_RULE_VERSION, StructuredSemanticConfig,
    load_semantic_settings, parse_algorithm_config, parse_semantic_settings,
    semantic_rule_identity, validate_semantic_rule_identity,
)
from tourism_ugc_study.segmentation.semantic_dispatch import choose_semantic_boundaries


class CharacterCounter:
    """每码点一个token，便于检查真实跨度及长度，而非模拟模型效果。"""

    def count(self, text):
        """不截断原文，完整返回长度。"""
        return len(text)


def make_units(parts, separator=""):
    """把指定合成句子组成无损正文，分隔空白保留原坐标。"""
    text = separator.join(parts)
    cursor = 0
    units = []
    for part in parts:
        units.append(TextUnit(cursor, cursor + len(part), len(part)))
        cursor += len(part) + len(separator)
    return text, tuple(units)


def assert_final_consistency(text, units, output):
    """独立从输出片段端点核对gap标记，不能只比较两份诊断字段。"""
    validate(text, [asdict(segment) for segment in output.segments])
    selected = set(output.decision.semantic_boundaries) | set(output.decision.length_boundaries)
    assert not set(output.decision.semantic_boundaries) & set(output.decision.length_boundaries)
    assert [units[index].end for index in sorted(selected)] == [segment.end for segment in output.segments[:-1]]
    assert {gap.index for gap in output.decision.gaps if gap.selected} == selected
    assert len(output.decision.gaps) == max(0, len(units) - 1)


def test_legacy_configuration_keeps_exact_fields_and_saved_fingerprint(tmp_path):
    """固定历史配置摘要，防止给旧dataclass加默认字段而破坏已封存证据。"""
    fields = {"unit_max_tokens", "window_tokens", "absolute_cohesion_threshold", "depth_floor",
              "depth_std_factor", "min_segment_nonspace_chars", "max_segment_tokens", "epsilon", "smoothing"}
    assert set(asdict(SemanticConfig())) == fields
    path = tmp_path / "historical-config.json"
    path.write_text(json.dumps(asdict(SemanticSettings(SemanticConfig(), EmbeddingConfig()))))
    settings = load_semantic_settings(path)
    assert type(settings.algorithm) is SemanticConfig
    assert settings.fingerprint == "b1bbabf69f20c0c65d87e8ded7e1f315eed37c007df79866a2c929b292638086"
    assert asdict(settings) == json.loads(path.read_text())
    assert parse_algorithm_config({}, require_complete=False) == SemanticConfig()


def test_explicit_structured_configuration_roundtrips_and_changes_fingerprint():
    """只有明确0.2才启用结构默认值，模式和长度策略都属于配置身份。"""
    new = parse_algorithm_config({"rule_version": STRUCTURED_RULE_VERSION})
    assert type(new) is StructuredSemanticConfig
    assert new.structure_mode == "context" and new.length_strategy == "global"
    settings = SemanticSettings(new, EmbeddingConfig())
    assert parse_semantic_settings(asdict(settings)) == settings
    assert parse_algorithm_config(asdict(new), require_complete=True) == new
    assert parse_algorithm_config(asdict(SemanticConfig()), require_complete=True) == SemanticConfig()
    legacy = SemanticSettings(SemanticConfig(), settings.embedding)
    assert settings.fingerprint != legacy.fingerprint
    assert replace(settings, algorithm=replace(new, structure_mode="blocks")).fingerprint != settings.fingerprint
    assert replace(settings, algorithm=replace(new, length_strategy="greedy")).fingerprint != settings.fingerprint


@pytest.mark.parametrize("raw,complete", [
    (None, False), ([], False), ({"structure_mode": "context"}, False),
    ({"rule_version": RULE_VERSION}, False), ({"rule_version": "future"}, False),
    ({"rule_version": None}, False), ({"rule_version": STRUCTURED_RULE_VERSION, "unknown": 1}, False),
    ({"rule_version": STRUCTURED_RULE_VERSION, "structure_mode": "guess"}, False),
    ({"rule_version": STRUCTURED_RULE_VERSION, "length_strategy": "guess"}, False),
    ({}, True), ({"rule_version": STRUCTURED_RULE_VERSION}, True),
])
def test_factory_rejects_inferred_upgrades_unknown_fields_and_incomplete_manifests(raw, complete):
    """版本不能凭新字段猜测；完整manifest不能借默认值补齐缺失证据。"""
    with pytest.raises(ValueError):
        parse_algorithm_config(raw, require_complete=complete)


@pytest.mark.parametrize("field,value", [
    ("rule_version", RULE_VERSION), ("structure_mode", None), ("length_strategy", []),
    ("max_segment_tokens", 1), ("window_tokens", True),
])
def test_structured_class_validates_both_inherited_and_new_parameters(field, value):
    """直接构造也不得绕过版本和继承参数校验。"""
    with pytest.raises(ValueError):
        StructuredSemanticConfig(**{field: value})


@pytest.mark.parametrize("algorithm,identity", [
    (SemanticConfig(), (RULE_VERSION, IMPLEMENTATION_VERSION)),
    (StructuredSemanticConfig(), (STRUCTURED_RULE_VERSION, STRUCTURED_IMPLEMENTATION_VERSION)),
])
def test_registered_identity_requires_matching_configuration_when_present(algorithm, identity):
    """已登记配对仍须与config相符，不能替换清单标签伪装算法升级。"""
    assert semantic_rule_identity(algorithm) == identity
    manifest = dict(zip(("rule_version", "implementation_version"), identity, strict=True))
    validate_semantic_rule_identity(manifest)
    manifest["config"] = asdict(SemanticSettings(algorithm, EmbeddingConfig()))
    validate_semantic_rule_identity(manifest)
    manifest["config"]["algorithm"] = asdict(StructuredSemanticConfig() if type(algorithm) is SemanticConfig else SemanticConfig())
    with pytest.raises(ValueError, match="semantic_rule_identity_mismatch"):
        validate_semantic_rule_identity(manifest)


@pytest.mark.parametrize("manifest", [
    None, {}, {"rule_version": RULE_VERSION, "implementation_version": STRUCTURED_IMPLEMENTATION_VERSION},
    {"rule_version": STRUCTURED_RULE_VERSION, "implementation_version": "unknown"},
    {"rule_version": RULE_VERSION, "implementation_version": IMPLEMENTATION_VERSION, "config": None},
])
def test_invalid_or_mismatched_rule_identity_is_rejected(manifest):
    """未知配对及坏配置统一拒绝，不回落到旧规则。"""
    with pytest.raises(ValueError, match="semantic_rule_identity_mismatch"):
        validate_semantic_rule_identity(manifest)


def test_unregistered_subclass_and_mutated_frozen_config_are_rejected():
    """未经登记的子类和绕过frozen的状态不能借旧身份执行。"""
    class UnknownConfig(SemanticConfig):
        """仅用于模拟未登记的扩展配置。"""
    with pytest.raises(ValueError, match="invalid_semantic_config_type"):
        semantic_rule_identity(UnknownConfig())
    changed = StructuredSemanticConfig()
    object.__setattr__(changed, "structure_mode", "unexpected")
    with pytest.raises(ValueError, match="unknown_structure_mode"):
        semantic_rule_identity(changed)


@pytest.mark.parametrize("parts,vectors,config", [
    ([], None, SemanticConfig()),
    (["海边散步。"], None, SemanticConfig()),
    (["甲" * 15 + "。", "乙" * 15 + "。"], [[1, 0], [0, 1]], SemanticConfig()),
    (["甲。", "乙。", "丙。", "丁。", "戊。"], [[1, 0]] * 5,
     SemanticConfig(unit_max_tokens=2, max_segment_tokens=4, min_segment_nonspace_chars=1)),
])
def test_legacy_dispatch_is_exactly_the_original_boundary_result(parts, vectors, config):
    """旧版的空、单单元、语义及长度路径须逐字段保持原结果。"""
    text, units = make_units(parts)
    expected_segments, expected_decision = choose_boundaries(text, units, vectors, config, CharacterCounter())
    actual = choose_semantic_boundaries(text, units, vectors, config, CharacterCounter())
    assert actual.segments == expected_segments and actual.decision == expected_decision
    assert actual.structure_trace is None
    assert_final_consistency(text, units, actual)


@pytest.mark.parametrize("parts,status", [([], "empty"), (["海边散步。"], "single_unit")])
def test_structured_empty_and_single_unit_do_not_invent_scores(parts, status):
    """0.2仍无需短帖向量，空结构trace不制造gap或边界。"""
    text, units = make_units(parts)
    actual = choose_semantic_boundaries(text, units, None, StructuredSemanticConfig(), CharacterCounter())
    assert actual.decision.status == status and actual.decision.gaps == ()
    assert actual.structure_trace.gaps == ()
    assert actual.structure_trace.suppressed_semantic_boundaries == ()
    assert actual.structure_trace.forced_structure_boundaries == ()
    assert_final_consistency(text, units, actual)


def test_context_suppresses_heading_cut_and_rebuilds_final_summary():
    """评分建议在题头后切，结构合并后最终selected及状态必须同步撤销。"""
    text, units = make_units(["山间漫步", "沿着步道到达山顶后可以看到远处的河流和村庄，傍晚还有温柔的夕阳。"], "\n")
    config = StructuredSemanticConfig(min_segment_nonspace_chars=1)
    actual = choose_semantic_boundaries(text, units, [[1, 0], [0, 1]], config, CharacterCounter())
    assert actual.structure_trace.initial_decision.semantic_boundaries == (0,)
    assert actual.structure_trace.suppressed_semantic_boundaries == (0,)
    assert actual.decision.semantic_boundaries == actual.decision.length_boundaries == ()
    assert actual.decision.status == "scored_keep"
    assert actual.decision.gaps[0].reason == "structure_suppressed_semantic"
    assert not actual.decision.gaps[0].selected
    assert actual.structure_trace.protected_blocks
    assert_final_consistency(text, units, actual)
    # 同一评分改为仅自然句／行模式，题头后的换行允许切，证明分派实际使用配置。
    natural = choose_semantic_boundaries(text, units, [[1, 0], [0, 1]],
        replace(config, structure_mode="sentence_only"), CharacterCounter())
    assert natural.decision.semantic_boundaries == (0,)
    assert natural.structure_trace.protected_blocks == ()


def test_global_partition_removes_stale_length_selections_and_reasons():
    """旧贪心长度点被全局分区替换后，旧点不得留selected或length原因。"""
    text, units = make_units(["甲。", "乙。", "丙。", "丁。", "戊。"])
    config = StructuredSemanticConfig(unit_max_tokens=2, max_segment_tokens=4,
        min_segment_nonspace_chars=1, absolute_cohesion_threshold=-1, structure_mode="sentence_only")
    actual = choose_semantic_boundaries(text, units, [[1, 0]] * 5, config, CharacterCounter())
    assert actual.structure_trace.initial_decision.length_boundaries == (1, 3)
    assert actual.decision.length_boundaries == (0, 2)
    for index in (1, 3):
        assert not actual.decision.gaps[index].selected
        assert not actual.decision.gaps[index].reason.startswith("length_limit_after_")
    assert actual.decision.status == "length_split"
    assert_final_consistency(text, units, actual)


def test_suppressed_semantic_cut_can_return_only_as_forced_length_boundary():
    """上下文保护无法满足段长时可明确退让，但同一gap不能重复归类计数。"""
    text, units = make_units(["Day1", "循着木栈道观赏山谷。"], "\n")
    config = StructuredSemanticConfig(unit_max_tokens=12, max_segment_tokens=12, min_segment_nonspace_chars=1)
    actual = choose_semantic_boundaries(text, units, [[1, 0], [0, 1]], config, CharacterCounter())
    assert actual.structure_trace.suppressed_semantic_boundaries == (0,)
    assert actual.structure_trace.forced_structure_boundaries == (0,)
    assert actual.decision.semantic_boundaries == () and actual.decision.length_boundaries == (0,)
    assert actual.decision.gaps[0].selected and actual.decision.status == "length_split"
    assert "forced_structure_boundary" in actual.decision.reason_codes
    assert_final_consistency(text, units, actual)


def test_oversized_structure_block_retains_explicit_relaxation_trace():
    """题头及首句都装不下时记录退让，不以空保护列表冒充未发现结构。"""
    text = "Day1\n沿着步道到达山顶后可以看到远处的河流和村庄傍晚还有温柔的夕阳。"
    units = (TextUnit(0, 4, 4), *(TextUnit(start, min(start + 8, len(text)), min(8, len(text) - start))
              for start in range(5, len(text), 8)))
    config = StructuredSemanticConfig(unit_max_tokens=8, max_segment_tokens=24, min_segment_nonspace_chars=1)
    actual = choose_semantic_boundaries(text, units, [[1, 0]] * len(units), config, CharacterCounter())
    assert len(actual.structure_trace.block_relaxations) == 1
    relaxation = actual.structure_trace.block_relaxations[0]
    assert (relaxation.start, relaxation.end) == (0, len(text))
    assert relaxation.reason == "whole_block_over_token_limit"
    assert relaxation.token_count == len(text) and relaxation.retained_end is None
    assert all(len(segment.raw_text) <= config.max_segment_tokens for segment in actual.segments)
    assert_final_consistency(text, units, actual)


def test_semantic_api_preserves_summary_keys_and_exposes_separate_structured_trace():
    """编排层结果身份跟随算法配置，结构字段不污染既有summary契约。"""
    body = "甲" * 15 + "。" + "乙" * 15 + "。"
    provider = SyntheticProvider()
    old = segment_body_semantic(body, provider=provider, token_counter=provider, config=SemanticConfig())
    new = segment_body_semantic(body, provider=provider, token_counter=provider, config=StructuredSemanticConfig())
    assert old.structure_trace is None and new.structure_trace is not None
    assert set(new.summary()) == set(old.summary())
    assert (new.result.rule_version, new.result.implementation_version) == semantic_rule_identity(StructuredSemanticConfig())
    assert (old.result.rule_version, old.result.implementation_version) == (RULE_VERSION, IMPLEMENTATION_VERSION)
    assert new.summary()["selected_boundary_count"] == len(new.result.text_segments) - 1
    assert json.loads(json.dumps(asdict(new.structure_trace)))["structure_mode"] == "context"


@pytest.mark.parametrize("body", [None, "", "海边真好看。"])
def test_structured_api_short_posts_still_skip_embedding(body):
    """新算法不以结构判断为由额外调用短帖模型。"""
    provider = SyntheticProvider(fail_on_call=1)
    actual = segment_body_semantic(body, provider=provider, token_counter=provider, config=StructuredSemanticConfig())
    assert provider.model_call_count == 0
    assert actual.result.status == "completed" and actual.structure_trace is not None


def test_structured_api_bad_projection_has_no_partial_structure_trace():
    """损坏投影不产生部分结构证据，失败记录仍带0.2规则身份。"""
    provider = SyntheticProvider(fail_on_call=1)
    actual = segment_body_semantic('{"ops":broken}', provider=provider, token_counter=provider,
                                   config=StructuredSemanticConfig())
    assert actual.result.status == "failed" and actual.result.text_segments is None
    assert actual.structure_trace is None and provider.model_call_count == 0
    assert actual.result.rule_version == STRUCTURED_RULE_VERSION
