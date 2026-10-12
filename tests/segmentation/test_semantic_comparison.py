"""跨平台缓存重放的末位浮点证据与严格决策反例；不请求模型。"""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from test_semantic_acceptance import ThresholdWitnessProvider, replay_witness
from test_semantic_model_comparison import model_comparisons
from tourism_ugc_study.segmentation import embedding, semantic_comparison
from tourism_ugc_study.segmentation.semantic_config import SemanticSettings
from tourism_ugc_study.segmentation.semantic_model_comparison import ModelComparisonInput, compare_semantic_models
from tourism_ugc_study.segmentation.source import file_sha256


def _inventory(paths):
    """记录来源、运行和向量字节，阻止重放通过改写证据制造一致。"""
    return {str(path): file_sha256(path) for root in paths for path in
            (root.rglob('*') if root.is_dir() else (root,)) if path.is_file()}


def _forbid_model(*args, **kwargs):
    """缓存重放只能读取原向量，不得加载权重或补算。"""
    pytest.fail('baseline replay unexpectedly requested model')


def _perturb_replay(monkeypatch, change):
    """模拟另一BLAS的计算结果，不改原trace或持久缓存。"""
    original = semantic_comparison._boundary_result

    def calculate(*args, **kwargs):
        segments, decision, elapsed = original(*args, **kwargs)
        decision = deepcopy(decision)
        if decision['gaps']:
            change(decision, segments)
        return segments, decision, elapsed

    monkeypatch.setattr(semantic_comparison, '_boundary_result', calculate)


def _roundoff(decision, segments):
    """仅两种float评分改变末位，选点、原因与全部结构仍严格相同。"""
    decision['gaps'][0]['cohesion'] += 5e-16
    decision['gaps'][0]['depth'] += 5e-16


def test_last_bit_replay_preserves_original_baseline_and_records_actual_candidates(
    replay_witness, tmp_path, monkeypatch,
):
    """接纳末位差但保留原run绑定；报告真实delta且零推理、零来源写入。"""
    source, members, run, cache, configured, provider = replay_witness
    before = _inventory((source.source_db, source.membership_db, source.candidate_manifest,
                         members.parent, run, cache))
    calls = provider.model_call_count
    monkeypatch.setattr(ThresholdWitnessProvider, 'encode', _forbid_model)
    monkeypatch.setattr(embedding.QwenEmbeddingProvider, '__init__', _forbid_model)
    _perturb_replay(monkeypatch, _roundoff)
    output = tmp_path / 'last-bit-comparison'
    manifest = semantic_comparison.compare_semantic_configs(source, members, run, output,
        configs=(configured.algorithm,), counter=provider)
    summary = json.loads((output / 'summary.json').read_text())
    replay = summary['baseline_replay']
    assert replay['absolute_tolerance'] == 1e-12 and replay['relative_tolerance'] == 0
    assert replay['float_fields'] == ['gaps[].cohesion', 'gaps[].depth']
    assert replay['post_count'] == 4 and replay['all_decisions_exact'] is False
    assert 0 < replay['max_absolute_delta'] < 1e-12
    assert sum(not row['decision_exact'] for row in replay['posts']) > 0
    config_id = SemanticSettings(configured.algorithm, configured.embedding).fingerprint
    rows = [json.loads(line) for line in (output / f'cases-{config_id}.jsonl').read_text().splitlines()]
    traces = [json.loads(line) for line in (run / 'semantic-trace.jsonl').read_text().splitlines()]
    assert [row['baseline_decision'] for row in rows] == [row['decision'] for row in traces]
    assert rows[0]['decision'] != rows[0]['baseline_decision']
    assert rows[0]['vector_content_sha256'] == traces[0]['summary']['vector_content_sha256']
    assert manifest['baseline_reproduction'] == 'passed' and manifest['actual_embedding_model_calls'] == 0
    assert provider.model_call_count == calls and _inventory((source.source_db, source.membership_db,
        source.candidate_manifest, members.parent, run, cache)) == before


def _damage(decision, segments, kind):
    """这些改变绝不属于末位计算差，哪怕Python宽松相等也必须拒绝。"""
    gap = decision['gaps'][0]
    if kind == 'selection':
        gap['selected'] = not gap['selected']
    elif kind == 'candidate':
        gap['candidate'] = not gap['candidate']
    elif kind == 'reason':
        gap['reason'] += '_changed'
    elif kind == 'bool_type':
        gap['selected'] = int(gap['selected'])
    elif kind == 'index_type':
        gap['index'] = float(gap['index'])
    elif kind == 'score_type':
        gap['depth'] = int(gap['depth'])
    elif kind == 'extra_key':
        gap['extra'] = None
    elif kind == 'missing_key':
        gap.pop('reason')
    elif kind == 'nan':
        gap['cohesion'] = float('nan')
    elif kind == 'infinity':
        gap['depth'] = float('inf')
    elif kind == 'over_limit':
        gap['cohesion'] += 2e-12
    elif kind == 'boundaries':
        decision['semantic_boundaries'].append(0)
    elif kind == 'segment_type':
        segments[0]['start'] = float(segments[0]['start'])
    elif kind == 'segment_span':
        segments[0]['end'] -= 1
    else:
        raise AssertionError(kind)


@pytest.mark.parametrize('kind', ['selection', 'candidate', 'reason', 'bool_type', 'index_type',
    'score_type', 'extra_key', 'missing_key', 'nan', 'infinity', 'over_limit', 'boundaries',
    'segment_type', 'segment_span'])
def test_last_bit_tolerance_never_admits_non_score_changes(replay_witness, tmp_path, monkeypatch, kind):
    """通过完整比较入口拒绝结构／类型／语义变化和非有限、超限评分。"""
    source, members, run, cache, configured, provider = replay_witness
    before = _inventory((members.parent, run, cache))
    monkeypatch.setattr(ThresholdWitnessProvider, 'encode', _forbid_model)
    monkeypatch.setattr(embedding.QwenEmbeddingProvider, '__init__', _forbid_model)
    _perturb_replay(monkeypatch, lambda decision, segments: _damage(decision, segments, kind))
    output = tmp_path / 'rejected-comparison'
    with pytest.raises(ValueError, match='comparison_baseline_reproduction_failed'):
        semantic_comparison.compare_semantic_configs(source, members, run, output,
            configs=(configured.algorithm,), counter=provider)
    assert not output.exists() and _inventory((members.parent, run, cache)) == before


def test_roundoff_comparison_remains_usable_by_cross_model_binding(model_comparisons, tmp_path, monkeypatch):
    """下游仍逐字段绑定原trace；新评分只进入本机候选重放与漂移报告。"""
    inputs, providers = model_comparisons
    before = _inventory(tuple(item.run for item in inputs))
    _perturb_replay(monkeypatch, _roundoff)
    monkeypatch.setattr(embedding.QwenEmbeddingProvider, '__init__', _forbid_model)
    rebuilt = []
    for index, (item, provider) in enumerate(zip(inputs, providers, strict=True)):
        run_manifest = json.loads((item.run / 'manifest.json').read_text())
        comparison = tmp_path / f'roundoff-comparison-{index}'
        # 合成来源与冻结成员仍是fixture创建的同一份本地证据。
        source = source_for_fixture(tmp_path)
        members = tmp_path / 'members/manifest.json'
        algorithm = semantic_comparison.parse_semantic_settings(run_manifest['config']).algorithm
        monkeypatch.setattr(provider, 'encode', _forbid_model)
        semantic_comparison.compare_semantic_configs(source, members, item.run, comparison,
            configs=tuple(replace(algorithm, absolute_cohesion_threshold=value) for value in (0.45, 0.65)),
            counter=provider)
        rebuilt.append(ModelComparisonInput(comparison, item.run))
    result = compare_semantic_models(tuple(rebuilt), tmp_path / 'roundoff-models')
    assert result['actual_embedding_model_calls'] == 0 and result['status'] == 'completed'
    assert _inventory(tuple(item.run for item in inputs)) == before


def source_for_fixture(tmp_path):
    """按fixture的已有路径重新解析来源，不另建或修改数据库。"""
    from tourism_ugc_study.segmentation.config import ResearchInput
    manifest = tmp_path / 'candidate-manifest.json'
    return ResearchInput(tmp_path / 'members.sqlite', tmp_path / 'source.sqlite', manifest, file_sha256(manifest))


@pytest.mark.parametrize('delta, accepted', [(1e-12, True), (1.0001e-12, False)])
def test_score_replay_absolute_limit_is_inclusive_without_relative_tolerance(delta, accepted):
    """零分值给出精确绝对界；大评分也不能借相对容差掩盖超限。"""
    original = {'gaps': [{'index': 0, 'cohesion': 0.0, 'depth': 0.0,
                         'candidate': False, 'selected': False, 'reason': 'keep'}]}
    replay = deepcopy(original)
    replay['gaps'][0]['depth'] = delta
    if accepted:
        assert semantic_comparison._baseline_decision_replay(original, replay)['max_absolute_delta'] == delta
    else:
        with pytest.raises(ValueError, match='comparison_baseline_reproduction_failed'):
            semantic_comparison._baseline_decision_replay(original, replay)
    original['gaps'][0]['cohesion'] = 1e6
    replay = deepcopy(original)
    replay['gaps'][0]['cohesion'] += 1e-6
    with pytest.raises(ValueError, match='comparison_baseline_reproduction_failed'):
        semantic_comparison._baseline_decision_replay(original, replay)
