"""统筹方核对跨模型汇总的独立反例：有效文件摘要不能证明计量口径正确。"""

import json

import pytest

from test_semantic_model_comparison import model_comparisons
from tourism_ugc_study.segmentation.semantic_model_comparison import compare_semantic_models
from tourism_ugc_study.segmentation.embedding import identity_sha256
from tourism_ugc_study.segmentation.source import file_sha256


@pytest.mark.parametrize("field", ["corpus_model_call_count", "preflight_model_call_count"])
def test_rehashed_report_cannot_overstate_baseline_inference_cost(model_comparisons, tmp_path, field):
    """模型选择的成本证据须对应原run，不能只核验派生报告的新SHA便照搬。"""
    inputs, _ = model_comparisons
    directory = inputs[1].comparison
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["baseline"][field] += 100
    summary_path.write_text(json.dumps(summary, ensure_ascii=False))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files_sha256"][summary_path.name] = file_sha256(summary_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False))

    output = tmp_path / "misleading-model-cost"
    with pytest.raises(ValueError):
        compare_semantic_models(inputs, output)
    assert not output.exists()


@pytest.mark.parametrize('field', ['rule_version', 'implementation_version'])
def test_rehashed_unknown_boundary_rule_cannot_enter_model_comparison(model_comparisons, tmp_path, field):
    """不同算法会混淆模型效果；文件SHA全部合法也必须核验实际规则身份。"""
    inputs, _ = model_comparisons
    source = inputs[1]
    run_path = source.run / 'manifest.json'
    run = json.loads(run_path.read_text())
    run[field] = 'unregistered-boundary-rule'
    run_path.write_text(json.dumps(run, ensure_ascii=False))
    manifest_path = source.comparison / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['run_manifest_sha256'] = file_sha256(run_path)
    manifest['run_validation']['run_manifest_sha256'] = file_sha256(run_path)
    manifest['comparison_id'] = identity_sha256({key: manifest[key] for key in
        ('run_manifest_sha256', 'members_manifest_sha256', 'frozen_units_sha256', 'configs', 'runtime')})
    for spec in manifest['configs']:
        review_path = source.comparison / f"review-{spec['config_id']}.json"
        review = json.loads(review_path.read_text())
        review['comparison_id'] = manifest['comparison_id']
        review_path.write_text(json.dumps(review, ensure_ascii=False))
        manifest['files_sha256'][review_path.name] = file_sha256(review_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False))
    output = tmp_path / 'confounded-model-comparison'
    with pytest.raises(ValueError, match='semantic_rule_identity_mismatch'):
        compare_semantic_models(inputs, output)
    assert not output.exists()
