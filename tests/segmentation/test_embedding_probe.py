"""真实模型探针的数值验收纯函数测试；不加载模型或执行设备推理。"""

import numpy as np
import pytest

from tourism_ugc_study.segmentation import embedding_probe as probe


def test_comparison_metrics_detects_permuted_rows_even_when_pairwise_scores_match():
    """成对相似度矩阵可能保持不变，仍须逐输入位置比较向量。"""
    base = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    swapped = base[::-1].copy()
    before = base.copy()
    metrics = probe._comparison_metrics(base, swapped)
    assert metrics["max_absolute_vector_difference"] == pytest.approx(1.0)
    assert metrics["max_pairwise_cosine_difference"] == pytest.approx(0.0)
    assert np.array_equal(base, before)
    with pytest.raises(ValueError):
        probe._require_numerical_consistency(metrics, tolerance=1e-5, label="permuted")


def test_comparison_metrics_reports_hand_calculable_cosine_difference():
    base = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    changed = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    metrics = probe._comparison_metrics(base, changed)
    assert metrics["max_absolute_vector_difference"] == pytest.approx(1.0)
    assert metrics["max_pairwise_cosine_difference"] == pytest.approx(1.0)


def test_identical_vectors_satisfy_repetition_tolerance():
    value = np.array([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
    metrics = probe._comparison_metrics(value, value.copy())
    assert all(difference == 0 for difference in metrics.values())
    probe._require_numerical_consistency(metrics, tolerance=1e-5, label="repeat")


@pytest.mark.parametrize("key", ["max_absolute_vector_difference", "max_pairwise_cosine_difference"])
def test_tolerance_includes_boundary_and_rejects_any_metric_above_it(key):
    tolerance = 1e-5
    metrics = {"max_absolute_vector_difference": 0.0, "max_pairwise_cosine_difference": 0.0}
    metrics[key] = tolerance
    probe._require_numerical_consistency(metrics, tolerance=tolerance, label="boundary")
    metrics[key] = float(np.nextafter(tolerance, np.inf))
    with pytest.raises(ValueError):
        probe._require_numerical_consistency(metrics, tolerance=tolerance, label="over-boundary")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("key", ["max_absolute_vector_difference", "max_pairwise_cosine_difference"])
def test_nonfinite_comparison_cannot_pass_tolerance(value, key):
    metrics = {"max_absolute_vector_difference": 0.0, "max_pairwise_cosine_difference": 0.0}
    metrics[key] = value
    with pytest.raises(ValueError):
        probe._require_numerical_consistency(metrics, tolerance=1e-4, label="nonfinite")


def test_unit_vector_check_accepts_float32_normalization_rounding():
    vectors = np.array([[1.0000001192092896, 0.0], [0.6, 0.8]], dtype=np.float32)
    probe._require_unit_vectors(vectors)


@pytest.mark.parametrize("vectors", [
    [[0.0, 0.0]],
    [[1.001, 0.0]],
    [[0.999, 0.0]],
    [[float("nan"), 0.0]],
    [[float("inf"), 0.0]],
])
def test_invalid_norm_or_nonfinite_vector_is_rejected(vectors):
    with pytest.raises(ValueError):
        probe._require_unit_vectors(np.array(vectors, dtype=np.float32))


@pytest.mark.parametrize("vectors", [
    np.array([1.0, 0.0], dtype=np.float32),
    np.empty((0, 2), dtype=np.float32),
    np.array([[1.0, 0.0]], dtype=np.float64),
])
def test_unit_vector_shape_and_storage_precision_are_explicit(vectors):
    with pytest.raises(ValueError):
        probe._require_unit_vectors(vectors)


def test_comparison_does_not_broadcast_different_row_counts():
    base = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        probe._comparison_metrics(base, base[:1])


def test_comparison_rejects_missing_measurement():
    with pytest.raises(ValueError):
        probe._require_numerical_consistency(
            {"max_absolute_vector_difference": 0.0}, tolerance=1e-5, label="missing-cosine")
