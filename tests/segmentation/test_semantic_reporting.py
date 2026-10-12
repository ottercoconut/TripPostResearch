"""分层统计空／失败／单单元分母与压力成员唯一计数的边界验证。"""

import pytest

from tourism_ugc_study.segmentation.semantic_reporting import summarize_comparison_cases


def case(pid, text, *, units, failed=False, pressure=False):
    """构造统计输入，明确不宣称这些人工数据具有真实主题效果。"""
    segment = {"seg_id": 1, "raw_text": text, "start": 0, "end": len(text)}
    return {"source_post_id": pid, "source_text": text, "status": "failed" if failed else "completed",
            "segments": [] if failed or not text else [segment], "unit_count": units,
            "embedded_unit_count": units if units >= 2 else 0,
            "cache_hit_count": units if units >= 2 else 0, "model_call_count": 0,
            "semantic_boundary_count": 0, "length_boundary_count": 0,
            "sample_role": "pressure" if pressure else "development",
            "segment_token_counts": [] if failed or not text else [len(text)],
            "segment_nonspace_counts": [] if failed or not text else [len(text)],
            "decision": {"reason_codes": ["invalid_window_vector"] if failed else ["kept"]},
            "boundary_seconds": 0.1}


def test_retention_denominator_excludes_empty_and_failed_but_reports_single_units():
    """单单元可整帖保留，但不是已评分整帖；失败与空帖不能制造保留率。"""
    report = summarize_comparison_cases([
        case(1, "", units=0), case(2, "甲", units=1), case(3, "乙丙", units=2),
        case(4, "丁戊", units=2, failed=True), case(5, "海" * 1001, units=20, pressure=True),
    ])
    all_posts = report["all"]
    assert all_posts["post_count"] == 5
    assert all_posts["empty_completed_count"] == 1 and all_posts["failed_count"] == 1
    assert all_posts["single_unit_count"] == 1 and all_posts["candidate_post_count"] == 3
    assert all_posts["whole_post_retention_ratio"] == {"numerator": 3, "denominator": 3, "value": 1.0}
    assert all_posts["scored_whole_post_retention_ratio"] == {"numerator": 2, "denominator": 2, "value": 1.0}
    assert all_posts["model_participation_ratio"]["value"] == 0
    assert report["length_strata"]["101-300"]["whole_post_retention_ratio"]["value"] is None
    assert report["length_strata"]["1001+"]["role_counts"] == {"pressure": 1}
    assert all_posts["failure_counts"] == {"invalid_window_vector": 1}


def test_pressure_member_cannot_be_counted_twice():
    """同一压力帖同时以开发角色传入会直接拒绝，而不会污染分层分母。"""
    duplicate = case(1, "海" * 1001, units=20, pressure=True)
    with pytest.raises(ValueError, match="comparison_duplicate_post"):
        summarize_comparison_cases([duplicate, {**duplicate, "sample_role": "development"}])


def test_real_processing_time_stays_separate_from_replay_boundary_time():
    """trace全帖处理时间与重放边界时间有不同范围，不能互填或误称模型耗时。"""
    original = case(1, "海" * 150, units=3)
    original.update(processing_seconds=7.5, boundary_seconds=None)
    replay = case(2, "山" * 150, units=3)
    replay["boundary_seconds"] = 0.2
    stats = summarize_comparison_cases([original, replay])["length_strata"]["101-300"]
    assert stats["processing_seconds"]["count"] == 1
    assert stats["total_processing_seconds"] == 7.5
    assert stats["boundary_seconds"]["count"] == 1
    assert stats["total_boundary_seconds"] == 0.2
