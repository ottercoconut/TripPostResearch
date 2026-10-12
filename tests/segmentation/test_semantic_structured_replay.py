"""结构规则重放的反例：文件摘要正确仍须核对真实结构计算证据。

全部来源与向量是临时合成数据。先生成实际0.2运行，再有意改写结构字段
并同步外层SHA，以区别文件损坏检测与结构语义重放。重放禁止任何forward。
"""

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from test_semantic_batch import make_input, read_rows
from test_semantic_structured_integration import _provider, _settings
from tourism_ugc_study.segmentation import embedding
from tourism_ugc_study.segmentation.semantic_batch import run_semantic_batch
from tourism_ugc_study.segmentation.semantic_comparison import compare_semantic_configs
from tourism_ugc_study.segmentation.semantic_config import SemanticConfig, SemanticSettings
from tourism_ugc_study.segmentation.semantic_evaluation import prepare_development_sample
from tourism_ugc_study.segmentation.source import file_sha256


@dataclass(frozen=True)
class StructuredReplayWitness:
    """原运行与冻结输入的合成证据；原trace用于独立核对候选结构输出。"""

    source: object
    members: Path
    run: Path
    cache: Path
    settings: SemanticSettings
    provider: object
    original_traces: list[dict]
    original_cache_hashes: dict[str, str]
    model_call_count: int


def _cache_hashes(cache: Path) -> dict[str, str]:
    """逐文件绑定缓存，验证比较既不补向量也不改写现有条目。"""
    return {str(path.relative_to(cache)): file_sha256(path)
            for path in cache.rglob("*") if path.is_file()}


@pytest.fixture
def structured_replay_witness(tmp_path, monkeypatch):
    """四长度层均有成员，并含真实超长题头保护块退让，冻结单位仍使用0.1。"""
    source = make_input(tmp_path, rows=[
        (1, "甲" * 15 + "。" + "乙" * 15 + "。", "keep"),
        (2, "Day2\n" + "甲" * 90 + "。" + "乙" * 40 + "。", "keep"),
        (3, "山" * 500, "keep"), (4, "路" * 1100, "keep"),
    ])
    configured = _settings()
    provider = _provider(configured)
    members_dir, run, cache = tmp_path / "members", tmp_path / "run", tmp_path / "vectors"
    sampled = prepare_development_sample(source, members_dir,
        settings=SemanticSettings(SemanticConfig(), configured.embedding), per_bucket=1, counter=provider)
    members = members_dir / "manifest.json"
    run_semantic_batch(source, run, settings=configured, code_version="synthetic-structure-replay",
        provider=provider, cache_dir=cache, post_ids=tuple(sampled["post_ids"]),
        seed=sampled["seed"], members_path=members)
    traces = read_rows(run / "semantic-trace.jsonl")
    assert all("structure" in row for row in traces)
    assert any(row["structure"]["block_relaxations"] for row in traces)

    def forbidden(*args, **kwargs):
        """拒绝模型加载和补算，使重放成功只能来自现有缓存。"""
        pytest.fail("结构重放只能只读使用已绑定向量")

    monkeypatch.setattr(provider, "encode", forbidden)
    monkeypatch.setattr(embedding.QwenEmbeddingProvider, "__init__", forbidden)
    return StructuredReplayWitness(source, members, run, cache, configured, provider, traces,
                                    _cache_hashes(cache), provider.model_call_count)


def _rewrite_structure(witness: StructuredReplayWitness, change: str) -> None:
    """只更改结构证据并同步trace SHA；片段、最终decision和单位均保持原字节含义。"""
    rows = json.loads(json.dumps(witness.original_traces))
    row = next(item for item in rows if item["structure"]["block_relaxations"])
    structure = row["structure"]
    if change == "missing_structure":
        row.pop("structure")
    elif change == "unknown_field":
        structure["invented_evidence"] = "不在计算契约中的字段"
    elif change == "changed_structure_mode":
        structure["structure_mode"] = "sentence_only"
    elif change == "changed_length_strategy":
        structure["length_strategy"] = "greedy"
    elif change == "invalid_forced_index":
        structure["forced_structure_boundaries"] = [len(row["units"]) + 100]
    elif change == "changed_relaxation_count":
        structure["block_relaxations"][0]["token_count"] += 1
    elif change == "changed_relaxation_retained_end":
        structure["block_relaxations"][0]["retained_end"] = 1
    elif change == "changed_initial_reason":
        structure["initial_decision"]["reason_codes"] = ["invented_reason"]
    elif change in {"initial_score_outside_tolerance", "initial_score_roundoff"}:
        structure["initial_decision"]["gaps"][0]["cohesion"] += (
            1e-6 if change == "initial_score_outside_tolerance" else 1e-13)
    else:
        raise AssertionError("未知合成篡改类型")
    trace_path = witness.run / "semantic-trace.jsonl"
    trace_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows))
    manifest_path = witness.run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["trace_sha256"] = file_sha256(trace_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False))


def _compare(witness: StructuredReplayWitness, output: Path) -> dict:
    """重放唯一显式新配置，保持原模型身份、冻结成员和完整缓存绑定。"""
    return compare_semantic_configs(witness.source, witness.members, witness.run, output,
        configs=(witness.settings.algorithm,), cache_dir=witness.cache, counter=witness.provider)


@pytest.mark.parametrize("change", [
    "missing_structure", "unknown_field", "changed_structure_mode", "changed_length_strategy",
    "invalid_forced_index", "changed_relaxation_count", "changed_relaxation_retained_end",
    "changed_initial_reason", "initial_score_outside_tolerance",
])
def test_rehashed_structure_changes_cannot_claim_successful_baseline_replay(
    structured_replay_witness, tmp_path, change,
):
    """即使重算SHA，缺失或伪造结构证据也不得获得baseline_reproduction=passed。"""
    witness = structured_replay_witness
    _rewrite_structure(witness, change)
    output = tmp_path / "rejected-comparison"
    with pytest.raises(ValueError):
        _compare(witness, output)
    assert not output.exists()
    assert witness.provider.model_call_count == witness.model_call_count
    assert _cache_hashes(witness.cache) == witness.original_cache_hashes


@pytest.mark.parametrize("roundoff", [False, True])
def test_valid_structure_replays_readonly_and_candidate_keeps_recomputed_evidence(
    structured_replay_witness, tmp_path, roundoff,
):
    """合法结构通过；仅初始评分允许既定末位差，候选仍保存本次真实重算结构。"""
    witness = structured_replay_witness
    if roundoff:
        _rewrite_structure(witness, "initial_score_roundoff")
    output = tmp_path / "accepted-comparison"
    manifest = _compare(witness, output)
    assert manifest["baseline_reproduction"] == "passed"
    assert manifest["actual_embedding_model_calls"] == 0
    cases = read_rows(output / f"cases-{witness.settings.fingerprint}.jsonl")
    assert [row["source_post_id"] for row in cases] == [row["source_post_id"] for row in witness.original_traces]
    for case, trace in zip(cases, witness.original_traces, strict=True):
        assert case["structure"] == trace["structure"]
    assert witness.provider.model_call_count == witness.model_call_count
    assert _cache_hashes(witness.cache) == witness.original_cache_hashes
