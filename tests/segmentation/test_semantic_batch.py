"""同步语义批处理的只读来源、身份、缓存重跑与失败隔离；不加载真实模型。"""

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import stat

import pytest

from test_semantic_api import SyntheticProvider
from tourism_ugc_study.segmentation.config import ResearchInput
from tourism_ugc_study.segmentation.embedding import EmbeddingError
from tourism_ugc_study.segmentation.semantic_batch import run_semantic_batch
from tourism_ugc_study.segmentation.semantic_config import EmbeddingConfig, CoherentSemanticConfig, SemanticSettings
from tourism_ugc_study.segmentation.source import file_sha256


def make_input(root: Path, rows=None):
    """新建仅测试用配对SQLite，成员含keep及排除，manifest绑定真实摘要。"""
    rows = rows or [(1, "甲" * 15 + "。" + "乙" * 15 + "。", "keep"),
                    (2, "丁" * 15 + "。" + "丙" * 15 + "。", "keep"),
                    (3, '{"ops":broken}', "keep"), (4, None, "keep"),
                    (5, "不能参与本轮切分。", "exclude")]
    source, members, manifest_path = root / "source.sqlite", root / "members.sqlite", root / "candidate-manifest.json"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE web_posts(id INTEGER PRIMARY KEY,platform_key TEXT,title TEXT,content_text TEXT)")
        connection.executemany("INSERT INTO web_posts VALUES (?,'xhs',NULL,?)", [(i, body) for i, body, _ in rows])
    counts = Counter(decision for _, _, decision in rows)
    manifest = {
        "artifact_kind": "full-research-cleaning-candidates",
        "status": "CANDIDATES_READY_AWAITING_HUMAN_REVIEW", "database_filename": members.name,
        "source_snapshot_sha256": file_sha256(source), "round_manifest_sha256": "b" * 64,
        "round_name": "synthetic-semantic-round", "record_count": len(rows),
        "decision_counts": {name: counts[name] for name in ("keep", "exclude", "manual_review")},
    }
    with sqlite3.connect(members) as connection:
        connection.executescript("""
            CREATE TABLE cleaning_candidates(source_post_id INTEGER,source_version INTEGER,
                platform_key TEXT,title TEXT,body TEXT,decision TEXT);
            CREATE TABLE round_metadata(key TEXT PRIMARY KEY,value_json TEXT);
        """)
        connection.executemany("INSERT INTO cleaning_candidates VALUES (?,1,'xhs','',?,?)", rows)
        connection.executemany("INSERT INTO round_metadata VALUES (?,?)", [
            (key, json.dumps(manifest[key])) for key in
            ("status", "round_name", "source_snapshot_sha256", "round_manifest_sha256")])
    manifest["database_sha256"] = file_sha256(members)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return ResearchInput(members, source, manifest_path, file_sha256(manifest_path))


def settings():
    """显式CPU测试身份，避免使用正式默认MPS路径或权重加载。"""
    return SemanticSettings(CoherentSemanticConfig(), EmbeddingConfig(device="cpu"))


def read_rows(path):
    """读取私有合成JSONL，检查持久化契约而非内存返回值。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_full_run_preserves_sources_and_separates_preflight_and_post_statistics(tmp_path):
    """源库只读、只切keep；预检真实调用独立于语料计数，损坏正文保留failed。"""
    config = make_input(tmp_path)
    sources = (config.source_db, config.membership_db, config.candidate_manifest)
    before = [file_sha256(path) for path in sources]
    provider = SyntheticProvider(settings().embedding)
    output = tmp_path / "run"
    progress = []
    manifest = run_semantic_batch(config, output, settings=settings(), code_version="synthetic-test",
                                  provider=provider, progress=lambda seen, total: progress.append((seen, total)))
    rows = read_rows(output / "posts.jsonl")
    trace = read_rows(output / "semantic-trace.jsonl")
    persisted = json.loads((output / "manifest.json").read_text())
    assert persisted == manifest
    assert manifest["scope"] == "full_corpus" and manifest["selected_post_ids"] is None
    assert manifest["status"] == "completed_with_failures"
    assert manifest["counts"]["post_count"] == 4 and manifest["counts"]["failed_count"] == 1
    assert manifest["counts"]["completed_count"] == 3 and manifest["counts"]["empty_count"] == 1
    assert [row["source_post_id"] for row in rows] == [1, 2, 3, 4]
    assert [row["source_post_id"] for row in trace] == [1, 2, 3, 4]
    assert rows[2]["text_segments"] is None and rows[3]["text_segments"] == []
    assert rows[3]["original_body"] is None and rows[3]["body_was_null"] is True
    assert provider.model_call_count == manifest["counts"]["model_call_count"] + 1
    assert provider.calls[0] == ("海边散步，海风很轻。", "公交车站就在路口。")
    assert provider.asset_checks >= 1
    assert file_sha256(output / "posts.jsonl") == manifest["posts_sha256"]
    assert file_sha256(output / "semantic-trace.jsonl") == manifest["trace_sha256"]
    assert Path(manifest["cache_directory"]) == tmp_path / ".semantic-vector-cache"
    assert [file_sha256(path) for path in sources] == before
    assert not list(tmp_path.glob("*.sqlite-*"))
    assert manifest["source_database_write_count"] == manifest["membership_database_write_count"] == 0
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]
    # 私有运行证据现在包含可重放的代码目录；目录必须可遍历，文件仍不可公开读取。
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE((output / "code-snapshot").stat().st_mode) == 0o700
    for path in output.rglob("*"):
        if path.is_file():
            assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_explicit_members_are_development_even_if_they_cover_all_keep(tmp_path):
    """显式成员顺序规范化为来源post顺序；覆盖全部成员也不能冒充正式全量。"""
    config = make_input(tmp_path)
    manifest = run_semantic_batch(config, tmp_path / "development", settings=settings(), code_version="test",
                                  provider=SyntheticProvider(), post_ids=(4, 2, 3, 1), seed=20261011)
    assert manifest["scope"] == "development" and manifest["selected_post_ids"] == [1, 2, 3, 4]
    assert manifest["sampling"] == "explicit_development_members" and manifest["seed"] == 20261011
    assert manifest["input"]["record_count"] == 4


@pytest.mark.parametrize("post_ids", [(), (1, 1), (True,), (0,), (-1,), (1.0,)])
def test_invalid_development_members_are_rejected_before_preflight(tmp_path, post_ids):
    """空、重复或非法成员标识不能生成有歧义的开发清单。"""
    config = make_input(tmp_path)
    provider = SyntheticProvider()
    with pytest.raises(ValueError, match="invalid_development_members"):
        run_semantic_batch(config, tmp_path / "invalid", settings=settings(), code_version="test",
                           provider=provider, post_ids=post_ids)
    assert provider.calls == [] and not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("missing_id", [5, 999])
def test_missing_or_excluded_development_member_fails_instead_of_silently_shrinking(tmp_path, missing_id):
    """指定排除成员或不存在成员时整轮失败，不能少一帖后声明清单完整。"""
    config = make_input(tmp_path)
    output = tmp_path / "missing"
    with pytest.raises(ValueError, match="development_members_not_found"):
        run_semantic_batch(config, output, settings=settings(), code_version="test",
                           provider=SyntheticProvider(), post_ids=(1, missing_id))
    assert not output.exists() and not list(tmp_path.glob(".missing-*"))
    attempt = json.loads((tmp_path / "missing.attempt.json").read_text())
    assert attempt["status"] == "failed" and attempt["counts"]["post_count"] == 1


def test_failed_model_attempt_preserves_cache_and_clean_retry_reuses_it(tmp_path):
    """第二帖模型失败只清临时运行包；首帖合格向量在新attempt中可复用。"""
    config = make_input(tmp_path)
    provider = SyntheticProvider(fail_on_call=3)
    output = tmp_path / "failed-model"
    with pytest.raises(EmbeddingError, match="synthetic_model_failure"):
        run_semantic_batch(config, output, settings=settings(), code_version="test", provider=provider,
                           post_ids=(1, 2))
    assert not output.exists() and not list(tmp_path.glob(".failed-model-*"))
    evidence = (tmp_path / "failed-model.attempt.json").read_text()
    assert "private-raw-marker" not in evidence
    assert json.loads(evidence)["status"] == "failed" and json.loads(evidence)["counts"]["post_count"] == 1
    vectors = list((tmp_path / ".semantic-vector-cache").glob("*/*/vector.npy"))
    assert len(vectors) == 2
    retry = SyntheticProvider()
    manifest = run_semantic_batch(config, tmp_path / "retry", settings=settings(), code_version="test",
                                  provider=retry, post_ids=(1, 2))
    assert manifest["status"] == "completed" and manifest["counts"]["cache_hit_count"] == 2
    assert manifest["counts"]["model_call_count"] == 1 and retry.model_call_count == 2
    assert provider.calls[1] not in retry.calls
    assert json.loads(evidence)["status"] == "failed"


def test_repeated_successful_run_always_preflights_but_does_not_claim_cache_as_inference(tmp_path):
    """复跑全量语料命中缓存，仍必须实际预检且原包不得覆盖。"""
    config = make_input(tmp_path, rows=[(1, "甲。乙。", "keep"), (2, "甲。乙。", "keep")])
    first_provider, second_provider = SyntheticProvider(), SyntheticProvider()
    first_output = tmp_path / "first"
    first = run_semantic_batch(config, first_output, settings=settings(), code_version="test", provider=first_provider)
    before = file_sha256(first_output / "manifest.json")
    second = run_semantic_batch(config, tmp_path / "second", settings=settings(), code_version="test", provider=second_provider)
    assert first["counts"]["model_call_count"] == 1 and first["counts"]["cache_hit_count"] == 2
    assert second["counts"]["model_call_count"] == 0 and second["counts"]["cache_hit_count"] == 4
    assert first["counts"]["embedded_unit_count"] == second["counts"]["embedded_unit_count"] == 4
    assert len(second_provider.calls) == 1
    with pytest.raises(FileExistsError):
        run_semantic_batch(config, first_output, settings=settings(), code_version="test", provider=SyntheticProvider())
    assert file_sha256(first_output / "manifest.json") == before


def test_all_single_unit_corpus_still_performs_real_preflight(tmp_path):
    """逐帖无需向量不能跳过整个批次的模型可用性检查。"""
    config = make_input(tmp_path, rows=[(1, "海边真好看。", "keep"), (2, None, "keep")])
    provider = SyntheticProvider()
    manifest = run_semantic_batch(config, tmp_path / "single", settings=settings(), code_version="test", provider=provider)
    assert len(provider.calls) == 1 and manifest["counts"]["model_call_count"] == 0
    assert manifest["counts"]["embedded_unit_count"] == 0


def test_preflight_failure_does_not_create_a_run_even_for_single_unit_corpus(tmp_path):
    """不能用本批无需逐帖embedding为由绕过真实预检失败。"""
    config = make_input(tmp_path, rows=[(1, "海边真好看。", "keep")])
    with pytest.raises(EmbeddingError, match="synthetic_model_failure"):
        run_semantic_batch(config, tmp_path / "preflight", settings=settings(), code_version="test",
                           provider=SyntheticProvider(fail_on_call=1))
    assert not (tmp_path / "preflight").exists()
    attempt = json.loads((tmp_path / "preflight.attempt.json").read_text())
    assert attempt["status"] == "failed" and not attempt["counts"]


def test_provider_identity_drift_is_detected_even_if_no_post_needs_embedding(tmp_path):
    """预检后只有单单元帖子时，运行结束仍必须复验实际模型身份。"""
    config = make_input(tmp_path, rows=[(1, "海边真好看。", "keep")])
    provider = SyntheticProvider()

    def change_identity(seen, total):
        """在最后一条回调中模拟配置漂移，使结束阶段必须检查。"""
        provider.identity["device"] = "mps"

    with pytest.raises(ValueError, match="runtime_or_model_changed_during_run"):
        run_semantic_batch(config, tmp_path / "model-drift", settings=settings(), code_version="test",
                           provider=provider, progress=change_identity)
    assert not (tmp_path / "model-drift").exists()


@pytest.mark.parametrize("field,value", [("device", "mps"), ("model_revision", "0" * 40),
                                         ("instruction", "changed"), ("dimension", 2), ("truncation", True)])
def test_provider_identity_must_match_declared_settings_before_preflight(tmp_path, field, value):
    """声明配置与实际provider矛盾不能成为貌似可复现的成功清单。"""
    config = make_input(tmp_path)
    provider = SyntheticProvider()
    provider.identity[field] = value
    output = tmp_path / "conflict"
    with pytest.raises(ValueError, match="embedding_identity_settings_mismatch"):
        run_semantic_batch(config, output, settings=settings(), code_version="test", provider=provider)
    assert provider.calls == [] and not output.exists()
    assert json.loads((tmp_path / "conflict.attempt.json").read_text())["status"] == "failed"


def test_changed_source_manifest_during_run_prevents_publication(tmp_path):
    """运行中来源身份漂移须清理临时运行，不能发布混合身份的完整manifest。"""
    config = make_input(tmp_path)

    def change_manifest(seen, total):
        """仅修改合成测试manifest，制造封存身份漂移。"""
        if seen == 1:
            with config.candidate_manifest.open("a") as stream:
                stream.write(" ")

    with pytest.raises(ValueError, match="运行中"):
        run_semantic_batch(config, tmp_path / "drift", settings=settings(), code_version="test",
                           provider=SyntheticProvider(), progress=change_manifest)
    assert not (tmp_path / "drift").exists() and not list(tmp_path.glob(".drift-*"))


def test_bad_source_binding_is_rejected_and_attempt_is_recorded(tmp_path):
    """预检成功仍不能绕过来源封存摘要，失败证据与成功运行包分开。"""
    config = replace(make_input(tmp_path), candidate_manifest_sha256="0" * 64)
    with pytest.raises(ValueError, match="摘要"):
        run_semantic_batch(config, tmp_path / "bad-source", settings=settings(), code_version="test",
                           provider=SyntheticProvider())
    assert not (tmp_path / "bad-source").exists()
    assert json.loads((tmp_path / "bad-source.attempt.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("inside", [False, True])
def test_cache_cannot_create_run_output_before_atomic_publication(tmp_path, inside):
    """缓存与运行包生命周期独立；禁止缓存提前建立目标目录破坏原子发布。"""
    config = make_input(tmp_path)
    output = tmp_path / "run"
    cache_dir = output / "cache" if inside else output
    provider = SyntheticProvider()
    with pytest.raises(ValueError, match="cache_must_be_outside_run_output"):
        run_semantic_batch(config, output, settings=settings(), code_version="test",
                           provider=provider, cache_dir=cache_dir)
    assert not output.exists() and provider.calls == []


def _cache_only_fixture(tmp_path):
    """先用合成模型完成一次正式全量，提供只读重算所需的来源运行与缓存。"""
    config = make_input(tmp_path, rows=[(1, "甲" * 15 + "。" + "乙" * 15 + "。", "keep"),
                                        (2, "丁" * 15 + "。" + "丙" * 15 + "。", "keep"),
                                        (3, "海边真好看。", "keep")])
    source = run_semantic_batch(config, tmp_path / "model-run", settings=settings(), code_version="test",
                                provider=SyntheticProvider())
    assert source["status"] == "completed" and source["vector_source"] == {"mode": "model"}
    return config


def test_cache_only_run_reproduces_segments_without_any_inference(tmp_path):
    """只读缓存重算与来源运行切点一致，provider从不推理且清单声明来源运行。"""
    config = _cache_only_fixture(tmp_path)
    provider = SyntheticProvider()
    manifest = run_semantic_batch(config, tmp_path / "cache-run", settings=settings(), code_version="test",
                                  provider=provider, cache_dir=tmp_path / ".semantic-vector-cache",
                                  cache_only_source=tmp_path / "model-run")
    assert provider.calls == [] and manifest["status"] == "completed"
    assert manifest["counts"]["model_call_count"] == 0 and manifest["preflight_model_call_count"] == 0
    assert manifest["counts"]["cache_hit_count"] == manifest["counts"]["embedded_unit_count"] == 4
    assert manifest["vector_source"] == {
        "mode": "cache_only", "identity_source_manifest": str(tmp_path / "model-run" / "manifest.json"),
        "identity_source_manifest_sha256": file_sha256(tmp_path / "model-run" / "manifest.json")}
    segments = lambda run: [row["text_segments"] for row in read_rows(tmp_path / run / "posts.jsonl")]
    assert segments("cache-run") == segments("model-run")


def test_cache_only_run_fails_on_missing_vector_instead_of_recomputing(tmp_path):
    """缺任一缓存向量即整轮失败、不发布，也不借provider补算。"""
    config = _cache_only_fixture(tmp_path)
    vectors = sorted((tmp_path / ".semantic-vector-cache").glob("*/*/vector.npy"))
    __import__("shutil").rmtree(vectors[0].parent)
    provider = SyntheticProvider()
    with pytest.raises(EmbeddingError, match="cache_entry_missing"):
        run_semantic_batch(config, tmp_path / "cache-run", settings=settings(), code_version="test",
                           provider=provider, cache_dir=tmp_path / ".semantic-vector-cache",
                           cache_only_source=tmp_path / "model-run")
    assert provider.calls == [] and not (tmp_path / "cache-run").exists()
    attempt = json.loads((tmp_path / "cache-run.attempt.json").read_text())
    assert attempt["status"] == "failed" and attempt["error_code"] == "cache_entry_missing"


def test_cache_only_run_rejects_identity_other_than_source_run(tmp_path):
    """provider身份须与来源运行记录逐字节一致，否则缓存命名空间不是同一批向量。"""
    config = _cache_only_fixture(tmp_path)
    provider = SyntheticProvider()
    provider.identity["assets_sha256"] = {"synthetic-fixture": "c" * 64}
    with pytest.raises(ValueError, match="cache_only_identity_mismatch"):
        run_semantic_batch(config, tmp_path / "cache-run", settings=settings(), code_version="test",
                           provider=provider, cache_dir=tmp_path / ".semantic-vector-cache",
                           cache_only_source=tmp_path / "model-run")
    assert not (tmp_path / "cache-run").exists()


def test_cache_only_source_must_be_complete_full_run(tmp_path):
    """开发样本或被改动的来源清单不能作为只读重算的身份依据。"""
    config = _cache_only_fixture(tmp_path)
    run_semantic_batch(config, tmp_path / "development", settings=settings(), code_version="test",
                       provider=SyntheticProvider(), post_ids=(1, 2))
    with pytest.raises(ValueError, match="cache_only_source_run_invalid"):
        run_semantic_batch(config, tmp_path / "cache-run", settings=settings(), code_version="test",
                           provider=SyntheticProvider(), cache_dir=tmp_path / ".semantic-vector-cache",
                           cache_only_source=tmp_path / "development")
    assert not (tmp_path / "cache-run.attempt.json").exists()
