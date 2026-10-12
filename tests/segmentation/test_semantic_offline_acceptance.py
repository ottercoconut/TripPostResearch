"""独立验证冻结开发输入的隔离边界，不把可读包等同于已核验完整源库。

使用合成来源与固定向量；通过移走数据库、改变冻结文件及制造本地证据冲突，
检验离线能力和回传核验是否真实发生。测试不下载模型、不读取正式语料。
"""

import json
import shutil
import sqlite3
import subprocess
import sys
import textwrap

import pytest

from test_semantic_acceptance import replay_witness
from test_semantic_runtime import normal_sample
from tourism_ugc_study.segmentation import semantic_runtime
from tourism_ugc_study.segmentation.importer import import_segmentation_run
from tourism_ugc_study.segmentation.semantic_batch import run_semantic_batch
from tourism_ugc_study.segmentation.semantic_comparison import compare_semantic_configs
from tourism_ugc_study.segmentation.semantic_development_input import load_frozen_development_input
from tourism_ugc_study.segmentation.semantic_evaluation import (
    export_frozen_development_input,
    verify_frozen_development_run,
)
from tourism_ugc_study.segmentation.source import file_sha256


@pytest.fixture
def offline_witness(replay_witness, tmp_path):
    """从已有完整来源导出，保留原证据以便离线后在本地独立回验。"""
    source, members, old_run, cache, settings, provider = replay_witness
    # 补全发布门所需的合成身份，使拒绝断言确实检验来源隔离，而非先被缺字段拦下。
    provider.identity["assets_sha256"] = {name: "a" * 64 for name in (
        "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors")}
    provider.identity["dependencies"] = {name: "synthetic" for name in (
        "torch", "transformers", "tokenizers", "huggingface-hub", "numpy")}
    prior = json.loads((old_run / "manifest.json").read_text())
    old_run = tmp_path / "complete-synthetic-baseline"
    run_semantic_batch(source, old_run, settings=settings, code_version="offline-acceptance",
        cache_dir=cache, provider=provider, post_ids=tuple(prior["selected_post_ids"]), seed=prior["seed"])
    package = tmp_path / "frozen-input"
    export_frozen_development_input(source, members, package)
    frozen = load_frozen_development_input(package / "input-manifest.json")
    return source, members, old_run, cache, settings, provider, package, frozen


def test_offline_execution_survives_absent_databases_but_never_becomes_full_corpus(
    offline_witness, tmp_path, monkeypatch,
):
    """数据库确实不可达时仍切同一成员；自动成员不意味着全量发布资格。"""
    source, members, old_run, cache, settings, provider, package, frozen = offline_witness
    assert {p.name for p in package.iterdir()} == {"manifest.json", "cases.jsonl", "input-manifest.json"}
    for name in ("manifest.json", "cases.jsonl"):
        assert (package / name).read_bytes() == (members.parent / name).read_bytes()

    original = json.loads((old_run / "manifest.json").read_text())
    moved = []
    output = tmp_path / "offline-run"
    try:
        for path in (source.source_db, source.membership_db, source.candidate_manifest):
            hidden = path.with_name(path.name + ".unavailable")
            path.rename(hidden)
            moved.append((path, hidden))
        with monkeypatch.context() as patched:
            def forbid_database(*args, **kwargs):
                pytest.fail("离线开发运行仍依赖数据库连接")
            patched.setattr(sqlite3, "connect", forbid_database)
            manifest = run_semantic_batch(frozen, output, settings=settings,
                code_version="offline-acceptance", cache_dir=cache, provider=provider)
    finally:
        for path, hidden in moved:
            hidden.rename(path)

    assert manifest["scope"] == "development"
    assert manifest["input"]["dataset_kind"] == "frozen_development_cases"
    assert manifest["input"]["parent_source_identity"] == original["input"]
    assert manifest["selected_post_ids"] == original["selected_post_ids"]
    assert manifest["counts"]["model_call_count"] == 0
    assert manifest["counts"]["cache_hit_count"] == original["counts"]["embedded_unit_count"]
    old_posts = [json.loads(line) for line in (old_run / "posts.jsonl").read_text().splitlines()]
    new_posts = [json.loads(line) for line in (output / "posts.jsonl").read_text().splitlines()]
    assert [p["text_segments"] for p in new_posts] == [p["text_segments"] for p in old_posts]
    bundle = tmp_path / "forbidden-published-bundle"
    with pytest.raises(ValueError):
        import_segmentation_run(output, bundle, source, code_version="offline-acceptance")
    assert not bundle.exists()

    # 该合成样本恰好覆盖全部keep，仍不能仅改scope和成员标记冒充全库运行。
    forged = tmp_path / "renamed-as-full-run"
    shutil.copytree(output, forged)
    changed = dict(manifest, scope="full_corpus", selected_post_ids=None,
                   sampling="all_keep_ordered_by_source_post_id")
    (forged / "manifest.json").write_text(json.dumps(changed, ensure_ascii=False))
    with pytest.raises(ValueError, match="导入目标来源与切分运行来源不一致"):
        import_segmentation_run(forged, bundle, source, code_version="offline-acceptance")
    assert not bundle.exists()


def test_returned_run_requires_local_source_reconciliation_before_cached_comparison(
    offline_witness, tmp_path, monkeypatch,
):
    """包内运行可用后仍逐帖对回本地来源；比较不能为补证据调用embedding。"""
    source, members, _, cache, settings, provider, package, frozen = offline_witness
    run = tmp_path / "returned-run"
    manifest = run_semantic_batch(frozen, run, settings=settings,
        code_version="offline-acceptance", cache_dir=cache, provider=provider)

    def forbid_embedding(*args, **kwargs):
        pytest.fail("回传核验或缓存比较调用了embedding")
    monkeypatch.setattr(provider, "encode", forbid_embedding)
    verification = verify_frozen_development_run(source, run, tmp_path / "verified",
        frozen_input=package / "input-manifest.json", members=members, counter=provider)
    assert verification["status"] == "completed"
    compared = compare_semantic_configs(source, members, run, tmp_path / "compared",
        frozen_input=frozen, cache_dir=cache, counter=provider)
    assert compared["input"] == manifest["input"]
    assert compared["baseline_reproduction"] == "passed"
    assert compared["actual_embedding_model_calls"] == 0

    # 同步外层摘要，使冲突来自正文／来源证据，而不是坏JSON或旧文件摘要。
    sample_manifest = json.loads(members.read_text())
    cases_path = members.parent / "cases.jsonl"
    cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
    cases[0]["source_version"] += 1
    cases_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases))
    sample_manifest["cases_sha256"] = file_sha256(cases_path)
    members.write_text(json.dumps(sample_manifest, ensure_ascii=False))
    rejected = tmp_path / "must-not-verify"
    with pytest.raises(ValueError):
        verify_frozen_development_run(source, run, rejected,
            frozen_input=package / "input-manifest.json", members=members, counter=provider)
    assert not rejected.exists()


def test_input_changed_during_offline_run_prevents_publication_without_erasing_cache(
    offline_witness, tmp_path,
):
    """已读入内存的正文不能掩盖包被替换；失败必须留下可复用缓存。"""
    _, _, _, cache, settings, provider, package, frozen = offline_witness
    original_cache = {p: file_sha256(p) for p in cache.rglob("*") if p.is_file()}
    output = tmp_path / "changed-during-run"

    def mutate_input(done, total):
        if done == 1:
            with (package / "cases.jsonl").open("a") as stream:
                stream.write("\n")

    with pytest.raises(ValueError):
        run_semantic_batch(frozen, output, settings=settings, code_version="offline-acceptance",
            cache_dir=cache, provider=provider, progress=mutate_input)
    assert not output.exists()
    assert {p: file_sha256(p) for p in original_cache} == original_cache


def test_explicit_member_subset_cannot_shrink_a_frozen_offline_input(offline_witness, tmp_path):
    """包成员是本轮比较基底，不允许在调用时悄悄删除压力或难例。"""
    _, _, old_run, cache, settings, provider, _, frozen = offline_witness
    ids = json.loads((old_run / "manifest.json").read_text())["selected_post_ids"]
    output = tmp_path / "subset-run"
    with pytest.raises(ValueError):
        run_semantic_batch(frozen, output, settings=settings, code_version="offline-acceptance",
            cache_dir=cache, provider=provider, post_ids=tuple(ids[:-1]))
    assert not output.exists()


def test_final_progress_input_change_is_checked_after_worker_exit(
    offline_witness, tmp_path, monkeypatch,
):
    """真实worker正常退出后，末次用户回调改包也不能越过父进程发布门。"""
    _, _, _, cache, settings, _, package, frozen = offline_witness
    original_popen = subprocess.Popen
    worker = textwrap.dedent("""
        import json, pathlib, sys
        request = pathlib.Path(sys.argv[1])
        config = json.loads(request.read_text())
        output = pathlib.Path(config['output'])
        output.mkdir()
        (output / 'manifest.json').write_text(json.dumps({'status': 'completed',
            'counts': {'post_count': len(config['post_ids'])},
            'code_snapshot': config['code_snapshot'], 'config_sha256': sys.argv[2]}))
        (request.parent / 'worker-result.json').write_text(json.dumps({
            'process_peak_rss_bytes': 1024}))
    """)
    processes = []

    def launch(command, **kwargs):
        process = original_popen([sys.executable, "-c", worker, command[-1], settings.fingerprint], **kwargs)
        processes.append(process)
        return process

    def change_input(done, total):
        with (package / "input-manifest.json").open("a") as stream:
            stream.write(" ")

    monkeypatch.setattr(semantic_runtime.subprocess, "Popen", launch)
    monkeypatch.setattr(semantic_runtime, "_resource_sample", normal_sample)
    monkeypatch.setattr(semantic_runtime.platform, "system", lambda: "Darwin")
    output = tmp_path / "changed-in-final-progress"
    with pytest.raises(ValueError, match="frozen_development_input_changed"):
        semantic_runtime.run_guarded_semantic_batch(frozen, output, settings=settings,
            code_version="offline-acceptance", cache_dir=cache, progress=change_input,
            limits=semantic_runtime.SemanticRuntimeLimits(sampling_seconds=0.02))
    assert processes[0].poll() == 0
    assert not output.exists()
