"""真实子进程停止、资源读数故障、私有模型异常和代码漂移的边界回归。"""

from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import textwrap

import pytest

from test_semantic_api import SyntheticProvider
from test_semantic_batch import make_input, settings
from tourism_ugc_study.segmentation import semantic_batch, semantic_runtime
from tourism_ugc_study.segmentation.artifacts import write_json


RAW_MARKER = "private-research-body-must-stay-in-log"


def install_waiting_worker(monkeypatch):
    """以真实长时子进程替代模型worker，创建本轮临时包和已完成缓存后等待。"""
    original = subprocess.Popen
    launched = []
    worker = textwrap.dedent("""
        import json, os, pathlib, sys, time
        request = pathlib.Path(sys.argv[1])
        root = request.parent
        config = json.loads(request.read_text())
        staging = root / '.completed-run-stranded'
        staging.mkdir()
        (staging / 'posts.jsonl').write_text('incomplete-private-package')
        cache = pathlib.Path(config['cache_dir'])
        cache.mkdir(parents=True)
        (cache / 'validated-cache-sentinel').write_text('preserve')
        state = root / 'worker-state.json'
        pending = root / 'worker-state.pending'
        pending.write_text(json.dumps({'phase': 'corpus_processing',
            'active_source_post_id': 81, 'counts': {'post_count': 1}}))
        pending.chmod(0o600)
        pending.replace(state)
        print('private-worker-log-marker', flush=True)
        while True:
            time.sleep(0.01)
    """)

    def launch(command, **kwargs):
        """仍让父进程wait／kill实际OS进程，不用假的poll掩盖停止失败。"""
        process = original([sys.executable, "-c", worker, command[-1]], **kwargs)
        launched.append(process)
        return process

    monkeypatch.setattr(semantic_runtime.subprocess, "Popen", launch)
    return launched


def normal_sample(pid):
    """固定可用读数以隔离时间门；不读取模型、不靠当前整机负载决定断言。"""
    return {"rss_bytes": 1024 if pid is not None else None,
            "swap_bytes": 100, "memory_pressure_level": 1}


def test_single_post_stall_is_terminated_without_publication_and_cache_survives(tmp_path, monkeypatch):
    """同帖持续存在不能重置post计时；超时后真实进程回收且无半包可导入。"""
    source = make_input(tmp_path)
    processes = install_waiting_worker(monkeypatch)
    monkeypatch.setattr(semantic_runtime, "_resource_sample", normal_sample)
    output, cache = tmp_path / "stalled", tmp_path / "vectors"
    with pytest.raises(semantic_runtime.SemanticRuntimeError, match="semantic_phase_timeout"):
        semantic_runtime.run_guarded_semantic_batch(source, output, settings=settings(), code_version="test",
            cache_dir=cache, post_ids=(1, 2), limits=semantic_runtime.SemanticRuntimeLimits(
                load_seconds=2, post_seconds=0.08, total_seconds=5, sampling_seconds=0.02))
    assert processes[0].poll() is not None
    with pytest.raises(ProcessLookupError):
        os.kill(processes[0].pid, 0)
    assert not output.exists() and not list((tmp_path / "stalled.runtime").glob(".completed-run-*"))
    assert (cache / "validated-cache-sentinel").read_text() == "preserve"
    attempt = json.loads((tmp_path / "stalled.attempt.json").read_text())
    assert attempt["status"] == "interrupted" and attempt["active_source_post_id"] == 81
    assert attempt["error_code"] == "semantic_phase_timeout"
    assert attempt["code_snapshot"]["executed_from_snapshot"] is True
    assert "private-worker-log-marker" in (tmp_path / "stalled.runtime/worker.log").read_text()
    assert stat.S_IMODE((tmp_path / "stalled.runtime").stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "stalled.runtime/code-snapshot").stat().st_mode) == 0o700
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600
               for p in (tmp_path / "stalled.runtime").rglob("*") if p.is_file())


@pytest.mark.parametrize("changed,reason", [({"rss_bytes": 8 * 1024**3 + 1}, "rss_limit_exceeded"),
    ({"swap_bytes": 100 + 512 * 1024**2 + 1}, "additional_swap_limit_exceeded"),
    ({"memory_pressure_level": 4}, "critical_memory_pressure")])
def test_resource_limit_stops_live_worker_and_retains_validated_cache(tmp_path, monkeypatch, changed, reason):
    """三个固定资源门均实际停止，触发门的最后采样及RSS峰值须留在摘要。"""
    source = make_input(tmp_path)
    processes = install_waiting_worker(monkeypatch)
    output, cache = tmp_path / "protected", tmp_path / "vectors"

    def sample(pid):
        """等缓存与状态均存在后超门，验证中止不会连缓存一起清除。"""
        value = normal_sample(pid)
        if pid is not None and (tmp_path / "protected.runtime/worker-state.json").exists():
            value.update(changed)
        return value

    monkeypatch.setattr(semantic_runtime, "_resource_sample", sample)
    with pytest.raises(semantic_runtime.SemanticRuntimeError, match=reason):
        semantic_runtime.run_guarded_semantic_batch(source, output, settings=settings(), code_version="test",
            cache_dir=cache, limits=semantic_runtime.SemanticRuntimeLimits(sampling_seconds=0.02))
    assert processes[0].poll() is not None and not output.exists()
    assert (cache / "validated-cache-sentinel").exists()
    attempt = json.loads((tmp_path / "protected.attempt.json").read_text())
    observations = [json.loads(line) for line in (tmp_path / "protected.runtime/resources.jsonl").read_text().splitlines()]
    triggering_sample = {**normal_sample(processes[0].pid), **changed}
    assert attempt["status"] == "interrupted" and attempt["error_code"] == reason
    assert attempt["peak_sampled_rss_bytes"] == max(sample["rss_bytes"] for sample in observations)
    assert attempt["last_resources"] == triggering_sample
    assert all(observations[-1][key] == value for key, value in triggering_sample.items())


@pytest.mark.parametrize("changed,reason", [
    ({"swap_bytes": 100 + 512 * 1024**2 + 1}, "additional_swap_limit_exceeded"),
    ({"memory_pressure_level": 4}, "critical_memory_pressure")])
def test_resource_limit_after_worker_exit_retains_final_observation(tmp_path, monkeypatch, changed, reason):
    """退出后的系统复核超门仍留存final证据，并删除worker包而不发布。"""
    source = make_input(tmp_path)
    original = subprocess.Popen
    worker = textwrap.dedent("""
        import json, pathlib, sys
        config = json.loads(pathlib.Path(sys.argv[1]).read_text())
        output = pathlib.Path(config['output'])
        output.mkdir()
        (output / 'posts.jsonl').write_text('private-worker-package')
    """)
    monkeypatch.setattr(semantic_runtime.subprocess, "Popen",
        lambda command, **kwargs: original([sys.executable, "-c", worker, command[-1]], **kwargs))
    system_observations = []

    def sample(pid):
        """启动前读数正常，仅在worker退出后的无PID复核中触发固定资源门。"""
        value = normal_sample(pid)
        if pid is None:
            if system_observations:
                value.update(changed)
            system_observations.append(value)
        return value

    monkeypatch.setattr(semantic_runtime, "_resource_sample", sample)
    with pytest.raises(semantic_runtime.SemanticRuntimeError, match=reason):
        semantic_runtime.run_guarded_semantic_batch(source, tmp_path / "final", settings=settings(), code_version="test",
            limits=semantic_runtime.SemanticRuntimeLimits(sampling_seconds=0.02))
    assert len(system_observations) == 2
    assert not (tmp_path / "final").exists() and not (tmp_path / "final.runtime/completed-run").exists()
    attempt = json.loads((tmp_path / "final.attempt.json").read_text())
    assert attempt["status"] == "interrupted" and attempt["error_code"] == reason
    assert attempt["final_resources"] == system_observations[-1] == {**normal_sample(None), **changed}


def test_missing_resource_observation_is_failure_not_healthy_fallback(tmp_path, monkeypatch):
    """swap读数故障不能用0替代继续；未启动模型也必须有明确attempt终态。"""
    source = make_input(tmp_path)
    monkeypatch.setattr(semantic_runtime, "_swap_bytes", lambda: (_ for _ in ()).throw(RuntimeError(RAW_MARKER)))
    with pytest.raises(semantic_runtime.SemanticRuntimeError, match="resource_observation_failed"):
        semantic_runtime.run_guarded_semantic_batch(source, tmp_path / "unobserved", settings=settings(), code_version="test")
    evidence = (tmp_path / "unobserved.attempt.json").read_text()
    assert RAW_MARKER not in evidence and json.loads(evidence)["status"] == "interrupted"
    assert not (tmp_path / "unobserved").exists()


def test_unknown_pressure_flag_is_not_treated_as_normal(monkeypatch):
    """未识别的系统压力值不能落入normal，否则资源保护会被系统变化关闭。"""
    monkeypatch.setattr(semantic_runtime, "_swap_bytes", lambda: 0)
    monkeypatch.setattr(semantic_runtime, "_pressure_level", lambda: 0)
    with pytest.raises(semantic_runtime.SemanticRuntimeError, match="resource_observation_failed"):
        semantic_runtime._resource_sample(None)


def test_resource_sampling_failure_during_live_post_stops_worker(tmp_path, monkeypatch):
    """运行中失去读数也必须停止，不能把worker还存活当资源正常。"""
    source = make_input(tmp_path)
    processes = install_waiting_worker(monkeypatch)

    def sample(pid):
        """等worker进入正文后制造采样器故障，覆盖中途失去监督的边界。"""
        if pid is not None and (tmp_path / "lost.runtime/worker-state.json").exists():
            raise semantic_runtime.SemanticRuntimeError("resource_observation_failed")
        return normal_sample(pid)

    monkeypatch.setattr(semantic_runtime, "_resource_sample", sample)
    with pytest.raises(semantic_runtime.SemanticRuntimeError, match="resource_observation_failed"):
        semantic_runtime.run_guarded_semantic_batch(source, tmp_path / "lost", settings=settings(), code_version="test",
            cache_dir=tmp_path / "vectors", limits=semantic_runtime.SemanticRuntimeLimits(sampling_seconds=0.02))
    assert processes[0].poll() is not None and not (tmp_path / "lost").exists()
    assert (tmp_path / "vectors/validated-cache-sentinel").exists()


def test_worker_peak_rss_failure_after_exit_still_prevents_publication(tmp_path, monkeypatch):
    """两个采样点之间的RSS峰值靠worker复核，进程正常退出也不能绕过门。"""
    source = make_input(tmp_path)
    original = subprocess.Popen
    worker = textwrap.dedent("""
        import json, pathlib, sys
        request = pathlib.Path(sys.argv[1])
        config = json.loads(request.read_text())
        output = pathlib.Path(config['output'])
        output.mkdir()
        (output / 'manifest.json').write_text(json.dumps({'status': 'completed',
            'counts': {'post_count': 2}, 'code_snapshot': config['code_snapshot'],
            'config_sha256': 'unused-before-resource-gate'}))
        (request.parent / 'worker-result.json').write_text(json.dumps({
            'process_peak_rss_bytes': 8 * 1024**3 + 1}))
    """)
    monkeypatch.setattr(semantic_runtime.subprocess, "Popen",
        lambda command, **kwargs: original([sys.executable, "-c", worker, command[-1]], **kwargs))
    monkeypatch.setattr(semantic_runtime, "_resource_sample", normal_sample)
    with pytest.raises(semantic_runtime.SemanticRuntimeError, match="worker_peak_rss_limit_exceeded"):
        semantic_runtime.run_guarded_semantic_batch(source, tmp_path / "peak", settings=settings(), code_version="test",
            limits=semantic_runtime.SemanticRuntimeLimits(sampling_seconds=0.02))
    assert not (tmp_path / "peak").exists() and not (tmp_path / "peak.runtime/completed-run").exists()
    assert json.loads((tmp_path / "peak.attempt.json").read_text())["status"] == "interrupted"


@pytest.mark.parametrize("exception", [RuntimeError("OOM:" + RAW_MARKER), MemoryError(RAW_MARKER)])
def test_model_exception_is_private_and_attempt_records_failure(tmp_path, monkeypatch, capsys, exception):
    """完整traceback可在私有worker日志排障，attempt与CLI都不能夹带原文。"""
    source = make_input(tmp_path)
    request_path = tmp_path / "request.json"
    output = tmp_path / "run"
    provider = SyntheticProvider()
    provider.encode = lambda texts: (_ for _ in ()).throw(exception)
    monkeypatch.setattr(semantic_batch, "QwenEmbeddingProvider", lambda **kwargs: provider)
    request = {"input": {k: str(v) for k, v in asdict(source).items()}, "settings": asdict(settings()),
               "output": str(output), "cache_dir": str(tmp_path / "vectors"), "code_version": "test",
               "post_ids": None, "seed": None, "members_path": None,
               "code_snapshot": None, "resource_supervision": None}
    write_json(request_path, request)
    assert semantic_runtime._worker(request_path) == 1
    assert RAW_MARKER in capsys.readouterr().err
    attempt = (tmp_path / "run.attempt.json").read_text()
    assert RAW_MARKER not in attempt and json.loads(attempt)["error_type"] == type(exception).__name__
    assert json.loads(attempt)["status"] == "failed" and not output.exists()


@pytest.mark.parametrize("exception", [RuntimeError(RAW_MARKER), MemoryError(RAW_MARKER)])
def test_research_cli_never_prints_runtime_exception_text(tmp_path, monkeypatch, capsys, exception):
    """research入口捕获torch RuntimeError／OOM，控制台只显示类型无traceback。"""
    root = Path(semantic_runtime.__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("research_cli_test", root / "scripts/text_segment_research.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "load_research_input", lambda path: make_input(tmp_path))
    monkeypatch.setattr(semantic_runtime, "run_guarded_semantic_batch",
                        lambda *args, **kwargs: (_ for _ in ()).throw(exception))
    monkeypatch.setattr(sys, "argv", ["text_segment_research.py", "--output", str(tmp_path / "output"),
                                     "--algorithm-config", str(root / "configs/text-segmentation-semantic-8b-coherent.json")])
    assert cli.main() == 1
    captured = capsys.readouterr()
    assert RAW_MARKER not in captured.out + captured.err and "Traceback" not in captured.err
    assert type(exception).__name__ in captured.err


@pytest.mark.parametrize("structured", [False, True])
def test_research_cli_rejects_retired_generation_before_reading_source(tmp_path, monkeypatch, capsys, structured):
    """旧身份只作兼容核验；新生成入口在接触正文与模型前拒绝旧规则。"""
    from tourism_ugc_study.segmentation.semantic_config import (
        EmbeddingConfig, SemanticConfig, SemanticSettings, StructuredSemanticConfig,
    )
    root = Path(semantic_runtime.__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("retired_research_cli_test", root / "scripts/text_segment_research.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    def forbidden(*args, **kwargs):
        raise AssertionError("retired rule must fail before corpus or model access")
    monkeypatch.setattr(cli, "load_research_input", forbidden)
    monkeypatch.setattr(semantic_runtime, "run_guarded_semantic_batch", forbidden)
    algorithm = StructuredSemanticConfig() if structured else SemanticConfig()
    config = tmp_path / "historical.json"
    config.write_text(json.dumps(asdict(SemanticSettings(algorithm, EmbeddingConfig()))))
    monkeypatch.setattr(sys, "argv", ["text_segment_research.py", "--output", str(tmp_path / "output"),
                                     "--algorithm-config", str(config)])
    assert cli.main() == 1
    assert "ValueError" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("section,field,value", [
    ("algorithm", "window_tokens", 128),
    ("algorithm", "absolute_cohesion_threshold", 0.65),
    ("algorithm", "depth_floor", 0.15),
    ("algorithm", "min_segment_nonspace_chars", 32),
    ("embedding", "model_id", "Qwen/Qwen3-Embedding-0.6B"),
    ("embedding", "revision", "a" * 40),
    ("embedding", "dtype", "float32"),
    ("embedding", "batch_size", 2),
])
def test_research_cli_rejects_changed_frozen_parameters_before_source_access(
    tmp_path, monkeypatch, capsys, section, field, value,
):
    """同一正式类不等于冻结组合；合法但未经采纳的参数也须在读取来源前拒绝。"""
    root = Path(semantic_runtime.__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("frozen_settings_cli_test", root / "scripts/text_segment_research.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    accessed = []
    def forbidden(*args, **kwargs):
        """留证并阻止任何来源或运行访问，拒绝不能仅依靠后续失败。"""
        accessed.append(True)
        raise AssertionError("unfrozen settings reached source or runtime")
    monkeypatch.setattr(cli, "load_research_input", forbidden)
    monkeypatch.setattr(semantic_runtime, "run_guarded_semantic_batch", forbidden)
    raw = json.loads((root / "configs/text-segmentation-semantic-8b-coherent.json").read_text())
    raw[section][field] = value
    config = tmp_path / "changed-config.json"
    config.write_text(json.dumps(raw))
    monkeypatch.setattr(sys, "argv", ["text_segment_research.py", "--output", str(tmp_path / "output"),
                                     "--algorithm-config", str(config), "--code-version", "synthetic"])
    assert cli.main() == 1
    assert "ValueError" in capsys.readouterr().err and not accessed
    assert not (tmp_path / "output").exists()


def test_current_frozen_settings_validator_binds_registered_digest():
    """登记摘要独立保存；合法配置原样接受，不依赖外部文件重新定义基准。"""
    from tourism_ugc_study.segmentation.semantic_config import (
        CURRENT_FROZEN_SETTINGS_SHA256, load_semantic_settings, validate_current_frozen_settings,
    )
    root = Path(semantic_runtime.__file__).resolve().parents[3]
    current = load_semantic_settings(root / "configs/text-segmentation-semantic-8b-coherent.json")
    assert current.fingerprint == CURRENT_FROZEN_SETTINGS_SHA256
    assert validate_current_frozen_settings(current) is None


def test_snapshot_drift_before_final_validation_prevents_publication(tmp_path):
    """保存源码被改动必须失败；不能只保留HEAD+dirty且忽略实际执行内容。"""
    source = make_input(tmp_path)
    snapshot = semantic_runtime.freeze_code(tmp_path / "snapshot")
    snapshot["executed_from_snapshot"] = True

    def change_snapshot(seen, total):
        """模拟外部修改封存副本，只触碰测试复制文件而非工作区源码。"""
        if seen == 1:
            path = Path(snapshot["directory"]) / "src/tourism_ugc_study/segmentation/semantic_api.py"
            path.write_bytes(path.read_bytes() + b"\n# changed\n")

    with pytest.raises(ValueError, match="code_snapshot_changed_during_run"):
        semantic_batch.run_semantic_batch(source, tmp_path / "drift", settings=settings(), code_version="test",
            provider=SyntheticProvider(), code_snapshot=snapshot, progress=change_snapshot)
    assert not (tmp_path / "drift").exists()


@pytest.mark.parametrize("invalid", [0, -1, float("nan"), float("inf")])
def test_invalid_post_timeout_cannot_disable_stall_guard(invalid):
    """零、负值或NaN不能使运行时间门静默失效。"""
    with pytest.raises(ValueError, match="invalid_semantic_runtime_limits"):
        semantic_runtime.SemanticRuntimeLimits(post_seconds=invalid)
