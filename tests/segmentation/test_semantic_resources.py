"""Linux资源单位、GPU身份与超门终态的独立反例；本机不冒充CUDA实测。

/proc 与 nvidia-smi 使用受控的真实格式读数；进程停止测试仍启动实际OS子进程。
所有正文与来源库均为测试自造材料。
"""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from test_semantic_batch import make_input, settings
from test_semantic_runtime import install_waiting_worker
from tourism_ugc_study.segmentation import semantic_resources as resources, semantic_runtime as runtime

UUID = 'GPU-12345678-1234-5678-90ab-123456789abc'
OTHER_UUID = 'GPU-87654321-4321-8765-ab09-cba987654321'
MIB = 1024**2


def policy(**changes):
    """测试使用小门限，使未转换KiB/MiB的实现无法误过门。"""
    return resources.LinuxResourcePolicy(**{
        'rss_bytes': 2 * MIB, 'additional_swap_bytes': 0,
        'cuda_device_uuid': UUID, 'cuda_memory_bytes': 1024 * MIB, **changes})


def install_readings(monkeypatch, *, status='VmRSS:\t2049 kB\n', memory='SwapTotal: 1024 kB\nSwapFree: 1023 kB\n',
                     uuid=UUID, used='512', total='16384'):
    """只替换对应观测文件与SMI命令；其他文件操作保持真实。"""
    original = Path.read_text
    commands = []

    def read(path, *args, **kwargs):
        if str(path) == '/proc/418/status':
            if status is None:
                raise FileNotFoundError(str(path))
            return status
        if str(path) == '/proc/meminfo':
            return memory
        return original(path, *args, **kwargs)

    def command(arguments, **kwargs):
        commands.append(arguments)
        assert arguments[0] == 'nvidia-smi'
        assert f'--id={UUID}' in arguments
        values = {'uuid': uuid, 'memory.used': used, 'memory.total': total}
        query = next(arg.split('=', 1)[1] for arg in arguments if arg.startswith('--query-gpu='))
        return SimpleNamespace(returncode=0, stdout=', '.join(values[key] for key in query.split(','))+'\n')

    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(resources.subprocess, 'run', command)
    return commands


def test_proc_kib_conversion_is_required_for_rss_protection(monkeypatch):
    """2049KiB超过2MiB门；若把Linux kB当字节会错误通过。"""
    install_readings(monkeypatch)
    sample = resources.sample_linux_resources(418, policy())
    assert sample['rss_bytes'] == 2049 * 1024
    assert sample['swap_bytes'] == 1024 and sample['memory_pressure_level'] is None
    with pytest.raises(resources.SemanticResourceError, match='rss_limit_exceeded'):
        resources.require_linux_resources(sample, 1024, policy())


def test_smi_mib_conversion_and_selected_uuid_are_required_for_gpu_gate(monkeypatch):
    """1025MiB全卡用量须触发1GiB门，不能与allocator观察混作同一范围。"""
    commands = install_readings(monkeypatch, used='1025')
    sample = resources.sample_linux_resources(None, policy())
    assert commands and sample['rss_bytes'] is None
    assert sample['cuda_device_used_bytes'] == 1025 * MIB
    with pytest.raises(resources.SemanticResourceError, match='cuda_memory_limit_exceeded'):
        resources.require_linux_resources(sample, sample['swap_bytes'], policy())


def test_wrong_gpu_observation_cannot_be_accepted_by_requested_uuid(monkeypatch):
    """命令请求卡A而工具返回卡B时拒绝，不能只信调用参数。"""
    install_readings(monkeypatch, uuid=OTHER_UUID)
    with pytest.raises(resources.SemanticResourceError, match='cuda_device_identity_mismatch'):
        resources.sample_linux_resources(None, policy())


@pytest.mark.parametrize('used', ['N/A', '-1'])
def test_missing_or_negative_gpu_measurement_does_not_mean_zero_usage(monkeypatch, used):
    """驱动不可观察/非法读数不能关闭显存门。"""
    install_readings(monkeypatch, used=used)
    with pytest.raises(resources.SemanticResourceError, match='resource_observation_failed'):
        resources.sample_linux_resources(None, policy())


def test_missing_process_differs_from_system_telemetry_failure(monkeypatch):
    """PID退出竞态可由监督器复核；系统swap缺失不能冒充同一退出竞态。"""
    install_readings(monkeypatch, status=None, memory='MemTotal: 100000 kB\n')
    with pytest.raises(resources.SemanticResourceError, match='worker_rss_unavailable'):
        resources.sample_linux_resources(418, policy())
    with pytest.raises(resources.SemanticResourceError, match='resource_observation_failed'):
        resources.sample_linux_resources(None, policy())


def test_zero_swap_budget_still_rejects_one_byte_growth(monkeypatch):
    """0表示不允许新增swap，不能被if budget当作关闭保护。"""
    install_readings(monkeypatch, status='VmRSS: 100 kB\n')
    sample = resources.sample_linux_resources(None, policy())
    with pytest.raises(resources.SemanticResourceError, match='additional_swap_limit_exceeded'):
        resources.require_linux_resources(sample, sample['swap_bytes'] - 1, policy())


def test_policy_above_physical_gpu_capacity_cannot_claim_valid_protection(monkeypatch):
    """错误配置的门限大于所选卡物理显存时，在推理之前拒绝。"""
    install_readings(monkeypatch, total='512', used='1')
    sample = resources.sample_linux_resources(None, policy())
    with pytest.raises(resources.SemanticResourceError, match='cuda_memory_limit_exceeds_device_total'):
        resources.require_linux_resources(sample, sample['swap_bytes'], policy())


@pytest.mark.parametrize('system,expected', [('Linux', 9 * 1024**3), ('Darwin', 9 * 1024**2)])
def test_peak_rss_unit_depends_on_os_not_inferred_value(monkeypatch, system, expected):
    """同一ru_maxrss原值在Linux应触发GiB级门，不可沿用Darwin字节解释。"""
    monkeypatch.setattr(resources.platform, 'system', lambda: system)
    monkeypatch.setattr(resources.resource, 'getrusage', lambda who: SimpleNamespace(ru_maxrss=9 * 1024**2))
    assert resources.process_peak_rss_bytes() == expected


def test_linux_without_explicit_policy_never_launches_a_worker(tmp_path, monkeypatch):
    """不能在新平台静默沿用Mac资源门或绕过监督。"""
    source = make_input(tmp_path)
    monkeypatch.setattr(runtime.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(runtime.subprocess, 'Popen', lambda *a, **k: pytest.fail('unexpected worker'))
    actual = settings()
    actual = replace(actual, embedding=replace(actual.embedding, device='cuda'))
    with pytest.raises(runtime.SemanticRuntimeError, match='semantic_linux_resource_policy_required'):
        runtime.run_guarded_semantic_batch(source, tmp_path/'run', settings=actual, code_version='test')
    assert not (tmp_path/'run').exists()


def test_linux_policy_cannot_override_the_fixed_macos_caps(tmp_path, monkeypatch):
    """显式Linux政策并非放宽当前Mac 8GiB门的开关。"""
    source = make_input(tmp_path)
    monkeypatch.setattr(runtime.platform, 'system', lambda: 'Darwin')
    with pytest.raises(runtime.SemanticRuntimeError, match='linux_resource_policy_requires_linux'):
        runtime.run_guarded_semantic_batch(source, tmp_path/'run', settings=settings(), code_version='test',
                                          resource_policy=policy(rss_bytes=32 * 1024**3))
    assert not (tmp_path/'run').exists()


def linux_sample(pid, *, used=512 * MIB):
    """模拟可见系统读数，RSS与整卡显存保持不同范围。"""
    return {'rss_bytes': 1024 if pid is not None else None, 'swap_bytes': 100,
            'memory_pressure_level': None, 'cuda_device_uuid': UUID,
            'cuda_device_used_bytes': used, 'cuda_device_total_bytes': 16384 * MIB}


def test_cuda_limit_stops_real_worker_and_retains_trigger_and_cache(tmp_path, monkeypatch):
    """全卡超门必须回收真实进程，保留触发观测和既有缓存，不能发布半包。"""
    source = make_input(tmp_path)
    processes = install_waiting_worker(monkeypatch)
    launch = runtime.subprocess.Popen
    bound_devices = []

    def capture(command, **kwargs):
        bound_devices.append(kwargs['env'].get('CUDA_VISIBLE_DEVICES'))
        return launch(command, **kwargs)

    def sample(pid, selected_policy):
        triggered = pid is not None and (tmp_path/'gpu.runtime/worker-state.json').exists()
        return linux_sample(pid, used=1025 * MIB if triggered else 512 * MIB)

    monkeypatch.setattr(runtime.subprocess, 'Popen', capture)
    monkeypatch.setattr(runtime.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(resources, 'sample_linux_resources', sample)
    actual = settings()
    actual = replace(actual, embedding=replace(actual.embedding, device='cuda'))
    with pytest.raises(runtime.SemanticRuntimeError, match='cuda_memory_limit_exceeded'):
        runtime.run_guarded_semantic_batch(source, tmp_path/'gpu', settings=actual,
            code_version='test', cache_dir=tmp_path/'cache', resource_policy=policy(),
            limits=runtime.SemanticRuntimeLimits(sampling_seconds=0.02))
    assert bound_devices == [UUID] and processes[0].poll() is not None
    assert not (tmp_path/'gpu').exists()
    assert not list((tmp_path/'gpu.runtime').glob('.completed-run-*'))
    assert (tmp_path/'cache/validated-cache-sentinel').read_text() == 'preserve'
    attempt = json.loads((tmp_path/'gpu.attempt.json').read_text())
    assert attempt['error_code'] == 'cuda_memory_limit_exceeded'
    assert attempt['peak_sampled_cuda_device_used_bytes'] == 1025 * MIB
    assert attempt['last_resources']['cuda_device_used_bytes'] == 1025 * MIB


def test_cuda_peak_between_samples_prevents_publication_after_exit(tmp_path, monkeypatch):
    """整卡采样均正常时仍检查worker分配器峰值，退出成功不能掩盖瞬时超门。"""
    source = make_input(tmp_path)
    original = subprocess.Popen
    worker = textwrap.dedent('''
        import json, pathlib, sys
        request = pathlib.Path(sys.argv[1])
        config = json.loads(request.read_text())
        output = pathlib.Path(config['output'])
        output.mkdir()
        (output / 'manifest.json').write_text('{}')
        (request.parent / 'worker-result.json').write_text(json.dumps({
            'process_peak_rss_bytes': 1024,
            'cuda_visible_device_uuid': config['resource_policy']['cuda_device_uuid'],
            'cuda_peak_reserved_bytes': 1024 * 1024 * 1024 + 1}))
    ''')
    monkeypatch.setattr(runtime.subprocess, 'Popen',
        lambda command, **kwargs: original([sys.executable, '-c', worker, command[-1]], **kwargs))
    monkeypatch.setattr(runtime.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(resources, 'sample_linux_resources', lambda pid, policy: linux_sample(pid))
    actual = settings()
    actual = replace(actual, embedding=replace(actual.embedding, device='cuda'))
    with pytest.raises(runtime.SemanticRuntimeError, match='worker_peak_cuda_reserved_limit_exceeded'):
        runtime.run_guarded_semantic_batch(source, tmp_path/'peak', settings=actual,
            code_version='test', resource_policy=policy(),
            limits=runtime.SemanticRuntimeLimits(sampling_seconds=0.02))
    assert not (tmp_path/'peak').exists() and not (tmp_path/'peak.runtime/completed-run').exists()
    attempt = json.loads((tmp_path/'peak.attempt.json').read_text())
    assert attempt['peak_sampled_cuda_device_used_bytes'] == 512 * MIB
    assert attempt['worker_cuda_resources']['cuda_peak_reserved_bytes'] == 1024 * MIB + 1
