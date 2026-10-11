"""CUDA声明与实际参数不一致、精度门被放宽时的独立反例。

仅用参数元数据和合成输入测试拒绝行为，不加载权重或声称本机完成CUDA验收。
"""

from types import SimpleNamespace
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest

from tourism_ugc_study.segmentation import embedding, embedding_probe
from test_semantic_batch import settings
from test_semantic_resources import MIB, UUID, linux_sample, policy


def fake_parameters(*specifications):
    """构造实际参数观察，不借请求身份替代dtype和逻辑GPU检查。"""
    return SimpleNamespace(parameters=lambda: iter(
        SimpleNamespace(dtype=dtype, device=SimpleNamespace(type=device, index=index))
        for dtype, device, index in specifications))


@pytest.mark.parametrize('actual_device,index', [('cpu', None), ('cuda', 1)])
def test_cuda_declaration_cannot_hide_parameters_on_other_device(actual_device, index):
    """CPU参数或另一逻辑GPU不能只因请求写了cuda就被记成同一设备。"""
    import torch

    model = fake_parameters((torch.float32, actual_device, index))
    with pytest.raises(embedding.EmbeddingError, match='loaded_model_parameter_device_mismatch'):
        embedding._validate_model_parameters(model, device='cuda', expected_dtype=torch.float32,
                                              cuda_device_index=0)


def test_later_parameter_with_wrong_precision_cannot_hide_behind_first_parameter():
    """首个参数精度正确不足以证明整个Qwen权重符合声明。"""
    import torch

    model = fake_parameters((torch.float32, 'cuda', 0), (torch.bfloat16, 'cuda', 0))
    with pytest.raises(embedding.EmbeddingError, match='loaded_model_parameter_dtype_mismatch'):
        embedding._validate_model_parameters(model, device='cuda', expected_dtype=torch.float32,
                                              cuda_device_index=0)


def test_missing_floating_parameters_cannot_certify_requested_precision():
    """没有实际浮点观察时不能写入一个已经验证的FP32运行身份。"""
    import torch

    with pytest.raises(embedding.EmbeddingError, match='loaded_model_has_no_floating_parameters'):
        embedding._validate_model_parameters(fake_parameters((torch.int64, 'cuda', 0)),
            device='cuda', expected_dtype=torch.float32, cuda_device_index=0)


def test_emulated_bfloat16_is_rejected_before_weight_loading(monkeypatch):
    """可用CUDA不等于原生BF16；在下载或读权重前拒绝仅有模拟支持的设备。"""
    import torch

    observed = []
    monkeypatch.setattr(embedding.QwenTokenCounter, '__init__', lambda self, **kwargs: None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)

    def supported(*, including_emulation=True):
        observed.append(including_emulation)
        return including_emulation

    def weights(*args, **kwargs):
        raise AssertionError('hardware rejection must precede weights')

    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', supported)
    monkeypatch.setattr(embedding.QwenTokenCounter, '_bind_assets', weights)
    with pytest.raises(embedding.EmbeddingError, match='cuda_bfloat16_not_natively_supported'):
        embedding.QwenEmbeddingProvider(model_id='Qwen/Qwen3-Embedding-4B', revision='a' * 40,
            device='cuda', dtype='bfloat16', batch_size=1)
    assert observed == [False]


@pytest.mark.parametrize('dtype', ['float16', 'bfloat16'])
def test_lower_precision_cannot_silently_borrow_fp32_tolerances(dtype):
    """低精度必须在运行前明确容差，不能遇到误差后自动改门。"""
    with pytest.raises(ValueError, match='lower_precision_smoke_tolerances_required'):
        embedding_probe.parse_smoke_tolerances(dtype, None)


def test_explicit_values_cannot_relax_the_fixed_fp32_gate():
    """增加CLI参数不能成为放宽已冻结FP32数值门的旁路。"""
    raw = {'repeat': 1e-5, 'batch': 1e-3, 'norm': 1e-5}
    with pytest.raises(ValueError, match='fp32_smoke_tolerances_fixed'):
        embedding_probe.parse_smoke_tolerances('float32', raw)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, 0])
def test_nonfinite_or_nonpositive_tolerance_cannot_disable_a_gate(bad):
    """NaN会使比较恒假，bool和零也不是可信的精度政策。"""
    with pytest.raises(ValueError, match='invalid_smoke_numerical_tolerances'):
        embedding_probe.parse_smoke_tolerances('bfloat16', {'repeat': 1e-5, 'batch': bad, 'norm': 1e-5})


def test_omitting_norm_gate_is_not_an_accepted_partial_tolerance_policy():
    """显式声明必须同时覆盖逐向量、成对余弦和单位向量检查。"""
    with pytest.raises(ValueError, match='invalid_smoke_numerical_tolerances'):
        embedding_probe.parse_smoke_tolerances('bfloat16', {'repeat': 1e-5, 'batch': 1e-3})


@pytest.mark.parametrize('phase', ['baseline', 'live', 'final'])
def test_cuda_smoke_keeps_over_limit_observation_even_without_passed_result(tmp_path, monkeypatch, phase):
    """启动、推理和退出阶段超限均保存整卡采样峰值，不能仅在成功时汇总。"""
    probe = embedding_probe
    original, real_sleep = subprocess.Popen, time.sleep
    launched = []
    worker = textwrap.dedent('''
        import json, pathlib, sys, time
        request = json.loads(pathlib.Path(sys.argv[1]).read_text())
        root = pathlib.Path(request['output'])
        (root / 'ready').write_text('1')
        if sys.argv[2] == 'live':
            while True:
                time.sleep(0.01)
    ''')

    def launch(command, **kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == UUID
        snapshot_src = Path(kwargs['env']['PYTHONPATH'])
        assert (snapshot_src/'tourism_ugc_study/segmentation/semantic_resources.py').is_file()
        process = original([sys.executable, '-c', worker, command[-1], phase], **kwargs)
        launched.append(process)
        return process

    system_samples = []

    def sample(pid, selected_policy):
        triggered = phase == 'baseline'
        if phase == 'live':
            triggered = pid is not None and (tmp_path/'probe/cuda/ready').exists()
        elif phase == 'final':
            triggered = pid is None and bool(system_samples)
        if pid is None:
            system_samples.append(pid)
        return linux_sample(pid, used=1025 * MIB if triggered else 512 * MIB)

    monkeypatch.setattr(probe.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(probe.platform, 'platform', lambda: 'synthetic-linux-resource-test')
    monkeypatch.setattr(probe, 'verify_local_assets', lambda **kwargs: {'assets': []})
    monkeypatch.setattr(probe, 'sample_linux_resources', sample)
    monkeypatch.setattr(probe.subprocess, 'Popen', launch)
    monkeypatch.setattr(probe.time, 'sleep', lambda seconds: real_sleep(0.01))
    result = probe.run_smoke(tmp_path/'probe', embedding=replace(settings().embedding, device='cuda'),
                            resource_policy=policy())
    assert result['status'] == 'failed' and result['error'] == 'cuda_memory_limit_exceeded'
    assert result['peak_sampled_cuda_device_used_bytes'] == 1025 * MIB
    assert result['code_snapshot']['executed_from_snapshot'] is True
    assert all(process.poll() is not None for process in launched)
    assert bool(launched) is (phase != 'baseline')
    saved = json.loads((tmp_path/'probe/summary.json').read_text())
    assert saved['peak_sampled_cuda_device_used_bytes'] == result['peak_sampled_cuda_device_used_bytes']
