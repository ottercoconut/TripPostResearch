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
from tourism_ugc_study.segmentation.embedding_cache import EmbeddingCache
from tourism_ugc_study.segmentation.semantic_config import EmbeddingConfig, validate_identity_settings
from test_semantic_api import SyntheticProvider
from test_semantic_batch import settings
from test_semantic_resources import MIB, UUID, linux_sample, policy


def padding_provider(monkeypatch, *, device='cuda', dtype='bfloat16', max_input_tokens=12):
    """用合成tokenizer和参数元数据观察真实provider初始化及forward输入。

    张量始终留在CPU；设备元数据只让初始化到达padding身份分支，不证明
    CUDA计算。tokenizer支持最长及固定宽度左pad，不截断超长输入。
    """
    import torch
    from transformers import AutoModel

    declared = EmbeddingConfig(device=device, dtype=dtype, batch_size=2,
                              max_input_tokens=max_input_tokens)
    tokenizations, forwards = [], []

    class Batch(dict):
        """记录设备请求，保留CPU合成张量以免单测需要GPU。"""

        def to(self, requested_device):
            assert requested_device == device
            return self

    def tokenize(texts, **kwargs):
        """模拟HF无截断左pad，记录调用参数及实际模型输入。"""
        tokenizations.append(kwargs)
        assert kwargs['truncation'] is False
        assert kwargs['add_special_tokens'] is True and kwargs['return_tensors'] == 'pt'
        width = max(map(len, texts))
        if kwargs['padding'] == 'max_length':
            width = max(width, kwargs['max_length'])
        else:
            assert kwargs['padding'] is True
        ids = torch.tensor([[0] * (width - len(text)) + [ord(char) for char in text]
                            for text in texts])
        return Batch(input_ids=ids, attention_mask=ids.ne(0).long())

    def counter(self, **kwargs):
        """沿用完整声明身份，只省略下载、模板和真实tokenizer初始化。"""
        self.identity = SyntheticProvider(declared).identity
        self._asset_paths = {}
        self.max_input_tokens = kwargs['max_input_tokens']
        self.tokenizer = tokenize
        self.format_input = lambda text: text
        self.input_count = len

    class MetadataModel:
        """提供完整初始化观察与无权重forward，仅检验输入契约。"""

        config = SimpleNamespace(_commit_hash=declared.revision, use_cache=False)

        def to(self, requested_device):
            assert requested_device == device
            return self

        def eval(self):
            return self

        def parameters(self):
            return [SimpleNamespace(dtype=getattr(torch, dtype),
                                    device=torch.device('cuda:0' if device == 'cuda' else device))]

        def __call__(self, input_ids, attention_mask, use_cache):
            assert use_cache is False and not torch.is_grad_enabled()
            forwards.append({'input_ids': input_ids.clone(), 'attention_mask': attention_mask.clone()})
            hidden = torch.zeros((*input_ids.shape, 1024))
            hidden[..., 0], hidden[..., 1] = input_ids.float(), 1
            return SimpleNamespace(last_hidden_state=hidden)

    monkeypatch.setattr(embedding.QwenTokenCounter, '__init__', counter)
    monkeypatch.setattr(embedding.QwenTokenCounter, '_bind_assets', lambda self, names: None)
    monkeypatch.setattr(AutoModel, 'from_pretrained', lambda *args, **kwargs: MetadataModel())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda **kwargs: True)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda index:
                        SimpleNamespace(name='synthetic-GPU', major=8, minor=9, uuid='fixture'))
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    monkeypatch.setattr(torch.mps, 'set_per_process_memory_fraction', lambda fraction: None)
    monkeypatch.setattr(torch, 'set_float32_matmul_precision', lambda precision: None)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', False)
    provider = embedding.QwenEmbeddingProvider(model_id=declared.model_id,
        revision=declared.revision, device=device, dtype=dtype, batch_size=2,
        max_input_tokens=max_input_tokens)
    return provider, declared, tokenizations, forwards


@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
def test_cuda_padding_keeps_repeated_input_shape_and_ids_with_different_partners(monkeypatch, dtype):
    """同一行在不同长度伙伴下必须有相同宽度、pad位置与mask，适用于所有CUDA精度。"""
    import torch

    provider, declared, tokenizations, forwards = padding_provider(monkeypatch, dtype=dtype)
    provider.encode(['重复', '甲', '重复', '乙丙丁戊己庚辛'])
    assert [tuple(batch['input_ids'].shape) for batch in forwards] == [(2, 12), (2, 12)]
    for key in ('input_ids', 'attention_mask'):
        assert torch.equal(forwards[0][key][0], forwards[1][key][0])
    assert all(call['padding'] == 'max_length' and call['max_length'] == declared.max_input_tokens
               and call['truncation'] is False for call in tokenizations)
    assert provider.identity['padding_strategy'] == 'max_length'
    assert provider.identity['padding_max_length'] == declared.max_input_tokens
    validate_identity_settings(provider.identity, declared)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_non_cuda_padding_preserves_longest_call_and_legacy_identity(monkeypatch, device):
    """旧Mac/CPU仍按伙伴最长输入pad，调用和身份均不引入CUDA新字段。"""
    provider, declared, tokenizations, forwards = padding_provider(monkeypatch, device=device, dtype='float32')
    provider.encode(['重复', '甲', '重复', '乙丙丁戊己庚辛'])
    assert [tuple(batch['input_ids'].shape) for batch in forwards] == [(2, 2), (2, 7)]
    assert all(call['padding'] is True and 'max_length' not in call for call in tokenizations)
    assert 'padding_strategy' not in provider.identity and 'padding_max_length' not in provider.identity
    validate_identity_settings(provider.identity, declared)


@pytest.mark.parametrize('guard', ['count', 'encoded_width'])
def test_cuda_fixed_padding_cannot_truncate_overflow_or_reach_forward(monkeypatch, guard):
    """计数超限先拒绝；即使计数漂移，实际超宽输入仍不能截断后进入模型。"""
    provider, _, tokenizations, forwards = padding_provider(monkeypatch)
    if guard == 'encoded_width':
        provider.input_count = lambda text: 1
    with pytest.raises(embedding.EmbeddingError, match='model_input_limit_exceeded'):
        provider.encode(['超' * 13])
    assert not forwards and provider.model_call_count == 0
    assert bool(tokenizations) is (guard == 'encoded_width')


@pytest.mark.parametrize('changes', [
    {'padding_strategy': None, 'padding_max_length': None},
    {'padding_strategy': None}, {'padding_max_length': None},
    {'padding_strategy': 'longest'}, {'padding_max_length': 13},
    {'padding_max_length': True}, {'padding_max_length': '12'},
])
def test_cuda_identity_missing_or_wrong_padding_cannot_pass_settings(monkeypatch, changes):
    """旧CUDA身份、错误策略或不等于实际预算的宽度均拒绝，不静默升级历史运行。"""
    provider, declared, _, _ = padding_provider(monkeypatch)
    for key, value in changes.items():
        if value is None:
            provider.identity.pop(key, None)
        else:
            provider.identity[key] = value
    with pytest.raises(ValueError, match='embedding_identity_settings_mismatch'):
        validate_identity_settings(provider.identity, declared)


def test_cuda_fixed_padding_identity_cannot_reuse_legacy_longest_cache(tmp_path, monkeypatch):
    """相同文本和模板也不能命中旧CUDA最长padding的缓存。"""
    import numpy as np

    provider, _, _, forwards = padding_provider(monkeypatch)
    legacy_identity = dict(provider.identity)
    legacy_identity.pop('padding_strategy', None)
    legacy_identity.pop('padding_max_length', None)
    legacy_vectors = np.zeros((1, 1024), dtype=np.float32)
    legacy_vectors[:, 1] = 1
    legacy = SimpleNamespace(identity=legacy_identity, encode=lambda texts: legacy_vectors)
    old_cache = EmbeddingCache(tmp_path, legacy.identity)
    old_cache.encode(['重复'], legacy)
    current_cache = EmbeddingCache(tmp_path, provider.identity)
    vectors = current_cache.encode(['重复'], provider)
    assert current_cache.root != old_cache.root
    assert current_cache.cache_hit_count == 0 and current_cache.cache_miss_count == 1
    assert len(forwards) == 1 and not np.array_equal(vectors, legacy_vectors)


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
