"""向量输入、末 token pooling、身份与无截断契约的确定性验证。"""

from types import SimpleNamespace
import json

import numpy as np
import pytest

from tourism_ugc_study.segmentation.embedding import (
    EmbeddingError, QwenEmbeddingProvider, QwenTokenCounter, identity_sha256, validate_vectors,
)


@pytest.mark.parametrize("value,count,dimension,error", [
    (np.ones((1, 2)), 2, 2, "invalid_vector_shape"),
    (np.ones((1, 3)), 1, 2, "invalid_vector_shape"),
    ([[1, 2]], 1, 2, "invalid_vector_dtype"),
    ([[np.nan, 1.0]], 1, 2, "nonfinite_vector"),
    ([[np.inf, 1.0]], 1, 2, "nonfinite_vector"),
    ([[0.0, 0.0]], 1, 2, "zero_vector"),
])
def test_vector_validation_rejects_invalid_output(value, count, dimension, error):
    with pytest.raises(EmbeddingError, match=error):
        validate_vectors(value, count=count, dimension=dimension)


def test_vector_validation_preserves_order_and_returns_copy():
    original = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float64)
    result = validate_vectors(original, count=2, dimension=2)
    assert result.dtype == np.float32
    assert result.tolist() == original.tolist()
    result[0, 0] = 99
    assert original[0, 0] == 3


def test_identity_order_independent_and_nonfinite_rejected():
    assert identity_sha256({"a": 1, "b": 2}) == identity_sha256({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        identity_sha256({"bad": float("nan")})


def test_invalid_revision_fails_before_loading():
    with pytest.raises(EmbeddingError, match="revision_must_be_commit_sha"):
        QwenTokenCounter(model_id="unused", revision="main")


def test_offset_envelope_preserves_original_combining_grapheme():
    counter = object.__new__(QwenTokenCounter)
    counter._asset_paths = {}
    counter.tokenizer = lambda *args, **kwargs: {"input_ids": [1, 2], "offset_mapping": [(0, 2), (2, 4)]}
    assert counter.offsets("cafe\u0301") == ((0, 2), (2, 5))


def test_offset_gaps_are_not_guessed():
    counter = object.__new__(QwenTokenCounter)
    counter._asset_paths = {}
    counter.tokenizer = lambda *args, **kwargs: {"input_ids": [1, 2], "offset_mapping": [(0, 1), (2, 3)]}
    with pytest.raises(EmbeddingError, match="invalid_token_offsets"):
        counter.offsets("abc")


def test_asset_change_is_rejected(tmp_path):
    asset = tmp_path / "model.bin"
    asset.write_bytes(b"original")
    counter = object.__new__(QwenTokenCounter)
    counter._asset_paths = {"model.bin": asset}
    counter._asset_stats = {"model.bin": counter._stat_identity(asset)}
    counter.identity = {"assets_sha256": {"model.bin": counter._file_sha(asset)}}
    counter.verify_assets()
    asset.write_bytes(b"changed")
    with pytest.raises(EmbeddingError, match="model_asset_changed"):
        counter.verify_assets(strong=False)


@pytest.mark.parametrize("change_phase", ["after_read", "after_bind", "stable"])
def test_sharded_index_change_cannot_certify_unbound_loaded_weights(tmp_path, monkeypatch, change_phase):
    """列出分片后换索引，不能用新索引摘要与旧分片摘要证明实际新权重。

    资产为临时合成字节，loader只读取分片名和参数元数据，不运行模型。
    读取后或绑定后替换文件，必须由已有资产变化门拒绝返回provider；
    稳定分片作为对照，必须全部进入身份。旧实现的读取后竞态会成功且
    强复核通过，但最后的实际分片绑定断言失败。
    """
    import torch
    import transformers
    import huggingface_hub
    from pathlib import Path

    revision = "a" * 40
    index = tmp_path / "model.safetensors.index.json"
    old_shard, new_shard = "old-part.safetensors", "new-part.safetensors"
    index.write_text(json.dumps({"weight_map": {"weight": old_shard}}))
    for name in (old_shard, new_shard):
        (tmp_path / name).write_bytes(name.encode())
    original_read = Path.read_text
    original_bind = QwenTokenCounter._bind_assets
    replaced = False
    loaded = []

    def read_index_then_replace(path, *args, **kwargs):
        """返回原清单，随后换成另一有效清单，制造真实文件身份漂移。"""
        nonlocal replaced
        raw = original_read(path, *args, **kwargs)
        if path == index and not replaced and change_phase == "after_read":
            replaced = True
            index.write_text(json.dumps({"weight_map": {"weight": new_shard}}))
        return raw

    def bind_then_replace(self, names):
        """另一反例在索引已绑定后换文件，不能重新接受漂移后的身份。"""
        nonlocal replaced
        original_bind(self, names)
        if index.name in names and not replaced and change_phase == "after_bind":
            replaced = True
            index.write_text(json.dumps({"weight_map": {"weight": new_shard}}))

    def synthetic_counter(self, **kwargs):
        """只替换tokenizer初始化；保留真实资产绑定与复验实现。"""
        self._asset_paths, self._asset_stats = {}, {}
        self._asset_kwargs = {name: kwargs[name] for name in
                              ("model_id", "revision", "local_files_only", "cache_dir")}
        self.identity = {"dimension": 4096, "assets_sha256": {}, "dependencies": {}}

    class MetadataModel:
        """只提供加载后校验需要的实际参数元数据，没有forward接口。"""

        config = SimpleNamespace(_commit_hash=revision, use_cache=False)

        def to(self, device):
            """合成对象固定在CPU，不分配模型参数。"""
            return self

        def eval(self):
            """保留provider初始化接口，不执行推理。"""
            return self

        def parameters(self):
            """返回CPU／FP32元数据，使验收能够到达资产漂移检查。"""
            return [SimpleNamespace(dtype=torch.float32, device=torch.device("cpu"))]

    def load_declared_weights(*args, **kwargs):
        """像加载器一样重新读当前索引，记录其实际选择的分片。"""
        names = json.loads(original_read(index))["weight_map"].values()
        for name in names:
            assert (tmp_path / name).read_bytes() == name.encode()
            loaded.append(name)
        return MetadataModel()

    monkeypatch.setattr(QwenTokenCounter, "__init__", synthetic_counter)
    monkeypatch.setattr(QwenTokenCounter, "_bind_assets", bind_then_replace)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda model_id, name, **kwargs: str(tmp_path / name))
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", load_declared_weights)
    monkeypatch.setattr(Path, "read_text", read_index_then_replace)
    try:
        provider = QwenEmbeddingProvider(model_id="Qwen/Qwen3-Embedding-8B",
            revision=revision, device="cpu", dtype="float32", batch_size=1)
    except EmbeddingError as exc:
        assert change_phase != "stable" and replaced
        assert str(exc) == "model_asset_changed"
        assert loaded == [new_shard]
        return

    # 旧实现会到这里，而且strong=True仍通过；实际加载分片必须全部有摘要。
    provider.verify_assets()
    assert loaded == [old_shard if change_phase == "stable" else new_shard]
    assert set(loaded) <= set(provider.identity["assets_sha256"])
    assert change_phase == "stable"
    assert set(provider.identity["assets_sha256"]) == {index.name, old_shard}


def _fake_provider():
    torch = pytest.importorskip("torch")

    class Batch(dict):
        def to(self, device):
            return self

    def tokenizer(texts, **kwargs):
        assert kwargs["truncation"] is False
        assert kwargs["padding"] is True
        length = max(map(len, texts))
        rows = [[0] * (length - len(t)) + [ord(c) for c in t] for t in texts]
        inputs = torch.tensor(rows)
        return Batch(input_ids=inputs, attention_mask=inputs.ne(0).long())

    class Model:
        def __call__(self, input_ids, attention_mask, use_cache):
            assert use_cache is False
            assert not torch.is_grad_enabled()
            return SimpleNamespace(last_hidden_state=torch.stack((input_ids.float(), torch.ones_like(input_ids)), dim=-1))

    provider = object.__new__(QwenEmbeddingProvider)
    provider._asset_paths = {}
    provider.identity = {"dimension": 2}
    provider.input_count = len
    provider.format_input = lambda text: text
    provider.tokenizer = tokenizer
    provider.model = Model()
    provider.device = "cpu"
    provider.batch_size = 2
    provider.max_input_tokens = 8
    provider.model_call_count = 0
    return provider


def test_provider_pools_last_valid_token_and_preserves_input_order():
    provider = _fake_provider()
    result = provider.encode(["甲", "甲乙", "甲"])
    assert result.shape == (3, 2)
    assert result.dtype == np.float32
    assert np.array_equal(result[0], result[2])
    assert not np.array_equal(result[0], result[1])
    assert np.allclose(np.linalg.norm(result, axis=1), 1)
    assert provider.model_call_count == 2


def test_overlong_input_rejected_before_any_forward():
    provider = _fake_provider()
    with pytest.raises(EmbeddingError, match="model_input_limit_exceeded"):
        provider.encode(["可以", "超限" * 5])
    assert provider.model_call_count == 0


def test_empty_batch_and_invalid_strings():
    provider = _fake_provider()
    assert provider.encode([]).shape == (0, 2)
    for values in ("wrong", [""], [None]):
        with pytest.raises(EmbeddingError):
            provider.encode(values)
    assert provider.model_call_count == 0


def test_probe_swap_guard_terminates_worker_and_preserves_failure(tmp_path, monkeypatch):
    """监督退出本身必须可测试，不能依赖真实机器再次产生 swap。"""
    from tourism_ugc_study.segmentation import embedding_probe as probe

    processes = []

    class Process:
        pid = 100
        terminated = False

        def __init__(self, *args, **kwargs):
            processes.append(self)

        def poll(self):
            return -15 if self.terminated else None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return -15

    swaps = iter([0, 600 * 1024**2])
    monkeypatch.setattr(probe.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(probe.platform, "platform", lambda: "mock-macos")
    monkeypatch.setattr(probe, "verify_local_assets", lambda: {"test_assets": True})
    monkeypatch.setattr(probe, "_swap_bytes", lambda: next(swaps))
    monkeypatch.setattr(probe, "_pressure_level", lambda: 1)
    monkeypatch.setattr(probe.subprocess, "Popen", Process)
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="4096"))
    output = tmp_path / "probe"
    result = probe.run_smoke(output)
    assert result["status"] == "failed"
    assert result["error"] == "additional_swap_limit_exceeded"
    assert len(processes) == 1 and processes[0].terminated
    assert (output / "summary.json").exists()
    assert (output / "cpu/resources.json").exists()
    assert not (output / "mps").exists()


def test_probe_missing_assets_produces_failure_without_loading(tmp_path, monkeypatch):
    from tourism_ugc_study.segmentation import embedding_probe as probe

    def missing():
        raise FileNotFoundError("required_weight_missing")

    monkeypatch.setattr(probe.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(probe, "verify_local_assets", missing)
    result = probe.run_smoke(tmp_path / "probe")
    assert result["status"] == "failed"
    assert result["error_type"] == "FileNotFoundError"
    assert not result["devices"]
