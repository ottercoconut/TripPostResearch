"""向量输入、末 token pooling、身份与无截断契约的确定性验证。"""

from types import SimpleNamespace

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
