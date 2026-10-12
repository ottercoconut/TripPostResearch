"""缓存命中、身份隔离、损坏拒绝、原子发布及输入顺序验证。"""

import json
import stat

import numpy as np
import pytest

from tourism_ugc_study.segmentation.embedding import EmbeddingError
from tourism_ugc_study.segmentation.embedding_cache import EmbeddingCache


class FakeProvider:
    """记录真实编码输入，使用可区分向量验证缓存不会打乱顺序。"""

    def __init__(self):
        self.identity = {"dimension": 2, "instruction": "test",
                         "template": "Instruct: {instruction}\nQuery:{text}", "device": "cpu"}
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.array([[ord(text[0]), 1] for text in texts], dtype=np.float32)


def test_cache_deduplicates_missing_input_but_keeps_output_order(tmp_path):
    provider = FakeProvider()
    cache = EmbeddingCache(tmp_path, provider.identity)
    first = cache.encode(["乙", "甲", "乙"], provider)
    assert provider.calls == [["乙", "甲"]]
    assert first[:, 0].tolist() == [ord("乙"), ord("甲"), ord("乙")]
    assert cache.cache_miss_count == 3
    second = cache.encode(["甲", "乙"], provider)
    assert np.array_equal(second, first[[1, 0]])
    assert len(provider.calls) == 1
    assert cache.cache_hit_count == 2
    assert not list(cache.root.glob(".pending-*"))
    assert stat.S_IMODE(cache.root.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in cache.root.glob("*/*"))


def test_changed_provider_identity_rejected(tmp_path):
    provider = FakeProvider()
    cache = EmbeddingCache(tmp_path, provider.identity)
    provider.identity["device"] = "mps"
    with pytest.raises(EmbeddingError, match="cache_provider_identity_mismatch"):
        cache.encode(["甲"], provider)
    assert not provider.calls


@pytest.mark.parametrize("corruption", ["digest", "count", "order", "dtype", "missing"])
def test_cache_corruption_stops_without_reencoding(tmp_path, corruption):
    provider = FakeProvider()
    cache = EmbeddingCache(tmp_path, provider.identity)
    cache.encode(["甲"], provider)
    entry = next(cache.root.iterdir())
    manifest_path = entry / "manifest.json"
    metadata = json.loads(manifest_path.read_text())
    if corruption == "missing":
        (entry / "vector.npy").unlink()
    else:
        field, value = {"digest": ("vector_sha256", "0" * 64), "count": ("count", 2),
                        "order": ("index", 1), "dtype": ("dtype", "float64")}[corruption]
        metadata[field] = value
        manifest_path.write_text(json.dumps(metadata))
    with pytest.raises(EmbeddingError, match="invalid_cache_entry"):
        cache.encode(["甲"], provider)
    assert len(provider.calls) == 1


def test_invalid_provider_output_is_never_published(tmp_path):
    provider = FakeProvider()
    provider.encode = lambda texts: np.zeros((len(texts), 2), dtype=np.float32)
    cache = EmbeddingCache(tmp_path, provider.identity)
    with pytest.raises(EmbeddingError, match="zero_vector"):
        cache.encode(["甲"], provider)
    assert not list(cache.root.iterdir())


def test_empty_batch_does_not_call_provider(tmp_path):
    provider = FakeProvider()
    cache = EmbeddingCache(tmp_path, provider.identity)
    assert cache.encode([], provider).shape == (0, 2)
    assert not provider.calls


def test_readonly_cache_preserves_permissions_and_rejects_encode_or_missing_namespace(tmp_path):
    """只读重放不能靠mkdir／chmod改变证据，缺命名空间也不能创建空缓存。"""
    provider = FakeProvider()
    writable = EmbeddingCache(tmp_path / "cache", provider.identity)
    writable.encode(["甲", "乙"], provider)
    writable.root.chmod(0o755)
    before = [(path, path.stat().st_mode, path.stat().st_mtime_ns) for path in writable.root.rglob("*")]
    replay = EmbeddingCache(tmp_path / "cache", provider.identity, readonly=True)
    vectors = replay.read_cached(["乙", "甲", "乙"])
    assert vectors[:, 0].tolist() == [ord("乙"), ord("甲"), ord("乙")]
    assert len(provider.calls) == 1
    assert stat.S_IMODE(writable.root.stat().st_mode) == 0o755
    assert [(path, path.stat().st_mode, path.stat().st_mtime_ns) for path, _, _ in before] == before
    with pytest.raises(EmbeddingError, match="cache_readonly_encode_forbidden"):
        replay.encode(["甲"], provider)
    missing = tmp_path / "absent-cache"
    with pytest.raises(EmbeddingError, match="cache_namespace_missing"):
        EmbeddingCache(missing, provider.identity, readonly=True)
    assert not missing.exists()
