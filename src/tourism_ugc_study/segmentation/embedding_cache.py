"""单进程私有向量缓存：身份隔离、摘要校验与完整条目原子发布。

缓存不建业务表，不修复或覆盖损坏条目。每条记录一个精确模型输入的
float32 向量；重复输入复用同一条目，输出仍严格对应调用方输入顺序。
"""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import numpy as np

from .embedding import EmbeddingError, EmbeddingProvider, identity_sha256, validate_vectors


class EmbeddingCache:
    """以模型完整身份建立命名空间；累计 hit/miss 按输入位置计数。

    root 应由调用者放在 Git 忽略的私有目录。首版仅支持单进程单写者；
    并发运行必须使用不同 root。缓存失败向上抛出，不静默重算。
    """

    def __init__(self, root: str | Path, identity: dict[str, Any], *, readonly: bool = False) -> None:
        """绑定完整身份；readonly 模式只认已有命名空间且不改文件权限。"""
        self.identity = json.loads(json.dumps(identity, allow_nan=False))
        self.identity_hash = identity_sha256(self.identity)
        cache_parent = Path(root)
        self.root = cache_parent / self.identity_hash
        self.readonly = readonly
        if readonly:
            if not self.root.is_dir():
                raise EmbeddingError("cache_namespace_missing")
        else:
            cache_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            cache_parent.chmod(0o700)
            self.root.mkdir(parents=True, exist_ok=True)
            self.root.chmod(0o700)
        self.dimension = self.identity.get("dimension")
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension < 1:
            raise EmbeddingError("cache_identity_missing_dimension")
        self.cache_hit_count = 0
        self.cache_miss_count = 0

    @property
    def cache_hit(self) -> int:
        """累计已存在且验证通过的输入数。"""
        return self.cache_hit_count

    @property
    def cache_miss(self) -> int:
        """累计需本次生成的输入数，包含同次重复输入位置。"""
        return self.cache_miss_count

    def _input_key(self, text: str) -> str:
        """用命名空间中冻结的模板重建精确输入，仅保存其摘要。"""
        template = self.identity.get("template")
        if template != "Instruct: {instruction}\nQuery:{text}":
            raise EmbeddingError("unsupported_cache_input_template")
        value = template.format(instruction=self.identity["instruction"], text=text)
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _read(self, path: Path, key: str) -> np.ndarray:
        """验证 manifest、文件摘要、dtype/形状和向量；不加载 pickle。"""
        try:
            metadata = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            if (path / "vector.npy").stat().st_size > self.dimension * 4 + 4096:
                raise EmbeddingError("cache_file_size_exceeded")
            raw = (path / "vector.npy").read_bytes()
            expected = {"schema": 1, "identity_sha256": self.identity_hash,
                        "input_sha256": key, "index": 0, "count": 1,
                        "dimension": self.dimension, "dtype": "float32"}
            if any(metadata.get(k) != v for k, v in expected.items()):
                raise EmbeddingError("cache_metadata_mismatch")
            if metadata.get("vector_sha256") != hashlib.sha256(raw).hexdigest():
                raise EmbeddingError("cache_digest_mismatch")
            value = np.load(BytesIO(raw), allow_pickle=False)
            if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32"):
                raise EmbeddingError("cache_dtype_mismatch")
            return validate_vectors(value, count=1, dimension=self.dimension)[0]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError(f"invalid_cache_entry:{key}:{type(exc).__name__}:{exc}") from exc

    def _write(self, key: str, vector: np.ndarray) -> None:
        """先写完整临时目录，再 rename；已有不同内容绝不覆盖。"""
        if self.readonly:
            raise EmbeddingError("cache_readonly")
        path = self.root / key
        if path.exists():
            if not np.array_equal(self._read(path, key), vector):
                raise EmbeddingError("cache_existing_vector_conflict")
            return
        temp = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.root))
        try:
            np.save(temp / "vector.npy", vector[None, :], allow_pickle=False)
            (temp / "vector.npy").chmod(0o600)
            metadata = {"schema": 1, "identity_sha256": self.identity_hash,
                        "input_sha256": key, "index": 0, "count": 1,
                        "dimension": self.dimension, "dtype": "float32",
                        "vector_sha256": hashlib.sha256((temp / "vector.npy").read_bytes()).hexdigest()}
            (temp / "manifest.json").write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
            (temp / "manifest.json").chmod(0o600)
            self._read(temp, key)
            temp.rename(path)
        finally:
            if temp.exists():
                shutil.rmtree(temp)

    def read_cached(self, texts: Sequence[str]) -> np.ndarray:
        """只读返回精确输入的已有向量；缺条目立即失败，绝不调用 provider。

        与 encode 共用输入 key 和逐条验证；不创建目录或写入条目。重复输入
        按位置返回，每个唯一条目只读取一次；成功后才累计命中数量。
        """
        if isinstance(texts, (str, bytes)):
            raise EmbeddingError("texts_must_be_sequence")
        values = list(texts)
        if any(not isinstance(text, str) or not text for text in values):
            raise EmbeddingError("empty_or_invalid_text")
        keys = [self._input_key(text) for text in values]
        found = {}
        for key in keys:
            if key not in found:
                path = self.root / key
                if not path.is_dir():
                    raise EmbeddingError("cache_entry_missing")
                found[key] = self._read(path, key)
        vectors = (np.stack([found[key] for key in keys]) if keys else
                   np.empty((0, self.dimension), dtype=np.float32))
        result = validate_vectors(vectors, count=len(keys), dimension=self.dimension)
        self.cache_hit_count += len(keys)
        return result

    def encode(self, texts: Sequence[str], provider: EmbeddingProvider) -> np.ndarray:
        """返回按原输入顺序排列的有效向量；一次 encode 仅推理缺失唯一输入。"""
        if self.readonly:
            raise EmbeddingError("cache_readonly_encode_forbidden")
        if identity_sha256(provider.identity) != self.identity_hash:
            raise EmbeddingError("cache_provider_identity_mismatch")
        verify_assets = getattr(provider, "verify_assets", None)
        if verify_assets is not None:
            verify_assets(strong=False)
        if isinstance(texts, (str, bytes)):
            raise EmbeddingError("texts_must_be_sequence")
        values = list(texts)
        if any(not isinstance(text, str) or not text for text in values):
            raise EmbeddingError("empty_or_invalid_text")
        keys = [self._input_key(text) for text in values]
        found: dict[str, np.ndarray] = {}
        missing: dict[str, str] = {}
        for key, text in zip(keys, values, strict=True):
            path = self.root / key
            if path.exists():
                if key not in found:
                    found[key] = self._read(path, key)
                self.cache_hit_count += 1
            else:
                missing.setdefault(key, text)
                self.cache_miss_count += 1
        if missing:
            vectors = validate_vectors(provider.encode(list(missing.values())), count=len(missing),
                                       dimension=self.dimension)
            for (key, _), vector in zip(missing.items(), vectors, strict=True):
                self._write(key, vector)
                found[key] = vector
        if not values:
            return np.empty((0, self.dimension), dtype=np.float32)
        return validate_vectors(np.stack([found[key] for key in keys]), count=len(keys),
                                dimension=self.dimension)
