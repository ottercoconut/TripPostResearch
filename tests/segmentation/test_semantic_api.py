"""显式语义接口的投影、短帖、系统故障和无损契约；仅使用合成向量。"""

from dataclasses import asdict
import hashlib
import json

import numpy as np
import pytest

from tourism_ugc_study.segmentation.embedding import EmbeddingError
from tourism_ugc_study.segmentation.embedding_cache import EmbeddingCache
from tourism_ugc_study.segmentation.integrity import validate
from tourism_ugc_study.segmentation.semantic_api import segment_body_semantic
from tourism_ugc_study.segmentation.semantic_config import EmbeddingConfig, CoherentSemanticConfig


class SyntheticProvider:
    """实现完整计数协议并输出可手算的固定向量，不加载任何真实模型。

    身份字段与配置对齐以检验编排层校验，test_fixture明确其为合成替身。
    甲和乙正交、乙和丙同向；其他文本用相同向量。只有前两维非零。
    """

    def __init__(self, settings=None, *, fail_on_call=None):
        """保存身份与调用证据，可指定一次模型调用模拟系统性故障。"""
        settings = settings or EmbeddingConfig(device="cpu")
        dimension = {"0.6B": 1024, "4B": 2560, "8B": 4096}[settings.model_id.rsplit("-", 1)[1]]
        self.max_input_tokens = settings.max_input_tokens
        self.identity = {
            "provider": "qwen-transformers-local-v1", "test_fixture": True,
            "model_id": settings.model_id, "model_revision": settings.revision,
            "tokenizer_revision": settings.revision, "dimension": dimension,
            "instruction": settings.instruction, "template": "Instruct: {instruction}\nQuery:{text}",
            "special_tokens": "tokenizer_native_no_manual_eos", "padding_side": "left",
            "max_input_tokens": settings.max_input_tokens, "truncation": False,
            "pooling": "last_valid_token", "normalization": "l2_float32", "quantization": None,
            "device": settings.device, "dtype": settings.dtype, "attention_backend": "eager",
            "batch_size": settings.batch_size,
            "use_cache": False, "assets_sha256": {"synthetic-fixture": "a" * 64},
            "dependencies": {"synthetic-only": "1"},
        }
        if settings.device == "cuda":
            self.identity.update(padding_strategy="max_length", padding_max_length=settings.max_input_tokens)
        self.calls = []
        self.model_call_count = 0
        self.asset_checks = 0
        self.fail_on_call = fail_on_call

    def count(self, text):
        """合成tokenizer每码点一个token，便于检验精确位置和段长。"""
        return len(text)

    def input_count(self, text):
        """合成模板固定占8个token；不借真实模型作单元测试。"""
        return len(text) + 8

    def offsets(self, text):
        """按原文顺序逐码点给出完整位置，供长单元字素保护测试。"""
        return tuple((index, index + 1) for index in range(len(text)))

    def encode(self, texts):
        """记录输入并产生单位向量，指定故障不返回半份输出。"""
        self.calls.append(tuple(texts))
        self.model_call_count += 1
        if self.model_call_count == self.fail_on_call:
            raise EmbeddingError("synthetic_model_failure:private-raw-marker")
        vectors = np.zeros((len(texts), self.identity["dimension"]), dtype=np.float32)
        for index, text in enumerate(texts):
            vectors[index, int(text.startswith(("乙", "丙")))] = 1
        return vectors

    def verify_assets(self, **kwargs):
        """记录编排层是否执行资产复验，真实资产检验由provider单测覆盖。"""
        self.asset_checks += 1


def semantic(body, provider=None, **kwargs):
    """给接口提供一致的合成provider及计数器，避免误用真实加载默认值。"""
    provider = provider or SyntheticProvider()
    return segment_body_semantic(body, provider=provider, token_counter=provider,
                                 config=CoherentSemanticConfig(), **kwargs)


@pytest.mark.parametrize("body,decision", [(None, "empty"), ("", "empty"), (" \r\n\t", "empty"), ("海边真好看。", "single_unit")])
def test_zero_or_one_unit_skips_embedding_and_keeps_projection_reason(body, decision):
    """N≤1无逐帖模型调用，NULL与模型决策分别保存，不混淆两个原因域。"""
    provider = SyntheticProvider(fail_on_call=1)
    output = semantic(body, provider)
    assert provider.calls == [] and output.model_call_count == 0
    assert output.result.status == "completed" and output.decision.status == decision
    assert output.embedded_unit_count == 0 and output.vector_content_sha256 is None
    assert output.result.body_was_null == (body is None)
    assert output.result.reason_code == ("source_body_null" if body is None else "plain_text")
    assert output.result.source_text == ("" if body is None else body)
    if decision == "empty":
        assert output.result.text_segments == ()
        assert [issue.code for issue in output.result.issues] == ["EMPTY_BODY"]


def test_two_units_can_split_without_exceeding_length_limit():
    """两个足够长的正交句子由绝对阈值切开，旧超长门槛不再阻止评分。"""
    body = "甲" * 15 + "。" + "乙" * 15 + "。"
    provider = SyntheticProvider()
    output = semantic(body, provider)
    assert output.result.status == "completed" and output.decision.status == "scored_split"
    assert output.decision.semantic_boundaries == (0,)
    assert len(output.result.text_segments) == 2 and len(output.units) == 2
    assert provider.calls == [("甲" * 15 + "。", "乙" * 15 + "。")]
    assert output.summary()["embedded_unit_count"] == 2


def test_three_units_match_endpoint_valley_and_short_edges_are_merged():
    """接口不改变纯计算N=3决策；模型选中的切点若留下过短首末段，则并回相邻段。"""
    body = "甲" * 15 + "。" + "乙" * 15 + "。" + "丙" * 15 + "。"
    output = semantic(body)
    assert output.decision.semantic_boundaries == (0,)
    assert len(output.result.text_segments) == 2
    short = semantic("甲。乙。")
    assert short.decision.status == "scored_keep" and short.embedded_unit_count == 2
    assert [segment.raw_text for segment in short.result.text_segments] == ["甲。乙。"]
    assert short.decision.gaps[0].reason == "meaning_boundary_suppressed"


def test_delta_body_projection_hash_and_original_offsets_are_preserved():
    """JSON正文与可读投影各有摘要，图片仅按既有投影规则忽略，span不指向JSON。"""
    first = "甲" * 12 + "👨‍👩‍👧‍👦。\r\n"
    second = "乙" * 15 + "。"
    body = json.dumps({"ops": [{"insert": first}, {"insert": {"image": "synthetic-image"}}, {"insert": second}]})
    output = semantic(body)
    result = output.result
    assert result.source_text == first + second
    assert result.format_id == "quill_delta_json" and result.ignored_embed_count == 1
    assert result.text_insert_count == 2
    assert result.source_body_sha256 == hashlib.sha256(body.encode()).hexdigest()
    assert result.source_text_sha256 == hashlib.sha256((first + second).encode()).hexdigest()
    assert result.source_body_sha256 != result.source_text_sha256
    validate(result.source_text, [asdict(segment) for segment in result.text_segments])
    assert result.text_segments[-1].start == len(first)


@pytest.mark.parametrize("body", ['{"ops":broken}', '{"ops":[{"insert":{"unknown":1}}]}'])
def test_bad_projection_is_per_post_failure_and_never_calls_embedding(body):
    """投影失败是明确的单帖失败，不能成为空成功或旧规则回退。"""
    provider = SyntheticProvider(fail_on_call=1)
    output = semantic(body, provider)
    assert output.result.status == "failed" and output.result.text_segments is None
    assert output.result.source_text is None and output.decision.status == "failed"
    assert provider.calls == []
    assert output.decision.reason_codes == (output.result.reason_code,)


def test_provider_and_invalid_vector_failures_escape_to_stop_batch():
    """系统故障向调度层抛出，不能伪装成可继续的单帖failed。"""
    with pytest.raises(EmbeddingError, match="synthetic_model_failure"):
        semantic("甲。乙。", SyntheticProvider(fail_on_call=1))
    provider = SyntheticProvider()
    provider.encode = lambda texts: np.zeros((len(texts), provider.identity["dimension"]), dtype=np.float32)
    with pytest.raises(EmbeddingError, match="zero_vector"):
        semantic("甲。乙。", provider)


@pytest.mark.parametrize("failure", [MemoryError, OSError, EmbeddingError])
def test_tokenizer_system_failures_escape_instead_of_becoming_per_post_errors(failure):
    """token计数中的资产、内存或I/O故障同样必须终止attempt。"""
    provider = SyntheticProvider()

    def broken_count(text):
        """模拟计数依赖损坏；不将异常换成合成计数继续执行。"""
        raise failure("synthetic-system-failure")

    provider.count = broken_count
    with pytest.raises(failure, match="synthetic-system-failure"):
        semantic("甲。乙。", provider)


def test_cached_replay_retains_semantic_participation_but_reports_no_new_model_call(tmp_path):
    """缓存命中仍计入参与语义评分的单元，不能误称第二次重新运行了模型。"""
    provider = SyntheticProvider()
    cache = EmbeddingCache(tmp_path / "cache", provider.identity)
    first = semantic("甲。乙。", provider, cache=cache)
    second = semantic("甲。乙。", provider, cache=cache)
    assert first.result == second.result and first.decision == second.decision
    assert first.vector_content_sha256 == second.vector_content_sha256
    assert first.model_call_count == 1 and second.model_call_count == 0
    assert first.cache_hit_count == 0 and second.cache_hit_count == 2
    assert first.embedded_unit_count == second.embedded_unit_count == 2


def test_corrupt_cache_raises_without_model_retry(tmp_path):
    """缓存损坏不会静默重新推理，已有原文及旧缓存保持可调查。"""
    provider = SyntheticProvider()
    cache = EmbeddingCache(tmp_path / "cache", provider.identity)
    semantic("甲。乙。", provider, cache=cache)
    vector = next(cache.root.glob("*/vector.npy"))
    vector.write_bytes(b"synthetic-corruption")
    with pytest.raises(EmbeddingError, match="invalid_cache_entry"):
        semantic("甲。乙。", provider, cache=cache)
    assert provider.model_call_count == 1


def test_long_unpunctuated_body_is_kept_whole_as_length_exception():
    """无标点无空白的长帖只有句内位置，长度兜底不切断，整段保留并标记异常。"""
    body = "甲" * 1100
    provider = SyntheticProvider()
    output = semantic(body, provider)
    assert output.result.status == "completed" and output.decision.status == "scored_keep"
    assert "length_exception" in output.decision.reason_codes
    assert len(output.units) > 1 and all(len(text) <= 64 for text in provider.calls[0])
    assert [segment.raw_text for segment in output.result.text_segments] == [body]
    validate(body, [asdict(segment) for segment in output.result.text_segments])


def test_space_separated_long_body_uses_spaces_only_as_length_fallback():
    """以空格代替标点的长文，长度兜底可在空白处补切，覆盖全文且不超长。"""
    body = " ".join(["甲" * 40] * 30)
    output = semantic(body)
    assert output.decision.status == "length_split"
    assert "length_exception" not in output.decision.reason_codes
    assert all(len(segment.raw_text) <= 512 for segment in output.result.text_segments)
    validate(body, [asdict(segment) for segment in output.result.text_segments])


def test_nonstring_body_is_rejected_before_any_work():
    """调用契约错误不经JSON字符串化变成正文。"""
    with pytest.raises(TypeError):
        semantic({"content": "错误调用"})
