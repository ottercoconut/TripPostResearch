"""公开纯函数接口的无损、失败状态和依赖隔离测试；只使用合成正文。"""

from dataclasses import FrozenInstanceError
import hashlib
import json
import subprocess
import sys

import pytest

from tourism_ugc_study.segmentation import segment_body, segment_text
from tourism_ugc_study.segmentation.engine import validate


def test_offsets_hashes_and_immutable_results():
    """组合emoji与CRLF保持原样，码点位置可直接回切可读投影。"""
    body = "  👨‍👩‍👧‍👦出游。\r\n收费18.5元！ "
    result = segment_body(body)
    assert result.status == "completed"
    assert result.source_text == body
    assert result.source_body_sha256 == result.source_text_sha256 == hashlib.sha256(body.encode()).hexdigest()
    assert [s.raw_text for s in result.text_segments] == ["👨‍👩‍👧‍👦出游。", "收费18.5元！"]
    for item in result.text_segments:
        assert body[item.start:item.end] == item.raw_text
    with pytest.raises(FrozenInstanceError):
        result.text_segments[0].start = 0


def test_delta_projection_preserves_text_and_ignores_declared_embeds():
    """只拼接文本insert，原JSON和可读投影分别具有摘要。"""
    body = json.dumps({"ops": [
        {"insert": "甲。\r\n", "attributes": {"bold": True}},
        {"insert": {"image": "private-placeholder"}}, {"insert": "乙🌊。"},
    ]}, ensure_ascii=False)
    result = segment_body(body)
    assert result.status == "completed"
    assert result.source_text == "甲。\r\n乙🌊。"
    assert result.text_insert_count == 2 and result.ignored_embed_count == 1
    assert result.source_text_sha256 != result.source_body_sha256
    assert [s.raw_text for s in result.text_segments] == ["甲。", "乙🌊。"]


@pytest.mark.parametrize("body,reason", [
    ('{"ops": broken}', "structured_text_malformed"),
    ('{"ops": null}', "structured_text_operations_invalid"),
    ('{"ops":[{"retain":1}]}', "structured_text_operation_invalid"),
    ('{"ops":[{"insert":"甲","attributes":null}]}', "structured_text_operation_invalid"),
    ('{"ops":[{"insert":{"unknown":1}}]}', "structured_text_embed_unsupported"),
    ('{"ops":[{"insert":{"image":"x"}}]}', "structured_text_no_text"),
    ('{"ops":[{"insert":"\\ufeff\\u200b\\ufe0f\\t"}]}', "structured_text_no_text"),
])
def test_failed_projection_never_becomes_empty_success(body, reason):
    """损坏输入不能伪装为空片段成功，也不能泄漏部分投影结果。"""
    result = segment_body(body)
    assert result.status == "failed" and result.reason_code == reason
    assert result.text_segments is None and result.source_text is None


@pytest.mark.parametrize("body", [None, "", " \r\n\t"])
def test_empty_and_null_are_explicit(body):
    """空白成功输出空元组，NULL保留独立原因而不借标题补正文。"""
    result = segment_body(body)
    assert result.status == "completed" and result.text_segments == ()
    assert result.body_was_null == (body is None)
    assert [i.code for i in result.issues] == ["EMPTY_BODY"]
    if body is None:
        assert result.reason_code == "source_body_null" and result.source_body_sha256 is None


def test_plain_text_entry_does_not_parse_json():
    """已投影入口不把JSON外观的原文再次解释为文档结构。"""
    text = '{"ops": broken}'
    assert segment_text(text).source_text == text
    assert segment_body(text).status == "failed"
    with pytest.raises(TypeError):
        segment_text(None)
    with pytest.raises(TypeError):
        segment_body(12)


def test_integrity_failure_has_no_partial_segments(monkeypatch):
    """引擎不变量失败是单条失败，不能返回部分片段。"""
    def broken(text):
        raise ValueError("synthetic integrity failure")
    monkeypatch.setattr("tourism_ugc_study.segmentation.api.segment", broken)
    result = segment_text("甲。")
    assert result.status == "failed" and result.text_segments is None
    assert result.reason_code == "segment_integrity_failed"


def test_validator_rejects_lost_text():
    """主动丢失非空白字符必须被拒绝。"""
    with pytest.raises(ValueError, match="丢失"):
        validate("甲乙。", [{"seg_id": 1, "start": 1, "end": 3, "raw_text": "乙。"}])


@pytest.mark.parametrize("text,expected", [
    ("甲？\u200c乙。", ["甲？\u200c", "乙。"]),
    ("甲！\u200d\u200d乙。", ["甲！\u200d\u200d", "乙。"]),
    ("甲?\ufe0f乙。", ["甲?\ufe0f", "乙。"]),
    ("甲。\u200c\u200c乙。", ["甲。\u200c\u200c", "乙。"]),
    ("甲。 \ufe0f乙。", ["甲。", " \ufe0f乙。"]),
    (" \u0f80\u0f72甲。", [" \u0f80\u0f72甲。"]),
    (" \u0301\n乙。", [" \u0301", "乙。"]),
    ("甲。\u0301 \u0301乙。", ["甲。\u0301", " \u0301乙。"]),
])
def test_grapheme_protection_at_punctuation_and_trim(text, expected):
    """覆盖全量接入新发现的标点附加符及空格附加符，原文逐字不丢失。"""
    result = segment_text(text)
    assert result.status == "completed"
    assert [item.raw_text for item in result.text_segments] == expected
    assert result.implementation_version == "objective-structure-0.5-grapheme-safe-1"
    assert all(text[item.start:item.end] == item.raw_text for item in result.text_segments)


def test_pure_entry_is_independent_even_under_optimized_python():
    """新解释器中不导入清洗/模型，优化模式也不能关闭完整性检查。"""
    code = """
import sys
from tourism_ugc_study.segmentation import segment_body
from tourism_ugc_study.segmentation.engine import validate
if segment_body('甲。乙。').status != 'completed':
    raise RuntimeError('独立入口未成功执行')
if any(name.startswith(('tourism_ugc_study.cleaning', 'tourism_ugc_study.models',
                        'numpy', 'sklearn', 'torch', 'yaml', 'sqlite3')) for name in sys.modules):
    raise RuntimeError('切分入口加载了无关依赖')
try:
    validate('甲', [])
except ValueError:
    pass
else:
    raise RuntimeError('优化模式下完整性守卫失效')
"""
    subprocess.run([sys.executable, "-O", "-c", code], check=True)
