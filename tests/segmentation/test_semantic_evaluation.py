"""开发样本分层、只读来源和成员清单绑定的合成测试。"""

from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

import pytest

from tourism_ugc_study.segmentation.config import ResearchInput
from tourism_ugc_study.segmentation.semantic_config import (
    EmbeddingConfig, SemanticConfig, SemanticSettings, validate_identity_settings,
)
from tourism_ugc_study.segmentation.semantic_evaluation import (
    load_development_members, prepare_development_sample,
)
from tourism_ugc_study.segmentation.source import file_sha256


class CharacterCounter:
    """只用于测试的精确逐码点计数，不冒称真实 tokenizer 验证。"""

    max_input_tokens = 256

    def __init__(self):
        settings = EmbeddingConfig()
        self.identity = {
            "provider": "qwen-transformers-local-v1", "model_id": settings.model_id,
            "model_revision": settings.revision, "tokenizer_revision": settings.revision,
            "max_input_tokens": settings.max_input_tokens, "instruction": settings.instruction,
            "template": "Instruct: {instruction}\nQuery:{text}", "dimension": 1024,
            "special_tokens": "tokenizer_native_no_manual_eos", "padding_side": "left",
            "truncation": False, "pooling": "last_valid_token", "normalization": "l2_float32",
            "quantization": None, "fixture_only": True,
        }

    def count(self, text):
        return len(text)

    def input_count(self, text):
        return len(text) + 12

    def offsets(self, text):
        return tuple((i, i + 1) for i in range(len(text)))


def make_source(tmp_path: Path) -> ResearchInput:
    """四层各两篇，并加入一篇应被排除的更长正文。"""
    source = tmp_path / "source.sqlite"
    members = tmp_path / "members.sqlite"
    manifest_path = tmp_path / "manifest.json"
    rows = [(i + 1, "海" * size, "keep") for i, size in enumerate((20, 40, 150, 200, 500, 800, 1200, 1400))]
    rows.append((9, "海" * 2000, "exclude"))
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE web_posts(id INTEGER PRIMARY KEY,platform_key TEXT,title TEXT,content_text TEXT)")
        connection.executemany("INSERT INTO web_posts VALUES (?, 'xhs', NULL, ?)", [(i, body) for i, body, _ in rows])
    manifest = {
        "artifact_kind": "full-research-cleaning-candidates",
        "status": "CANDIDATES_READY_AWAITING_HUMAN_REVIEW", "database_filename": members.name,
        "source_snapshot_sha256": file_sha256(source), "round_manifest_sha256": "b" * 64,
        "round_name": "synthetic-development", "record_count": 9,
        "decision_counts": {"keep": 8, "exclude": 1, "manual_review": 0},
    }
    with sqlite3.connect(members) as connection:
        connection.executescript("""
            CREATE TABLE cleaning_candidates(source_post_id INTEGER,source_version INTEGER,
                platform_key TEXT,title TEXT,body TEXT,decision TEXT);
            CREATE TABLE round_metadata(key TEXT PRIMARY KEY,value_json TEXT);
        """)
        connection.executemany("INSERT INTO cleaning_candidates VALUES (?,1,'xhs','',?,?)", rows)
        connection.executemany("INSERT INTO round_metadata VALUES (?,?)", [
            (key, json.dumps(manifest[key])) for key in
            ("status", "round_name", "source_snapshot_sha256", "round_manifest_sha256")
        ])
    manifest["database_sha256"] = file_sha256(members)
    manifest_path.write_text(json.dumps(manifest))
    return ResearchInput(members, source, manifest_path, file_sha256(manifest_path))


def settings():
    return SemanticSettings(SemanticConfig(), EmbeddingConfig())


def test_development_sampling_is_reproducible_readonly_and_contains_longest(tmp_path):
    source = make_source(tmp_path)
    before = {path: file_sha256(path) for path in (source.source_db, source.membership_db, source.candidate_manifest)}
    first = prepare_development_sample(source, tmp_path / "one", settings=settings(),
                                       per_bucket=1, counter=CharacterCounter())
    second = prepare_development_sample(source, tmp_path / "two", settings=settings(),
                                        per_bucket=1, counter=CharacterCounter())
    assert first["status"] == "completed"
    assert first["post_ids"] == second["post_ids"]
    assert first["bucket_counts"] == {"1-100": 2, "101-300": 2, "301-1000": 2, "1001+": 2}
    assert len(first["development_post_ids"]) == 4
    assert 8 in first["post_ids"] and 9 not in first["post_ids"]
    assert first["actual_embedding_model_calls"] == 0
    assert first["semantic_quality_evaluated"] is False
    assert {path: file_sha256(path) for path in before} == before
    assert load_development_members(tmp_path / "one/manifest.json", source)["post_ids"] == first["post_ids"]
    assert not list(tmp_path.glob("*.sqlite-*"))


def test_source_change_during_unitization_prevents_publication(tmp_path):
    source = make_source(tmp_path)

    def mutate_after_read(done, total):
        if done == 1:
            with source.candidate_manifest.open("a") as stream:
                stream.write(" ")

    with pytest.raises(ValueError):
        prepare_development_sample(source, tmp_path / "run", settings=settings(), per_bucket=1,
                                   counter=CharacterCounter(), progress=mutate_after_read)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("tamper", ["add_nonmember", "drop_case", "replace_role"])
def test_development_member_manifest_cannot_diverge_from_cases(tmp_path, tamper):
    source = make_source(tmp_path)
    output = tmp_path / "sample"
    value = prepare_development_sample(source, output, settings=settings(),
                                       per_bucket=1, counter=CharacterCounter())
    if tamper == "add_nonmember":
        value["post_ids"].append(9999)
    elif tamper == "drop_case":
        value["post_ids"].pop()
    else:
        value["development_post_ids"] = [9]
    (output / "manifest.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_development_members(output / "manifest.json", source)


def test_provider_identity_validation_distinguishes_tokenizer_from_model():
    declared = EmbeddingConfig()
    identity = CharacterCounter().identity
    with pytest.raises(ValueError, match="embedding_identity_settings_mismatch"):
        validate_identity_settings(identity, declared)
    identity.update({key: asdict(declared)[key] for key in ("device", "dtype", "batch_size")})
    identity.update(attention_backend="eager", use_cache=False)
    validate_identity_settings(identity, declared)
    identity["max_input_tokens"] += 1
    with pytest.raises(ValueError, match="embedding_identity_settings_mismatch"):
        validate_identity_settings(identity, declared)
