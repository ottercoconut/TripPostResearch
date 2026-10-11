"""固定开发案例的离线输入与文件身份，不访问完整来源或加载模型。

原抽样 manifest/cases 保持原字节；新增输入清单只记录文件绑定和本地导出
核验声明。包身份代表本次实际读取的案例文件，完整源库仅为 parent 引用。
本模块负责输入文件，来源核验和 tokenizer 规则仍由 semantic_evaluation 负责。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator, Sequence

from .embedding import identity_sha256
from .integrity import validate
from .projection import project_structured_text
from .semantic_config import parse_semantic_settings
from .source import ResearchCorpus, ResearchPost, file_sha256

INPUT_KIND = "semantic-frozen-development-input"
INPUT_DATASET_KIND = "frozen_development_cases"


@dataclass(frozen=True)
class FrozenDevelopmentInput:
    """仅携带输入清单路径及首次读取摘要，进程请求不复制原文。

    路径可迁移；SHA 固定首次接收的精确字节。每次加载与退出复核均验证
    该摘要及清单绑定的两文件，变化抛 ValueError，不重新接受新身份。
    """

    manifest: Path
    manifest_sha256: str


def development_units_sha256(input_identity: dict, post_ids: Sequence[int], cases: Sequence[dict]) -> str:
    """绑定实际输入身份及有序成员、正文、版本、候选，不包含模型大小。

    同一包跨模型必须先实测相同候选，再共享本摘要；parent 来源摘要另存，
    不能把旧全库身份计算的摘要冒充新包身份。
    """
    binding = [{"source_post_id": row["source_post_id"], "source_version": row["source_version"],
                "source_text_sha256": hashlib.sha256(row["source_text"].encode()).hexdigest(),
                "units": row["units"]} for row in cases]
    return identity_sha256({"input": input_identity, "post_ids": list(post_ids), "cases": binding})


def read_development_cases(path: Path) -> tuple[dict, tuple[dict, ...]]:
    """核验冻结样本自身的文件、成员、投影及无损跨度，不宣称核验源库。

    供在线来源核验和离线包共用，避免两套案例解析。只认可完整成功样本，
    保留正文原码点；文件中途变化、成员矛盾或损坏投影立即拒绝。
    """
    path = Path(path)
    raw = path.read_bytes()
    value = json.loads(raw)
    if (not isinstance(value, dict) or value.get("artifact_kind") != "semantic-development-sample"
            or value.get("status") != "completed" or not isinstance(value.get("input"), dict)):
        raise ValueError("invalid_development_manifest")
    cases_path = path.parent / "cases.jsonl"
    cases_raw = cases_path.read_bytes()
    if (value.get("cases_filename") != "cases.jsonl"
            or hashlib.sha256(cases_raw).hexdigest() != value.get("cases_sha256")):
        raise ValueError("development_cases_digest_mismatch")
    members = value.get("post_ids")
    if (not isinstance(members, list) or not members or any(type(i) is not int or i < 1 for i in members)
            or sorted(set(members)) != members or type(value.get("seed")) is not int):
        raise ValueError("invalid_development_members")
    development, targeted = value.get("development_post_ids"), value.get("targeted_post_ids")
    if any(not isinstance(ids, list) or any(type(i) is not int or i < 1 for i in ids)
           or len(set(ids)) != len(ids) for ids in (development, targeted)):
        raise ValueError("invalid_development_roles")
    if set(development) & set(targeted) or sorted(development + targeted) != members:
        raise ValueError("development_members_mismatch")
    rows = tuple(json.loads(line) for line in cases_raw.decode("utf-8").splitlines())
    if (any(not isinstance(row, dict) for row in rows)
            or [row.get("source_post_id") for row in rows] != members
            or any(row.get("status") != "completed" for row in rows)):
        raise ValueError("development_cases_members_mismatch")
    settings = parse_semantic_settings(value.get("config"))
    if value.get("config_sha256") != settings.fingerprint:
        raise ValueError("development_config_digest_mismatch")
    for row in rows:
        body = row.get("original_body")
        if ("original_body" not in row or (body is not None and not isinstance(body, str))
                or type(row.get("source_version")) is not int or row["source_version"] < 1
                or not isinstance(row.get("source_text"), str)):
            raise ValueError("development_case_source_mismatch")
        projected = project_structured_text(body or "")
        if projected.status != "usable" or row["source_text"] != projected.text:
            raise ValueError("development_case_source_mismatch")
        units = row.get("units")
        if (not isinstance(units, list) or any(not isinstance(unit, dict)
                or set(unit) != {"start", "end", "token_count"}
                or any(type(unit[key]) is not int for key in unit)
                or unit["token_count"] < 1 for unit in units)):
            raise ValueError("development_units_invalid")
        validate(projected.text, [{"seg_id": i, "raw_text": projected.text[unit["start"]:unit["end"]],
                                   "start": unit["start"], "end": unit["end"]}
                                  for i, unit in enumerate(units, 1)])
    if file_sha256(path) != hashlib.sha256(raw).hexdigest():
        raise ValueError("development_manifest_changed")
    if file_sha256(cases_path) != value["cases_sha256"]:
        raise ValueError("development_cases_changed")
    return value, rows


def read_frozen_development_input(config: FrozenDevelopmentInput) -> tuple[dict, dict, tuple[dict, ...]]:
    """读取一份固定包并复核三文件摘要，返回输入清单、原样本清单与案例。

    本地导出核验是被引用的历史声明，远端只能核验文件和案例本身。固定
    文件名防止输入清单意外指向包外来源；任何变动不能被重新标成已接收。
    """
    path = Path(config.manifest)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config.manifest_sha256:
        raise ValueError("frozen_development_input_changed")
    wrapper = json.loads(raw)
    if (not isinstance(wrapper, dict) or wrapper.get("artifact_kind") != INPUT_KIND
            or wrapper.get("scope") != "development" or wrapper.get("status") != "completed"
            or wrapper.get("members_filename") != "manifest.json"
            or wrapper.get("cases_filename") != "cases.jsonl"
            or wrapper.get("source_validation") != {"status": "passed", "scope": "local_paired_source_on_export"}):
        raise ValueError("invalid_frozen_development_input")
    members_path = path.parent / "manifest.json"
    if file_sha256(members_path) != wrapper.get("members_manifest_sha256"):
        raise ValueError("frozen_development_members_digest_mismatch")
    members, cases = read_development_cases(members_path)
    if (wrapper.get("cases_sha256") != members["cases_sha256"]
            or wrapper.get("post_ids") != members["post_ids"]
            or wrapper.get("seed") != members["seed"]
            or wrapper.get("parent_source_identity") != members.get("input")
            or wrapper.get("parent_frozen_units_sha256") != development_units_sha256(
                members["input"], members["post_ids"], cases)
            or wrapper.get("counts") != {"post_count": len(cases), "unit_count": sum(len(row["units"]) for row in cases)}):
        raise ValueError("frozen_development_input_binding_mismatch")
    if (file_sha256(path) != config.manifest_sha256
            or file_sha256(members_path) != wrapper["members_manifest_sha256"]
            or file_sha256(path.parent / "cases.jsonl") != wrapper["cases_sha256"]):
        raise ValueError("frozen_development_input_changed")
    return wrapper, members, cases


def load_frozen_development_input(path: Path) -> FrozenDevelopmentInput:
    """首次固定输入清单SHA并完整核验包；不读取数据库或模型资产。"""
    path = Path(path).resolve(strict=True)
    config = FrozenDevelopmentInput(path, file_sha256(path))
    read_frozen_development_input(config)
    return config


def frozen_development_input_identity(config: FrozenDevelopmentInput) -> dict:
    """返回实际案例包身份；源库SHA只放在独立parent引用中。"""
    wrapper, members, _ = read_frozen_development_input(config)
    return {"dataset_kind": INPUT_DATASET_KIND, "selection": "fixed_development_members",
            "record_count": len(members["post_ids"]), "input_manifest_sha256": config.manifest_sha256,
            "members_manifest_sha256": wrapper["members_manifest_sha256"],
            "cases_sha256": wrapper["cases_sha256"],
            "parent_source_identity": wrapper["parent_source_identity"],
            "parent_frozen_units_sha256": wrapper["parent_frozen_units_sha256"],
            "parent_source_verification": "local_export_reference_only"}


def frozen_development_parameters(config: FrozenDevelopmentInput, post_ids: tuple[int, ...] | None,
                                  seed: int | None, members_path: Path | None) -> tuple[tuple[int, ...], int, Path]:
    """自动采用包内全部成员；显式冲突拒绝，None绝不解释为全量运行。

    监督入口与同步批处理共用此解析，确保直接API调用也受开发范围限制。
    """
    wrapper, members, _ = read_frozen_development_input(config)
    ids = tuple(members["post_ids"])
    if post_ids is not None and (len(post_ids) != len(ids)
            or any(type(i) is not int for i in post_ids) or set(post_ids) != set(ids)):
        raise ValueError("frozen_development_members_mismatch")
    if seed is not None and (type(seed) is not int or seed != members["seed"]):
        raise ValueError("frozen_development_seed_mismatch")
    if members_path is not None and file_sha256(Path(members_path)) != wrapper["members_manifest_sha256"]:
        raise ValueError("frozen_development_members_digest_mismatch")
    return ids, members["seed"], Path(config.manifest).parent / "manifest.json"


@contextmanager
def open_frozen_development_corpus(config: FrozenDevelopmentInput) -> Iterator[ResearchCorpus]:
    """仅从冻结案例产生ResearchPost，退出复验输入，不打开SQLite。

    返回身份是实际包，source_post_id/version继承parent；调用方须在正常
    退出后发布结果。输入变动使退出失败，不能落盘成功manifest。
    """
    _, _, cases = read_frozen_development_input(config)
    identity = frozen_development_input_identity(config)
    yield ResearchCorpus(identity, (ResearchPost(row["source_post_id"], row["source_version"], row["original_body"])
                                   for row in cases))
    read_frozen_development_input(config)
