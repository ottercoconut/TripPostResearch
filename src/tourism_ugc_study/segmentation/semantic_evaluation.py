"""固定开发成员并核查真实tokenizer单元，不加载embedding权重或评价语义效果。

四个可读正文码点长度层各20篇，用单一随机流和固定成员顺序抽取；另保留
最长帖作为压力样本。原文、旧切分和候选单元只写私有运行目录，公开摘要无原文。
这批材料用于调参，后续独立验收必须排除全部开发及压力成员。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Callable

from .api import segment_body
from .artifacts import canonical_json, new_run_directory, write_json
from .batch import runtime_identity
from .config import ResearchInput
from .embedding import EmbeddingError, QwenTokenCounter, identity_sha256
from .integrity import validate
from .projection import project_structured_text
from .semantic_config import (RUN_KIND, SemanticSettings, parse_semantic_settings,
                              validate_identity_settings, validate_semantic_rule_identity)
from .semantic_development_input import (
    INPUT_KIND, FrozenDevelopmentInput, development_units_sha256,
    frozen_development_input_identity, load_frozen_development_input,
    read_development_cases, read_frozen_development_input,
)
from .semantic_units import build_units
from .source import file_sha256, open_research_corpus

BUCKETS = (("1-100", 1, 100), ("101-300", 101, 300),
           ("301-1000", 301, 1000), ("1001+", 1001, None))


def prepare_development_sample(
    source: ResearchInput, output: Path, *, settings: SemanticSettings,
    seed: int = 20261011, per_bucket: int = 20, counter=None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """只读抽样并验证单元位置；任一来源变化或领域错误不伪造成功。

    counter仅供测试注入。默认只加载固定tokenizer/config，实际模型调用为0。
    小型测试语料可显式减小per_bucket；正式默认不以替补掩盖某层成员不足。
    """
    if type(seed) is not int or type(per_bucket) is not int or per_bucket < 1:
        raise ValueError("invalid_development_sampling")
    if Path(output).exists():
        raise FileExistsError("development_output_exists")
    runtime = runtime_identity()
    groups = {name: [] for name, _, _ in BUCKETS}
    posts, projections = {}, {}
    with open_research_corpus(source) as corpus:
        identity = corpus.identity
        for post in corpus.posts:
            projection = project_structured_text(post.body or "")
            if projection.status != "usable":
                raise ValueError("development_projection_failed")
            posts[post.source_post_id], projections[post.source_post_id] = post, projection.text
            for name, low, high in BUCKETS:
                if len(projection.text) >= low and (high is None or len(projection.text) <= high):
                    groups[name].append(post.source_post_id)
                    break
    rng = random.Random(seed)
    chosen = []
    for name, _, _ in BUCKETS:
        if len(groups[name]) < per_bucket:
            raise ValueError("insufficient_development_stratum")
        chosen.extend(sorted(rng.sample(sorted(groups[name]), per_bucket)))
    longest = max(projections, key=lambda pid: (len(projections[pid]), -pid))
    targeted = [] if longest in chosen else [longest]
    members = sorted(chosen + targeted)
    if counter is None:
        embedding = settings.embedding
        counter = QwenTokenCounter(model_id=embedding.model_id, revision=embedding.revision,
            max_input_tokens=embedding.max_input_tokens, instruction=embedding.instruction,
            local_files_only=True)
    counts = Counter()
    durations = []
    with new_run_directory(output) as staging:
        records = staging / "cases.jsonl"
        with records.open("x", encoding="utf-8") as stream:
            records.chmod(0o600)
            for i, pid in enumerate(members):
                started = time.monotonic()
                text = projections[pid]
                try:
                    built = build_units(text, counter, settings.algorithm)
                except (EmbeddingError, MemoryError, OSError):
                    raise
                except ValueError as exc:
                    unit_elapsed = time.monotonic() - started
                    counts["failed"] += 1
                    row = {"source_post_id": pid, "status": "failed", "reason": str(exc)}
                else:
                    unit_elapsed = time.monotonic() - started
                    old = segment_body(posts[pid].body)
                    row = {"source_post_id": pid, "status": "completed", "source_text": text,
                           "original_body": posts[pid].body, "source_version": posts[pid].source_version,
                           "units": [asdict(u) for u in built.units], "diagnostics": list(built.diagnostics),
                           "old_status": old.status, "old_reason_code": old.reason_code,
                           "old_segments": [asdict(s) for s in old.text_segments or ()]}
                    counts["completed"] += 1
                    counts["units"] += len(built.units)
                    key = f"posts_with_{len(built.units)}_units" if len(built.units) < 4 else "posts_with_4plus_units"
                    counts[key] += 1
                elapsed = time.monotonic() - started
                row["unitization_seconds"] = unit_elapsed
                row["total_case_seconds"] = elapsed
                row["targeted_longest"] = pid in targeted
                durations.append(unit_elapsed)
                stream.write(canonical_json(row) + "\n")
                if progress:
                    progress(i + 1, len(members))
        if hasattr(counter, "verify_assets"):
            counter.verify_assets()
        # 单元化可能耗时，重新验证配对源库，避免样本计算期间换源仍发布成功。
        with open_research_corpus(source) as corpus:
            if corpus.identity != identity:
                raise ValueError("development_source_changed")
            for _ in corpus.posts:
                pass
        if runtime_identity() != runtime:
            raise ValueError("runtime_changed_during_sample")
        manifest = {"artifact_kind": "semantic-development-sample", "status": "completed" if not counts["failed"] else "completed_with_failures",
            "input": identity, "seed": seed, "per_bucket": per_bucket, "bucket_counts": {k: len(v) for k, v in groups.items()},
            "sampling": "single_rng_sorted_strata_then_longest_pressure_case", "development_post_ids": chosen,
            "targeted_post_ids": targeted, "post_ids": members, "counts": dict(counts),
            "config": asdict(settings), "config_sha256": settings.fingerprint, "tokenizer_identity": counter.identity,
            "runtime": runtime, "cases_filename": records.name, "cases_sha256": file_sha256(records),
            "max_unitization_seconds": max(durations), "total_unitization_seconds": sum(durations),
            "actual_embedding_model_calls": 0, "semantic_quality_evaluated": False,
            "source_database_write_count": 0, "membership_database_write_count": 0}
        write_json(staging / "manifest.json", manifest)
    return manifest


def load_development_members(path: Path, source: ResearchInput | FrozenDevelopmentInput) -> dict:
    """共用冻结文件核验，再核对配对来源或显式离线包绑定。

    ResearchInput仍严格打开完整只读来源；FrozenDevelopmentInput只检查包，
    不把parent引用当远端已验证源库。文件漂移或正文／版本不一致拒绝。
    """
    path = Path(path)
    manifest_sha = file_sha256(path)
    value, rows = read_development_cases(path)
    if isinstance(source, FrozenDevelopmentInput):
        wrapper, _, _ = read_frozen_development_input(source)
        if (manifest_sha != wrapper["members_manifest_sha256"]
                or value["cases_sha256"] != wrapper["cases_sha256"]):
            raise ValueError("frozen_development_members_digest_mismatch")
    else:
        with open_research_corpus(source) as corpus:
            if corpus.identity != value.get("input"):
                raise ValueError("development_source_mismatch")
            available = {post.source_post_id: post for post in corpus.posts}
        if not set(value["post_ids"]) <= set(available):
            raise ValueError("development_members_mismatch")
        for row in rows:
            post = available[row["source_post_id"]]
            if row["original_body"] != post.body or row["source_version"] != post.source_version:
                raise ValueError("development_case_source_mismatch")
    if file_sha256(path) != manifest_sha:
        raise ValueError("development_manifest_changed")
    if file_sha256(path.parent / "cases.jsonl") != value["cases_sha256"]:
        raise ValueError("development_cases_changed")
    return value


def export_frozen_development_input(source: ResearchInput, members: Path, output: Path) -> dict:
    """从本地已配对核验的固定开发样本导出三文件离线包，不加载权重。

    原manifest/cases逐字节复制；只有新增清单记录导出核验与parent身份。
    输出必须新建，复制前后均核验原文件及本地来源，任何变化不发布半包。
    """
    if not isinstance(source, ResearchInput):
        raise ValueError("development_local_research_source_required")
    members = Path(members)
    source_manifest_sha = file_sha256(members)
    value = load_development_members(members, source)
    _, cases = read_development_cases(members)
    wrapper = {"artifact_kind": INPUT_KIND, "scope": "development", "status": "completed",
               "members_filename": "manifest.json", "members_manifest_sha256": source_manifest_sha,
               "cases_filename": "cases.jsonl", "cases_sha256": value["cases_sha256"],
               "parent_source_identity": value["input"],
               "parent_frozen_units_sha256": development_units_sha256(value["input"], value["post_ids"], cases),
               "post_ids": value["post_ids"], "seed": value["seed"],
               "source_validation": {"status": "passed", "scope": "local_paired_source_on_export"},
               "counts": {"post_count": len(cases), "unit_count": sum(len(row["units"]) for row in cases)},
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "actual_embedding_model_calls": 0, "semantic_quality_evaluated": False,
               "source_database_write_count": 0, "membership_database_write_count": 0}
    with new_run_directory(Path(output)) as staging:
        for original, filename in ((members, "manifest.json"), (members.parent / "cases.jsonl", "cases.jsonl")):
            target = staging / filename
            with target.open("xb") as stream:
                target.chmod(0o600)
                stream.write(original.read_bytes())
        write_json(staging / "input-manifest.json", wrapper)
        load_frozen_development_input(staging / "input-manifest.json")
        if (load_development_members(members, source) != value
                or file_sha256(members) != source_manifest_sha):
            raise ValueError("development_manifest_changed")
    return wrapper


@dataclass(frozen=True)
class FrozenDevelopment:
    """已经核对本次 tokenizer 的冻结成员和候选，供运行层只读使用。

    fingerprint 绑定来源、成员、正文摘要及候选跨度／token 数；不包含模型
    大小或窗口参数，因此跨模型比较必须各自实测候选一致后才能复用。
    """

    manifest: dict
    cases: tuple[dict, ...]
    fingerprint: str


def validate_counter_settings(counter, settings: SemanticSettings, *, model_identity: dict | None = None) -> None:
    """核对实际 tokenizer 的声明与资产，拒绝用错模型或计数器身份。

    model_identity 存在时，tokenizer 资产和依赖必须是其同名身份子集；
    device、dtype 等仅属于模型，不要求纯 tokenizer 虚构这些字段。
    """
    expected = {"model_id": settings.embedding.model_id,
                "model_revision": settings.embedding.revision,
                "tokenizer_revision": settings.embedding.revision,
                "max_input_tokens": settings.embedding.max_input_tokens,
                "instruction": settings.embedding.instruction}
    identity = getattr(counter, "identity", None)
    if (not isinstance(identity, dict) or any(identity.get(key) != value for key, value in expected.items())
            or counter.max_input_tokens != settings.embedding.max_input_tokens):
        raise ValueError("development_tokenizer_settings_mismatch")
    if model_identity is not None:
        for key in ("assets_sha256", "dependencies"):
            if any(model_identity.get(key, {}).get(name) != digest
                   for name, digest in identity.get(key, {}).items()):
                raise ValueError("development_tokenizer_model_identity_mismatch")
        for key in ("template", "native_normalizer", "offset_policy", "special_tokens",
                    "padding_side", "truncation", "eos_token_id", "pad_token_id"):
            if key in identity and identity[key] != model_identity.get(key):
                raise ValueError("development_tokenizer_model_identity_mismatch")


def validate_development_units(path: Path, source: ResearchInput | FrozenDevelopmentInput, *, settings: SemanticSettings,
                               counter=None) -> FrozenDevelopment:
    """重建同一冻结成员的候选并逐字段核对，不抽样、不加载 embedding 权重。

    跨模型 tokenizer 可以有不同身份，但同一正文的 start/end/token_count
    必须完全相同。变化时明确失败，由调用方另建开发证据，不能静默换单位。
    """
    manifest = load_development_members(path, source)
    frozen_settings = parse_semantic_settings(manifest["config"])
    if settings.algorithm.unit_max_tokens != frozen_settings.algorithm.unit_max_tokens:
        raise ValueError("development_unitization_config_mismatch")
    if counter is None:
        counter = QwenTokenCounter(**{key: value for key, value in asdict(settings.embedding).items()
                                   if key not in {"device", "dtype", "batch_size"}}, local_files_only=True)
    validate_counter_settings(counter, settings)
    cases = tuple(json.loads(line) for line in (path.parent / "cases.jsonl").read_text(encoding="utf-8").splitlines())
    manifest_sha = file_sha256(path)
    for row in cases:
        built = build_units(row["source_text"], counter, settings.algorithm)
        if [asdict(unit) for unit in built.units] != row["units"]:
            raise ValueError("development_frozen_units_mismatch")
        if list(built.diagnostics) != row.get("diagnostics"):
            raise ValueError("development_frozen_diagnostics_mismatch")
    if hasattr(counter, "verify_assets"):
        counter.verify_assets()
    if file_sha256(path.parent / "cases.jsonl") != manifest["cases_sha256"]:
        raise ValueError("development_cases_changed")
    if file_sha256(path) != manifest_sha:
        raise ValueError("development_manifest_changed")
    if isinstance(source, FrozenDevelopmentInput):
        read_frozen_development_input(source)
    fingerprint = development_units_sha256(manifest["input"], manifest["post_ids"], cases)
    return FrozenDevelopment(manifest, cases, fingerprint)


def verify_development_sample(source: ResearchInput, members: Path, output: Path, *,
                              settings: SemanticSettings, counter=None) -> dict:
    """保存另一模型／配置对同一冻结单位的 tokenizer 核验，不产生新成员。

    原始 cases 留在其私有目录；本核验清单绑定其SHA与当前计数器身份，
    不运行 embedding，不把技术核验标为主题审查通过。
    """
    if counter is None:
        counter = QwenTokenCounter(**{key: value for key, value in asdict(settings.embedding).items()
                                   if key not in {"device", "dtype", "batch_size"}}, local_files_only=True)
    frozen = validate_development_units(members, source, settings=settings, counter=counter)
    with new_run_directory(output) as staging:
        result = {"artifact_kind": "semantic-development-unit-verification", "status": "completed",
                  "input": frozen.manifest["input"], "post_ids": frozen.manifest["post_ids"],
                  "members_manifest_sha256": file_sha256(members), "cases_sha256": frozen.manifest["cases_sha256"],
                  "frozen_units_sha256": frozen.fingerprint, "config": asdict(settings),
                  "config_sha256": settings.fingerprint, "tokenizer_identity": counter.identity,
                  "counts": {"post_count": len(frozen.cases), "unit_count": sum(len(row["units"]) for row in frozen.cases)},
                  "actual_embedding_model_calls": 0, "semantic_quality_evaluated": False}
        load_development_members(members, source)
        write_json(staging / "manifest.json", result)
    return result


def validate_development_run(run: Path, source: ResearchInput | FrozenDevelopmentInput, *, members: Path,
                             counter=None, frozen_input: FrozenDevelopmentInput | None = None) -> dict:
    """运行后核对冻结单位、逐帖来源与 trace；允许有明确逐帖领域失败。

    本函数不判断主题质量，不读取向量或重算边界。用于原子发布之前的技术门。
    离线worker传FrozenDevelopmentInput；回传本地传ResearchInput并显式提供
    frozen_input，才能同时核验完整本地来源及实际包身份，不能借用parent声明。
    任何成员、正文、配置、模型摘要或单位漂移均抛 ValueError。
    """
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("artifact_kind") != RUN_KIND or manifest.get("scope") != "development"
            or manifest.get("status") not in {"completed", "completed_with_failures"}
            or manifest.get("posts_filename") != "posts.jsonl"
            or manifest.get("trace_filename") != "semantic-trace.jsonl"):
        raise ValueError("invalid_development_run")
    validate_semantic_rule_identity(manifest)
    settings = parse_semantic_settings(manifest.get("config"))
    if settings.fingerprint != manifest.get("config_sha256"):
        raise ValueError("development_run_config_digest_mismatch")
    model_identity = manifest.get("model_identity")
    validate_identity_settings(model_identity, settings.embedding)
    if identity_sha256(model_identity) != manifest.get("model_identity_sha256"):
        raise ValueError("development_run_model_digest_mismatch")
    frozen = validate_development_units(members, source, settings=settings, counter=counter)
    if (isinstance(source, FrozenDevelopmentInput) and frozen_input is not None
            and frozen_input.manifest_sha256 != source.manifest_sha256):
        raise ValueError("frozen_development_input_binding_mismatch")
    input_config = source if isinstance(source, FrozenDevelopmentInput) else frozen_input
    expected_input = frozen.manifest["input"]
    if input_config is not None:
        wrapper, _, _ = read_frozen_development_input(input_config)
        if (file_sha256(members) != wrapper["members_manifest_sha256"]
                or frozen.manifest["input"] != wrapper["parent_source_identity"]
                or frozen.fingerprint != wrapper["parent_frozen_units_sha256"]):
            raise ValueError("frozen_development_input_binding_mismatch")
        expected_input = frozen_development_input_identity(input_config)
    if (manifest.get("input") != expected_input
            or manifest.get("selected_post_ids") != frozen.manifest["post_ids"]):
        raise ValueError("development_run_members_source_mismatch")
    actual_units_sha = development_units_sha256(expected_input, frozen.manifest["post_ids"], frozen.cases)
    if input_config is not None and (manifest.get("frozen_units_sha256") != actual_units_sha
            or manifest.get("input_verification_scope") != "frozen_files_cases_and_tokenizer_units"):
        raise ValueError("frozen_development_run_units_digest_mismatch")
    for filename, key in (("posts.jsonl", "posts_sha256"), ("semantic-trace.jsonl", "trace_sha256")):
        if file_sha256(run / filename) != manifest.get(key):
            raise ValueError("development_run_file_digest_mismatch")
    posts = [json.loads(line) for line in (run / "posts.jsonl").read_text(encoding="utf-8").splitlines()]
    traces = [json.loads(line) for line in (run / "semantic-trace.jsonl").read_text(encoding="utf-8").splitlines()]
    if any([row.get("source_post_id") for row in rows] != frozen.manifest["post_ids"] for rows in (posts, traces)):
        raise ValueError("development_run_record_members_mismatch")
    for case, row, trace in zip(frozen.cases, posts, traces, strict=True):
        text_hash = hashlib.sha256(case["source_text"].encode()).hexdigest()
        if (row.get("original_body") != case["original_body"] or row.get("source_text") != case["source_text"]
                or row.get("source_version") != case["source_version"]
                or row.get("source_text_sha256") != text_hash or trace.get("source_text_sha256") != text_hash
                or trace.get("units") != case["units"]):
            raise ValueError("development_run_frozen_case_mismatch")
        if input_config is not None and (row.get("source_snapshot_sha256") != frozen.manifest["input"]["source_snapshot_sha256"]
                or row.get("source_snapshot_identity_scope") != "parent_source_reference"):
            raise ValueError("frozen_development_run_parent_reference_mismatch")
        if (row.get("semantic_diagnostics") != trace.get("summary")
                or trace["summary"].get("model_identity_sha256") != manifest["model_identity_sha256"]
                or trace["summary"].get("unit_count") != len(case["units"])):
            raise ValueError("development_run_trace_summary_mismatch")
        if row.get("status") == "completed":
            validate(case["source_text"], row.get("text_segments"))
        elif row.get("status") != "failed" or row.get("text_segments") is not None:
            raise ValueError("development_run_failed_segments_invalid")
    result = {"validation": "passed", "post_count": len(posts),
              "frozen_units_sha256": actual_units_sha,
              "members_manifest_sha256": file_sha256(members), "run_manifest_sha256": file_sha256(run / "manifest.json")}
    if input_config is not None:
        read_frozen_development_input(input_config)
        result.update(input_manifest_sha256=input_config.manifest_sha256,
                      parent_frozen_units_sha256=frozen.fingerprint,
                      source_validation_scope="frozen_input_only" if isinstance(source, FrozenDevelopmentInput)
                      else "local_paired_source_and_frozen_input")
    return result


def verify_frozen_development_run(source: ResearchInput, run: Path, output: Path, *,
                                  frozen_input: Path, members: Path, counter=None) -> dict:
    """将回传离线运行与本地配对来源重新核对，保存无原文核验记录。

    只加载本次run声明的tokenizer，不加载embedding权重或补算向量；原run和
    输入包只读且不改摘要。output必须新建，来源或文件变化不发布核验成功。
    本核验不重放边界、不评价主题，后续compare还须读取真实缓存复现基线。
    """
    if not isinstance(source, ResearchInput):
        raise ValueError("development_local_research_source_required")
    run = Path(run)
    config = load_frozen_development_input(frozen_input)
    manifest_sha = file_sha256(run / "manifest.json")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    settings = parse_semantic_settings(manifest.get("config"))
    if counter is None:
        counter = QwenTokenCounter(**{key: value for key, value in asdict(settings.embedding).items()
                                   if key not in {"device", "dtype", "batch_size"}}, local_files_only=True)
    validate_counter_settings(counter, settings, model_identity=manifest.get("model_identity"))
    with new_run_directory(Path(output)) as staging:
        validation = validate_development_run(run, source, members=members, counter=counter, frozen_input=config)
        if validation["run_manifest_sha256"] != manifest_sha:
            raise ValueError("development_run_changed")
        result = {"artifact_kind": "semantic-frozen-development-run-verification", "status": "completed",
                  **validation, "input": manifest["input"], "model_identity_sha256": manifest["model_identity_sha256"],
                  "config_sha256": settings.fingerprint, "tokenizer_identity": counter.identity,
                  "actual_embedding_model_calls": 0, "semantic_quality_evaluated": False,
                  "source_database_write_count": 0, "membership_database_write_count": 0}
        write_json(staging / "manifest.json", result)
        repeated = validate_development_run(run, source, members=members, counter=counter, frozen_input=config)
        if repeated != validation or file_sha256(run / "manifest.json") != manifest_sha:
            raise ValueError("development_run_changed")
    return result
