"""片段单表发布包的全量复核与随机抽查材料生成。

核验只读数据库与原始冻结来源逐字对应，不调用边界算法产生新结果。抽样是
固定种子的帖子级简单随机抽样；程序生成材料，审阅结论必须另行逐篇填写。
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from collections import Counter
from contextlib import closing
from itertools import zip_longest
from pathlib import Path

from .artifacts import canonical_json, new_run_directory, write_json
from .config import ResearchInput, load_research_input
from .import_validation import validate_post_record, validate_run_manifest
from .projection import project_structured_text
from .source import file_sha256, open_research_corpus
from .storage import get_post_segments, open_segment_database, segment_identifier


def _load_bundle(bundle: Path) -> tuple[dict, dict]:
    """校验发布包文件身份；不允许清单重定向到包外路径或缺失的文件。"""
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("artifact_kind") != "single-table-text-segmentation"
            or manifest.get("schema_version") != 1 or manifest.get("table") != "text_segments"
            or manifest.get("database_filename") != "text-segments.sqlite"):
        raise ValueError("片段单表发布契约不匹配")
    for name, expected in (
        ("text-segments.sqlite", manifest["database_sha256"]),
        ("post-evidence.jsonl", manifest["post_evidence_sha256"]),
        ("input-config.json", manifest["input_config_sha256"]),
        ("source-run-manifest.json", manifest["batch_id"]),
    ):
        if file_sha256(bundle / name) != expected:
            raise ValueError(f"片段发布包文件摘要不匹配：{name}")
    run = json.loads((bundle / "source-run-manifest.json").read_text(encoding="utf-8"))
    validate_run_manifest(run)
    for key in ("input", "rule_version", "implementation_version", "projection_id"):
        if manifest[key] != run[key]:
            raise ValueError(f"导入与原切分清单不一致：{key}")
    return manifest, run


def _verify_table_schema(connection: sqlite3.Connection) -> None:
    """只读核验七列、STRICT和索引，在内存副本验证实际写入约束。

    核验不会尝试写发布数据库；约束试写发生在只复制CREATE TABLE语句的
    内存库。禁止额外表、视图和触发器，避免不明对象改变表的写入语义。
    SQLite完整性检查只能证明文件结构完整，不能替代本契约检查。
    """
    objects = list(connection.execute("SELECT type,name,sql FROM sqlite_master WHERE type IN ('table','view','trigger')"))
    if len(objects) != 1 or objects[0][0:2] != ("table", "text_segments"):
        raise ValueError("切分数据库必须只有一张text_segments表")
    expected_columns = [
        (0, "segment_id", "TEXT", 1, None, 1, 0), (1, "source_post_id", "INTEGER", 1, None, 0, 0),
        (2, "seg_no", "INTEGER", 1, None, 0, 0), (3, "text", "TEXT", 1, None, 0, 0),
        (4, "start", "INTEGER", 1, None, 0, 0), (5, "end", "INTEGER", 1, None, 0, 0),
        (6, "batch_id", "TEXT", 1, None, 0, 0),
    ]
    if [tuple(row) for row in connection.execute("PRAGMA table_xinfo(text_segments)")] != expected_columns:
        raise ValueError("片段表字段、类型、主键或NOT NULL约束不匹配")
    table = [tuple(row) for row in connection.execute(
        "SELECT schema,name,type,ncol,wr,strict FROM pragma_table_list WHERE name='text_segments'")]
    if table != [("main", "text_segments", "table", 7, 0, 1)]:
        raise ValueError("片段表必须使用现有七列STRICT契约")
    unique = []
    for row in connection.execute("PRAGMA index_list(text_segments)"):
        _, name, is_unique, origin, partial = tuple(row)
        if is_unique:
            columns = tuple(item[2] for item in connection.execute("SELECT * FROM pragma_index_info(?)", (name,)))
            unique.append((origin, partial, columns))
    if sorted(unique) != [("pk", 0, ("segment_id",)), ("u", 0, ("batch_id", "source_post_id", "seg_no"))]:
        raise ValueError("片段表的主键或批次帖子序号唯一约束不匹配")
    if list(connection.execute("PRAGMA foreign_key_list(text_segments)")):
        raise ValueError("片段表不得引入跨库或额外外键")
    with closing(sqlite3.connect(":memory:")) as probe:
        probe.execute("PRAGMA trusted_schema=OFF")
        try:
            probe.execute(objects[0][2])
            baseline = ("a" * 64, 1, 1, "甲", 0, 1, "b" * 64)
            probe.execute("INSERT INTO text_segments VALUES (?,?,?,?,?,?,?)", baseline)
            invalid = [(index, None) for index in range(7)]
            invalid += [(0, "a" * 63), (0, "a" * 65), (1, 0), (1, -1), (2, 0), (2, -1),
                        (3, ""), (4, -1), (5, 0), (5, -1), (6, "b" * 63), (6, "b" * 65),
                        (1, "not-an-integer"), (2, 1.5), (4, 0.5), (5, 1.5), (3, b"raw-bytes")]
            for index, value in invalid:
                row = ["c" * 64, 2, 2, "乙", 0, 1, "d" * 64]
                row[index] = value
                try:
                    probe.execute("INSERT INTO text_segments VALUES (?,?,?,?,?,?,?)", row)
                except sqlite3.IntegrityError:
                    continue
                raise ValueError("片段表缺少必要CHECK、类型或NOT NULL约束")
            # 主键和联合唯一约束分别验证，避免同一行同时命中两项而掩盖缺失。
            for row in (("a" * 64, 2, 2, "乙", 0, 1, "d" * 64),
                        ("c" * 64, 1, 1, "乙", 0, 1, "b" * 64)):
                try:
                    probe.execute("INSERT INTO text_segments VALUES (?,?,?,?,?,?,?)", row)
                except sqlite3.IntegrityError:
                    continue
                raise ValueError("片段表唯一约束未生效")
        except sqlite3.DatabaseError:
            raise ValueError("片段表约束无法独立验证") from None


def verify_segment_bundle(bundle: Path, config: ResearchInput | None = None) -> dict:
    """逐帖读取新表并与冻结来源、正文投影、原切分证据对照，返回无原文摘要。

    检查唯一业务表、批次、计数、片段ID、顺序、文字、偏移、字素和来源哈希。
    任一失败抛异常；不改库或静默修复。config省略时使用包内已绑定的解析配置。
    """
    manifest, run = _load_bundle(bundle)
    config = config or load_research_input(bundle / "input-config.json")
    batch_id = manifest["batch_id"]
    total = post_count = 0
    empty_ids = []
    output_digest = hashlib.sha256()
    semantic_counts: Counter = Counter()
    semantic_decisions: Counter = Counter()
    with closing(open_segment_database(bundle / "text-segments.sqlite")) as connection:
        _verify_table_schema(connection)
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("片段数据库完整性失败")
        counts = dict(connection.execute("SELECT batch_id,count(*) FROM text_segments GROUP BY batch_id"))
        expected = {batch_id: manifest["segment_count"]} if manifest["segment_count"] else {}
        if counts != expected:
            raise ValueError("数据库批次或片段数不匹配")
        with open_research_corpus(config) as corpus, (bundle / "post-evidence.jsonl").open(encoding="utf-8") as evidence:
            if corpus.identity != manifest["input"]:
                raise ValueError("来源身份与片段库不一致")
            for post, line in zip_longest(corpus.posts, evidence):
                if post is None or line is None:
                    raise ValueError("帖子证据数量与来源不一致")
                record = json.loads(line)
                rows = get_post_segments(connection, batch_id, post.source_post_id)
                expected_count = record.pop("segment_count")
                if expected_count != len(rows):
                    raise ValueError("单帖片段数与原运行证据不一致")
                for row in rows:
                    if row["segment_id"] != segment_identifier(batch_id, post.source_post_id, row["seg_no"]):
                        raise ValueError("片段稳定ID不一致")
                record["original_body"] = post.body
                record["source_text"] = project_structured_text("" if post.body is None else post.body).text
                record["text_segments"] = [
                    {"seg_id": r["seg_no"], "raw_text": r["text"], "start": r["start"], "end": r["end"]}
                    for r in rows
                ]
                validate_post_record(record, post, run)
                summary = record["semantic_diagnostics"]
                semantic_counts.update({"post_count": 1, "completed_count": 1, "segment_count": len(rows),
                    "empty_count": int(not rows), "null_body_count": int(record["body_was_null"]),
                    "structured_body_count": int(record["format_id"] == "quill_delta_json"),
                    "posts_with_issues": int(bool(record["issues"])),
                    **{key: summary[key] for key in ("embedded_unit_count", "cache_hit_count", "model_call_count")}})
                semantic_decisions[summary["decision"]] += 1
                # 重建逐字JSONL摘要，证明压缩证据加新表足以还原被替代的大文件。
                output_digest.update((canonical_json(record) + "\n").encode())
                post_count += 1
                total += len(rows)
                if not rows:
                    empty_ids.append(post.source_post_id)
    if (total != manifest["segment_count"] or post_count != manifest["post_count"]
            or empty_ids != manifest["empty_post_ids"]
            or output_digest.hexdigest() != run["posts_sha256"]
            or manifest["source_records_sha256"] != run["posts_sha256"]):
        raise ValueError("全量入库结果无法精确还原原始切分运行")
    if (any(semantic_counts[key] != value for key, value in run["counts"].items())
            or dict(semantic_decisions) != run.get("decision_counts") or run.get("failure_counts") != {}):
        raise ValueError("语义运行汇总与逐帖证据计数不一致")
    _load_bundle(bundle)
    return {**manifest, "verification": "passed", "reconstructed_jsonl_sha256": output_digest.hexdigest()}


def _sampling_exclusions(paths: tuple[Path, ...], explicit: Path | None, *, source: ResearchInput,
                         identity: dict, population: set[int]) -> tuple[list[int], list[dict]]:
    """合并来源绑定的开发／压力／调参成员，拒绝跨来源或人口外的排除ID。

    显式调参清单使用 semantic-development-exclusions、input、post_ids；
    开发和比较清单自动排除整个 post_ids，不能只排除随机80帖而漏掉压力帖。
    """
    exclusions: set[int] = set()
    provenance = []
    for path in (*paths, *((explicit,) if explicit is not None else ())):
        value = json.loads(path.read_text(encoding="utf-8"))
        kind = value.get("artifact_kind")
        if kind == "semantic-development-sample":
            from .semantic_evaluation import load_development_members

            value = load_development_members(path, source)
        elif kind not in {"semantic-development-comparison", "semantic-development-exclusions"}:
            raise ValueError("独立抽查排除清单类型不匹配")
        if value.get("input") != identity:
            raise ValueError("独立抽查排除清单来源不一致")
        members = value.get("post_ids")
        if (not isinstance(members, list) or any(type(pid) is not int or pid < 1 for pid in members)
                or len(set(members)) != len(members) or not set(members) <= population):
            raise ValueError("独立抽查排除成员非法或不属于来源人口")
        if kind == "semantic-development-comparison":
            roles = value.get("development_post_ids", []) + value.get("targeted_post_ids", [])
            if set(roles) != set(members) or len(roles) != len(members):
                raise ValueError("比较清单开发与压力角色不完整")
        exclusions.update(members)
        provenance.append({"path": str(path.resolve()), "artifact_kind": kind,
                           "manifest_sha256": file_sha256(path), "post_count": len(members)})
    return sorted(exclusions), provenance


def sample_segment_bundle(bundle: Path, output: Path, *, count: int, seed: int,
                          exclude_manifests: tuple[Path, ...] = (), exclusion_manifest: Path | None = None,
                          independent: bool = False) -> dict:
    """全量复核后随机抽取count篇帖子，保存正文、分段及待填写审阅材料。

    普通抽查从全人口简单随机抽样。independent 要求显式来源绑定排除
    清单，排除后不足 count 直接失败；不补抽、预看或混入已参与调参的个案。
    只产生随机材料与自动验证状态，质量审阅始终 pending。
    """
    manifest = verify_segment_bundle(bundle)
    if type(count) is not int or count < 1 or count > manifest["post_count"] or type(seed) is not int:
        raise ValueError("抽查数量超出有效范围")
    if independent and not (exclude_manifests or exclusion_manifest is not None):
        raise ValueError("独立抽查必须显式提供全部开发压力与调参排除清单")
    config = load_research_input(bundle / "input-config.json")
    with open_research_corpus(config) as corpus:
        posts = {p.source_post_id: p for p in corpus.posts}
    excluded, exclusion_sources = _sampling_exclusions(tuple(exclude_manifests), exclusion_manifest,
        source=config, identity=manifest["input"], population=set(posts))
    candidates = sorted(set(posts) - set(excluded))
    if count > len(candidates):
        raise ValueError("排除开发压力与调参成员后抽查人口不足")
    selected = random.Random(seed).sample(candidates, count)
    diagnostics = {}
    for line in (bundle / "post-evidence.jsonl").open(encoding="utf-8"):
        item = json.loads(line)
        if item["source_post_id"] in selected:
            diagnostics[item["source_post_id"]] = item["issues"]
    with new_run_directory(output) as staging, closing(open_segment_database(bundle / "text-segments.sqlite")) as connection:
        samples = []
        for number, post_id in enumerate(selected, 1):
            post = posts[post_id]
            samples.append({
                "sample_no": number, "source_post_id": post_id, "source_version": post.source_version,
                "original_body": post.body,
                "source_text": project_structured_text("" if post.body is None else post.body).text,
                "segments": get_post_segments(connection, manifest["batch_id"], post_id),
                "issues": diagnostics[post_id],
            })
        write_json(staging / "samples.json", samples)
        summary = {
            "batch_id": manifest["batch_id"], "database_sha256": manifest["database_sha256"],
            "population_count": len(posts), "sample_count": count, "seed": seed,
            "sampling": "simple_random_without_replacement_by_post",
            "source_post_ids": selected, "samples_sha256": file_sha256(staging / "samples.json"),
            "automatic_full_verification": "passed", "review_status": "pending",
        }
        if exclude_manifests or exclusion_manifest is not None or independent:
            summary.update({"sample_role": "independent" if independent else "random_audit",
                "input": manifest["input"], "candidate_population_count": len(candidates),
                "candidate_post_ids": candidates,
                "candidate_population_sha256": hashlib.sha256(canonical_json(candidates).encode()).hexdigest(),
                "excluded_post_ids": excluded, "excluded_post_count": len(excluded),
                "exclusion_post_ids_sha256": hashlib.sha256(canonical_json(excluded).encode()).hexdigest(),
                "exclusion_sources": exclusion_sources,
                "exclusion_manifests_sha256": hashlib.sha256(canonical_json(exclusion_sources).encode()).hexdigest(),
                "source_bundle_manifest_sha256": file_sha256(bundle / "manifest.json"),
                "config_id": _load_bundle(bundle)[1].get("config_sha256", manifest["batch_id"]),
            })
        write_json(staging / "selection.json", summary)
        reviews = [{"sample_no": i, "source_post_id": post_id, "result": "pending", "note": ""}
                   for i, post_id in enumerate(selected, 1)]
        if independent:
            for row in reviews:
                row.update({"post_id": row["source_post_id"], "sample_role": "independent",
                    "config_id": summary["config_id"], "technical_pass": True,
                    "assistant_judgment": "pending", "issue_codes": [], "affected_spans": [],
                    "explanation": "", "suggested_change": "", "reviewer_kind": "assistant", "review_time": None})
        write_json(staging / "review.json", {"review_status": "pending", "samples": reviews})
    return summary
