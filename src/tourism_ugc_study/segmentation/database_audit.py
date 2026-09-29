"""片段单表发布包的全量复核与随机抽查材料生成。

核验只读数据库与原始冻结来源逐字对应，不调用边界算法产生新结果。抽样是
固定种子的帖子级简单随机抽样；程序生成材料，审阅结论必须另行逐篇填写。
"""

from __future__ import annotations

import hashlib
import json
import random
from contextlib import closing
from itertools import zip_longest
from pathlib import Path

from .artifacts import canonical_json, new_run_directory, write_json
from .config import ResearchInput, load_research_input
from .import_validation import validate_post_record
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
    for key in ("input", "rule_version", "implementation_version", "projection_id"):
        if manifest[key] != run[key]:
            raise ValueError(f"导入与原切分清单不一致：{key}")
    return manifest, run


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
    with closing(open_segment_database(bundle / "text-segments.sqlite")) as connection:
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        if tables != ["text_segments"]:
            raise ValueError("切分数据库必须只有一张text_segments表")
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
    _load_bundle(bundle)
    return {**manifest, "verification": "passed", "reconstructed_jsonl_sha256": output_digest.hexdigest()}


def sample_segment_bundle(bundle: Path, output: Path, *, count: int, seed: int) -> dict:
    """全量复核后随机抽取count篇帖子，保存正文、分段及待填写审阅材料。

    抽样不按平台、长度或是否带提示筛选；数量超出人口直接失败，不补抽替换。
    只产生随机材料与自动验证状态，不将程序检查冒充人工质量审阅。
    """
    manifest = verify_segment_bundle(bundle)
    if type(count) is not int or count < 1 or count > manifest["post_count"]:
        raise ValueError("抽查数量超出有效范围")
    config = load_research_input(bundle / "input-config.json")
    with open_research_corpus(config) as corpus:
        posts = {p.source_post_id: p for p in corpus.posts}
    selected = random.Random(seed).sample(sorted(posts), count)
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
        write_json(staging / "selection.json", summary)
        write_json(staging / "review.json", {"review_status": "pending", "samples": [
            {"sample_no": i, "source_post_id": post_id, "result": "pending", "note": ""}
            for i, post_id in enumerate(selected, 1)
        ]})
    return summary
