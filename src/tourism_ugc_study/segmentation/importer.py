"""将已验收切分结果原样导入唯一text_segments表，保持来源库只读。

新库及清单成组发布；相同批次重复导入只校验已有产物，不生成重复行。
正文与片段转入数据库后，辅助证据仅保存哈希和诊断，避免再复制大份原文。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

from .artifacts import canonical_json, new_run_directory, write_json
from .config import ResearchInput
from .database_audit import verify_segment_bundle
from .import_validation import load_run_manifest, require_records_hash, validate_post_record
from .source import file_sha256, open_research_corpus
from .storage import create_segment_database, insert_segments


def import_segmentation_run(
    run: Path, output: Path, config: ResearchInput, *, code_version: str,
) -> dict:
    """校验完整JSONL与冻结原文后，事务性写入单表并返回产物清单。

    run为已有成功切分运行，output为新发布目录。已有目录仅允许同一批次且
    数据库/证据摘要仍有效；其他情况拒绝覆盖。输入失败回滚并清理临时目录，
    不写源库，不自动更改管理端配置。空正文成员保存在清单，不伪造片段行。
    """
    manifest, batch_id = load_run_manifest(run)
    if output.exists():
        existing = verify_segment_bundle(output, config)
        if existing["batch_id"] != batch_id:
            raise FileExistsError("目标目录属于另一切分批次")
        return {**existing, "import_status": "already_imported"}
    require_records_hash(run, manifest)
    ordered_input = hashlib.sha256()
    post_count = segment_count = 0
    empty_posts = []
    with new_run_directory(output) as staging:
        database = staging / "text-segments.sqlite"
        evidence_path = staging / "post-evidence.jsonl"
        connection = create_segment_database(database)
        try:
            with open_research_corpus(config) as corpus:
                if corpus.identity != manifest["input"]:
                    raise ValueError("导入目标来源与切分运行来源不一致")
                with (run / "posts.jsonl").open(encoding="utf-8") as inputs, evidence_path.open("x", encoding="utf-8") as evidence:
                    evidence_path.chmod(0o600)
                    for source, line in zip_longest(corpus.posts, inputs):
                        if source is None or line is None:
                            raise ValueError("切分JSONL成员数与来源不一致")
                        record = json.loads(line)
                        validate_post_record(record, source, manifest)
                        insert_segments(connection, batch_id, source.source_post_id, record["text_segments"])
                        ordered_input.update((canonical_json([
                            source.source_post_id, source.source_version, record["source_body_sha256"],
                        ]) + "\n").encode())
                        # 只保存来源身份、正文摘要与诊断；原body仍在冻结源快照，片段已入表。
                        compact = {k: v for k, v in record.items() if k not in
                                   {"original_body", "source_text", "text_segments"}}
                        compact["segment_count"] = len(record["text_segments"])
                        evidence.write(canonical_json(compact) + "\n")
                        if not record["text_segments"]:
                            empty_posts.append(source.source_post_id)
                        post_count += 1
                        segment_count += len(record["text_segments"])
                if (post_count != manifest["counts"]["post_count"]
                        or segment_count != manifest["counts"]["segment_count"]
                        or ordered_input.hexdigest() != manifest["ordered_input_sha256"]):
                    raise ValueError("导入数量或有序成员摘要不一致")
            require_records_hash(run, manifest)
            if file_sha256(run / "manifest.json") != batch_id:
                raise ValueError("切分清单在导入过程中变化")
            connection.commit()
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("片段SQLite完整性检查失败")
        finally:
            connection.close()
        # 原清单按精确字节保留，后续可在删除重复JSONL后仍确定批次身份。
        (staging / "source-run-manifest.json").write_bytes((run / "manifest.json").read_bytes())
        (staging / "source-run-manifest.json").chmod(0o600)
        parsed_config = {k: str(v) for k, v in asdict(config).items()}
        write_json(staging / "input-config.json", parsed_config)
        result = {
            "artifact_kind": "single-table-text-segmentation", "schema_version": 1,
            "status": "IMPORTED_VERIFIED", "batch_id": batch_id,
            "table": "text_segments", "database_filename": database.name,
            "database_sha256": file_sha256(database), "post_count": post_count,
            "segment_count": segment_count, "empty_post_ids": empty_posts,
            "input": manifest["input"], "rule_version": manifest["rule_version"],
            "implementation_version": manifest["implementation_version"],
            "projection_id": manifest["projection_id"],
            "source_records_sha256": manifest["posts_sha256"],
            "post_evidence_sha256": file_sha256(evidence_path),
            "input_config_sha256": file_sha256(staging / "input-config.json"),
            "import_code_version": code_version,
            "import_module_sha256": {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "imported_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_database_write_count": 0, "membership_database_write_count": 0,
        }
        write_json(staging / "manifest.json", result)
        verify_segment_bundle(staging, config)
    return {**result, "import_status": "imported"}
