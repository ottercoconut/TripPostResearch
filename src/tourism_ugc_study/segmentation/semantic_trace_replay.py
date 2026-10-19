"""从已完成正式运行的 trace 重放当前分区规则，供开发对比；不读向量、源库或模型。

1.3只改选点、不改候选单元（适配方案§5.2.3），而正式运行的`semantic-trace.jsonl`
已保存每帖候选单元跨度、token数与初始 gap 评分（cohesion、depth及语义提名），
足以用当前规则重算分区。本模块只服务开发阶段的全量对比：结果标记为
development，不能导入单表或冒充正式运行；正式运行仍须经只读向量缓存重算评分
（§7），两者应逐帖一致。

输入为旧运行的`posts.jsonl`与`semantic-trace.jsonl`（同序、同帖），输出只含
原文坐标、切点理由和汇总计数，不复制正文。任何帖子身份、正文摘要或单元覆盖
不一致都抛 ValueError，不跳过。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator, Sequence

from .contracts import GapScore, SemanticDecision, TextUnit, TokenCounter
from .integrity import WHITE
from .semantic_config import CoherentSemanticConfig
from .semantic_meaning import MeaningPartition, partition_by_meaning

# 新增切点按以下优先顺序归类，取首个命中的理由；撤销切点同理。撤销类中的
# not_nominated 表示该处在新规则下没有任何提名（多为旧运行的长度补切或标签上移
# 切点，随其他切点变化后不再需要）。
_ADDED_CLASSES = ("selected_section_heading", "selected_day_transition", "selected_before_label_heading",
                  "selected_length_fallback", "selected_local_semantic")
_REMOVED_CLASSES = ("merged_heading_only_segment", "merged_section_closure", "merged_short_first_segment",
                    "merged_short_last_segment", "merged_wordless_segment", "rejected_inside_coherent_block",
                    "rejected_open_contrast_or_reason", "rejected_structure_below_valley",
                    "rejected_section_dependency", "rejected_semantic_dependency",
                    "rejected_structure_dependency", "shift_before_label_heading_rejected",
                    "not_nominated")


@dataclass(frozen=True)
class ReplayCase:
    """一帖的重放输入：原文、候选单元、初始评分及旧运行的片段起点。"""

    post_id: int
    text: str
    units: tuple[TextUnit, ...]
    decision: SemanticDecision
    old_starts: tuple[int, ...]


def _text_sha(text: str) -> str:
    """与运行记录相同口径的正文摘要（UTF-8字节的SHA-256）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def case_from_records(post: dict, trace: dict) -> ReplayCase:
    """把同一帖的 posts.jsonl 与 trace 记录还原成重放输入；不一致即失败。

    初始评分取 trace 的`meaning.initial_decision`，即分区之前的语义评分结果，
    与正式运行中 choose_boundaries 的返回值相同；不使用旧分区的最终选择。
    """
    if post.get("source_post_id") != trace.get("source_post_id"):
        raise ValueError("trace_replay_post_mismatch")
    text = post.get("source_text")
    if not isinstance(text, str) or _text_sha(text) != trace.get("source_text_sha256"):
        raise ValueError("trace_replay_text_mismatch")
    units = tuple(TextUnit(unit["start"], unit["end"], unit["token_count"]) for unit in trace["units"])
    initial = trace["meaning"]["initial_decision"]
    gaps = tuple(GapScore(gap["index"], gap["cohesion"], gap["depth"], gap["candidate"], gap["selected"], gap["reason"])
                 for gap in initial["gaps"])
    decision = SemanticDecision(initial["status"], tuple(initial["reason_codes"]), gaps,
                                tuple(initial["semantic_boundaries"]), tuple(initial["length_boundaries"]))
    segments = post.get("text_segments") or []
    return ReplayCase(post["source_post_id"], text, units, decision,
                      tuple(segment["start"] for segment in segments[1:]))


def iter_cases(run: Path) -> Iterator[ReplayCase]:
    """按原顺序逐帖读取旧运行目录，不一次载入全部 trace。"""
    with (run / "posts.jsonl").open(encoding="utf-8") as posts, \
            (run / "semantic-trace.jsonl").open(encoding="utf-8") as traces:
        for post_line, trace_line in zip(posts, traces, strict=True):
            yield case_from_records(json.loads(post_line), json.loads(trace_line))


def _classify(reasons: Sequence[str], classes: Sequence[str]) -> str:
    """取理由列表中优先级最高的已知类别；都不命中时记 other。"""
    present = set(reasons)
    return next((name for name in classes if name in present), "other")


def replay_case(case: ReplayCase, config: CoherentSemanticConfig, counter: TokenCounter,
                ) -> tuple[MeaningPartition, dict]:
    """用当前规则重算一帖，返回分区结果和只含坐标、理由的对比记录。

    切点以片段起点（原文码点）比较：旧有新无为撤销，旧无新有为新增。理由取
    新分区对应 gap 的 reason_codes，用于说明为何新增或为何不再选中。
    """
    partition = partition_by_meaning(case.text, case.units, case.decision, config, counter)
    gap_of = {unit.start: index - 1 for index, unit in enumerate(case.units) if index}
    new_starts = tuple(segment.start for segment in partition.segments[1:])
    old, new = set(case.old_starts), set(new_starts)

    def reasons_at(start: int) -> tuple[str, ...]:
        """片段起点对应 gap 的全部理由；起点不在单元起点上时说明输入已损坏。"""
        if start not in gap_of:
            raise ValueError("trace_replay_boundary_not_on_gap")
        return partition.gap_decisions[gap_of[start]].reason_codes

    added = [{"start": start, "class": _classify(reasons_at(start), _ADDED_CLASSES)} for start in sorted(new - old)]
    removed = [{"start": start, "class": _classify(reasons_at(start), _REMOVED_CLASSES)} for start in sorted(old - new)]
    record = {"source_post_id": case.post_id, "segment_count": len(partition.segments),
              "old_segment_count": len(case.old_starts) + 1 if case.units else 0,
              "segments": [[segment.start, segment.end] for segment in partition.segments],
              "added": added, "removed": removed, "length_exceptions": list(partition.length_exceptions)}
    return partition, record


def summarize(records: Sequence[dict], texts: dict[int, str] | None = None, short_limit: int = 16) -> dict:
    """汇总全量对比：段数、变化帖数、按理由分类的新增／撤销切点及短段数。

    texts 可选，提供时按非空白字符统计不足 short_limit 的片段数；不保存正文。
    """
    added, removed = Counter(), Counter()
    for record in records:
        added.update(item["class"] for item in record["added"])
        removed.update(item["class"] for item in record["removed"])
    summary = {"post_count": len(records),
               "old_segment_count": sum(record["old_segment_count"] for record in records),
               "segment_count": sum(record["segment_count"] for record in records),
               "changed_post_count": sum(bool(record["added"] or record["removed"]) for record in records),
               "added_boundary_count": sum(added.values()), "removed_boundary_count": sum(removed.values()),
               "added_by_class": dict(sorted(added.items())), "removed_by_class": dict(sorted(removed.items())),
               "length_exception_post_count": sum(bool(record["length_exceptions"]) for record in records)}
    if texts is not None:
        summary[f"short_segment_count_lt_{short_limit}"] = sum(
            sum(char not in WHITE for char in texts[record["source_post_id"]][start:end]) < short_limit
            for record in records for start, end in record["segments"])
    return summary


_WORKER: dict = {}


def _init_worker(config: CoherentSemanticConfig, counter_kwargs: dict) -> None:
    """子进程各自加载同 revision tokenizer；只读本地文件，不访问网络。"""
    from .embedding import QwenTokenCounter

    _WORKER["config"] = config
    _WORKER["counter"] = QwenTokenCounter(**counter_kwargs)


def _replay_one(case: ReplayCase) -> dict:
    """子进程内重放一帖，只回传坐标与理由记录。"""
    return replay_case(case, _WORKER["config"], _WORKER["counter"])[1]


def replay_run(run: Path, output: Path, *, config: CoherentSemanticConfig, counter_kwargs: dict,
               code_version: str, config_sha256: str, workers: int = 1) -> dict:
    """对旧运行全量重放当前规则，写出逐帖对比、汇总与开发清单，返回汇总。

    output 必须不存在；先写入同级临时目录，全部成功后整体改名，失败不留半包。
    清单记录输入运行清单摘要、当前配置摘要、代码版本和 tokenizer 参数，并明确
    `scope=development`、`vector_source=trace_scores`，不能当作正式运行导入。
    """
    from multiprocessing import get_context

    if output.exists():
        raise FileExistsError("trace_replay_output_exists")
    manifest_path = run / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    temporary = output.with_name(output.name + ".partial")
    temporary.mkdir(parents=True, exist_ok=False)
    records: list[dict] = []
    texts: dict[int, str] = {}

    def cases() -> Iterator[ReplayCase]:
        """顺序读取输入，同时留下正文供短段统计（只在内存中，不写盘）。"""
        for case in iter_cases(run):
            texts[case.post_id] = case.text
            yield case

    with (temporary / "posts.jsonl").open("w", encoding="utf-8") as handle:
        if workers <= 1:
            _init_worker(config, counter_kwargs)
            results = map(_replay_one, cases())
            for record in results:
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        else:
            with get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(config, counter_kwargs)) as pool:
                for record in pool.imap(_replay_one, cases(), chunksize=16):
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = summarize(records, texts)
    (temporary / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    replay_manifest = {"artifact_kind": "semantic-trace-replay", "scope": "development",
                       "vector_source": "trace_scores", "source_run_manifest": str(manifest_path),
                       "source_run_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                       "source_rule_version": source_manifest.get("rule_version"),
                       "source_implementation_version": source_manifest.get("implementation_version"),
                       "config_sha256": config_sha256, "rule_version": config.rule_version,
                       "code_version": code_version, "tokenizer": {key: value for key, value in counter_kwargs.items()
                                                                   if key != "cache_dir"},
                       "posts_sha256": hashlib.sha256((temporary / "posts.jsonl").read_bytes()).hexdigest()}
    (temporary / "manifest.json").write_text(json.dumps(replay_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.rename(output)
    return summary
