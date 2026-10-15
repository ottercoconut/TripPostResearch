"""组合正文投影、候选单元、向量和纯边界计算的显式语义接口。

调用者负责初始化provider与缓存；本接口不打开数据库、不下载权重。
只读缓存时向量全部来自缓存，provider只提供身份与token计数。
逐帖投影／单元／边界失败带明确原因且无部分片段；模型或缓存异常向上抛出，
使批处理停止系统性故障。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

from .contracts import SegmentationIssue, SegmentationResult, SemanticDecision, TextUnit
from .embedding import EmbeddingError, EmbeddingProvider, TokenCounter, identity_sha256, validate_vectors
from .embedding_cache import EmbeddingCache
from .integrity import validate
from .projection import PROJECTION_ID, project_structured_text
from .semantic_config import CoherentSemanticConfig, semantic_rule_identity
from .semantic_dispatch import MeaningTrace, choose_semantic_boundaries
from .semantic_units import build_units


@dataclass(frozen=True)
class SemanticOutput:
    """切分结果加语义trace，只有summary随逐帖证据进入发布包。

    units及decision用于本地检查，不是新增数据库列。实际调用数按模型forward
    计数，embedded_unit_count按参与评分单元计数，缓存命中仍属于语义参与。
    summary只使用最终decision，不混入初始临时决策。
    meaning_trace仅当前连贯块规则成功时提供，保存原始语义／结构来源；
    投影或领域失败时为None。
    """

    result: SegmentationResult
    units: tuple[TextUnit, ...]
    decision: SemanticDecision
    model_identity_sha256: str
    vector_content_sha256: str | None
    embedded_unit_count: int
    cache_hit_count: int
    model_call_count: int
    meaning_trace: MeaningTrace | None = None

    def summary(self) -> dict:
        """不含原文的紧凑诊断；当前规则的semantic计数包括结构来源的非长度点。

        两类来源重叠只计一次，纯语义／结构来源另见meaning_trace。投影reason_code继续只说明正文投影，不评价切分质量。
        """
        return {
            "decision": self.decision.status, "reason_codes": list(self.decision.reason_codes),
            "unit_count": len(self.units), "candidate_gap_count": max(0, len(self.units) - 1),
            "embedded_unit_count": self.embedded_unit_count, "cache_hit_count": self.cache_hit_count,
            "model_call_count": self.model_call_count,
            "semantic_boundary_count": len(self.decision.semantic_boundaries),
            "length_boundary_count": len(self.decision.length_boundaries),
            "selected_boundary_count": len(self.decision.semantic_boundaries) + len(self.decision.length_boundaries),
            "model_identity_sha256": self.model_identity_sha256,
            "vector_content_sha256": self.vector_content_sha256,
        }


def segment_body_semantic(
    body: str | None, *, provider: EmbeddingProvider, token_counter: TokenCounter,
    config: CoherentSemanticConfig, cache: EmbeddingCache | None = None,
) -> SemanticOutput:
    """处理单帖原始正文，位置始终相对既有可读投影S。

    空白和NULL成功返回空片段，损坏Delta失败。N≥2必须有有效向量，N≤1不作
    无意义模型调用；批次级模型预检由调度层完成。所有异常诊断不包含原文。
    """
    if body is not None and not isinstance(body, str):
        raise TypeError("body必须是字符串或None")
    rule_version, implementation_version = semantic_rule_identity(config)
    projection = project_structured_text("" if body is None else body)
    source = projection.text if projection.status == "usable" else None
    reason = "source_body_null" if body is None else projection.reason_code
    model_hash = identity_sha256(provider.identity)
    units = ()
    segments = None
    issues = ()
    vector_hash = None
    embedded_count = 0
    meaning_trace = None
    decision = SemanticDecision("failed", (reason,), (), (), ())
    hits_before = cache.cache_hit_count if cache else 0
    calls_before = getattr(provider, "model_call_count", 0)
    if source is not None:
        try:
            built = build_units(source, token_counter, config)
            units = built.units
        except EmbeddingError:
            raise
        except ValueError as exc:
            decision = SemanticDecision("failed", (str(exc),), (), (), ())
        else:
            vectors = None
            if len(units) >= 2:
                texts = [source[u.start:u.end] for u in units]
                # 模型和cache错误必须向上终止attempt，不能混作可继续的单帖错误。
                # 只读缓存只取已有向量，缺条目即失败，绝不回落到provider推理。
                if cache is None:
                    vectors = provider.encode(texts)
                elif cache.readonly:
                    vectors = cache.read_cached(texts)
                else:
                    vectors = cache.encode(texts, provider)
                vectors = validate_vectors(vectors, count=len(units), dimension=provider.identity["dimension"])
                vector_hash = hashlib.sha256(vectors.astype("<f4").tobytes(order="C")).hexdigest()
                embedded_count = len(units)
            try:
                selected = choose_semantic_boundaries(source, units, vectors, config, token_counter)
                segments, decision = selected.segments, selected.decision
                meaning_trace = selected.meaning_trace
                validate(source, [asdict(s) for s in segments])
            except EmbeddingError:
                raise
            except ValueError as exc:
                segments = None
                meaning_trace = None
                decision = SemanticDecision("failed", (str(exc),), (), (), ())
            else:
                codes = tuple(dict.fromkeys((*built.diagnostics, *decision.reason_codes)))
                decision = SemanticDecision(decision.status, codes, decision.gaps,
                                            decision.semantic_boundaries, decision.length_boundaries)
                if not units:
                    issues = (SegmentationIssue("EMPTY_BODY", 0, len(source)),)
    result = SegmentationResult(
        rule_version=rule_version, implementation_version=implementation_version,
        projection_id=PROJECTION_ID, status="completed" if segments is not None else "failed",
        reason_code=reason, body_was_null=body is None,
        source_body_sha256=None if body is None else hashlib.sha256(body.encode()).hexdigest(),
        source_text=source, source_text_sha256=None if source is None else hashlib.sha256(source.encode()).hexdigest(),
        format_id=projection.format_id, text_insert_count=projection.text_insert_count,
        ignored_embed_count=projection.ignored_embed_count, text_segments=segments, issues=issues,
    )
    return SemanticOutput(result, units, decision, model_hash, vector_hash, embedded_count,
                          (cache.cache_hit_count - hits_before) if cache else 0,
                          getattr(provider, "model_call_count", 0) - calls_before, meaning_trace)
