"""语义切分的固定经验参数和严格配置解析，不加载模型或打开数据库。

默认值是首轮工程起点，不代表科研最优值。配置含算法与embedding两部分，
所有字段进入谱系；路径由CLI显式提供，不在配置中存凭据或源正文。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re

RULE_VERSION = "semantic-depth-ugc-0.1"
IMPLEMENTATION_VERSION = "semantic-depth-ugc-0.1-original-1"
STRUCTURED_RULE_VERSION = "semantic-depth-ugc-0.2"
STRUCTURED_IMPLEMENTATION_VERSION = "semantic-depth-ugc-0.2-structure-global-1"
COHERENT_RULE_VERSION = "semantic-coherent-ugc-1.0"
COHERENT_IMPLEMENTATION_VERSION = "semantic-coherent-ugc-1.0-meaning-v4-1"
COHERENT_PARTITION_ID = "experimental-meaning-units-4"
CURRENT_FROZEN_SETTINGS_SHA256 = "cef1f8dd506b3ddd3a1a8b31171f7e4737da8d671a2afa405bb192fefb7d22da"
RUN_KIND = "semantic-text-segmentation-run"


def validate_semantic_rule_identity(manifest: dict) -> None:
    """只认可当前已登记的语义规则与实现配对，不把合法文件摘要当算法身份。

    本门不检查开发／全量范围，也不打开文件或加载模型；开发运行核验及跨模型
    汇总均可复用。缺失、未知或错配的身份抛 ValueError，不能静默解释为当前规则。
    """
    registered = ((RULE_VERSION, IMPLEMENTATION_VERSION),
                  (STRUCTURED_RULE_VERSION, STRUCTURED_IMPLEMENTATION_VERSION),
                  (COHERENT_RULE_VERSION, COHERENT_IMPLEMENTATION_VERSION))
    if not isinstance(manifest, dict):
        raise ValueError("semantic_rule_identity_mismatch")
    identity = (manifest.get("rule_version"), manifest.get("implementation_version"))
    if identity not in registered:
        raise ValueError("semantic_rule_identity_mismatch")
    if "config" in manifest:
        try:
            configured = semantic_rule_identity(parse_semantic_settings(manifest["config"]).algorithm)
        except (TypeError, ValueError):
            raise ValueError("semantic_rule_identity_mismatch") from None
        if configured != identity:
            raise ValueError("semantic_rule_identity_mismatch")


@dataclass(frozen=True)
class SemanticConfig:
    """边界计算与候选单元参数；任何非法值在模型运行之前拒绝。

    token限制约束正文，模型输入含模板上限由EmbeddingConfig约束。
    epsilon用于峰值／平台容差，不能作为跨硬件效果验收阈值。
    """

    unit_max_tokens: int = 64
    window_tokens: int = 64
    absolute_cohesion_threshold: float = 0.55
    depth_floor: float = 0.15
    depth_std_factor: float = 0.5
    min_segment_nonspace_chars: int = 12
    max_segment_tokens: int = 384
    epsilon: float = 1e-6
    smoothing: int = 0

    def __post_init__(self) -> None:
        """拒绝bool伪装整数、非有限阈值和无意义的长度组合。"""
        for name in ("unit_max_tokens", "window_tokens", "min_segment_nonspace_chars", "max_segment_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"invalid_algorithm_parameter:{name}")
        for name in ("absolute_cohesion_threshold", "depth_floor", "depth_std_factor", "epsilon"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"invalid_algorithm_parameter:{name}")
        if not -1 <= self.absolute_cohesion_threshold <= 1:
            raise ValueError("invalid_cohesion_threshold")
        if not 0 < self.depth_floor <= 4 or self.depth_std_factor < 0 or not 0 < self.epsilon < 0.01:
            raise ValueError("invalid_depth_parameters")
        if self.unit_max_tokens > self.max_segment_tokens:
            raise ValueError("unit_limit_exceeds_segment_limit")
        if type(self.smoothing) is not int or self.smoothing != 0:
            raise ValueError("smoothing_not_supported")


@dataclass(frozen=True)
class StructuredSemanticConfig(SemanticConfig):
    """显式启用0.2开发规则；不向旧配置增加字段或改变历史指纹。

    继承原有评分和长度参数，结构证据与分区策略另行进入配置摘要。
    context/global仅在调用者选择本类或声明0.2时成为默认值，不能据模型
    大小或缺失字段把历史0.1配置升级。登记算法身份不等于科研质量已通过。
    """

    rule_version: str = STRUCTURED_RULE_VERSION
    structure_mode: str = "context"
    length_strategy: str = "global"

    def __post_init__(self) -> None:
        """复用原参数约束，另拒绝未知身份与无法重放的结构／长度策略。"""
        super().__post_init__()
        if self.rule_version != STRUCTURED_RULE_VERSION:
            raise ValueError("unknown_semantic_rule_version")
        if self.structure_mode not in ("sentence_only", "structure", "blocks", "context"):
            raise ValueError("unknown_structure_mode")
        if self.length_strategy not in ("global", "greedy"):
            raise ValueError("unknown_partition_length_strategy")


@dataclass(frozen=True)
class CoherentSemanticConfig(SemanticConfig):
    """冻结的完整介绍／推荐块规则；默认参数等于用户采纳的v4运行。

    只新增显式算法身份，不改旧配置字段或指纹。该身份表示采用同一确定性
    规则，不代表每篇正文质量均经人工确认；参数变化仍进入配置摘要。
    """

    window_tokens: int = 32
    depth_floor: float = 0.05
    min_segment_nonspace_chars: int = 16
    max_segment_tokens: int = 512
    rule_version: str = COHERENT_RULE_VERSION

    def __post_init__(self) -> None:
        """拒绝非法参数及借用其他规则标签，不接受实验策略开关。"""
        super().__post_init__()
        if self.rule_version != COHERENT_RULE_VERSION:
            raise ValueError("unknown_semantic_rule_version")


def parse_algorithm_config(raw: object, require_complete: bool = False) -> SemanticConfig:
    """按显式规则恢复算法配置；旧无版本字段格式始终恢复为0.1。

    普通配置允许省略已声明版本的经验默认值；运行manifest使用
    require_complete=True要求完整保存该版本全部字段。未知字段、未知版本
    或把新策略塞入旧配置均抛ValueError，不静默忽略或推断升级。
    """
    if not isinstance(raw, dict):
        raise ValueError("semantic_config_fields_invalid:algorithm")
    cls = SemanticConfig
    if "rule_version" in raw:
        classes = {STRUCTURED_RULE_VERSION: StructuredSemanticConfig,
                   COHERENT_RULE_VERSION: CoherentSemanticConfig}
        if not isinstance(raw["rule_version"], str) or raw["rule_version"] not in classes:
            raise ValueError("unknown_semantic_rule_version")
        cls = classes[raw["rule_version"]]
    fields = set(cls.__dataclass_fields__)
    if set(raw) - fields or (require_complete and set(raw) != fields):
        raise ValueError("semantic_config_fields_invalid:algorithm")
    return cls(**raw)


def semantic_rule_identity(config: SemanticConfig) -> tuple[str, str]:
    """返回已登记配置类的规则／实现配对，同时防御性复验配置。

    仅接受登记的确切类型，避免未登记子类借旧身份携带额外行为。
    实现身份属于算法；模型身份仍由EmbeddingConfig及provider独立绑定。
    """
    if type(config) is SemanticConfig:
        SemanticConfig.__post_init__(config)
        return RULE_VERSION, IMPLEMENTATION_VERSION
    if type(config) is StructuredSemanticConfig:
        StructuredSemanticConfig.__post_init__(config)
        return STRUCTURED_RULE_VERSION, STRUCTURED_IMPLEMENTATION_VERSION
    if type(config) is CoherentSemanticConfig:
        CoherentSemanticConfig.__post_init__(config)
        return COHERENT_RULE_VERSION, COHERENT_IMPLEMENTATION_VERSION
    raise ValueError("invalid_semantic_config_type")


@dataclass(frozen=True)
class EmbeddingConfig:
    """显式本地Qwen运行身份；同配置不得静默更换设备、模型或精度。"""

    model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    revision: str = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    device: str = "mps"
    dtype: str = "float32"
    batch_size: int = 1
    max_input_tokens: int = 256
    instruction: str = "Retrieve semantically similar text"

    def __post_init__(self) -> None:
        """型号仅限定已计划的Qwen系列，revision必须完整固定提交。"""
        if self.model_id not in {f"Qwen/Qwen3-Embedding-{s}" for s in ("0.6B", "4B", "8B")}:
            raise ValueError("unsupported_embedding_model")
        if not isinstance(self.revision, str) or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("embedding_revision_must_be_commit")
        if self.device not in {"cpu", "mps", "cuda"} or self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported_embedding_runtime")
        if any(type(v) is not int or v < 1 for v in (self.batch_size, self.max_input_tokens)):
            raise ValueError("invalid_embedding_limit")
        if self.max_input_tokens > 32768 or not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("invalid_embedding_input")


@dataclass(frozen=True)
class SemanticSettings:
    """同一运行的算法与模型声明，实际权重及设备身份另由provider补齐。"""

    algorithm: SemanticConfig
    embedding: EmbeddingConfig

    @property
    def fingerprint(self) -> str:
        """返回规范化配置摘要，不是完成运行后的batch_id。"""
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def validate_current_frozen_settings(settings: SemanticSettings) -> None:
    """新生成入口只接受用户采纳的完整算法、模型与运行参数组合。

    登记摘要独立于外部JSON，修改配置文件不能同时改写核验基准。纯计算、
    历史读取和实验重放仍可显式参数化，不把该门加入一般配置解析器。
    身份、参数或模型漂移均抛ValueError，调用者须在读取正文前执行。
    """
    if (type(settings) is not SemanticSettings
            or type(settings.algorithm) is not CoherentSemanticConfig
            or settings.fingerprint != CURRENT_FROZEN_SETTINGS_SHA256):
        raise ValueError("current_frozen_settings_mismatch")


def load_semantic_settings(path: Path) -> SemanticSettings:
    """解析JSON，两层均拒绝未知字段；不允许缺少算法或模型声明。"""
    return parse_semantic_settings(json.loads(path.read_text(encoding="utf-8")))


def parse_semantic_settings(raw: object) -> SemanticSettings:
    """从清单内JSON对象恢复同一配置契约；不接受未知字段或隐式类型转换。"""
    if not isinstance(raw, dict) or set(raw) != {"algorithm", "embedding"}:
        raise ValueError("semantic_config_fields_invalid")
    algorithm = parse_algorithm_config(raw["algorithm"])
    embedding = raw["embedding"]
    if not isinstance(embedding, dict) or set(embedding) - set(EmbeddingConfig.__dataclass_fields__):
        raise ValueError("semantic_config_fields_invalid:embedding")
    return SemanticSettings(algorithm, EmbeddingConfig(**embedding))


def validate_identity_settings(identity: dict, settings: EmbeddingConfig) -> None:
    """比较实际provider身份与声明，防止测试注入或配置漂移伪造科研谱系。

    完整资产摘要另由provider复验；这里不打开模型文件，可用于运行及发布包
    的独立验收。测试provider也必须提供同一显式身份，不能例外发布。
    CUDA必须显式证明固定padding宽度与声明预算一致；旧CUDA身份不静默升级，
    CPU/MPS仍接受既有不含这两个字段的身份。
    """
    expected = {
        "provider": "qwen-transformers-local-v1",
        "model_id": settings.model_id, "model_revision": settings.revision,
        "tokenizer_revision": settings.revision, "device": settings.device,
        "dtype": settings.dtype, "batch_size": settings.batch_size,
        "max_input_tokens": settings.max_input_tokens, "instruction": settings.instruction,
        "template": "Instruct: {instruction}\nQuery:{text}",
        "dimension": {"0.6B": 1024, "4B": 2560, "8B": 4096}[settings.model_id.rsplit("-", 1)[-1]],
        "special_tokens": "tokenizer_native_no_manual_eos", "padding_side": "left",
        "truncation": False, "pooling": "last_valid_token", "normalization": "l2_float32",
        "quantization": None, "attention_backend": "eager", "use_cache": False,
    }
    if settings.device == "cuda":
        expected.update(padding_strategy="max_length", padding_max_length=settings.max_input_tokens)
    if not isinstance(identity, dict) or any(
        key not in identity or type(identity[key]) is not type(value) or identity[key] != value
        for key, value in expected.items()
    ):
        raise ValueError("embedding_identity_settings_mismatch")
