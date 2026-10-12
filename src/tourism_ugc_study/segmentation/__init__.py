"""切分的历史兼容导出：旧客观规则仅供既有调用及历史结果复现。

保留旧名称和身份以免历史调用静默变成新算法；新批次使用semantic_api
及显式冻结配置，正式生成入口为scripts/text_segment_research.py。
导入本包不会读取配置、接触数据库或加载模型。
"""

from .api import segment_body, segment_text
from .contracts import SegmentationIssue, SegmentationResult, TextSegment
from .engine import IMPLEMENTATION_VERSION, RULE_VERSION

__all__ = [
    "IMPLEMENTATION_VERSION", "RULE_VERSION", "SegmentationIssue", "SegmentationResult", "TextSegment",
    "segment_body", "segment_text",
]
