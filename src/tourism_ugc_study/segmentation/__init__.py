"""客观文本切分独立模块：纯函数接口只依赖标准库及固定版本regex。

数据库读取、配置加载与运行包导出由本包独立适配层提供；导入此入口不会
读取配置、接触数据库或加载清洗模型。这里的结果类型不是未来持久化表设计。
"""

from .api import segment_body, segment_text
from .contracts import SegmentationIssue, SegmentationResult, TextSegment
from .engine import IMPLEMENTATION_VERSION, RULE_VERSION

__all__ = [
    "IMPLEMENTATION_VERSION", "RULE_VERSION", "SegmentationIssue", "SegmentationResult", "TextSegment",
    "segment_body", "segment_text",
]
