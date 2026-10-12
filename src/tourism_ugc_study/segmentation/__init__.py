"""旅游UGC正文切分；唯一生成接口为semantic_api及冻结连贯块配置。

导入本包不读取配置、接触数据库或加载模型。旧算法已删除，不提供旧名称别名。
"""

from .contracts import SegmentationIssue, SegmentationResult, TextSegment

__all__ = ["SegmentationIssue", "SegmentationResult", "TextSegment"]
