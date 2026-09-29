# 测试策略

测试目录与 `src/tourism_ugc_study/` 的实际模块对应，例如 `tests/cleaning/`、`tests/annotation/`、`tests/models/text/` 和 `tests/models/vision/`。只有出现第一份相应测试时才创建目录，不预建空的 `unit/`、`integration/` 或 `regression/` 层级。

单元、集成和回归测试使用文件名区分，例如 `test_normalize.py`、`test_cleaning_pipeline_integration.py`、`test_counts_regression.py`。共享合成或脱敏样本增多后再创建 `tests/fixtures/`。

测试不得连接平台、登录账号或真实抓取环境；不得依赖正式数据库才能完成基础单元测试。

`tests/segmentation/`对应独立切分包，使用合成正文和临时SQLite验证冻结边界、
Unicode字素保护、失败状态、来源配对、依赖隔离及只读批处理；单表测试覆盖
重复导入、失败回滚、内容篡改、批次冲突、唯一约束和可复现抽样。真实研究语料全量
对照另存本地`results/`，不作为基础测试的私有数据依赖。
