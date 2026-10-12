# 运行结果

本目录统一保存过去分散在 `artifacts/`、`reports/`、`output/` 和 `provenance/` 的运行产物。默认不提交大型或可重建文件。

每次正式运行直接使用 `results/<run_id>/`，在同一目录内保存解析后的配置、命令、数据与切分 manifest、环境、日志、指标、预测、模型权重和人类可读摘要。谱系信息随运行包保存，不再维护独立的 `provenance/` 目录。

- `exports/`：当前确有的本地 DOCX、PDF 或 HTML 导出；默认忽略。
- 论文实际使用的图表可在需要时创建 `paper/`，并记录来源 `run_id`。

正式冻结数据所依赖的完整运行证据，无论是否已经被论文引用，都须长期保留；未冻结的临时探索产物才按具体生命周期另行清理。大模型权重或其他大型产物不进入普通Git历史。

当前完整清洗轮次为`research-cleaning-20260911-autodl-r2/`：36批输入、预测、17,696份缓存向量、逐条引用、回传/释放回执、日志、装配及最终验收共同保全。不能把这些向量当普通可清理缓存；它们是本次冻结结果的证据。另有三份环境元数据由`environment-references.json`绑定，按冻结清单保护。

候选SQLite和原始内容/图片归档位于`data/processed/`；冻结登记为`governance/research-data-freezes.json`。具体根、字节校验及不可变保护见[研究数据冻结与校验](../docs/protocols/研究数据冻结与校验.md)。既有MPS/Runpod取消尝试的推理产物已经销毁，不恢复、不混入本轮；其他历史训练/人工证据仍保留。

切分算法的活跃复现包为 `semantic-coherent-freeze-20261012-01/`，原展示／复核已集中到冻结包的 `evidence/`。旧试验入口、旧配置与运行副本已按逐文件SHA校验后收入 `segmentation-archive-20261012-01/archive.tar.gz`，索引为同目录 `inventory.json`；旧路径即归档成员名。当前输入包与NAS来源核验保留，复现不再依赖已清理目录。采纳范围及后续事项只在[总备忘录B05](../docs/planning/研究问题与待办总备忘录.md#b05embedding自然语义分段与工程接入)维护。
