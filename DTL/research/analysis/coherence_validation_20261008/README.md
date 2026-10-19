# 2026-10-08 方向—均值兼容性候选验证

结果和论文取舍见[完整报告](/home/lgc/TSLib/DTL/research/coherence_validation_results_2026-10-08.md)与[创新点文档第14节](/home/lgc/TSLib/DTL/research/trend_accuracy_literature_and_ideas_2026-09-26.md:875)。本目录使用已有冻结模型的预测，不新增拟合/λ选择；旧test明确为已审阅开发资料。

主要工件：

- `controls_rule_lock.json`、`controls_completion.json`：固定首步控制协议、37项来源SHA和完成状态。
- `cases/`：12case × validation/legacy两个stage的输入、修复、完整路径与逐case诊断；原路径和仅归一重播逐位一致。
- `per_case_diagnostics.csv`、`per_case_metrics.csv`、`mean_metrics.csv`、`paired_differences.json`：首步诊断、所有控制指标及1/3日配对区间。
- `supplement_scope.json`、`whole_horizon_necessary_conditions.csv`、`whole_horizon_conditional_errors.csv`、`supplement_paired_comparisons.json`：之后追加的全H必要条件与固定对照。必要条件没有失败不代表完整可行。
- `theory_probe.py`、`theory_results.json`、`theory_note.md`：小H合成机制与无界整数首步条件。
- `real_flow_protocol.md`、`real_flow_certificates.py/json`、`real_flow_certificates_note.md`、`real_flow_certificates/`：108条固定validation曲线的数值flow诊断和214份见证。107可行/1必要违规；抽样未触发首步修复，不能外推全量违规率或当修复收益。
- `moment_floor_validation_protocol.md`、`moment_floor_validation/`：全段必要矩floor，事后、仅validation，无test应用、拟合或重选。
- `audit_inputs.*`、`audit_flow.*`、`audit_controls.*`、`audit_witnesses.*`、`audit_moment_floor.*`和`audit_note.md`：独立数值、密集DP、源哈希与bootstrap复算。检查数是工程断言，不是统计样本量。
- `coherence_fixed_repair_effects.png`与`plot_render_manifest.json`：固定控制估计/区间图；独立绘图脚本仅读取已保存统计结果。
- `main_document_before.md`：主文档更新前备份；第0–13节内容保持不变。
- `delivery_checks.json`、`delivery_manifest.json`：最后工件/来源核验与交付哈希，非新增科学结果。

PPT及其独立文字、布局、来源审计在[汇报目录](/home/lgc/TSLib/DTL/research/presentations/2026-10-08)。原TIME实验目录和checkpoint保留不变。复现命令在完整报告中，所有事后规则都保留其真实提出顺序。
