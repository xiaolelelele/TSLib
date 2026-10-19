# 本轮结果复查

实验执行日期2026-09-29；报告整理日期2026-09-30。原始训练采用协议v2，快照修订号为v1。数据、模型和结果均保存在本地工作区，无需重新下载。

- [研究报告](../../repair_and_followup_results_2026-09-29.md)给出判断及限制。
- [独立指标与选择审计](independent_audit.json)、[完整汇总](summary/evidence.md)和CSV可逐项复查。
- [原始运行目录](../../../trend_lab_runs/repair_posterior_20260929/)保留源码/协议/配置冻结、CSV哈希、运行命令、checkpoint、路径、validation锁及完成记录。
- `source/`保存生成本轮分析的Python源码与最终研究文档快照；文档快照的相对链接保持原文件写法，日常阅读请使用上面的原位置。
- [机制分解](mechanisms/README.md)及[昨日同小时基线](seasonal_naive/README.md)是结果后诊断，未改变模型或选择。

在 `/home/lgc/TSLib` 下，用已有带NumPy的解释器复查。下面输出路径必须尚不存在；已有工件不会被覆盖。

```sh
/home/lgc/.conda/envs/TFB/bin/python DTL/research/analysis/repair_posterior_20260929/source/audit_repaired_experiment.py \
  --runs \
  DTL/trend_lab_runs/repair_posterior_20260929/results_v1/repair_electricity_dlinear \
  DTL/trend_lab_runs/repair_posterior_20260929/results_v1/repair_traffic_dlinear \
  DTL/trend_lab_runs/repair_posterior_20260929/results_v1/repair_traffic_linear \
  DTL/trend_lab_runs/repair_posterior_20260929/results_v1/additional_traffic_dlinear \
  DTL/trend_lab_runs/repair_posterior_20260929/results_v1/additional_electricity_dlinear \
  --output /tmp/trend_repair_new_audit.json

/home/lgc/.conda/envs/TFB/bin/python DTL/research/analysis/repair_posterior_20260929/source/summarize_repaired_experiment.py \
  --audit /tmp/trend_repair_new_audit.json \
  --output-dir /tmp/trend_repair_new_summary
```

审计重算训练归一化、标签、阈值/scale、时间切分、validation选择和最终指标；不依赖PyTorch或原评分器。完整test候选库未保存，审计未独立重建该库；神经网络推理另由checkpoint重载验证。投影梯度残差来自已保存的求解诊断，不是第二种优化器的独立复解。20,327是工程检查项数，不是独立样本量。

全部150次网络拟合已完成，GPU3已释放。重新训练须创建新的输出目录并自行保留相应执行环境；本轮最终报告不依赖继续运行的进程。
