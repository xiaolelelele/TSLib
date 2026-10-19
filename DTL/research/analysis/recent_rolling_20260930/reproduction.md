# 重现与文件入口

正式轮次使用 `/home/lgc/TSLib/DTL/trend_lab_runs/recent_rolling_20260930/source/v2`，不是当前工作目录里可能继续编辑的代码。v1 是有 TF32 重载差异的技术冒烟，保留但不进入主结论。主协议含训练前数据质量修订；事后控制另有分阶段协议。

- [主配置](../../../trend_lab_runs/recent_rolling_20260930/configs/matrix_v2.json)：两个数据、两个年份、两个骨干、三个 seed，40 epoch/patience8。
- [冻结输入 hash](../../../trend_lab_runs/recent_rolling_20260930/input_sha256_v2.json)：65 项源码/配置/数据及其 metadata。
- [原始数据 manifest](../../data/recent_hourly_20260930/download_manifest.json)：1,279 文件及 URL、SHA256；下载 18.03 MB。
- [完成记录](../../../trend_lab_runs/recent_rolling_20260930/results_v2/completion.json)：24 组，1282.55 秒，PyTorch 峰值已分配显存 46,859,776 字节；CUDA 上下文另占显存，已结束释放本轮任务。
- 每个 `results_v2/<data>/<year>/<backbone>_seed<seed>/` 有 `validation_lock.json`、`checkpoint.pt`、`predictions.npz`、`solver_diagnostics.json`、`result.json`；CaReTS 对照在同级 `carets_seed<seed>/`，供两个骨干共用。
- [主审计](independent_audit_v2.json)及[控制审计](controls_independent_audit.json)不调用 runner 的计分、选择或 ridge 实现。QP 检查使用 NumPy 的独立梯度和残差；范围检查用原 CSV 标签，避免浮点逆变换的零点误判。
- [逐 seed](summary/per_seed.csv)、[组均值](summary/summary.csv)、[条件性时间块区间](summary/paired_intervals.csv)、[门控与同候选](summary/gate_increment_vs_alpha1.csv)、[范围控制增量](summary/clipped_increment_vs_clipped_base.csv)。范围预测在 `summary/*_ranges.npz`，其余控制在 `controls/*.npz`。
- 本轮分析源码快照在 `source/`，检查日志在 `checks/`；`artifact_sha256.json` 记录最终分析产物。主实验源文件由 input hash 单独覆盖。

原执行使用 TFB Python 和物理 GPU 3，单进程 PyTorch 显存比例上限 .045、关闭 matmul/cuDNN TF32。重跑时先选择当前有容量的 GPU；下面是原执行参数，新输出目录避免与正式结果混合：

```bash
cd /home/lgc/TSLib/DTL/trend_lab_runs/recent_rolling_20260930/source/v2
CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/lgc/.conda/envs/TFB/bin/python -m DTL.trend_lab.recent_run --config /home/lgc/TSLib/DTL/trend_lab_runs/recent_rolling_20260930/configs/matrix_v2.json --output /home/lgc/TSLib/DTL/trend_lab_runs/recent_rolling_20260930/reproduction_v2
```

重新核验正式已有结果：

```bash
cd /home/lgc/TSLib
/home/lgc/.conda/envs/TFB/bin/python DTL/research/audit_recent_rolling.py --root DTL/trend_lab_runs/recent_rolling_20260930 --output /tmp/recent_main_audit.json
/home/lgc/.conda/envs/TFB/bin/python DTL/research/audit_recent_controls.py --root DTL/trend_lab_runs/recent_rolling_20260930 --analysis DTL/research/analysis/recent_rolling_20260930
```

`analyze_recent_rolling.py` 汇总主矩阵、条件性区间和范围控制；`diagnose_recent_controls.py` 生成固定 alpha=10 的四类事后控制；`plot_recent_rolling.py` 从已汇总 CSV 导出科学图。分析不能替代新样本、再训练/选择不确定性或正式风险假设。

原始正式方法选择先落盘，随后只计算冻结的测试输出。分析中的范围组合按第二阶段 post hoc 记录生成，没有重选原模型、alpha 或 gate。已看过的 Q4 数据不再用于追加主网格。
