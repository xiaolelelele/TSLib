**独立审查工件与复现说明，2026-09-29**

总报告：[independent_review_2026-09-29.md](/home/lgc/TSLib/DTL/research/independent_review_2026-09-29.md)。没有修改原实验、源代码或选择结果。

本目录保存了审查时使用的 `source/trend_lab` 源码副本。复制的 `trend_*probe.py` 仅将import路径改为该副本，数据路径仍指向原存档，避免日后修复原代码后无法复现本次发现。`source/research` 保存审计/汇总代码；本目录不是新的训练工程。

- `trend_calibration_probe.py`：重载Electricity两个骨干及Traffic–DLinear的seed0 checkpoint；在原calibration段拆分MSE、0.1×CE及输出梯度范数。关键P1证据。
- `trend_surrogate_probe.py`：辅助分类边界的分布反例，以及ETTh旧预测中的损失比例。
- `trend_code_probe.py`：边界最小例子、旧ETTh存档40步与CPU400步的初步敏感性检查。
- `trend_projection_convergence_probe.py`：同输入、同CPU的40/400/1000步对照，排除设备差异。增迭代仅为诊断，没有据此重选alpha或替换结果。
- `trend_label_precision_probe.py`：训练float32与评分float64的三分类边界差异。
- 各探针的 `_results.txt`：本轮实际运行输出。
- `trend_review_diagnostics.py` / `.json`：另写的NumPy诊断，直接读保存NPZ与派生CSV，核对raw sign、逐通道/预测步、改对改错及小差分区域贡献。此脚本输出JSON到 `/tmp/trend_review_diagnostics.json`，不会修改本目录JSON。
- `reviewer_fixed_threeway_{old,new}_audit.json`：本轮重跑审计输出，分别14112、9504项通过。
- `table_reproduction_checks.json`：5张重新生成CSV与原CSV逐字节相同的哈希记录。重复的大CSV留在临时目录，本目录不再保存副本。
- `trend_{code,eval,novelty}_review.md`：三路分工笔记；其中临时路径已替换为本目录持久路径。总报告的综合判断优先。
- `manifest.json`：本目录文件及总报告哈希，另记录当前核心源代码与存档源代码一致性。

CPU探针运行示例（其他探针替换文件名即可，输出可重定向到新的临时文件）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/lgc/.conda/envs/TFB/bin/python /home/lgc/TSLib/DTL/research/analysis/review_20260929/trend_calibration_probe.py
```

原14项unittest的本轮运行方式：工作目录 `/home/lgc/TSLib/DTL`，运行下列命令，14项通过。它验证管线，不覆盖本次发现的所有研究对照缺陷。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/lgc/.conda/envs/TFB/bin/python -m unittest discover -s trend_lab/tests -v
```

新旧审计与5张CSV可用下列包装脚本复现。输出目录必须不存在；它直接调用审计函数，避免原审计CLI向原实验目录写入文件。此包装脚本经过语法检查；本轮审计和CSV核对已通过相同函数完成，未为验证包装入口再重复整轮计算。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/lgc/.conda/envs/TFB/bin/python /home/lgc/TSLib/DTL/research/analysis/review_20260929/reproduce_review_tables.py --output /tmp/trend_review_rerun
```

这些检查条件于现有数据、模型和选择。审计通过不是新颖性证明，也不是对新时间段或重新训练的统计保证。checkpoint加载只针对本地已有工件；所有计算使用CPU。
