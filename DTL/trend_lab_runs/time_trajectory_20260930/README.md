# TIME轨迹方案：实验归档与复现

整理日期2026-10-03，实验标识保持20260930。[最终报告](/home/lgc/TSLib/DTL/research/time_trajectory_validation_results_2026-10-03.md)给出科学结论；本文件解释版本和命令。原论文正文未改。

## 结果与版本

| 路径 | 用途 |
|---|---|
| `source/v2`、`source_manifest_v2.json` | 正式训练冻结版本：76源文件、两个原配置；包含训练前质量修订 |
| `configs/matrix.json`、`matrix_v2.log`、`results_v2` | 正式18case、9shared、153拟合；每case有checkpoint、原预测、validation锁及decoder诊断 |
| `anchor64_repair_v2` | canonical推理修复：真实anchor64加直接signed32；13条竞争候选替换规则；18case、9shared |
| `anchor64_repair` | 历史v1修复档案，15候选池包含旧bug版，仅供过程复核，不能作最终竞争比较 |
| `precision_controls` | native12case、192条无训练/无选择控制；保留原来源哈希 |
| `analysis` | 510指标复算、逐case/通道/幅度/相位/应用诊断、1/3日bootstrap、修复竞争前沿、成本和图 |
| `source/v1`、`smoke_v1`、`smoke_v2` | 初始技术检查与重复smoke，非正式科学结果 |
| `source/v3`、`source_manifest_v3.json`、`smoke_v3_cpu` | 未来代码技术版本：round-trip CSV解析、真实锚点接口、源码/数据/配置resume保护，另有OCE公式测试源；没有用于重训正式矩阵 |

v2保留默认pandas解析，后发现的末位差异不改变本轮float32训练张量/真实标签；规范锁定重解码敏感性结果单独保存。v3不能冒充v2模型的训练来源。正式训练与未来源码各自有manifest，不能将修改后的live目录覆盖到冻结目录。

数据位于[prepared目录](/home/lgc/TSLib/DTL/research/data/time_validation_20260930/prepared)。主数据版本、原源LFS哈希和处理记录见[获取审计](/home/lgc/TSLib/DTL/research/data/time_validation_20260930/independent_acquisition_audit.json)、[质量说明](/home/lgc/TSLib/DTL/research/time_validation_data_audit_2026-09-30.md)。路径为本机绝对路径，迁移时需要明确修改配置副本并重新记录哈希，不宣称原配置未变。

## 环境

本轮Python3.8.20、PyTorch2.4.1+cu121、NumPy1.24.4、pandas1.5.3；解释器`/home/lgc/.conda/envs/TFB/bin/python`。正式训练物理GPU3、CUDA进程显存比例.045、TF32/flash-SDP/memory-efficient-SDP/MHA-fastpath关闭、CPU2线程。峰值allocated约412MiB；这是本进程统计，不含其它进程或完整GPU预留。

[技术检查](/home/lgc/TSLib/DTL/trend_lab_runs/time_trajectory_20260930/technical_checks.json)记录smoke重复数值差：79个数组非逐位相同，最大差2.5034e−6，容差失败0；474方向数组类别一致，6case选择一致。不能据此声称正式训练逐位复现或独立重复训练已完成。

## 重建统计，无需训练

下列命令仅读取锁定输出，重写派生`analysis`。root后处理不选择模型/λ；期望18主case、18修复case及12控制，缺失或评分不一致会失败。

```bash
cd /home/lgc/TSLib
TASK_RUN=/home/lgc/TSLib/DTL/trend_lab_runs/time_trajectory_20260930
TASK_PY=/home/lgc/.conda/envs/TFB/bin/python
MPLCONFIGDIR=/tmp/time_trajectory_mpl "$TASK_PY" DTL/research/analyze_time_trajectory_validation.py --run "$TASK_RUN"
```

`analysis/input_artifact_manifest.json`绑定读取的预测/结果/锁；`summary.json`记录后处理脚本哈希；`analysis_manifest.json`记录派生文件哈希。bootstrap按每origin先平均3个seed，再对连续非重叠1/3天日期组配对重采样2000次，保留起报数量权重。它是给定模型和测试期的敏感性统计，不是训练分布总体CI、多个比较统一95%保证或条件风险界。汇总百分比采用平均MSE之比；某些历史precision表采用逐seed百分比再平均，分母顺序不同。

## 独立核验命令

将输出写到新目录，保留原审计。runner和修复强制完整18case；decoder完整报告另含实际case数及样例范围。精度审计要求12case的身份集合由最终完整性审计补查。

```bash
TASK_BASE=/home/lgc/TSLib/DTL/research/analysis/time_baseline_sources_20260930
TASK_DEC=/home/lgc/TSLib/DTL/research/analysis/time_trajectory_20260930
mkdir -p "$TASK_RUN/rechecks"
"$TASK_PY" "$TASK_BASE/audit_runner_outputs.py" --run "$TASK_RUN/results_v2" --checkpoint-audit --expected-cases 18 --output "$TASK_RUN/rechecks/runner.json"
"$TASK_PY" "$TASK_BASE/audit_anchor64_repair.py" --run "$TASK_RUN/results_v2" --repair "$TASK_RUN/anchor64_repair_v2" --expected-cases 18 --output "$TASK_RUN/rechecks/repair.json"
"$TASK_PY" "$TASK_DEC/audit_decoder_matrix.py" --results "$TASK_RUN/results_v2" --output "$TASK_RUN/rechecks/decoder.json"
"$TASK_PY" "$TASK_DEC/precision_controls_audit.py" --controls "$TASK_RUN/precision_controls" --frozen-source "$TASK_RUN/source/v2" --output "$TASK_RUN/rechecks/precision"
```

完整独立审计20586/4260/6462/660项分别通过。decoder的2844条dense复解为origin0/中点、全通道与部分预声明控制；checkpoint独立重载为18×7模型的前11个test origin。它们不是全测试路径穷举或第二次独立拟合。[最终核验说明](/home/lgc/TSLib/DTL/research/analysis/time_trajectory_20260930/final_artifact_audit_note.md)另记录550项完整性检查；[独立统计复核](/home/lgc/TSLib/DTL/research/analysis/time_trajectory_20260930/final_claim_statistical_checks.json)141项通过。

## 完整重训的归档布局

以下是复现实验入口示例，**本次收尾没有再次启动**。必须使用新的目录，且所用物理GPU需空闲；不停止其它训练。v2 runner中的.045显存比例为代码固定值。

```bash
TASK_REPRO=/home/lgc/TSLib/DTL/trend_lab_runs/time_trajectory_reproduction
set -e
mkdir "$TASK_REPRO"
ln -s "$TASK_RUN/source" "$TASK_REPRO/source"
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$TASK_RUN/source/v2" OMP_NUM_THREADS=2 "$TASK_PY" -m trend_lab.trajectory_run --config "$TASK_RUN/configs/matrix.json" --output "$TASK_REPRO/results_v2"
"$TASK_PY" "$TASK_BASE/repair_shared_anchor64.py" --run "$TASK_REPRO/results_v2" --output "$TASK_REPRO/anchor64_repair"
"$TASK_PY" "$TASK_BASE/rebuild_anchor64_competitor_v2.py" --run "$TASK_REPRO/results_v2" --archive "$TASK_REPRO/anchor64_repair" --output "$TASK_REPRO/anchor64_repair_v2"
"$TASK_PY" /home/lgc/TSLib/DTL/research/run_time_precision_controls.py --results "$TASK_REPRO/results_v2" --output "$TASK_REPRO/precision_controls" --frozen-source "$TASK_RUN/source/v2"
MPLCONFIGDIR=/tmp/time_trajectory_mpl "$TASK_PY" /home/lgc/TSLib/DTL/research/analyze_time_trajectory_validation.py --run "$TASK_REPRO"
```

配置的reference_root和数据仍指向已冻结原归档，因此保留该归档。修复流程先用原checkpoint生成直接signed trace，再以v2替换规则选竞争基准；不可从old z32减x32反推分支。重训只期望容差级重现，不保证CPU/GPU或不同硬件逐位一致；选择靠近边界时也可能发生变化。

## 未来v3单元测试

冻结测试历史上混用`DTL.trend_lab`与`trend_lab`命名空间。直接在`/tmp`只设source/v3路径，会有一个导入失败；下列别名明确使两种名字都指向冻结v3，而不误用live代码。2026-10-03复核24项通过，0失败；测试包含短H穷举、dense DP、greedy、true-anchor公式和runner锁序。该别名只为测试启动，不修改冻结源。

```bash
cd /tmp
PYTHONPATH="$TASK_RUN/source/v3" OMP_NUM_THREADS=2 "$TASK_PY" - <<'PY'
import sys, types, unittest
import trend_lab
package = types.ModuleType('DTL')
package.__path__ = []
sys.modules['DTL'] = package
sys.modules['DTL.trend_lab'] = trend_lab
names = ['trend_lab.tests.test_trajectory_decision',
         'trend_lab.tests.test_trajectory_baselines',
         'trend_lab.tests.test_trajectory_runner_audit']
result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromNames(names))
raise SystemExit(not result.wasSuccessful())
PY
```

live及v3的resume guard保存配置/包源码/数据/reference hashes，拒绝不同来源或旧无版本目录继续混写。本轮v2矩阵的来源保护由预先冻结的manifest和最终独立审计提供，不能声称它已经使用后来增加的guard。[guard检查](/home/lgc/TSLib/DTL/trend_lab_runs/time_trajectory_20260930/resume_guard_checks.json)3项通过；[v3 CPU技术smoke](/home/lgc/TSLib/DTL/trend_lab_runs/time_trajectory_20260930/smoke_v3_cpu/completion.json)6case、51个小拟合仅证明接口可执行，不代表新科学结果。
