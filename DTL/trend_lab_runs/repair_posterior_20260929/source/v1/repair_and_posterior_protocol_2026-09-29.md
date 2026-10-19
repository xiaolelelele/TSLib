# 修复与完整后验候选路径研究：冻结协议

日期2026-09-29。回应独立审查及用户“补文献、修复实验、继续研究”的要求。先完成代码/CPU/CUDA smoke验证，再冻结代码与本协议哈希，随后启动下列真实数据矩阵。所有结果写新目录；不覆盖9月27–28日工件。本文使用的“冻结”是本地带哈希的预定记录，不是外部注册或首创证明。

## 问题与证据范围

1. 修复死区阈值兼作损失除数的问题后，方向监督残差是否仍明显逊于学习投影？
2. Traffic的学习方向校正在拆分早停/校准、求解精度一致及更直接损失基线下能否重复？
3. 同一个方向头与同一个有限路径库，完整后验的逐窗口路径决策是否优于validation统一选择投影强度？

旧前8列测试已有多轮结果，重跑为**修复后的探索性复验**。新增列8–15为本轮预先固定的额外通道验证；同一数据集、同一时间轴可能高度相关，不称新独立数据域或新独立时间测试。所有组固定种子[0,1,2]，不根据结果增加种子/选择其他列。

| 组 | 数据列，0-based数值列 | 骨干 | 定位 |
|---|---|---|---|
| repair_electricity_dlinear | Electricity 0–7 | DLinear | 最严重损失尺度缺陷的重训复验 |
| repair_traffic_dlinear | Traffic 0–7 | DLinear | 主局部信号复验 |
| repair_traffic_linear | Traffic 0–7 | joint linear | 已有另一骨干复验 |
| additional_traffic_dlinear | Traffic 8–15 | DLinear | 额外通道验证 |
| additional_electricity_dlinear | Electricity 8–15 | DLinear | 额外通道验证及负结果检查 |

## 固定数据/训练规则

- 源文件使用本地Time-Series-Library Electricity/Traffic CSV。派生8列文件只按列序复制原始日期和数值字符串；不补值、不删行、不根据测试选择列。记录原CSV和派生CSV SHA256。预检检查结构和时间顺序，数值质量统计只在训练段完成。
- L96/H24、scales=[1,2,4,8]；原时间60/15/10/15分割。原15% calibration段按原始时间中点分成earlystop与temperature calibration；丢弃预测目标跨中点的窗口，历史仍可使用已观测过去。二者目标不重叠，不声称统计独立。
- train stride4，其他stride24；train-only标准化；deadzone仍为训练绝对斜率0.2分位数、floor1e-6。标签统一先转float64差分，再与**同一保存的float32阈值**比较。
- 40epochs、patience8、batch64、Adam lr.001、hidden32；所有网络只用earlystop选epoch。训练顺序、初始化与batch shuffle种子记录。base/difference/TDAlign/CosDir共享初始base权重及shuffle序列；普通/方向残差同初始权重及shuffle序列。
- 残差方向loss以训练斜率绝对值中位数作为scale，floor=.02（训练标准化单位）；logits=[(-d-eps)/scale,0,(d-eps)/scale]。eps仅定义分类边界，不作除数。非边界argmax与deadzone一致；等号统一按flat解释。这个一维CE对所有三分类混合分布不保证Fisher一致，属于稳定的同标签监督控制，不是新loss贡献。
- 方向残差权重固定网格[.01,.1]，0对应普通residual另报；CosDir权重[.1,1]，0对应base。温度网格[.5,1,2,4]仅在独立于earlystop的校准目标上选择NLL。它们网格大小明确列出，不声称所有方法总计算预算完全相同。

## 方法与选择

- base、difference-MSE、TDAlign官方损失、CosDir官方损失。TDAlign使用当前官方逐sample/channel的rho，严格dp*dt<0计异号；不冒充与2024v1符号零处理完全相同。CosDir用官方cosine_similarity范数clamp语义，不含UW版本。评价一律保留真实零方向；与CosDir官方DA忽略零的口径不同。
- 普通residual、修复方向监督residual；隐藏宽度尽量匹配多尺度头参数量。记录损失分量与输出梯度比，不能用该比值直接推断模型性能。
- multi、single共享同一个多尺度训练头；另训练一个独立单尺度头作为`projection_single_trained`。它参数更少，记录实际数目；不称参数完全匹配。
- multi的T=1消融、flat、训练prior。所有投影alpha固定[0,.1,.3,1,3,10]；RMS修改预算.5；最大400步、每条路径投影梯度映射RMS容差1e-5；记录是否耗尽预算和实际迭代数。对所有方法使用同一准则；不根据测试改步数。到达最大步数不能标成收敛。
- `learned_selected`仅在multi/single之间按validation联合条件选择。`standalone_selected`在预定各方法之间用同样条件选择。选择只读validation：raw sign严格提高，标准化MSE<=base+1e-8；按方向降序、MSE升序、固定列表顺序打破平局；无合格候选返回base。回退不计方向成功。

**完整后验探索方案**：固定候选库为base、multi的5个非零alpha、single的5个非零alpha，共11条路径；不生成额外测试最优路径。同一个多尺度头计算每条路径实际类别的平均后验概率 `U(z)=mean p(C_eps(A_k z)|x)`。每个窗口选择 `argmax[U(z)-beta*mean((z-base)^2)]`，beta网格[0,.1,1,10]；平局优先base/固定路径顺序。随后只在validation用上述raw sign/MSE联合规则选择beta，否则整体返回base。

这里完整后验效用针对**带死区三分类**，与raw sign不是同一效用；两者必须分别报告。路径到base的距离是修改成本，不是真实MSE增量估计或风险保证。离散候选库有限，不能声称求出连续路径Bayes最优解；该方案只是检验已有后验是否可用于路径决策的低成本探针。

## 判据与报告

- 主要检验最终路径raw sign和标准化MSE，逐尺度三分类、头概率指标、原单位MSE为完整补充。原单位MSE增加须明示，不能将标准化通过表述成所有口径通过。
- 按组/方法逐seed报告；平均MSE百分比用seed均值之比。局部重复信号预定为3/3 seed方向提高且MSE不升，均值方向增益至少0.2pp；这只是筛选阈值，不是显著性或总体成功概率保证。
- posterior候选必须与base及learned_selected直接配对。只有超过base不算完整后验机制得到支持；应同时检查三分类效用、修改量、改对/改错。
- 若学习校正被修复残差/CosDir/TDAlign稳定解释，收缩D主张；若posterior选择不能超过共享路径库的统一alpha选择，停止将该具体候选选择器作为下一轮主线。不在相同测试上继续扩大beta/alpha网格。
- 新旧结果的差异同时包含早停区间、标签精度、解码精度和损失修复，不能将所有变化单独归因于某一修复。严格单因素归因需后续固定其他因素再设计。
- 本轮不完整复现CaReTS、不运行更强骨干/长H/全通道，也未做cross-fitted base、多个新时间切点或有理论保证的双风险选择。保留为明确后续欠缺，不能称正式benchmark已完成。

## 工程验证

必须通过CPU边界/梯度/分段/解码/基线单测及v2端到端；通过GPU3 bounded-training smoke后再完整训练。GPU0/1/2已有任务，不占用。每组保存config、源代码/协议/数据哈希、验证锁定记录、checkpoint、validation与test路径、求解诊断和训练诊断；checkpoint重载逐路径验证。独立NumPy审计重新计算指标、验证选择及后验候选选择，保存新的结果表。

并行的已知分布短H合成实验使用单独的先冻结协议，不与真实训练结果混合。真实后验仅可作oracle；可达前沿只能在明确的有限网格/候选集合上声称精确。
