# 第四轮：固定三元均值复核协议

2026-09-28。用户要求“固定三元均值继续复核”。本协议在新增 Electricity / Traffic 测试结果产生前固定。只复核既定均值及必要的去方向头对照，不根据新测试表现添加成员、搜索权重或改变验证门槛。

## 1. 问题与数据范围

- 主方案为 `learned_threeway = (learned_selected + flat + difference) / 3`。验证它是否提高原始方向准确率，同时保持或降低同一 backbone 的原始预测 MSE；进一步检查是否优于 validation 选出的单方案，以及学习方向头是否有贡献。
- 新增验证：本地 Electricity、Traffic 的原 CSV **按原列序取前8个数值列**（不含日期），保留全部观测行；joint linear 与 DLinear；各 seeds `[0,1,2,3,4]`。这是8通道原型验证，不能表述为完整 Electricity / Traffic benchmark。
- 原始路径为 `TSLib/Time-Series-Library/dataset/electricity/electricity.csv` 和 `TSLib/Time-Series-Library/dataset/traffic/traffic.csv`（相对于项目根 `/home/lgc/TSLib`）。列选择不使用目标相关性、误差或测试效果。可生成8列派生 CSV 以减少加载开销，必须保留数值字符串、时间戳、行序，并记录原/派生文件 SHA256 和列映射。
- 训练前检查全时间戳、重复/缺口、行列数；数值质量统计只针对训练段，记录非有限值、方差与常数列。异常不能按测试收益决定修复；需先记录明确规则。先 CUDA smoke，再运行全部已固定组合。
- 探索性复核：复用以下已冻结融合工件中的8个数据×骨干设置、各3 seeds，不重训、不重选已有原子方案。这些测试段已经看过，本轮新增去头对照不能算新的独立验证：
  - `/home/lgc/TSLib/DTL/trend_lab_runs/fusion_validation_exploratory_etth_4cells_20260927T162300Z`：ETTh1 / ETTh2 × linear / DLinear，共4组。
  - `/home/lgc/TSLib/DTL/trend_lab_fusion_runs/fusion_evaluation_additional_validation_20260927T162105Z/WeatherGPU2`：Weather × linear / DLinear，共2组。
  - `/home/lgc/TSLib/DTL/trend_lab_fusion_runs/fusion_evaluation_additional_validation_20260927T162105Z/ExchangeRateGPU3`：Exchange Rate × linear / DLinear，共2组。
  - 读取各根目录 `results.json` 的 `groups` 和对应 `predictions.npz`；利用记录的路径键和选择身份，不猜测原子方案路径。新 `learned_threeway` 的部署路径必须复现原 `threeway_mean`。

## 2. 不变的训练与原子选择

- L=96、H=24、尺度 `[1,2,4,8]`；按时间分 **train / calibration / validation / test = 60/15/10/15**；train stride=4、eval stride=24；40 epochs、patience=8、batch=64、lr=0.001、hidden=32；方向阈值只用训练标签的0.2分位数（下限1e-6）。尺度单位为观测行。切分顺序以一直使用的 `trend_lab/data.py` 第170行及实际窗口为准；第三轮协议将中间两段名称写反，本轮在训练/融合评分前核对并更正文档描述，不改变既有代码或数据边界。
- 每 seed 延续 base、difference、direction_head、residual、residual_direction 五个训练网络及原11项工件兼容对照。没有新增学习架构；不跑已暂停的 event / hazard。`difference` 是普通差分 MSE 训练对照，不能称为已实现完整 TDT。
- 温度只由 calibration NLL 选择。`learned_single` 与 `learned_multi` 共用原多尺度方向头，前者只在解码时保留 k=1 约束。
- learned_single、learned_multi、prior、flat 均从 base 校正；alpha=`[0,0.1,0.3,1,3,10]`，40投影步，max_rms=0.5。prior 只从训练标签频率计算；flat 固定 flat 概率为1，并不等于直接输出常数序列。
- 每 seed 只在 validation 选择：相对 base 的 **raw sign accuracy 严格提高，标准化 MSE <= base + 1e-8**；候选按方向准确率降序、MSE升序、alpha升序；无候选则 alpha=0。alpha=0直接复用 base 原路径。
- `learned_selected` 在选定的 learned_single / learned_multi 中按 validation 方向准确率降序、MSE升序选择，完全相同优先 single。`standalone_selected` 在 base、difference、residual、residual_direction、learned_single、learned_multi、prior、flat 中按同一联合门槛选择；无合格候选回退 base，完全相同依上述列序。
- 全部选择先于 test 评分。测试数据不用于选 alpha、成员、门槛或融合形式。保存完整 validation alpha 表和路径，不能只存赢家。

## 3. 恰好四个固定组合

| 标识 | 固定数值路径 | 用途 |
|---|---|---|
| learned_threeway | (learned_selected + flat + difference) / 3 | 主方案 |
| nohead_threeway | (base + flat + difference) / 3 | 用 base 替代学习校正，判断方向头贡献 |
| base_difference_mean | (base + difference) / 2 | 普通预测均值对照 |
| flat_difference_mean | (flat + difference) / 2 | 无学习头的两方案均值对照 |

- 每个组合均保存 **raw（直接均值）** 和 **deployed（按 validation 联合门槛判断；失败则回退 base）** 两种路径，身份和得分分开。沿用 float32 固定均值运算；成员完全相同时直接返回原路径，避免舍入差异被误认为方法收益。
- 主复核评价 deployed learned_threeway 分别相对 base、standalone_selected 和 deployed nohead_threeway。机制归因必须另外比较 **raw learned_threeway vs raw nohead_threeway**，不能把两者验证回退不同造成的差异归因给方向头。
- 记录 validation / test 上实际不同成员数、方向头是否改变路径、raw learned_threeway 是否异于 raw nohead_threeway、部署回退和最终路径身份。至少两个不同成员才能算真正均值融合；learned_selected 等于 base 时不算实际学习校正贡献。
- 不新增可学习门控，不搜索组合或权重，不对四个组合再用测试挑选“最优方法”。

## 4. 统计与审计

- 主方向指标为相邻未来步相对前一步（首步相对最后历史值）的原始 `sign` 准确率，真实精确零计为第三种 sign；不是默认50%机会水平的二分类指标。方向标签由同一最终数值路径产生。
- 主 MSE 为原训练标准化空间的原始数值预测 MSE；同时报告原单位 MSE、各尺度三分类准确率及 macro-F1，不能以补充指标替代主判据。
- 按数据×骨干分别报告每 seed、均值、seed标准差、联合通过次数。方向用百分点差；MSE百分比用 seed 均值之比。只算均值有改善不能称所有 seed 成功，回退为 base 不计严格方向成功。
- 以配对窗口差值先跨拟合seed取均值，再进行8窗口时间块 bootstrap，2000次、随机种子0、95%区间。各域独立；不能将5次拟合当作5个独立数据集，也不能跨域拼接窗口。区间条件于这些数据、训练和选择，不作已校正多重检验的正式显著性声明。
- 四个组合各自相对 base / standalone_selected，以及主方案相对 nohead 的 raw、deployed 比较均保留窗口输入、差值和区间。
- 新 evaluator 在旧数据 smoke 和独立审计后冻结，之后才评分新数据。保留源码快照/哈希、配置、原/派生数据、checkpoint、原始预测和新 NPZ 的哈希及重载记录。新输出不覆盖任何旧实验。
- 独立审计从保存的数值路径重新计算 float64 方向/MSE/三分类指标、float32组合、validation选择、门槛回退、身份和seed均值bootstrap。prior若只复核保存计数到概率而未从原CSV重建，必须明示这一审计范围。

## 5. 预先固定的解释边界

- **每个设置的稳定支持**：主方案均值相对 base 和 validation 选出的 standalone 均满足“方向严格提高、MSE不增加”，并分别至少4/5 seeds满足同一联合条件。未达到时如实保留均值信号及通过次数，称为局部/不稳定证据，不扩大为稳定改进。
- **跨域推荐固定三元作为默认方案**：需要上述稳定支持在两个新增域的 DLinear 上都成立，并检查 linear 是否存在退化；否则最多保留特定条件下的简单集成候选。
- **学习方向头的增量价值**：必须有 raw-to-raw 主方案相对 nohead 的实际路径差异和重复的方向/MSE联合改善；强支持同样要求均值及至少4/5 seeds联合改善，并在两个新增域的 DLinear 上复现。单靠deployed回退差异或单个seed不能恢复一般性方向头创新主张。
- 若 nohead 或固定两方案均值达到/超过三元主方案，应优先更简单实现；若固定均值无法反复超过验证选出的单方案，降低融合优先级。阴性结果只否定/降级当前训练预算、成员选择、均值实现及8通道范围，不宣称否定所有集成或所有方向学习。
- 本轮只为既定三元方案和必要消融提供证据；不根据新测试弱点立即调整方法再在同一测试段证明有效。

## 6. 交付

保存4个新训练设置（共20次seed拟合、每次5个网络）、8个旧设置的新消融复算、独立审计、全量结果表和可重现命令。更新原 `trend_accuracy_literature_and_ideas_2026-09-26.md` 的第9节：固定三元是否值得继续、方向头是否必要、简单均值或单方案是否更合适，以及先前结论如何修订。
