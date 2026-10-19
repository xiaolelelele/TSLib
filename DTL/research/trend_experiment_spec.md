# 趋势方向研究：本地原型实验契约

日期：2026-09-27。目标是编写可移至服务器的研究原型，不下载数据，不在本地开展完整实验。轻量测试只验证代码和数据边界，不作为方法有效性证据。创新判断见 [文献与创新备忘录](trend_accuracy_literature_and_ideas_2026-09-26.md)。

## 1. 本轮要区分的假设

| 编号 | 假设 | 必要对照 | 失败判据 |
|---|---|---|---|
| H1 | 从历史直接估计方向概率，比对条件均值轨迹取符号更准确 | 基础预测、多数类、历史方向保持、差分 MSE | 分类头没有超越合适的简单基线 |
| H2，主线 | 把多尺度方向决策作为约束作用到同一条数值轨迹，能提高轨迹方向准确率 | 基础预测、单尺度校正、独立方向头、同输入同预算的直接残差头 | 只有分类头改善；或普通残差头解释全部收益；或必须突破预先规定的 MSE 容忍度 |
| H3，条件分支 | 未来转折状态递推能减少转折附近的方向错误 | 直接残差头、无转折状态的斜率积分头 | 转折并非主要错误来源，或状态递推没有独立增益 |

本轮 H2 采用后验类别约束的凸软投影原型，不宣称它是新的优化算法或已证明的新颖方法。先学习方向，再检验怎样把方向信息兑现为整条预测轨迹的改善。

## 2. 目录和接口

新增独立包 `DTL/trend_lab/`，不修改原投稿代码。仅依赖 Python >=3.10、NumPy、PyTorch；配置用 JSON；测试用 unittest。从 `DTL/` 运行 `python -m trend_lab.run --config trend_lab/configs/synthetic.json`。

模块分工：`operators.py`（标签/差分）、`data.py`（输入/合成/切分）、`metrics.py`（评价）、`models.py`（模型）、`decoding.py`（轨迹投影）、`run.py`（训练/选择/存档）、`tests/`、`configs/` 和 `README.md`。

统一形状：历史 `x=[N,L,C]`，未来 `y/base/path=[N,H,C]`，方向 logits/probs 为 `[N,S,H,C,3]`；类别顺序固定为 `0=down,1=flat,2=up`。scales 为预先固定、无重复且首项为 1 的整数列表。底层算子允许 k<=L；实验 runner 要求 k<L，使历史方向保持对照有完整 k 步历史跨度。

`operators.py` 公开函数：

- `multiscale_slopes(history, future, scales)`：Torch 张量，返回 `[N,S,H,C]`，第 h 步尺度 k 为 `(y[t+h]-y[t+h-k])/k`。`t+h-k<=t` 的点从历史取；第一步锚定最后历史值。禁止把真实未来当成预测输入。
- `direction_labels(slopes, epsilon)`：epsilon 为 `[S,C]`，按 `>epsilon`、`<-epsilon`、其余映射到 up/down/flat；等于阈值属于 flat。
- `fit_thresholds(train_x, train_y, scales, quantile=0.2)`：仅训练集拟合各尺度/变量绝对斜率分位数，floor=1e-6。这是低幅度死区定义，不能称为已知噪声标准差。
- `turn_targets(history, future, epsilon)`：epsilon 为 `[C]`；维护最近非 flat 的一阶方向，遇到相反非 flat 方向时发出转折，构造 `[N,H,C]` 的 0/1 标签。flat 不发事件也不更新方向；第一个未来方向与历史最近非 flat 方向比较；历史全 flat 时首个非 flat 不算转折。

`data.py` 提供 dataclass `Split(x,y,origins)` 和 `DatasetBundle(train,calibration,validation,test,mean,std,metadata)`，均存 NumPy float32 数组，origins 是最后历史点的全局索引。`load_data(config)` 接收下文 data 子字典。

`models.py` 公开 `build_base(name, lookback, horizon, channels, hidden=32)`，name 为 `linear` 或 `mlp`，返回 Torch 模型 `forward(x)->path`；所有路径锚定最后历史值再预测偏移。原型不用外部基准框架。

插件接口：`DirectionHead(lookback,horizon,channels,scales,hidden=32)` 的 `forward(x,base)->logits`；`ResidualAdapter(lookback,horizon,channels,hidden=32)` 的 `forward(x,base)->path`；`EventAdapter(lookback,horizon,channels,hidden=32,use_events=True)` 的 `forward(x,base)->dict(path,turn_logits)`。插件推理只接收 x/base，基础预测冻结。

`decoding.py` 公开 `project_trajectory(history,base,probabilities,scales,epsilon,strength=1.0,steps=40,max_rms=0.5)`，返回和 base 同形的 Torch 路径。保持设备与 dtype，输出 detach；strength=0 精确返回基础路径。

`metrics.py` 公开 `evaluate_predictions(history,truth,pred,scales,epsilon,probabilities=None)` 返回 JSON 可序列化 dict；`paired_block_interval(differences,block_size=8,repeats=500,seed=0)` 返回标量均值差的连续块 bootstrap 95% 区间。无定义指标用 None，禁止 JSON NaN。

## 3. 数据和无泄漏切分

默认按原始时间索引分成四段：train 60%、calibration 15%、validation 10%、test 15%。一个样本的整个预测区间必须落入对应段；历史可以使用该段开始前已经观测到的值。生成窗口后再随机拆分是不允许的。训练可打乱窗口批次。

标准化均值/标准差只用 train 时间段；所有方法、随机种子复用同一数据划分和阈值。默认评估 stride=H 降低预测区间重叠；保留 origins 支持关联误差分析。训练 stride 可取 1。

支持两类输入：

1. 合成序列（在内存生成，不联网）：`persistent_trend`（平滑持续趋势）、`periodic_turn`（历史可推断的周期反转）、`noisy_multiscale`（低频趋势叠加快变项和观测噪声）、`random_walk`（独立对称增量，方向不可预测负对照）、`skewed_increments`（独立增量，70% 为 +1、30% 为 -4，均值 -0.5；多数类 up 是重要对照）。各场景提供不同通道相位/噪声，固定 data seed；不向模型提供潜在状态或未来事件。
2. 用户放在服务器的本地 CSV：必须显式 `value_columns`，可指定 `time_column` 检查严格递增，不静默重排/补值；数值缺失/非有限报错。CSV 不在仓库中，不提供下载函数。

## 4. 可核查的 H2 解码

先用同一个冻结的基础预测器，训练方向头对各尺度/未来步/通道输出概率。方向头监督来自训练目标；模型输入只用 x 和基础预测。

校准集选温度 T（例如 0.5、1、2、4），用多类 NLL 评分。校准不使用测试数据。随后每个约束取后验概率最大的类 c，权重 `w=p_top1-p_top2`，不把概率高等同于修正必然有用。

对单一候选路径 z 优化：

`0.5 * sum((z-base)^2) + alpha/S * sum_k sum(w * penalty(A_k(z)))`

其中 `A_k` 是上述多尺度斜率，up 的 penalty=`0.5*relu(1.5*eps-d)^2`，down=`0.5*relu(1.5*eps+d)^2`，flat=`0.5*relu(abs(d)-eps)^2`。每个约束落到同一 z，不能把每个尺度的分类序列分别积分成多条相互矛盾的路径。该目标是给定概率后的凸软约束问题，但不等同于精确最小化 0/1 方向错误，也没有泛化改进保证。

用固定步数投影梯度下降，步长可取 `1/(1+4*alpha*mean(1/k^2))`。额外约束每个样本修正的 RMS 不超过 max_rms（在训练标准化单位中）；梯度目标对 batch 求和，保证批次大小不会改变同一样本的解。零强度必须逐元素等于 base。禁止从真实未来计算概率或在线选 alpha。

在 validation 上从 `alpha=[0,0.1,0.3,1,3,10]` 选预注册尺度平均方向 accuracy 最高者，约束同一 validation 的标准化 MSE 不超过 base 的 1.01 倍，且平均三分类 macro-F1 不下降（数值容差 1e-12）；并列选择较低 MSE、较小 alpha。accuracy 无严格改善就选 0。单尺度校正独立选择 alpha，评价仍用相同全尺度集合。这是经验选择条件，不是测试分布的风险保证。报告完整验证集 Pareto 表和最终固定 alpha 的测试表现。

## 5. 实验矩阵和实现范围

共享一个已训练 MSE 基础预测器：

- `base`：基础路径。
- `direction_head`：独立方向概率，路径仍为 base；明确单独报告头指标与路径指标。
- `projection`：所有尺度约束的 H2。
- `projection_single`：只启用 k=1 约束，用同一方向头，隔离多尺度约束收益。
- `residual`：同一 x/base 输入、相同训练预算的普通残差 MLP，隐藏宽度选择使参数量尽量靠近 DirectionHead；报告实际参数量，不声称绝对一致。
- `residual_direction`：复用同一个 ResidualAdapter，加与方向头相同的多尺度标签监督。用 `r=path_slopes/eps` 的三个分数 `[-r,1-abs(r),r]` 做可微交叉熵，默认系数 0.1；作为额外方向监督的控制，不能把其收益误认成结构化解码的贡献。首轮固定系数，正式比较需要同预算调参。
- `event` / `event_nohazard`：H3 分支；event 用预测 hazard 在旧斜率与候选新斜率间递推，再从最后历史值积分；nohazard 直接积分逐步候选斜率。输出分未来步，不能将一个类别向量重复 H 次。
- `difference`：与 base 相同初始化/预算，额外一阶差分 MSE，含首步历史锚点；明确标成 difference-MSE baseline，不能冒充 TDT 完整复现。
- 数值简单对照：last-value、按最近 k=1 斜率线性延伸；方向对照：训练集多数类、最后历史方向保持。

所有学习方法使用 Adam 和相同 train/calibration 规则、早停预算。可以只运行配置中的 methods；小规模 smoke 只需 1-2 epochs。主实验配置五个 seeds；数据 seed 独立，保持相同 test samples 以供配对。

## 6. 评价、证据和保存

每个尺度分别报告路径的方向 accuracy、macro-F1、balanced accuracy、真实非平区的 up/down accuracy、真实非平区覆盖率、混淆矩阵；另报未使用死区的一阶 sign accuracy，避免通过扩大 flat 类提高分数。概率头另报 accuracy、macro-F1、Brier、ECE（10 bins）、NLL 及头/路径类别冲突率。

数值指标 MSE/MAE 与逐 horizon 曲线；对一阶转折做容忍窗口 1 的一对一事件 precision/recall/F1、匹配时间误差；分别报告转折邻域和非转折邻域方向错误率及样本数，无事件时返回 None。bootstrap 以每个预测窗口的成对方向正确率差和 MSE 差为输入，用连续块重采样，不能把各 horizon/变量当独立样本。

输出目录必须保留：解析后的 config、数据/切分/归一化/阈值信息、训练 seed、每种方法训练参数量与耗时、校准温度、验证集 alpha 表、所有方法测试指标、按 seed 的均值/标准差汇总、预测与 origins 的 NPZ、可重载模型 checkpoint。测试只在各方法与超参数固定后运行。

本轮不实现 PIR/GARS/CORTEX/FreDF/PS/DILATE 全复现，不将普通 residual 称为其复现；这些是通过机制筛选后必须补的外部强基线。合成收益不能外推真实数据，随机游走负对照失败要优先查泄漏。

负对照的 50% 机会水平仅指对称独立增量的未阈值化一阶方向；三分类死区和跨越已知历史的多尺度标签不能统一以 50% 为界。正式合成验证需多个独立 data seeds，训练 seeds 不能代替数据重复。短 smoke 的区间只检查程序，不提供显著性证据。温度不改变头 argmax；校准是否改善路径需要后续 T=1 消融，本轮未实现。

## 7. 必须通过的少量测试

1. 手算验证多尺度历史锚点、flat 边界、第一步转折和路径/概率两套指标。
2. CSV 时间顺序、目标跨段禁止、测试段数值改变不影响 train scaler/threshold。
3. strength=0 恒等、投影 RMS 上界、batch 分割不改变投影、概率约束有方向作用。
4. 所有模型推理输入无 y；逐 horizon 输出形状与反向传播正常；事件无 hazard 对照可训练。
5. 微型合成端到端运行能输出完整 JSON/NPZ/checkpoint 并重新加载；它只证明管线可用，不报告趋势增益结论。

## 8. 本地代码审查与验证记录

2026-09-27：实现由 GPT-6 Luna（medium）子 agent 完成，主 agent 独立审查并要求修复后复验。原投稿代码未修改，未下载数据，未运行服务器任务或完整研究实验。

已修复的关键问题：多数类/历史方向对照必须覆盖全部尺度和测试样本；路径指标不能冒充分类头指标；单尺度与多尺度独立选 alpha；残差消融共享初始化与训练顺序；首步转折和 flat 保持规则统一；CSV smoke 限制读取量；checkpoint 保存重建参数、归一化、阈值、温度、投影步数和修正上界。投影推理时间包含方向头与解码，插件时间不含共享 base，不能与完整模型耗时混淆。

独立验证环境：Python 3.11、NumPy 2.2.5、PyTorch 2.7.0，CPU。

| 检查 | 结果与边界 |
|---|---|
| `python -m unittest discover -s trend_lab/tests -v` | 13 项通过；覆盖算子、数据切分、CSV、事件、模型梯度、投影、全方法/子集运行与重载。 |
| 默认 synthetic 配置加 `--smoke` | 5 场景 × 2 骨干全部通过，每个组合 13 方法、1 seed、1 epoch；不是完整实验。 |
| 独立输出检查 | 10 组 JSON 无 NaN，预测/概率数组有限且与 origins 对齐；两种投影的 alpha 都符合各自验证表与选择规则。 |
| CSV 模板端到端 | 使用临时生成的 400 行数值 fixture；无需 length/seed，smoke 限读前缀，6 种模板方法及 checkpoint 重载通过。 |

上述本地检查结束时尚未验证：CUDA 执行与性能、大规模数据内存占用、真实数据准确率、强骨干适配和完整文献基线。当前 runner 将窗口数据放入内存/所选设备，大数据应先估算容量，不能视为流式训练框架。概率校准也不提供分布漂移下的保证。正式结论必须按上面的 H1/H2/H3 判据另行实验，不以通过测试代替研究证据。

## 9. 服务器首轮 ETTh 验证

2026-09-27，用户授权服务器 GPU 实验后，GPT-6 Luna（max）执行、主 agent 独立审查：完成 ETTh1 和 ETTh2 各 3 seeds、11 方法的真实数据 pilot。先通过 CUDA CSV smoke，再依次完成两组固定参数实验；两组均通过 13 项现有测试、checkpoint 重载及独立 NumPy 结果核对。

完整配置、结果、分尺度诊断、证据边界和复跑入口见 [ETTh GPU 首轮报告](etth_gpu_pilot_2026-09-27.md)。协议为联合线性骨干、L=96、H=24、七变量、尺度 [1,2,4,8]、原型四段切分，不是标准 ETTh benchmark 配置。

预先固定的平均路径方向 accuracy：ETTh1 base 55.89% → 多尺度 55.73%；ETTh2 base 45.94% → 多尺度 46.44%，但单尺度为 46.78%。方向分类头分别达到 58.33% 和 52.05%，尚未体现为稳定的多尺度轨迹增益。当前 H1 有可学性信号，H2 主张未通过本轮检验；H3 分支未运行。两组平均 MSE 和原始一阶 sign accuracy 有改善，详见报告，不能据此替换已固定的主指标。

CUDA 和这两组小规模真实数据管线现已验证；强骨干、其他数据域、大规模内存、完整文献基线及充分公平调参仍未验证。优先在训练/验证段诊断尺度冲突和死区附近校正，而非直接扩大实验规模。

## 10. 第二轮：按用户联合目标筛选

2026-09-27，已按[追加协议](etth_idea_screening_protocol_2026-09-27.md)完成 ETTh1/ETTh2 × joint linear/MLP/DLinear × 3 seeds，每设置 13 方法，包含 event/nohazard。GPU 矩阵使用两张 A100，14 项代码检查、DLinear 参考一致性、CUDA smoke、6 格 checkpoint 重载及独立保存结果审计通过。旧 linear 的预测与首轮逐元素相同，只计作复验。

用户要求的测试联合目标是每种方法相对同一 base 的方向 accuracy 严格提高且标准化 MSE 不升（容忍 1e-8 数值误差）；原始 sign 与带固定阈值的三分类分别判定。训练与原验证选参规则未变，验证允许的 1% MSE 余量不计作用户目标中的“保持”。这是追加的探索性评估，不回溯冒充最初主指标。

原始方向/MSE 均值联合达标：多尺度投影 4/6 设置、单尺度 3/6、差分 MSE 4/6、方向监督残差 2/6、普通残差 1/6、事件模块 1/6。多尺度/单尺度在两个 DLinear 设置都失败；joint linear 的 ETTh2 上，固定 flat 控制又优于学习方向概率的校正。事件模块在六个设置中均未优于 nohazard 的最终路径转折 F1 与邻域错误率。

因此不把当前 H2/H3 写为已验证创新：D 限于一次有明确停止条件的小规模归因复验，A 当前实现暂停，C/E 留作组件和诊断，B/F 仍是未测试候选。各设置/种子的证据、强弱骨干差异、控制归因与完整取舍已写入[创新点文档第 7 节](trend_accuracy_literature_and_ideas_2026-09-26.md#7-按方向准确率与-mse-联合目标重新筛选)，详表见[第二轮逐方法结果](analysis/etth_matrix_final/evidence.md)。
