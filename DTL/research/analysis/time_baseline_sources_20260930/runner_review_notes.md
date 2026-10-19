# 2026-09-30：独立runner审查与数值修复记录

审查范围是`trajectory_run.py`、冻结v2协议/loader、保存的选择与模型输出。以下检查由另一个代理实施，与主runner的评分/选择实现分开；不等同外部审稿或独立重做全部训练。

## 已定位与处理的问题

1. 初始runner的callable预测和`del`语法由root修复；审查时尚未正式训练。
2. shared基线初稿只把现有模型移到CPU，并没有实际disk reload。已改为从保存checkpoint加载到新实例，并检查OCE/CaReTS的val和test输出。
3. shared测试曾早于case选择锁。已改为shared先训练/validation，case选择锁保存后才生成shared test路径及分数。
4. int lambda端点会产生validation `posterior_1`、test `posterior_1.0`而查找失败。已在生成key之前转float；原配置本身的端点已经是float，技术smoke未受这一潜在错误影响。冻结v2包含此修复。
5. 正式训练开始后发现CaReTS的历史末值锚在float32中加signed deviation，会丢掉小偏差；评价使用原float64观测锚时，又可能把零偏差第一步误记up/down。这是实际数值问题，不能用改true标签或换评估锚掩盖。新增**post-protocol推理修复**从不变的learned branch直接得到signed32，再用原观测anchor64相加；原checkpoint/主方法锁/输出完整保留。修复公式、选择规则和读盘核对见`repair_shared_anchor64.py`、`anchor64_repair/repair_protocol.json`。不从已经加锚后的z32−x32反推signed。

第5项修复不保证方向准确率提高：NEM三seed的CaReTS3 raw变化分别−0.5942/−0.9190/−1.0329 pp，原舍入产生的假方向有时碰巧命中真实方向。数值语义正确与指标较高须分开。NEM六case验证选定的修复竞争基准没有改变。完整修复及最终审计以全部18case工件为准。

修复竞争池的v1把旧CaReTS和修复版同时放入，形成15条候选，且可能再次选中已确认有数值问题的旧路径。此规则已纠正：canonical `anchor64_repair_v2`用两个anchor64版**替换**旧CaReTS两个位置，其他11条保持不变，候选总数仍13。先根据保存validation重算v2 lock，再读取/评分固定heldtest路径，没有新神经前向或重新训练。v1原工件完整留档供历史审查，不进入正式修复比较；此规则纠正有公式语义依据，不以test指标选择。

## 检查通过的机制

训练回调同步执行，未发现循环闭包把一个方法训练成另一个方法的问题。CosDir anchor固定eval且冻结；方向头训练按CE，所有回归组件早停按point MSE。MSE/CosDir/TDAlign同初始化与shuffle；普通residual和direction residual同参数/初始化。独立mock拟合e2e检查了实际交给trainer的state一致性，以及锁→测试→保存/重载顺序。

NEM/Finland完整窗口，Carpark按训练前数据质量修订仅历史做past-only ffill；origin必须原值有效，未来y必须全部原值有效，原train nanstats不改。五段目标不越界，完整时间轴保留。有限训练相邻对中float32与float64方向**不一致数量为0**：NEM259,190、Carpark67,560、Finland26,303对。最终另检查三个域五个时间段全部窗口1,918,560个标签（含窗口重叠），相邻raw三类和CaReTS相对锚二类标签的float32/64不一致数量都为0。这不表示预测的锚加法没有舍入问题。

季节性基准`x[:H]`对应前一日同时间段，因为这三个任务L均为一天且H<L。温度只由单独calibration段决定；独立CPU重建原模型并重选温度。有限候选的各消融共用网格，主λ和旧soft α均只从validation选。未看到test标签进入选择。竞争selector只挑相对CosDir方向严格提高且MSE不增加的方法，因此是预设联合目标的竞争基准，不应泛称任何单项意义的最强模型；报告仍需列全部竞争前沿。

`mean_is_base`消融保留同一个由learned mu制定的候选范围，关闭的是**误差代价项里的均值信息**，不是从算法的所有部位删除mu信息。

## 独立证据与适用限度

- 最终live基线10项与runner集成4项联合复测通过，共14项，0.401秒；原初8项基线公式/因果/相位测试也已保存。
- live `CaReTSPhaseTransformer.forward_with_true_anchor(x_float64)`接口已将第5项修复固化供后续实验使用：仅encoder输入转为权重dtype，保留真实float64末值，再直接添加learned signed deviation；普通forward/loss及冻结v2不变。增加真实anchor跨float32舍入、精确零和微小正/负signed、两variant NumPy公式测试后，live基线10项测试通过。
- 技术smoke v1独立评分/选择检查6,326项无失败；增强v2增加trainstats/窗口重建、CPU disk inference、校准温度及source/config hash，6,686项无失败。
- 两次技术smoke数值有小差异，不是bitwise重现；474个方向路径shape数组的raw符号不一致数为0，6case的choices/temperature一致。不把2epoch smoke重现外推为完整训练重现。
- `audit_runner_outputs.py`对完整结果独立重算指标、全部保存validation候选/选择、最终路径fallback、raw head指标、时序锁、数据/source/config hash；可选重建全部时间窗口、CPU读盘推理及温度选择。未独立训练，也没有第二套优化器复解整个DP矩阵。
- `audit_anchor64_repair.py`不导入修复脚本，独立重建signed/prob/deviation trace到真实anchor64、val/test指标、完整竞争库和原联合selector，并核对原hash/choices及修复锁时序。其最终报告应覆盖全部case。
- 全18case完整runner检查20,586项无失败，报告`matrix_independent_audit.json`；canonical v2修复检查4,260项无失败，报告`anchor64_repair_v2_independent_audit.json`。两脚本均以`--expected-cases 18`强制完整性。v2额外确认13条候选的替换规则、禁选已确认有问题的旧CaReTS，并检查trace是v1档案的逐文件原样副本。
- CPU/GPU原公式与anchor修复分开的完整诊断：9个shared seed、val/test两variant合计36组路径中，原CPU与GPU输出raw符号仅1个不一致（NEM seed2026 CaReTS4 validation后续步），test为0；固定CPU signed32的真实anchor修复改变37,811个符号，报告`anchor64_numerical_drift.json`。允许浮点容差不等于严格方向一致性；该旧公式漂移检查也不证明修复版CPU与GPU逐符号一致，不称bitwise重现。
- 整理数据的上游centered/global清洗及未知flat来源仍是数据限制；本地因果loader不能撤销上游已发生的未来清洗。

正式矩阵被冻结，不因结果不利扩λ/修改方法。数值修复被单独保存并加入最终公平比较，不声称这是此前预设的方法创新。
