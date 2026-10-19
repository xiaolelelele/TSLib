# 第三轮：方向校正归因与融合验证协议

2026-09-27。用户授权继续验证，并考虑是否需要多种方案融合。本协议在本轮新数据结果产生前固定；不根据测试表现增加融合成员、权重或门控规则。

## 数据与训练

- 探索性复核：复用 ETTh1/ETTh2 的 joint linear 与 DLinear，各 seeds [0,1,2]，来自 `ett_backbone_matrix_20260927T145732Z`。这些测试段已看过，不能作为新独立验证。
- 额外数据验证：本地 `weather/weather.csv` 与 `exchange_rate/exchange_rate.csv`，全部数值变量，joint linear 与 DLinear，各 seeds [0,1,2]。本轮方案固定前只检查文件模式、长度、时间格式与训练段质量，不读取新测试指标。
- 延续 L=96/H=24、尺度 [1,2,4,8]、时间四段 60/15/10/15、train stride=4/eval stride=24、训练 40 epochs/patience=8、batch=64、lr=0.001、hidden=32、训练阈值分位数0.2。尺度单位是数据行；不同频率不解释成相同小时数。
- 冻结与原型相同的 base、difference、direction_head、residual、residual_direction 训练结构与预算。保留原11项作为工件兼容对照；不再运行已暂停的 event/nohazard。温度依旧只用 calibration NLL 选择。
- Exchange Rate 日期若不符合现有加载器，只生成日期转换为 ISO 的派生 CSV；数值字符串、行序和列序不变，验证时间严格递增并保存原/派生 SHA256。原始 CSV 不覆盖。
- 数据预检修订（尚未开始新训练/查看新测试指标）：Weather 有一条整行完全相同的重复观测（原0-based行19044，2020-05-12 06:00:00，与上一行同时间且所有数值字符串相同）。在派生 CSV 中只删除这种完全重复的观测，保留首次出现；相同时间但数值不同则停止，不任意择一。保留其余数值字符串与行序，记录删除行号与原/派生哈希。另有一次100分钟缺口（原row21514），不插值，窗口尺度解释为观测步而非处处严格等距的物理时间。这两处位于训练段；本轮不因测试效果调整清理规则。
- 使用当前空闲 GPU；正式小数据训练前做 CUDA smoke。保存源码、配置、数据哈希、模型、预测和重载检查。新增验证仍是本项目小样本原型协议，非标准 benchmark 或完整外部方法比较。

## 预先固定的原子方案

数值路径包括 `base`、`difference`、`residual`、`residual_direction`。校正均从同一个 base 出发：

1. `learned_multi`：原多尺度学习方向概率。
2. `learned_single`：同一概率头，仅 k=1 约束。
3. `prior`：只从训练标签统计每尺度/步长/变量类别频率，与当前输入无关。
4. `flat`：固定 flat 概率为1。

每种校正独立使用 alpha=[0,0.1,0.3,1,3,10]，40步，max_rms=0.5。**本轮只在 validation 上选择原始 sign accuracy 严格提高且标准化 MSE <= base+1e-8 的候选，按 accuracy 降序、MSE 升序、alpha 升序排序；无候选就 alpha=0。** 三分类各尺度、均值、macro-F1 另外完整报告，不用另一指标事后替代原始 sign 判据。当前 runner 的旧三分类 alpha 可保留为历史对照，不作为本轮新选择。

`learned_selected` 只在 validation 上从 learned_single/learned_multi 中选择（accuracy 高、MSE 低优先；完全相同则优先 single）。`standalone_selected` 同样从 base、difference、residual、residual_direction、learned_single、learned_multi、prior、flat 中选择，必须满足相对 base 的严格联合验证条件，无候选就 base。

所有 alpha/权重均按拟合 seed 分别在本数据的 validation 上选择。这评价的是每域、每次拟合后校准的流程，不声称一套固定权重无需校准即可跨域迁移；三套 seed 搜索的调参波动通过完整选择记录和逐 seed 测试结果公开。

## 有限融合集合

只融合最终数值轨迹，再从同一条融合轨迹重新计算方向和 MSE。对每对 `(a,b)`，用 `z=(1-w)*a+w*b`，w=[0,0.25,0.5,0.75,1]；先冻结上述成员自己的 validation 选择，再仅用 validation 选择融合 w，仍遵守相对 base 的严格联合条件。权重端点等于成员方案；不得把端点或两成员完全相同的情况计作真正融合收益。

- base + difference：简单预测集成对照。
- base + residual：收缩普通残差对照。
- base + residual_direction：收缩方向监督残差。
- learned_selected + flat：方向信息与收缩是否互补。
- learned_selected + prior：学习概率与总体先验是否互补。
- learned_selected + difference：方向校正与差分训练是否互补。
- 一个固定三方案均值 `(learned_selected+flat+difference)/3`：不搜索三维权重。

无满足验证联合条件的融合候选时回退 base。`fusion_selected` 在上述7个已选方案中用 validation 同规则选择，无候选回退 base。完整记录每个权重、每个方案的验证表及选择身份；不看测试再挑“最佳融合”。本轮不训练自适应门控、事件融合或按真实未来选择成员。

## 互补性与决策

- 在 validation 上对冻结成员记录逐窗口 raw-sign 正确率和 MSE 差值、方向错误互补比例与误差相关性。互补性是诊断，不能代替可实现的融合结果；不把按真实未来选择成员的 oracle 作为方法分数。
- 对每个数据×骨干×seed，在 test 分别评价候选相对 base、相对 `standalone_selected`、融合相对其两个成员的方向/MSE 联合变化。全量报告每种预先固定融合，不只报告正结果。
- 均值与逐seed次数分开；MSE +1% 不算保持，alpha=0/w端点不算新机制获益。分清“降低单个残差的幅度”“选择更好的单成员”“两个方案真正互补”。
- 只有在新增数据上反复超过验证选出的单方案、至少在较强 DLinear 上出现可重复收益，才建议继续融合。若改善只来自端点、弱 base 或一个偶然设置，优先采用简单单方案，不增加门控复杂度。
- 本轮先选每个成员的 alpha，再选融合权重；这限制了搜索范围，可能错过单独不达标但联合有效的 alpha 组合。因此阴性结果只用于降低当前有限融合实现的优先级，不宣称穷尽或否证所有联合优化方案。
- 配对时间块 bootstrap 使用8窗口、2000次，先对拟合seed取窗口差值均值再重采样；另报seed标准差。区间条件于当前数据和模型，多个权重/方法带来验证选择自由度，不能当正式多重检验已校正的发现。

## 交付

保存第三轮训练/推理/融合工件、独立复算、全量差值与选择记录。更新原 `trend_accuracy_literature_and_ideas_2026-09-26.md`：D 是否继续、是否需要融合、哪些具体组合值得进一步验证、哪些停止，以及新结果对第7节旧结论的修订。
