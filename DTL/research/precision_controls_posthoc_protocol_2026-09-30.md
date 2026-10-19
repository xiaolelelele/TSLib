# TIME 方向轨迹方案：测量精度负对照（事后诊断）

2026-09-30。本规则在首批主实验结果已经产生之后追加，由独立审查指出的公式限制触发：方向监督残差使用 eps=0 的 logits `[-d,0,d]/s`，flat 概率 `1/(1+2cosh(d/s)) <= 1/3`，因此不能把它当成统计表达能力等价的任意三分类方向模型。同时必须排除“更大容量加普通整数精度”解释复杂 posterior+mean 组合的收益。

这些是 **post-hoc 证伪控制**，不是新的确认集、候选选择库或调参。固定使用已冻结的 train mean/std、模型输出和 val/test 窗口；不训练、不改变主方法、竞争基准或 validation lock，不凭结果增删规则。NEM 的物理精度没有核实，故不使用 q=1。

## 1. 原生整数普通舍入

对 Carpark 和 Finland 全部两骨干、三 seed，共 12 个 case；每个 val/test 既有 `mean_raw`、`residual_capacity`、`residual_direction_capacity`、`mse_base`、`cosdir`、`tdalign` 输出应用同一规则：

1. 逆变换到原单位：`a = z * train_std + train_mean`。
2. 原生网格 `q=1, r=0` 上普通 `numpy.rint(a)`，采用 ties-to-even；不根据数据或性能选择舍入方式。
3. 同一非负范围：`max(rounded, 0)`，不使用未知容量上界。
4. 使用同一 train mean/std 回到 float64 标准化坐标。

命名为 `{method}_native_round`。保留 val/test 的原 x/y/origins，报告 MSE、MAE、raw accuracy、macro-F1、混淆矩阵、各类 precision/recall 和预测 change/zero rate。不进行 validation 选择；这些结果不回写主竞争者。

## 2. 原支持库上的残差点估计投影

额外固定两个控制：`residual_capacity_shared_grid`、`residual_direction_capacity_shared_grid`。使用主方法的 x、CosDir b 和原 mean μ 构造 **同一 common grid**，沿用原规则、同一 q/r、非负范围、padding=.25、max_states=257 和可见范围稀疏化。将对应容量残差的点输出作为误差中心，方向奖励固定为零，使用主 decoder 的 λ=0，不进行 λ 选择。保留与主候选库相同的完整 b 备选，不把 b 拼成逐步备选。

λ=0 时 p 不参与目标值；保存 p 只是复用相同冻结接口。控制说明点估计在现有支持库上的投影能实现多少收益，不能把它称为新的学习算法。支持库由原 μ/b 建立，残差预测若超出该库范围，其损失可能受支持限制；因此同时保留第1节无额外库裁剪的原生普通舍入。

## 3. 工件和结论边界

输出到 `DTL/trend_lab_runs/time_trajectory_20260930/precision_controls/{domain}/{case}/predictions.npz` 和 `result.json`。保存来源 predictions/validation_lock SHA256 和固定变换元数据。案例只有在原 `result.json`、lock 和完整预测已经保存后才运行。

若普通舍入或同库点估计达到/超过复杂主方法，这削弱“完整方向后验/全局轨迹决策超越容量和量化”的主张；不能因此回选其它 λ 或改测试切分。若复杂方法更好，这仍只是事后对照证据，需要新数据上的预先固定确认。原 eps=0 残差方向 surrogate 的 flat 表达限制保持公开，不能用它较差的结果证明所有方向监督残差不如主方法。
