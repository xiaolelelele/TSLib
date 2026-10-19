# 现实 validation flow 证书：固定采样与算法规则

2026-10-08。本规则先于本程序的真实输入评分运行写入；仅使用旧实验的 validation 可见预测与历史锚点，不加载未来真值，不加载 test 数组，不训练，不选择任何论文方法参数。

- 数据域固定为 Carpark、Finland，各 DLinear/iTransformer × seeds 2024/2025/2026，共 12 例。
- 每例固定 validation origin 数组位置 `0` 和 `floor(N/2)`；每个通道各检查一条完整 H16 曲线。预期合计 108 条，不按观测矛盾/误差采样。
- p 从已保存 validation float32 概率转为 float64，逐时刻三类重新归一。当前 x、预测 μ、CosDir b 仅按训练 mean/std 逆标准化；x 核验到原生非负整数的误差≤1e−9，再 `rint`。μ 和 b 不作为整数真值。
- original pair 不修改。simple pair 只在第一步真实 x=0 时屏蔽 down，并归一 surviving zero/up；按该 masked p 的无界一步精确区间投影首步 μ，且仅当原单位违反>1e−9时修改。其他 p/μ 不变。
- 每个 pair 先检查原生解析必要条件：第一步边界与上下界、各时刻 μ_t≥p_up_t、μ_(t−1)≥p_down_t；只有相应概率恰为0时检查无 down/no up 的加强矩条件。真实 x0=0 时另检查 p_down_2≤p_up_1。条件数值比较容差为原单位1e−9。违反即记 `NECESSARY_INCOMPATIBLE`，不运行 LP。
- 未发现必要违规时，在**原生无上界支持中的有限子集**求可行见证。初始候选包含 0、1、x、合法的 x−1、x+1；所有 μ 与 b 的非负 floor/ceil。跨度 `span=max(1, max({x,mu,b})-min({x,mu,b}))` 仅由可见值确定。按 tail 因子 1、2、4、8 分批扩大，添加 `ceil(max_visible+factor*span)` 与其相邻合法整数，保留之前状态；首个通过残差核验的可行解即停。没有把候选最大值当物理 U，也未使用训练/未来最大值。
- 使用独立 `theory_probe.flow_lp` 的 HM² 稀疏 equality flow 模型，moment 系数按该通道已知训练 std 缩放；SciPy HiGHS primal/dual feasibility tolerance1e−9。单 LP wall `time_limit≤5s`，全局总预算300s；剩余预算不足时记录 `UNKNOWN_BUDGET`。阶段性扩容失败、求解超时及数值残差未通过均记 UNKNOWN，绝不记为原生不兼容。
- 每个可行解复算非负、源/层间流连续性、三类方向概率、原单位均值，以及缩放 equality residual。接受阈值：概率/flow/归一≤1e−8、原单位均值≤max(1,train_std)*1e−8。保存稀疏 e 与完整 v/G 的 compact npz 证书、p/μ/x、源文件 SHA256、支持/耗时/状态/残差。
- `CERTIFIED_FEASIBLE_NUMERICAL` 表示经上述浮点容差核验的合法原生有限支持流见证，不等于有理数或区间算术的精确形式化证书。有限支持成功不依赖任何物理容量；有限支持失败不能否认无界支持的可行性。
- 汇总 original/simple 的三类结果及 paired transition；样本为固定验证子样本，不能外推全量矛盾率或当成新测试收益。报告所有阶段，不根据结果修改这份规则。
