# 方向轨迹方案验证：固定首轮协议

日期2026-09-30。用户授权验证主文档第12节。此协议在任何真实数据模型训练/评分前写入，阶段一是机制与有限规模排除实验，不是完整会议benchmark。历史工件不覆盖。

## 1. 数据与首轮范围

固定TIME导出数据版本：ProcessedCSV `683bb21b470f5a26586e1ab2295f0a2a1529ef0f`。第一批NEM全部10列、Finland单列、Carpark按源列名/原次序预选8列，选择发生在训练前。公开CSV经镜像传输并与锁定LFS SHA256核对。PM25/Oil已获取作暂不评分的后续任务。

原文件NEM是单宽表、Carpark也是单宽表而非多个item：按列名排序无法识别numeric ID，因此按照原列顺序固定RHM,TGM2,PM28,TM43,KLM,SE50,JBM,KBM。初始获取阶段检查了全档案的质量和值域，未查看模型结果；这不是声称所有标签均保持盲态的临床式预注册。

重要限制：官方预处理源码使用centered rolling IQR、全序列IQR，并有ffill/bfill；导出CSV无修改mask。本轮评价的是**整理后数据上的方法比较**，不能推出原生实时观测/真实plateau上的效果。后续确认必须用可核验原始数据及因果质量处理。NEM南澳operational demand存在负值，所有NEM通道不施加非负范围；计数域只用非负范围，不从test最大值创造容量上界。

| 数据 | L/H；训练/评价stride | 训练结束 | 早停结束 | 温度校准结束 | 方法选择结束 | 测试结束 |
|---|---|---|---|---|---|---|
| NEM 5min | 288/24；12/6步 | 2025-04-01 | 04-15 | 05-01 | 05-15 | 06-01 |
| Carpark 15min | 96/16；4/2步 | 2025-04-01 | 04-15 | 05-01 | 05-15 | 06-01 |
| Finland 15min | 96/16；4/2步 | 2024-10-01 | 10-15 | 11-01 | 12-01 | 2025-01-01 |

日期按导出轴解释；naive源不自行推断时区。训练从源首日开始，五段目标区间互不交叉，历史可跨先前分界。保留完整轴和原始NaN，不插值未来标签。NEM/Finland要求整个L+H有效；Carpark按下述训练前修订仅历史输入做因果ffill。各方法使用同一窗口。所有统计量仅训练段。torch训练为float32，原始归一化/评价及网格为float64，避免native锚点误差制造方向。

两骨干DLinear和原参考iTransformer，seeds2024/2025/2026，共18case。每case训练MSE/CosDir(.5)/官方TDAlign共用初始state及shuffle。**插件固定以同骨干CosDir输出为b**，主动检验已具方向监督的竞争基准，而不是从test选择最弱base。

## 2. 方法与训练预算

Adam .001、batch64（相位Transformer因显存使用32）、最多100epoch、patience12；所有回归checkpoint按独立早停段MSE，方向头按CE。TF32关闭，CPU/GPU重载容差仍2e-5。仅使用空闲且有显存余量的GPU；不停止既有任务。

方向头hidden32，raw down/zero/up、epsilon0；温度候选[.5,1,2,4]仅在独立校准段选择。mu为MSE残差组件，参数数匹配方向头；另训练参数约为p+mu总和的普通残差和相同初始化的方向监督残差（稳定scale、固定权重.1），排除额外容量与监督解释。

共享补充基线每数据/seed训练一次：OCE-TS固定训练支持适配、相位保留CaReTS3/4 Transformer适配。OCE保留官方DLinear概率输出、truncated Gaussian targets、ordinal cumulative BCE和概率均值；**不复制官方测试使用未来batch_y extrema缩放的代码**，使用训练段逐通道固定affine。bins45、sigma.015按官方入口；不是原论文完整复现。CaReTS保留相对历史末值二分类/幅度公式，编码保留时间位置与flatten，dmodel64、2layers、4heads；使用相同输入、预算、MSE选择，不声称最优版本完整复现。

## 3. 候选、选择与控制

NEM共同连续网格：标准化step=.1、padding=.25、包含当前历史锚点。计数域native q=1,r=0，用训练mean/std转换网格原点和步长；只有训练值通过整数检查才使用。每sample/channel由当前xlast、b、mu确定覆盖范围，状态上限257，超限按可见跨度确定性稀疏化，保持native格点；报告稀疏/边界/anchor可达性。所有解码控制共用**同一个**候选集合，额外保留完整实数b路径，逐实例按同一估计目标比较，不逐步拼接b。

lambda固定[0,.001,.01,.1,1]，越大方向奖励越强。完整后验DP主线、同库top1-gap DP、mu=b DP、训练方向先验DP、同库贪心各给相同5项选择预算；mu-only lambda0、原mu输出、普通rounding/范围、末值/季节性单列。旧软投影alpha[0,.1,1,10]同一个head；margin=.1训练历史绝对差中位数、RMS cap .5、tol1e-5/max600。

选择仅validation：相对**同一范围处理后的CosDir b**方向严格增加，MSE<=b+1e-8；否则完整b回退，避免把无修改算收益。竞争基准从所有训练/规则对照中按相同validation规则选，不看test。保存完整validation候选及选择，锁文件写入后才生成该case的test路径/评分。训练数据监督插件不称TIME官方zero-shot。

主问题是联合改善，因此首轮预算tau=0采用**验证段聚合经验条件**，不称逐起报条件风险保证。保存未选候选以解释方向—MSE取舍，test上不改lambda/网格/模型。额外范围或quantization仅作为预先声明的对照，不在失败后扩大库。

## 4. 判据和证据

机制先完成：短H独立穷举对DP、同p异mu风险、无信号、Bayes-mean零预算不可能性、真实zero与量化artifact、非凸Pareto不受支持点。标准技术/恒等式不称理论首创。

真实数据报告每case相对b及验证选定竞争基准的方向pp/MSE%；逐类precision/recall、macroF1、预测change rate、head/path差、按需求/价格/通道/起报相位及目标幅度分项。预先用1天origin blocks、2000配对bootstrap（先平均已拟合seeds），3天block作依赖敏感性；时间短和同源通道限制必须写清。

继续条件沿主文档：至少2/3任务有相对竞争基准的方向增量且MSE不增加，至少两个骨干/各3seed可查；方向组件不能被mu-only、top1或范围/rounding/容量残差复制。否则停止“该完整后验DP具有额外价值”的主张，完整报告失败，不在同test扩lambda。满足首轮门槛才锁算法开展第二批PM25/Oil/更长H/原始数据确认；否则仅做已声明机制定位。短期curated-data pilot通过也不足以认定完整CCF A论文已验证。

## 5. 训练前质量修订（尚未训练或评分）

独立loader审计发现Carpark共享缺测约每日出现：在L96+H16全部有效要求下训练窗口为零，因此该原要求不可执行。Carpark改为historical covariates只做past-only forward fill，不bfill、不压缩轴、不填future y；origin处全部8个原观测必须有效，未来H全部原标签必须有效，填充历史起点缺测也不保留。训练均值/标准差仍取原始训练段nanstats，各模型共同输入/窗口，报告每段history填充比例。NEM/Finland保持complete policy。此修订由数据质量检查触发，发生在第一次模型训练前，不由模型结果触发。基于整理数据和有限历史填充的结论仍需原始因果数据复核。

补充基准锁：共享OCE/CaReTS先生成validation预测及保存checkpoint；其test预测与评分仅在第一个case validation_lock写入后生成。竞争基准采用明确的联合改善选择规则，不称所有基准中的无条件最强；报告全部基准和验证Pareto集合以暴露MSE—方向取舍。

## 6. 训练后独立审计触发的推理修复

冻结v2矩阵开始后，独立审计发现CaReTS的float32历史锚点相加会把公式中零signed deviation，转换成相对真实float64锚点的微小涨跌。追加两个修复基线：从原checkpoint重建相同signed32分支，不从已相加结果相减反推，改为xlast64+signed32.double；无需重新拟合。保存完整val/test和signed分支trace，用原相同validation联合改善规则在原库用两修复基线替换旧CaReTS后选择competitor_corrected（仍13项，旧bug版仅历史展示），先持久化该repair validation锁，再生成修复test输出。主方法lambda/model/网格及原选择全部保留。最终以修复比较判断结论，同时报告原比较；这是因代码公式及精度证据而实施的post-protocol技术修复，不是从test选超参数。完全公开修改时间、旧结果和效果，不能将其称严格盲态预注册。

## 7. 审稿所需的精度负对照（事后诊断，不扩主选择库）

独立源码审稿指出eps=0残差方向surrogate的flat概率最多1/3，不能称为与自由三类头统计等强的zero基线。因此补固定、无训练、无选择的native域precision controls：普通μ/两容量残差/MSE/Cos/TD的q1,r0舍入及同非负范围，另有两容量残差向primary同一库的λ0最近值投影；不新增main λ或CV选择，不从test筛选规则。该组只用于诊断或证伪“超越容量和量化”主张，不能用事后胜出升级为独立确认；普通Cos rounding本来已经在主比较库。NEM不采用未经核实的物理q1新控制。规则在计算前另存precision_controls_posthoc_protocol_2026-09-30.md，保留所有输出及原锁SHA。
