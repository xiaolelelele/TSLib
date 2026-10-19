# 2026-09-30：强方向基线的公式来源与适配边界

本目录保存公开主来源、OCE-TS固定revision及逐文件hash，供本轮监督实验复算；不构成原论文完整复现。正式源为[OCE-TS AAAI 2026论文](https://ojs.aaai.org/index.php/AAAI/article/view/39831/43792)、[其arXiv完整HTML](https://arxiv.org/html/2511.10200v1)、[官方代码](https://github.com/Shi-hm/OCE-TS/tree/a574b99e2e2673f7647b06ccda09de7d4b69f0e5)、[CaReTS原论文](https://arxiv.org/html/2511.09789v1)。AAAI PDF通过web工具阅读；直接下载连接关闭，arXiv PDF传输停滞后仅中断该下载任务。已成功保存两篇完整HTML和文本提取。

## OCE-TS：保留公式，固定因果支持

实现`OCETimeSeries`保留论文式2的K−1个累计概率二分类交叉熵之和；式8–11的截断高斯soft target；式18的bin中心概率加权均值。骨干按官方`models/DLinear.py`，先做端点复制的移动均值分解，再分别以共享通道的L→H*K线性层生成趋势/残差logits，softmax后输出[B,H,C,K]。这不同于正文式14较简略的scalar→K投影图示，选择遵循可见官方实现。默认K=45、sigma=.015来自官方`run_longExp.py`，不是事后搜索。

官方固定commit的`exp/exp_main.py`存在需要规避的评价问题：训练/validation将未来标签窗口以自己的min/max映射到[-1,1]，历史输入则以另一个历史窗口min/max映射；test第236–238行用`batch_y`未来极值覆盖`min_val/max_val`，第251行将预测均值按这些未来极值还原。这使可见官方test输出依赖未来标签。这里只判断该固定代码路径，不由此断言论文全部已发表结果均来自这一实现。

本轮采用明确的**OCE-TS因果固定支持适配**：训练时间段每通道最小/最大值，先在runner固定标准化坐标中存为持久buffer；历史与soft标签用同一固定仿射映射到[-1,1]，输出用同一buffer还原。未来极值、validation/test统计均不进入预测。常量训练通道支持扩成以该值为中心、宽度1的标准化区间。soft标签越界时仅将损失构造的高斯中心裁到固定边界，评价真值不改；输入不裁剪。预测天然限制在训练bin中心范围，因此漂移/尖峰下可能受限，必须披露。累计概率与Gaussian CDF差用float64防止饱和后log(0)，并非另加MSE或伪造ordinal损失。

## CaReTS3/4：保留任务，保留时间相位

`CaReTSPhaseTransformer`复用此前CaReTS的分类/幅度分支及不确定性加权损失：两层64维FC+ReLU；3使用论文式13的真类别幅度监督、式15的up/down概率加权非负偏差；4按式17–19将类别概率与历史编码特征拼接后回归有符号偏差。类别是**相对最后历史值**的二分类，不改写成相邻三类方向；论文未明确的equal情况固定为up。

论文§2.2明确提出log-variance限制在[-10,10]及附加soft penalty；本实现沿用前一版本的loss内clamp和固定`0.001 * sum(log_variance²)`系数。该系数不是从失效匿名仓库确认的原始超参数，属于需披露的实现选择，未按本轮validation/test搜索。

编码器采用附录A.2/A.4的64维历史投影、位置编码、2层4头Transformer，FF维度128、ReLU、dropout=.1；然后按时间位置flatten全部历史隐状态并线性投影到64维，不做抹掉相位的平均池化。flatten readout是明确适配选择，原匿名仓库已失效，无法声称逐行复现。只输入历史变量，不新增未来calendar/协变量。

本轮训练段固定标准化、连续时间留出、100 epoch/patience12且point-MSE早停，均不同于原论文min–max、10fold、600epoch/patience50设置。应同时报告实际epoch及触顶比例；不能将表现较弱的有界预算适配解释为原框架无效。

实际审计发现float32末值加法会吞掉微小signed deviation；在评价使用原float64观测锚时，零deviation还会产生假第一步方向。因此后续live推理应调用`forward_with_true_anchor(x_float64)`，在权重dtype中计算原分支signed，再转float64与原历史末值相加，普通训练forward/loss不变。本轮冻结结果不回写，单独post-protocol修复保留原工件。canonical `anchor64_repair_v2`将竞争池原两个CaReTS位置替换为修复版，保持其余11条及13条总预算；v1将旧版并列在扩大的15条池中，现仅作历史档案。

## 独立检查

8项unittest通过：官方OCE loss源码数值核对；SciPy Gaussian CDF独立soft mass核对；NumPy分解/线性/softmax/中心重建；同一仿射变换协变；未来标签扰动不改support/输出；CaReTS融合、last-value二类及真类别幅度掩码；保持均值/末值但置换中间历史的相位敏感性；所有模型state reload及逐样本/批次一致性。另检查梯度有限、关键分支获得梯度和非法范围拒绝。测试约0.114秒，结果不是模型效用证明。

新增真实anchor跨float32舍入、零signed、被旧加法吞掉的微小正负signed及两variant独立NumPy公式检查后，live基线共10项unittest通过，约0.135秒；冻结v2没有这些后加接口/测试。

普通`residual_direction`使用同一个`ResidualAdapter`（与普通residual相同参数），可复用`losses.fit_slope_scales`和`direction_surrogate_loss`：训练拟合正的斜率scale，epsilon=0的raw边界，MSE加固定weight的有界方向CE。该控制是已修复的普通监督目标，不列为本轮新方法贡献。

该CE构造在epsilon=0时logits为`[-d/scale,0,d/scale]`，flat类概率最大只有1/3，不能表达高置信exact-zero类别。这是原控制的实质限制；native count域的诊断需另看固定rounding负对照，未来确认性实验需预定flat-consistent或hurdle目标，不能把当前控制不足解释为新算法超越所有方向监督方法。
