# TSLib 工作区代码

本仓库同步 TSLib、TFB 和 DTL 的 Python 源码、实验脚本、配置、测试及相关说明。数据集、训练权重、预测数组、运行日志、结果表格、论文文件和演示文稿生成环境不在本次同步范围内。

## 目录

| 目录 | 内容 |
| --- | --- |
| `Time-Series-Library/` | TSLib 主程序、模型定义、趋势分类头、损失函数及实验脚本 |
| `Namdm/`、`lxh/` | GRU 等预测模型与超参数优化代码 |
| `TFB/` | TFB 基准框架及本地对 DUET、PDF、趋势损失和指标的改动 |
| `DTL/trend_lab/` | 趋势方法原型、数据处理、评估、配置与测试 |
| `DTL/research/` | 数据获取、实验审计、机制分析脚本及研究协议 |
| `DTL/supplementary materials/code/` | 补充材料中的代码版本 |
| `DTL/trend_lab*_runs/` | 历史实验保留的源码和配置快照；运行产物未同步 |

`TSLib/Time-Series-Library`、`TSLib/Namdm` 和 `TSLib/lxh` 是指向上述现有目录的相对符号链接，保留 DTL 中 `../TSLib/...` 的路径兼容性。使用 Windows 时请在支持符号链接的环境（如 WSL）中克隆和运行。

## 运行

各子项目有独立的依赖要求，建议分别创建 Python 环境。请阅读对应的说明：

- [TSLib 使用说明](Time-Series-Library/README.md)及[依赖](Time-Series-Library/requirements.txt)
- [TFB 中文说明](TFB/README_CN.md)及[依赖](TFB/requirements.txt)
- [Trend Lab 说明](DTL/trend_lab/README.md)及[依赖](DTL/trend_lab/requirements.txt)

Trend Lab 的合成数据检查不需要下载真实数据。在安装其依赖后，从仓库根目录执行：

```bash
cd DTL
python -m trend_lab.run --config trend_lab/configs/repaired_synthetic.json --smoke --output /tmp/trend_v2_smoke
```

真实数据实验需要自行准备数据并配置路径。部分历史脚本还保留原机器的 Python、数据和 GPU 设置，请按各脚本提供的环境变量或命令行参数调整。审计历史实验结果的脚本需要对应的本地运行产物。

## 同步边界

`.gitignore` 排除数据、模型权重、缓存和运行产物，保留 `models/*.py` 等模型实现。历史运行目录中已选择保留的源码和配置仍由 Git 跟踪，新生成的运行文件会被忽略。

此前已在远程版本中跟踪的数据、权重和日志从当前目录树移除，本地原件保留；既有 Git 历史保留。

本次同步日期：2026-10-19。
