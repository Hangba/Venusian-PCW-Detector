# 分析与出图工作流

[workflow.ipynb](../workflow.ipynb) 是数据处理、训练、评价和搜索的交互入口。

[figures.ipynb](../figures.ipynb) 是科学图表入口。它调用项目的 Python 模块，按检测、年际与空间统计、密度、边界和观测示例分组出图。

## 运行 Notebook

在仓库根目录执行：

```bash
uv pip install -e '.[notebook]'
jupyter lab workflow.ipynb figures.ipynb
```

使用安装了项目的 Python 内核。在 `workflow.ipynb` 的参数单元选择数据来源、训练阶段、种子和设备，再开启所需步骤。默认只校验输入并重算参考分数的指标。

在 `figures.ipynb` 中先运行参数与统计准备单元，再运行所需的绘图单元。默认用冻结派生输入重算统计并绘制 10 张图。为 `MAG_DIR` 或 `VENUS_EXPRESS_RAW_DATA_DIR` 指定原始 MAG 目录后，Notebook 可再绘制 3 张原始观测示例图。每组图保存为 PNG 和 PDF，输出在 `build/figures/`。

`venusian-pcw replay` 可单独重算数值，结果写入 `build/summary.json`。

| 参考结果 | 数值 |
|---|---|
| 2013 年留出复审 | 245 个最终候选，排名前 50 项中确认 38 个 PCW |
| 全任务期发生率 | 阈值 0.70 下得到 1,150 个事件簇，平均 0.42468 簇/观测日 |
| 距离校正后的年际波能 | 796 个事件，与年均太阳黑子数的 Pearson 相关系数为 0.695 |
| 日侧中性氢密度 | 78 个事件，25,000 km 处为 6.55 cm⁻³，95% 区间为 4.99–8.72 cm⁻³ |
| 准径向行星际磁场样本 | 50 个时段中有 32 个识别出感应磁层边界 |

## 从原始 MAG 重算

```bash
export VENUS_EXPRESS_RAW_DATA_DIR=/path/to/VEX_1Hz
venusian-pcw catalogue --workers 4
venusian-pcw psd-audit --workers 4
venusian-pcw search holdout --workers 4
venusian-pcw search mission --workers 6
venusian-pcw exposure --workers 6
```

`catalogue` 从原始记录重新测量参考清单中的 946 个 PCW 事件，结果写入 `build/raw_catalogue/`。全任务期搜索生成独立的自动检出目录。搜索可用 `--day YYYY-MM-DD` 指定单日，用 `--resume` 复用同一次运行已完成的逐日粗分数。

完成两条搜索后，可在 Notebook 中将 `USE_NEW_SEARCH` 设为 `True` 来使用新搜索的结果。将 `RECOMPUTE_DENSITY_PSD` 设为 `True` 可重新测量密度反演所用的波谱。弓激波图使用从归档矢量图恢复的数值表，留出复审和准径向行星际磁场样本使用提供的人工标注。[输入定义](data.md)

## 模型训练与固定窗口评价

在 `workflow.ipynb` 中，设置 `STAGE` 和 `TRAIN_SEEDS`。从 BAM 开始时，先开启 `PREPARE_TRAINING_WINDOWS`，再开启 `RUN_TRAIN`。需要对本地训练结果做固定窗口评价时，设置 `EVAL_TRAINED_SEED`，并提前准备与 `TRAIN_INPUT` 对应的固定测试特征。参考检查点评价与本地检查点评价分别输出。

也可通过命令行批量运行：

```bash
venusian-pcw prepare-windows MStandard
venusian-pcw train MStandard --seed 2026 --from-raw

for stage in MSeed MExtend1 MExtend2 MStandard; do
  venusian-pcw prepare-windows "$stage" --role validation
done
venusian-pcw prepare-windows fixed_test
venusian-pcw evaluate-windows
```

MSeed、MExtend1、MExtend2 和 MStandard 支持种子 2026、2027、2028。`holdout2013` 使用种子 2026。若要与原始窗口特征做精确对照，可将 `VENUS_EXPRESS_TRAINING_INPUT_DIR` 指向[输入协议](../inputs/protocol.json)所列的冻结 NPZ 根目录，再运行 `venusian-pcw replay --rescore-windows`。

## 字体配置

在 Notebook 参数单元按回退顺序填写 `FONT_FAMILIES`，通过 `FONT_PATHS` 添加自有字体目录。两项留空时使用随附的 New Computer Modern 和可用系统字体。中文图表需要包含中文字形的字体，可按需要选择 Noto、思源或其他中文字体。

```python
FONT_FAMILIES = ["DejaVu Sans", "Noto Sans CJK SC"]
FONT_PATHS = ["/path/to/fonts"]
```

也可用环境变量 `VENUSIAN_PCW_FONT_PATH` 添加字体目录，多个目录以系统路径分隔符分开。重新运行统计准备单元和需要的绘图单元后，设置会应用到新图。
