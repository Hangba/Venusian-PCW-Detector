# 金星快车 PCW 研究复现指南

这套工作流对应[中文论文](../paper/main.typ)。参考模型包含四个训练阶段，每阶段三个随机种子。2013 年留出评价使用单独的检查点。

## 使用冻结输入复算

出图需要 Poppler 的 `pdftocairo`，论文编译需要 Typst 0.14 或更高版本。渲染中文图表或正文时，请使用包含中文字形的字体。

```bash
venusian-pcw verify
venusian-pcw replay
venusian-pcw paper --generated
```

`replay` 用冻结输入重算表格并重绘 10 张图，其余 3 张依赖原始 MAG 的图采用归档版本。接入 `VENUS_EXPRESS_RAW_DATA_DIR` 后，运行 `replay --raw-figures` 可重绘全部 13 张图。直接运行 `paper` 会使用归档图件编译。

| 参考结果 | 数值 |
|---|---|
| 2013 年留出复审 | 245 个最终候选，排名前 50 项中确认 38 个 PCW |
| 全任务期发生率 | 阈值 0.70 下得到 1,150 个事件簇，平均 0.42468 簇/观测日 |
| 距离校正后的年际波能 | 796 个事件，与年均太阳黑子数的 Pearson 相关系数为 0.695 |
| 日侧中性氢密度 | 78 个事件，25,000 km 处为 6.55 cm⁻³，95% 区间为 4.99–8.72 cm⁻³ |
| 准径向行星际磁场样本 | 50 个时段中有 32 个识别出感应磁层边界 |

数值结果写入 `build/summary.json`，编译后的论文写入 `build/paper/venus-pickup-ion-paper.pdf`。

## 从原始 MAG 重算

```bash
export VENUS_EXPRESS_RAW_DATA_DIR=/path/to/VEX_1Hz
venusian-pcw catalogue --workers 4
venusian-pcw psd-audit --workers 4
venusian-pcw search holdout --workers 4
venusian-pcw search mission --workers 6
venusian-pcw exposure --workers 6
venusian-pcw replay --raw-search --raw-density
venusian-pcw paper --generated
```

`catalogue` 从原始记录重新测量参考清单中的 946 个 PCW 事件，结果写入 `build/raw_catalogue/`。全任务期搜索生成独立的自动检出目录。搜索支持用 `--day YYYY-MM-DD` 指定单日，也支持用 `--resume` 复用同一次运行已完成的逐日粗分数。

弓激波几何基准使用从原矢量图恢复的穿越点及已报告的圆锥曲线参数。留出复审和准径向行星际磁场样本使用提供的人工标注。[输入定义](data.md)

## 模型训练与固定窗口评价

```bash
venusian-pcw prepare-windows MStandard
venusian-pcw train MStandard --seed 2026 --from-raw

for stage in MSeed MExtend1 MExtend2 MStandard; do
  venusian-pcw prepare-windows "$stage" --role validation
done
venusian-pcw prepare-windows fixed_test
venusian-pcw evaluate-windows
```

MSeed、MExtend1、MExtend2 和 MStandard 支持种子 2026、2027、2028。`holdout2013` 使用种子 2026。若要与原始窗口特征做精确对照，可将 `VENUS_EXPRESS_TRAINING_INPUT_DIR` 指向[输入协议](../inputs/protocol.json)所列的冻结 NPZ 根目录，再运行 `replay --rescore-windows`。

## 字体配置

`replay` 和 `paper` 都支持重复传入 `--font-family` 和 `--font-path`。可按回退顺序指定字体家族，也可添加一个或多个字体目录。默认使用随仓库提供的 New Computer Modern 和可用系统字体。图表与正文可以分别选字体。

```bash
venusian-pcw replay --font-family "DejaVu Sans" --font-family "Noto Sans CJK SC"
venusian-pcw paper --generated --font-path /path/to/fonts --font-family "Noto Serif CJK SC"
```

也可用 `VENUSIAN_PCW_FONT_PATH` 为两个命令添加字体目录。多个目录用系统路径分隔符分开。字体配置在图件重绘时生效，归档 PDF 中的文字随原 PDF 字体显示。
