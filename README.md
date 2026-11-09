# 揭榜挂帅论文复现

对应当前中文论文《金星拾起离子及其对感应磁层的影响》，以项目中的 `揭榜挂帅报告.typ` 为稿件来源。仓库保留当前论文的 13 张图、正式模型、计算脚本和最小冻结派生数据。

## 快速复现

需要 Python 3.12、Poppler 的 `pdftocairo`、Typst ≥ 0.14 和中文宋体类字体。重绘图件使用 Source Han Serif SC、Noto Serif CJK SC 或系统 Songti SC。若 Source Han Serif SC 的 OTF 不在系统字体目录，可用 `JIEBANG_CJK_FONT_DIR` 指向它。

```bash
uv venv --python 3.12
uv pip install -r requirements-repro.txt -e .
source .venv/bin/activate

python -m jiebang verify
python -m jiebang replay
python -m jiebang paper --generated
```

`replay` 从冻结派生输入重新统计窗口性能、2013 年复审、任务期发生率、年际波能、氢密度、弓激波和准径向 IMF 结果。它会重绘十张图，并为依赖 BAM 的三张图使用冻结原图。带 `--raw-figures` 时全部 13 张图都走绘图代码。直接运行 `python -m jiebang paper` 可用全部原图编译。

| 论文结果 | 复现结果 |
|---|---|
| 2013 年连续记录盲审 | 245 个最终候选，前 50 项确认 38 个 PCW |
| 全任务期基准发生率 | 1,150 簇，0.42468 簇/观测日，年率与年均黑子数相关 0.96815 |
| 年度波能 | 796 个太阳风事件，距离校正后相关约 0.695 |
| 氢密度及逃逸量级 | 78 个事件，25,000 km 处 6.55 cm⁻³，球面通量估算 2.30×10²⁵ s⁻¹ |
| 弓激波 | 日下点与晨昏面外移约 4.9% 和 7.3%，可重算压缩比及其 Welch 检验 |
| 准径向 IMF | 50 个时段中 32 个识别出 IMB，17 个时段记录到 PCW，共 55 个 PCW 事件 |

输出总表为 `build/summary.json`。论文 PDF 位于 `build/paper/`。

## 接入外部原始数据

```bash
export VENUS_EXPRESS_RAW_DATA_DIR=/path/to/VEX_1Hz

python -m jiebang replay --raw-figures
python -m jiebang psd-audit --day 2014-02-22
python -m jiebang paper --generated

# 从 BAM 重跑论文两条连续搜索
python -m jiebang search holdout --workers 4
python -m jiebang search mission --workers 6
python -m jiebang exposure --workers 6
python -m jiebang psd-audit --workers 4
python -m jiebang replay --raw-search --raw-density
python -m jiebang paper --generated
```

搜索支持 `--day YYYY-MM-DD` 做单日对照，支持 `--resume` 复用同一次运行的逐日粗分数。建议依次执行共享数据源的命令。检测使用任务期和 2013 年各自冻结的 MStandard 检查点。

## 模型重训和固定窗口前向

```bash
export VENUS_EXPRESS_TRAINING_INPUT_DIR=/path/to/frozen-feature-root
python -m jiebang replay --rescore-windows
python -m jiebang train MStandard --seed 2026 --device cpu
python -m jiebang train holdout2013 --seed 2026 --device cpu
```

正式四阶段为 MSeed、MExtend1、MExtend2、MStandard，各支持种子 2026、2027、2028。留出训练使用种子 2026。冻结特征的文件路径和 SHA-256 见 `inputs/protocol.json` 和 `inputs/windows/external_test.json`。若已有完整 VenusExpress 科研工程，特征根目录可指向该工程，MAG 目录可指向其 `data/raw/VEX_1Hz`。

手工复审、目录选择及 IMF 边界判读作为固定输入。弓激波的压缩比和几何点来自原矢量图，几何参数沿用论文报告值。外部数据规格和各步骤的复现边界见 [数据接口](docs/data.md)。
