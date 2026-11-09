# 揭榜挂帅论文复现

对应当前中文论文《金星拾起离子及其对感应磁层的影响》，以项目中的 `揭榜挂帅报告.typ` 为稿件来源。仓库保留当前论文的 13 张图、正式模型、计算脚本和最小冻结派生数据。

## 从原始 MAG 和事件清单复算

安装依赖后，把 1 Hz VSO BAM 文件放到外部目录。仓库内的 `inputs/manual/event_list.csv` 是论文的 946 个 PCW 事件清单，包含起止时间和人工区域标签。

```bash
export VENUS_EXPRESS_RAW_DATA_DIR=/path/to/VEX_1Hz

# 先用一天检查冷缓存和单事件结果
python -m jiebang cache --day 2006-04-24
python -m jiebang catalogue --day 2006-04-24

# 全量计算，也可通过 --events /path/to/events.csv 指定自有清单
python -m jiebang cache --workers 4
python -m jiebang catalogue --workers 4
python -m jiebang psd-audit --workers 4
```

缓存从 BAM 生成 FAC 磁场、局地回旋频率、椭圆率和波法向角，写入 `build/feature_cache/`。运行 `catalogue` 或 `search` 时也会自动补建所需缓存。`catalogue` 从原始记录计算位置、带内波能、静态弓激波连通性和扣连续谱后的氢密度，并按样本覆盖重算年际波能和氢密度拟合。逐事件输出和摘要在 `build/raw_catalogue/`，人工目录的 10 分钟归并与论文的全任务自动事件簇分别统计。

同一数据源还能重建训练、验证和固定测试的模型输入。可选用 `cache --scope mission --workers 4` 并行预建全任务的可评分日缓存。窗口清单只包含时间、标签、父组和权重，张量从 BAM 和日缓存生成。例子如下。

```bash
python -m jiebang prepare-windows MStandard
python -m jiebang train MStandard --seed 2026 --from-raw

for stage in MSeed MExtend1 MExtend2 MStandard; do
  python -m jiebang prepare-windows "$stage" --role validation
done
python -m jiebang prepare-windows fixed_test
python -m jiebang evaluate-windows
```

重建的模型输入和重新评分结果分别位于 `build/window_features/` 和 `build/windows_from_bam/`。原始搜索、原始事件物理复算和冻结对照是可独立运行的入口。

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

## 使用作者冻结特征做精确对照

```bash
export VENUS_EXPRESS_TRAINING_INPUT_DIR=/path/to/frozen-feature-root
python -m jiebang replay --rescore-windows
python -m jiebang train MStandard --seed 2026 --device cpu
python -m jiebang train holdout2013 --seed 2026 --device cpu
```

正式四阶段为 MSeed、MExtend1、MExtend2、MStandard，各支持种子 2026、2027、2028。留出训练使用种子 2026。冻结特征的文件路径和 SHA-256 见 `inputs/protocol.json` 和 `inputs/windows/external_test.json`。若已有完整 VenusExpress 科研工程，特征根目录可指向该工程，MAG 目录可指向其 `data/raw/VEX_1Hz`。

手工复审、目录选择及 IMF 边界判读作为固定输入。弓激波的压缩比和几何点来自原矢量图，几何参数沿用论文报告值。外部数据规格和各步骤的复现边界见 [数据接口](docs/data.md)。
