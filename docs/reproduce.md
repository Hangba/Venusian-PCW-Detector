# 从原始 MAG 运行

[workflow.ipynb](../workflow.ipynb) 用于预处理、训练、评价与搜索。[figures.ipynb](../figures.ipynb) 用于科学图表。仓库自带模型权重、事件和窗口清单、SILSO 黑子观测、弓激波位置清单及参考模型参数。

## 使用 Notebook

```bash
uv pip install -e '.[notebook]'
jupyter lab workflow.ipynb figures.ipynb
```

选用安装了项目的 Python 内核。设置 `MAG_DIR`，或通过 `VENUS_EXPRESS_RAW_DATA_DIR` 指定自己的 1 Hz VSO BAM 目录。原始记录由使用者提供。特征、分数、曝光和事件测量从 BAM 生成，保存在本地 `build/`。

## 从单日开始

```bash
export VENUS_EXPRESS_RAW_DATA_DIR=/path/to/VEX_1Hz
venusian-pcw verify
venusian-pcw catalogue --day 2006-04-24
venusian-pcw search mission --day 2007-02-04
venusian-pcw replay
```

`replay` 汇总本地任务期搜索，按该次搜索的日期重算缺失的曝光，并生成发生率和空间网格。单日运行用于检查链路。全量搜索扫描数据目录内的全部观测日，可使用 `--resume` 续跑同一配置。固定测试与验证窗口均按随附清单从 BAM 重建。

## 模型训练与评价

```bash
venusian-pcw prepare-windows MStandard
venusian-pcw train MStandard --seed 2026

for stage in MSeed MExtend1 MExtend2 MStandard; do
  venusian-pcw prepare-windows "$stage" --role validation
done
venusian-pcw prepare-windows fixed_test
venusian-pcw evaluate-windows
```

四个训练阶段支持种子 2026、2027、2028。若要执行 2013 年留出搜索，先运行 `venusian-pcw prepare-windows holdout2013 --role validation`，再运行 `venusian-pcw search holdout`。留出阈值由重建的验证窗口和留出权重现场计算，人工复审标签只与相同起止时间的候选匹配。

## 出图

在 `figures.ipynb` 中配置数据目录并运行。事件分布和密度图使用本次逐事件测量。年际波能面板需要足够的逐年样本。开启 `RUN_MISSION_SEARCH` 后会从 BAM 计算候选、曝光和发生率，再生成相应图表。太阳黑子和弓激波位置图可单独运行。弓激波图按低、高活动期分别绘制穿越点和参考模型曲线，第三个面板对比两条曲线。可设置 `SHOCK_POSITIONS_CSV` 使用论文位置清单，或设置 `SHOCK_CROSSINGS_CSV` 从 UTC 穿越时刻和 BAM 提取位置并重新拟合。两项只选一种。从 BAM 重算的图保存为 `09_shock_geometry_from_bam`，便于与论文版并排查看。详细字段见[数据接口](data.md#弓激波三联图输入)。

PNG 和 PDF 写入 `build/figures/`。字体通过参数单元的 `FONT_FAMILIES` 和 `FONT_PATHS` 设置，也可用 `VENUSIAN_PCW_FONT_PATH` 添加字体目录。中文标签需要包含相应字形的字体。
