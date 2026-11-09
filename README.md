# Venusian PCW Detector

用于在金星快车（Venus Express）的磁强计观测中检测质子回旋波（PCW），并分析事件的发生率、波能及其与金星等离子体环境的关系。

- **连续记录检测**：用训练好的模型筛选候选区间，进一步细化事件边界并执行物理核验。
- **事件目录分析**：用原始磁场和 UTC 事件区间重算波能、轨道位置、弓激波连通性及氢密度。可使用自有清单或仓库附带的 946 个标注事件。
- **模型输入重建**：从原始磁场生成可复用的日特征缓存，按窗口清单构建训练、验证和测试输入。
- **科学图表**：用 [figures.ipynb](figures.ipynb) 生成事件分布、发生率、波能、氢密度与参考弓激波模型图。

## 安装

推荐使用 Python 3.12 安装锁定版本的依赖。在仓库检出目录内执行以下命令，命令行工具会从该目录读取模型和事件清单。

```bash
uv venv --python 3.12
uv pip install -r requirements-repro.txt -e .
source .venv/bin/activate
venusian-pcw --help
```

## 从单日数据开始

将环境变量指向存放校准后 1 Hz VSO BAM 文件的目录。磁场文件格式、下载来源和事件 CSV 的字段见[数据接口与来源](docs/data.md)。

```bash
export VENUS_EXPRESS_RAW_DATA_DIR=/path/to/VEX_1Hz
venusian-pcw cache --day 2006-04-24
venusian-pcw catalogue --day 2006-04-24
venusian-pcw search mission --day 2007-02-04
```

`cache` 生成日特征缓存。`catalogue` 测量已知事件，可用 `--events /path/to/events.csv` 指定清单。`search` 在连续记录中检出新候选。原始 MAG 数据由用户提供，缓存和结果写入 `build/`。

## 用 Notebook 运行工作流

```bash
uv pip install -e '.[notebook]'
jupyter lab workflow.ipynb figures.ipynb
```

- [workflow.ipynb](workflow.ipynb) 用于缓存、事件分析、窗口构建、模型训练、评价和连续搜索。默认只校验权重和事件清单。训练、全任务搜索等耗时步骤由参数单元里的开关控制。
- [figures.ipynb](figures.ipynb) 用于绘图。在参数单元配置数据目录和字体。接入原始 MAG 后，从事件测量和连续搜索结果出图。连续搜索通过参数单独开启。PNG 和 PDF 输出到 `build/figures/`。

## 常用工作流

| 任务 | 命令 |
|---|---|
| 分析整份事件清单 | `venusian-pcw catalogue --events events.csv --workers 4` |
| 使用参考检查点搜索全任务期 | `venusian-pcw search mission --workers 6` |
| 重建训练和验证窗口 | `venusian-pcw prepare-windows MStandard` |
| 用重建的窗口特征训练 | `venusian-pcw train MStandard --seed 2026 --from-raw` |
| 汇总本地生成的搜索与曝光结果 | `venusian-pcw replay` |

仓库随附权重、事件与窗口清单、SILSO 黑子观测、弓激波位置清单和模型参数。MAG 由使用者提供。特征、分数、曝光和事件测量从原始记录生成。

分析参数、训练划分和字体设置见[工作流指南](docs/reproduce.md)。
