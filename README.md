# Venusian PCW Detector

用于在金星快车（Venus Express）的磁强计观测中检测质子回旋波（PCW），并分析事件的发生率、波能及其与金星等离子体环境的关系。

- **连续记录检测**：用训练好的模型筛选候选区间，进一步细化事件边界并执行物理核验。
- **事件目录分析**：用原始磁场和 UTC 事件区间重算波能、轨道位置、弓激波连通性及氢密度。可使用自有清单或仓库附带的 946 个参考事件。
- **模型输入重建**：从原始磁场生成可复用的日特征缓存，按窗口清单构建训练、验证和测试输入。
- **研究结果复现**：提供参考检查点、冻结输入和图表生成脚本。

## 安装

推荐使用 Python 3.12 安装锁定版本的依赖。在仓库检出目录内执行以下命令，命令行工具会从该目录读取模型和参考输入。

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

## 常用工作流

| 任务 | 命令 |
|---|---|
| 分析整份事件清单 | `venusian-pcw catalogue --events events.csv --workers 4` |
| 使用参考检查点搜索全任务期 | `venusian-pcw search mission --workers 6` |
| 重建训练和验证窗口 | `venusian-pcw prepare-windows MStandard` |
| 用重建的窗口特征训练 | `venusian-pcw train MStandard --seed 2026 --from-raw` |
| 使用冻结输入重算参考结果 | `venusian-pcw replay` |

研究基准、训练划分和图表生成方法见[复现指南](docs/reproduce.md)。绘图和论文编译均支持自选字体，详见[字体配置](docs/reproduce.md#字体配置)。
