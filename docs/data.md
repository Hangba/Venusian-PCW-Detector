# 数据接口与来源

## MAG

`VENUS_EXPRESS_RAW_DATA_DIR` 指向只读的 1 Hz VSO BAM 文本目录。文件平铺，命名如 `BAM_2006-04-24T00-00-00_DOY_114_D001_3_VSO_SAP.dat`，也支持 `*_VSO_SAPV.dat`。文件由空白分隔，`#` 为注释。第 1 列为 UTC，第 7–10 列为 Bx、By、Bz、|B|（nT），第 11–13 列为 VSO 轨道位置（km）。正式搜索以 `inputs/mag_inventory.json` 核验日期、文件名与字节数，数据快照为论文使用的仪器团队 1 Hz 文件。预处理缓存写到 `build/`。[ESA MAG 档案](https://archives.esac.esa.int/psa/ftp/VENUS-EXPRESS/MAG/)

BAM 入口支持当前论文的两条连续搜索、空间曝光、未平滑谱峰复核、氢反演背景扣除和三个时域或轨道图。准径向 IMF 的 50 个样本来自冻结人工判读。论文提及的 32 Hz 观测无需接入上述 1 Hz 数值复算。

## IMA

附录 A 的计算入口为 `python -m jiebang ima-moments /path/to/calibrated-bins.npz`，输出 `build/ima_moments.csv`。它使用数据提供者整理后的轴序、地面标定几何因子和有效质荷比映射。原始 CDF 的能表版本、后加速档位和仪器标定由外部数据适配器解析。仓库的 IMA 积分用于公式复算，当前正文的数值表和 13 张图均可独立生成。

| NPZ 字段 | 约定 |
|---|---|
| `utc` | 各扫描的 UTC 字符串 |
| `counts` | `(scan, ring, sector, energy, pitch)`，无效或填充档置 NaN |
| `geometric_factor` | 可广播至 counts，满足 `counts/(G·E·0.1209 s)` 输出每 cm²、每 s、每 sr、每 eV 的差分通量 |
| `mass_to_charge` | 各档位的质量环等效质荷比，可广播至 counts，无效映射置 NaN |
| `energy_eV`、`width_eV` | 一维共同能量网格中心及能宽 |
| `solid_angle_sr`、`unit_vector` | `(sector,pitch)` 立体角和 `(sector,pitch,3)` IMAS 单位视向 |

不同能表或有效视场应分别归一化后积分。二级矩数据可按 UTC 与输出对齐。[CDPP 原始 IMA 产品](https://spase-metadata.org/CNES/NumericalData/CDPP-Archive/VEX/ASPERA4/VEX_ASPERA4_IMA.html)、[CDPP IMA 二级矩产品](https://spase-metadata.org/CNES/NumericalData/CDPP-Archive/VEX/ASPERA4/VEX_ASPERA4_SWM_L2.html)

## 正式模型和复算输入

- 训练和固定窗口前向读取外部 NPZ，窗口特征为 `X_mean`（4×54）、`fcp_median`、`freq_pol`、`y`、`w`、`group`、`win_start` 和 `source`。输入按论文快照散列检查。重新训练的权重与论文固定检查点分开保存。
- 默认统计以冻结的逐父候选输出、覆盖秒数、人工复审、窗口 logit、参考弓激波条件下的人工目录等为起点。`--raw-search` 使用新搜索结果。原有的人工标签按事件起止时间关联，若重新搜索改变受审时段会中止统计。
- 年度波能拟合基于冻结的 796 个纳入事件及事件位置和波能，并重新计算参考激波距离、自然三次样条、年份截距和 1500 次分年观测日 bootstrap。氢反演由 78 个事件的谱和能量派生输入复算，或从 BAM 重算谱与连续谱扣除后拟合。
- 50 例边界与 PCW 交叉表按原工作表 Sheet1 的第 4–53 行、L/Q/W 列提取，仓库只保留统计需要的派生列。弓激波压缩比和散点由两份冻结矢量图提取，重新生成比较图、Welch 检验、特征距离变化。637／343 次穿越的原始选样和拟合属于未随稿保存的上游来源。
- 月均及年均黑子数采用冻结的 WDC-SILSO V2.0 数据。年际相关使用完整公历年均值。[WDC-SILSO 数据文件](https://www.sidc.be/SILSO/datafiles)
- `reference/input_lineage.json` 记录输入的源路径和 SHA-256。源路径用于溯源，命令从当前仓库与明确指定的数据目录取数。
