# 数据接口与来源

## MAG

`VENUS_EXPRESS_RAW_DATA_DIR` 指向只读的 1 Hz VSO BAM 文本目录。文件平铺，命名如 `BAM_2006-04-24T00-00-00_DOY_114_D001_3_VSO_SAP.dat`，也支持 `*_VSO_SAPV.dat`。文件由空白分隔，`#` 为注释。第 1 列为 UTC，第 7–10 列为 Bx、By、Bz、|B|（nT），第 11–13 列为 VSO 轨道位置（km）。正式搜索以 `inputs/mag_inventory.json` 核验日期、文件名与字节数，参考快照为仪器团队提供的 1 Hz 文件。预处理缓存写到 `build/`。[ESA MAG 档案](https://archives.esac.esa.int/psa/ftp/VENUS-EXPRESS/MAG/)

BAM 入口支持两条连续搜索工作流、空间曝光、未平滑谱峰复核、氢反演背景扣除和三个时域或轨道图。准径向 IMF 的 50 个样本来自冻结人工判读。这些计算使用 1 Hz 记录。

## 事件清单和缓存

`catalogue --events` 读取 CSV。必需列是 `start,end`，用整秒 UTC。参考事件清单还包含 `plasma_region`（`solar_wind`、`magnetosheath` 或 `unknown`）、`crossing_region` 和布尔值 `overlaps_recorded_sheath`。后两列是原穿越表对人工标签的支持和磁鞘重叠标注，默认分别为 `unknown` 和 `false`。只有给出太阳风标签的事件才进入参考分析口径的太阳风波能样本和日侧氢反演；只提供起止时间时仍可计算轨道、波能和静态弓激波连通性。异常或缺测事件逐行写明状态。参考分析的日重采样沿用原始随机种子和日编号顺序。该顺序单独保存，参与反演的事件仍由原始几何和谱计算决定。

`cache` 按 UTC 日从 BAM 构建并复用日特征，缓存按数据目录隔离。为保持与正式推理一致，椭圆率按 float16 存储，波法向角以 0.5° 间隔量化。替换同一路径下的原始文件后，清理该源对应的 `build/feature_cache/` 子目录再重算。`inputs/window_lists/` 固定了各阶段、留出和测试窗口的顺序、标签、父组及权重。正例清单本身没有负样本和划分信息，因此重训使用这些额外的小型清单。不同原始版本或训练后端可产生数值差异。

## IMA

IMA 矩计算入口为 `venusian-pcw ima-moments /path/to/calibrated-bins.npz`，输出 `build/ima_moments.csv`。它使用数据提供者整理后的轴序、地面标定几何因子和有效质荷比映射。原始 CDF 的能表版本、后加速档位和仪器标定由外部数据适配器解析。仓库的 IMA 积分用于公式复算，参考统计表和 13 张图均可独立生成。

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

- 训练和固定窗口前向读取外部 NPZ，窗口特征为 `X_mean`（4×54）、`fcp_median`、`freq_pol`、`y`、`w`、`group`、`win_start` 和 `source`。输入按冻结输入清单散列检查。重新训练的权重与参考检查点分开保存。
- 默认统计以冻结的逐父候选输出、覆盖秒数、人工复审、窗口 logit、参考弓激波条件下的人工目录等为起点。`--raw-search` 使用新搜索结果。原有的人工标签按事件起止时间关联，若重新搜索改变受审时段会中止统计。
- 年度波能拟合基于冻结的 796 个纳入事件及事件位置和波能，并重新计算参考激波距离、自然三次样条、年份截距和 1500 次分年观测日 bootstrap。冻结复算使用 78 个事件的派生输入。原始入口从事件时段重新计算位置、连通性、回旋频带、连续谱、拾起质子和中性氢。
- 50 例边界与 PCW 交叉表按原工作表 Sheet1 的第 4–53 行、L/Q/W 列提取，仓库只保留统计需要的派生列。弓激波压缩比和散点是从归档矢量图提取的数值输入，已保存为 CSV。Notebook 读取它们绘图，统计模块重算 Welch 检验和特征距离变化。637／343 次穿越的原始选样和拟合属于未随稿保存的上游来源。
- 月均及年均黑子数采用冻结的 WDC-SILSO V2.0 数据。年际相关使用完整公历年均值。[WDC-SILSO 数据文件](https://www.sidc.be/SILSO/datafiles)
- `reference/input_lineage.json` 记录输入的源路径和 SHA-256。源路径用于溯源，命令从当前仓库与明确指定的数据目录取数。
