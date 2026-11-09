# 数据接口与来源

## MAG

`VENUS_EXPRESS_RAW_DATA_DIR` 指向只读的 1 Hz VSO BAM 文本目录。文件平铺，命名如 `BAM_2006-04-24T00-00-00_DOY_114_D001_3_VSO_SAP.dat`，也支持 `*_VSO_SAPV.dat`。文件由空白分隔，`#` 为注释。第 1 列为 UTC，第 7–10 列为 Bx、By、Bz、|B|（nT），第 11–13 列为 VSO 轨道位置（km）。搜索从数据目录发现可用日期，也可以用 `--day` 指定日期。预处理缓存写到 `build/`。[ESA MAG 档案](https://archives.esac.esa.int/psa/ftp/VENUS-EXPRESS/MAG/)

BAM 入口支持两条连续搜索工作流、空间曝光、未平滑谱峰复核、氢反演背景扣除和三个时域或轨道图。准径向 IMF 的 50 个样本为人工标注。这些计算使用 1 Hz 记录。

## 事件清单和缓存

`catalogue --events` 读取 CSV。必需列是 `start,end`，用整秒 UTC。参考事件清单还包含 `plasma_region`（`solar_wind`、`magnetosheath` 或 `unknown`）、`crossing_region` 和布尔值 `overlaps_recorded_sheath`。后两列是原穿越表对人工标签的支持和磁鞘重叠标注，默认分别为 `unknown` 和 `false`。只有给出太阳风标签的事件才进入参考分析口径的太阳风波能样本和日侧氢反演；只提供起止时间时仍可计算轨道、波能和静态弓激波连通性。异常或缺测事件逐行写明状态。日重采样使用固定随机种子。

`cache` 按 UTC 日从 BAM 构建并复用日特征，缓存按数据目录隔离。为保持与正式推理一致，椭圆率按 float16 存储，波法向角以 0.5° 间隔量化。替换同一路径下的原始文件后，清理该源对应的 `build/feature_cache/` 子目录再重算。`inputs/window_lists/` 记录各阶段、留出和测试窗口的起点、标签与父组。训练时按父组内窗口数生成权重。正例清单本身没有负样本和划分信息，因此重训使用这些额外的小型清单。不同原始版本或训练后端可产生数值差异。

## IMA

IMA 矩计算入口为 `venusian-pcw ima-moments /path/to/calibrated-bins.npz`，输出 `build/ima_moments.csv`。它使用数据提供者整理后的轴序、地面标定几何因子和有效质荷比映射。原始 CDF 的能表版本、后加速档位和仪器标定由外部数据适配器解析。仓库的 IMA 积分用于公式复算，输入由使用者单独提供。

| NPZ 字段 | 约定 |
|---|---|
| `utc` | 各扫描的 UTC 字符串 |
| `counts` | `(scan, ring, sector, energy, pitch)`，无效或填充档置 NaN |
| `geometric_factor` | 可广播至 counts，满足 `counts/(G × E × 0.1209 s)` 输出每 cm²、每 s、每 sr、每 eV 的差分通量 |
| `mass_to_charge` | 各档位的质量环等效质荷比，可广播至 counts，无效映射置 NaN |
| `energy_eV`、`width_eV` | 一维共同能量网格中心及能宽 |
| `solid_angle_sr`、`unit_vector` | `(sector,pitch)` 立体角和 `(sector,pitch,3)` IMAS 单位视向 |

不同能表或有效视场应分别归一化后积分。二级矩数据可按 UTC 与输出对齐。[CDPP 原始 IMA 产品](https://spase-metadata.org/CNES/NumericalData/CDPP-Archive/VEX/ASPERA4/VEX_ASPERA4_IMA.html)、[CDPP IMA 二级矩产品](https://spase-metadata.org/CNES/NumericalData/CDPP-Archive/VEX/ASPERA4/VEX_ASPERA4_SWM_L2.html)

## 随附数据与生成结果

- `checkpoints/` 提供参考权重，`inputs/protocol.json` 记录模型参数、检查点校验和和搜索协议。
- `inputs/manual/`、`inputs/holdout/review.csv`、`inputs/boundaries/radial_imf.csv` 和 `inputs/window_lists/` 为事件时段、人工标签和窗口清单。
- `inputs/silso.csv` 与 `inputs/silso_monthly.csv` 是 SILSO V2.0 年均、月均黑子观测。年度统计使用完整公历年均值。[WDC-SILSO 数据文件](https://www.sidc.be/SILSO/datafiles)
- 弓激波参考曲线由模型参数计算。`inputs/boundaries/shock_positions.csv` 含低、高活动期各 637、343 个 AVSO 穿越位置，坐标由归档矢量图恢复。字段 `X_AVSO_Rv` 和 `R_AVSO_Rv` 以金星半径为单位，`period` 为 `low` 或 `high`。候选、曝光、轨道点、波能、密度、窗口张量和模型分数均由工作流生成到 `build/`。

### 弓激波三联图输入

| 输入 | 必需字段 | 用途 |
|---|---|---|
| `SHOCK_POSITIONS_CSV` | `X_AVSO_Rv,R_AVSO_Rv,period` | 用已有 AVSO 位置和随附模型参数复绘，或开启 `SHOCK_REFIT_POSITIONS` 从散点重拟合 |
| `SHOCK_CROSSINGS_CSV` | `utc,period` | 从原始 BAM 量取穿越位置并拟合两组圆锥曲线 |

两种清单的 `period` 都用 `low` 或 `high`。两项都留空时使用仓库附带的位置清单。UTC 入口为每个穿越时刻匹配 2 秒内最近的有效 MAG 位置，转换到 AVSO 后分别对两期拟合圆锥隐式方程残差，输出位置与模型参数到 `build/boundaries/`。从 UTC 重算论文散点需要原始穿越时刻清单；随附的 980 个位置来自原矢量图，未附这些时刻。
