# 当前训练与推理全流程计时（2026-10-16）

环境：Apple M4 Pro，24 GiB统一内存，macOS-27.0.1-arm64-arm-64bit，PyTorch 2.13.0，MPS后端。
本机 MPS，新架构 dual CNN（24,084 参数），四模型 MSeed / MExtend1 / MExtend2 / MStandard。正式模型和论文未修改。训练为各自从头初始化的独立模型，四阶段耗时相加表示顺序训练成本，不是同一个模型逐阶段继续训练。

历史正式训练：四模型×三种子×100 epoch，共 **1332.43 s（22.21 min）**。本次逐阶段计时重跑每模型 seed2026，共 **467.40 s**；计时含阶段间 MPS 同步，不能将它和原来无阶段同步的 fit 时间当作完全相同的测量。

## 训练总量：原有三种子真实记录

| 模型 | seed2026 (s) | seed2027 (s) | seed2028 (s) | 均值 ± SD (s) | 三种子合计 (s) |
| --- | --- | --- | --- | --- | --- |
| MSeed | 104.60 | 103.97 | 103.90 | 104.15 ± 0.39 | 312.46 |
| MExtend1 | 105.13 | 104.19 | 104.25 | 104.52 ± 0.53 | 313.57 |
| MExtend2 | 106.50 | 105.13 | 104.62 | 105.42 ± 0.97 | 316.25 |
| MStandard | 132.28 | 128.88 | 128.98 | 130.05 ± 1.93 | 390.14 |

## 训练各部分：本次 seed2026 全100 epoch重跑

| 部分 | MSeed | MExtend1 | MExtend2 | MStandard |
| --- | --- | --- | --- | --- |
| Load / setup / final checks | 0.6991 | 0.6060 | 0.6437 | 0.6143 |
| Sampling | 0.0224 | 0.0230 | 0.0236 | 0.0288 |
| Optimize (5200 steps) | 107.6405 | 107.4002 | 107.7394 | 134.5298 |
| Validation forward | 0.5570 | 0.5903 | 1.0485 | 1.0886 |
| Validation metrics | 0.0359 | 0.0468 | 0.0645 | 0.0679 |
| 100 checkpoint writes | 0.8441 | 0.8322 | 0.9367 | 0.8869 |
| History / progress | 0.1008 | 0.1029 | 0.1024 | 0.1043 |
| 完整 fit 墙钟 (s) | 109.9042 | 109.6049 | 110.5626 | 137.3245 |
| 新进程含导入/退出 (s) | 112.0104 | 111.3527 | 112.3582 | 139.0994 |

每个模型固定52步/epoch、共5200次优化；MStandard每步另加16个F窗口。优化项包括输入选择及传输、前向、损失、反向和AdamW更新，未在每个算子间强制同步。100次checkpoint写入另计。原正式训练与重跑的训练损失序列最大绝对差分别为：MSeed=0, MExtend1=0, MExtend2=0, MStandard=0。

数据准备（原有特征NPZ与已审计筛选掩码→子集、父事件权重、隔离检查、9个压缩NPZ及来源校验）：**1.863 s**。八份训练/验证数组与正式输入逐项完全一致。

12模型训练后选择：**0.148 s**；CPU重新加载验证、逐项验证及60个工作点：**1.175 s**。这是读取已有12模型权重的独立重放，不是本次只跑的四模型被冒充成12模型。

训练计时起点是已冻结的原始特征NPZ与筛选掩码。最初目录标注、样本收集、历史负样本扩展、原始训练特征生成不在该计时内；没有将它们记作0秒。先前P窗门控审计记录28.897 s仅覆盖当时新增1199个窗及后续审计，并复用了已有物理结果，不能当成全体训练窗从零物理审计耗时。

![本次训练与日推理分项](phase_times.png)

## 推理各部分：原始MAG到已保存事件

四个完整UTC日，独立新Python进程、空应用特征缓存，OS页缓存未清空。MStandard seed2026、原验证选择epoch65，阈值0.70，80 s窗/20 s步长、batch4096、CPU线程4。原论文的端点/中心时长拯救，未含后续gap4/STFT扩展。所有日的时间网格一致、分数与正式全任务缓存逐项allclose通过。

| 日期 | 样本 | 窗口 | 初始候选 | 细化种子 | 保留事件 | 原始数据到保存 (s) | 含导入/初始化/退出 (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2006-05-15 | 83380 | 4165 | 0 | 0 | 0 | 1.175 | 14.301 |
| 2009-10-12 | 84650 | 4229 | 3 | 5 | 5 | 1.237 | 3.044 |
| 2011-02-20 | 82598 | 4126 | 20 | 27 | 26 | 1.756 | 3.343 |
| 2014-02-22 | 82924 | 4143 | 19 | 32 | 29 | 2.068 | 3.653 |

| 部分 | 均值 (s) | 范围 (s) |
| --- | --- | --- |
| Raw MAG / FAC / spectral features + cache | 0.3063 | 0.2988–0.3269 |
| Cache read / windows / CNN scores | 0.7387 | 0.6723–0.8366 |
| Score encoding | 0.0082 | 0.0081–0.0085 |
| Candidate assembly | 0.0003 | 0.0000–0.0005 |
| Physical refinement + rescue | 0.5019 | 0.0000–1.0029 |
| Final six gates | 0.0000 | 0.0000–0.0000 |
| Export | 0.0036 | 0.0033–0.0040 |

| 预处理/网络内部（包含于上表） | 四日均值 (s) |
| --- | --- |
| preprocessing/load_data | 0.1452 |
| preprocessing/polarization_analysis_on_xy | 0.0195 |
| preprocessing/propagation_angle_analysis | 0.0183 |
| preprocessing/tofac | 0.0116 |
| refinement/load_data | 0.1031 |
| refinement/tofac | 0.0083 |
| CNN forward (MPS同步，不含输入组装) | 0.1248 |

预处理内的raw_and_features_seconds与daily_cache_write_seconds属于不同操作；二者都已包含在preprocess_seconds内，不能再次加到预处理总量。窗口评分包含缓存重读、窗口特征组装、网络前向与输出整理，因此不能称为纯神经网络耗时。细化内部的FFT、偏振、WNA和wide_rescue_inclusive有嵌套，原始JSON保留所有子计时，禁止直接加总。首个观测日的新进程时间还包含首次Matplotlib字体缓存初始化；原始数据到保存的内部计时不含该启动开销。

| 细化内部计时（有嵌套） | 四日均值 (s) |
| --- | --- |
| _fft_metrics | 0.3072 |
| batched_polarization_analysis | 0.0044 |
| batched_propagation_analysis | 0.0041 |
| wide_rescue_inclusive | 0.1452 |

## 最新gap≤4 + STFT全局搜索：已有真实运行

完整2888日、四模型、阈值0.5/0.6/0.7/0.8的最新增量运行：34分41.82秒。它复用粗评分、已有细评分与部分物理结果，含中断和诊断等待；不包含所有原始数据从零预处理及所有粗评分。其阶段如下，文件时间线估计与perf_counter实测分开。

| 阶段 | 耗时 (s) | 来源 |
| --- | --- | --- |
| initialization_and_reuse_checks | 1.792 | filesystem estimate |
| dense_rescoring_all_four_models | 162.540 | measured perf_counter, captured from completed dense progress |
| interrupted_candidate_assembly_overhead | 265.175 | filesystem estimate; incomplete attempt, separately charged |
| resume_preparation_and_equivalent_candidate_assembly | 35.980 | filesystem estimate; includes cached dense-stage skip |
| physics_before_speed_optimization | 943.305 | measured perf_counter |
| cached_physics_attempt_with_edge_guard_exit | 377.606 | filesystem duration; includes completed work and pool shutdown after assertion |
| remaining_edge_physics_checks | 5.500 | measured perf_counter |
| overlap_merge_and_aggregation | 24.363 | measured perf_counter |

[既有完整计时与中断说明](../four_model_gap4_stft_global_20261016/timing.md)。该总墙钟还包含265.557秒的进程切换/诊断/等待间隔；数据校验与绘图另外19.509秒。并行任务elapsed之和10429.54秒不是墙钟，不能加到2081.82秒上。

## STFT内部：相同12个真实失败候选重跑

两个真实UTC日上的同一12个scope、串行，分别重算原始DataAnalysis、按日FAC缓存冷启动和缓存命中。三种模式的完整物理量、STFT审计、输出边界和结果严格相同。

| 部分 | 原版 (s) | 按日缓存冷启动 (s) | 按日缓存命中 (s) |
| --- | --- | --- | --- |
| raw_FAC_local_evaluator_setup | 1.8189 | 0.5484 | 0.0929 |
| whole_interval_six_gates | 0.0121 | 0.0117 | 0.0108 |
| STFT_support_profile | 0.8130 | 0.8147 | 0.8225 |
| STFT_endpoint_proposals | 0.0023 | 0.0021 | 0.0023 |
| STFT_proposal_six_gates | 0.6974 | 0.6950 | 0.6954 |
| 其他处理和保存 | 0.0083 | 0.0083 | 0.0088 |
| 总墙钟 | 3.3519 | 2.0802 | 1.6326 |

## 证据与限制

所有计时以本机实际负载为准，未关闭用户进程；4日均值不能冒充2888日逐日冷启动实测。正式训练时间、重跑计时、历史全任务增量时间分别列出，不相加制造一个不存在的完整冷启动总时间。完整原始JSON保留在training/、inference/与sequential_cpu/，子进程日志位于本目录。最初三项CPU补测存在并行重叠，已保留但不采用；报告只采用sequential_cpu/下顺序执行的独立补测。

生成脚本：experiments/paper_runtime/profile_current_pipeline_20261016.py 和 report_current_pipeline_20261016.py。
