# Street View 与 Satellite：Qwen3-VL-8B-Instruct SFT Results

记录日期：2026-10-08。两项实验均完成正式训练、全部候选评测、checkpoint 选择、全量 validation/OOD 及独立核验。训练配置见 [Experiment Setup](experiment_setup.md)。

## 1. 核心结果

| 实验 | 总更新 | 最佳 checkpoint | Probe EM | 完整 validation EM | Base OOD EM | SFT OOD EM | OOD 提升 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Street View | 626 | checkpoint-500 | 69.14% | 932/1,445 (64.50%) | 29/3,525 (0.82%) | 2,269/3,525 (64.37%) | +63.55 pp |
| Satellite | 752 | checkpoint-700 | 83.20% | 1,337/1,684 (79.39%) | 39/4,299 (0.91%) | 3,394/4,299 (78.95%) | +78.04 pp |

Streetview, Satellite 均从原始 base model 独立训练。各模态的 base/SFT OOD 比较使用完全相同的数据与生成协议。两种模态的数据集和样本数不同，不能将二者分数差直接解释为图像模态带来的因果提升。`pp` 表示百分点。

## 2. Training Dynamics

### 2.1 训练 loss、学习率与梯度

![两项实验的训练 loss、validation loss、学习率与梯度范数](figures/training_dynamics.png)

图 1（图件更新：2026-10-14）：StreetView 为 panels (a–c)，Satellite 为 panels (d–f)；三列分别为 loss、学习率和梯度范数。蓝色/橙色实线表示各模态的训练日志，loss 面板中的灰色虚线与圆点表示合格 CoT 验证集 loss，竖直彩色虚线标出选中 checkpoint。横轴为 optimizer update；Street View 有 62 个训练日志点、8 个 loss validation 点，Satellite 有 75 个训练日志点、9 个 loss validation 点。Training loss 是日志窗口统计，validation loss 是独立验证集统计，二者不能作为完全等条件的 loss 差比较。图中未平滑或补造首尾数据点。梯度图使用保存的 `grad_norm` 日志值，不将其解释为裁剪后范数。

| 训练指标 | Street View | Satellite |
| --- | --- | --- |
| 首个记录 training loss | 1.7806（step 10） | 1.7805（step 10） |
| 最后记录 training loss | 0.8818（step 620） | 0.7787（step 750） |
| 首个 validation loss | 1.023509（step 100） | 0.912707（step 100） |
| 最终 validation loss | 0.903486（step 626） | 0.785577（step 752） |
| 最佳 checkpoint 的 validation loss | 0.905951（step 500） | 0.785732（step 700） |

两个实验的 loss 均整体下降，后期变化幅度减小；学习率按各自步数 warmup 后余弦衰减。最终选择依据是验证集自由生成排序指标，而不是最低 loss。

### 2.2 候选 checkpoint 的生成验证表现

![候选 checkpoint 的 Exact Match 和 directed adjacency 动态](figures/checkpoint_validation_dynamics_with_baseline.png)

图 2：panel (a) 为 Exact Match，panel (b) 为 directed adjacency；蓝色为 StreetView，橙色为 Satellite。Step=0 表示未经 SFT 的原始 base model；其余点表示训练后的候选 checkpoint。所有点使用本模态同一固定 256 条 validation probe 和同一 vLLM 生成协议，不使用 OOD 分数作为起点。Exact Match 分母为 256 条轨迹，有向邻接指标分母为 1,792 条边（256 × 7）；连线仅用于观察趋势，不代表中间步骤已评测。星号为选中候选，OOD 未用于选择。Base model 的监督 validation loss 未列入本次生成评测记录，因此表中标为“—”。

#### 2.2.1 Street View

| Checkpoint | Exact / 256 | Exact Match | 正确有向边 / 1,792 | Validation loss |
| --- | --- | --- | --- | --- |
| 0（Base model） | 0 | 0.00% | 318 | — |
| 100 | 66 | 25.78% | 1009 | 1.02350938 |
| 200 | 120 | 46.88% | 1269 | 0.95906740 |
| 300 | 147 | 57.42% | 1370 | 0.92862260 |
| 313 | 145 | 56.64% | 1361 | 0.92635530 |
| 400 | 162 | 63.28% | 1436 | 0.91498142 |
| **500（最佳）** | 177 | 69.14% | 1509 | 0.90595096 |
| 600 | 165 | 64.45% | 1432 | 0.90351391 |
| 626 | 168 | 65.62% | 1498 | 0.90348625 |

第 500 步达到最高 probe Exact Match 177/256（69.14%）；继续训练后 loss 略降，但第 600、626 步的完整排序准确率未超过它，因此选择 checkpoint-500。

#### 2.2.2 Satellite

| Checkpoint | Exact / 256 | Exact Match | 正确有向边 / 1,792 | Validation loss |
| --- | --- | --- | --- | --- |
| 0（Base model） | 0 | 0.00% | 370 | — |
| 100 | 52 | 20.31% | 901 | 0.91270691 |
| 200 | 147 | 57.42% | 1343 | 0.84882349 |
| 300 | 174 | 67.97% | 1538 | 0.82052380 |
| 376 | 194 | 75.78% | 1604 | 0.80707228 |
| 400 | 192 | 75.00% | 1595 | 0.80433953 |
| 500 | 199 | 77.73% | 1619 | 0.79470742 |
| 600 | 206 | 80.47% | 1638 | 0.78802103 |
| **700（最佳）** | 213 | 83.20% | 1669 | 0.78573221 |
| 752 | 213 | 83.20% | 1648 | 0.78557748 |

第 700、752 步均达到 213/256（83.20%）。按预定同分规则，第 700 步正确有向边更多（1,669 对 1,648），因此选择 checkpoint-700，虽然第 752 步的 loss 略低。

## 3. 完整 Validation 结果

### 3.1 整体排序质量

| 实验 | 模型 | N | 完全正确 / N | Exact Match | Pairwise | Kendall τ | Adjacency | Position |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Street View | SFT-500 | 1,445 | 932/1,445 | 64.50% | 88.86% | 0.7771 | 81.87% | 78.88% |
| Satellite | SFT-700 | 1,684 | 1,337/1,684 | 79.39% | 91.83% | 0.8366 | 90.74% | 85.21% |

Base validation 仅评测固定 256 条 probe，两项实验的 EM 均为 0/256；SFT 评测的是各自完整 1,445 / 1,684 条 validation。两者不是同集合，不能直接计算 validation 提升。

### 3.2 验证城市

| 实验 | 城市 | N | 完全正确 / N | Exact Match | Adjacency | Position |
| --- | --- | --- | --- | --- | --- | --- |
| Street View | Toronto | 749 | 487/749 | 65.02% | 83.16% | 79.84% |
| Street View | Vienna | 696 | 445/696 | 63.94% | 80.48% | 77.84% |
| Satellite | Toronto | 931 | 722/931 | 77.55% | 89.83% | 83.55% |
| Satellite | Vienna | 753 | 615/753 | 81.67% | 91.88% | 87.25% |

## 4. 全量 OOD：Base 与 SFT 配对比较

### 4.1 整体排序质量（Micro Average）

| 实验 | 模型 | N | 完全正确 / N | Exact Match | Pairwise | Kendall τ | Adjacency | Position |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Street View | Base | 3,525 | 29/3,525 | 0.82% | 47.72% | -0.0456 | 20.49% | 14.64% |
| Street View | SFT-500 | 3,525 | 2,269/3,525 | **64.37%**| 88.04% | 0.7607 | 81.69% | 77.75% |
| Satellite | Base | 4,299 | 39/4,299 | 0.91% | 41.99% | -0.1603 | 20.86% | 13.28% |
| Satellite | SFT-700 | 4,299 | 3,394/4,299 | **78.95%**| 91.64% | 0.8328 | 90.28% | 85.24% |

### 4.2 格式、合法性与生成长度

| 实验 | 集合 / 模型 | 合法排列 | 严格 CoT 格式 | 截断率 | 平均输出 tokens |
| --- | --- | --- | --- | --- | --- |
| Street View | Validation / SFT | 94.95% | 96.40% | 3.60% | 1014.1 |
| Street View | OOD / Base | 93.11% | 0.00% | 6.07% | 980.1 |
| Street View | OOD / SFT | 95.12% | 96.94% | 3.06% | 1005.9 |
| Satellite | Validation / SFT | 97.62% | 99.58% | 0.42% | 804.2 |
| Satellite | OOD / Base | 86.02% | 0.00% | 13.10% | 1183.8 |
| Satellite | OOD / SFT | 97.44% | 99.70% | 0.30% | 808.6 |

严格 CoT 格式率检验冻结的四步 think/answer 格式，和答案是否为合法排列、排序是否正确是不同指标。无效排列计入全部样本分母，pairwise/adjacency/position 计 0，Kendall τ 计 −1。

### 4.3 五城 Exact Match

| 实验 | 城市 | N | Base 完全正确 | Base EM | SFT 完全正确 | SFT EM | 提升 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Street View | Berlin | 743 | 6/743 | 0.81% | 504/743 | 67.83% | +67.03 pp |
| Street View | Buenos Aires | 891 | 10/891 | 1.12% | 628/891 | 70.48% | +69.36 pp |
| Street View | Melbourne | 668 | 5/668 | 0.75% | 389/668 | 58.23% | +57.49 pp |
| Street View | Osaka | 493 | 3/493 | 0.61% | 304/493 | 61.66% | +61.05 pp |
| Street View | Philadelphia | 730 | 5/730 | 0.68% | 444/730 | 60.82% | +60.14 pp |
| Satellite | Berlin | 718 | 11/718 | 1.53% | 599/718 | 83.43% | +81.89 pp |
| Satellite | Buenos Aires | 984 | 3/984 | 0.30% | 689/984 | 70.02% | +69.72 pp |
| Satellite | Melbourne | 953 | 12/953 | 1.26% | 814/953 | 85.41% | +84.16 pp |
| Satellite | Osaka | 754 | 7/754 | 0.93% | 630/754 | 83.55% | +82.63 pp |
| Satellite | Philadelphia | 890 | 6/890 | 0.67% | 662/890 | 74.38% | +73.71 pp |


## 5. 耗时与显存

### 5.1 正式训练

| 项目 | Street View | Satellite |
| --- | --- | --- |
| 训练完成 | 626 updates / 2 epochs | 752 updates / 2 epochs |
| 可核实 Trainer runtime | 最后接续段 300→626：6.63 小时 | 完整 0→752：13.53 小时 |
| 纯更新窗口平均耗时 | 47.30 秒/update（610→620） | 45.72 秒/update（710→720） |
| Rank 0 / 1 峰值 allocated | 34.103 / 34.594 GiB | 34.376 / 34.376 GiB |
| Rank 0 / 1 峰值 reserved | 37.297 / 37.863 GiB | 37.438 / 37.312 GiB |

Trainer runtime 包含 loss validation 与保存，不包含训练后 vLLM generation。Street View 的 6.63 小时只覆盖最后接续段，不能当作全部 626 步总训练时间。纯更新窗口不含验证/保存，受共享 GPU 负载影响；epoch 尾部不足完整累积组的更新未用于通常步速估计。

### 5.2 最终生成评测

| 实验 | 生成阶段 | N | Wall time（分钟） | Output tokens/s（双卡合计） |
| --- | --- | --- | --- | --- |
| Street View | SFT 完整 validation | 1,445 | 36.93 | 661.32 |
| Street View | Base 全量 OOD | 3,525 | 128.06 | 449.65 |
| Street View | SFT 全量 OOD | 3,525 | 113.49 | 520.73 |
| Satellite | SFT 完整 validation | 1,684 | 35.95 | 627.87 |
| Satellite | Base 全量 OOD | 4,299 | 109.67 | 773.36 |
| Satellite | SFT 全量 OOD | 4,299 | 93.14 | 621.97 |

生成 wall time 包含图片准备、生成、保存及评分，排除引擎初始化。正式协议为每卡活跃 8、已提交请求上限 32、持续补充；逐请求排队延迟不等于训练时间，也不能相加当作阶段 wall time。

## 6. 最终模型与结果来源

### 6.1 模型交付

| 实验 | 最佳 adapter（服务器绝对路径） | 保留完整恢复状态 |
| --- | --- | --- |
| Street View | `/path/to/visual_trajectory/runs/streetview_sft_qwen3vl8b_lora_v1/training/checkpoint-500` | 500、600、626 |
| Satellite | `/path/to/visual_trajectory/runs/satellite_sft_qwen3vl8b_lora_v1/training/checkpoint-700` | 700、752 |

Adapter 推理需要同一原始 base model。全部候选 adapter、processor/tokenizer、训练元数据、选择证据和逐条预测保留；仅清理不再保留候选的 optimizer/scheduler/RNG 中间状态。

### 6.2 Training dynamics 数据来源

图 1 使用各正式 run 的最终 checkpoint `trainer_state.json` 中 `log_history`；与训练目录最终 `trainer_state.json` 的对应历史逐条核对一致。图 2 的 step=0 使用 `vllm/evaluations/baseline_validation_probe/summary.json`；其余点使用 `vllm/evaluations/checkpoint-*/summary.json`，与当前步 validation loss 配对，并重算正确轨迹数、有向边整数计数及最佳选择。

| 实验 | 训练曲线源文件（服务器绝对路径） | 日志点：train / loss validation |
| --- | --- | --- |
| Street View | `/path/to/visual_trajectory/runs/streetview_sft_qwen3vl8b_lora_v1/training/checkpoint-626/trainer_state.json` | 62 / 8 |
| Satellite | `/path/to/visual_trajectory/runs/satellite_sft_qwen3vl8b_lora_v1/training/checkpoint-752/trainer_state.json` | 75 / 9 |

各 run 的 `experiment_result.json`、`vllm/evaluations/*/summary.json` 和逐条 predictions 为评测来源；`training/configuration.json` 与 `final_audit.json` 为实际配置及完整核验入口。原始冻结数据、来源哈希及基础模型文件已核验，清理前后核验通过。

本地图源 `sft_plot_data.json`（外部图源，不随公开代码上传） 保留绘图所需原始日志、probe 整数计数及源文件 SHA-256。只读采集脚本为 [collect_sft_plot_data.py](../../code/training/collect_sft_plot_data.py)，绘图脚本为 [plot_sft_results.py](../../code/training/plot_sft_results.py)。复现图件：

```powershell
python code/training/plot_sft_results.py --source training/SFT/figures/sft_plot_data.json --output training/SFT/figures
```

### 6.3 解释范围

Street View OOD 中 352 条轨迹曾用于 teacher 比较与选择，完整结果包含该子集，不能宣称全部测试数据此前完全未使用。Satellite 五城 teacher 输出未参与本次 SFT 或 checkpoint 选择。两项实验均未对全部 teacher 推理人工逐条视觉审核；格式合规和答案正确不等于推理中的每个视觉细节都可靠。

正式保留 checkpoint 的恢复核验包括 CPU optimizer/scheduler、CPU/Python/NumPy RNG 和零学习率更新；CUDA RNG 检查结构。实际双卡 25→26 恢复有预检证据，CPU 核验不等于从最佳正式 checkpoint 在 GPU 上继续训练。
