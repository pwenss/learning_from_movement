# StreetView 与 Satellite：GRPO + OPSD Results

更新日期：2026-10-14。两项训练均已按用户要求提前停止，选中模型的完整validation/OOD及合并SFT起点全量OOD均已完成并独立验收。正式配置见 [Experiment Setup](experiment_setup.md)。

## 1. 最终模型与总体结果

### 1.1 核心结果

| 实验 | 选中cycle / update | Probe EM | 完整validation EM | 合并SFT起点OOD EM | 后训练OOD EM | OOD提高 |
|---|---|---:|---:|---:|---:|---:|
| StreetView | 68 / 272 | 177/256（69.14%） | 1007/1445（69.69%） | 2248/3525（63.77%） | **2404/3525（68.20%）** | **+156条 / +4.43 pp** |
| Satellite | 51 / 204 | 226/256（88.28%） | 1396/1684（82.90%） | 3358/4299（78.11%） | **3515/4299（81.76%）** | **+157条 / +3.65 pp** |

`pp`表示百分点；OOD提高相对于实际合并SFT训练起点。两个模态的数据、prompt与样本数不同，不将跨模态分数差解释为图像模态的因果效果。联合训练没有GRPO-only或OPSD-only消融，结果不分摊为两个loss各自的贡献。

### 1.2 实际停止状态

| 项目 | StreetView | Satellite |
|---|---|---|
| 原计划epoch | 174 cycles / 696 updates | 182 cycles / 728 updates |
| 实际有效更新 | 272 | 279 |
| 最后完整保存点 | cycle68 / update272 | cycle68 / update272 |
| 最终采用模型 | cycle68 | cycle51 |
| 选中保存点数据游标 | 6,528条轨迹 | 4,896条轨迹 |
| 停止方式 | cycle68保存后受控停止 | 69完整cycles后，cycle70完成3/4次更新时受控停止 |

两项均未完成计划的完整epoch。Satellite cycle70的768个rollout回答已经生成，但只有前三个optimizer minibatches完成；不将该cycle描述为完整训练批次。StreetView从cycle51迁移到四卡后重算旧未保存分支，只计最终有效updates1–272。

## 2. Training Dynamics

### 2.1 Rollout EM、Entropy (MC) 与 Reference KL

![正式训练动态：StreetView panels a–c，Satellite panels d–f；rollout EM、采样熵估计与reference KL](figures/training_dynamics.png)

Panels (a/d)：每cycle768个 **temperature=1采样回答的平均EM**，不是best-of-8或greedy probe EM；轨迹批次随cycle变化。Panels (b/e)：Entropy (MC)，取日志 `rollout_corr/rollout_log_ppl`，先对每个回答的有效response tokens计算平均 −log p，再对回答及等大小rank取均值，单位nats/token。生成采用T=1、top_p=1、top_k=−1，因此它是采样rollout policy的熵估计；正式训练关闭全词表entropy计算，不能将它当作全词表精确熵。每cycle内的四个点来自同一rollout policy的不同minibatches，不是四次更新后重新生成的熵。

Panels (c/f)：原始reference KL估计量 `kl_loss`，对rank取均值，**未乘0.001损失系数**，也不是old/current policy之间的PPO KL。

浅线为原始值，粗线为向后均值：EM使用5个完整cycles，熵与KL使用8次updates；不足窗口时使用已观测点，不补造cycle0指标。

彩色虚线标出最终选中保存点；StreetView灰色点线为cycle51三卡→四卡迁移。Satellite空心点为cycle70已经生成的rollout，未纳入完整cycle平滑线；熵与KL保留实际完成的第277–279次更新。

| 训练观测 | StreetView | Satellite |
|---|---:|---:|
| 前10个完整cycle rollout EM均值 | 44.48% | 66.20% |
| 最后10个完整cycle rollout EM均值 | 54.05%（59–68） | 76.41%（60–69） |
| 裁剪前grad norm范围 | 0.03461–0.08651 | 0.03545–0.06732 |
| 图中有效optimizer updates | 272 | 279 |
| 全部rank同步有限更新检查 | 通过 | 通过 |

Joint loss包含组内中心化的policy-gradient项，正负波动且接近0不等于没有梯度，也不具有SFT交叉熵loss那样的单调下降解释。全部保留更新的loss/metrics有限，跨rank梯度一致，并核对 `joint = GRPO + 0.01×OPSD + 0.001×reference KL`。

### 2.2 固定validation probe动态

![固定256条probe的EM和directed adjacency随update变化](figures/checkpoint_probe_dynamics.png)

Panel (a)为EM，panel (b)为directed adjacency。Update0使用实际部署的合并SFT起点；每cycle4次updates。每个点均为本模态同一256条固定probe，temperature0；星号是最终选中模型。连线仅展示已评估保存点间的变化，不代表中间checkpoint已经测评。

| 实验 | Cycle / update | EM正确数 / 256 | EM | Directed adjacency | Position | Kendall τ |
|---|---|---:|---:|---:|---:|---:|
| StreetView | 0 / 0，合并SFT | 162 | 63.28% | 81.92% | 77.44% | 0.7718 |
| StreetView | 17 / 68 | 176 | 68.75% | 83.43% | 81.15% | 0.8083 |
| StreetView | 34 / 136 | 173 | 67.58% | 83.93% | 80.47% | 0.8016 |
| StreetView | 51 / 204 | 176 | 68.75% | 84.93% | 82.52% | 0.8351 |
| **StreetView** | **68 / 272，选中** | **177** | **69.14%** | **84.77%** | **81.45%** | **0.8103** |
| Satellite | 0 / 0，正式fresh合并SFT | 211 | 82.42% | 92.52% | 86.52% | 0.8482 |
| Satellite | 17 / 68 | 205 | 80.08% | 92.41% | 86.04% | 0.8638 |
| Satellite | 34 / 136 | 212 | 82.81% | 91.91% | 87.79% | 0.8569 |
| **Satellite** | **51 / 204，选中** | **226** | **88.28%** | **94.70%** | **91.55%** | **0.9071** |
| Satellite | 68 / 272 | 224 | 87.50% | 93.86% | 91.21% | 0.9012 |

StreetView选中保存点比合并起点多正确15条（+5.86 pp），但只比cycles17/51多1条，且cycle51的部分结构指标更高。Satellite选中保存点比合并起点多正确15条（+5.86 pp），cycle68则少于cycle51两条。未进行显著性检验，不据少量probe差异认定保存点有显著优势。OOD不参与选择。

## 3. 完整Validation与OOD

### 3.1 选中模型的总体排序指标

| 实验 / Split | N | EM | Directed adjacency | Position | Pairwise | Kendall τ |
|---|---:|---:|---:|---:|---:|---:|
| StreetView cycle68 / validation | 1,445 | 69.69% | 84.78% | 81.83% | 90.42% | 0.8084 |
| StreetView cycle68 / OOD | 3,525 | 68.20% | 84.14% | 80.51% | 89.83% | 0.7966 |
| Satellite cycle51 / validation | 1,684 | 82.90% | 92.70% | 87.73% | 93.53% | 0.8706 |
| Satellite cycle51 / OOD | 4,299 | 81.76% | 92.12% | 87.42% | 93.17% | 0.8633 |

总体为全部轨迹的样本平均；无效排列保留在分母中。Validation由Toronto/Vienna组成，与固定probe的范围不同。

### 3.2 合并SFT起点与后训练的全量OOD对照

| 实验 / 模型 | EM正确数 / N | EM | Adjacency | Position | Pairwise | Kendall τ |
|---|---:|---:|---:|---:|---:|---:|
| StreetView / 合并SFT-500 | 2248/3525 | 63.77% | 81.11% | 77.10% | 87.67% | 0.7533 |
| StreetView / cycle68 | 2404/3525 | 68.20% | 84.14% | 80.51% | 89.83% | 0.7966 |
| Satellite / 合并SFT-700 | 3358/4299 | 78.11% | 90.20% | 84.48% | 91.37% | 0.8273 |
| Satellite / cycle51 | 3515/4299 | 81.76% | 92.12% | 87.42% | 93.17% | 0.8633 |

各模态对照使用相同合并底座、完整ID/GT/shuffle/prompt、图片处理、vLLM0.29/PyTorch2.13和greedy解码；起点关闭RL adapter。StreetView已有cycle68 OOD为六卡分片、起点补测为三卡；Satellite两次都为GPU0/2/3。提交batch、活跃并发和token预算按各自模态保持一致，但不宣称逐请求调度完全相同。

历史原始base+SFT adapter OOD为StreetView2269/3525（64.37%）、Satellite3394/4299（78.95%）；其合并状态和生成环境不同，不能代替实际训练起点。历史完整SFT记录见 [SFT Results](../SFT/sft_results.md)。

### 3.3 分城市结果

| 实验 | OOD城市 | N | 合并SFT正确数 / EM | 后训练正确数 / EM |
|---|---|---:|---|---|
| StreetView | Berlin | 743 | 469 / 63.12% | 532 / 71.60% |
| StreetView | Buenos Aires | 891 | 622 / 69.81% | 668 / 74.97% |
| StreetView | Melbourne | 668 | 401 / 60.03% | 413 / 61.83% |
| StreetView | Osaka | 493 | 296 / 60.04% | 306 / 62.07% |
| StreetView | Philadelphia | 730 | 460 / 63.01% | 485 / 66.44% |
| Satellite | Berlin | 718 | 573 / 79.81% | 610 / 84.96% |
| Satellite | Buenos Aires | 984 | 680 / 69.11% | 727 / 73.88% |
| Satellite | Melbourne | 953 | 827 / 86.78% | 844 / 88.56% |
| Satellite | Osaka | 754 | 612 / 81.17% | 649 / 86.07% |
| Satellite | Philadelphia | 890 | 666 / 74.83% | 685 / 76.97% |

完整validation分城市EM：StreetView Toronto518/749（69.16%）、Vienna489/696（70.26%）；Satellite Toronto739/931（79.38%）、Vienna657/753（87.25%）。

### 3.4 输出质量与实测耗时

| OOD模型 | 排列有效率 | CoT格式有效率 | 截断率 | 平均tokens |
|---|---:|---:|---:|---:|
| StreetView合并SFT | 95.01% | 97.56% | 2.44% | 993.60 |
| StreetView cycle68 | 96.40% | 97.45% | 2.55% | 980.3 |
| Satellite合并SFT | 97.60% | 99.77% | 0.23% | 808.75 |
| Satellite cycle51 | 98.28% | 99.74% | 0.26% | 804.27 |

| 耗时观测 | 实测 |
|---|---:|
| StreetView正式三卡cycle1–51中位数 | 32.27分钟/cycle |
| StreetView正式四卡cycle52–67中位数 | 26.50分钟/cycle |
| Satellite完整cycle1–69中位数 | 30.68分钟/cycle |
| StreetView cycle68完整validation / OOD | 26.81 / 31.46分钟（六卡） |
| Satellite cycle51完整validation / OOD生成阶段 | 19.99 / 52.37分钟（三卡） |
| 合并SFT OOD生成：StreetView / Satellite | 44.19 / 41.18分钟（三卡顺序执行） |

训练耗时来自实际完整cycle日志，包含不同响应长度、资源竞争和保存开销；StreetView cycle68暂停发生在保存后、最终cycle日志打印前，没有为它补造耗时。上述配置、样本数和输出长度不同，不作为严格硬件加速比对照。

## 4. 验收、证据与图件复现

### 4.1 完整性

选中保存点LoRA/Adam、独立RNG/scheduler及数据游标读回通过。固定probe、完整validation/OOD均有完整唯一ID、原始回答和严格重评分；报告与summary一致，prompt/split/model/adapter/source哈希核验通过，推理worker实际退出码0。

合并SFT补测共7,824条原始回答，所有权重分片与冻结源码/数据在结束时复验未变。原协调器在Satellite生成完成后因参考protocol路径错误退出；独立CPU验收修正路径后通过，保留原异常，未重复推理或更改生成结果。两项联合独立验收为passed。

### 4.2 权威结果路径

ROOT为 `/path/to/visual_trajectory`。SV_RUN为ROOT下 `runs/streetview_grpo_opsd_qwen3vl8b_lora_v1`，SAT_RUN为 `runs/satellite_grpo_opsd_qwen3vl8b_lora_v1_gpu023`。

| 记录 | 服务器位置 |
|---|---|
| StreetView最终状态/保存验收 | SV_RUN下 `final_result.json`；四卡正式分支下 `pause_checkpoint_verification.json` |
| StreetView probe / validation / OOD | 四卡正式分支下 `probe_evaluation_step68/`、`validation_cycle68_gpu2_7/`、`ood_cycle68_gpu2_7_fast/` |
| Satellite受控停止 | SAT_RUN下 `provision/controlled_stop_for_cycle51_evaluation.json` |
| Satellite正式probe | 正式分支下 `probe_evaluations/cycle_XXX/`；独立核验在run的provision目录 |
| Satellite选中模型全量评测 | SAT_RUN下 `evaluation/cycle_051_full/acceptance.json` 与 `independent_final_verification.json` |
| 合并SFT全量OOD与独立验收 | ROOT下 `runs/merged_sft_ood_20261014/{streetview,satellite}/`，根目录 `acceptance.json`、`independent_final_verification.json` |
| 原始轨迹回答 | 各评测目录 `predictions.jsonl`、`summary.json`、`report.json` |

### 4.3 图源与复现

外部 `grpo_opsd_plot_data.json`（不随公开代码上传）保留实际有效分支的逐update/逐cycle指标、probe、最终评测及服务器源文件SHA-256，用于复现；没有保留本地原始response副本。

图源从服务器只读导出，StreetView严格拼接三卡cycles1–51与四卡52–68，排除旧三卡updates205–214及所有预检/失败试跑。采集脚本为 [collect_grpo_opsd_plot_data.py](../../code/training/collect_grpo_opsd_plot_data.py)，绘图脚本为 [plot_grpo_opsd_results.py](../../code/training/plot_grpo_opsd_results.py)。本地复现：

```powershell
python code/training/plot_grpo_opsd_results.py --source training/OPD/figures/grpo_opsd_plot_data.json --output training/OPD/figures
```

当前两项训练均已停止，所有请求的评测已完成，监测自动化已暂停。原完整epoch计划不作为已完成训练结果报告。
