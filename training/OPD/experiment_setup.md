# StreetView 与 Satellite：GRPO + OPSD Experiment Setup

整理日期：2026-10-14。本文记录两项实验实际采用的正式配置；训练进度、checkpoint选择、评测和训练动态图见 [Results](results.md)。两项训练均按用户要求提前停止，未完成原计划的完整epoch。

## 1. 任务、初始化与数据

### 1.1 任务与模型

Qwen3-VL-8B-Instruct接收同一轨迹的8张乱序图片，生成既有四步CoT和 `<answer>` 排列，恢复时间顺序。StreetView与Satellite分别从各自最佳SFT模型启动，互不继承权重、RL adapter或optimizer状态。

| 项目 | StreetView | Satellite |
|---|---|---|
| 初始SFT | checkpoint-500 | checkpoint-700 |
| 合并底座 | `streetview_sft_checkpoint500_single_round_bf16` | `satellite_sft_checkpoint700_single_round_bf16` |
| 全部训练轨迹 | 16,704条，23城 | 17,452条，23城 |
| 训练筛选 | 全部train，不限制为合格CoT子集 | 全部train，不限制为合格CoT子集 |
| 图片 | 8张1024×512 JPEG / 轨迹 | 8张768×768 JPEG / 轨迹 |
| Processor grid / 每轨迹视觉tokens | 每图 `[1,32,64]` / 4,096 | 每图 `[1,44,44]` / 3,872 |
| 固定validation probe | 256条，Toronto/Vienna各128条 | 256条，Toronto/Vienna各128条 |
| 完整validation | 1,445条 | 1,684条 |
| 完整OOD | 3,525条，五城 | 4,299条，五城 |
| 原计划 | 174 cycles、696 optimizer updates / rank | 182 cycles、728 optimizer updates / rank |

OOD五城为Philadelphia、Buenos Aires、Berlin、Osaka、Melbourne。固定manifest保存ID、城市、图片路径、shuffle、GT与split；GT为呈现顺序到原时间顺序映射的逆排列。保留原SFT prompt与图像processor：patch16、merge2、min_pixels65,536、max_pixels524,288。训练/validation/OOD无ID交集，probe是完整validation的固定子集。

### 1.2 合并权重与可训练参数

合并方式为 `BF16(FP32(base) + FP32(B @ A) × alpha/rank)`，合并权重实际读回验证。Student在合并SFT底座上新建零增量RL LoRA，重新初始化Adam与scheduler；冻结teacher和reference使用同一初始合并SFT底座。

| 项目 | 两项共同配置 |
|---|---|
| LoRA | rank64、alpha128、dropout0、bias none |
| 适配范围 | 语言层 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| 可训练规模 | 252个模块、504个FP32 LoRA tensors、174,587,904个参数 |
| 冻结范围 | 全部基础参数，包括视觉encoder、merger/DeepStack、embedding、lm_head |
| 精度 | 基础权重/推理BF16；LoRA与Adam moments FP32；不量化 |

## 2. GRPO + OPSD 联合目标

### 2.1 GRPO与reference正则

每条轨迹采样8个student回答。Reward为完整排序Exact Match：正确1、错误0；采用组内标准差归一化优势、vanilla PPO clipping（0.2）。每组奖励相同时task优势为0，但保留这些回答供OPSD与reference正则使用。

Reference固定为合并SFT，读取不含GT的student上下文。启用loss中的low-variance reference KL，系数0.001；关闭reward中的KL，避免重复施加。

### 2.2 Frozen SFT teacher与GT privilege

Teacher使用与reference相同的冻结SFT底座，额外接收以下模板，并对student已经采样的tokens计算条件log-prob。Student rollout和评测都不接收GT。

```text
You are provided with the following verified ground-truth information to help solve this task:
### The correct chronological order of the presented image labels is {GT_ORDER}. ###

Use the ground truth solely as internal guidance. Reason as if you had independently derived the answer, presenting a natural and coherent reasoning process. Do not mention, cite or allude to the ground truth. Do not imply that the answer was provided to you in advance.
```

冻结模板SHA-256：`6c19e1c06dc8b07953dc0a5bd9d8ff4e43fa94f9a25cc0d5252e309d1b75fd92`。

采用verl原生 **k1 policy-gradient distillation**，`use_policy_gradient=true`、`policy_loss_mode=vanilla`、clipping0.2、`loss_max_clamp=10`，固定OPSD系数0.01。Teacher只提供student采样token的log-prob，不生成额外训练回答；student/teacher按response内token位置对齐，额外GT导致的prompt长度差单独处理。

### 2.3 联合优化与归一化

\[
L = L_{\mathrm{GRPO}} + 0.01L_{\mathrm{OPSD}} + 0.001L_{\mathrm{reference\ KL}}.
\]

三项采用同一有效response mask，在同一次梯度累积中相加后更新LoRA。Token-mean的分母是本次optimizer minibatch的有效response tokens，跨micro-batch、跨rank使用一致的真实token总数。系数固定，不引入融合网络、门控、可学习组合权重或reward model。这里的OPSD实现是上述PG-OPD设置，不等同于全词表KL蒸馏。

## 3. 正式训练配置与执行方式

### 3.1 共同超参数

| 参数 | 实际正式值 |
|---|---:|
| `train_batch_size` | 96条轨迹 |
| `rollout.n` | 8 |
| 每完整rollout cycle | 768个回答 |
| `ppo_mini_batch_size` | 24条轨迹，对应192个回答 |
| `ppo_epochs` | 1 |
| Actor micro-batch | 每卡1个回答 |
| 每cycle optimizer updates | 4，每个回答在该cycle使用一次 |
| Learning rate / scheduler | 1e−6 / constant，无warmup |
| Optimizer | AdamW，betas(0.9,0.999)、eps1e−8、weight_decay0.01 |
| Gradient clipping | 1.0 |
| Gradient checkpointing | non-reentrant；训练forward关闭KV cache |
| Rollout sampling | temperature1、top_p1、top_k−1；penalties 1/0/0 |
| Student / teacher prompt上限 | 5,120 / 5,376 tokens |
| Response / vLLM context上限 | 3,072 / 9,216 tokens |
| 推理结束条件 | stop151645；ignore_eos=True |
| vLLM每卡 | TP1、maxseq16、batched tokens8192、memory0.45 |
| 随机种子 | 20261009 |

`ppo_mini_batch_size=24`在锁定verl实现中再乘 `rollout.n=8`，实际为192 responses/update。四次更新共用本批采样时的old-policy log-prob、reference与teacher分数；全部更新结束后同步LoRA，再生成下一批。Old-policy采用student vLLM返回的raw log-prob；当前采样T1/p1/k−1不进行top-k/top-p截断。

### 3.2 多卡部署与阶段顺序

| 实验 | 实际正式GPU / DDP部署 |
|---|---|
| StreetView cycles1–51 | A800 80GB ×3，物理GPU1/2/3，rank顺序2/3/1 |
| StreetView cycles52–68 | A800 80GB ×4，物理GPU0/1/2/3，rank顺序0/2/3/1；从已验收cycle51/update204恢复 |
| Satellite | A800 80GB ×3，物理GPU0/2/3 |

Actor使用实际Torch DDP/NCCL，每卡复制模型并同步梯度。三卡每次更新各64回答；四卡各48回答，global batch及192-response optimizer minibatch不变。

每卡一个共享冻结SFT vLLM副本，依次执行：student开启RL adapter生成回答 → reference关闭RL adapter、无GT评分 → teacher关闭RL adapter、含GT评分。阶段之间等待全部请求完成并清除旧策略cache。随后vLLM sleep，DDP actor更新；actor基础权重常驻GPU，Adam状态在更新阶段之外卸载CPU。全部4次更新后恢复vLLM并同步新LoRA。

StreetView换卡时保留cycle51模型、Adam、scheduler与数据位置；旧三卡未保存的updates205–214被重算，不能混入有效训练曲线。新增rank使用独立RNG状态。

### 3.3 Satellite计划尾批与保存

Satellite完整epoch原计划最后76条轨迹、608个真实回答。若执行尾批，仅actor dispatch加1条零loss占位，分三rank；真实minibatch为192/192/192/32，最后一批rank真实数11/11/10。占位mask/优势/梯度为0，不新增生成请求或真实轨迹。此机制经过独立预检；正式训练提前停止，未到达尾批。

两项均每17 cycles保存，计划最后一个cycle额外保存。实际有效保存点均为17/34/51/68。保存canonical LoRA/Adam、各rank独立RNG/scheduler及dataloader游标，并读回验证一致性。StreetView候选probe在暂停后统一生成；Satellite每次保存后同步刚保存的LoRA并立即评测固定probe。Cycle0使用实际合并SFT起点，预检或历史未合并SFT分数不代替它。

## 4. 模型选择与生成评测

### 4.1 选择规则与固定协议

候选按固定256条validation probe的EM排序，同分比较directed adjacency，再选较早checkpoint；OOD不参与选择。最终采用StreetView cycle68/update272、Satellite cycle51/update204。

所有生成评测greedy temperature0、top_p1、top_k−1、n1，最多3,072 tokens，仅stop151645、ignore_eos=True；使用同一模态冻结ID/GT/shuffle/prompt/processor。无效排列仍保留分母，pairwise/adjacency/position计0，Kendall τ计−1。

### 4.2 全量评测实际配置

| 项目 | StreetView cycle68 | Satellite cycle51 | 合并SFT起点OOD补测 |
|---|---|---|---|
| GPU / TP | GPU2–7，各TP1 | GPU0/2/3，各TP1 | GPU0/2/3，各TP1，两模态顺序执行 |
| Validation提交batch / maxseq / memory | 4 / 4 / 0.4 | 96 / 16 / 0.6 | 本次只补测OOD |
| OOD提交batch / maxseq / memory | 32 / 16 / 0.6 | 96 / 16 / 0.6 | StreetView32、Satellite96 / 16 / 0.6 |
| Context / batched tokens | 8,192 / 8,192 | 9,216 / 8,192 | 各模态沿用对应设置 |
| 推理模型 | 合并SFT + RL LoRA | 合并SFT + RL LoRA | 合并SFT，adapter=null，无GT |

所有worker实际退出码0后，合并分片并核对完整唯一ID、原始token IDs解码、逐条重评分与summary一致；验证prompt/split/model/adapter/source哈希。两项合并SFT全量OOD已补齐。StreetView原后训练OOD为六卡、本次起点为三卡，分片调度不完全相同；表中提交batch是队列深度，maxseq才是活跃并发上限。

## 5. 技术栈与复现路径

### 5.1 锁定环境

Python3.12.15、PyTorch2.13.0+cu130、Transformers5.12.1、PEFT0.19.1、Ray2.55.1、vLLM0.29.0、FlashAttention2.8.3、OmegaConf2.3.1/Hydra1.3.3。训练使用verl原生GRPO与PG-OPD，锁定源码commit `704a9f01bda8b9863c02c113547191bf167bc88f`（包版本0.10.0.dev0），项目代码适配实际DDP与串行共享推理服务。

### 5.2 权威文件位置

服务器：`training-host`，用户 `training_user`。ROOT：`/path/to/visual_trajectory`；环境：`/path/to/workspace/conda-envs/streetview_verl_grpo_opsd_py312`。

| 内容 | ROOT下路径 |
|---|---|
| 冻结数据/prompt/teacher模板 | `training/{streetview,satellite}_grpo_opsd_qwen3vl8b_lora_v1/` |
| 合并底座 | `models/{streetview_sft_checkpoint500,satellite_sft_checkpoint700}_single_round_bf16/` |
| StreetView run | `runs/streetview_grpo_opsd_qwen3vl8b_lora_v1/` |
| StreetView有效三卡 / 四卡分支 | 上述run下 `deployment/serial_ddp/ddp_resident/formal/` 与 `deployment/serial_ddp4/ddp_resident/formal/` |
| Satellite run | `runs/satellite_grpo_opsd_qwen3vl8b_lora_v1_gpu023/` |
| Satellite正式分支 | 上述run下 `deployment/ddp_resident/formal/` |
| 正式配置/实现 | `code/training/grpo_opsd/serial_ddp/`、`code/training/satellite_grpo_opsd3/`；运行冻结参数及四卡迁移记录保存在对应run |
| verl源码 | `runtime/verl/` |
| 选中adapter | 各正式分支下 `global_step_68/actor/` 或 `global_step_51/actor/` |

失败预检、部署试跑与未保存回滚分支不计入正式结果；复现以实际run的冻结配置、保存状态和验收文件为准。
