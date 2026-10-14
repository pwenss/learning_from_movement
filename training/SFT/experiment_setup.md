# Street View 与 Satellite：Qwen3-VL-8B-Instruct SFT Experiment Setup

记录日期：2026-10-08。本文汇总两项已完成实验的最终配置；训练和评测结果见 [SFT Results](sft_results.md)。

## 1. 实验目标与模型来源

### 1.1 任务与独立实验

模型接收一条轨迹的 8 张打乱顺序图片，生成四步 CoT 及 `<answer>` 中的图片排序，恢复原始行进顺序。Street View 和 Satellite 分别训练独立 LoRA，均从同一原始 **Qwen3-VL-8B-Instruct** 初始化；Satellite 未加载 Street View adapter 或训练状态。

| 项目 | Street View | Satellite |
| --- | --- | --- |
| Experiment ID | `streetview_sft_qwen3vl8b_lora_v1` | `satellite_sft_qwen3vl8b_lora_v1` |
| 输入类型 | 街景图片序列 | 卫星图片序列 |
| 初始化 | 原始 base model + 新建 LoRA | 原始 base model + 新建独立 LoRA |
| 训练轮数 / 更新次数 | 2 epochs / 626 updates | 2 epochs / 752 updates |

两项实验使用各自的数据、prompt 与冻结排列；两种模态的轨迹不逐条配对。

### 1.2 运行资源

| 项目 | 最终使用配置 |
| --- | --- |
| 服务器 / 用户 | `training-host` / `training_user` |
| 项目根目录 ROOT | `/path/to/visual_trajectory` |
| 基础模型目录 | `/path/to/workspace/models/Qwen3-VL-8B-Instruct` |
| 基础模型 revision | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| 显卡 | 物理 GPU 1、2：2 × NVIDIA A800 80GB PCIe |
| 训练方式 | 两进程 DDP / NCCL，每个 rank 一张卡 |

GPU 1 同时存在既有任务。本实验的训练和推理用时受共享负载影响；不将两卡显存视为一块共享内存。

## 2. 数据与监督

### 2.1 城市划分与样本数

训练使用 23 城；Toronto 和 Vienna 整城留出做 validation；Berlin、Buenos Aires、Melbourne、Osaka、Philadelphia 为五城 OOD。各实验内部的 train、validation、OOD 轨迹 ID 无交集。

| 集合 | Street View | Satellite | 用途 |
| --- | ---: | ---: | --- |
| 全部轨迹 | 21,674 | 23,435 | 原始正式数据 |
| 全部图片 | 173,392 | 187,480 | 每轨迹 8 张 |
| SFT train | 9,999 | 12,021 | 23 城合格 teacher CoT，参与更新 |
| Loss validation | 932 | 1,143 | 两个验证城市的合格 CoT，计算监督 loss |
| Generation validation | 1,445 | 1,684 | 两个验证城市全量轨迹，包含 teacher 不合格样本 |
| Generation probe | 256 | 256 | 上一集合的固定子集，选择 checkpoint |
| OOD test | 3,525 | 4,299 | 五城全量轨迹，统一比较 base 与 SFT |

| 验证城市 | Street View：loss / generation | Satellite：loss / generation |
| --- | ---: | ---: |
| Toronto | 480 / 749 | 581 / 931 |
| Vienna | 452 / 696 | 562 / 753 |

Probe 在两个验证城市各固定取 128 条，以 `SHA256(seed + city + trajectory_id)` 排序抽样，seed 为 `20261006`；抽样不依赖 teacher 答案正确性。Generation validation 和 OOD 不按 teacher 合格性筛选。

### 2.2 Teacher 筛选与监督目标

训练及 loss validation 仅使用冻结的合格 teacher 记录：API 成功、答案完全正确、严格四步 CoT 格式、正常 `stop`、think tokens 严格小于 2,048。

监督目标是 teacher 的完整 **`content`**，包含 `<think>` 四步推理和 `<answer>` 排列；不使用 provider 的 `reasoning_content`。采用模型 chat template，仅对 assistant 正文与结束 token `<|im_end|>` 计算 causal cross-entropy；system、user、图片 token、assistant header 和 padding 的 labels 均为 `-100`。

### 2.3 图片排列与信息边界

设 `s[k]` 为呈现图片 `Image k` 对应的原始帧编号，则正确答案为 `argsort(s)`。按冻结 `s` 读取 8 张图片，使用各模态冻结的 system/user prompt，以及 `Image 0` 至 `Image 7` 标签。原始帧编号、文件名、坐标、轨迹 ID 和正确顺序不进入模型文本输入。

每个 epoch 只重排训练样本，单条轨迹内部图片排列固定。不开启图片裁切、翻转、颜色增强、contact sheet、video 输入或样本 packing，避免 CoT 引用的图片标签与视觉输入失配。

### 2.4 图片与序列预算

| 项目 | Street View | Satellite |
| --- | ---: | ---: |
| 磁盘 JPEG 尺寸（宽 × 高） | 1024 × 512 | 768 × 768 |
| 模型实际处理尺寸（宽 × 高） | 1024 × 512 | 704 × 704 |
| 每张 `image_grid_thw` | `[1,32,64]` | `[1,44,44]` |
| 每张 / 每轨迹视觉 tokens | 512 / 4,096 | 484 / 3,872 |
| `min_pixels` / `max_pixels` | 65,536 / 524,288 | 65,536 / 524,288 |
| 完整 input + target 最大长度（实测） | 6,786 tokens | 6,718 tokens |
| 完整序列上限 | 8,192 tokens | 8,192 tokens |

Satellite 在相同像素预算下由 processor 处理为 704 × 704；磁盘图片仍为 768 × 768。完整序列长度经过核验，禁止静默截断图片、推理正文或答案。

## 3. SFT 训练配置

### 3.1 共同超参数

| 参数 | 最终值 |
| --- | --- |
| 实现 | Transformers Trainer + PEFT + 多模态 collator |
| 精度 / attention | BF16 mixed precision / FlashAttention-2；训练 TF32 开启 |
| LoRA | rank 64、alpha 128、dropout 0.05、bias none |
| LoRA 范围 | 语言模型 36 层的 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| 可训练规模 | 252 个模块、504 个 FP32 LoRA tensor、174,587,904 个参数 |
| 冻结范围 | 全部基础参数，包括视觉 encoder、merger/DeepStack、embedding、lm_head |
| Train / loss eval micro-batch | 每卡 1 条完整轨迹 |
| Gradient accumulation | 16 |
| 有效训练 batch | 通常 32 条轨迹（256 张图片）；epoch 尾部保留不足完整累积组的更新 |
| Learning rate | `5e-5` |
| Optimizer | AdamW，β=(0.9,0.999)、eps=`1e-8`、weight decay 0.01；bias/norm 不衰减 |
| Scheduler / warmup | Cosine / 3%，分别按本实验总步数建立 |
| Gradient clipping | `max_grad_norm=1.0` |
| Gradient checkpointing | 开启，non-reentrant；训练 `use_cache=False` |
| 序列处理 | 动态 padding，max sequence 8,192；packing 关闭 |
| DataLoader | 每 rank 4 workers，pin memory，prefetch 2 |
| 随机种子 | `seed=data_seed=20261006` |
| 日志 | 每 10 optimizer updates 记录 loss、LR、grad norm |

### 3.2 步数、验证与保存

| 项目 | Street View | Satellite |
| --- | --- | --- |
| 每 epoch 更新次数 | 313 | 376 |
| 总更新次数 | 626 | 752 |
| 候选 checkpoint | 100、200、300、313、400、500、600、626 | 100、200、300、376、400、500、600、700、752 |
| Loss validation | 每 100 步及 epoch 末尾 / 最后一步 | 每 100 步及 epoch 末尾 / 最后一步 |
| 最终候选 generation | 全部训练完成后，统一 vLLM 协议 | 全部训练完成后，统一 vLLM 协议 |

保存的候选使用该步骤新计算的 validation loss。Street View 曾从本实验 checkpoint-100、checkpoint-300 接续，optimizer/scheduler/随机状态与核心配置已核验；最终 vLLM 候选评测全部在训练后重新生成。Satellite 正式训练从原始 base 新建 LoRA，`resume_from=None`。

### 3.3 锁定环境

| 组件 | 训练环境 | 生成评测环境 |
| --- | --- | --- |
| Python | 3.11 | 独立推理环境，完整版本见 lock |
| PyTorch | 2.8.0+cu128 | 2.10.0 |
| Transformers | 4.57.1 | 4.57.1 |
| PEFT / Accelerate | 0.17.1 / 1.10.1 | vLLM 原生 LoRA |
| FlashAttention | 2.8.3 | 由锁定的 vLLM 环境管理 |
| vLLM | — | 0.17.1 |
| Pillow / Tokenizers | 11.3.0 / 0.22.1 | 完整版本见 lock |

训练环境：`/path/to/workspace/conda-envs/streetview_sft_qwen3vl`。推理环境：`/path/to/workspace/inference-envs/streetview_vllm_qwen3vl`。各 RUN 的 `environment.lock.txt` 记录训练依赖；两项实验共享的推理环境 lock 位于 ROOT 的 `runs/streetview_sft_qwen3vl8b_lora_v1/vllm/environment.lock.txt`，各 RUN 的冻结生成协议记录实际推理设置。

## 4. 生成评测与 checkpoint 选择

### 4.1 vLLM 最终协议

| 参数 | 两项实验共同值 |
| --- | --- |
| 引擎部署 | 物理 GPU 1、2 各一个引擎，TP=1 |
| Base / LoRA 推理精度 | BF16，无量化；保存的 FP32 adapter 不改写 |
| 活跃并发 | 每卡最多 8 条轨迹，两卡合计 16 条 |
| 已提交请求上限 | 每卡 32；完成请求后持续补充，不等待整批完成 |
| `max_model_len` / `max_num_batched_tokens` | 8,192 / 8,192 |
| `gpu_memory_utilization` | 0.55 |
| 图片处理 | 每条 8 张；像素预算同训练；实际 grid 和完整 prompt token IDs 已对齐 |
| 缓存 | Prefix cache 关闭，multimodal processor cache 为 0 |
| 解码 | Greedy：temperature=0、top_p=1、top_k=-1 |
| 输出上限 | 3,072 tokens |
| Penalty | repetition=1，presence=0，frequency=0 |
| 结束条件 | 仅 token 151645（`<|im_end|>`），忽略额外 native EOS |
| 随机种子 / TF32 | 20261006 / 推理 TF32 关闭 |

### 4.2 模型选择流程

1. Base model 与每个候选 checkpoint 均生成同一固定 256 条 validation probe。
2. 依次比较 Exact Match 正确条数、正确有向邻接边数、较低 validation loss；完全相同保留较早候选。
3. 最佳 checkpoint 生成完整 validation，然后冻结用于最终 OOD 评测。
4. Base 和冻结 SFT checkpoint 在本模态同一全量 OOD 上比较，保持轨迹、图片、排列、prompt、processor 和解码参数相同。

最终比较只使用本次冻结 vLLM 协议的输出；旧后端、旧并发的分数不参与候选选择。OOD 不参与 checkpoint 选择。

### 4.3 评分口径

| 指标 | 含义 |
| --- | --- |
| Exact Match | 8 张图片的完整排列与 ground truth 完全一致 |
| Pairwise accuracy / Kendall τ | 28 对图片的相对顺序准确率 / 顺序相关性 |
| Directed adjacency | 7 条有向相邻边的准确率 |
| Position accuracy | 8 个位置的逐位置准确率 |
| 合法排列率 | 可解析为 0–7 各出现一次的答案比例 |
| CoT 格式率 | 符合冻结的四步 think/answer 严格格式的比例 |
| 截断率 / 输出长度 | 触及生成上限的比例 / 平均输出 tokens |

无效排列保留在全部指标分母中：Exact Match 计错，pairwise/adjacency/position 计 0，Kendall τ 计 −1。整体 micro average 按全部轨迹计算，城市 macro average 对城市指标等权平均。

## 5. 数据与复现入口

### 5.1 数据位置

以下相对路径均相对于 ROOT：`/path/to/visual_trajectory`。

| 内容 | Street View | Satellite |
| --- | --- | --- |
| 图片根目录 | `dataset/streetview/cities/` | `dataset/satellite/cities/` |
| 冻结 teacher 数据 | `dataset/streetview/cot/train_qwen38_flash_cot_20260927/` | `dataset/satellite/cot/satellite_cot_qwen38_flash_all_20261004/` |
| 派生训练 / 验证 manifest | `training/streetview_sft_qwen3vl8b_lora_v1/` | `training/satellite_sft_qwen3vl8b_lora_v1/` |
| 运行结果根目录 RUN | `runs/streetview_sft_qwen3vl8b_lora_v1/` | `runs/satellite_sft_qwen3vl8b_lora_v1/` |

Street View 图片路径由 `image_path_mapping.jsonl` 解析，不能直接读取 teacher 中旧 NAS 绝对路径。Satellite 中历史 `dataset/satellite/final/cities/` 引用只在派生加载时映射至当前图片目录。源图片、teacher、manifest 和原始城市划分未改写。

### 5.2 权威记录

各 RUN 下保留 `training/configuration.json`、环境 lock、派生来源哈希、`training/trainer_state.json`、`vllm/generation_protocol.json`、`vllm/checkpoint_selection.json`、逐条预测及 `final_audit.json`。训练和评测代码位于 ROOT 的 `code/training/`。

### 5.3 评测范围说明

Street View OOD 中 352 条轨迹曾用于 teacher 比较与选择，因此不能将全部 3,525 条描述为此前完全未使用的测试数据。Satellite 五城 teacher 输出未用于本次 SFT 或 checkpoint 选择。两项实验均未人工逐条审核全部 teacher 推理，自动筛选正确不等于每一处视觉描述均可靠；Street View teacher 原先使用 1920 × 960 图片，SFT 使用 1024 × 512。
