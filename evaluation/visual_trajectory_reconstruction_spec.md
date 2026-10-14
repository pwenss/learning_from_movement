# Visual Trajectory Reconstruction: Data, Splits, CoT and Prompts

本文件统一记录 `streetview` 与 `satellite` 两类任务的数据、划分、CoT 构建和输入 prompt。Satellite 最终数据筛选与重新编号核对日期为 **2026-10-04**；其他运行及筛选日期以各节说明为准。

## 0. Shared Task Protocol

### 0.1 Task Definition and Post-Training

输入为同一轨迹上打乱顺序的 8 张观察图，目标是恢复沿运动方向的空间顺序。`streetview` 使用朝向运动方向的街景观察；`satellite` 使用图像上方向与运动方向对齐的俯视观察，并允许尺度变化。两类任务使用各自的 system/user prompt，不能混用视觉约束。

后训练策略为 **CoT SFT + C3-OPSD**：先由 teacher 生成带解释的回答，经自动筛选后构建 SFT 数据；再进行 C3-OPSD。当前已完成 streetview 和 satellite 的 teacher 数据生成，本文不将尚未执行的模型训练写成已完成实验。

### 0.2 Input Labels and Leakage Control

每条轨迹作为不可拆分的数据单位，8 张图必须属于同一 split。先冻结 train/test 城市、样本及打乱映射，再构建模型请求。`Image 0` 至 `Image 7` 仅表示打乱后的呈现位置，答案使用这些呈现索引。

模型输入不包含城市名、轨迹 ID、原始帧号、原始顺序、坐标、时间戳或 ground truth。图片采用匿名对象名；原始路径、顺序及标签映射仅在离线 manifest 中保留，用于评分和复现。

### 0.3 Server Locations and Current Paths

路径记录更新于 **2026-10-06**。当前后训练数据副本位于 **`training-host`**，SSH 用户为 `training_user`，项目根目录为 **`/path/to/visual_trajectory`**。原服务器 `pangshuai_l20` 的 `/path/to/visual_trajectory` 保留；下表相对路径可用于两台服务器，本文后续的当前绝对路径均指新服务器。

| 内容 | 相对于项目根目录的当前路径 |
| --- | --- |
| Street View 数据（1024×512） | `dataset/streetview/` |
| Street View 图片 | `dataset/streetview/cities/<编号_city_id>/trajectory_sequences_180/<trajectory_id>/frame_XX.jpg` |
| Street View CoT | `dataset/streetview/cot/train_qwen38_flash_cot_20260927/` |
| Street View 评测及完整划分 | `dataset/streetview/evaluations/`、`dataset/streetview/split_manifest.json` |
| Satellite 数据及辅助文件 | `dataset/satellite/` |
| Satellite 图片（768×768） | `dataset/satellite/cities/<city_id>/trajectory_sequences/<trajectory_id>/frame_XX.jpg` |
| Satellite 城市索引 | `dataset/satellite/cities/<city_id>/index.json`、`cities/<city_id>/summary.json` |
| Satellite CoT 及冻结划分 | `dataset/satellite/cot/satellite_cot_qwen38_flash_all_20261004/`，含 `split_manifest.json` |
| 两类数据的来源记录 | `dataset/streetview/provenance/`、`dataset/satellite/provenance/` |
| 生成与核验代码 | `code/evaluation/src/` |
| Thinking token 计数用 tokenizer 缓存 | `runtime/cot/tokenizer.json` |
| Street View / Satellite 运行记录 | `runs/train_qwen38_flash_cot_20260927/`、`runs/satellite_cot_qwen38_flash_all_20261004/` |
| 配置及可视化代码 | `config/`、`viewer/` |

整个项目目录已于 2026-10-06 完成迁移，410,397 个文件逐文件 SHA-256、大小及权限校验通过，目录结构一致。项目目录外的 base model 未迁移；原模型来源仍为 `/path/to/trajectory_generation/model/Qwen3-VL-8B-Instruct`，缓存的 tokenizer 不等同于完整模型权重。

冻结 manifest、CoT、图片映射和历史报告中的绝对路径保留生成时记录，迁移未改写其内容；匿名 R2 URL 及呈现映射也保持不变。在新服务器读取本地图片时，应由加载程序根据上表解析当前路径：Street View 应读取 `image_path_mapping.jsonl` 中的 `images[].local_path`，并相对当前 `dataset/streetview/` 解析（如 `cities/01_nyc_new_york/trajectory_sequences_180/nyc_new_york__000003/frame_00.jpg`）；冻结 CoT 中的旧 NAS 路径不能仅通过替换绝对根前缀解析；Satellite 历史 `dataset/satellite/final/cities/` 或 `dataset/satellite/cities_1024/` 对应当前 `dataset/satellite/cities/`。历史路径仅用于溯源，不能假定这些旧目录仍存在。新项目根目录含空格，在命令行中需将完整路径用引号包裹。

---

## 1. Streetview

### 1.1 Trajectory Dataset

#### 1.1.1 Version and Generation Constraints

当前后训练数据根目录：`/path/to/visual_trajectory/dataset/streetview`。原始 1920×960 数据来源为 `/path/to/trajectory_generation/datasets/sv30_20260920_r100_r3`；当前目录内图片已转换为 1024×512，轨迹、帧和 pano 对应关系保持不变，转换设置和全量核验见 `dataset_info.json` 与 `validation_report.json`。

- 已完成 30 个城市，共 **21,674 条轨迹、173,392 张图片**，每条 8 张图。
- 原始生成图像为 `1920 x 960` JPEG；当前本地数据为 `1024 x 512` JPEG，采用 Lanczos 缩放、quality 92、4:4:4，不裁切。水平视野 180 度、垂直视野 90 度，pitch 为 0；相机中心方向与下一段运动方向对齐，第 8 帧沿用上一帧方向。
- 每条轨迹包含 8 个 pano，轨迹内及该街景数据集全局不重复；相邻距离严格处于 `(1, 30) m`，导航转角不超过 90 度。
- 日期精确到月份时要求年月一致，否则要求年份一致；8 帧均完成生成校验后才计入成功轨迹。
- 每城每轮采样 1,500 个查询点，最多 3 轮；同城累计查询点间距严格大于 100 m，查询半径为 100 m。每轮结束后，去重起点达到至少 1,000 个则停止追加。起点数量不等于最终轨迹数量。

完整生成规则见 [Street View 批量轨迹生成方案](../trajectory/streetview/docs/trajectory_generation.md)。

#### 1.1.2 City Counts by Continent

| 大洲 | 城市 | city_id | 轨迹数 | Split |
| --- | --- | --- | ---: | --- |
| 北美洲 | 纽约 | `nyc_new_york` | 841 | Train |
| 北美洲 | 旧金山 | `sf_san_francisco` | 389 | Train |
| 北美洲 | 芝加哥 | `chicago` | 774 | Train |
| 北美洲 | 费城 | `philadelphia` | 730 | OOD test |
| 北美洲 | 多伦多 | `toronto` | 749 | Train |
| 北美洲 | 温哥华 | `vancouver` | 686 | Train |
| 北美洲 | 墨西哥城 | `mexico_city` | 860 | Train |
| 南美洲 | 圣保罗 | `sao_paulo` | 1,119 | Train |
| 南美洲 | 里约热内卢 | `rio_de_janeiro` | 1,131 | Train |
| 南美洲 | 布宜诺斯艾利斯 | `buenos_aires` | 891 | OOD test |
| 欧洲 | 伦敦 | `london` | 674 | Train |
| 欧洲 | 巴黎 | `paris` | 823 | Train |
| 欧洲 | 柏林 | `berlin` | 743 | OOD test |
| 欧洲 | 马德里 | `madrid` | 943 | Train |
| 欧洲 | 罗马 | `rome` | 1,080 | Train |
| 欧洲 | 阿姆斯特丹 | `amsterdam` | 549 | Train |
| 欧洲 | 苏黎世 | `zurich` | 480 | Train |
| 欧洲 | 维也纳 | `vienna` | 696 | Train |
| 欧洲 | 斯德哥尔摩 | `stockholm` | 669 | Train |
| 欧洲 | 布拉格 | `prague` | 644 | Train |
| 亚洲 | 东京 | `tokyo` | 518 | Train |
| 亚洲 | 大阪 | `osaka` | 493 | OOD test |
| 亚洲 | 首尔 | `seoul` | 652 | Train |
| 亚洲 | 台北 | `taipei` | 671 | Train |
| 亚洲 | 新加坡 | `singapore` | 663 | Train |
| 亚洲 | 曼谷 | `bangkok` | 687 | Train |
| 亚洲 | 吉隆坡 | `kuala_lumpur` | 832 | Train |
| 大洋洲 | 悉尼 | `sydney` | 692 | Train |
| 大洋洲 | 墨尔本 | `melbourne` | 668 | OOD test |
| 大洋洲 | 奥克兰 | `auckland` | 327 | Train |
| **总计** | **30 城** | | **21,674** | |

| 大洲 | 城市数 | 全部轨迹 | Train 轨迹 | OOD test 轨迹 |
| --- | ---: | ---: | ---: | ---: |
| 北美洲 | 7 | 5,029 | 4,299 | 730 |
| 南美洲 | 3 | 3,141 | 2,250 | 891 |
| 欧洲 | 10 | 7,301 | 6,558 | 743 |
| 亚洲 | 7 | 4,516 | 4,023 | 493 |
| 大洋洲 | 3 | 1,687 | 1,019 | 668 |
| **总计** | **30** | **21,674** | **18,149** | **3,525** |

### 1.2 Train/Test Split

#### 1.2.1 City-Level OOD Holdout

固定将 **费城、布宜诺斯艾利斯、柏林、大阪、墨尔本** 五城的全部轨迹作为 OOD test，其余 25 城作为训练池。

| 集合 | 城市数 | 轨迹数 | 图片数 | 占全部轨迹 |
| --- | ---: | ---: | ---: | ---: |
| Train | 25 | 18,149 | 145,192 | 83.74% |
| OOD test | 5 | 3,525 | 28,200 | 16.26% |
| 总计 | 30 | 21,674 | 173,392 | 100% |

#### 1.2.2 Frozen 10% OOD Evaluation Subset

此前模型比较采用五城各自约 10% 的固定子集，共 **352 条轨迹、2,816 张图**。按城市内 SHA-256 排名抽取，样本数为各城总量乘 10% 后四舍五入，seed 为 `20260925`。所有模型复用相同样本与呈现映射。

| OOD 城市 | 完整 test 轨迹 | 10% 测评轨迹 |
| --- | ---: | ---: |
| 费城 | 730 | 73 |
| 布宜诺斯艾利斯 | 891 | 89 |
| 柏林 | 743 | 74 |
| 大阪 | 493 | 49 |
| 墨尔本 | 668 | 67 |
| **总计** | **3,525** | **352** |

该子集包含于完整 OOD test，不是独立于五城的额外测试集。它已用于比较和选择 teacher，后续实验应披露这一用途；若需要未参与选择的最终测试，应另行冻结未用于选择的样本。

冻结记录见 `Qwen3.8-Flash sample_manifest.json`（私有测评记录，不随公开代码上传）。

### 1.3 CoT Construction

#### 1.3.1 Teacher and Batch Settings

| 设置 | 实际配置 |
| --- | --- |
| Teacher | `qwen3.8-flash` |
| Provider | Alibaba Cloud DashScope Batch API |
| Run ID | `train_qwen38_flash_cot_20260927` |
| 输入样本 | 25 城训练池的全部 18,149 条轨迹 |
| Prompt | 与此前 Qwen3.8-Flash 测评一致的 streetview CoT system/user prompt，见 1.4.2 |
| Thinking | 开启；`thinking_budget=20000` |
| 输出上限 | `max_tokens=20000` |
| 打乱 seed | `20260927`；由 seed 与轨迹 ID 的 SHA-256 派生每条轨迹的固定排列 |
| Batch 单位 | 每城市 1 个主 Batch；API 失败项按同城补交，最多共 4 次尝试 |
| 图片存储 | Cloudflare R2，145,192 张图片已上传；对象文件名为随机 UUID，无城市、轨迹或顺序信息；保留图片 |

教师自由推断轨迹顺序，ground truth 不放入输入。正确性由生成后离线比较判定，不是向 teacher 提供正确顺序后让其解释。图片首次上传读取本地字节；提交及失败重试复用冻结 manifest 中的 URL，不重新上传图片。公开 URL 仅按城市抽查，不逐张重复下载。

#### 1.3.2 SFT CoT Selection Criteria

仅在 API 成功返回非空文本的前提下，以下条件必须**同时满足**，记录才标记为 `sft_eligible=true`：

| 条件 | 判定规则 |
| --- | --- |
| 顺序完全正确 | `<answer>` 中的答案为 `0` 至 `7` 的合法排列，且与 ground truth 完全一致；不接受部分正确 |
| CoT 格式合规 | 完整输出符合 `<think>...</think><answer>...</answer>`，标签外仅允许空白；只有一个 `<think>` 块，四个 Step 标题与冻结 prompt 的文字和顺序一致；标题匹配时将 U+2011 非断行连字符视为普通 `-` |
| 正常结束 | `finish_reason == "stop"`，排除截断输出 |
| Thinking 长度 | **`output_thinking_tokens < 2048`**； |

Thinking tokens 使用本地 SFT 模型 `Qwen3-VL-8B-Instruct` 的 `tokenizer.json` 计数，生成时来源路径为原服务器的 `/path/to/trajectory_generation/model/Qwen3-VL-8B-Instruct/tokenizer.json`；新服务器已迁移的同一 tokenizer 缓存位于 `/path/to/visual_trajectory/runtime/cot/tokenizer.json`，完整 base model 未迁移。对唯一 `<think>...</think>` 内的原始文本执行 `encode(..., add_special_tokens=False)`，保留首尾空白，包含 Step 标题及正文；不计 `<think>` 标签、`<answer>` 内容或 API 返回的 internal `reasoning_content`。该长度不是 DashScope 的计费 token 数，也不是完整 completion tokens。Tokenizer SHA-256 为 `a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7`。

筛选结果写入 `cot_data.jsonl` 的 `sft_eligible`，并记录 `output_thinking_tokens` 和 `thinking_length_valid`；无法提取唯一 think 块时，长度为 `null`，长度条件不通过。`summary.json` 记录 tokenizer、计数口径及长度阈值。上述条件仅验证答案、格式、结束状态和长度，尚未验证推理正文的视觉依据或逻辑质量。

#### 1.3.3 Generation Outcome and Artifacts

运行于北京时间 **2026-09-29 03:24** 结束。25 个主 Batch 加 20 个补交 Batch，共 45 个 Batch 均结束，无待处理 Batch；台北 1 条轨迹共尝试 4 次仍因 DashScope `data_inspection_failed` 失败，因此运行状态为 `incomplete`。

| 项目 | 数量 | 占训练池 18,149 条 |
| --- | ---: | ---: |
| 全量结果记录 | 18,149 | 100% |
| API 成功输出 | 18,148 | 99.994% |
| 有效答案 | 18,069 | 99.56% |
| CoT 格式合规 | 17,400 | 95.87% |
| 顺序完全正确 | 11,357 | 62.58% |
| 通过当前 SFT 筛选（含 thinking tokens < 2,048） | **10,931** | **60.23%** |

`cot_data.jsonl` 包含全部 18,149 条记录（包括失败），无重复 custom ID；10,931 是其中通过筛选的样本数，不是原始 train split 的轨迹数。生成于 2026-09-29 结束，筛选统计于 2026-10-03 按 1.3.2 的条件重新计算。

当前后训练 CoT 目录：`/path/to/visual_trajectory/dataset/streetview/cot/train_qwen38_flash_cot_20260927`。2026-10-06 从旧 NAS 的 `training/train_qwen38_flash_cot_20260927` 迁移，逐文件 SHA-256 校验，冻结 manifest、CoT 正文、评分、原始响应和 UUID URL 保持不变。既有 teacher 输出基于原始 1920×960 图片生成，并非对缩放后的图片重新调用 API。

| 文件/目录 | 用途 |
| --- | --- |
| `manifest.json` | 冻结训练样本、teacher 配置、完整 prompt、图片 URL 及打乱映射 |
| `batch_submission.json` | 各城市 Batch 提交记录 |
| `cot_data.jsonl` | 逐条 CoT、答案、reasoning、usage、评分及筛选标志 |
| `summary.json` | 总体及各城市统计 |
| `responses/<batch_id>.jsonl` | 完整原始 Batch 响应 |
| `image_path_mapping.jsonl` | 冻结原始图片路径/teacher URL 与当前 1024×512 本地帧路径及 SHA-256 的对应关系，保留打乱映射 |
| `split_manifest.json` | 本次 teacher 输入的训练子集，18,149 条 Train，0 条 OOD |
| `migration_verification.json` | 迁移完整性、CoT 唯一覆盖、当前图片映射及原有成功/失败数量的核验 |

R2 prefix 为 `sv30_20260920_r100_r3/training/train_qwen38_flash_cot_20260927`；运行记录确认图片保留。运行状态与日志副本当前位于 `/path/to/visual_trajectory/runs/train_qwen38_flash_cot_20260927`。原服务器旧 NAS 的 CoT、evaluations 和运行目录以符号链接兼容旧路径；这些旧 NAS 路径不属于新服务器的项目目录。复现既有结果应使用冻结 manifest 中的 prompt 与 UUID 映射；读取当前缩放图片时使用 `image_path_mapping.jsonl` 的帧及打乱映射，并按 0.3 节解析本地路径，不重新生成 CoT。

Street View 数据根目录的 `evaluations/` 保存此前 OOD 模型比较、完整结果、冻结抽样 manifest 和分片结果；根目录 `split_manifest.json` 覆盖全部 21,674 条轨迹，Train 18,149 / OOD 3,525。`provenance/` 保存原始生成配置、边界、采样记录和迁移文件哈希报告 `migration_20261006.json`。原有 1 条 API 失败记录仍保留，迁移不改变运行的 `incomplete` 结论。

### 1.4 Prompts

#### 1.4.1 Prompt w/o CoT

##### System Prompt

```text
You are a visual spatial reasoning assistant specializing in trajectory reconstruction from multiple observations.
```

##### User Prompt

Construct a single multimodal `user` message using the template below. Each `Image k:` label must be a text content block immediately before its corresponding image content block; `k` is the presentation position after shuffling, not the original frame number. Replace the image placeholders with actual image attachments, not literal placeholder text or original filenames. Include only the prompt text, displayed image labels, and image content; keep ground-truth order and original frame identifiers outside the model input.

```text
You are given 8 observation images sampled along a spatial trajectory. The images have been shuffled and are presented in an arbitrary order, labeled Image 0 through Image 7. Each label immediately precedes its corresponding image. The camera's central viewing direction in each image is aligned with the direction of movement along the trajectory. Reconstruct the spatial order in which these 8 observations would be encountered when following the trajectory.

Output using exactly the following format:

<answer>
[i0, i1, i2, i3, i4, i5, i6, i7]
</answer>

Image 0:
[Image content for presentation position 0]
Image 1:
[Image content for presentation position 1]
Image 2:
[Image content for presentation position 2]
Image 3:
[Image content for presentation position 3]
Image 4:
[Image content for presentation position 4]
Image 5:
[Image content for presentation position 5]
Image 6:
[Image content for presentation position 6]
Image 7:
[Image content for presentation position 7]
```

#### 1.4.2 CoT Prompt

##### Shared CoT System Prompt

```text
You are a visual spatial reasoning assistant specializing in trajectory reconstruction from multiple observations.

Analyze the provided images and produce a detailed, evidence-grounded explanation following the four steps specified in the user prompt. Base your conclusions on visible scene evidence and the stated camera-orientation constraint. Distinguish clear visual correspondences from uncertain matches, and do not invent visual details.

Refer to each observation by its displayed label, Image 0 through Image 7. These labels identify the order of presentation, not the spatial order along the trajectory.

Follow the required output format exactly. Within <think>, use all four step titles from the user prompt verbatim and in the specified order. Within <answer>, output only a list of eight integer indices, using each index from 0 to 7 exactly once. Do not include any text outside these two blocks.
```

##### User Prompt

Construct a single multimodal `user` message using the template below. Each `Image k:` label must be a text content block immediately before its corresponding image content block; `k` is the presentation position after shuffling, not the original frame number. Replace the image placeholders with actual image attachments, not literal placeholder text or original filenames. Include only the prompt text, displayed image labels, and image content; keep ground-truth order and original frame identifiers outside the model input.

```text
You are given 8 observation images sampled along a spatial trajectory. The images have been shuffled and are presented in an arbitrary order, labeled Image 0 through Image 7. Each label immediately precedes its corresponding image. The camera's central viewing direction in each image is aligned with the direction of movement along the trajectory. Reconstruct the spatial order in which these 8 observations would be encountered when following the trajectory.

Provide a detailed, evidence-grounded explanation using strictly the following four steps, followed by the final order.

Step 1: Global Scene Understanding
Provide a high-level overview of the environment and the spatial characteristics of the route depicted across the images. Focus on the overall character of the scene along the route rather than inferring their relative order at this stage.

Step 2: Cross-View Correspondence Identification
Identify visual correspondences across different observations by locating recurring landmarks that appear in multiple views and distinctive local cues that only appear in a subset of observations. Focus on correspondences that provide useful evidence for determining the trajectory order.

Step 3: Viewpoint Transformation Analysis
Compare observations linked by shared landmarks, focusing on potentially adjacent pairs. For each pair, analyze how the scene changes across observations by examining the transformations of corresponding landmarks, including spatial displacement, apparent scale, viewing angle, and visibility changes. Use these visual transformations to infer relative order under the forward-facing camera constraint. Check whether their viewpoint transitions are physically consistent and whether they form a continuous movement path.

Step 4: Final Sequence Determination
Combine the inferred pairwise spatial relationships to construct the final trajectory order, ensuring that every image index appears exactly once.

Output using exactly the following format:

<think>
Step 1: Global Scene Understanding
...
Step 2: Cross-View Correspondence Identification
...
Step 3: Viewpoint Transformation Analysis
...
Step 4: Final Sequence Determination
...
</think>

<answer>
[i0, i1, i2, i3, i4, i5, i6, i7]
</answer>

Image 0:
[Image content for presentation position 0]
Image 1:
[Image content for presentation position 1]
Image 2:
[Image content for presentation position 2]
Image 3:
[Image content for presentation position 3]
Image 4:
[Image content for presentation position 4]
Image 5:
[Image content for presentation position 5]
Image 6:
[Image content for presentation position 6]
Image 7:
[Image content for presentation position 7]
```

---

## 2. Satellite

### 2.1 Trajectory Dataset

#### 2.1.1 Independent Routes and Generation Rules

Satellite 在与 streetview 相同的 30 城行政边界内独立生成轨迹，拥有独立 ID，**不与 streetview 轨迹逐条配对**。每条包含 8 个观察中心、8 帧图像及 7 次移动，输出为 `1024 x 1024` JPEG。

当前正式生成方案为 `satellite_v4_extent_100_400_truncated_normal_dynamic_zoom`：

- 每帧覆盖旋转正方形，边长 extent 范围为 `[100, 400] m`。
- 相邻移动距离为当前 extent 的 `[0.2, 0.6]` 倍；从均值 0.4、标准差 0.2 的正态分布拒绝采样。
- 有符号 heading 变化从均值 0、标准差 30 度的正态分布截断到 `[-30, 30]` 度；最后一帧沿用前一帧 heading。
- 相邻 extent 变化从均值 0、标准差 50 m 的正态分布截断到 `[-50, 50] m`，同时保持 extent 的全局上下界。
- 每帧上方向与 heading 对齐；尺度变化可能来自观察高度变化，判断前进顺序应主要依据观察中心位置变化。
- 全部 footprint 位于城市边界内，中心连线不自交；瓦片 zoom 随 extent、纬度和分辨率动态选择。

完整规则见 [Satellite Generation Pipeline](../trajectory/satellite/docs/trajectory_generation.md)。

#### 2.1.2 Final Dataset after Landcover Filtering

原始生成数据曾完成：30 城各 1,000 条，共 **30,000 条轨迹、240,000 张图片**。2026-10-06 按用户要求清理原始 `cities_1024/` 及逐帧统计；生成汇总、筛选阈值分析和原始统计核验记录保留于 `/path/to/visual_trajectory/dataset/satellite/provenance/`，当前实验输入仅使用下述最终数据。

当前 Satellite 数据根目录为 **`/path/to/visual_trajectory/dataset/satellite`**，图片位于其 `cities/` 子目录。筛选发布时 dataset ID 为 `satellite_trajectory_v4_filtered_n1`；当前 768×768 副本的 `cities/dataset_info.json` 记录为 `satellite_trajectory_v4_filtered_n1_768`。历史 `final/` 路径已失效。以 2.1.3 的规则按整条轨迹筛选，排除 **6,565 条（21.88%）**，保留 **23,435 条轨迹、187,480 张图片（78.12%）**；每条仍包含完整的 8 帧。后续 Satellite 实验和网站使用当前 `cities/` 图片及 `cot/` 结果目录；图像转换设置和全量核验见 `cities/dataset_info.json` 与 `cities/validation_report.json`。

| 城市 | city_id | 最终轨迹 | 排除轨迹 | 最终图片 | 按五城规则的归属 |
| --- | --- | ---: | ---: | ---: | --- |
| 纽约 | `nyc_new_york` | 669 | 331 | 5,352 | Train |
| 旧金山 | `sf_san_francisco` | 310 | 690 | 2,480 | Train |
| 芝加哥 | `chicago` | 958 | 42 | 7,664 | Train |
| 费城 | `philadelphia` | 890 | 110 | 7,120 | OOD test |
| 多伦多 | `toronto` | 931 | 69 | 7,448 | Train |
| 温哥华 | `vancouver` | 858 | 142 | 6,864 | Train |
| 墨西哥城 | `mexico_city` | 987 | 13 | 7,896 | Train |
| 圣保罗 | `sao_paulo` | 667 | 333 | 5,336 | Train |
| 里约热内卢 | `rio_de_janeiro` | 646 | 354 | 5,168 | Train |
| 布宜诺斯艾利斯 | `buenos_aires` | 984 | 16 | 7,872 | OOD test |
| 伦敦 | `london` | 917 | 83 | 7,336 | Train |
| 巴黎 | `paris` | 919 | 81 | 7,352 | Train |
| 柏林 | `berlin` | 718 | 282 | 5,744 | OOD test |
| 马德里 | `madrid` | 793 | 207 | 6,344 | Train |
| 罗马 | `rome` | 767 | 233 | 6,136 | Train |
| 阿姆斯特丹 | `amsterdam` | 808 | 192 | 6,464 | Train |
| 苏黎世 | `zurich` | 729 | 271 | 5,832 | Train |
| 维也纳 | `vienna` | 753 | 247 | 6,024 | Train |
| 斯德哥尔摩 | `stockholm` | 848 | 152 | 6,784 | Train |
| 布拉格 | `prague` | 732 | 268 | 5,856 | Train |
| 东京 | `tokyo` | 859 | 141 | 6,872 | Train |
| 大阪 | `osaka` | 754 | 246 | 6,032 | OOD test |
| 首尔 | `seoul` | 777 | 223 | 6,216 | Train |
| 台北 | `taipei` | 572 | 428 | 4,576 | Train |
| 新加坡 | `singapore` | 413 | 587 | 3,304 | Train |
| 曼谷 | `bangkok` | 924 | 76 | 7,392 | Train |
| 吉隆坡 | `kuala_lumpur` | 938 | 62 | 7,504 | Train |
| 悉尼 | `sydney` | 993 | 7 | 7,944 | Train |
| 墨尔本 | `melbourne` | 953 | 47 | 7,624 | OOD test |
| 奥克兰 | `auckland` | 368 | 632 | 2,944 | Train |
| **总计** | **30 城** | **23,435** | **6,565** | **187,480** | |

#### 2.1.3 Landcover Criterion, Renumbering and Provenance

Landcover 来源为服务器本地的 **Dynamic World 2025 全年 annual mode、名义 10 m 分辨率**栅格。按每帧旋转正方形 footprint 内的像素中心统计全部 9 类 counts 和 shares，使用有效性掩膜区分有效 water（类别 0）与缺失像素；重叠 tile 不重复计数。

帧排除条件为：`valid_pixel_count > 0`，且存在一个 **built 以外的单一类别**，其 `class_count == valid_pixel_count`，即在有效像素中的占比精确为 100%。这不等同于所有非 built 类相加达到 100%；不对 shares 四舍五入后判定。Built 占比 100% 的帧不触发该条件。Shares 的分母仅为有效像素，缺失像素单独记录；无有效像素的帧不触发该条件。

最终保留数据中没有完全缺失的帧，16 帧含部分缺失像素；这些帧按上述有效像素口径参与判断，其缺失像素数随逐帧统计保留。

令 K 为一条轨迹中满足此条件的帧数。最终采用 **N=1：排除 K ≥ 1 的整条轨迹，仅保留 K=0 的轨迹**。不会只删除单帧或改变其余帧的时间顺序；各帧 extent、heading、坐标、图像字节及 ground truth 顺序保持生成时的值。

在每个城市内按原始 trajectory ID 升序排序保留轨迹，重新从 `000000` 连续编号，ID 格式仍为 `<city_id>__satellite_v4__<六位编号>`。帧文件名保持 `frame_00.jpg` 至 `frame_07.jpg`，`gt_order` 保持 `[0,1,2,3,4,5,6,7]`。目录、城市 index 和 trajectory manifest 中的 ID 同步更新；manifest 的 `source_trajectory_id` 保留原编号。最终 JPEG 最初通过同文件系统硬链接发布；删除原始目录后，最终文件仍完整保留。

下表记录筛选发布时的产物名称，历史 `final/` 目录已不存在；当前可读路径及保留情况以本节后文和 0.3 节为准。

| 筛选发布时的文件/目录 | 内容 |
| --- | --- |
| `index.json`、`summary.json` | 30 城最终索引、数据版本、筛选规则及各城统计 |
| `cities/<city_id>/index.json`、`summary.json` | 城市内连续编号的轨迹索引及筛选后数量 |
| `cities/<city_id>/trajectory_sequences/<new_id>/` | 新 ID manifest 和完整 8 帧 JPEG |
| `trajectory_id_mapping.jsonl` | 全部 30,000 条原始轨迹的保留/排除结果、原编号、新编号及触发条件的帧与类别；排除项的新编号为 null |
| `landcover_2025_mode_10m/<city_id>/frames.jsonl` | 保留帧的新旧 trajectory ID、9 类 counts/shares、有效/缺失像素数及 `matches_exclusion_condition=false` |
| `verification.json` | 全量验证：筛选判定、连续编号、manifest/index 一致性、图像来源一致性和完整帧统计 |

原始 240,000 帧的详细统计目录已于 2026-10-06 删除；其总体汇总、核验记录及各阈值分析分别保留于 `provenance/source_landcover_summary.json`、`source_landcover_verification.json` 和 `retention_by_threshold.json`。当前轨迹的新旧编号可在 `cities/<city_id>/index.json` 及轨迹 manifest 的 `source_trajectory_id` 中查阅；CoT 和原始 API 响应位于独立 `cot/` 目录。历史 `final/trajectory_id_mapping.jsonl`、`final/landcover_2025_mode_10m/` 和 `final/verification.json` 未在当前目录布局中找到，不作为现行加载入口。发布脚本仍位于 `tools/publish_filtered_satellite.py`，但原始完整输入已删除，不可直接重跑原始全量发布。最终数据的 manifest 和映射仅用于离线管理与评估，不加入模型输入。

当前数据根目录包含 `cities/`、`cot/`、`dynamic_world_2025_mode_10m/`、`provenance/`、`tools/` 和 `README.md`。使用 `cities/<city_id>/index.json` 与 `cities/<city_id>/summary.json` 获取城市轨迹，使用 `cities/dataset_info.json` 与 `cities/validation_report.json` 查看当前全量图片统计；不再使用 `final/index.json` 或 `final/summary.json`。`provenance/cleanup_report_20261006.json` 记录先前清理时的目录状态，其路径、inode 和大小属于历史核验口径；当前缩放副本以 `cities/validation_report.json` 为准。根目录 README 中仍出现的 `final/` 同样为旧路径。

### 2.2 Train/Test Split

Satellite 全量 CoT 运行沿用 streetview 的五城 OOD holdout：费城、布宜诺斯艾利斯、柏林、大阪、墨尔本；其余 25 城为训练池。已在本次 CoT 运行的 `split_manifest.json` 冻结按最终 ID 划分的城市归属，全部 30 城均生成 teacher 输出，OOD 输出不进入 SFT 导出。

| 口径 | Train 轨迹/图片 | OOD test 轨迹/图片 | 总轨迹/图片 |
| --- | ---: | ---: | ---: |
| N=1 筛选后的最终数据按五城规则归类 | **19,136 / 153,088** | **4,299 / 34,392** | **23,435 / 187,480** |
| 筛选前完整生成数据（溯源） | 25,000 / 200,000 | 5,000 / 40,000 | 30,000 / 240,000 |

最终数据已覆盖全部 30 城，本次 CoT 在独立 manifest 中冻结新 ID、城市归属和逐轨迹呈现映射；未额外划出 ID test 或 validation。不能套用 streetview 的 352 条样本 ID 或将两类数据视为配对样本。若增加同城 ID test/validation，需按地理区域留出并控制 footprint 重叠；随机按 trajectory ID 划分可能使高度重叠的卫星观察跨 split。

### 2.3 CoT Construction

#### 2.3.1 Full-Dataset Teacher Run

2026-10-04 启动筛选后最终数据的全量 CoT 运行，覆盖 **30 城、23,435 条轨迹、187,480 张图片**。运行于 **2026-10-05 23:53（Asia/Shanghai）完成**，23,435 条轨迹全部 API 成功；2026-10-06 00:28 完成独立全量核验，`verification.json` 的 `verified=true`。

共提交并收集 **53 个 Batch：30 个城市主 Batch、23 个失败或缺失项补交 Batch**，全部状态为 `completed`。全部图片上传及匿名 URL 映射保留，主 Batch 和补交 Batch 的原始响应均保留；最终结果不含 API 失败或缺失轨迹。

| 设置 | 本次配置 |
| --- | --- |
| Run ID | `satellite_cot_qwen38_flash_all_20261004` |
| Teacher / Provider | `qwen3.8-flash` / Alibaba Cloud DashScope Batch API |
| API endpoint | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| 输入数据 | `satellite_trajectory_v4_filtered_n1`，完整 23,435 条轨迹 |
| Prompt | **2.4.2 Satellite CoT system/user prompt**；与既有 Satellite prompt 校验一致 |
| Thinking / 输出 | `enable_thinking=true`、`thinking_budget=20000`、`max_tokens=20000`、`temperature=0` |
| 呈现打乱 | seed `20261004`；由 seed 与最终轨迹标识的 SHA-256 派生，每条固定排列 |
| 主 Batch | 每城 1 个，共 30 个；只对 API 失败或缺失项按城市补交，每条最多共 4 次尝试 |
| 图片上传 | 复用 Street View 的 R2 bucket 和 HTTPS 公网域名；32 并发，随机 UUID 文件名，无城市、轨迹或顺序标签 |
| R2 prefix | `satellite_trajectory_v4_filtered_n1/cot/satellite_cot_qwen38_flash_all_20261004` |
| API credential | 本次独立密钥文件的当前副本 `/path/to/visual_trajectory/config/satellite_cot_20261004.key`，权限 `0600`；不使用 Street View 密钥 |

Ground truth 只保存在离线 manifest，不加入 teacher 请求。请求正文仅含 Satellite prompt、`Image 0` 至 `Image 7` 的呈现标签和匿名图片 URL。每城上传完成后对首、中、末图片做公网字节校验，然后提交该城市 Batch；在其他城市上传过程中，已提交城市可由服务端执行。创建对象采用仅创建写入，断点恢复复用匹配原图 SHA-256 和长度的对象，保留已上传图片。

Batch 采用 `24h` completion window。图片上传、请求文件上传和 HTTP 轮询均有超时；任务在后台运行，日志及状态持久保存。提交接收情况不明时先按 input file ID 核对服务端 Batch，不重复提交接收不明的付费请求；限流及可恢复网络故障退避续跑。模型已成功输出但答案或 CoT 不合格的记录不因质量原因重复生成。

#### 2.3.2 Output Selection and Artifacts

复用 1.3.2 的 SFT 条件：API 成功、顺序完全正确、严格 CoT 格式、`finish_reason=stop`、`<think>` 内 thinking tokens **严格小于 2,048**。Step 标题使用 Satellite prompt。计数采用与 Street View 相同的 `Qwen3-VL-8B-Instruct` tokenizer，当前缓存路径为 `/path/to/visual_trajectory/runtime/cot/tokenizer.json`；SHA-256 为 `a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7`。计数保留 think 内首尾空白及 Step 标题，不含标签或 API 的 internal `reasoning_content`。

`cot_data.jsonl` 最终为全部 23,435 条轨迹保存结果或失败记录、呈现映射、原始 ID、正文、internal reasoning、usage 和评分。`sft_eligible` 表示输出满足上述质量规则；**`sft_export_eligible` 还要求 `split=train`**，因此五城 OOD 输出始终不进入 SFT 导出。该筛选不等同于已完成推理正文的视觉依据人工审查，也不代表已执行 SFT 或 GRPO。

当前输出目录：`/path/to/visual_trajectory/dataset/satellite/cot/satellite_cot_qwen38_flash_all_20261004`。冻结记录中出现的旧 `final/cot/` 路径仅作生成时溯源。

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 全量样本、新旧 ID、完整 prompt、UUID URL、固定呈现排列和 ground truth |
| `split_manifest.json` | 19,136 条训练轨迹、4,299 条五城 OOD 轨迹的冻结归属 |
| `batch_submission.json` | 各城市主 Batch、补交记录、API batch/file ID 及状态 |
| `responses/<batch_id>.jsonl` | 每个 Batch 的完整原始响应及错误记录 |
| `cot_data.jsonl` | 全部城市 Batch 收集及必要补交结束后生成的完整逐轨迹 CoT 结果 |
| `summary.json` | 执行中进度或最终总体、各城市、usage 和 SFT 筛选统计 |
| `verification.json` | 独立全量核验：结果覆盖与唯一性、图片与呈现映射、原始响应、split、评分、thinking token 重计数及 SFT 标志一致性 |

运行记录副本目录：`/path/to/visual_trajectory/runs/satellite_cot_qwen38_flash_all_20261004`，包含 `state.json`、`worker.log` 和 `worker_pid.json`。执行代码位于服务器 `code/evaluation/src/generate_satellite_qwen38_flash_cot_batch.py`，沿用 Street View 的提交核对、原始响应收集、失败项补交和严格评分实现。密钥不写入 manifest、日志或请求正文。

#### 2.3.3 Final Results and Verification

最终 `cot_data.jsonl` 包含 **23,435 条唯一轨迹结果**，Train **19,136**、五城 OOD **4,299**。核验确认 187,480 张本地图片及匿名 URL 映射完整、每条呈现排列与 ground truth 一致、53 个 Batch 全部结束并收集。原始响应共 23,694 行，包含各次尝试，不能作为最终唯一轨迹数量。

| 最终统计 | 数量 | 比例与分母 |
| --- | ---: | --- |
| API 成功 | 23,435 | 100% / 全部轨迹 |
| 最终 API 失败或缺失 | 0 | 0% / 全部轨迹 |
| 有效顺序列表 | 23,153 | 98.80% / 全部轨迹 |
| 顺序完全正确 | 16,580 | 70.75% / 全部轨迹 |
| 严格 CoT 格式有效 | 22,307 | 95.19% / 全部轨迹 |
| `finish_reason=stop` | 23,394 | 99.83% / 全部轨迹 |
| 长度筛选前满足其余 SFT 条件 | 16,130 | 全部城市 |
| 因 think tokens ≥ 2,048 排除 | 76 | 长度筛选前合格记录 |
| 全部城市满足 SFT 质量条件 | 16,054 | 68.50% / 全部轨迹 |
| Train `sft_export_eligible=true` | **13,164** | **68.79% / 19,136 条训练轨迹** |
| OOD 满足质量条件但不用于 SFT | 2,890 | 4,299 条 OOD 中的子集 |

SFT 质量条件同时要求 API 成功、顺序完全正确、严格单一 `<think>` / `<answer>` 格式及四步标题、`finish_reason=stop`、think tokens 严格小于 2,048。核验逐条对照原始响应，并使用上述冻结 tokenizer 对全量 think 正文重新计数；五城 OOD 的 `sft_export_eligible` 均为 false。13,164 表示可用于训练的合格记录数量，不表示已另行导出训练文件、完成人工视觉审核或执行模型训练。

所有尝试累计 usage：prompt tokens **212,297,665**，completion tokens **356,410,486**，total tokens **568,708,151**。结果核验脚本位于服务器 `code/evaluation/src/verify_satellite_cot_completion.py`，核验日志及 PID 记录位于上述运行目录。生成进程正常结束；完成后停止本次定时监督。

### 2.4 Prompts

#### 2.4.1 Prompt w/o CoT

##### System Prompt

```text
You are a visual spatial reasoning assistant specializing in trajectory reconstruction from multiple satellite observations.
```

##### User Prompt

Construct a single multimodal `user` message using the template below. Each `Image k:` label must be a text content block immediately before its corresponding image content block; `k` is the presentation position after shuffling, not the original frame number. Replace the image placeholders with actual image attachments, not literal placeholder text or original filenames. Include only the prompt text, displayed image labels, and image content; keep ground-truth order and original frame identifiers outside the model input.

```text
You are given 8 satellite observation images sampled along a spatial trajectory. The images have been shuffled and are presented in an arbitrary order, labeled Image 0 through Image 7. Each label immediately precedes its corresponding image. Each image is oriented so that its heading, corresponding to the upward direction of the image, is aligned with the direction of movement along the trajectory. Different observations may have different scales due to changes in observation altitude; infer movement from changes in observation center position rather than from scale changes. Reconstruct the spatial order in which these 8 observations would be encountered when following the trajectory.

Output using exactly the following format:

<answer>
[i0, i1, i2, i3, i4, i5, i6, i7]
</answer>

Image 0:
[Image content for presentation position 0]
Image 1:
[Image content for presentation position 1]
Image 2:
[Image content for presentation position 2]
Image 3:
[Image content for presentation position 3]
Image 4:
[Image content for presentation position 4]
Image 5:
[Image content for presentation position 5]
Image 6:
[Image content for presentation position 6]
Image 7:
[Image content for presentation position 7]
```

#### 2.4.2 CoT Prompt

##### System Prompt

```text
You are a visual spatial reasoning assistant specializing in trajectory reconstruction from multiple satellite observations.

Analyze the provided images and produce a detailed, evidence-grounded explanation following the four steps specified in the user prompt. Base your conclusions on visible overhead scene evidence and the stated image-orientation constraint. Account for possible differences in spatial extent and apparent scale across observations. Distinguish clear visual correspondences from uncertain matches, and do not invent visual details.

Refer to each observation by its displayed label, Image 0 through Image 7. These labels identify the order of presentation, not the spatial order along the trajectory.

Follow the required output format exactly. Within <think>, use all four step titles from the user prompt verbatim and in the specified order. Within <answer>, output only a list of eight integer indices, using each index from 0 to 7 exactly once. Do not include any text outside these two blocks.
```

##### User Prompt

Construct a single multimodal `user` message using the template below. Each `Image k:` label must be a text content block immediately before its corresponding image content block; `k` is the presentation position after shuffling, not the original frame number. Replace the image placeholders with actual image attachments, not literal placeholder text or original filenames. Include only the prompt text, displayed image labels, and image content; keep ground-truth order and original frame identifiers outside the model input.

```text
You are given 8 satellite observation images sampled along a spatial trajectory. The images have been shuffled and are presented in an arbitrary order, labeled Image 0 through Image 7. Each label immediately precedes its corresponding image. Each image is oriented so that its heading, corresponding to the upward direction of the image, is aligned with the direction of movement along the trajectory. Different observations may have different scales due to changes in observation altitude; infer movement from changes in observation center position rather than from scale changes. Reconstruct the spatial order in which these 8 observations would be encountered when following the trajectory.

Provide a detailed, evidence-grounded explanation using strictly the following four steps, followed by the final order.

Step 1: Global Scene Understanding
Provide a high-level overview of the geographic environment and the spatial characteristics of the route depicted across the images. Identify major land-cover patterns, transportation structures, building distributions, water bodies, or other large-scale geographic features that characterize the observed area. Focus on the overall character of the scene along the route rather than inferring their relative order at this stage.

Step 2: Cross-View Correspondence Identification
Identify visual correspondences across different observations by locating recurring geographic features that appear in multiple images and distinctive local cues that only appear in a subset of observations. Useful correspondences may include roads, intersections, building complexes, blocks, water features, vegetation boundaries, open spaces, or other persistent spatial structures. Focus on correspondences that provide useful evidence for determining the trajectory order.

Step 3: Viewpoint Transformation Analysis
Compare observations linked by shared geographic features, focusing on potentially adjacent pairs. For each pair, analyze how corresponding features change across observations as the image center moves along the trajectory. Examine their relative position within the image, orientation, visibility, overlap, and apparent size. Treat scale changes as possible altitude changes rather than evidence of forward or backward movement, and infer movement primarily from changes in observation center position. Use these transformations together with the constraint that the upward direction of each image is aligned with the direction of movement to infer relative order. Check whether the inferred transformations are spatially consistent and whether the observations can form a continuous movement path.

Step 4: Final Sequence Determination
Combine the inferred pairwise spatial relationships to construct the final trajectory order, ensuring that the sequence represents a spatially continuous path and that every image index appears exactly once.

Output using exactly the following format:

<think>
Step 1: Global Scene Understanding
...

Step 2: Cross-View Correspondence Identification
...

Step 3: Viewpoint Transformation Analysis
...

Step 4: Final Sequence Determination
...
</think>

<answer>
[i0, i1, i2, i3, i4, i5, i6, i7]
</answer>

Image 0:
[Image content for presentation position 0]
Image 1:
[Image content for presentation position 1]
Image 2:
[Image content for presentation position 2]
Image 3:
[Image content for presentation position 3]
Image 4:
[Image content for presentation position 4]
Image 5:
[Image content for presentation position 5]
Image 6:
[Image content for presentation position 6]
Image 7:
[Image content for presentation position 7]
```
