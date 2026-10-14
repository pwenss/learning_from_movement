# Reproduction Notes

## 1. Release boundary

This is a source and documentation release. It includes small, final figures and aggregate result tables; it excludes full image datasets, teacher records, dataset manifests, raw predictions, training histories, audits, weights, and checkpoints. One selected eight-image StreetView input is included as a README illustration. A public dataset download is not provided by this repository.

The public source preserves the recorded algorithm, model, dataset contracts, and experiment structure. Environment-specific directories are replaced by placeholders. The formal entry points require the original external resources or an independently prepared dataset matching the documented contracts; merely cloning this repository does not reproduce the reported numbers.

## 2. Configure external resources

### 2.1 Paths

Use `.env.example` as a list of configuration variables, and export values in your shell. It is not automatically loaded.

```bash
export LFM_ROOT=/absolute/path/to/visual_trajectory
export LFM_MODEL=/absolute/path/to/models/Qwen3-VL-8B-Instruct
export LFM_SFT_ENV=/absolute/path/to/sft_environment
export LFM_RL_ENV=/absolute/path/to/grpo_opsd_environment
export LFM_INFERENCE_ENV=/absolute/path/to/vllm_environment
```

Core Python project/model paths and formal RL launchers accept these variables. Remaining `/path/to/...` values in generation utilities, audit provenance, or example configurations must be reviewed for your deployment. Audit scripts that verify frozen source/model hashes refer to the private original experiment, not the sanitized public snapshot.

### 2.2 Images and manifests

Arrange the eight images for each route, frozen city split, image path mapping, shuffle, ground-truth permutation, teacher response, and quality flags as described in the SFT setup. Store derived manifests and prompts under `training/<experiment_id>/`; those files are external and deliberately ignored by Git.

SFT export scripts enforce the published sample counts and sequence budgets. Using another dataset requires a deliberate adaptation of those assertions and a new experimental record.

### 2.3 GPU placement

Recorded launchers encode the original GPU counts and placement. Review `CUDA_VISIBLE_DEVICES`, the DDP world size, and GPU UUID checks before execution. `LFM_GPU_UUIDS` supplies UUIDs to source files that used a fixed UUID list; obtain values from your own machine. A three-rank execution and a four-rank execution have different state layouts.

Do not treat the four-rank StreetView migration launcher as an automatic fresh run: its recorded recovery path expects a complete three-rank cycle-51 checkpoint. The three-rank path initializes from merged SFT.

## 3. Pipeline entry points

### 3.1 Dataset generation

| Function | Source |
|---|---|
| StreetView production sampling | `trajectory/streetview/server/run_all.py`, `run_city.py`, `sampling.py` |
| Panorama metadata | `trajectory/streetview/server/maps.py` |
| Aerial path and image sampling | `trajectory/satellite/src/generate_satellite_trajectories.py` |
| Satellite geometry/image validation | `trajectory/satellite/src/validate_satellite_dataset.py` |
| Dynamic World boundary exports | `trajectory/satellite/src/download_dynamic_world_boundaries.py` |
| Teacher inference and ordering metrics | `evaluation/src/` |

Generation requires city boundaries and imagery access configured by the caller. StreetView reads `GOOGLE_MAPS_API_KEY` from the environment. Satellite sampling and Dynamic World checks do not include the final private land-cover tables; their filtering contract is documented in the main README and experiment specification.

### 3.2 SFT

| Stage | StreetView | Satellite |
|---|---|---|
| Prepare manifests | `export_streetview_sft.py` | `export_satellite_sft.py` |
| Training | `train_streetview_sft.py` | `train_satellite_sft.py` |
| Input/masking contract | `streetview_sft_common.py` | `satellite_sft_common.py` |
| vLLM evaluation | `evaluate_streetview_vllm.py` | `evaluate_satellite_vllm.py` |

These files are in `code/training/`. Use separate SFT and inference environments matching the documented versions. Review each training script's arguments and deferred-generation settings; the reported candidate generations were performed after training.

### 3.3 GRPO + OPSD

| Function | Source |
|---|---|
| Full StreetView training set | `code/training/grpo_opsd/prepare_streetview.py` |
| Full Satellite training set | `code/training/satellite_grpo_opsd/prepare_satellite.py` |
| StreetView formal three-rank runtime | `code/training/grpo_opsd/serial_ddp/` |
| StreetView formal four-rank runtime | `code/training/grpo_opsd/serial_ddp4/` |
| Satellite formal three-rank runtime | `code/training/satellite_grpo_opsd3/` |
| Later generation evaluation | `evaluate_satellite_rl_checkpoint.py`, `evaluate_merged_sft_ood.py` |

The parent StreetView `experiment.yaml` and configuration inheritance helpers originate from earlier deployments. The formal three/four-rank configuration overrides them to **B96 / n8 / mini24 / epochs1 / micro1**. Use the formal `run_serial.py` entry point rather than interpreting the parent pilot defaults as the reported experiment.

Install upstream verl at commit `704a9f01bda8b9863c02c113547191bf167bc88f` under `runtime/verl/`. The project integration uses pinned internal APIs for native GRPO and PG distillation, shared vLLM scoring, DDP actor updates, response alignment, and checkpoint handling. Runtime adaptation files are included; the upstream source, environments, model weights, and private run-specific migrations are not vendored.

The RL configs refer to merged SFT models. Merge the selected SFT adapter using the single-round FP32 accumulation/BF16 conversion recorded in the setup, verify it, and then initialize a new RL LoRA. Student and evaluation prompts must never include the privileged teacher's GT suffix.

## 4. Visualization

### 4.1 Trajectory browser

The Python viewer has no third-party server dependency. With external datasets configured:

```bash
export STREETVIEW_DATASET_ROOT=/path/to/streetview_dataset
export SATELLITE_DATASET_ROOT=/path/to/satellite_dataset
export STREETVIEW_EVALUATION_ROOT=/path/to/evaluation_records
python trajectory/viewer/serve_production_viewer.py
```

Open `http://127.0.0.1:8787/` for the trajectory view or `?mode=evaluation` for model outputs. The repository does not include the images or evaluation records consumed by the service. Google basemap tiles also require network access.

### 4.2 Training figures

The collectors `collect_sft_plot_data.py` and `collect_grpo_opsd_plot_data.py` read the external run directories and print compact JSON with source hashes. Save that output outside version control, then use the corresponding `plot_*_results.py` script as shown in the README.

The GRPO collector excludes the discarded StreetView three-rank branch after cycle 51 and retains Satellite's incomplete final actor cycle honestly. The SFT collector verifies that final-checkpoint and final-root loss histories match, allowing the root state to append its runtime summary.

The conceptual overview images `figures/output/figure1.png` and `figure2.png` are final author-supplied assets. Map rendering utilities require external boundaries, routes, and tile inputs; those inputs are not bundled.

## 5. Validation performed for this release

Before publication, Python syntax, local import availability for the formal runtime, Markdown/image links, selected source/asset size limits, and the staged file list are checked. Figures are read from verified local final outputs. No GPU training or generation job is launched as part of publication.

Full reruns depend on external data, frozen prompts, source hashes, environments, and service access. The repository does not claim that the full training/evaluation suite runs without those inputs.
