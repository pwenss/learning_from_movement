#!/usr/bin/env bash
set -euo pipefail
ROOT="${LFM_ROOT:?Set LFM_ROOT}"
ENV="${LFM_RL_ENV:?Set LFM_RL_ENV}"
CODE="$ROOT/code/training/satellite_grpo_opsd3"
export CUDA_HOME="$ENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$ENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$CODE:$ROOT/code/training:$ROOT/runtime/verl"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,2,3
export STREETVIEW_DEPLOYMENT_WORLD_SIZE=3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VERL_USE_UV=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export WANDB_MODE=disabled
cd "$ROOT"
exec "$ENV/bin/python" -u "$CODE/run_serial.py" "$@"
