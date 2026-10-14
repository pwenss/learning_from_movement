#!/usr/bin/env bash
set -euo pipefail
ROOT="${LFM_ROOT:?Set LFM_ROOT}"
ENV="${LFM_RL_ENV:?Set LFM_RL_ENV}"
CODE="$ROOT/code/training/grpo_opsd"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,3,1
export STREETVIEW_DEPLOYMENT_WORLD_SIZE=4
export CUDA_HOME="$ENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$ENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$CODE/serial_ddp4:$CODE/three_gpu:$CODE/deployment:$ROOT/code/training:$CODE:$ROOT/runtime/verl"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VERL_USE_UV=0
cd "$ROOT/runtime/verl"
exec "$ENV/bin/python" -u "$CODE/serial_ddp4/run_serial.py" "$@"
