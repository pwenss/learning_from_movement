#!/usr/bin/env bash
set -euo pipefail
ROOT="${LFM_ROOT:?Set LFM_ROOT}"
RUN="$ROOT/runs/streetview_grpo_opsd_qwen3vl8b_lora_v1"
REPO="$ROOT/runtime/verl"
ENV="${LFM_RL_ENV:?Set LFM_RL_ENV}"
UV='/path/to/workspace/conda-envs/streetview_verl_grpo_opsd/bin/uv'
PY='/path/to/workspace/conda-envs/streetview_sft_qwen3vl/bin/python'
REV='704a9f01bda8b9863c02c113547191bf167bc88f'
mkdir -p "$RUN/provision" "$ROOT/runtime"
trap 'rc=$?; printf "%s\n" "$rc" > "$RUN/provision/setup.exitcode"' EXIT
if [ ! -d "$REPO/.git" ]; then
  mkdir -p "$REPO"
  git -C "$REPO" init
  git -C "$REPO" remote add origin https://github.com/verl-project/verl.git
fi
if ! git -C "$REPO" cat-file -e "$REV^{commit}" 2>/dev/null; then
  timeout 600 git -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=60 -C "$REPO" fetch --depth=1 origin "$REV"
fi
git -C "$REPO" checkout --detach "$REV"
git -C "$REPO" rev-parse HEAD > "$RUN/provision/verl.commit"
timeout 600 "$UV" python install 3.12
export UV_PROJECT_ENVIRONMENT="$ENV"
export UV_HTTP_TIMEOUT=120
export UV_HTTP_RETRIES=3
export UV_LINK_MODE=copy
export MAX_JOBS=8
cd "$REPO"
WHEEL="$RUN/provision/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
if [ -f "$WHEEL" ]; then
  echo "ea61a750254b3f0b79171d58ee3fe4627d601411ee9801dd1705855c1ece6bf3  $WHEEL" | sha256sum -c -
  timeout 7200 "$UV" sync --frozen --all-packages --extra vllm --extra fsdp --python 3.12 --no-install-package flash-attn
  timeout 300 "$UV" pip install --python "$ENV/bin/python" --no-deps "$WHEEL"
else
  timeout 7200 "$UV" sync --frozen --all-packages --extra vllm --extra fsdp --python 3.12
fi
# The upstream lock mixes CUDA 13.0 runtime headers with newer nvcc/CRT/NVVM.
# Keep the compiler toolchain on 13.0 too; this does not alter torch/vLLM or kernels' math.
timeout 600 "$UV" pip install --python "$ENV/bin/python" --no-deps \
  nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88
CUDA_ROOT="$ENV/lib/python3.12/site-packages/nvidia/cu13"
[ -e "$CUDA_ROOT/lib64" ] || ln -s lib "$CUDA_ROOT/lib64"
[ -e "$CUDA_ROOT/lib/libcudart.so" ] || ln -s libcudart.so.13 "$CUDA_ROOT/lib/libcudart.so"
"$ENV/bin/python" -c 'import torch,transformers,vllm,ray,verl; print(torch.__version__,transformers.__version__,vllm.__version__,ray.__version__)'
"$UV" pip freeze --python "$ENV/bin/python" > "$RUN/provision/environment.lock.txt"
