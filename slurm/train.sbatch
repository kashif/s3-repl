#!/bin/bash
#SBATCH --job-name=s3-train
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=22
#SBATCH --time=10:00:00
#SBATCH -o /fsx/kashif/s3-repl/logs/%x-%j.out
# s3 RL (train_s3.py): vLLM with NCCL weight transfer on GPU 0, the AsyncGRPOTrainer on GPU 1.
# vLLM runs offline (fast start); the trainer online (`kernels` resolves flash-attn3 "version 1" on the Hub).
#   sbatch slurm/train.sbatch [train_s3.py args...]     e.g. --teams-per-update 128 --updates 50
set -euo pipefail
ROOT=/fsx/kashif/s3-repl
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
PORT=${PORT:-$((20000 + SLURM_JOB_ID % 20000))}
export HF_HOME=${HF_HOME:-/fsx/kashif/.cache/huggingface}
export PYTHONPATH=/fsx/kashif/trl${PYTHONPATH:+:$PYTHONPATH} PYTHONUNBUFFERED=1
cd "$ROOT"
echo "node=$(hostname) model=$MODEL port=$PORT args=$*"
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_SERVER_DEV_MODE=1 /fsx/kashif/miniconda3/bin/vllm serve "$MODEL" \
  --served-model-name "$MODEL" --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --max-model-len 40960 \
  --max-num-seqs 512 --gpu-memory-utilization 0.85 --enable-prefix-caching \
  --logprobs-mode processed_logprobs --weight-transfer-config '{"backend":"nccl"}' \
  > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  kill -0 $VLLM_PID 2>/dev/null || { echo "vllm died"; tail -40 "logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "vllm ready after $((i * 10)) s"; break; }
  sleep 10
done
CUDA_VISIBLE_DEVICES=1 /fsx/kashif/miniconda3/bin/accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_s3.py --vllm-url "http://127.0.0.1:$PORT" --output-dir "checkpoints/s3-$SLURM_JOB_ID" "$@"
