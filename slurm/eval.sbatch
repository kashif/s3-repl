#!/bin/bash
#SBATCH --job-name=s3-eval
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=11
#SBATCH --time=2:00:00
#SBATCH -o /fsx/kashif/s3-repl/logs/%x-%j.out
# Serve a model (hub id or checkpoint dir) with vLLM on this job's GPU, then run eval.py against it.
#   sbatch slurm/eval.sbatch NAME [eval.py args...]          e.g. base --episodes 192
#   MODEL=checkpoints/s3-364160/checkpoint-50 sbatch slurm/eval.sbatch ckpt50
# Output: runs/NAME.jsonl. Env knobs: MODEL (Qwen/Qwen3-0.6B), PORT (20000 + job id % 20000).
set -euo pipefail
NAME=${1:?run name}; shift || true
ROOT=/fsx/kashif/s3-repl
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
PORT=${PORT:-$((20000 + SLURM_JOB_ID % 20000))}
export HF_HOME=${HF_HOME:-/fsx/kashif/.cache/huggingface} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$ROOT"
echo "node=$(hostname) model=$MODEL port=$PORT name=$NAME args=$*"
/fsx/kashif/miniconda3/bin/vllm serve "$MODEL" --served-model-name "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --max-model-len 40960 --max-num-seqs 512 --gpu-memory-utilization 0.85 \
  --enable-prefix-caching > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
  kill -0 $VLLM_PID 2>/dev/null || { echo "vllm died"; tail -40 "logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "vllm ready after $((i * 10)) s"; break; }
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "vllm not ready"; exit 1; }
/fsx/kashif/miniconda3/bin/python eval.py --url "http://127.0.0.1:$PORT" --model "$MODEL" --name "$NAME" "$@"
/fsx/kashif/miniconda3/bin/python analysis/episodes.py "runs/$NAME.jsonl" --quiet
