#!/bin/bash
#SBATCH --job-name=s3-serve
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=11
#SBATCH --time=6:00:00
#SBATCH -o /fsx/kashif/s3-repl/logs/%x-%j.out
# Serve a model with vLLM on this job's GPU and keep it up; evaluate inside the job with
#   srun --jobid=<id> --overlap python eval.py --url http://127.0.0.1:$(cat logs/port-<id>) --name NAME
# The port is written to logs/port-<jobid>. Env knobs: MODEL (Qwen/Qwen3-0.6B), MAX_LEN (40960).
set -euo pipefail
ROOT=/fsx/kashif/s3-repl
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
PORT=${PORT:-$((20000 + SLURM_JOB_ID % 20000))}
export HF_HOME=${HF_HOME:-/fsx/kashif/.cache/huggingface} HF_HUB_OFFLINE=1
cd "$ROOT"
echo "$PORT" > "logs/port-$SLURM_JOB_ID"
echo "node=$(hostname) model=$MODEL port=$PORT"
exec "${VLLM_BIN:-/fsx/kashif/miniconda3/bin/vllm}" serve "$MODEL" --served-model-name "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --max-model-len "${MAX_LEN:-40960}" \
  --max-num-seqs 512 --gpu-memory-utilization 0.85 --enable-prefix-caching
