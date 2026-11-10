#!/bin/bash
#SBATCH --job-name=s3-run
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=11
#SBATCH --time=4:00:00
#SBATCH -o /fsx/kashif/s3-repl/logs/%x-%j.out
# Serve a model with vLLM on this job's GPU, then run a swarmlab spec against it.
#   sbatch slurm/serve_and_run.sbatch specs/smoke.yaml [swarmlab run args...]
# Env knobs: MODEL (Qwen/Qwen3-0.6B), PORT (20000 + job id % 20000), MAX_LEN (40960), PARALLEL (8).
set -euo pipefail
SPEC=${1:?spec path}; shift || true
ROOT=/fsx/kashif/s3-repl
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
PORT=${PORT:-$((20000 + SLURM_JOB_ID % 20000))}
MAX_LEN=${MAX_LEN:-40960}
PARALLEL=${PARALLEL:-8}
export HF_HOME=${HF_HOME:-/fsx/kashif/.cache/huggingface} HF_HUB_OFFLINE=1
cd "$ROOT"
echo "node=$(hostname) model=$MODEL port=$PORT spec=$SPEC args=$*"
"${VLLM_BIN:-/fsx/kashif/miniconda3/bin/vllm}" serve "$MODEL" --served-model-name "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --max-model-len "$MAX_LEN" --max-num-seqs 256 --gpu-memory-utilization 0.85 \
  --enable-prefix-caching > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
  kill -0 $VLLM_PID 2>/dev/null || { echo "vllm died"; tail -40 "logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "vllm ready after $((i * 10)) s"; break; }
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "vllm not ready"; exit 1; }
RUN_SPEC="logs/spec-$SLURM_JOB_ID.yaml"
sed "s#http://127.0.0.1:8000#http://127.0.0.1:$PORT#" "$SPEC" > "$RUN_SPEC"
grep -q "127.0.0.1:$PORT" "$RUN_SPEC" || { echo "no base_url to rewrite in $SPEC"; exit 1; }
SL="uv run --project /fsx/kashif/swarmlab swarmlab"
$SL run "$RUN_SPEC" --out runs/ --parallel "$PARALLEL" --yes "$@"
$SL report runs/ --out "logs/report-$SLURM_JOB_ID.md" || true
