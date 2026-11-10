#!/bin/bash
# Run a swarmlab spec against the vLLM server of the serve job this runs in (srun --overlap).
set -euo pipefail
SPEC=${1:?spec path}; shift || true
cd /fsx/kashif/s3-repl
PORT=$(cat "logs/port-$SLURM_JOB_ID")
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "vllm on $PORT not ready"; exit 1; }
RUN_SPEC="logs/spec-$SLURM_JOB_ID-$(basename "$SPEC")"
sed "s#http://127.0.0.1:8000#http://127.0.0.1:$PORT#" "$SPEC" > "$RUN_SPEC"
uv run --project /fsx/kashif/swarmlab swarmlab run "$RUN_SPEC" --out runs/ --parallel "${PARALLEL:-8}" --yes "$@"
