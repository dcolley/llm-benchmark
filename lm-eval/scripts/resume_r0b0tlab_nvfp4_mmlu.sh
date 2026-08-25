#!/usr/bin/env bash
# Repair NVFP4 HF cache symlinks, start server, run MMLU.
set -euo pipefail

STATE=/home/derek/llm-benchmark/lm-eval/results/qwen38-quant-pipeline.state
RECIPE=/home/derek/llm-benchmark/vllm/recipes/r0b0tlab-qwen3.8-27b-nvfp4-mtp-sm121
LM_EVAL=/home/derek/llm-benchmark/lm-eval
OUT_DIR=$LM_EVAL/results/r0b0tlab-qwen3.8-27b-nvfp4-mtp-sm121

printf '%s\trepairing-nvfp4-cache\twiring weight shard symlinks\n' "$(date -Is)" >"$STATE"

# Stop competing slim downloaders / old server
docker ps -q --filter ancestor=python:3.12-slim | xargs -r docker rm -f || true
docker rm -f qwen38-27b-nvfp4-mtp >/dev/null 2>&1 || true

docker run --rm \
  -v /home/derek/.cache/huggingface:/root/.cache/huggingface \
  -v "$LM_EVAL/scripts/repair_r0b0tlab_nvfp4_cache.py:/repair.py:ro" \
  python:3.12-slim python /repair.py

printf '%s\tstarting-nvfp4\trepaired cache; launching server\n' "$(date -Is)" >"$STATE"
cd "$RECIPE"
FOLLOW_LOGS=0 ./start-docker.sh

for i in $(seq 1 180); do
  if curl -fsS --max-time 3 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q qwen3.8-27b-nvfp4-mtp; then
    echo "HEALTHY after $i checks"
    break
  fi
  status=$(docker inspect -f '{{.State.Status}}' qwen38-27b-nvfp4-mtp 2>/dev/null || echo missing)
  echo "wait $i status=$status"
  if [[ "$status" == "exited" ]]; then
    docker logs --tail 80 qwen38-27b-nvfp4-mtp 2>&1 | tail -80
    exit 1
  fi
  sleep 10
done

curl -fsS http://127.0.0.1:8000/v1/models | grep -q qwen3.8-27b-nvfp4-mtp
printf '%s\tbenchmarking-nvfp4\tserver healthy; MMLU starting\n' "$(date -Is)" >"$STATE"

cd "$LM_EVAL"
MODEL_ID=qwen3.8-27b-nvfp4-mtp OUT_DIR="$OUT_DIR" NUM_CONCURRENT=8 \
  ./scripts/run_mmlu_llama_qwen38_27b_fp8.sh --foreground

printf '%s\tnvfp4-complete\tmmlu finished\n' "$(date -Is)" >"$STATE"
echo NVFP4_BENCH_DONE
