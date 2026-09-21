#!/bin/bash
# One-command latency run for a rented GPU (H100/A100): serve reflex, sweep the grid,
# and, if TYPESAFE_API_KEY is set, probe the hosted Jev from the same machine.
#
#   git clone https://github.com/kshetrajna12/reflex && cd reflex && pip install uv && uv sync
#   MODEL=Qwen/Qwen3.8-27B PERMS=2 bash scripts/bench_h100.sh            # 27B bf16, two orders
#   MODEL=Qwen/Qwen3.5-4B  PERMS=2 bash scripts/bench_h100.sh            # the 4B `stable` shape
#   MODEL=Qwen/Qwen3.8-27B-FP8 bash scripts/bench_h100.sh               # FP8 checkpoint (H100)
#
# Outputs: bench_<label>.json and a printed table (p50/p95, req/s, q/s per cell), plus
# the Jev probe lines. Cold rows are the ones comparable to Jev; warm rows need a repeated
# state. Nothing here needs SGLang.
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3.8-27B}
PERMS=${PERMS:-2}
PORT=${PORT:-8010}
LABEL=${LABEL:-$(basename "$MODEL")-p$PERMS-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-')}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."

REFLEX_API_KEY=bench uv run reflex-serve --model "$MODEL" --permutations "$PERMS" \
    --served-name reflex --host 127.0.0.1 --port "$PORT" > "serve_$LABEL.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
for i in $(seq 1 180); do curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q healthy && break; sleep 5; done
curl -s "http://127.0.0.1:$PORT/health"; echo
echo "== $LABEL: grid (cold rows are the Jev-comparable ones) =="
uv run python scripts/bench_latency.py --url "http://127.0.0.1:$PORT" --api-key bench \
    --concurrency 1,2,4,8,16 --questions 1,3,10 --requests 32 --label "$LABEL" --out "bench_$LABEL.json"
echo "== $LABEL: 2,000-token state, 3 questions =="
uv run python scripts/bench_latency.py --url "http://127.0.0.1:$PORT" --api-key bench \
    --concurrency 1,8 --questions 3 --requests 16 --state-tokens 2000 --label "$LABEL-2k" --out "bench_$LABEL-2k.json"
if [ -n "${TYPESAFE_API_KEY:-}" ]; then
  echo "== hosted Jev from this machine =="
  uv run --with typesafe-sdk python scripts/jev_latency_probe.py || echo "(jev probe failed; is typesafe-sdk installable here?)"
fi
echo "BENCH_DONE $LABEL"
