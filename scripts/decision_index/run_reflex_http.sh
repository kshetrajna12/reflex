#!/usr/bin/env bash
set -euo pipefail

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENDPOINT="${1:?usage: run_reflex_http.sh ENDPOINT TOKEN_FILE OUT_DIR [full|SAMPLE_SIZE] [MODEL] [WORKERS]}"
TOKEN_FILE="${2:?usage: run_reflex_http.sh ENDPOINT TOKEN_FILE OUT_DIR [full|SAMPLE_SIZE] [MODEL] [WORKERS]}"
OUT_DIR="${3:?usage: run_reflex_http.sh ENDPOINT TOKEN_FILE OUT_DIR [full|SAMPLE_SIZE] [MODEL] [WORKERS]}"
MODE="${4:-100}"
MODEL="${5:-reflex-27b}"
WORKERS="${6:-1}"
SUITE_DIR="${DECISION_INDEX_SUITE_DIR:-${KIT_DIR}/suite}"
REBUILT_DIR="${DECISION_INDEX_REBUILT_DIR:-${KIT_DIR}/work/artifacts/benchmark-suite/release-v1-rebuilt}"
BASE_URL="${ENDPOINT%/}"
BASE_URL="${BASE_URL%/v1/systemone}"

if [[ ! -r "${TOKEN_FILE}" ]]; then
  echo "token file is not readable: ${TOKEN_FILE}" >&2
  exit 2
fi

mkdir -p "$(dirname "${OUT_DIR}")"

# The HTTP engine reads only this environment variable. Command substitution
# removes a trailing newline without printing the credential.
DECISION_INDEX_API_KEY="$(<"${TOKEN_FILE}")"
export DECISION_INDEX_API_KEY
export HF_HUB_DISABLE_XET=1

run_index() {
  uv run --project "${KIT_DIR}" python -m decision_index "$@"
}

if [[ ! -f "${SUITE_DIR}/selected-rows.jsonl.gz" ]]; then
  if [[ ! -f "${REBUILT_DIR}/selected-rows.jsonl.gz" ]]; then
    echo "rebuilt suite is missing: ${REBUILT_DIR}/selected-rows.jsonl.gz" >&2
    echo "run: uv run python -m decision_index suite rebuild --work work" >&2
    exit 2
  fi
  run_index suite import \
    --dir "${SUITE_DIR}" \
    --rows "${REBUILT_DIR}/selected-rows.jsonl.gz" \
    --exclusions "${KIT_DIR}/hub/excluded-questions.json" \
    --manifest "${KIT_DIR}/hub/manifest.json"
fi
run_index suite verify --dir "${SUITE_DIR}"

common=(
  --engine decision_index.engines.reflex_http:ReflexHttpSystemOne
  --suite-dir "${SUITE_DIR}"
  --option "base_url=${BASE_URL}"
  --option "model=${MODEL}"
  --option 'extra={"permutations":2}'
  --option timeout=600
  --out "${OUT_DIR}"
  --compact
  --workers "${WORKERS}"
)

if [[ "${MODE}" == "full" ]]; then
  run_index pipeline "${common[@]}"
else
  if [[ ! "${MODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "fourth argument must be 'full' or a positive sample size" >&2
    exit 2
  fi
  sample="${OUT_DIR%/}-sample-${MODE}.jsonl.gz"
  run_index suite sample --dir "${SUITE_DIR}" --n "${MODE}" --out "${sample}"
  run_index pipeline "${common[@]}" --rows "${sample}"
fi
