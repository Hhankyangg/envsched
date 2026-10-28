#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
SELECTOR_PYTHON=${SELECTOR_PYTHON:-python3}
RUN_DIR=${ROLL_CURRICULUM_RUN_DIR:?ROLL_CURRICULUM_RUN_DIR is required}
HISTORY_MAX_STEP=${ROLL_CURRICULUM_HISTORY_MAX_STEP:?ROLL_CURRICULUM_HISTORY_MAX_STEP is required}
TARGET_STEP=${ROLL_CURRICULUM_TARGET_STEP:-$((HISTORY_MAX_STEP + 1))}
OUTPUT_DIR=${CURRICULUM_OUTPUT_DIR:?CURRICULUM_OUTPUT_DIR is required}

export PYTHONPATH=${WORKSPACE}/selector:${PYTHONPATH:-}
exec "${SELECTOR_PYTHON}" "${WORKSPACE}/selector/select_online_high_variance.py" \
  --run-dir "${RUN_DIR}" \
  --selection-history-dir "${OUTPUT_DIR}" \
  --scenarios "${CURRICULUM_SCENARIOS}" \
  --topologies "${CURRICULUM_TOPOLOGIES}" \
  --environment-metadata "${CURRICULUM_ENVIRONMENT_METADATA}" \
  --history-min-step "${ROLL_CURRICULUM_HISTORY_MIN_STEP:-0}" \
  --history-max-step "${HISTORY_MAX_STEP}" \
  --target-policy-step "${TARGET_STEP}" \
  --output-dir "${OUTPUT_DIR}" \
  --exact-task-exposure-scale 16 --within-env-graph-weight 0.5 \
  --least-observed-per-batch 32 --max-per-environment 4 \
  --environment-window-steps 3 --max-per-environment-window 11 \
  --max-consecutive-selections 2 --diversity-penalty 0.08 \
  --recency-guard-half-life 2 --recency-guard-strength 0.5 \
  --progress-environment-prior-weight 0.5 \
  "$@"
