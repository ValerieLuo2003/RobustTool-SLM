#!/usr/bin/env bash
set -euo pipefail

exec python scripts/run_grpo.py \
  --config configs/grpo/qwen2_5_1_5b_grpo_outcome_smoke.json \
  "$@"
