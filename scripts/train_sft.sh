#!/usr/bin/env bash
set -euo pipefail

python scripts/build_sft_data.py
exec python scripts/run_sft.py --config configs/sft/qwen2_5_1_5b_lora_smoke.json "$@"
