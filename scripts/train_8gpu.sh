#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 CONFIG OUTPUT_DIR [extra train arguments...]" >&2
  exit 2
fi

config_path=$1
output_dir=$2
shift 2
torchrun --standalone --nproc_per_node=8 -m resflow.train \
  --config "$config_path" --output "$output_dir" "$@"

