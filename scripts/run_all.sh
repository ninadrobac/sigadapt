#!/usr/bin/env bash
# Reproduce every experiment of the paper.
#   bash scripts/run_all.sh            # all datasets
#   bash scripts/run_all.sh rte gfc12  # only these datasets
# TiDE and Chronos-2 need a GPU (in practice); the other models run on a CPU.
set -euo pipefail
cd "$(dirname "$0")/.."

datasets=("${@:-rte gfc12 gfc14 gfc17 entsoe epf}")
for d in ${datasets[@]}; do
  for cfg in configs/"$d"/*.yaml; do
    echo "=== $cfg"
    python scripts/run.py "$cfg"
  done
done

python scripts/collect_rte.py
python scripts/make_tables.py
