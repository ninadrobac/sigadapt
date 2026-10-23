"""Collect the metrics of all runs into two tables.

    python scripts/make_tables.py

results/summary_per_series.csv : one row per (dataset, model, series)
results/summary.csv            : one row per (dataset, model), metrics averaged over the series
                                 (for the standardised metrics, this is the fev-bench convention)
For TiDE, the metrics are averaged over the seeds (columns *_mean) before averaging over series.
"""

import json
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parent.parent / "results"
rows = []
for f in sorted(root.glob("*/*/**/metrics.json")):
    rel = f.relative_to(root).parts            # dataset / model / [series /] metrics.json
    dataset, model = rel[0], rel[1]
    series = rel[2] if len(rel) == 4 else dataset
    m = json.load(open(f))
    m = {k.removesuffix("_mean"): v for k, v in m.items() if not k.endswith("_sd") and k != "lookback_steps"}
    rows.append({"dataset": dataset, "model": model, "series": series, **m})

if not rows:
    raise SystemExit("no metrics.json found under results/")

per_series = pd.DataFrame(rows)
per_series.to_csv(root / "summary_per_series.csv", index=False)
summary = per_series.drop(columns="series").groupby(["dataset", "model"]).mean()
summary.to_csv(root / "summary.csv")
print(summary.round(4).to_string())
