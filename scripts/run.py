"""Run one experiment.

    python scripts/run.py configs/rte/sigadapt.yaml
    python scripts/run.py configs/gfc12/sigadapt.yaml --series 1 2

Results go to results/<dataset>/<model>/<series>/ (or results/<dataset>/<model>/ for a
dataset with one series), with a copy of the configuration that produced them.
"""

import argparse
import json
from pathlib import Path

import yaml

from sigadapt.config import load_config
from sigadapt.data import load_dataset
from sigadapt.metrics import scores, train_scale


def get_model(name):
    if name == "sigadapt":
        from sigadapt.sigadapt import run
    elif name == "linear":
        from sigadapt.baselines.linear import run
    elif name == "gam":
        from sigadapt.baselines.gam import run
    elif name == "tide":
        from sigadapt.baselines.tide import run
    elif name == "chronos2":
        from sigadapt.baselines.chronos2 import run
    else:
        raise ValueError(f"unknown model '{name}'")
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("config", help="experiment config, e.g. configs/rte/sigadapt.yaml")
    parser.add_argument("--series", nargs="*", help="run only these series (zones, markets, ...)")
    parser.add_argument("--root", default=".", help="repository root (data/ and results/ live here)")
    parser.add_argument("--out", default=None, help="output folder (default: results/<dataset>/<model>)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run = get_model(cfg["model"])
    all_series = load_dataset(cfg["data"], root=args.root)
    if args.series:
        all_series = [s for s in all_series if s.name in args.series]

    base = Path(args.out) if args.out else Path(args.root) / "results" / cfg["data"]["name"] / cfg["model"]
    base.mkdir(parents=True, exist_ok=True)
    (base / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    for series in all_series:
        out_dir = base / series.name if "series" in cfg["data"] else base   
        print(f"{cfg['model']} | {series.dataset} | {series.name}")
        pred = run(series, cfg, out_dir)
        if pred is not None:      # models with several seeds (TiDE) write their own metrics
            metrics = scores(pred["y_true"], pred["y_pred"], scale=train_scale(series))
            json.dump(metrics, open(out_dir / "metrics.json", "w"), indent=1)
        metrics = json.load(open(out_dir / "metrics.json"))
        print("   " + "  ".join(f"{k} {v:.4g}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
