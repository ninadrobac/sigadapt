"""Chronos-2 baseline (zero-shot), with the same information set as SigAdapt.

Every point is forecast exactly one lead time ahead. The context length is chosen on the
validation period. Many forecasts are batched in one call by giving each its own series ID.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..calendar import rte_calendar
from ..metrics import scores, train_scale



def rte_special_flags(df):
    """Special-day flags only (no day of week, no position in the year): Chronos-2 can infer
    the daily, weekly and yearly patterns from the context. """
    cal, _ = rte_calendar(df)
    dow = df["DateTime"].dt.dayofweek.to_numpy()
    out = pd.DataFrame({"public_holiday": ((dow < 5) & (df["DayOff"] == 1)).astype(float)}, index=df.index)
    for c in ["bridge", "christmas", "winter_holiday", "july_holiday", "august_holiday"]:
        out[c] = cal[c]
    return out


def run(series, cfg, out_dir):
    import torch
    from chronos import Chronos2Pipeline

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = series.df.copy()
    step = pd.Timedelta(series.freq)
    assert (df["DateTime"].diff().dropna() == step).all(), "the series must be regular"

    covariates = list(series.covariates)
    if cfg.get("rte_special_flags", False):
        flags = rte_special_flags(df)
        df = pd.concat([df, flags], axis=1)
        covariates += list(flags.columns)

    H = series.lead_time
    target = series.target
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = Chronos2Pipeline.from_pretrained(cfg["checkpoint"], device_map=device)

    times = df["DateTime"].to_numpy()
    y = df[target].to_numpy()
    position = pd.Series(np.arange(len(df)), index=df["DateTime"])
    chunk = cfg.get("chunk", 64)

    def run_rolling(targets, context_len):
        """For each target row t: context = the context_len rows before o = t - (H - 1),
        future = rows o, ..., t; the forecast for t is kept."""
        results = []
        for start in range(0, len(targets), chunk):
            ctx_parts, fut_parts, ids = [], [], {}
            for t in targets[start:start + chunk]:
                o = t - (H - 1)
                assert o - context_len >= 0, "not enough history for this context length"
                sid = f"s{t:07d}"
                ids[sid] = t
                ctx_parts.append(df.iloc[o - context_len:o][["DateTime", target] + covariates].assign(id=sid))
                fut_parts.append(df.iloc[o:o + H][["DateTime"] + covariates].assign(id=sid))
            with torch.inference_mode():
                out = pipeline.predict_df(
                    pd.concat(ctx_parts, ignore_index=True), future_df=pd.concat(fut_parts, ignore_index=True),
                    prediction_length=H, quantile_levels=[0.1, 0.5, 0.9],
                    id_column="id", timestamp_column="DateTime", target=target)
            last = out.groupby("id", sort=False).tail(1)
            t_rows = last["id"].map(ids).to_numpy()
            assert (pd.to_datetime(last["DateTime"]).to_numpy() == times[t_rows]).all()
            results.append(pd.DataFrame({"DateTime": times[t_rows], "y_true": y[t_rows],
                                         "y_pred": last["predictions"].to_numpy(),
                                         "q10": last["0.1"].to_numpy(), "q50": last["0.5"].to_numpy(),
                                         "q90": last["0.9"].to_numpy()}))
        return pd.concat(results, ignore_index=True).sort_values("DateTime").reset_index(drop=True)

    # Validation: choose the context length
    first_val, first_test = position[series.val_start], position[series.test_start]
    val_targets = np.arange(first_val, first_test, cfg.get("val_stride", 1))
    val_rmse = {}
    for cd in cfg["context_days"]:
        res = run_rolling(val_targets, cd * series.slots_per_day)
        val_rmse[cd] = scores(res["y_true"], res["y_pred"])["RMSE"]
        print(f"  [{series.name}] context {cd} days: validation RMSE {val_rmse[cd]:.4g}")
    best_cd = min(val_rmse, key=val_rmse.get)
    pd.Series(val_rmse, name="val_rmse").rename_axis("context_days").to_csv(out_dir / "validation_rmse.csv")

    # Test: every point, one lead time ahead
    test_targets = np.arange(first_test, len(df), 1)
    result = run_rolling(test_targets, best_cd * series.slots_per_day)
    result.to_csv(out_dir / "predictions_test.csv", index=False)
    json.dump({"context_days": best_cd, "covariates": covariates}, open(out_dir / "choice.json", "w"), indent=1)
    return result
