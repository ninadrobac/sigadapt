"""TiDE baseline (Darts), with the same information set as SigAdapt.

Every test point is forecast with a rolling origin exactly 'lead_time' steps ahead, by keeping only the 
last point of the forecast horizon, with a stride of one step.
Future covariates: the calendar of SigAdapt plus the time of day, and the dataset covariates. 
Target and covariates are scaled with scalers fitted on the training period.

Procedure:
  1. for each look-back and seed: fit on train, forecast the validation period;
  2. choose the look-back with the lowest validation RMSE, averaged over the seeds;
  3. for each seed: refit on train + validation and forecast the test period.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from ..calendar import build_pointwise, time_of_day
from ..metrics import scores, train_scale


def tide_calendar(df, kind, slots_per_day, n_train):
    """Calendar inputs of TiDE: the same calendar as SigAdapt, plus the time of day.
    """
    cal, _ = build_pointwise(df, kind, n_train)
    return pd.concat([cal, time_of_day(df, slots_per_day)], axis=1)


def run(series, cfg, out_dir):
    from darts import TimeSeries, concatenate
    from darts.dataprocessing.transformers import Scaler
    from darts.models import TiDEModel

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = series.df.copy()
    freq, H = series.freq, series.lead_time
    step = pd.Timedelta(freq)
    val_cfg = cfg["validation"]

    cal_df = tide_calendar(df, cfg["calendar"], series.slots_per_day, series.n_train)
    cal_cols = list(cal_df.columns)
    df = pd.concat([df, cal_df], axis=1)

    y = TimeSeries.from_dataframe(df, "DateTime", series.target, freq=freq)
    cov = TimeSeries.from_dataframe(df, "DateTime", series.covariates, freq=freq)
    cal = TimeSeries.from_dataframe(df, "DateTime", cal_cols, freq=freq)

    train_end = series.val_start - step  
    val_end = series.test_start - step

    # Scaler, fitted on the training period only
    scaler_y, scaler_cov = Scaler(), Scaler()
    scaler_y.fit(y.split_after(train_end)[0])
    scaler_cov.fit(cov.split_after(train_end)[0])
    y_s = scaler_y.transform(y)
    future_cov = cal.stack(scaler_cov.transform(cov))

    params = dict(cfg["model_params"])
    if params.get("temporal_width_future") == "all":     # no compression of the future covariates
        params["temporal_width_future"] = len(cal_cols) + len(series.covariates)
    trainer = cfg.get("trainer")
    predict_kwargs = {"batch_size": cfg["predict_batch_size"]} if cfg.get("predict_batch_size") else None

    def make_tide(input_len, seed):
        kw = dict(input_chunk_length=input_len, output_chunk_length=H, random_state=seed,
                  optimizer_kwargs={"lr": cfg["lr"]}, **params)
        if trainer:
            kw["pl_trainer_kwargs"] = trainer
        return TiDEModel(**kw)

    y_true = pd.Series(y.values().ravel(), index=y.time_index)

    def forecast_last_point(model, context, future_cov, first_target, scaler, stride):
        """The last point of each forecast is kept."""
        kw = dict(series=context, future_covariates=future_cov, start=first_target - (H - 1) * step,
                  forecast_horizon=H, stride=stride, retrain=False, last_points_only=True, verbose=False)
        if predict_kwargs:
            kw["predict_kwargs"] = predict_kwargs
        pred_ts = scaler.inverse_transform(model.historical_forecasts(**kw))
        pred = pd.Series(pred_ts.values().ravel(), index=pred_ts.time_index)
        return pred, y_true.loc[pred.index]

    def forecast_all_points(model, context, future_cov, first_target, scaler):
        """Validation variant: consecutive H-step forecasts, all points kept."""
        blocks = model.historical_forecasts(series=context, future_covariates=future_cov,
                                            start=first_target, forecast_horizon=H, stride=H,
                                            retrain=False, last_points_only=False, verbose=False)
        pred_ts = scaler.inverse_transform(concatenate(blocks))
        pred = pd.Series(pred_ts.values().ravel(), index=pred_ts.time_index)
        pred = pred.loc[pred.index.isin(y_true.index)]
        return pred, y_true.loc[pred.index]

    # 1-2. Look-back selection on validation
    lookbacks = [d * series.slots_per_day for d in cfg["lookback_days"]]
    seeds = cfg["seeds"]
    y_s_train = y_s.split_after(train_end)[0]
    y_s_context_val = y_s.split_after(val_end)[0]
    val_rmse = {lb: [] for lb in lookbacks}
    for lb in lookbacks:
        for seed in seeds:
            model = make_tide(lb, seed)
            fit_kw = dict(series=y_s_train, future_covariates=future_cov, verbose=False)
            if val_cfg.get("pass_val_series", False):
                fit_kw |= dict(val_series=y_s.split_after(train_end)[1].split_after(val_end)[0],
                               val_future_covariates=future_cov)
            model.fit(**fit_kw)
            if val_cfg.get("points", "last") == "all":
                pred, truth = forecast_all_points(model, y_s_context_val, future_cov, series.val_start, scaler_y)
            else:
                pred, truth = forecast_last_point(model, y_s_context_val, future_cov, series.val_start, scaler_y,
                                                  stride=val_cfg.get("stride", 1))
            val_rmse[lb].append(root_mean_squared_error(truth, pred))
            print(f"  [{series.name}] look-back {lb} seed {seed}: validation RMSE {val_rmse[lb][-1]:.4g}")

    best_lb = min(lookbacks, key=lambda lb: np.mean(val_rmse[lb]))
    pd.DataFrame(val_rmse, index=seeds).rename_axis("seed").to_csv(out_dir / "validation_rmse.csv")

    # 3. Final models: refit on train + validation, forecast the test period
    y_s_trainval = y_s.split_after(val_end)[0]
    scale_std = train_scale(series)
    rows = []
    for seed in seeds:
        model = make_tide(best_lb, seed)
        model.fit(series=y_s_trainval, future_covariates=future_cov, verbose=False)
        pred, truth = forecast_last_point(model, y_s, future_cov, series.test_start, scaler_y, stride=1)
        pd.DataFrame({"DateTime": pred.index, "y_true": truth.to_numpy(), "y_pred": pred.to_numpy()}) \
          .to_csv(out_dir / f"predictions_test_seed{seed}.csv", index=False)
        rows.append({"seed": seed, **scores(truth, pred, scale=scale_std)})

    table = pd.DataFrame(rows).set_index("seed")
    summary = {"lookback_steps": int(best_lb), **{f"{k}_mean": float(table[k].mean()) for k in table},
               **{f"{k}_sd": float(table[k].std()) for k in table}}
    table.to_csv(out_dir / "test_scores_per_seed.csv")
    json.dump(summary, open(out_dir / "metrics.json", "w"), indent=1)
    return None