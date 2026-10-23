"""Linear baseline without signatures (ablation of SigAdapt, RTE).

Same pipeline as SigAdapt (one model per slot of the day, calendar, ridge with the bridge
unpenalised, Kalman filter with the same tuning, delay and pause on special days), but the
signatures are replaced by raw inputs:

  linear      : covariate at t and target one lead time before, plus the calendar; Kalman filter;
  linear_T{k} : the same, plus the covariate k days before, for k in `covariate_lags_days`
                (the SigAdapt windows); these experts are aggregated with BOA, as in SigAdapt.

In the Kalman filter, the intercept and the coefficients of the covariate and target drift.
The calendar is treated as in SigAdapt: 'calendar: residual' holds its effect fixed at the ridge
value and filters the residual; 'calendar: in_state' keeps it in the state without drift.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from viking_kalman import iterative_grid_search

from ..calendar import build_pointwise
from ..config import grid
from ..io import assemble
from ..kalman import KalmanRegression
from ..sigadapt import aggregate


def run(series, cfg, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = series.df
    n_train, spd = series.n_train, series.slots_per_day
    kalman_delay = series.lead_time_days
    is_val = ((df["DateTime"] >= series.val_start) & (df["DateTime"] < series.test_start)).to_numpy()

    alphas = grid(cfg["ridge"]["alphas"])
    q_list = list(grid(cfg["kalman"]["q_grid"]))
    cal, special = build_pointwise(df, cfg.get("pointwise", {}).get("calendar", "rte"), n_train)
    if special is None or not cfg["kalman"].get("skip_special_days", False):
        special = np.zeros(len(df), dtype=bool)
    cal_scale = np.where(cal.columns.isin(cfg["ridge"].get("unpenalised", [])),
                         cfg["ridge"].get("c_cal", 1.0), 1.0)
    cal_np = cal.to_numpy()

    in_state = cfg["kalman"].get("calendar", "residual") == "in_state"
    lags = cfg["covariate_lags_days"]
    experts = {"linear": []} | {f"linear_T{k}": [k] for k in lags}

    cov = df[series.covariates[0]]
    target = df[series.target]

    def inputs(lag_days):
        cols = {"temp": cov, "load_2": target.shift(series.lead_time)}
        for k in lag_days:
            cols[f"temp_{k}"] = cov.shift(spd * k)
        return pd.DataFrame(cols)

    y_scaler = StandardScaler().fit(target[:n_train].to_numpy().reshape(-1, 1))
    y_s = y_scaler.transform(target.to_numpy().reshape(-1, 1)).ravel()
    slots = series.slots()
    rows = np.arange(len(df))
    dt = df["DateTime"].to_numpy()

    def to_orig(v):
        return y_scaler.inverse_transform(v.reshape(-1, 1)).ravel()

    kalman_preds = {}
    for name, lag_days in experts.items():
        print(f"  [{series.name}] expert {name}")
        x_cont = inputs(lag_days)
        complete = x_cont.notna().all(axis=1).to_numpy()
        n_cont = x_cont.shape[1]
        store, preds = {}, {}
        for h in range(spd):
            tr = (rows < n_train) & (slots == h) & complete
            te = (rows >= n_train) & (slots == h) & complete
            scaler = StandardScaler().fit(x_cont.to_numpy()[tr])

            def design(m):
                return np.hstack([scaler.transform(x_cont.to_numpy()[m]), cal_np[m]])

            def joint(a):
                return np.hstack([a[:, :n_cont], cal_scale * a[:, n_cont:]])

            xtr, xte = design(tr), design(te)
            ytr, yte = y_s[tr], y_s[te]
            val = is_val[tr]

            best_alpha, best_rmse = alphas[0], np.inf
            for a in alphas:
                r = Ridge(alpha=a).fit(joint(xtr[~val]), ytr[~val])
                rmse = root_mean_squared_error(ytr[val], r.predict(joint(xtr[val])))
                if rmse < best_rmse:
                    best_rmse, best_alpha = rmse, a

            ridge = Ridge(alpha=best_alpha).fit(joint(xtr), ytr)
            coef = ridge.coef_.copy()
            coef[n_cont:] *= cal_scale
            theta_static = np.concatenate([[ridge.intercept_], coef])

            x_all = np.vstack([xtr, xte])
            x_full = np.hstack([np.ones((len(x_all), 1)), x_all])     # [1 | inputs | calendar]
            static_pred = x_full @ theta_static                      # static ridge forecast
            y_seq = np.concatenate([ytr, yte])
            n, n_tr = len(y_seq), len(ytr)

            if in_state:   # calendar in the state (no drift); intercept and inputs drift
                x_seq, offset, beta0 = x_full, np.zeros(n), theta_static
                restrict = list(range(n_cont + 1))
            else:          # residual: calendar effect fixed at its ridge value
                x_seq = x_full[:, :n_cont + 1]
                offset = x_all[:, n_cont:] @ coef[n_cont:]
                beta0, restrict = theta_static[:n_cont + 1], None
            y_kal = y_seq - offset
            d = x_seq.shape[1]

            res = iterative_grid_search(x_seq[:n_tr], y_kal[:n_tr], q_list=q_list, Q_init=np.zeros((d, d)),
                                        restrict=restrict, delay=kalman_delay, p1=1, verbose=False)
            kalman = KalmanRegression(n_features=d, Q=np.asarray(res["Q"]), R=res["sig"] ** 2,
                                      P_init=np.asarray(res["P"]), beta_init=beta0)
            special_seq = np.concatenate([special[tr], special[te]])
            y_pred = np.zeros(n)
            for j in range(n):
                if j >= kalman_delay and not special_seq[j - kalman_delay]:
                    kalman.update(x_seq[j - kalman_delay], y_kal[j - kalman_delay])
                y_pred[j] = kalman.predict(x_seq[j]) + offset[j]

            store[h] = {"tr_true": to_orig(ytr), "tr_pred": to_orig(static_pred[:n_tr]), "tr_dt": dt[tr],
                        "te_true": to_orig(yte), "te_pred": to_orig(static_pred[n_tr:]), "te_dt": dt[te]}
            preds[h] = {"tr_true": to_orig(ytr), "tr_pred": to_orig(y_pred[:n_tr]), "tr_dt": dt[tr],
                        "te_true": to_orig(yte), "te_pred": to_orig(y_pred[n_tr:]), "te_dt": dt[te]}

        assemble(store, "te").to_csv(out_dir / f"predictions_test_batch_{name}.csv", index=False)
        kalman_preds[name] = assemble(preds, "te")
        kalman_preds[name].to_csv(out_dir / f"predictions_test_kalman_{name}.csv", index=False)

    kalman_preds["linear"].to_csv(out_dir / "predictions_test_single.csv", index=False)
    agg, weights = aggregate({k: v for k, v in kalman_preds.items() if k != "linear"},
                             spd, series.lead_time_days)
    agg.to_csv(out_dir / "predictions_test.csv", index=False)
    weights.to_csv(out_dir / "mixture_weights.csv", index=False)
    return agg