"""GAM and GAMAdapt (GAM + Kalman filter), RTE.

One GAM per slot of the day, fitted on the rows before the test period. Two sets of test
predictions are saved:
  predictions_test_static.csv : the static GAM;
  predictions_test.csv        : GAMAdapt, where the intercept, the temperature terms
                                (temperature and the two smoothed temperatures) and the past
                                loads are re-weighted online by a Kalman filter that starts
                                from the static GAM; calendar terms held fixed; no update on
                                special days, as in SigAdapt.
Same calendar variables as SigAdapt; same information set (load used from one lead time before
the target).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pygam import LinearGAM, f, s
from viking_kalman import iterative_grid_search

from ..calendar import rte_calendar
from ..config import grid
from ..kalman import KalmanRegression


def exponential_smoothing(series, smooth_param):
    """x_s[t] = a x[t] + (1 - a) x_s[t-1], started at x_s[0] = x[0]."""
    series = np.asarray(series, dtype=float)
    smoothed = np.zeros(len(series))
    smoothed[0] = series[0]
    for t in range(1, len(series)):
        smoothed[t] = smooth_param * series[t] + (1 - smooth_param) * smoothed[t - 1]
    return smoothed


def run(series, cfg, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = series.df
    spd, n_train = series.slots_per_day, series.n_train
    kalman_delay = series.lead_time_days
    temp_col, load_col = series.covariates[0], series.target

    cal, special = rte_calendar(df)

    g = df.copy()
    g["dow"] = g["DateTime"].dt.dayofweek
    flags = ["is_workday", "bridge", "christmas", "winter_holiday", "july_holiday", "august_holiday"]
    for c in flags:
        g[c] = cal[c]
    g["load_2"] = g[load_col].shift(series.lead_time)              # same slot, one lead time before
    g["load_7"] = g[load_col].shift(spd * 7)                       # same slot, one week before
    a_slow, a_fast = cfg["smoothing"]
    g["temp_s99"] = exponential_smoothing(g[temp_col], a_slow)
    g["temp_s95"] = exponential_smoothing(g[temp_col], a_fast)

    feat_cols = ["year_position", temp_col, "dow", *flags, "load_2", "load_7", "temp_s99", "temp_s95"]
    terms = (s(0, n_splines=20, basis="cp")                 # position in the year, cyclic
             + s(1, n_splines=10)                           # temperature
             + f(2)                                         # day of week
             + f(3) + f(4) + f(5) + f(6) + f(7) + f(8)      # calendar flags
             + s(9, n_splines=10) + s(10, n_splines=10)     # load one lead time and one week before
             + s(11, n_splines=10) + s(12, n_splines=10))   # smoothed temperatures
    drift_terms = [feat_cols.index(c) for c in [temp_col, "temp_s99", "temp_s95", "load_2", "load_7"]]

    # one smoothing parameter per candidate, shared by all terms except the bridge (almost unpenalised)
    lams = grid(cfg["lams"])
    c_cal = cfg.get("c_cal", 1.0)
    lam_grid = np.tile(lams[:, None], (1, len(feat_cols)))
    lam_grid[:, feat_cols.index("bridge")] = lams / c_cal ** 2
    q_list = list(grid(cfg["kalman"]["q_grid"]))

    slots = series.slots()
    complete = g[feat_cols].notna().all(axis=1).to_numpy()   # the first week has no load_7
    rows = np.arange(len(g))
    store = {}
    for h in range(spd):
        tr = (rows < n_train) & (slots == h) & complete
        te = (rows >= n_train) & (slots == h) & complete
        xtr, ytr = g.loc[tr, feat_cols].to_numpy(), g.loc[tr, load_col].to_numpy()
        xte, yte = g.loc[te, feat_cols].to_numpy(), g.loc[te, load_col].to_numpy()

        # static GAM, smoothing parameters chosen by GCV
        gam = LinearGAM(terms).gridsearch(xtr, ytr, objective="GCV", lam=lam_grid, progress=False)
        te_pred_static = gam.predict(xte)

        # Kalman design: one column per GAM term (its partial effect), plus an intercept
        x_all = np.vstack([xtr, xte])
        y_all = np.concatenate([ytr, yte])
        n, n_tr = len(y_all), len(ytr)
        y_mu, y_sd = ytr.mean(), ytr.std()
        y_all_s = (y_all - y_mu) / y_sd
        x_kal = np.column_stack([np.asarray(gam.partial_dependence(term=col, X=x_all)).ravel() / y_sd
                                 for col in range(len(feat_cols))])
        x_kal = np.hstack([np.ones((n, 1)), x_kal])
        d = x_kal.shape[1]
        theta_static = np.concatenate([[(gam.coef_[-1] - y_mu) / y_sd], np.ones(len(feat_cols))])
        assert np.allclose(x_kal @ theta_static * y_sd + y_mu, gam.predict(x_all))

        kal_idx = [0] + [c + 1 for c in drift_terms]
        fix_idx = [c + 1 for c in range(len(feat_cols)) if c not in drift_terms]
        res = iterative_grid_search(x_kal[:n_tr], y_all_s[:n_tr], q_list=q_list, Q_init=np.zeros((d, d)),
                                    restrict=kal_idx, delay=kalman_delay, p1=1, verbose=False)
        p0, q0, r0 = np.array(res["P"], dtype=float), np.asarray(res["Q"]), res["sig"] ** 2
        p0[fix_idx, :] = 0.0                    # fixed coefficients do not drift
        p0[:, fix_idx] = 0.0
        kalman = KalmanRegression(n_features=d, Q=q0, R=r0, P_init=p0, beta_init=theta_static)
        special_seq = np.concatenate([special[tr], special[te]])
        y_pred_s = np.zeros(n)
        for j in range(n):
            if j >= kalman_delay and not special_seq[j - kalman_delay]:
                kalman.update(x_kal[j - kalman_delay], y_all_s[j - kalman_delay])
            y_pred_s[j] = kalman.predict(x_kal[j])

        store[h] = {"te_dt": g.loc[te, "DateTime"].to_numpy(), "te_true": yte,
                    "static": te_pred_static, "adapt": (y_pred_s * y_sd + y_mu)[n_tr:]}
        print(f"  [{series.name}] slot {h} done")

    def collect(key):
        return (pd.concat([pd.DataFrame({"DateTime": st["te_dt"], "y_true": st["te_true"], "y_pred": st[key]})
                           for st in store.values()])
                  .sort_values("DateTime").reset_index(drop=True))

    static, adapt = collect("static"), collect("adapt")
    static.to_csv(out_dir / "predictions_test_static.csv", index=False)
    adapt.to_csv(out_dir / "predictions_test.csv", index=False)
    return adapt
