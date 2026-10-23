"""SigAdapt: linear model on sliding-window signatures, adapted online by a Kalman filter,
with online aggregation of the experts obtained for several window lengths.

For each window length w (one expert per w):
  1. Signature features. Each signature block is the signature of a path (time, channels...)
     over a window of w steps, ending either at the forecast time or at the forecast origin.
  2. Linear model: ridge regression on the standardised signature features and the pointwise features 
     (e.g. calendar); the ridge penalty is selected on the validation period.
  3. Per-level decomposition: the signature part of the ridge prediction is split by contribution per level;
     these become the states of a Kalman filter that starts from the ridge solution and tracks their drift. 
     Two variants:
       calendar: in_state  - the calendar features are also in the state (diagonal Q element is set to 0);
       calendar: residual  - the calendar effect is held fixed at its ridge value and the
                             filter runs on the residual.
     The filter is updated with each observation once it is known, i.e. one lead time later.
  4. Aggregation of the experts (one per window) with Bernstein Online Aggregation (opera),
     one mixture per slot of the day, with the weights delayed by the lead time.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from opera import Mixture
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from viking_kalman import iterative_grid_search

from .calendar import build_pointwise
from .config import grid
from .io import assemble, save_matrices
from .kalman import KalmanRegression
from .scaling import scale_columns
from .signatures import keep_mask, sig_sliding_basepoint, signature_words


# ---------------------------------------------------------------------------
# Signature blocks
# ---------------------------------------------------------------------------

def _channel_names(block, series):
    names = []
    for c in block["channels"]:
        if c == "target":
            names.append(series.target)
        elif c == "covariates":
            names.extend(series.covariates)
        else:
            names.append(c)
    return names


def _letters(names, block_names, series):
    out = []
    for c in names:
        if c == "time":
            out.append(0)
        elif c == "target":
            out.append(1 + block_names.index(series.target))
        else:
            out.append(1 + block_names.index(c))
    return out

# Which signature components to keep
def block_layout(block, series, depth):
    names = _channel_names(block, series)
    words = signature_words(len(names) + 1, depth)
    keep = keep_mask(words,
                     drop_pure=_letters(block.get("drop_pure", []), names, series),
                     require_any=_letters(block.get("require_any", []), names, series))
    levels = np.array([len(w) for w in words])[keep]
    return names, keep, levels


# ---------------------------------------------------------------------------
# One expert (one window length)
# ---------------------------------------------------------------------------

def run_expert(series, cfg, w, scaled, y_scaler, pointwise, special, out_dir):
    """Ridge per slot, then Kalman filter per slot, for the window length w (in steps)."""
    sig_cfg, ridge_cfg, kal_cfg = cfg["signature"], cfg["ridge"], cfg["kalman"]
    delay = series.lead_time
    kalman_delay = series.lead_time_days        
    n_train = series.n_train
    df = series.df
    y_scaled, y_raw = scaled[series.target], df[series.target].to_numpy()

    # Signatures
    blocks, path_ids, level_ids = [], [], []
    for p, block in enumerate(sig_cfg["blocks"]):
        depth = block["depth"] if "depth" in block else sig_cfg["depth"]
        names, keep, levels = block_layout(block, series, depth)
        path = np.column_stack([scaled[c] for c in names])
        sig = sig_sliding_basepoint(path, w, trunc_order=depth, drop=False)[:, keep]
        lag = delay if block["anchor"] == "origin" else 0
        blocks.append((sig, lag))
        path_ids.append(np.full(len(levels), p))
        level_ids.append(levels)
    path_ids, level_ids = np.concatenate(path_ids), np.concatenate(level_ids)
    groups = [(p, k) for p in range(len(blocks)) for k in np.unique(level_ids[path_ids == p])]
    n_groups = len(groups)

    def sig_rows(start, stop):
        # alignment of the outlook and history signature
        return np.hstack([sig[start - lag: stop - lag] for sig, lag in blocks])

    n = len(df)
    tr_rows, te_rows = slice(w + delay, n_train), slice(n_train, n)
    sig_train, sig_test = sig_rows(w + delay, n_train), sig_rows(n_train, n)
    cal_np = pointwise.to_numpy()
    cal_train, cal_test = cal_np[tr_rows], cal_np[te_rows]
    n_sig = sig_train.shape[1]

    # calendar columns unpenalised 
    unpen = ridge_cfg.get("unpenalised", [])
    cal_scale = np.where(pointwise.columns.isin(unpen), ridge_cfg.get("c_cal", 1.0), 1.0)

    y_train, y_true_train = y_scaled[tr_rows], y_raw[tr_rows]
    y_test, y_true_test = y_scaled[te_rows], y_raw[te_rows]
    slots = series.slots()
    slot_train, slot_test = slots[tr_rows], slots[te_rows]
    dt = df["DateTime"].to_numpy()
    dt_train, dt_test = dt[tr_rows], dt[te_rows]
    is_val = ((df["DateTime"] >= series.val_start) & (df["DateTime"] < series.test_start)).to_numpy()
    val_train = is_val[tr_rows]
    if special is None:         # special = ((dow < 5) & public_holiday) | bridge_any | xmas_period
        special = np.zeros(len(df), dtype=bool)
    special_train, special_test = special[tr_rows], special[te_rows]

    alphas = grid(ridge_cfg["alphas"])
    in_state = kal_cfg["calendar"] == "in_state"

    def inv(v):
        return y_scaler.inverse_transform(np.asarray(v).reshape(-1, 1)).ravel()

    # Ridge per slot
    store = {}
    for h in range(series.slots_per_day):
        tr, te = slot_train == h, slot_test == h
        str_raw, ste_raw = sig_train[tr], sig_test[te]
        ctr, cte = cal_train[tr], cal_test[te]
        ytr = y_train[tr]
        val = val_train[tr]

        sig_scaler = StandardScaler().fit(str_raw)
        xtr_j = np.hstack([sig_scaler.transform(str_raw), cal_scale * ctr])

        # penalty chosen on validation (fit on train), then refit on train + validation
        best_alpha, best_rmse = alphas[0], np.inf
        for a in alphas:
            r = Ridge(alpha=a).fit(xtr_j[~val], ytr[~val])
            rmse = root_mean_squared_error(ytr[val], r.predict(xtr_j[val]))
            if rmse < best_rmse:
                best_rmse, best_alpha = rmse, a

        ridge = Ridge(alpha=best_alpha).fit(xtr_j, ytr)
        beta_sig = ridge.coef_[:n_sig]
        beta_cal = ridge.coef_[n_sig:] * cal_scale

        def levels(s_raw):
            contrib = sig_scaler.transform(s_raw) * beta_sig
            return np.column_stack([contrib[:, (path_ids == p) & (level_ids == k)].sum(axis=1)
                                    for (p, k) in groups])

        if in_state:
            xtr, xte = np.hstack([levels(str_raw), ctr]), np.hstack([levels(ste_raw), cte])
            theta_static = np.concatenate([[ridge.intercept_], np.ones(n_groups), beta_cal])
            tr_pred = inv(theta_static[0] + xtr @ theta_static[1:])
            te_pred = inv(theta_static[0] + xte @ theta_static[1:])
            off_tr, off_te = np.zeros(len(xtr)), np.zeros(len(xte))
        else:
            xtr, xte = levels(str_raw), levels(ste_raw)
            theta_static = np.concatenate([[ridge.intercept_], np.ones(n_groups)])
            xte_j = np.hstack([sig_scaler.transform(ste_raw), cal_scale * cte])
            tr_pred, te_pred = inv(ridge.predict(xtr_j)), inv(ridge.predict(xte_j))
            off_tr, off_te = ctr @ beta_cal, cte @ beta_cal

        store[h] = {"xtr": xtr, "xte": xte, "ytr": ytr, "yte": y_test[te],
                    "off_tr": off_tr, "off_te": off_te, "theta_static": theta_static,
                    "alpha": best_alpha,
                    "tr_true": y_true_train[tr], "tr_pred": tr_pred,
                    "te_true": y_true_test[te], "te_pred": te_pred,
                    "tr_dt": dt_train[tr], "te_dt": dt_test[te],
                    "tr_special": special_train[tr], "te_special": special_test[te]}

    pd.Series({h: store[h]["alpha"] for h in store}, name="alpha") \
      .rename_axis("slot").to_csv(out_dir / f"alpha_{w}.csv")
    assemble(store, "tr").to_csv(out_dir / f"predictions_train_batch_{w}.csv", index=False)
    assemble(store, "te").to_csv(out_dir / f"predictions_test_batch_{w}.csv", index=False)

    # Kalman filter per slot; variances tuned by grid search on the rows before the test period
    q_list = list(grid(kal_cfg["q_grid"]))
    skip_special = kal_cfg.get("skip_special_days", False)
    preds, theta_kal, p_kal, q_kal, r_kal = {}, {}, {}, {}, {}
    for h, s in store.items():
        x_seq = np.vstack([s["xtr"], s["xte"]])
        x_seq = np.hstack([np.ones((len(x_seq), 1)), x_seq])
        off_seq = np.concatenate([s["off_tr"], s["off_te"]])
        y_seq = np.concatenate([s["ytr"], s["yte"]]) - off_seq
        d, n, n_tr = x_seq.shape[1], len(y_seq), len(s["ytr"])

        restrict = list(range(n_groups + 1)) if in_state else None   # intercept + levels drift
        res = iterative_grid_search(x_seq[:n_tr], y_seq[:n_tr], q_list=q_list,
                                    Q_init=np.zeros((d, d)), restrict=restrict,
                                    delay=kalman_delay, p1=1, verbose=False)
        theta_kal[h], p_kal[h], q_kal[h], r_kal[h] = (s["theta_static"], np.asarray(res["P"]),
                                                      np.asarray(res["Q"]), res["sig"] ** 2)

        kalman = KalmanRegression(n_features=d, Q=q_kal[h], R=r_kal[h],
                                  P_init=p_kal[h], beta_init=theta_kal[h])
        special_seq = np.concatenate([s["tr_special"], s["te_special"]])

        y_pred = np.zeros(n)
        for j in range(n):
            if j >= kalman_delay and not (skip_special and special_seq[j - kalman_delay]):
                kalman.update(x_seq[j - kalman_delay], y_seq[j - kalman_delay])
            y_pred[j] = kalman.predict(x_seq[j]) + off_seq[j]

        y_pred = inv(y_pred)
        preds[h] = {"tr_true": s["tr_true"], "tr_pred": y_pred[:n_tr],
                    "te_true": s["te_true"], "te_pred": y_pred[n_tr:],
                    "tr_dt": s["tr_dt"], "te_dt": s["te_dt"]}

    save_matrices(p_kal, out_dir / f"P_kal_{w}.csv")
    save_matrices(q_kal, out_dir / f"Q_kal_{w}.csv")
    pd.DataFrame.from_dict(theta_kal, orient="index").sort_index() \
      .rename_axis("slot").to_csv(out_dir / f"theta_kal_{w}.csv")
    pd.DataFrame({"R": pd.Series(r_kal)}).sort_index() \
      .rename_axis("slot").to_csv(out_dir / f"R_kal_{w}.csv")
    assemble(preds, "tr").to_csv(out_dir / f"predictions_train_kalman_{w}.csv", index=False)
    test = assemble(preds, "te")
    test.to_csv(out_dir / f"predictions_test_kalman_{w}.csv", index=False)
    return test


# ---------------------------------------------------------------------------
# Aggregation of the experts
# ---------------------------------------------------------------------------

def aggregate(expert_preds, slots_per_day, lead_time_days):
    """BOA mixture per slot of the day.

    expert_preds : DataFrame(DateTime, y_true, y_pred)
    opera's weights are shifted by (lead_time_days - 1) rows so each weight only uses observations available 
    at the forecast origin.
    """
    preds, truths = {}, {}
    for name, f in expert_preds.items():
        f = f.set_index("DateTime").sort_index()
        preds[name], truths[name] = f["y_pred"], f["y_true"]
    experts = pd.concat(preds, axis=1).dropna()
    y_true_all = pd.concat(truths, axis=1).loc[experts.index]
    assert (y_true_all.nunique(axis=1) <= 1).all(), "y_true disagrees across experts"
    y_obs = y_true_all.iloc[:, 0]

    from .data import slot_of_day
    slot = slot_of_day(experts.index, slots_per_day)
    lag = lead_time_days - 1

    pred_parts, weight_parts = [], []
    for h in range(slots_per_day):
        rows = slot == h
        x_h, y_h = experts[rows], y_obs[rows]
        mix = Mixture(y=y_h, experts=x_h, model="BoA", loss_type="mse", loss_gradient=True)
        w_mat = np.asarray(mix.weights)
        w_h = w_mat[np.maximum(np.arange(len(w_mat)) - lag, 0)]    # first rows: initial weights
        pred_parts.append(pd.Series((w_h * x_h.to_numpy()).sum(axis=1), index=x_h.index))
        weight_parts.append(pd.DataFrame(w_h, index=x_h.index, columns=x_h.columns).assign(slot=h))

    y_pred = pd.concat(pred_parts).sort_index()
    result = pd.DataFrame({"DateTime": y_pred.index, "y_true": y_obs.loc[y_pred.index].to_numpy(),
                           "y_pred": y_pred.to_numpy()})
    weights = pd.concat(weight_parts).sort_index().rename_axis("DateTime").reset_index()
    return result, weights


# ---------------------------------------------------------------------------
# Full model on one series
# ---------------------------------------------------------------------------

def run(series, cfg, out_dir):
    """Run SigAdapt on one series; returns the test predictions of the aggregated model."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sig_cfg = cfg["signature"]
    n_train = series.n_train

    # Standardise every channel that enters a path (scalers fitted on train set)
    kind = cfg.get("scaling", "standard")
    channels = {series.target}
    for block in sig_cfg["blocks"]:
        channels.update(_channel_names(block, series))
    scaled, scalers = {}, {}
    for c in sorted(channels):
        scaled[c], scalers[c] = scale_columns(series.df[c].to_numpy(), n_train, kind)

    pw_cfg = cfg.get("pointwise", {})
    pointwise, special = build_pointwise(series.df, pw_cfg.get("calendar", "generic"), n_train,
                                         extra=pw_cfg.get("extra", []))

    expert_preds = {}
    for wd in sig_cfg["window_days"]:
        w = wd * series.slots_per_day
        print(f"  [{series.name}] window {wd} days")
        expert_preds[f"w{wd}"] = run_expert(series, cfg, w, scaled, scalers[series.target],
                                             pointwise, special, out_dir)

    result, weights = aggregate(expert_preds, series.slots_per_day, series.lead_time_days)
    weights.to_csv(out_dir / "mixture_weights.csv", index=False)
    result.to_csv(out_dir / "predictions_test.csv", index=False)
    json.dump({"pointwise_features": list(pointwise.columns)},
              open(out_dir / "features.json", "w"), indent=1)
    return result
