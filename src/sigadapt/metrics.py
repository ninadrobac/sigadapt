"""Error metrics and Diebold-Mariano test."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             root_mean_squared_error)

def scores(y_true, y_pred, scale=None):
    out = {"RMSE": root_mean_squared_error(y_true, y_pred),
           "MAE": mean_absolute_error(y_true, y_pred),
           "MAPE": 100 * mean_absolute_percentage_error(y_true, y_pred)}
    if scale is not None:                       
        out["RMSE_stand"] = out["RMSE"] / scale
        out["MSE_stand"] = out["RMSE_stand"] ** 2
        out["MAE_stand"] = out["MAE"] / scale
    return out

def train_scale(series):
    """Standard deviation of the target on the training period."""
    df = series.df
    return df.loc[df["DateTime"] < series.val_start, series.target].std()


def diebold_mariano(loss_a, loss_b, lag=3, one_sided=False):
    """Diebold-Mariano test on the daily-averaged loss differential, Newey-West variance.

    loss_a, loss_b : pandas Series indexed by DateTime.
    A negative statistic means that model a is more accurate.
    one_sided=False : two-sided p-value;
    one_sided=True  : p-value of H0: "Model a is not more accurate than model b".
    """
    d = (loss_a - loss_b).groupby(loss_a.index.normalize()).mean().to_numpy()
    T, mean_d = len(d), d.mean()
    c = d - mean_d
    gamma = [c @ c / T] + [c[k:] @ c[:-k] / T for k in range(1, lag + 1)]
    long_run_var = gamma[0] + 2 * sum((1 - k / (lag + 1)) * gamma[k] for k in range(1, lag + 1))
    dm = mean_d / np.sqrt(long_run_var / T)
    p = stats.norm.cdf(dm) if one_sided else 2 * stats.norm.sf(abs(dm))
    return dm, p
