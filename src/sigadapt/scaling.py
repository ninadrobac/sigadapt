"""Scalers fitted on the rows before the test period.

"standard"  : (x - mean) / sd                       (sklearn StandardScaler)
"invariant" : asinh((x - median) / MAD)             (Uniejewski et al., 2018; as in epftoolbox)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.robust import mad


class InvariantScaler:
    """Median/MAD normalisation followed by asinh, as epftoolbox's DataScaler('Invariant')."""

    def fit(self, x):
        x = np.asarray(x)
        self.median = np.median(x, axis=0)
        self.mad = mad(x, axis=0)
        return self

    def transform(self, x):
        return np.arcsinh((np.asarray(x) - self.median) / self.mad)

    def inverse_transform(self, x):
        return np.sinh(np.asarray(x)) * self.mad + self.median


def make_scaler(kind):
    if kind == "standard":
        return StandardScaler()
    if kind == "invariant":
        return InvariantScaler()
    raise ValueError(f"unknown scaler '{kind}'")


def scale_columns(values, n_fit, kind):
    """Scale one series (1-D array) with a scaler fitted on its first n_fit values.
    Returns (scaled values, fitted scaler)."""
    scaler = make_scaler(kind).fit(values[:n_fit].reshape(-1, 1))
    return scaler.transform(values.reshape(-1, 1)).ravel(), scaler
