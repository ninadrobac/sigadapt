"""Reading and writing results."""

import numpy as np
import pandas as pd


def assemble(store, split):
    """Concatenate the per-slot results into one DataFrame (DateTime, y_true, y_pred) sorted by time.
    split = "tr" (train) or "te" (test)."""
    keys = sorted(store)
    dt = np.concatenate([store[h][f"{split}_dt"] for h in keys])
    true = np.concatenate([store[h][f"{split}_true"] for h in keys])
    pred = np.concatenate([store[h][f"{split}_pred"] for h in keys])
    order = np.argsort(dt)
    return pd.DataFrame({"DateTime": dt[order], "y_true": true[order], "y_pred": pred[order]})


def save_matrices(mats, path):
    """Save a dict slot as a matrix in long format (slot, row, col, value)."""
    frames = []
    for h, m in mats.items():
        m = np.asarray(m)
        r, c = np.indices(m.shape)
        frames.append(pd.DataFrame({"slot": h, "row": r.ravel(), "col": c.ravel(), "value": m.ravel()}))
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def load_matrices(path):
    """Read the saved matrices."""
    df = pd.read_csv(path)
    return {h: g.pivot(index="row", columns="col", values="value").sort_index().sort_index(axis=1).to_numpy()
            for h, g in df.groupby("slot")}


def read_predictions(path):
    """Read a predictions file, indexed by DateTime."""
    return pd.read_csv(path, parse_dates=["DateTime"]).set_index("DateTime").sort_index()
