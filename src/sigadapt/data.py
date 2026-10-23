"""Datasets.

A dataset is described by a YAML file in configs/data/. It may contain several series
(zones, countries, markets); every model is run on each series separately, with the
same splits.

Splits:
    train      : DateTime <  val_start
    validation : val_start <= DateTime < test_start
    test       : DateTime >= test_start
Hyperparameters are selected on validation after fitting on train; the final model is
refitted on train + validation and evaluated on test.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class Series:
    dataset: str
    name: str                 # series identifier, e.g. a zone; equal to the dataset name if single
    df: pd.DataFrame         
    target: str
    covariates: list
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    freq: str
    slots_per_day: int
    lead_time_days: int
    slot_column: str = None   # column giving the slot of the day; computed from DateTime if None


    @property
    def n_train(self):
        """Number of rows before the test period (train + validation)."""
        return int((self.df["DateTime"] < self.test_start).sum())

    @property
    def lead_time(self):
        """Lead time in number of time steps."""
        return self.lead_time_days * self.slots_per_day

    def slots(self):
        if self.slot_column is not None:
            return self.df[self.slot_column].to_numpy()
        return slot_of_day(pd.DatetimeIndex(self.df["DateTime"]), self.slots_per_day)


def slot_of_day(index, slots_per_day):
    minutes = index.hour * 60 + index.minute
    return (minutes // (24 * 60 // slots_per_day)).to_numpy()


def load_dataset(data_cfg, root="."):
    """Load every series of a dataset. Returns a list of Series."""
    root = Path(root)
    names = data_cfg.get("series") or [data_cfg["name"]]
    out = []
    for name in names:
        path = root / data_cfg["file"].format(series=name)
        df = pd.read_csv(path)
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df = df.sort_values("DateTime").reset_index(drop=True)

        end = (data_cfg.get("end_per_series") or {}).get(name, data_cfg.get("end"))
        if end is not None:
            df = df[df["DateTime"] < pd.Timestamp(end)].reset_index(drop=True)

        target = data_cfg["target"]
        covariates = data_cfg["covariates"]
        if covariates == "auto":   # every other column, in file order
            covariates = [c for c in df.columns if c not in ("DateTime", target)]

        val_start, test_start = _splits(df, data_cfg["splits"])
        out.append(Series(dataset=data_cfg["name"], name=str(name), df=df, target=target,
                          covariates=list(covariates), val_start=val_start, test_start=test_start,
                          freq=data_cfg["freq"], slots_per_day=data_cfg["slots_per_day"],
                          lead_time_days=data_cfg["lead_time_days"],
                          slot_column=data_cfg.get("slot_column")))
    return out


def _splits(df, splits):
    """Absolute dates, or periods counted back from the end of the data (EPF)."""
    if "test_start" in splits:
        return pd.Timestamp(splits["val_start"]), pd.Timestamp(splits["test_start"])
    data_end = df["DateTime"].max()
    test_start = data_end - pd.DateOffset(years=splits["test_years"])
    val_start = test_start - pd.DateOffset(years=splits["val_years"])
    return val_start, test_start
