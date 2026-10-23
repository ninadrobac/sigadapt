"""Calendar variables.

`rte_calendar`     : French calendar used for the RTE case study.
`generic_calendar` : position in the year and day-of-week dummies (fev-bench datasets).
"""

import numpy as np
import pandas as pd


def rte_calendar(df):
    """Calendar design for the RTE data, one row per half-hour. For day-of-week dummy variables, Monday is 
    the reference day).

    Columns:
      sin_poy, cos_poy   annual seasonality
      dow_1 ... dow_6    day-of-week dummies
      is_workday         Monday-Friday and not a public holiday
      bridge             working day between two days off, outside the end-of-year break
      christmas          24 Dec - 1 Jan, Monday-Friday only
      winter_holiday     school Christmas break (all zones) outside 24 Dec - 1 Jan
      july_holiday       school summer holidays (all zones) in July
      august_holiday     school summer holidays (all zones) in August

    Also returns `special`: the days on which the Kalman filter does not update
    (public holiday on a weekday, any bridge, the whole 24 Dec - 1 Jan period).

    Requires the columns DateTime, year_position, DayOff, vacancesZoneA/B/C.
    """
    dt = df["DateTime"]
    day = dt.dt.normalize()
    dow = dt.dt.dayofweek.to_numpy()
    month, dom = dt.dt.month.to_numpy(), dt.dt.day.to_numpy()
    public_holiday = df["DayOff"].to_numpy() == 1
    all_zones = ((df["vacancesZoneA"] == 1) & (df["vacancesZoneB"] == 1)
                 & (df["vacancesZoneC"] == 1)).to_numpy()

    # school holidays in all zones
    zones_day = pd.Series(all_zones, index=day).groupby(level=0).max()
    school_day = zones_day & zones_day.shift(-1, fill_value=False)
    school = school_day.reindex(day).to_numpy(dtype=bool)

    workday = (dow < 5) & ~public_holiday
    xmas_period = ((month == 12) & (dom >= 24)) | ((month == 1) & (dom == 1))
    xmas_school = school & np.isin(month, [12, 1]) & ~xmas_period

    # bridge: a working day with a day off on both sides (computed on daily values)
    off = pd.Series(~workday, index=day).groupby(level=0).max()
    off_prev = off.reindex(off.index - pd.Timedelta(days=1), fill_value=False).to_numpy()
    off_next = off.reindex(off.index + pd.Timedelta(days=1), fill_value=False).to_numpy()
    bridge_any = (pd.Series(~off.to_numpy() & off_prev & off_next, index=off.index)
                  .reindex(day).to_numpy(dtype=bool))
    bridge = bridge_any & ~xmas_period & ~xmas_school

    cal = {}
    poy = 2 * np.pi * df["year_position"].to_numpy()
    cal["sin_poy"], cal["cos_poy"] = np.sin(poy), np.cos(poy)
    for k in range(1, 7):
        cal[f"dow_{k}"] = (dow == k).astype(float)
    cal["is_workday"] = workday.astype(float)
    cal["bridge"] = bridge.astype(float)
    cal["christmas"] = (xmas_period & (dow < 5)).astype(float)
    cal["winter_holiday"] = xmas_school.astype(float)
    cal["july_holiday"] = (school & (month == 7)).astype(float)
    cal["august_holiday"] = (school & (month == 8)).astype(float)

    special = ((dow < 5) & public_holiday) | bridge_any | xmas_period
    return pd.DataFrame(cal, index=df.index), special


def generic_calendar(df):
    """Position in the year (sine and cosine) and day-of-week dummies (Monday dropped)."""
    idx = pd.DatetimeIndex(df["DateTime"])
    year_start = pd.to_datetime(idx.year.astype(str) + "-01-01")
    doy = (idx - pd.DatetimeIndex(year_start)) / np.timedelta64(1, "D")
    year_len = np.where(idx.is_leap_year, 366.0, 365.0)
    poy = (doy / year_len).to_numpy()
    dow = pd.get_dummies(idx.dayofweek, prefix="dow", drop_first=True)
    cal = pd.DataFrame({"sin_poy": np.sin(2 * np.pi * poy), "cos_poy": np.cos(2 * np.pi * poy)},
                       index=df.index)
    for c in dow.columns:
        cal[c] = dow[c].to_numpy(float)
    return cal


def build_pointwise(df, kind, n_train, extra=()):
    """Pointwise (non-signature) features of SigAdapt.

    kind  : "rte" or "generic"
    extra : covariates entering directly, standardised on the first n_train rows
            (e.g. solar radiation for ENTSO-E)

    Returns (DataFrame of features, boolean array of special days or None).
    """
    if kind == "rte":
        cal, special = rte_calendar(df)
    elif kind == "generic":
        cal, special = generic_calendar(df), None
    else:
        raise ValueError(f"unknown calendar '{kind}'")
    if extra:
        x = df[list(extra)].to_numpy(float)
        x = (x - x[:n_train].mean(0)) / x[:n_train].std(0)
        for j, c in enumerate(extra):
            cal[c] = x[:, j]
    return cal, special
