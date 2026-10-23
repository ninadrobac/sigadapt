"""Download the fev-bench datasets used in the paper and save them as CSV files in this folder.

    pip install datasets
    python data/download_fevbench.py

The datasets come from the Hugging Face repository autogluon/fev_datasets (Shchur et al., fev-bench).
Each configuration contains one or several series (zones, countries, markets). Every series is saved
in its own file, with the columns DateTime, the target, and the covariates:

    GEFCom2012  proenfo_gfc12  as  gfc12_<zone>.csv        (load, temperature)
    GEFCom2014  proenfo_gfc14  as  gfc14_load.csv          (load, temperature)
    GEFCom2017  proenfo_gfc17  as  gfc17_<zone>.csv        (load, temperature)
    ENTSO-E     entsoe_1H      as  entsoe_1H_<country>.csv (load, temperature, rad_direct, rad_diffuse)
    EPF         epf_<market>   as  epf_<market>.csv        (target = price, and the two exogenous columns)
"""

from pathlib import Path

import datasets
import pandas as pd


data_dir = Path(__file__).resolve().parent              
configs_dir = data_dir.parent / "configs" / "data"


ENTSOE = {"target": "load", "radiation_direct_horizontal": "rad_direct",
          "radiation_diffuse_horizontal": "rad_diffuse"}

DATASETS = {
    "gfc12": (["proenfo_gfc12"], None, "gfc12_{id}.csv"),
    "gfc14": (["proenfo_gfc14"], None, "gfc14_load.csv"),
    "gfc17": (["proenfo_gfc17"], None, "gfc17_{id}.csv"),
    "entsoe": (["entsoe_1H"], ENTSOE, "entsoe_1H_{id}.csv"),
    "epf": ([f"epf_{m}" for m in ["fr", "de", "be", "np", "pjm"]], {}, "{config}.csv"),
}

def gfc_rename(columns):
    """GFC: the target is the load, and the only other column is the temperature."""
    covariates = [c for c in columns if c not in ("id", "timestamp", "target")]
    assert len(covariates) == 1, f"expected 1 covariate, got {covariates}"
    return {"target": "load", covariates[0]: "temperature"}


def to_frame(row, columns, rename):
    """One series (one row of the Hugging Face dataset) to DataFrame(DateTime, target, covariates)."""
    df = pd.DataFrame({"DateTime": row["timestamp"],
                       **{c: row[c] for c in columns if c not in ("id", "timestamp")}})
    return df.rename(columns=rename)


for name, (hf_configs, rename, pattern) in DATASETS.items():
    written = []
    for config in hf_configs:
        ds = datasets.load_dataset("autogluon/fev_datasets", config, split="train")
        ds.set_format("numpy")
        cols = gfc_rename(ds.column_names) if rename is None else rename
        for row in ds:
            series_id = str(row["id"]).strip()
            path = data_dir / pattern.format(id=series_id, config=config)
            df = to_frame(row, ds.column_names, cols)
            df.to_csv(path, index=False)
            written.append(series_id)
            print(f"{config:15s} {series_id:12s} -> {path.name}  "
                  f"{len(df)} rows, {df.DateTime.iloc[0]} -> {df.DateTime.iloc[-1]}")