"""Reading the YAML configuration files."""

from pathlib import Path

import numpy as np
import yaml


def load_config(path):
    """Read an experiment config and attach its dataset config under the key "data".
    """
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    data_path = path.parent.parent / "data" / f"{cfg['dataset']}.yaml"
    cfg["data"] = yaml.safe_load(data_path.read_text())
    cfg["config_file"] = str(path)
    return cfg


def grid(spec):
    """A list of values, or {logspace: [start, stop, num]} for numpy.logspace;
    to read the ridge penalties and Kalman variance grid. """
    if isinstance(spec, dict) and "logspace" in spec:
        start, stop, num = spec["logspace"]
        return np.logspace(start, stop, int(num))
    return np.asarray(spec, dtype=float)
