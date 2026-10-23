# SigAdapt ✍️

Code for the paper *Encoding Time Series Dynamics with Sliding-Window Signatures for Online Forecasting*.

SigAdapt forecasts a time series with exogenous covariates using an adaptive linear model on
sliding-window path signatures:

1. **Signature features.** For each window length *w*, the *outlook signature* is the
   signature of (time, covariates) over the window ending at the forecast time, and the
   *history signature* is the signature of (time, covariates, target) over the window ending
   at the forecast origin. Paths are time- and basepoint-augmented, and signatures are
   updated sequentially with Chen's identity.
2. **Linear model.** One ridge regression per slot of the day, on the signature features
   and calendar variables.
3. **Online adaptation.** A Kalman filter tracks the drift of the contribution of each
   signature level; i.e. the signature dynamic effects.
4. **Aggregation.** The experts obtained for the different window lengths are combined with
   Bernstein Online Aggregation.


## Repository structure 🗂️

```
configs/          dataset and experiment configurations
src/sigadapt/     the package
  signatures.py   sliding-window signatures (Chen's identity, basepoint augmentation)
  sigadapt.py     the SigAdapt model (ridge per slot, Kalman filter, aggregation)
  kalman.py       Kalman filter for regression with drifting coefficients
  calendar.py     calendar variables
  data.py         datasets and splits
  baselines/      TiDE, Chronos-2, GAM / GAMAdapt, linear baseline without signatures
scripts/          run.py (one experiment), run_all.sh, make_tables.py
notebooks/        analysis of the results (tests, figures)
```


## Configuration files

- `configs/data/<dataset>.yaml` describes a dataset: file(s), target, covariates,
  frequency, lead time, and the train / validation / test split. 
- `configs/<dataset>/<model>.yaml` contains the hyperparameters of one model on one
  dataset: e.g. the window lengths, signature depth, ridge penalty grid, and Kalman variance grid
  for SigAdapt, and the look-backs, seeds, and network size for TiDE.



## Reproducing the results 🚀

Each experiment is defined by one configuration file and run with one command:

```bash
python scripts/run.py configs/rte/sigadapt.yaml
```

| Result | Command |
|---|---|
| RTE: SigAdapt | `python scripts/run.py configs/rte/sigadapt.yaml` |
| RTE: baselines | `python scripts/run.py configs/rte/{tide,chronos2,gam}.yaml` |
| RTE: no-signature linear baseline (ablation) | `python scripts/run.py configs/rte/linear.yaml` |
| RTE: tests and figures | `python scripts/collect_rte.py`, then `notebooks/rte_results_analysis.ipynb` |
| fev-bench (GFC12, GFC14, GFC17, ENTSO-E, EPF) | `bash scripts/run_all.sh gfc12 gfc14 gfc17 entsoe epf` |
| Summary tables | `python scripts/make_tables.py` |
| fev-bench figure | `notebooks/fevbench_figure.ipynb` |
| Everything | `bash scripts/run_all.sh` |

Results are written to `results/<dataset>/<model>/[<series>/]`:

- `predictions_test.csv`: test forecasts (`DateTime, y_true, y_pred`);
- `metrics.json`: RMSE, MAE, MAPE, and the errors standardised by the standard deviation of
  the target on the training period.

TiDE and Chronos-2 are run on a GPU. The other models run on a CPU.


## Acknowledgements 🧩

This code builds on the following:

- [iisignature](https://github.com/bottler/iisignature/pull/25): signature computations.
- [viking_kalman](https://github.com/EDF-Lab/viking_kalman): estimation of the
  Kalman filter variances (iterative grid search). It is a Python version of the R package
  viking by J. de Vilmarest.
- [opera-python](https://github.com/Dralliag/opera-python): online aggregation of experts.
- [Darts](https://github.com/unit8co/darts) (TiDE) and
  [Chronos-2](https://github.com/amazon-science/chronos-forecasting).
- The *invariant* (asinh) scaler follows [epftoolbox](https://github.com/jeslago/epftoolbox)
  (Uniejewski, Weron and Ziel, 2018).
- The benchmark datasets come from [fev-bench](https://github.com/autogluon/fev).


