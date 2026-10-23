# Data


| Dataset | Files | Columns |
|---|---|---|
| RTE (French national load, half-hourly) | `rte_2018_2026.csv` | `Consumption`, `Temperature`, `halfhour` (0-47), `year_position` (in [0, 1)), `DayOff` (public holiday), `vacancesZoneA`, `vacancesZoneB`, `vacancesZoneC` (school holidays) |
| GEFCom2012 (11 zones, hourly) | `gfc12_<zone>.csv`, zones 1-11 | `load`, `temperature` |
| GEFCom2014 (hourly) | `gfc14_load.csv` | `load`, `temperature` |
| GEFCom2017 (8 zones, hourly) | `gfc17_<zone>.csv`, zones CT, ME, NH, RI, VT, SEMASS, WCMASS, NEMASSBOST | `load`, `temperature` |
| ENTSO-E (6 countries, hourly) | `entsoe_1H_<country>.csv`, countries AT, BE, DE, HU, LU, NL | `load`, `temperature`, `rad_direct`, `rad_diffuse` |
| EPF (5 markets, hourly) | `epf_<market>.csv`, markets fr, de, be, np, pjm | `target` (price) and two exogenous columns |

The GEFCom, ENTSO-E and EPF series are taken from fev-bench
(<https://huggingface.co/datasets/autogluon/fev_datasets>).

The RTE dataset was obtained by combining half-hourly French national electricity load from Eco2mix(<https://www.rte-france.com/eco2mix>), the open data platform of the French transmission system operator RTE with with temperature recorded by Météo-France (<https://donneespubliques.meteofrance.fr>).
