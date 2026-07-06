# gmdh-energy-pricing

Reproducibility package for **"Hybrid GMDH-Based Energy Pricing for
Territorial Communities with Renewable and Biofuel Sources: A
Multi-Objective Optimization Framework"** (Osypenko, Smolarz, Kaplun,
Lytvynenko), Section IX.

## Repository layout

```
.
├── R/
│   ├── gmdh_combinatorial.R   # GMDH algorithm implementation (Section IV.A)
│   ├── cv_pipeline.R          # Stratified 5-fold cross-validation (Section VI)
│   └── mc_uncertainty.R       # Monte Carlo uncertainty propagation, K=10,000 (Section IV.C)
├── Python/
│   ├── baselines.py           # RF, XGBoost, NN baselines
│   └── optimization_models.py # MILP / two-stage stochastic programming (eq. 7-11)
├── data/
│   └── energy_balance_2019_2023.csv   # monthly, 60 observations (2019-01 .. 2023-12)
└── results/                   # metrics/figures written by the scripts above
```

## Data

`data/energy_balance_2019_2023.csv` — monthly energy balance for the
territorial community (TC) microgrid:

| column | description |
|---|---|
| `date` | month, `YYYY-MM` |
| `E_SPP`, `E_WPP` | solar / wind generation (MWh), eq. (3)-(4) |
| `E_HPP`, `E_BCP` | biofuel heat plant / cogeneration generation (MWh) |
| `E_CPSS` | centralized grid draw (MWh) |
| `B` | available biofuel resource (t or MWh-eq.), eq. (5) |
| `D` | community electricity demand (MWh) |
| `C` | realised specific electricity cost (EUR/kWh), GMDH cost-model target |

> Note: this CSV is a synthetic, seeded (seed = 42) reconstruction that
> reproduces the variable definitions, seasonality, and scale described
> in the manuscript, generated with `gen_data.py`. Replace it with the
> original metered dataset for exact result reproduction.

## Environment

- **R 4.3.2** — packages: `Metrics`, `caret`, `ggplot2`
- **Python 3.11** — `scikit-learn` 1.4, `xgboost` 2.0, `torch` 2.2, `pulp`

```bash
# R
install.packages(c("Metrics", "caret", "ggplot2"))

# Python
pip install scikit-learn==1.4.* xgboost==2.0.* torch==2.2.* pulp pandas numpy
```

`Seed = 42` everywhere for reproducibility.

## Running

From the repository root:

```bash
# GMDH demand/cost models
Rscript R/gmdh_combinatorial.R

# 5-fold CV comparison
Rscript R/cv_pipeline.R

# Monte Carlo uncertainty (K = 10,000)
Rscript R/mc_uncertainty.R

# Baselines: RF / XGBoost / NN
python Python/baselines.py

# MILP / stochastic-programming dispatch
python Python/optimization_models.py
```

Outputs (metrics CSVs, figures) are written to `results/`.
