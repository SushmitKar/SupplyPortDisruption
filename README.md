# Supply Chain Contagion: Does Port Disruption Spread?

Most port-monitoring dashboards track one port at a time. This project asks a
different question: **when one port is disrupted, does the disruption spread
to other ports — and if so, how fast and how far?**

## Data

- **Port activity**: daily vessel call counts (`portcalls`) for global ports,
  2019–2026.
- **Disruption events**: labeled real-world events (COVID-19 lockdowns, Suez
  Canal blockage, Red Sea crisis) with severity levels, used to cross-reference
  the disruption signal against known ground truth.
- Analysis is restricted to the top 20 ports by total vessel-call volume.

![Daily vessel calls, top 5 ports](outputs/figures/01_port_trends.png)

## Method

**1. Defining disruption.**
A port is flagged as having a *Port Disruption Event (PDE)* on any day its
vessel calls fall more than 1.5 standard deviations below its own trailing
30-day rolling mean. This turns a noisy count series into a binary signal
that can be compared across ports.

**2. Classical forecasting baseline.**
ADF stationarity testing, ACF/PACF-guided and auto-selected ARIMA, and
Holt-Winters exponential smoothing, benchmarked per port. Classical models
forecast stable hubs (Singapore, Rotterdam) well but produce flat, uninformative
forecasts for volatile ports (Nagoya, Shanghai) — they can't anticipate
external shocks.

**3. Cross-port correlation, with multiple-comparisons correction.**
Pairwise, lagged correlation between every port's PDE series (20 ports ×
19 possible leaders × 14 lags = ~5,300 tests), to test whether disruption at
one port predicts disruption at another shortly after. Run at that scale,
an uncorrected significance threshold would flag roughly 250–270 pairs from
noise alone. A Benjamini-Hochberg FDR correction was applied across all tests,
which reduced the "significant" pair count from 262 to 172. Each port's
single strongest correlated "leader" — the one used downstream as an XGBoost
feature — held stable before and after correction, meaning the core leader
relationships were not among the noisy ones filtered out.

**4. Advanced models.**
- **VAR** on two port clusters (Japanese ports; Ningbo–Shanghai) to jointly
  model co-movement.
- **Granger causality** to test directional predictive relationships between
  specific port pairs.
- **Prophet** with known disruption events (COVID, Suez, Red Sea) added as
  explicit regressors.
- **XGBoost** using each port's own lag features plus its statistically
  strongest "leading" port's lags (post-correction), with feature importance
  to interpret which lags actually drive predictions.

## Results

XGBoost outperforms classical models for most of the Japanese port cluster
and Shanghai (5 of 7 ports with leader-lag features), though not uniformly —
Chiba and Sakai-Semboku still favor Prophet despite the same feature set.

![Model comparison — MAPE by port](outputs/figures/13_final_comparison.png)

![XGBoost feature importance — Japanese cluster](outputs/figures/12_xgb_importance.png)

The importance plots show why: `own_lag1` and `leader_lag1` dominate for
every port in this cluster, meaning the model is genuinely leaning on the
lag structure, not treating features arbitrarily.

Full per-port comparison: `outputs/processed/final_comparison.csv`.

## Known limitations

Being upfront about the current state, not just the results:

- **Zero-value edge case in MAPE.** A small number of ports have occasional
  zero-vessel-call days in their test window, which makes standard MAPE
  undefined at that point. Metric calculations mask these points explicitly
  rather than silently propagating NaN, so per-port scores are always
  computed from the well-defined subset of days.
- **Weather/exogenous data** aside from the three labeled disruption events
  isn't used; the model relies on vessel-call history and event flags only.
- **Regional contagion is inferred from correlation and Granger causality,
  not a causal experiment.** Even after correction, this identifies
  statistically robust lead-lag relationships — it doesn't rule out shared
  external causes (e.g. a common trade-lane shock) producing correlated
  timing without one port literally causing the other's disruption.

## Tech stack

Python · pandas · NumPy · statsmodels (ADF, ARIMA, VAR, Granger causality,
seasonal decomposition) · pmdarima (auto-ARIMA) · Prophet · XGBoost ·
scikit-learn (metrics) · statsmodels (multiple-testing correction) ·
matplotlib · seaborn

## Repo structure

```
notebooks/
├── 01_eda.ipynb                # PDE definition, event validation, correlation
├── 02_classical_models.ipynb   # ADF, ARIMA, Holt-Winters, cross-correlation
└── 03_advanced_models.ipynb    # VAR, Granger causality, Prophet, XGBoost

outputs/
├── figures/                    # all saved plots
└── processed/                  # intermediate CSVs (PDE flags, model results)
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_eda.ipynb   # run in order: 01 → 02 → 03
```