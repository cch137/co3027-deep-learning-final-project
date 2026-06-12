# Lead-Lag Relationship Estimation Without Labels

**Problem in one sentence:** given two time series and no human annotations,
train a model that tells you who leads whom, by how many days, how strongly
they are correlated, and whether the relationship is positive, negative, or
absent.

The model is trained purely on **synthetic data** — pairs generated as
`B = beta * shift(A, tau) + noise`, so the labels (lag `tau`, correlation
`rho`, scale `beta`) come for free from the generator. Real market data is
never trained on; it serves as out-of-sample validation against classical
cross-correlation and known market facts (e.g., the S&P 500 close leads Asian
indices by one day).

## Setup

```powershell
uv venv
uv pip install -r requirements.txt
.venv\Scripts\python data\download.py   # one-shot, ~5 MB of parquet
```

## Data

- **Training:** synthetic pairs, generated on the fly (`synth/`).
  Base signals: AR(1), sine mixtures, random-walk increments.
  `tau` in [-10, +10], `|beta|` in [0.3, 3] (both signs), ~15% uncorrelated pairs.
- **Validation/test (real, daily, 2016-01-01 to date, via yfinance):**
  14 assets across markets — US equities (^GSPC, ^IXIC), volatility (^VIX),
  Asian equities (^N225, ^HSI, ^TWII), European equities (^FTSE, ^GDAXI),
  crypto (BTC-USD, ETH-USD), commodities (GC=F, CL=F), rates (^TNX),
  US dollar (DX-Y.NYB).
- **Splits (by window end date):** train/reference <= 2025-12-31,
  validation = 2026 Q1, test = 2026 Q2 to date.

## Model

Input: two aligned, normalized windows of length 128, shape `(2, 128)`.
Siamese 1D-CNN encoder -> fusion -> three heads:

| Head | Output | Question answered |
|---|---|---|
| lag  | 21-class softmax, tau in [-10, +10] | who leads, by how many days |
| rho  | tanh in [-1, 1] | correlation strength and sign (near 0 = unrelated) |
| beta | derived analytically at readout: `beta = rho * std(B) / std(A)` | scale factor of the relationship (no learned head needed — with z-scored inputs the OLS slope equals rho) |

## Evaluation

- **Synthetic test set** (fixed seed): lag accuracy (exact and within ±1),
  rho MAE, 3-way direction accuracy (positive / negative / uncorrelated at
  |rho| < 0.2), beta MAE.
- **Real data:** agreement with the cross-correlation scan baseline (lag within
  ±1, rho MAE), plus case studies — S&P 500 -> TWII/N225 (expect lag +1,
  rho > 0), VIX <-> S&P 500 (lag 0, rho < 0), BTC <-> ETH (lag 0, high rho).

## Future work (deliberately out of scope)

Multi-asset cross-market attention (full CMAN), return forecasting and trading
backtests, time-varying online lag tracking, nonlinear relationship types,
applying the beta head to real market data.
