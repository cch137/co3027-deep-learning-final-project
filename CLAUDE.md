# CLAUDE.md

## Project

Self-supervised lead-lag relationship estimation between time series. Given two
aligned windows, a small neural model predicts: who leads whom and by how many
days (lag), correlation strength and sign (rho), and optionally the scale
factor (beta). Trained entirely on synthetic data (labels are free from the
generator); real market data is used only for validation/test case studies.
See `proposal.md` for the original (broader) proposal and README.md for the
current, narrowed scope.

## Environment

- Windows 11, CPU-only laptop. Training budget: 1-2 hours max.
- Virtual env managed with uv: `uv venv` then `uv pip install -r requirements.txt`.
- Run scripts with `.venv\Scripts\python <script>` (do not assume an activated shell).

## Key decisions (do not silently change)

- All constants live in `config.py`: 14-ticker universe, WINDOW_LEN=128,
  MAX_LAG=10, split dates.
- Splits are by window END date: train <= 2025-12-31, val = 2026 Q1,
  test = 2026 Q2 onward. Never shuffle across time.
- Features: daily log returns (`^TNX` uses first difference), rolling z-score
  normalization using past data only (no lookahead).
- Lag head is a 21-class classifier (tau in [-10, +10]), not regression.
- Classical cross-correlation scan is the baseline AND the reference-answer
  generator for real data; it must exist before any model work.
- Models stay small (~0.5M params); 1D-CNN encoder first, Transformer only as
  an ablation.

## Layout

- `data/download.py` — one-shot yfinance download, writes parquet (gitignored)
- `synth/` — synthetic pair generator (the actual training data)
- `models/` — baselines + neural model
- `train.py`, `eval.py` — config-driven entry points
