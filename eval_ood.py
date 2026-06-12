"""Out-of-distribution generalization tests — everything here is unseen in training.

Test A  semi-real pairs: real market returns as the base signal, lag/scale/noise
        applied by us, so labels are exact while the signal statistics (fat
        tails, volatility clustering) were never in the training generator.
Test B  OOD synthetic: GARCH(1,1) base signals + Student-t(3) noise innovations.
Test C  unseen markets: tickers never used anywhere in the pipeline (KOSPI,
        ASX 200, SOL, EURUSD), scored against known market relationships.

Run:  python eval_ood.py [--ckpt outputs/model.pt]
Writes outputs/figures/ood_summary.md
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yfinance as yf

import config
from eval import model_predict, real_windows
from models.baselines import xcorr_scan_batch
from synth.generator import BETA_RANGE, SIGMA_RANGE, _rowwise_corr, _zscore

OOD_TICKERS = {
    "^KS11": "KOSPI (Korea)",
    "^AXJO": "ASX 200 (Australia)",
    "SOL-USD": "Solana",
    "EURUSD=X": "EUR/USD",
}
OOD_CLOSE_PARQUET = "data/ood_close.parquet"
OOD_PERIOD = ("2026-01-01", "2026-12-31")  # both 2026 quarters; tickers are fully unseen

OOD_CASES = [
    ("^GSPC", "^KS11", "S&P500 -> KOSPI (expect tau=+1, rho>0)"),
    ("^GSPC", "^AXJO", "S&P500 -> ASX200 (expect tau=+1, rho>0)"),
    ("BTC-USD", "SOL-USD", "BTC <-> SOL (expect tau=0, rho>>0)"),
    ("DX-Y.NYB", "EURUSD=X", "DXY <-> EURUSD (expect tau=0, rho<<0)"),
]


def pair_metrics(model, a: np.ndarray, b: np.ndarray, tau_true, rho_true) -> dict:
    out = {}
    preds = {
        "model": model_predict(model, a, b),
        "baseline": xcorr_scan_batch(a, b, config.MAX_LAG),
    }
    for name, p in preds.items():
        out[name] = {
            "lag_acc": float((p["tau"] == tau_true).mean()),
            "lag_acc1": float((np.abs(p["tau"] - tau_true) <= 1).mean()),
            "rho_mae": float(np.abs(p["rho"] - rho_true).mean()),
        }
    return out


def build_pairs_from_base(base: np.ndarray, rng: np.random.Generator):
    """Apply the synthetic construction B = beta*shift(A,tau) + sigma*eps to
    given base rows of length WINDOW_LEN + 2*MAX_LAG (same protocol as training)."""
    L, K = config.WINDOW_LEN, config.MAX_LAG
    n = len(base)
    base = (base - base.mean(axis=1, keepdims=True)) / (base.std(axis=1, keepdims=True) + 1e-12)
    a = base[:, K : K + L]
    tau = rng.integers(-K, K + 1, size=n)
    sign = rng.choice([-1.0, 1.0], size=n)
    beta = sign * np.exp(rng.uniform(np.log(BETA_RANGE[0]), np.log(BETA_RANGE[1]), size=n))
    sigma = np.exp(rng.uniform(np.log(SIGMA_RANGE[0]), np.log(SIGMA_RANGE[1]), size=n))
    rows = np.arange(n)[:, None]
    cols = (K - tau)[:, None] + np.arange(L)[None, :]
    a_shift = base[rows, cols]
    b = beta[:, None] * a_shift + sigma[:, None] * rng.standard_normal((n, L))
    rho = _rowwise_corr(a_shift, b)
    return _zscore(a), _zscore(b), tau, rho.astype(np.float32)


def test_a_semireal(model, ret: pd.DataFrame, rng) -> dict:
    """Real return series as base, known labels (n ~ 5000)."""
    L, K = config.WINDOW_LEN, config.MAX_LAG
    seg = L + 2 * K
    train = ret.loc[: config.TRAIN_END]
    bases = []
    per_ticker = 5000 // len(config.TICKERS) + 1
    for t in config.TICKERS:
        s = train[t].dropna().to_numpy()
        starts = rng.integers(0, len(s) - seg, size=per_ticker)
        bases.extend(s[i : i + seg] for i in starts)
    base = np.array(bases)
    a, b, tau, rho = build_pairs_from_base(base, rng)
    return pair_metrics(model, a, b, tau, rho)


def _garch_base(n: int, length: int, rng) -> np.ndarray:
    alpha = rng.uniform(0.02, 0.15, size=(n, 1))
    beta_g = rng.uniform(0.80, 0.98 - alpha)  # keep alpha + beta < 1 (stationary)
    omega = 1.0 - alpha - beta_g  # unconditional variance 1
    z = rng.standard_normal((n, length))
    x = np.empty((n, length))
    var = np.ones((n, 1))
    prev = np.zeros((n, 1))
    for t in range(length):
        var = omega + alpha * prev**2 + beta_g * var
        prev = np.sqrt(var) * z[:, t : t + 1]
        x[:, t] = prev[:, 0]
    return x


def test_b_ood_synthetic(model, rng) -> dict:
    """GARCH(1,1) bases + Student-t(3) noise — neither exists in training."""
    L, K = config.WINDOW_LEN, config.MAX_LAG
    n, seg = 5000, L + 2 * K
    base = _garch_base(n, seg, rng)
    base = (base - base.mean(axis=1, keepdims=True)) / (base.std(axis=1, keepdims=True) + 1e-12)
    a = base[:, K : K + L]
    tau = rng.integers(-K, K + 1, size=n)
    sign = rng.choice([-1.0, 1.0], size=n)
    beta = sign * np.exp(rng.uniform(np.log(BETA_RANGE[0]), np.log(BETA_RANGE[1]), size=n))
    sigma = np.exp(rng.uniform(np.log(SIGMA_RANGE[0]), np.log(SIGMA_RANGE[1]), size=n))
    rows = np.arange(n)[:, None]
    cols = (K - tau)[:, None] + np.arange(L)[None, :]
    a_shift = base[rows, cols]
    t_noise = rng.standard_t(df=3, size=(n, L)) / np.sqrt(3.0)  # unit variance, fat tails
    b = beta[:, None] * a_shift + sigma[:, None] * t_noise
    rho = _rowwise_corr(a_shift, b).astype(np.float32)
    return pair_metrics(model, _zscore(a), _zscore(b), tau, rho)


def load_ood_returns(ret: pd.DataFrame) -> pd.DataFrame:
    """Combined returns: existing universe + never-before-seen tickers."""
    path = Path(OOD_CLOSE_PARQUET)
    if path.exists():
        close = pd.read_parquet(path)
    else:
        raw = yf.download(list(OOD_TICKERS), start="2025-01-01",
                          auto_adjust=True, progress=False)
        close = raw["Close"].sort_index()
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close = close[close.index.dayofweek < 5]
        close.to_parquet(path)
    new_ret = np.log(close.ffill(limit=config.FFILL_LIMIT)).diff()
    return pd.concat([ret, new_ret], axis=1)


def test_c_unseen_markets(model, ret: pd.DataFrame) -> list[dict]:
    combined = load_ood_returns(ret)
    rows = []
    for a, b, desc in OOD_CASES:
        A, B = real_windows(combined, a, b, OOD_PERIOD)
        if A is None:
            rows.append({"pair": desc, "error": "no valid windows"})
            continue
        mp = model_predict(model, A, B)
        bp = xcorr_scan_batch(A, B, config.MAX_LAG)
        mode = lambda v: int(pd.Series(v).mode().iloc[0])
        rows.append({
            "pair": desc, "n": len(A),
            "model_tau": mode(mp["tau"]), "base_tau": mode(bp["tau"]),
            "model_rho": float(mp["rho"].mean()), "base_rho": float(bp["rho"].mean()),
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/model.pt")
    args = p.parse_args()

    from models.net import LeadLagNet
    ckpt = torch.load(args.ckpt, weights_only=True)
    model = LeadLagNet(ckpt["max_lag"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ret = pd.read_parquet(config.RETURNS_PARQUET)
    rng = np.random.default_rng(2026)

    md = ["# Out-of-distribution generalization tests\n",
          "Nothing below appears in the training distribution.\n"]

    print("Test A — semi-real pairs (real returns as base, exact labels)...")
    ra = test_a_semireal(model, ret, rng)
    print("Test B — OOD synthetic (GARCH base, Student-t noise)...")
    rb = test_b_ood_synthetic(model, rng)
    for title, r in [("Test A — semi-real pairs (real market returns as base, n≈5000)", ra),
                     ("Test B — OOD synthetic (GARCH(1,1) base + Student-t(3) noise, n=5000)", rb)]:
        md.append(f"## {title}\n")
        md.append("| method | lag exact | lag ±1 | ρ MAE |")
        md.append("|---|---|---|---|")
        for name in ["model", "baseline"]:
            m = r[name]
            md.append(f"| {name} | {m['lag_acc']:.3f} | {m['lag_acc1']:.3f} | {m['rho_mae']:.3f} |")
        md.append("")
        print(f"  model: lag {r['model']['lag_acc']:.3f} / ±1 {r['model']['lag_acc1']:.3f} "
              f"/ rho MAE {r['model']['rho_mae']:.3f}   "
              f"baseline: {r['baseline']['lag_acc']:.3f} / {r['baseline']['lag_acc1']:.3f} "
              f"/ {r['baseline']['rho_mae']:.3f}")

    print("Test C — unseen markets (KOSPI, ASX, SOL, EURUSD)...")
    rc = test_c_unseen_markets(model, ret)
    md.append("## Test C — markets never seen anywhere in the pipeline (2026 windows)\n")
    md.append("| pair | n windows | baseline τ | model τ | baseline ρ | model ρ |")
    md.append("|---|---|---|---|---|---|")
    for c in rc:
        if "error" in c:
            md.append(f"| {c['pair']} | — | — | — | — | {c['error']} |")
            print(f"  {c['pair']}: {c['error']}")
            continue
        md.append(f"| {c['pair']} | {c['n']} | {c['base_tau']:+d} | {c['model_tau']:+d} | "
                  f"{c['base_rho']:+.3f} | {c['model_rho']:+.3f} |")
        print(f"  {c['pair']:<48} base tau={c['base_tau']:+d} rho={c['base_rho']:+.3f}   "
              f"model tau={c['model_tau']:+d} rho={c['model_rho']:+.3f}")

    out = Path("outputs/figures/ood_summary.md")
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
