"""Evaluate the trained model.

Part 1 — synthetic test set (seed 999, never used for training/selection):
         model vs classical cross-correlation baseline.
Part 2 — real market data: agreement with the baseline on all asset pairs over
         the validation (2026 Q1) and test (2026 Q2) periods, plus named
         case studies with known ground truth.

Run:  python eval.py [--ckpt outputs/model.pt]
"""

import argparse
import itertools
import sys

import numpy as np
import pandas as pd
import torch

import config
from models.baselines import xcorr_scan_batch
from models.net import LeadLagNet
from synth.generator import make_fixed_set

TEST_SEED = 999
UNCORR_THRESHOLD = 0.2  # |rho| below this => "uncorrelated" in 3-way direction

CASE_STUDIES = [
    ("^GSPC", "^TWII", "S&P500 -> TWII (expect tau=+1, rho>0)"),
    ("^GSPC", "^N225", "S&P500 -> N225 (expect tau=+1, rho>0)"),
    ("^VIX", "^GSPC", "VIX <-> S&P500 (expect tau=0, rho<0)"),
    ("BTC-USD", "ETH-USD", "BTC <-> ETH (expect tau=0, rho>>0)"),
]


def direction3(rho: np.ndarray) -> np.ndarray:
    """-1 negative, 0 uncorrelated, +1 positive."""
    d = np.sign(rho)
    d[np.abs(rho) < UNCORR_THRESHOLD] = 0
    return d


@torch.no_grad()
def model_predict(model: LeadLagNet, a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: (N, L) already z-scored. Returns tau (N,), rho (N,)."""
    x = torch.from_numpy(np.stack([a, b], axis=1).astype(np.float32))
    out_tau, out_rho = [], []
    for i in range(0, len(x), 1024):
        logits, rho = model(x[i : i + 1024])
        out_tau.append(logits.argmax(dim=1).numpy() - config.MAX_LAG)
        out_rho.append(rho.numpy())
    return {"tau": np.concatenate(out_tau), "rho": np.concatenate(out_rho)}


def eval_synthetic(model: LeadLagNet) -> None:
    print("=" * 72)
    print(f"PART 1 — synthetic test set (seed {TEST_SEED}, n=20000)")
    print("=" * 72)
    d = make_fixed_set(20000, seed=TEST_SEED)
    tau_true = d["lag_class"] - config.MAX_LAG
    m = d["corr_mask"] == 1

    preds = {
        "model": model_predict(model, d["x"][:, 0], d["x"][:, 1]),
        "xcorr baseline": xcorr_scan_batch(d["x"][:, 0], d["x"][:, 1], config.MAX_LAG),
    }
    print(f"{'method':<16} {'lag acc':>8} {'lag ±1':>8} {'rho MAE':>8} {'dir3 acc':>9}")
    for name, p in preds.items():
        exact = (p["tau"][m] == tau_true[m]).mean()
        within1 = (np.abs(p["tau"][m] - tau_true[m]) <= 1).mean()
        rho_mae = np.abs(p["rho"] - d["rho"]).mean()
        dir_acc = (direction3(p["rho"].copy()) == direction3(d["rho"].copy())).mean()
        print(f"{name:<16} {exact:>8.3f} {within1:>8.3f} {rho_mae:>8.4f} {dir_acc:>9.3f}")

    # difficulty breakdown for the model: strong vs weak correlation
    p = preds["model"]
    for lo, hi, label in [(0.5, 1.01, "|rho|>=0.5"), (0.2, 0.5, "0.2<=|rho|<0.5"),
                          (0.0, 0.2, "|rho|<0.2")]:
        sel = m & (np.abs(d["rho"]) >= lo) & (np.abs(d["rho"]) < hi)
        if sel.sum():
            w1 = (np.abs(p["tau"][sel] - tau_true[sel]) <= 1).mean()
            print(f"  model lag ±1 acc on {label:<16}: {w1:.3f}  (n={sel.sum()})")


def real_windows(ret: pd.DataFrame, a: str, b: str, period: tuple[str, str]):
    """All length-L windows of pair (a,b) whose END date falls in period.

    Returns (A, B) z-scored arrays of shape (n_windows, L). Windows with a
    small share of NaN (long market holidays) are kept with NaN treated as a
    zero return; windows with >10% NaN are skipped.
    """
    L = config.WINDOW_LEN
    sub = ret[[a, b]]
    end_dates = ret.index[(ret.index >= period[0]) & (ret.index <= period[1])]
    wa, wb = [], []
    for end in end_dates:
        pos = ret.index.get_loc(end)
        if pos + 1 < L:
            continue
        w = sub.iloc[pos + 1 - L : pos + 1].to_numpy()
        if np.isnan(w).mean(axis=0).max() > 0.10:
            continue
        w = np.nan_to_num(w, nan=0.0)
        wa.append(w[:, 0])
        wb.append(w[:, 1])
    if not wa:
        return None, None
    A, B = np.array(wa), np.array(wb)
    z = lambda x: (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-12)
    return z(A), z(B)


def eval_real(model: LeadLagNet) -> None:
    ret = pd.read_parquet(config.RETURNS_PARQUET)
    periods = {
        "val 2026Q1": ("2026-01-01", config.VAL_END),
        "test 2026Q2": (str((pd.Timestamp(config.VAL_END) + pd.Timedelta(days=1)).date()),
                        "2026-12-31"),
    }

    print()
    print("=" * 72)
    print("PART 2 — real market data: model vs xcorr baseline")
    print("=" * 72)

    for period_name, period in periods.items():
        # aggregate agreement over all pairs
        agree, rho_diff, n_win, n_sig = 0, [], 0, 0
        for a, b in itertools.combinations(config.TICKERS, 2):
            A, B = real_windows(ret, a, b, period)
            if A is None:
                continue
            mp = model_predict(model, A, B)
            bp = xcorr_scan_batch(A, B, config.MAX_LAG)
            n_win += len(A)
            rho_diff.append(np.abs(mp["rho"] - bp["rho"]))
            sig = np.abs(bp["rho"]) >= UNCORR_THRESHOLD  # only count lag where it's meaningful
            n_sig += sig.sum()
            agree += (np.abs(mp["tau"][sig] - bp["tau"][sig]) <= 1).sum()
        rho_mae = np.concatenate(rho_diff).mean()
        print(f"\n[{period_name}] {n_win} windows over {len(list(itertools.combinations(config.TICKERS, 2)))} pairs")
        print(f"  lag agreement (±1, where baseline |rho|>={UNCORR_THRESHOLD}): "
              f"{agree / max(n_sig, 1):.3f}  (n={n_sig})")
        print(f"  rho MAE vs baseline: {rho_mae:.4f}")

        print(f"  {'case study':<42} {'base tau':>8} {'mdl tau':>8} {'base rho':>9} {'mdl rho':>8}")
        for a, b, desc in CASE_STUDIES:
            A, B = real_windows(ret, a, b, period)
            if A is None:
                print(f"  {desc:<42} (no valid windows)")
                continue
            mp = model_predict(model, A, B)
            bp = xcorr_scan_batch(A, B, config.MAX_LAG)
            mode = lambda v: int(pd.Series(v).mode().iloc[0])
            print(f"  {desc:<42} {mode(bp['tau']):>8d} {mode(mp['tau']):>8d} "
                  f"{bp['rho'].mean():>9.3f} {mp['rho'].mean():>8.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/model.pt")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, weights_only=True)
    model = LeadLagNet(ckpt["max_lag"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    eval_synthetic(model)
    eval_real(model)


if __name__ == "__main__":
    main()
