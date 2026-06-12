"""Validate downloaded market data: coverage, gaps, sanity, and alignment.

Run:  python data/validate.py
Exits non-zero if any CHECK fails. INFO lines are expected quirks, not errors.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

FAILURES = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def nan_runs(s: pd.Series) -> list[tuple[str, str, int]]:
    """Runs of consecutive NaN between first and last valid index."""
    s = s.loc[s.first_valid_index() : s.last_valid_index()]
    isna = s.isna().to_numpy()
    runs, start = [], None
    for i, v in enumerate(isna):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((str(s.index[start].date()), str(s.index[i - 1].date()), i - start))
            start = None
    return sorted(runs, key=lambda r: -r[2])


def main() -> None:
    close = pd.read_parquet(config.RAW_CLOSE_PARQUET)
    ret = pd.read_parquet(config.RETURNS_PARQUET)

    # --- structure ---
    check("all 14 tickers present in both files",
          set(config.TICKERS) == set(close.columns) == set(ret.columns))
    check("index is unique and sorted",
          close.index.is_unique and close.index.is_monotonic_increasing)
    check("weekdays only", (close.index.dayofweek < 5).all())
    check("calendar starts 2016", str(close.index[0].date()) <= "2016-01-04")

    # --- freshness: every ticker must have data near the calendar end ---
    last_cal = close.index[-1]
    for t in config.TICKERS:
        last = close[t].last_valid_index()
        check(f"{t} data is fresh", (last_cal - last).days <= 7,
              f"last valid {last.date()}")

    # --- prices sane ---
    for t in config.TICKERS:
        px = close[t].dropna()
        nonpos = px[px <= 0]
        if t == "CL=F" and len(nonpos):
            info(f"CL=F non-positive prices on {[str(d.date()) for d in nonpos.index]} "
                 f"(April 2020 negative-oil event; log return -> NaN there, expected)")
        else:
            check(f"{t} prices strictly positive", len(nonpos) == 0,
                  f"{len(nonpos)} non-positive rows")

    # --- gaps: NaN runs inside each series' live range ---
    for t in config.TICKERS:
        runs = nan_runs(ret[t])
        longest = runs[0][2] if runs else 0
        total = sum(r[2] for r in runs)
        ok = longest <= 10  # longest market closure we tolerate (long holidays)
        check(f"{t} no gap longer than 10 trading days", ok,
              f"longest {longest}d {runs[0][:2] if runs else ''}, total NaN {total}")
    info("ETH-USD has no data before 2017-11-09 (listing date) — windows there are skipped")

    # --- ffill artifact level: fraction of exact-zero returns ---
    for t in config.TICKERS:
        r = ret[t].dropna()
        zfrac = (r == 0).mean()
        check(f"{t} zero-return fraction < 10%", zfrac < 0.10, f"{zfrac:.1%}")

    # --- return magnitudes ---
    for t in config.TICKERS:
        r = ret[t].dropna()
        extreme = r[r.abs() > 0.5]
        if len(extreme):
            info(f"{t} has {len(extreme)} day(s) with |move| > 50%: "
                 f"{[str(d.date()) for d in extreme.index[:5]]}")
        check(f"{t} daily std plausible", 0.0005 < r.std() < 0.15, f"std={r.std():.4f}")

    # --- split sizes (window END date convention) ---
    dates = ret.index
    train_end = pd.Timestamp(config.TRAIN_END)
    val_end = pd.Timestamp(config.VAL_END)
    n_train = (dates <= train_end).sum()
    n_val = ((dates > train_end) & (dates <= val_end)).sum()
    n_test = (dates > val_end).sum()
    info(f"split row counts — train/reference: {n_train}, val (2026 Q1): {n_val}, "
         f"test (Q2 to date): {n_test}")
    check("enough history for first full window",
          n_train > config.WINDOW_LEN + 2 * config.MAX_LAG)
    check("validation split non-trivial", n_val >= 55, f"{n_val} rows")
    check("test split non-trivial", n_test >= 40, f"{n_test} rows")

    # --- end-to-end alignment sanity: known market relationships ---
    r = ret
    c1 = r["^GSPC"].shift(1).corr(r["^TWII"])
    check("S&P500(t-1) vs TWII(t) positively correlated (US leads Asia)",
          c1 > 0.15, f"corr={c1:.3f}")
    c2 = r["^GSPC"].shift(1).corr(r["^N225"])
    check("S&P500(t-1) vs N225(t) positively correlated", c2 > 0.15, f"corr={c2:.3f}")
    c3 = r["^VIX"].corr(r["^GSPC"])
    check("VIX vs S&P500 strongly negative same-day", c3 < -0.55, f"corr={c3:.3f}")
    c4 = r["BTC-USD"].corr(r["ETH-USD"])
    check("BTC vs ETH strongly positive same-day", c4 > 0.5, f"corr={c4:.3f}")
    c5 = r["^GSPC"].shift(-1).corr(r["^TWII"])
    check("reverse direction (TWII leads S&P) is weaker than forward",
          c1 > abs(c5), f"forward={c1:.3f}, reverse={c5:.3f}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
