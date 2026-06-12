"""Download daily close prices for the asset universe and build aligned return series.

Run once:  python data/download.py
Outputs:   data/raw_close.parquet  (wide close prices, union calendar)
           data/returns.parquet    (aligned daily log returns / diffs)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def download_close() -> pd.DataFrame:
    raw = yf.download(
        list(config.TICKERS),
        start=config.START_DATE,
        auto_adjust=True,
        progress=False,
    )
    close = raw["Close"].sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    # Crypto trades 7 days a week; keep weekdays only so equity series are not
    # padded with fake 0-return weekend rows. Crypto weekend moves fold into
    # Monday's return, and lags are measured in trading days.
    close = close[close.index.dayofweek < 5]
    return close


def build_returns(close: pd.DataFrame) -> pd.DataFrame:
    # union calendar with bounded forward-fill across market holidays
    filled = close.ffill(limit=config.FFILL_LIMIT)
    parts = {}
    for ticker in close.columns:
        if ticker in config.DIFF_TICKERS:
            parts[ticker] = filled[ticker].diff()
        else:
            parts[ticker] = np.log(filled[ticker]).diff()
    ret = pd.DataFrame(parts).dropna(how="all")
    return ret


def summarize(close: pd.DataFrame, ret: pd.DataFrame) -> None:
    print(f"calendar: {close.index[0].date()} .. {close.index[-1].date()}  ({len(close)} rows)")
    print(f"{'ticker':<10} {'first':<12} {'last':<12} {'NaN%':>6}")
    for ticker in config.TICKERS:
        col = close[ticker]
        nan_pct = 100 * ret[ticker].isna().mean()
        print(
            f"{ticker:<10} {str(col.first_valid_index().date()):<12} "
            f"{str(col.last_valid_index().date()):<12} {nan_pct:>5.1f}%"
        )


def main() -> None:
    close = download_close()
    missing = [t for t in config.TICKERS if t not in close.columns or close[t].dropna().empty]
    if missing:
        raise SystemExit(f"no data returned for: {missing}")

    ret = build_returns(close)
    Path(config.DATA_DIR).mkdir(exist_ok=True)
    close.to_parquet(config.RAW_CLOSE_PARQUET)
    ret.to_parquet(config.RETURNS_PARQUET)
    summarize(close, ret)
    print(f"\nsaved {config.RAW_CLOSE_PARQUET} and {config.RETURNS_PARQUET}")


if __name__ == "__main__":
    main()
