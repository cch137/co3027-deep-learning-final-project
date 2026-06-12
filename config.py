"""Project-wide constants: asset universe, date ranges, window/lag settings."""

# yfinance tickers -> human-readable name
TICKERS = {
    "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq Composite",
    "^VIX": "CBOE VIX",
    "^N225": "Nikkei 225",
    "^HSI": "Hang Seng",
    "^TWII": "Taiwan Weighted",
    "^FTSE": "FTSE 100",
    "^GDAXI": "DAX",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "GC=F": "Gold Futures",
    "CL=F": "WTI Crude Futures",
    "^TNX": "US 10Y Yield",
    "DX-Y.NYB": "US Dollar Index",
}

# ^TNX is a yield level: use first difference instead of log return
DIFF_TICKERS = {"^TNX"}

START_DATE = "2016-01-01"

# Splits are applied to the window END date
TRAIN_END = "2025-12-31"   # train / reference period
VAL_END = "2026-03-31"     # validation = 2026 Q1
# everything after VAL_END = test (2026 Q2 to date)

WINDOW_LEN = 128           # input window length (days)
MAX_LAG = 10               # lag head covers tau in [-MAX_LAG, +MAX_LAG]
FFILL_LIMIT = 3            # max consecutive days to forward-fill across holidays

DATA_DIR = "data"
RAW_CLOSE_PARQUET = "data/raw_close.parquet"
RETURNS_PARQUET = "data/returns.parquet"
