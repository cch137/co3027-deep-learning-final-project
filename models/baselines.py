"""Classical baseline: cross-correlation scan over discrete lags.

Convention used across the whole project:
    rho(tau) = corr(a[t - tau], b[t])
    tau > 0  means  A LEADS B by tau trading days (b follows a).
"""

import numpy as np


def xcorr_scan(a: np.ndarray, b: np.ndarray, max_lag: int) -> dict:
    """Scan tau in [-max_lag, +max_lag]; pick the lag with max |corr|.

    Returns {"tau": int, "rho": float, "beta": float, "curve": (2K+1,) array}
    where beta is the OLS slope of b on a at the chosen lag.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    assert len(b) == n
    taus = np.arange(-max_lag, max_lag + 1)
    curve = np.full(len(taus), np.nan)

    for i, tau in enumerate(taus):
        if tau >= 0:
            x, y = a[: n - tau], b[tau:]
        else:
            x, y = a[-tau:], b[: n + tau]
        if x.std() > 0 and y.std() > 0:
            curve[i] = np.corrcoef(x, y)[0, 1]

    best = int(np.nanargmax(np.abs(curve)))
    tau = int(taus[best])
    rho = float(curve[best])
    if tau >= 0:
        x, y = a[: n - tau], b[tau:]
    else:
        x, y = a[-tau:], b[: n + tau]
    beta = float(rho * y.std() / x.std()) if x.std() > 0 else np.nan
    return {"tau": tau, "rho": rho, "beta": beta, "curve": curve}


def xcorr_scan_batch(a: np.ndarray, b: np.ndarray, max_lag: int) -> dict:
    """Vectorized scan for batches. a, b: (N, L). Returns arrays per sample."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    N, L = a.shape
    taus = np.arange(-max_lag, max_lag + 1)
    curves = np.full((N, len(taus)), np.nan)

    for i, tau in enumerate(taus):
        if tau >= 0:
            x, y = a[:, : L - tau], b[:, tau:]
        else:
            x, y = a[:, -tau:], b[:, : L + tau]
        xm = x - x.mean(axis=1, keepdims=True)
        ym = y - y.mean(axis=1, keepdims=True)
        denom = np.sqrt((xm**2).sum(axis=1) * (ym**2).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            curves[:, i] = (xm * ym).sum(axis=1) / denom

    best = np.nanargmax(np.abs(curves), axis=1)
    tau = taus[best]
    rho = curves[np.arange(N), best]
    return {"tau": tau, "rho": rho, "curves": curves}
