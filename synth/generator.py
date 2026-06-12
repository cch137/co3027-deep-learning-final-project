"""Synthetic training data: pairs with known lag/correlation labels, zero annotation cost.

Each correlated pair is built as
    B(t) = beta * A(t - tau) + sigma * eps(t)
with tau in [-MAX_LAG, +MAX_LAG] (tau > 0: A leads B), |beta| in [0.3, 3] with
random sign, and sigma controlling correlation strength. ~15% of pairs are
independent (uncorrelated): label rho = 0 and the lag loss is masked.

The rho label is the REALIZED Pearson correlation at the true lag, so it
reflects the actual noise draw, not just the theoretical value.

Inputs handed to the model are per-window z-scored, matching how real-data
windows are normalized at evaluation time.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

P_UNCORRELATED = 0.15
BETA_RANGE = (0.3, 3.0)
SIGMA_RANGE = (0.1, 3.0)


def _ar1(n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    phi = rng.uniform(-0.9, 0.9, size=(n, 1))
    e = rng.standard_normal((n, length))
    x = np.empty((n, length))
    x[:, 0] = e[:, 0]
    for t in range(1, length):
        x[:, t] = phi[:, 0] * x[:, t - 1] + e[:, t]
    return x


def _sines(n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(length)
    x = 0.3 * rng.standard_normal((n, length))
    n_comp = rng.integers(2, 4, size=n)  # 2-3 incommensurate sines: avoids pure periodicity
    for k in range(3):
        active = (n_comp > k).astype(float)[:, None]
        amp = rng.uniform(0.5, 1.5, size=(n, 1))
        freq = rng.uniform(0.02, 0.45, size=(n, 1))
        phase = rng.uniform(0, 2 * np.pi, size=(n, 1))
        x += active * amp * np.sin(2 * np.pi * freq * t[None, :] + phase)
    return x


def _white(n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((n, length))

_BASE_GENS = [_white, _ar1, _sines]


def _base_signals(n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    """Random mix of base signal types, each row standardized."""
    kind = rng.integers(0, len(_BASE_GENS), size=n)
    x = np.empty((n, length))
    for k, gen in enumerate(_BASE_GENS):
        m = kind == k
        if m.any():
            x[m] = gen(int(m.sum()), length, rng)
    x -= x.mean(axis=1, keepdims=True)
    x /= x.std(axis=1, keepdims=True) + 1e-12
    return x


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-12)


def _rowwise_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xm = x - x.mean(axis=1, keepdims=True)
    ym = y - y.mean(axis=1, keepdims=True)
    denom = np.sqrt((xm**2).sum(axis=1) * (ym**2).sum(axis=1)) + 1e-12
    return (xm * ym).sum(axis=1) / denom


def make_batch(n: int, rng: np.random.Generator) -> dict:
    """Generate n labelled pairs.

    Returns dict of arrays:
      x          (n, 2, L) float32  z-scored [A, B] windows
      lag_class  (n,)      int64    tau + MAX_LAG (only valid where corr_mask=1)
      rho        (n,)      float32  realized Pearson corr at the true lag
      corr_mask  (n,)      float32  1 = correlated pair, 0 = independent pair
    """
    L, K = config.WINDOW_LEN, config.MAX_LAG
    buf = L + 2 * K
    a_full = _base_signals(n, buf, rng)
    a = a_full[:, K : K + L]

    tau = rng.integers(-K, K + 1, size=n)
    sign = rng.choice([-1.0, 1.0], size=n)
    beta = sign * np.exp(rng.uniform(np.log(BETA_RANGE[0]), np.log(BETA_RANGE[1]), size=n))
    sigma = np.exp(rng.uniform(np.log(SIGMA_RANGE[0]), np.log(SIGMA_RANGE[1]), size=n))

    # a_shift[i, t] = a_full[i, K - tau[i] + t]  (the lagged copy B is built from)
    rows = np.arange(n)[:, None]
    cols = (K - tau)[:, None] + np.arange(L)[None, :]
    a_shift = a_full[rows, cols]

    b = beta[:, None] * a_shift + sigma[:, None] * rng.standard_normal((n, L))

    uncorr = rng.random(n) < P_UNCORRELATED
    if uncorr.any():
        b[uncorr] = _base_signals(int(uncorr.sum()), buf, rng)[:, K : K + L]

    rho = _rowwise_corr(a_shift, b)
    rho[uncorr] = 0.0

    x = np.stack([_zscore(a), _zscore(b)], axis=1).astype(np.float32)
    return {
        "x": x,
        "lag_class": (tau + K).astype(np.int64),
        "rho": rho.astype(np.float32),
        "corr_mask": (~uncorr).astype(np.float32),
    }


def make_fixed_set(n: int, seed: int) -> dict:
    """Deterministic labelled set (for validation / test)."""
    return make_batch(n, np.random.default_rng(seed))
