"""Sanity cross-check: the classical baseline must recover the generator's labels."""

import time

import numpy as np

import config
from models.baselines import xcorr_scan_batch
from synth.generator import make_fixed_set

t0 = time.time()
d = make_fixed_set(4000, seed=42)
gen_t = time.time() - t0

m = d["corr_mask"] == 1
t0 = time.time()
res = xcorr_scan_batch(d["x"][:, 0], d["x"][:, 1], config.MAX_LAG)
scan_t = time.time() - t0

tau_true = d["lag_class"] - config.MAX_LAG
exact = (res["tau"][m] == tau_true[m]).mean()
within1 = (np.abs(res["tau"][m] - tau_true[m]) <= 1).mean()
rho_mae = np.abs(res["rho"][m] - d["rho"][m]).mean()
abs_rho = np.abs(d["rho"][m])

print(f"gen 4000 samples: {gen_t:.2f}s, xcorr scan: {scan_t:.2f}s")
print(f"correlated n={m.sum()}, uncorrelated n={(~m).sum()}")
print(f"baseline lag exact acc: {exact:.3f}, within +-1: {within1:.3f}")
print(f"baseline rho MAE vs label: {rho_mae:.4f}")
print(f"|rho| label dist: mean={abs_rho.mean():.3f}, "
      f"q10={np.quantile(abs_rho, 0.1):.3f}, q90={np.quantile(abs_rho, 0.9):.3f}")
print(f"baseline spurious |rho| on uncorrelated pairs: {np.abs(res['rho'][~m]).mean():.3f}")
counts = np.bincount(d["lag_class"][m], minlength=2 * config.MAX_LAG + 1)
print(f"lag class balance: min={counts.min()}, max={counts.max()}")
