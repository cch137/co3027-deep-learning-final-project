"""Check the quiz path (make_batch with n=1) matches known test-set accuracy."""

import numpy as np
import torch
import torch.nn.functional as F

import config
from models.net import LeadLagNet
from synth.generator import make_batch

ckpt = torch.load("outputs/model.pt", weights_only=True)
model = LeadLagNet(ckpt["max_lag"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

rng = np.random.default_rng(11)
K = config.MAX_LAG
hits1, exact, n_corr, n_uncorr, uncorr_ok, weak = 0, 0, 0, 0, 0, 0

with torch.no_grad():
    for _ in range(500):
        d = make_batch(1, rng)
        x = torch.from_numpy(d["x"])
        logits, rho = model(x)
        tau_pred = int(logits.argmax(dim=1)) - K
        if d["corr_mask"][0] == 1:
            n_corr += 1
            tau_true = int(d["lag_class"][0]) - K
            exact += tau_pred == tau_true
            hits1 += abs(tau_pred - tau_true) <= 1
            weak += abs(d["rho"][0]) < 0.2
        else:
            n_uncorr += 1
            uncorr_ok += abs(float(rho[0])) < 0.2

print(f"correlated n={n_corr}: exact {exact / n_corr:.3f}, within ±1 {hits1 / n_corr:.3f}")
print(f"  of which weak (|rho|<0.2, lag nearly unidentifiable): {weak / n_corr:.1%}")
print(f"uncorrelated n={n_uncorr}: correctly flagged {uncorr_ok / max(n_uncorr, 1):.3f}")
