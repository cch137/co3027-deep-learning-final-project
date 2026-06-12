"""Generate all PPT-ready figures and a metrics summary.

Run:  python figures.py [--ckpt outputs/model.pt]
Writes PNGs (200 dpi) and metrics_summary.md to outputs/figures/.
"""

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
from eval import CASE_STUDIES, UNCORR_THRESHOLD, direction3, model_predict, real_windows
from models.baselines import xcorr_scan, xcorr_scan_batch
from models.net import LeadLagNet
from synth.generator import make_fixed_set

OUT = Path("outputs/figures")
TEST_SEED = 999
VAL_PERIOD = ("2026-01-01", config.VAL_END)
TEST_PERIOD = ("2026-04-01", "2026-12-31")

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name: str) -> None:
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  wrote {OUT / name}")


# ---------------------------------------------------------------- fig 1
def fig_training_curves() -> None:
    h = pd.read_csv("outputs/val_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(h["step"], h["lag_acc"], "o-", label="lag exact acc")
    ax.plot(h["step"], h["lag_acc1"], "s-", label="lag acc within ±1")
    ax.axhline(1 / 21, color="gray", ls="--", lw=1, label="random guess (4.8%)")
    ax.set_xlabel("training step"); ax.set_ylabel("validation accuracy")
    ax.set_title("Lag estimation accuracy during training")
    ax.legend(); ax.set_ylim(0, 1)
    ax = axes[1]
    ax.plot(h["step"], h["rho_mae"], "o-", color="tab:red", label="rho MAE")
    ax2 = ax.twinx()
    ax2.plot(h["step"], h["val_loss"], "s-", color="tab:gray", alpha=0.6, label="val loss")
    ax2.set_ylabel("validation loss", color="tab:gray")
    ax.set_xlabel("training step"); ax.set_ylabel("validation rho MAE", color="tab:red")
    ax.set_title("Correlation error and loss during training")
    save(fig, "fig1_training_curves.png")


# ---------------------------------------------------------------- fig 2
def fig_dataset_overview(close: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    cmap = plt.get_cmap("tab20")
    for i, t in enumerate(config.TICKERS):
        px = close[t].dropna()
        ax.plot(px.index, 100 * px / px.iloc[0], lw=1.1, color=cmap(i % 20),
                label=f"{t} ({config.TICKERS[t]})")
    ax.set_yscale("log")
    ax.set_ylabel("indexed level (start = 100, log scale)")
    ax.set_title("Asset universe: 14 markets, daily, 2016 – 2026")
    ax.axvspan(pd.Timestamp("2026-01-01"), pd.Timestamp(config.VAL_END),
               color="orange", alpha=0.15)
    ax.axvspan(pd.Timestamp("2026-04-01"), close.index[-1], color="red", alpha=0.12)
    ax.legend(fontsize=7.5, ncol=2, loc="upper left", framealpha=0.9)
    save(fig, "fig2_dataset_overview.png")


# ---------------------------------------------------------------- fig 3
def fig_leadlag_motivation(ret: pd.DataFrame) -> None:
    train = ret.loc[: config.TRAIN_END]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    taus = np.arange(-config.MAX_LAG, config.MAX_LAG + 1)
    for b, color in [("^TWII", "tab:blue"), ("^N225", "tab:orange"), ("^HSI", "tab:green")]:
        pair = train[["^GSPC", b]].dropna()
        curve = xcorr_scan(pair["^GSPC"].to_numpy(), pair[b].to_numpy(), config.MAX_LAG)["curve"]
        ax.plot(taus, curve, "o-", color=color, label=f"S&P 500 → {b}")
    ax.axvline(0, color="gray", lw=1, ls="--")
    ax.set_xticks(taus)
    ax.set_xlabel("lag τ (trading days; τ > 0 means S&P 500 leads)")
    ax.set_ylabel("Pearson correlation")
    ax.set_title("Motivation: the US close leads Asian indices by exactly 1 day\n"
                 "(cross-correlation, train period 2016–2025)")
    ax.legend()
    save(fig, "fig3_leadlag_motivation.png")


# ---------------------------------------------------------------- fig 4
def fig_synthetic_examples() -> None:
    d = make_fixed_set(3000, seed=7)
    tau = d["lag_class"] - config.MAX_LAG
    m = d["corr_mask"] == 1
    picks = [
        np.flatnonzero(m & (tau == 5) & (d["rho"] > 0.85))[0],
        np.flatnonzero(m & (tau == -3) & (d["rho"] < -0.6))[0],
        np.flatnonzero(~m)[0],
    ]
    titles = [
        lambda i: f"correlated: τ = {tau[i]:+d} (A leads B), ρ = {d['rho'][i]:+.2f}",
        lambda i: f"anti-correlated: τ = {tau[i]:+d} (B leads A), ρ = {d['rho'][i]:+.2f}",
        lambda i: "independent pair: ρ = 0, lag undefined",
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.6), sharex=True)
    for ax, i, title in zip(axes, picks, titles):
        ax.plot(d["x"][i, 0], lw=1.2, label="series A")
        ax.plot(d["x"][i, 1] - 4, lw=1.2, label="series B (offset)")
        ax.set_title(title(i), fontsize=10)
        ax.set_yticks([])
    axes[0].legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("time step")
    fig.suptitle("Synthetic training pairs: B = β·shift(A, τ) + noise — labels are free", y=1.0)
    save(fig, "fig4_synthetic_examples.png")


# ---------------------------------------------------------------- figs 5-7 + metrics
def synthetic_test(model) -> dict:
    d = make_fixed_set(20000, seed=TEST_SEED)
    tau_true = d["lag_class"] - config.MAX_LAG
    m = d["corr_mask"] == 1
    mp = model_predict(model, d["x"][:, 0], d["x"][:, 1])
    bp = xcorr_scan_batch(d["x"][:, 0], d["x"][:, 1], config.MAX_LAG)

    def metrics(p):
        return {
            "lag_acc": (p["tau"][m] == tau_true[m]).mean(),
            "lag_acc1": (np.abs(p["tau"][m] - tau_true[m]) <= 1).mean(),
            "rho_mae": np.abs(p["rho"] - d["rho"]).mean(),
            "dir3": (direction3(p["rho"].copy()) == direction3(d["rho"].copy())).mean(),
        }

    buckets = [(0.5, 1.01, "|ρ| ≥ 0.5"), (0.2, 0.5, "0.2 ≤ |ρ| < 0.5"), (0.0, 0.2, "|ρ| < 0.2")]
    bucket_rows = []
    for lo, hi, label in buckets:
        sel = m & (np.abs(d["rho"]) >= lo) & (np.abs(d["rho"]) < hi)
        bucket_rows.append({
            "label": label, "n": int(sel.sum()),
            "model": (np.abs(mp["tau"][sel] - tau_true[sel]) <= 1).mean(),
            "baseline": (np.abs(bp["tau"][sel] - tau_true[sel]) <= 1).mean(),
        })
    return {"d": d, "tau_true": tau_true, "m": m, "mp": mp, "bp": bp,
            "model": metrics(mp), "baseline": metrics(bp), "buckets": bucket_rows}


def fig_model_vs_baseline(s: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [3, 1]})
    names = ["lag exact acc", "lag acc ±1", "direction acc\n(pos/neg/none)"]
    keys = ["lag_acc", "lag_acc1", "dir3"]
    xpos = np.arange(len(names))
    ax = axes[0]
    for off, (label, src, color) in enumerate([
            ("LeadLagNet (ours)", s["model"], "tab:blue"),
            ("xcorr baseline", s["baseline"], "tab:gray")]):
        vals = [src[k] for k in keys]
        bars = ax.bar(xpos + (off - 0.5) * 0.36, vals, 0.36, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=9)
    ax.set_xticks(xpos, names); ax.set_ylim(0, 1.1)
    ax.set_title("Synthetic test set (n = 20,000): accuracy")
    ax.legend()
    ax = axes[1]
    bars = ax.bar(["LeadLagNet", "baseline"],
                  [s["model"]["rho_mae"], s["baseline"]["rho_mae"]],
                  color=["tab:blue", "tab:gray"])
    ax.bar_label(bars, fmt="%.3f", fontsize=9)
    ax.set_title("ρ MAE (lower = better)")
    save(fig, "fig5_model_vs_baseline.png")


def fig_confusion(s: dict) -> None:
    K = config.MAX_LAG
    m, tt, pt = s["m"], s["tau_true"], s["mp"]["tau"]
    cm = np.zeros((2 * K + 1, 2 * K + 1))
    for t, p in zip(tt[m], pt[m]):
        cm[t + K, p + K] += 1
    cm /= cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.8, 6))
    im = ax.imshow(cm, cmap="Blues", origin="lower",
                   extent=[-K - 0.5, K + 0.5, -K - 0.5, K + 0.5], vmin=0, vmax=1)
    ax.plot([-K, K], [-K, K], color="red", lw=0.8, alpha=0.5)
    ax.set_xlabel("predicted τ"); ax.set_ylabel("true τ")
    ax.set_title("Lag confusion matrix (model, correlated pairs)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="row-normalized frequency", shrink=0.85)
    save(fig, "fig6_lag_confusion.png")


def fig_difficulty(s: dict) -> None:
    rows = s["buckets"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    xpos = np.arange(len(rows))
    for off, key, label, color in [(-0.5, "model", "LeadLagNet (ours)", "tab:blue"),
                                   (0.5, "baseline", "xcorr baseline", "tab:gray")]:
        bars = ax.bar(xpos + off * 0.36, [r[key] for r in rows], 0.36,
                      label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=9)
    ax.set_xticks(xpos, [f"{r['label']}\n(n={r['n']})" for r in rows])
    ax.set_ylabel("lag acc within ±1"); ax.set_ylim(0, 1.1)
    ax.set_title("Both methods degrade as correlation weakens — lag is\n"
                 "fundamentally unidentifiable when |ρ| is small")
    ax.legend()
    save(fig, "fig7_difficulty_buckets.png")


# ---------------------------------------------------------------- fig 8
def fig_case_studies(model, ret: pd.DataFrame) -> list[dict]:
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    taus = np.arange(-config.MAX_LAG, config.MAX_LAG + 1)
    for ax, (a, b, desc) in zip(axes.flat, CASE_STUDIES):
        A, B = real_windows(ret, a, b, VAL_PERIOD)
        mp = model_predict(model, A, B)
        bp = xcorr_scan_batch(A, B, config.MAX_LAG)
        mode_tau = int(pd.Series(mp["tau"]).mode().iloc[0])
        mean_curve = np.nanmean(bp["curves"], axis=0)
        ax.plot(taus, mean_curve, "o-", color="tab:gray", label="xcorr (baseline)")
        ax.axvline(mode_tau, color="tab:blue", lw=2, ls="--",
                   label=f"model: τ = {mode_tau:+d}, ρ = {mp['rho'].mean():+.2f}")
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xticks(taus[::2])
        ax.set_title(desc, fontsize=10)
        ax.set_xlabel("τ (days)"); ax.set_ylabel("correlation")
        ax.legend(fontsize=8.5)
        rows.append({"pair": desc, "model_tau": mode_tau,
                     "model_rho": float(mp["rho"].mean()),
                     "base_tau": int(pd.Series(bp["tau"]).mode().iloc[0]),
                     "base_rho": float(bp["rho"].mean())})
    fig.suptitle("Real-data case studies (validation period 2026 Q1):\n"
                 "model output vs classical cross-correlation", y=1.0)
    fig.tight_layout()
    save(fig, "fig8_case_studies.png")
    return rows


# ---------------------------------------------------------------- fig 9
def fig_pair_heatmap(model, ret: pd.DataFrame) -> None:
    tickers = list(config.TICKERS)
    n = len(tickers)
    mat = np.eye(n)
    for i, j in itertools.combinations(range(n), 2):
        A, B = real_windows(ret, tickers[i], tickers[j], VAL_PERIOD)
        if A is None:
            mat[i, j] = mat[j, i] = np.nan
            continue
        rho = model_predict(model, A, B)["rho"].mean()
        mat[i, j] = mat[j, i] = rho
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), tickers, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(range(n), tickers, fontsize=9)
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(mat[i, j]) > 0.55 else "black")
    ax.set_title("Model-estimated correlation structure across all pairs\n"
                 "(mean ρ̂ at best lag, validation period 2026 Q1)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="ρ̂", shrink=0.8)
    save(fig, "fig9_pair_heatmap.png")


# ---------------------------------------------------------------- summary
def real_agreement(model, ret: pd.DataFrame, period) -> dict:
    agree, n_sig, n_win, rho_diff = 0, 0, 0, []
    for a, b in itertools.combinations(config.TICKERS, 2):
        A, B = real_windows(ret, a, b, period)
        if A is None:
            continue
        mp = model_predict(model, A, B)
        bp = xcorr_scan_batch(A, B, config.MAX_LAG)
        n_win += len(A)
        rho_diff.append(np.abs(mp["rho"] - bp["rho"]))
        sig = np.abs(bp["rho"]) >= UNCORR_THRESHOLD
        n_sig += sig.sum()
        agree += (np.abs(mp["tau"][sig] - bp["tau"][sig]) <= 1).sum()
    return {"windows": n_win, "lag_agree": agree / max(n_sig, 1),
            "rho_mae": float(np.concatenate(rho_diff).mean())}


def write_summary(s: dict, cases: list[dict], real_val: dict, real_test: dict) -> None:
    md = ["# Metrics summary (for PPT)\n"]
    md.append("## Model\n")
    md.append("- LeadLagNet: siamese 1D-CNN + correlation fusion, **130,502 params**")
    md.append("- Trained 8,000 steps (~14 min, CPU only) on on-the-fly synthetic pairs")
    md.append("- Best checkpoint at step 7,000 (validation composite score)\n")
    md.append("## Synthetic test set (n = 20,000, seed 999)\n")
    md.append("| method | lag exact | lag ±1 | ρ MAE | direction acc |")
    md.append("|---|---|---|---|---|")
    for name, r in [("LeadLagNet (ours)", s["model"]), ("xcorr baseline", s["baseline"])]:
        md.append(f"| {name} | {r['lag_acc']:.3f} | {r['lag_acc1']:.3f} | "
                  f"{r['rho_mae']:.3f} | {r['dir3']:.3f} |")
    md.append("\nLag ±1 accuracy by correlation strength:\n")
    md.append("| bucket | n | model | baseline |")
    md.append("|---|---|---|---|")
    for r in s["buckets"]:
        md.append(f"| {r['label']} | {r['n']} | {r['model']:.3f} | {r['baseline']:.3f} |")
    md.append("\n## Real market data — agreement with baseline\n")
    md.append("| period | windows | lag agreement (±1) | ρ MAE vs baseline |")
    md.append("|---|---|---|---|")
    md.append(f"| validation 2026 Q1 | {real_val['windows']} | "
              f"{real_val['lag_agree']:.3f} | {real_val['rho_mae']:.3f} |")
    md.append(f"| test 2026 Q2 | {real_test['windows']} | "
              f"{real_test['lag_agree']:.3f} | {real_test['rho_mae']:.3f} |")
    md.append("\n## Case studies (validation 2026 Q1)\n")
    md.append("| pair | baseline τ | model τ | baseline ρ | model ρ |")
    md.append("|---|---|---|---|---|")
    for c in cases:
        md.append(f"| {c['pair']} | {c['base_tau']:+d} | {c['model_tau']:+d} | "
                  f"{c['base_rho']:+.3f} | {c['model_rho']:+.3f} |")
    md.append("\n## Figures\n")
    for f, use in [
        ("fig1_training_curves.png", "training progress slide"),
        ("fig2_dataset_overview.png", "dataset slide"),
        ("fig3_leadlag_motivation.png", "motivation slide (why lead-lag exists)"),
        ("fig4_synthetic_examples.png", "method slide (label-free training data)"),
        ("fig5_model_vs_baseline.png", "results slide (synthetic benchmark)"),
        ("fig6_lag_confusion.png", "results slide (error structure)"),
        ("fig7_difficulty_buckets.png", "analysis slide (identifiability limit)"),
        ("fig8_case_studies.png", "results slide (real-market validation)"),
        ("fig9_pair_heatmap.png", "results slide (market structure map)"),
    ]:
        md.append(f"- `{f}` — {use}")
    (OUT / "metrics_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  wrote {OUT / 'metrics_summary.md'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/model.pt")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, weights_only=True)
    model = LeadLagNet(ckpt["max_lag"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    close = pd.read_parquet(config.RAW_CLOSE_PARQUET)
    ret = pd.read_parquet(config.RETURNS_PARQUET)

    print("generating figures...")
    fig_training_curves()
    fig_dataset_overview(close)
    fig_leadlag_motivation(ret)
    fig_synthetic_examples()
    s = synthetic_test(model)
    fig_model_vs_baseline(s)
    fig_confusion(s)
    fig_difficulty(s)
    cases = fig_case_studies(model, ret)
    fig_pair_heatmap(model, ret)
    print("computing real-data aggregates...")
    real_val = real_agreement(model, ret, VAL_PERIOD)
    real_test = real_agreement(model, ret, TEST_PERIOD)
    write_summary(s, cases, real_val, real_test)
    print("done.")


if __name__ == "__main__":
    main()
