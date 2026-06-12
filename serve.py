"""FastAPI demo server for LeadLagNet.

Run:  .venv\\Scripts\\python -m uvicorn serve:app --port 8000
Open: http://127.0.0.1:8000
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
from models.baselines import xcorr_scan
from models.net import LeadLagNet
from synth.generator import make_batch

app = FastAPI(title="LeadLagNet demo")

_ckpt = torch.load("outputs/model.pt", weights_only=True)
MODEL = LeadLagNet(_ckpt["max_lag"])
MODEL.load_state_dict(_ckpt["state_dict"])
MODEL.eval()
RET = pd.read_parquet(config.RETURNS_PARQUET)
RNG = np.random.default_rng()
L, K = config.WINDOW_LEN, config.MAX_LAG


class AnalyzeReq(BaseModel):
    ticker_a: str
    ticker_b: str
    end_date: str | None = None


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-12)


@torch.no_grad()
def _predict(a_z: np.ndarray, b_z: np.ndarray) -> dict:
    x = torch.from_numpy(np.stack([a_z, b_z])[None].astype(np.float32))
    logits, rho = MODEL(x)
    probs = F.softmax(logits[0], dim=0).numpy()
    return {
        "tau": int(probs.argmax()) - K,
        "rho": float(rho[0]),
        "lag_probs": [round(float(p), 4) for p in probs],
    }


@app.get("/")
def index():
    return FileResponse("web/index.html")


@app.get("/quiz")
def quiz_page():
    return FileResponse("web/quiz.html")


@app.get("/api/assets")
def assets():
    return {
        "tickers": [{"symbol": t, "name": n} for t, n in config.TICKERS.items()],
        "min_date": str(RET.index[L].date()),
        "max_date": str(RET.index[-1].date()),
        "window_len": L,
        "max_lag": K,
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeReq):
    for t in (req.ticker_a, req.ticker_b):
        if t not in RET.columns:
            raise HTTPException(400, f"unknown ticker: {t}")
    if req.ticker_a == req.ticker_b:
        raise HTTPException(400, "pick two different assets")

    if req.end_date:
        pos = RET.index.searchsorted(pd.Timestamp(req.end_date), side="right") - 1
    else:
        pos = len(RET) - 1
    if pos + 1 < L:
        raise HTTPException(400, "not enough history before this date")

    w = RET[[req.ticker_a, req.ticker_b]].iloc[pos + 1 - L : pos + 1]
    if np.isnan(w.to_numpy()).mean(axis=0).max() > 0.10:
        raise HTTPException(400, "too many missing values in this window")
    raw = np.nan_to_num(w.to_numpy(), nan=0.0)
    a_raw, b_raw = raw[:, 0], raw[:, 1]
    a_z, b_z = _zscore(a_raw), _zscore(b_raw)

    pred = _predict(a_z, b_z)
    base = xcorr_scan(a_z, b_z, K)
    beta = pred["rho"] * (b_raw.std() + 1e-12) / (a_raw.std() + 1e-12)

    return {
        "window": {
            "start": str(w.index[0].date()),
            "end": str(w.index[-1].date()),
            "dates": [str(d.date()) for d in w.index],
            "series_a": [round(float(v), 4) for v in a_z],
            "series_b": [round(float(v), 4) for v in b_z],
        },
        "model": {**pred, "beta": round(float(beta), 4)},
        "baseline": {
            "tau": base["tau"],
            "rho": round(base["rho"], 4),
            "curve": [None if np.isnan(c) else round(float(c), 4) for c in base["curve"]],
        },
        "lags": list(range(-K, K + 1)),
    }


class QuizReq(BaseModel):
    difficulty: str = "easy"  # "easy": correlated with |rho|>=0.5; "any": full distribution


@app.post("/api/quiz")
def quiz(req: QuizReq):
    for _ in range(500):
        d = make_batch(1, RNG)
        if req.difficulty != "easy":
            break
        if d["corr_mask"][0] == 1 and abs(float(d["rho"][0])) >= 0.5:
            break
    a_z, b_z = d["x"][0, 0], d["x"][0, 1]
    pred = _predict(a_z, b_z)
    base = xcorr_scan(a_z.astype(float), b_z.astype(float), K)
    correlated = bool(d["corr_mask"][0] == 1)
    return {
        "truth": {
            "correlated": correlated,
            "tau": int(d["lag_class"][0]) - K if correlated else None,
            "rho": round(float(d["rho"][0]), 4),
        },
        "model": pred,
        "baseline": {
            "tau": base["tau"],
            "rho": round(base["rho"], 4),
            "curve": [None if np.isnan(c) else round(float(c), 4) for c in base["curve"]],
        },
        "series_a": [round(float(v), 4) for v in a_z],
        "series_b": [round(float(v), 4) for v in b_z],
        "lags": list(range(-K, K + 1)),
    }
