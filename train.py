"""Train LeadLagNet on on-the-fly synthetic pairs. CPU budget: well under 1 hour.

Run:  python train.py [--steps 8000] [--batch 128] [--lr 2e-3]
Saves best checkpoint (by validation composite score) to outputs/model.pt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
from models.net import LeadLagNet, count_params
from synth.generator import make_batch, make_fixed_set

RHO_LOSS_WEIGHT = 4.0
VAL_SEED = 123  # test set uses seed 999 in eval.py — never train/select on it


def to_tensors(d: dict) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(d["x"]),
        torch.from_numpy(d["lag_class"]),
        torch.from_numpy(d["rho"]),
        torch.from_numpy(d["corr_mask"]),
    )


def compute_loss(model, x, lag_class, rho, mask):
    lag_logits, rho_hat = model(x)
    # lag is undefined for independent pairs -> mask their CE term
    ce = F.cross_entropy(lag_logits, lag_class, reduction="none")
    ce = (ce * mask).sum() / mask.sum().clamp(min=1)
    mse = F.mse_loss(rho_hat, rho)
    return ce + RHO_LOSS_WEIGHT * mse, lag_logits, rho_hat


@torch.no_grad()
def evaluate(model, tensors) -> dict:
    model.eval()
    x, lag_class, rho, mask = tensors
    loss, lag_logits, rho_hat = compute_loss(model, x, lag_class, rho, mask)
    pred = lag_logits.argmax(dim=1)
    m = mask.bool()
    err = (pred[m] - lag_class[m]).abs()
    model.train()
    return {
        "loss": loss.item(),
        "lag_acc": (err == 0).float().mean().item(),
        "lag_acc1": (err <= 1).float().mean().item(),
        "rho_mae": (rho_hat - rho).abs().mean().item(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/model.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = LeadLagNet(config.MAX_LAG)
    print("=" * 78)
    print(f"LeadLagNet training  |  params: {count_params(model):,}  |  device: CPU "
          f"({torch.get_num_threads()} threads)")
    print(f"steps: {args.steps}  batch: {args.batch}  lr: {args.lr}  seed: {args.seed}")
    print(f"val set: 4096 synthetic pairs (seed {VAL_SEED})  |  checkpoint: {args.out}")
    print(f"random-guess reference: lag_acc ~{1 / (2 * config.MAX_LAG + 1):.3f}")
    print("=" * 78, flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    val_tensors = to_tensors(make_fixed_set(4096, seed=VAL_SEED))

    best_score, best_state = -1.0, None
    running = 0.0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, lag_class, rho, mask = to_tensors(make_batch(args.batch, rng))
        loss, _, _ = compute_loss(model, x, lag_class, rho, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item()

        if step % args.log_every == 0:
            speed = step / (time.time() - t0)
            eta_min = (args.steps - step) / speed / 60
            print(
                f"step {step:5d}/{args.steps}  train_loss {running / args.log_every:.4f}  "
                f"lr {sched.get_last_lr()[0]:.2e}  {speed:.1f} it/s  ETA {eta_min:4.1f} min",
                flush=True,
            )
            running = 0.0

        if step % args.val_every == 0 or step == args.steps:
            v = evaluate(model, val_tensors)
            score = v["lag_acc1"] - v["rho_mae"]  # composite for model selection
            mark = ""
            if score > best_score:
                best_score = score
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
                mark = "  *best -> checkpoint*"
            print(
                f"[VAL] step {step:5d}  loss {v['loss']:.4f}  lag_acc {v['lag_acc']:.3f}  "
                f"lag_acc±1 {v['lag_acc1']:.3f}  rho_mae {v['rho_mae']:.4f}{mark}",
                flush=True,
            )

    Path(args.out).parent.mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "max_lag": config.MAX_LAG,
                "window_len": config.WINDOW_LEN}, args.out)
    print(f"\nbest val score {best_score:.4f}; saved {args.out}  "
          f"(total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
