"""Instructor utility: regenerate the small pretrained 1D score checkpoint.

Run from the workspace root: .venv/bin/python L2-DiffusionSamplers/train_demo_score.py
Students load the bundled weights; they do not run this training script.
"""

from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from lesson2_kernel_demo import DemoConfig, DemoScoreNetwork, marginal_parameters, mixture_score


# Fixed target and diffusion configuration shipped with the checkpoint.
CONFIG = DemoConfig(prior_std=1.0, num_steps=600, schedule="linear",
                    delta_start=0.005, delta_end=0.035, mode_location=3.0, component_std=0.55)
MODEL_WIDTH = 64         # Width of both MLP hidden layers.
TRAINING_STEPS = 8000    # Supervised score-fitting updates (instructor only).
BATCH_SIZE = 1024        # Fresh states/time indices sampled per update.
LEARNING_RATE = 1e-3     # Initial Adam learning rate; cosine decay to 1e-5.
SEED = 42               # Reproducible initialization and training samples.
CPU_THREADS = 2         # Small CPU model; many threads add overhead.
LOG_EVERY = 1000         # Progress-report interval.
VALIDATION_SAMPLES = 8192  # Independent noisy states for score validation.
OUTPUT = Path(__file__).resolve().parent / "assets" / "bimodal_score.pt"


def score_batch(means, variances, count, *, generator=None, include_tails=False):
    # Half the indices emphasize the small-k, strongly bimodal marginals.
    times = torch.rand(count, 1, generator=generator)
    times[:count // 2].square_()
    indices = (times * CONFIG.num_steps).long().clamp(0, CONFIG.num_steps)
    mean, variance = means[indices], variances[indices]
    signs = 2 * torch.randint(0, 2, (count, 1), generator=generator) - 1
    states = signs * mean + variance.sqrt() * torch.randn(count, 1, generator=generator)
    if include_tails:
        # Cover the low-density gap and tails as well as typical noisy samples.
        states[:count // 5] = 14 * torch.rand(count // 5, 1, generator=generator) - 7
    return states, indices, mixture_score(states, mean, variance), variance


def main():
    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(SEED)
    model = DemoScoreNetwork(MODEL_WIDTH, CONFIG.prior_std)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAINING_STEPS, eta_min=1e-5)
    means, variances = marginal_parameters(CONFIG)
    history = []
    for step in range(1, TRAINING_STEPS + 1):
        states, indices, target, variance = score_batch(means, variances, BATCH_SIZE, include_tails=True)
        predicted = model(states, indices, CONFIG.num_steps)
        loss = (variance * (predicted - target).square()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % LOG_EVERY == 0:
            value = float(loss.detach())
            history.append({"step": step, "weighted_mse": value})
            print(f"update {step:5d}: weighted score MSE {value:.6g}", flush=True)
    model.eval().requires_grad_(False)
    states, indices, target, variance = score_batch(
        means, variances, VALIDATION_SAMPLES, generator=torch.Generator().manual_seed(SEED + 1))
    with torch.no_grad():
        error = model(states, indices, CONFIG.num_steps) - target
    weighted_rmse = float((variance * error.square()).mean().sqrt())
    relative_rmse = float(error.square().mean().sqrt() / target.square().mean().sqrt())
    metadata = {
        "format_version": 1,
        "diffusion": asdict(CONFIG),
        "model_width": MODEL_WIDTH,
        "training": {"steps": TRAINING_STEPS, "batch_size": BATCH_SIZE,
                     "learning_rate": LEARNING_RATE, "seed": SEED,
                     "objective": "variance-weighted squared error to the exact forward-mixture score",
                     "state_sampling": "80% forward marginals, 20% uniform [-7,7]; extra early-time samples",
                     "history": history},
        "validation": {"samples": VALIDATION_SAMPLES, "seed": SEED + 1,
                       "weighted_score_rmse": weighted_rmse, "relative_score_rmse": relative_rmse},
        "note": "A trained neural score, not an analytic-score replacement. Reverse Euler sampling is approximate.",
    }
    if not math.isfinite(weighted_rmse) or weighted_rmse > 0.08:
        raise RuntimeError(f"Score fit is not accurate enough: weighted RMSE={weighted_rmse}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT)
    OUTPUT.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {OUTPUT}; validation weighted RMSE={weighted_rmse:.5f}, relative RMSE={relative_rmse:.5f}")


if __name__ == "__main__":
    main()
