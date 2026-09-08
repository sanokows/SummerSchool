#!/usr/bin/env python3
"""Plot Gaussian forward/reverse KL fits to the same bimodal target.

Forward KL matches the target's moments exactly. Reverse KL is optimized
with deterministic Gauss-Hermite quadrature and several initializations;
the right-hand solution is shown (the left-hand one is equally good).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODE = 3.0
COMPONENT_STD = 0.65
INK = "#17243A"
SLATE = "#64748B"
BLUE = "#1E90FF"
TARGET = "#9E6424"
LOG_2PI = math.log(2 * math.pi)


def log_target(x: torch.Tensor) -> torch.Tensor:
    components = torch.stack(
        [-0.5 * ((x - mean) / COMPONENT_STD) ** 2 for mean in (-MODE, MODE)]
    )
    return (
        torch.logsumexp(components, dim=0)
        - math.log(2 * COMPONENT_STD)
        - 0.5 * LOG_2PI
    )


def fit_reverse() -> tuple[float, float]:
    nodes, weights = np.polynomial.hermite.hermgauss(160)
    noise = torch.tensor(nodes * math.sqrt(2), dtype=torch.float64)
    weights = torch.tensor(weights / math.sqrt(math.pi), dtype=torch.float64)
    candidates = []
    for initial_mean in (-MODE, 0.0, MODE):
        for initial_std in (COMPONENT_STD, math.hypot(MODE, COMPONENT_STD)):
            parameters = torch.tensor(
                [initial_mean, math.log(initial_std)],
                dtype=torch.float64,
                requires_grad=True,
            )
            optimizer = torch.optim.LBFGS(
                [parameters], max_iter=100, line_search_fn="strong_wolfe",
                tolerance_grad=1e-10, tolerance_change=1e-12,
            )

            def objective() -> torch.Tensor:
                mean, log_std = parameters
                x = mean + log_std.exp() * noise
                return -0.5 * (1 + LOG_2PI) - log_std - weights @ log_target(x)

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                loss = objective()
                loss.backward()
                return loss

            optimizer.step(closure)
            mean, log_std = parameters.detach().tolist()
            candidates.append((objective().detach().item(), mean, math.exp(log_std)))
    _, mean, std = min(candidates)
    return abs(mean), std


def log_normal(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return -0.5 * ((x - mean) / std) ** 2 - math.log(std) - 0.5 * LOG_2PI


def render_panel(direction: str, mean: float, std: float) -> None:
    x = np.linspace(-8, 8, 1601)
    target = np.exp(log_target(torch.from_numpy(x)).numpy())
    approximation = np.exp(log_normal(x, mean, std))
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.23, top=0.86)
    ax.fill_between(x, target, color=TARGET, alpha=0.12)
    ax.plot(x, target, color=TARGET, linewidth=2.4, label=r"Target $p$")
    ax.fill_between(x, approximation, color=BLUE, alpha=0.10)
    ax.plot(x, approximation, color=BLUE, linewidth=2.4, linestyle="--",
            label=r"Gaussian $q_\theta$")
    ax.set(xlim=(-8, 8), ylim=(0, 0.68), xticks=[-6, -3, 0, 3, 6],
           yticks=[0, 0.3, 0.6], xlabel=r"$x$", ylabel="Density")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color(SLATE)
    ax.tick_params(colors=SLATE, length=3)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              frameon=False, borderaxespad=0, handlelength=2.2)
    if direction == "forward":
        ax.annotate("Covers both modes", xy=(-3, 0.081), xytext=(0, 0.48),
                    ha="center", color=BLUE,
                    arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.2})
        ax.annotate("", xy=(3, 0.081), xytext=(1.1, 0.44),
                    arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.2})
    else:
        ax.annotate("Misses a mode", xy=(-3, 0.30), xytext=(-3, 0.49),
                    ha="center", color=SLATE,
                    arrowprops={"arrowstyle": "->", "color": SLATE, "lw": 1.2})
    output = ROOT / "assets" / f"kl_{direction}.pdf"
    fig.savefig(output, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"Wrote {output.relative_to(ROOT)}")


def main() -> None:
    torch.set_num_threads(1)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "text.color": INK, "pdf.fonttype": 42})
    forward = (0.0, math.hypot(MODE, COMPONENT_STD))
    reverse = fit_reverse()

    # Independently evaluate both directions by dense trapezoidal integration.
    x = np.linspace(-30, 30, 60001)
    log_p = log_target(torch.from_numpy(x)).numpy()
    p = np.exp(log_p)
    values = {}
    for name, (mean, std) in (("forward", forward), ("reverse", reverse)):
        log_q = log_normal(x, mean, std)
        q = np.exp(log_q)
        assert abs(np.trapezoid(q, x) - 1) < 1e-10
        values[name] = (
            np.trapezoid(p * (log_p - log_q), x),
            np.trapezoid(q * (log_q - log_p), x),
        )
        print(f"{name}: mean={mean:.6f}, std={std:.6f}, "
              f"KL(p||q)={values[name][0]:.6f}, KL(q||p)={values[name][1]:.6f}")
    assert abs(np.trapezoid(p, x) - 1) < 1e-10
    assert values["forward"][0] < values["reverse"][0]
    assert values["reverse"][1] < values["forward"][1]
    for direction, (mean, std) in (("forward", forward), ("reverse", reverse)):
        render_panel(direction, mean, std)


if __name__ == "__main__":
    main()
