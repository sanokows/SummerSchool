#!/usr/bin/env python3
"""Animate the discrete forward process in arXiv:2512.02019v3, Sec. 2.3.

Every density is the exact Gaussian-mixture marginal of the displayed Euler
update, not an interpolation between endpoint pictures. The dots follow the
same Markov transitions. Small finite steps approach an approximately Gaussian
prior: the Euler stationary variance is nu^2 / (1 - delta / 4).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "vp_diffusion_1d"
STEPS = 600
DELTA = 0.02
PRIOR_STD = 1.0
INITIAL_MEANS = np.array([-3.0, 3.0])
INITIAL_STD = 0.55
PARTICLES = 160
FPS = 10
SEED = 17
INK = "#17243A"
SLATE = "#64748B"
BLUE = "#1E90FF"
TRAIL = "#9E6424"


def normal_pdf(x: np.ndarray, mean: float, variance: float) -> np.ndarray:
    return np.exp(-0.5 * (x - mean) ** 2 / variance) / np.sqrt(2 * np.pi * variance)


def simulate() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    means = np.empty((STEPS + 1, 2))
    variances = np.empty(STEPS + 1)
    samples = np.empty((STEPS + 1, PARTICLES))
    means[0] = INITIAL_MEANS
    variances[0] = INITIAL_STD**2
    samples[0] = rng.choice(INITIAL_MEANS, PARTICLES) + INITIAL_STD * rng.normal(size=PARTICLES)
    contraction = 1 - DELTA / 2
    noise_variance = PRIOR_STD**2 * DELTA
    for k in range(1, STEPS + 1):
        means[k] = contraction * means[k - 1]
        variances[k] = contraction**2 * variances[k - 1] + noise_variance
        samples[k] = contraction * samples[k - 1] + np.sqrt(noise_variance) * rng.normal(size=PARTICLES)

    # Check the recurrences against their closed forms and integrate the
    # densities independently on a wide grid.
    powers = contraction ** np.arange(STEPS + 1)
    stationary_variance = PRIOR_STD**2 / (1 - DELTA / 4)
    np.testing.assert_allclose(means, powers[:, None] * INITIAL_MEANS)
    np.testing.assert_allclose(
        variances,
        powers**2 * INITIAL_STD**2 + (1 - powers**2) * stationary_variance,
    )
    grid = np.linspace(-12, 12, 24001)
    for k in (0, 60, 120, STEPS):
        density = sum(normal_pdf(grid, mean, variances[k]) for mean in means[k]) / 2
        assert abs(np.trapezoid(density, grid) - 1) < 1e-10
    prior = normal_pdf(grid, 0, PRIOR_STD**2)
    final_kl = float(np.trapezoid(density * np.log(density / prior), grid))
    assert final_kl < 1e-4
    print(f"Final means: {means[-1]}; component variance: {variances[-1]:.6f}")
    print(f"KL(final marginal || Gaussian prior): {final_kl:.8f}")
    return means, variances, samples


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16,
                         "text.color": INK, "axes.labelcolor": INK,
                         "pdf.fonttype": 42})
    means, variances, samples = simulate()
    steps_shown = [0] * 8 + np.rint(STEPS * np.linspace(0, 1, 60)**2).astype(int).tolist() + [STEPS] * 12
    x = np.linspace(-6, 6, 1801)
    prior = normal_pdf(x, 0, PRIOR_STD**2)
    fig = plt.figure(figsize=(6.4, 4.6), dpi=180)
    ax = fig.add_axes([0.13, 0.34, 0.83, 0.49])
    dots_ax = fig.add_axes([0.13, 0.22, 0.83, 0.065], sharex=ax)
    progress_ax = fig.add_axes([0.16, 0.065, 0.77, 0.025])
    current_line, = ax.plot([], [], color=BLUE, linewidth=2.8, label=r"Density $p_k$")
    ax.plot(x, prior, color=INK, linestyle="--", linewidth=2, label=r"Prior $\mathrm{N}(0,1)$")
    ax.set(xlim=(-6, 6), ylim=(0, 0.46), ylabel="Density", yticks=[0, 0.2, 0.4])
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.tick_params(labelbottom=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              frameon=False, borderaxespad=0, fontsize=14, handlelength=1.7)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(SLATE)
    ax.tick_params(colors=SLATE, length=3)
    dots_ax.set(ylim=(-1, 1), yticks=[], xlabel=r"$x$")
    dots_ax.spines[["top", "left", "right"]].set_visible(False)
    dots_ax.spines["bottom"].set_color(SLATE)
    dots_ax.tick_params(colors=SLATE, length=3)
    dots_ax.text(-0.02, 0.5, "Samples", transform=dots_ax.transAxes,
                 ha="right", va="center", fontsize=10, color=SLATE)
    dots_y = np.random.default_rng(SEED + 1).uniform(-0.7, 0.7, PARTICLES)
    dots = dots_ax.scatter(samples[0], dots_y, s=9, color=BLUE, alpha=0.5, linewidths=0)
    progress_ax.set(xlim=(-10, STEPS + 10), ylim=(0, 1))
    progress_ax.axis("off")
    progress_ax.plot([0, STEPS], [0.5, 0.5], color="#DCE5EF", linewidth=4)
    progress_line, = progress_ax.plot([], [], color=BLUE, linewidth=4)
    progress_dot, = progress_ax.plot([], [], "o", color=BLUE, markersize=7)
    progress_ax.text(0, -1.1, "Bimodal target", ha="left", color=TRAIL, fontsize=12)
    progress_ax.text(STEPS, -1.1, "Gaussian prior", ha="right", color=INK, fontsize=12)
    title = fig.text(0.55, 0.965, "", ha="center", va="top", fontsize=17, weight="bold")

    frames = []
    fill = None
    for index, k in enumerate(steps_shown):
        density = sum(normal_pdf(x, mean, variances[k]) for mean in means[k]) / 2
        current_line.set_data(x, density)
        if fill is not None:
            fill.remove()
        fill = ax.fill_between(x, density, color=BLUE, alpha=0.13)
        dots.set_offsets(np.column_stack((samples[k], dots_y)))
        progress_line.set_data([0, k], [0.5, 0.5])
        progress_dot.set_data([k], [0.5])
        title.set_text(f"Forward noising    k = {k} / {STEPS}")
        fig.canvas.draw()
        frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        frame.save(OUTPUT / f"frame-{index}.png", optimize=True)
        frames.append(frame)
    plt.close(fig)
    frames[0].save(ROOT / "assets" / "vp_diffusion_1d.gif", save_all=True,
                   append_images=frames[1:], duration=1000 // FPS, loop=0,
                   disposal=2, optimize=False)
    metadata = {
        "source": "https://arxiv.org/html/2512.02019v3#S2.SS3",
        "update": "X[k] = (1-delta[k-1]/2)*X[k-1] + nu*sqrt(delta[k-1])*epsilon[k-1]",
        "steps": STEPS, "delta": DELTA, "nu": PRIOR_STD,
        "initial_means": INITIAL_MEANS.tolist(), "initial_std": INITIAL_STD,
        "seed": SEED, "fps": FPS, "frame_steps": steps_shown,
        "stationary_variance": PRIOR_STD**2 / (1 - DELTA / 4),
        "note": "Exact mixture marginals of the discrete update; Gaussian prior is approximate at finite step size.",
    }
    (ROOT / "assets" / "vp_diffusion_1d.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(frames)} PNG frames and assets/vp_diffusion_1d.gif")


if __name__ == "__main__":
    main()
