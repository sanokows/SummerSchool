#!/usr/bin/env python3
"""Render the first Lesson 2 exercises using their actual solution kernels.

Requires the local instructor solution notebook. Only its two sampling function
definitions are loaded; no notebook training or other cells are executed.
"""

from __future__ import annotations

# Playback settings; the simulation settings come from the notebook/checkpoint.
FPS = 10
MOVING_FRAMES = 60
INITIAL_HOLD = 8
FINAL_HOLD = 12
DPI = 140

import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
LESSON = ROOT / "L2-DiffusionSamplers"
ASSETS = ROOT / "Slides" / "assets"
OUTPUT = ASSETS / "kernel_preview"
sys.path.insert(0, str(LESSON))
from lesson2_kernel_demo import DemoConfig, load_pretrained_demo, simulate_forward, simulate_reverse

INK = "#17243A"
SLATE = "#64748B"
BLUE = "#1E90FF"


def notebook_code(path):
    notebook = json.loads(path.read_text())
    return [ast.parse("".join(cell["source"])) for cell in notebook["cells"]
            if cell["cell_type"] == "code"]


def load_exercise():
    settings = {}
    for cell in notebook_code(LESSON / "diffusion_sampler_student.ipynb"):
        for node in cell.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id.startswith("DEMO_")):
                settings[node.targets[0].id] = ast.literal_eval(node.value)
    config = DemoConfig(**{
        field: settings["DEMO_" + name] for field, name in {
            "prior_std": "PRIOR_STD", "num_steps": "DIFFUSION_STEPS",
            "schedule": "SCHEDULE", "delta_start": "DELTA_START",
            "delta_end": "DELTA_END", "mode_location": "MODE_LOCATION",
            "component_std": "COMPONENT_STD",
        }.items()
    })
    checkpoint = LESSON / settings["DEMO_CHECKPOINT"]
    pretrained = load_pretrained_demo(checkpoint)
    assert config == pretrained.config, "The preview must match both notebook presets."
    namespace = {"torch": torch, "PRIOR_STD": config.prior_std}
    source = {}
    for cell in notebook_code(LESSON / "diffusion_sampler_solution.ipynb"):
        for node in cell.body:
            if isinstance(node, ast.FunctionDef) and node.name in {
                "forward_sde_step", "reverse_sde_step",
            }:
                source[node.name] = ast.unparse(node)
                exec(compile(ast.Module(body=[node], type_ignores=[]),
                             "diffusion_sampler_solution.ipynb", "exec"), namespace)
    kwargs = {"num_samples": settings["DEMO_PARTICLES"], "seed": settings["DEMO_SEED"]}
    forward = simulate_forward(namespace["forward_sde_step"], config, **kwargs)
    reverse = simulate_reverse(namespace["reverse_sde_step"], pretrained, **kwargs)
    return forward, reverse, {
        "diffusion": asdict(config), **kwargs, "kernel_source": source,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "note": "Histograms use the notebook kernels and pretrained neural score; dashed curves are destination densities.",
    }


def normal_pdf(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std)**2) / (np.sqrt(2 * np.pi) * std)


def main():
    torch.set_num_threads(2)
    forward, reverse, metadata = load_exercise()
    config = forward.config
    forward_final = forward.states[-1, :, 0].numpy()
    reverse_final = reverse.states[0, :, 0].numpy()
    left, right = reverse_final[reverse_final < 0], reverse_final[reverse_final >= 0]
    metrics = {
        "forward_final_mean": float(forward_final.mean()),
        "forward_final_std": float(forward_final.std()),
        "reverse_left_mass": float(len(left) / len(reverse_final)),
        "reverse_mode_means": [float(left.mean()), float(right.mean())],
        "reverse_mode_stds": [float(left.std()), float(right.std())],
    }
    assert abs(metrics["forward_final_mean"]) < 0.06 * config.prior_std
    assert abs(metrics["forward_final_std"] / config.prior_std - 1) < 0.06
    assert abs(metrics["reverse_left_mass"] - 0.5) < 0.04
    np.testing.assert_allclose(metrics["reverse_mode_means"],
                               [-config.mode_location, config.mode_location], atol=0.08)
    np.testing.assert_allclose(metrics["reverse_mode_stds"], config.component_std, atol=0.06)
    metadata["endpoint_checks"] = metrics
    print(json.dumps(metrics, indent=2), flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15,
                         "text.color": INK, "axes.labelcolor": INK})
    # Leave room for the code above the plots on the assignment slide.
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 2.5), dpi=DPI)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.23, top=0.73, wspace=0.22)
    bound = max(config.mode_location + 4 * config.component_std, 4.5 * config.prior_std)
    edges = np.linspace(-bound, bound, 71)
    grid = np.linspace(-bound, bound, 1001)
    prior = normal_pdf(grid, 0, config.prior_std)
    target = (normal_pdf(grid, -config.mode_location, config.component_std)
              + normal_pdf(grid, config.mode_location, config.component_std)) / 2
    histograms, labels = [], []
    ymax = max(prior.max(), target.max()) * 1.28
    for axis, destination in zip(axes, (prior, target)):
        histograms.append(axis.stairs(np.zeros(len(edges) - 1), edges,
                                      fill=True, color=BLUE, alpha=0.7, linewidth=0))
        axis.plot(grid, destination, color=INK, ls="--", lw=2)
        axis.set(xlim=(-bound, bound), ylim=(0, ymax), xlabel=r"$x$",
                 xticks=[-4, -2, 0, 2, 4], yticks=[0, 0.2, 0.4])
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(SLATE)
        axis.tick_params(colors=SLATE, length=3)
        labels.append(axis.text(0.5, 1.07, "", transform=axis.transAxes,
                                ha="center", fontsize=15))
    axes[0].set_ylabel("Density")
    fig.legend(handles=[Patch(color=BLUE, alpha=0.7, label="Kernel samples"),
                        Line2D([], [], color=INK, ls="--", lw=2, label="Destination density")],
               loc="upper center", bbox_to_anchor=(0.52, 1.025), ncol=2,
               frameon=False, fontsize=14)
    moving = np.rint(config.num_steps * np.linspace(0, 1, MOVING_FRAMES)**2).astype(int).tolist()
    pairs = ([(0, config.num_steps)] * INITIAL_HOLD
             + list(zip(moving, reversed(moving)))
             + [(config.num_steps, 0)] * FINAL_HOLD)
    frames = []
    for index, steps in enumerate(pairs):
        for panel, (path, k) in enumerate(zip((forward, reverse), steps)):
            samples = path.states[k, :, 0].numpy()
            histogram = np.histogram(samples, bins=edges)[0] / (len(samples) * np.diff(edges))
            histograms[panel].set_data(histogram)
            labels[panel].set_text(f"k = {k} / {config.num_steps}")
        fig.canvas.draw()
        frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        frame.save(OUTPUT / f"frame-{index}.png", optimize=True)
        frames.append(frame)
    plt.close(fig)
    frames[0].save(ASSETS / "kernel_preview.gif", save_all=True,
                   append_images=frames[1:], duration=1000 // FPS, loop=0,
                   disposal=2, optimize=False)
    metadata["animation"] = {"fps": FPS, "frame_count": len(frames), "frame_step_pairs": pairs}
    (ASSETS / "kernel_preview.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (ASSETS / "kernel_preview.tex").write_text(
        "% Generated by scripts/generate_kernel_preview.py\n"
        "\\animategraphics[autoplay,loop,poster=first,width=\\linewidth]"
        f"{{{FPS}}}{{assets/kernel_preview/frame-}}{{0}}{{{len(frames) - 1}}}\n"
    )
    print(f"Wrote {len(frames)} frames and assets/kernel_preview.gif", flush=True)


if __name__ == "__main__":
    main()
