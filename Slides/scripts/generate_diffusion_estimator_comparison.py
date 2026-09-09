"""Render the fixed-temperature, 26,000-update comparison used in Lesson 2.

Run without arguments to use the checked-in plotting data. To refresh that
data from the completed experiments, pass --checkpoints /path/to/checkpoints.
The exercise's shorter GIF remains separate from this longer slide animation.
"""
from __future__ import annotations

# Animation settings (training settings are recorded in the companion JSON).
FPS = 4                    # Frames per second in the GIF and PDF.
DPI = 100                  # Export resolution for the four-panel comparison.
UNIFORM_FRAMES = 65         # Evenly spaced frames across the full training run.
EARLY_FRAMES = 10           # Additional frames to show early optimization.
FINAL_HOLD_SECONDS = 2      # Pause at the final result before looping.

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "Slides" / "assets"
DATA = ASSETS / "diffusion_estimator_training.npz"
METADATA = DATA.with_suffix(".json")
GIF = DATA.with_suffix(".gif")
FRAMES = ASSETS / "l4_training"
ORDER = [(estimator, langevin) for langevin in (False, True)
         for estimator in ("reparameterization", "log_derivative")]
sys.path.insert(0, str(ROOT / "L2-DiffusionSamplers"))


def cache_checkpoints(directory: Path) -> None:
    import torch
    arrays, sources = {}, []
    reference_steps = reference_settings = None
    for index, (estimator, langevin) in enumerate(ORDER):
        name = f"{estimator}_langevin{int(langevin)}_fixed.pt"
        source = directory / name
        record = torch.load(source, map_location="cpu", weights_only=True)
        history, settings = record["history"], record["settings"]
        assert record["step"] == history["step"][-1] == 26_000
        assert all(t == 1.0 for t in history["temperature"])
        if reference_steps is None:
            reference_steps, reference_settings = history["step"], settings
        assert history["step"] == reference_steps
        assert settings == reference_settings
        arrays[f"samples_{index}"] = torch.stack(history["samples"]).numpy()
        arrays[f"loss_{index}"] = np.asarray(history["target_one_loss"])
        sources.append({"checkpoint": name,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "estimator": estimator, "langevin": langevin,
                        "final_loss": history["target_one_loss"][-1],
                        "recovered_modes": history["recovered_modes"][-1]})
    arrays["steps"] = np.asarray(reference_steps)
    assert all(np.isfinite(a).all() for a in arrays.values())
    np.savez_compressed(DATA, **arrays)
    # Retain training provenance, excluding the historical annealing treatment.
    settings = {k: v for k, v in reference_settings.items()
                if k not in {"INITIAL_TEMPERATURE", "TEMPERATURE_ANNEAL_STEPS"}}
    settings["TRAINING_STEPS"] = 26_000
    settings["LOG_EVERY"] = 200
    METADATA.write_text(json.dumps({
        "temperature": 1.0, "temperature_annealing": False,
        "training_settings": settings,
        "learning_rate": {"initial": 5e-4, "decay_start": 20_000,
                          "decay_end": 24_000, "final": 2e-5, "decay": "cosine"},
        "evaluation_paths": 5000, "sources": sources,
    }, indent=2) + "\n")


def render() -> None:
    import lesson2_common as common
    metadata = json.loads(METADATA.read_text())
    settings = metadata["training_settings"]
    common.configure_target(
        num_modes=settings["NUM_MODES"], mode_bound=settings["MODE_BOUND"],
        target_variance=settings["TARGET_VARIANCE"], seed=settings["TARGET_SEED"],
        plot_bound=settings["PLOT_BOUND"], grid_resolution=settings["GRID_RESOLUTION"],
        contour_levels=settings["CONTOUR_LEVELS"], reward_plot_range=settings["REWARD_PLOT_RANGE"],
    )
    with np.load(DATA) as data:
        steps = data["steps"].tolist()
        experiments = [(estimator, langevin, {
            "step": steps, "samples": data[f"samples_{index}"],
            "evaluation_loss": data[f"loss_{index}"].tolist(),
            "temperature": [metadata["temperature"]] * len(steps),
        }) for index, (estimator, langevin) in enumerate(ORDER)]
    selected = sorted(set(list(range(min(EARLY_FRAMES, len(steps))))
                          + np.linspace(0, len(steps) - 1, UNIFORM_FRAMES, dtype=int).tolist()))
    selected += [selected[-1]] * (FINAL_HOLD_SECONDS * FPS - 1)
    common.save_training_comparison_animation(
        experiments, GIF, fps=FPS, dpi=DPI, frame_indices=selected,
    )
    FRAMES.mkdir(exist_ok=True)
    count = 0
    with Image.open(GIF) as movie:
        for i in range(movie.n_frames):
            movie.seek(i)
            # GIF merges identical held frames; expand them for fixed-rate PDF playback.
            repeats = max(1, round(movie.info["duration"] * FPS / 1000))
            for _ in range(repeats):
                movie.convert("RGB").save(FRAMES / f"frame-{count}.png")
                count += 1
    for old in FRAMES.glob("frame-*.png"):
        if int(old.stem.split("-")[-1]) >= count:
            old.unlink()
    metadata["animation"] = {"fps": FPS, "frame_count": count,
                             "displayed_updates": [steps[i] for i in selected]}
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n")
    DATA.with_suffix(".tex").write_text(
        "% Generated by scripts/generate_diffusion_estimator_comparison.py\n"
        "\\animategraphics[autoplay,loop,poster=last,height=0.68\\textheight]\n"
        f"  {{{FPS}}}{{assets/l4_training/frame-}}{{0}}{{{count - 1}}}\n"
    )
    print(f"Saved {GIF}; PDF frames 0–{count - 1}, {FPS} fps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path)
    args = parser.parse_args()
    if args.checkpoints is not None:
        cache_checkpoints(args.checkpoints)
    render()
