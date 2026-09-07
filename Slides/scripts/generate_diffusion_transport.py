#!/usr/bin/env python3
"""Render the forward/reverse diffusion transport animation used in the slides."""

from __future__ import annotations

import math
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1600, 500
SCALE = 2
FRAME_COUNT = 48
FPS = 4

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "assets" / "diffusion_transport"
GIF_PATH = ROOT / "assets" / "diffusion_transport.gif"

WHITE = (255, 255, 255, 255)
PAPER = (248, 251, 255, 255)
INK = (23, 36, 58, 255)
SLATE = (100, 116, 139, 255)
BLUE = (30, 144, 255, 255)
BLUE_DARK = (17, 92, 181, 255)
BLUE_PALE = (30, 144, 255, 36)
ORANGE = (216, 154, 67, 255)
ORANGE_DARK = (158, 100, 36, 255)
TERRAIN_LIGHT = (220, 232, 229, 255)
TERRAIN_DARK = (118, 158, 154, 255)
GREEN_DARK = (74, 112, 108, 255)

LEFT_CENTER = (245, 260)
RIGHT_CENTER = (1355, 260)
TARGET_DATA_BOX = (1215, 120, 1495, 400)


def rgba(color: tuple[int, int, int, int], alpha: int) -> tuple[int, int, int, int]:
    return color[:3] + (alpha,)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size * SCALE)


def rounded_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    radius: float,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int] | None = None,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(
        tuple(int(v * SCALE) for v in box),
        radius=int(radius * SCALE),
        fill=fill,
        outline=outline,
        width=width * SCALE,
    )


def ellipse(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    fill: tuple[int, int, int, int] | None = None,
    outline: tuple[int, int, int, int] | None = None,
    width: int = 1,
) -> None:
    draw.ellipse(
        tuple(int(v * SCALE) for v in box),
        fill=fill,
        outline=outline,
        width=width * SCALE,
    )


def line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    draw.line(
        [(int(x * SCALE), int(y * SCALE)) for x, y in points],
        fill=fill,
        width=width * SCALE,
        joint="curve",
    )


def text_center(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    value: str,
    text_font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int],
) -> None:
    box = draw.textbbox((0, 0), value, font=text_font)
    draw.text(
        (
            int(xy[0] * SCALE - (box[2] - box[0]) / 2),
            int(xy[1] * SCALE - (box[3] - box[1]) / 2),
        ),
        value,
        font=text_font,
        fill=fill,
    )


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int, int],
    width: int,
) -> None:
    line(draw, [start, end], color, width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head = 14 + width
    wing = 0.55
    p1 = (
        end[0] - head * math.cos(angle - wing),
        end[1] - head * math.sin(angle - wing),
    )
    p2 = (
        end[0] - head * math.cos(angle + wing),
        end[1] - head * math.sin(angle + wing),
    )
    draw.polygon(
        [
            (int(end[0] * SCALE), int(end[1] * SCALE)),
            (int(p1[0] * SCALE), int(p1[1] * SCALE)),
            (int(p2[0] * SCALE), int(p2[1] * SCALE)),
        ],
        fill=color,
    )


def ease(value: float) -> float:
    """Cosine easing keeps both endpoints readable for a few frames."""
    return 0.5 - 0.5 * math.cos(math.pi * value)


def timeline(frame_index: int) -> tuple[float, str]:
    """Return noising amount in [0, 1] and the active transport direction."""
    if frame_index <= 2:
        return 0.0, "forward"
    if frame_index <= 20:
        return ease((frame_index - 3) / 17), "forward"
    if frame_index <= 23:
        return 1.0, "reverse"
    if frame_index <= 41:
        return 1.0 - ease((frame_index - 24) / 17), "reverse"
    return 0.0, "forward"


def build_geometry() -> tuple[
    list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]]
]:
    generator = torch.Generator().manual_seed(0)
    means_tensor = torch.empty((40, 2)).uniform_(-40.0, 40.0, generator=generator)
    means = [(float(x), float(y)) for x, y in means_tensor]

    rng = random.Random(11)
    target_samples: list[tuple[float, float]] = []
    prior_samples: list[tuple[float, float]] = []
    for index in range(320):
        mx, my = means[index % len(means)]
        target_samples.append((mx + rng.gauss(0.0, 1.0), my + rng.gauss(0.0, 1.0)))
        prior_samples.append((rng.gauss(0.0, 22.0), rng.gauss(0.0, 22.0)))
    return means, target_samples, prior_samples


def build_cost_landscape(means: list[tuple[float, float]]) -> Image.Image:
    """Rasterize the relative negative log-density of the unit-variance GMM."""
    left, top, right, bottom = TARGET_DATA_BOX
    width = int((right - left) * SCALE)
    height = int((bottom - top) * SCALE)
    xs = (torch.arange(width, dtype=torch.float32) + 0.5) / SCALE + left
    ys = (torch.arange(height, dtype=torch.float32) + 0.5) / SCALE + top
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack(
        (
            (grid_x - RIGHT_CENTER[0]) / 3.0,
            -(grid_y - RIGHT_CENTER[1]) / 3.0,
        ),
        dim=-1,
    )
    mean_tensor = torch.tensor(means, dtype=torch.float32)
    squared_distance = (points[..., None, :] - mean_tensor).square().sum(dim=-1)
    log_density = torch.logsumexp(-0.5 * squared_distance, dim=-1)
    relative_cost = -log_density
    relative_cost -= relative_cost.min()

    # Values beyond 18 nats are visually equivalent high-cost regions. The
    # clipping keeps every unit-variance basin legible on a projected slide.
    normalized = relative_cost.clamp(0.0, 18.0) / 18.0
    low_cost = torch.tensor(TERRAIN_DARK[:3], dtype=torch.float32)
    high_cost = torch.tensor(TERRAIN_LIGHT[:3], dtype=torch.float32)
    rgb = low_cost * (1.0 - normalized[..., None]) + high_cost * normalized[..., None]
    alpha = torch.full((*rgb.shape[:2], 1), 255.0)
    rgba_pixels = torch.cat((rgb, alpha), dim=-1).to(torch.uint8).numpy()
    return Image.fromarray(rgba_pixels, mode="RGBA")


def draw_prior(draw: ImageDraw.ImageDraw) -> None:
    cx, cy = LEFT_CENTER
    for radius, alpha in [(137, 13), (110, 19), (83, 27), (55, 37)]:
        ellipse(
            draw,
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=rgba(BLUE, alpha),
            outline=rgba(BLUE, min(alpha + 28, 70)),
            width=1,
        )
    line(draw, [(cx - 150, cy), (cx + 150, cy)], rgba(SLATE, 38), 1)
    line(draw, [(cx, cy - 150), (cx, cy + 150)], rgba(SLATE, 38), 1)
    ellipse(draw, (cx - 3, cy - 3, cx + 3, cy + 3), fill=rgba(BLUE_DARK, 180))


def map_target(point: tuple[float, float]) -> tuple[float, float]:
    return RIGHT_CENTER[0] + 3.0 * point[0], RIGHT_CENTER[1] - 3.0 * point[1]


def draw_target(
    canvas: Image.Image,
    means: list[tuple[float, float]],
    cost_landscape: Image.Image,
) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    cx, cy = RIGHT_CENTER
    rounded_rectangle(
        draw,
        (1188, 82, 1536, 438),
        23,
        rgba(TERRAIN_LIGHT, 75),
        rgba(GREEN_DARK, 70),
        1,
    )
    canvas.alpha_composite(
        cost_landscape,
        (int(TARGET_DATA_BOX[0] * SCALE), int(TARGET_DATA_BOX[1] * SCALE)),
    )
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle(
        tuple(int(value * SCALE) for value in TARGET_DATA_BOX),
        outline=rgba(GREEN_DARK, 75),
        width=SCALE,
    )
    line(draw, [(cx - 155, cy), (cx + 155, cy)], rgba(SLATE, 34), 1)
    line(draw, [(cx, cy - 167), (cx, cy + 167)], rgba(SLATE, 34), 1)

    # Thin contours emphasize that the colored background is a cost surface,
    # rather than another cloud of target samples.
    for mean in means:
        x, y = map_target(mean)
        ellipse(draw, (x - 12, y - 12, x + 12, y + 12), outline=rgba(GREEN_DARK, 42))
        ellipse(draw, (x - 7, y - 7, x + 7, y + 7), outline=rgba(GREEN_DARK, 68))

    text_center(
        draw,
        (1360, 102),
        "C(x)  ·  LOW-COST BASINS",
        font(14, True),
        GREEN_DARK,
    )

    # Compact high-to-low cost key. It intentionally sits outside the plotted
    # [-40, 40]^2 region so it never hides a mode or a sample.
    legend_x0, legend_x1 = 1507, 1518
    legend_y0, legend_y1 = 148, 352
    steps = 52
    for index in range(steps):
        fraction = index / (steps - 1)
        color = tuple(
            round(
                TERRAIN_LIGHT[channel] * (1.0 - fraction)
                + TERRAIN_DARK[channel] * fraction
            )
            for channel in range(3)
        ) + (255,)
        y0 = legend_y0 + (legend_y1 - legend_y0) * index / steps
        y1 = legend_y0 + (legend_y1 - legend_y0) * (index + 1) / steps
        draw.rectangle(
            (legend_x0 * SCALE, int(y0 * SCALE), legend_x1 * SCALE, int(y1 * SCALE)),
            fill=color,
        )
    text_center(draw, (1513, 136), "high", font(8, True), SLATE)
    text_center(draw, (1513, 366), "low", font(8, True), GREEN_DARK)


def draw_progress_rail(draw: ImageDraw.ImageDraw, amount: float, active: str) -> None:
    left_x, right_x, rail_y = LEFT_CENTER[0], RIGHT_CENTER[0], 260
    # Five subtle time stations make the Markov chain visually explicit.
    for index in range(6):
        x = left_x + (right_x - left_x) * index / 5
        ellipse(draw, (x - 4, rail_y - 4, x + 4, rail_y + 4), fill=rgba(SLATE, 75))
    for index in range(5):
        x0 = left_x + (right_x - left_x) * index / 5 + 10
        x1 = left_x + (right_x - left_x) * (index + 1) / 5 - 10
        line(draw, [(x0, rail_y), (x1, rail_y)], rgba(SLATE, 50), 2)

    marker_x = RIGHT_CENTER[0] + (LEFT_CENTER[0] - RIGHT_CENTER[0]) * amount
    active_color = ORANGE if active == "forward" else BLUE
    ellipse(
        draw,
        (marker_x - 14, rail_y - 14, marker_x + 14, rail_y + 14),
        fill=rgba(active_color, 35),
    )
    ellipse(
        draw,
        (marker_x - 7, rail_y - 7, marker_x + 7, rail_y + 7),
        fill=rgba(active_color, 225),
    )


def draw_particles(
    canvas: Image.Image,
    amount: float,
    active: str,
    target_samples: list[tuple[float, float]],
    prior_samples: list[tuple[float, float]],
) -> None:
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    alpha = math.cos(amount * math.pi / 2.0)
    noise_weight = math.sqrt(max(0.0, 1.0 - alpha * alpha))
    anchor_x = RIGHT_CENTER[0] + (LEFT_CENTER[0] - RIGHT_CENTER[0]) * amount
    anchor_y = RIGHT_CENTER[1]

    # A translucent envelope reads as probability mass while leaving both
    # endpoint distributions visible underneath it.
    spread_x = 140 * alpha + 100 * noise_weight
    spread_y = 140 * alpha + 100 * noise_weight
    glow = BLUE
    ellipse(
        draw,
        (
            anchor_x - spread_x,
            anchor_y - spread_y,
            anchor_x + spread_x,
            anchor_y + spread_y,
        ),
        fill=rgba(glow, 9),
        outline=rgba(glow, 62),
        width=2,
    )

    dot_color = BLUE_DARK
    for target, prior in zip(target_samples, prior_samples):
        x_offset = alpha * 3.0 * target[0] + noise_weight * 3.0 * prior[0]
        y_offset = -(alpha * 3.0 * target[1] + noise_weight * 3.0 * prior[1])
        x, y = anchor_x + x_offset, anchor_y + y_offset
        if 45 <= x <= WIDTH - 45 and 72 <= y <= HEIGHT - 66:
            ellipse(draw, (x - 3.4, y - 3.4, x + 3.4, y + 3.4), fill=rgba(WHITE, 205))
            ellipse(
                draw, (x - 2.2, y - 2.2, x + 2.2, y + 2.2), fill=rgba(dot_color, 190)
            )

    canvas.alpha_composite(layer)


def render_frame(
    frame_index: int,
    means: list[tuple[float, float]],
    target_samples: list[tuple[float, float]],
    prior_samples: list[tuple[float, float]],
    cost_landscape: Image.Image,
) -> Image.Image:
    amount, active = timeline(frame_index)
    canvas = Image.new("RGBA", (WIDTH * SCALE, HEIGHT * SCALE), PAPER)
    draw = ImageDraw.Draw(canvas, "RGBA")

    rounded_rectangle(
        draw, (12, 10, WIDTH - 12, HEIGHT - 10), 28, WHITE, rgba(BLUE, 34), 1
    )
    # Soft endpoint cards visually anchor the two distributions.
    rounded_rectangle(draw, (58, 83, 430, 437), 26, rgba(BLUE, 7), rgba(BLUE, 30), 1)
    draw_prior(draw)
    draw_target(canvas, means, cost_landscape)
    draw_progress_rail(draw, amount, active)

    forward_color = ORANGE_DARK if active == "forward" else rgba(ORANGE_DARK, 60)
    reverse_color = BLUE_DARK if active == "reverse" else rgba(BLUE_DARK, 60)
    arrow(draw, (1190, 58), (410, 58), forward_color, 5 if active == "forward" else 2)
    arrow(draw, (410, 454), (1190, 454), reverse_color, 5 if active == "reverse" else 2)

    forward_text = "FORWARD  ·  ADD NOISE"
    reverse_text = "REVERSE  ·  LEARN DENOISING"
    text_center(draw, (800, 43), forward_text, font(18, True), forward_color)
    text_center(draw, (800, 440), reverse_text, font(18, True), reverse_color)

    draw_particles(canvas, amount, active, target_samples, prior_samples)

    draw = ImageDraw.Draw(canvas, "RGBA")
    state = f"noise level  {amount:0.2f}"
    pill_color = ORANGE if active == "forward" else BLUE
    rounded_rectangle(draw, (706, 224, 894, 270), 23, rgba(pill_color, 232), None)
    text_center(draw, (800, 245), state, font(16, True), WHITE)

    # ImageDraw stores the requested alpha on RGBA canvases. Flatten onto the
    # paper color explicitly so the translucent density layers remain subtle
    # in PDF viewers as well as in the standalone PNGs.
    flattened = Image.alpha_composite(
        Image.new("RGBA", canvas.size, PAPER), canvas
    ).convert("RGB")
    return flattened.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    means, target_samples, prior_samples = build_geometry()
    cost_landscape = build_cost_landscape(means)
    frames: list[Image.Image] = []
    for index in range(FRAME_COUNT):
        frame = render_frame(
            index,
            means,
            target_samples,
            prior_samples,
            cost_landscape,
        )
        frame.save(OUTPUT_DIR / f"frame-{index}.png", optimize=True)
        frames.append(frame)
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / FPS),
        loop=0,
        disposal=2,
        optimize=False,
    )


if __name__ == "__main__":
    main()
