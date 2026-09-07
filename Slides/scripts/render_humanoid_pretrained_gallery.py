#!/usr/bin/env python3
"""Render selected data-only rollouts, with passive physics after failure.

Run with the humanoid_diff environment. Four archived failure examples are
included alongside five successful pickups. Recorded motion is unchanged;
after a failure the robot and box settle under gravity with actuation disabled.
This continuation is illustrative, not a continuation of the learned policy.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image


SOURCE_FPS = 50
VIDEO_FPS = 5
VIDEO_FRAMES = 60

# Keep the camera/grid while choosing four distinct recorded failure examples.
FAILURE_ENTRIES = {
    0: "sub10_largebox_086_original_sbto-v2-top-all:v0_b002",
    3: "sub3_largebox_030_original_sbto-v2-top-all:v0_b000",
    5: "sub16_largebox_046_original_sbto-v2-top-all:v0_b000",
    8: "sub7_largebox_047_original_sbto-v2-top-all:v0_b025",
}


def continue_failure(model, recorded, length):
    """Preserve the recording and simulate the unrecorded tail at 50 Hz."""
    state = mujoco.MjData(model)
    state.qpos[:] = recorded[-1]
    mujoco.mj_differentiatePos(
        model, state.qvel, 1 / SOURCE_FPS, recorded[-2], recorded[-1]
    )
    mujoco.mj_forward(model, state)
    output = np.empty((length, model.nq), dtype=recorded.dtype)
    output[:len(recorded)] = recorded
    steps = round(1 / SOURCE_FPS / model.opt.timestep)
    for index in range(len(recorded), length):
        mujoco.mj_step(model, state, nstep=steps)
        if not np.isfinite(state.qpos).all() or not np.isfinite(state.qvel).all():
            raise ValueError("Non-finite state in passive continuation")
        if any(warning.number for warning in state.warning):
            raise ValueError("MuJoCo warning in passive continuation")
        output[index] = state.qpos
    assert np.array_equal(output[:len(recorded)], recorded)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--humanoid-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.humanoid_root.resolve()
    assets = Path(__file__).resolve().parents[1] / "assets"
    gallery = json.loads((assets / "humanoid_sim_gallery.json").read_text())
    results = root / "runs/sanokows_new_all_motion_tracker_eval_5030886_latest_ema_visualized"
    wanted = list(gallery["track_labels"])
    for index, entry in FAILURE_ENTRIES.items():
        wanted[index] = entry
    rows = {}
    with (results / "per_rollout.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["entry_name"] in wanted and row["repeat"] == 0:
                rows[row["entry_name"]] = row
    selected = [rows[name] for name in wanted]
    assert sum(not row["completion_success"] for row in selected) == 4
    tracks = []
    for row in selected:
        with np.load(results / row["trajectory_shard"], allow_pickle=False) as data:
            index = row["trajectory_row"]
            track = np.concatenate([
                data["simulation__" + field][index]
                for field in ("root_pos", "root_rot", "dof_pos", "object_pos", "object_rot")
            ], axis=1)
        # Evaluator discards every pose at/after failure_frame. Do not bridge
        # interior gaps or accidentally replay any padded/reset states.
        end = row["failure_frame"] if not row["completion_success"] else row["num_frames"]
        track = track[:end]
        if len(track) < 2 or not np.isfinite(track).all():
            raise ValueError(f"Invalid recording: {row['entry_name']}")
        tracks.append(track)

    xml = str(root / "assets/robots/g1/scene_g1_box.xml")
    model = mujoco.MjModel.from_xml_path(xml)
    passive_model = mujoco.MjModel.from_xml_path(xml)
    passive_model.opt.timestep = 0.002
    passive_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    # The tracker scene uses a restricted set of explicit contact pairs.
    # Let every physical body collider touch the floor during a full fall.
    for index in range(passive_model.ngeom):
        if passive_model.geom_group[index] == 3:
            passive_model.geom_conaffinity[index] |= 1
    continuations = []
    length = VIDEO_FRAMES * SOURCE_FPS // VIDEO_FPS
    for index, (row, track) in enumerate(zip(selected, tracks)):
        if row["completion_success"]:
            continue
        extended = continue_failure(passive_model, track, length)
        continuations.append({
            "entry_name": row["entry_name"],
            "recorded_frames": len(track),
            "continuation_frames": len(extended) - len(track),
            "initial_root_height_m": float(track[-1, 2]),
            "final_root_height_m": float(extended[-1, 2]),
            "final_box_height_m": float(extended[-1, 38]),
        })
        tracks[index] = extended
        print(f"Continued {row['entry_name']}: root height "
              f"{track[-1, 2]:.2f} -> {extended[-1, 2]:.2f} m", flush=True)
    data = mujoco.MjData(model)
    extra = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=1280, max_geom=20000)
    camera = mujoco.MjvCamera()
    settings = gallery["render"]["camera"]
    camera.lookat[:] = [0, 0, settings["lookat_height"]]
    camera.distance = settings["distance"]
    camera.azimuth = settings["azimuth"]
    camera.elevation = settings["elevation"]
    option, perturb = mujoco.MjvOption(), mujoco.MjvPerturb()
    origins = np.array([
        [(i // 3 - 1) * 2.5, (i % 3 - 1) * 2.5, 0] for i in range(9)
    ])

    output = assets / "humanoid_sim_pretrained_gallery"
    output.mkdir(exist_ok=True)
    video = imageio_ffmpeg.write_frames(
        str(output.with_suffix(".mp4")), (1280, 720), fps=VIDEO_FPS,
        codec="libx264", quality=8, macro_block_size=1,
    )
    video.send(None)
    try:
        for frame in range(VIDEO_FRAMES):
            for index, track in enumerate(tracks):
                pose = track[min(frame * SOURCE_FPS // VIDEO_FPS, len(track) - 1)].copy()
                pose[0:3] += origins[index]
                pose[36:39] += origins[index]
                current = data if index == 0 else extra
                current.qpos[:] = pose
                mujoco.mj_forward(model, current)
                if index == 0:
                    renderer.update_scene(current, camera)
                else:
                    mujoco.mjv_addGeoms(
                        model, current, option, perturb,
                        mujoco.mjtCatBit.mjCAT_DYNAMIC, renderer.scene,
                    )
            pixels = renderer.render().copy()
            Image.fromarray(pixels).save(output / f"frame-{frame}.jpg", quality=90)
            video.send(pixels)
            if (frame + 1) % 10 == 0:
                print(f"Rendered {frame + 1}/{VIDEO_FRAMES} frames", flush=True)
    finally:
        video.close()
        renderer.close()

    manifest = {
        "video": str(output.relative_to(assets.parent)) + ".mp4",
        "source_results": str(results),
        "selection": "repeat 0, four selected recorded failures and five successful pickups; not an aggregate evaluation",
        "checkpoint": selected[0]["checkpoint"],
        "checkpoint_sha256": selected[0]["checkpoint_sha256"],
        "camera": settings,
        "spacing": 2.5,
        "fps": VIDEO_FPS,
        "frames": VIDEO_FRAMES,
        "post_failure": {
            "method": "passive MuJoCo continuation from the last recorded pose",
            "velocity": "finite difference of the final two recorded poses at 50 Hz",
            "actuation": "disabled; learned tracker is not continued",
            "contacts": "original scene pairs plus floor contact for every group-3 body collider",
            "timestep_s": float(passive_model.opt.timestep),
            "tracks": continuations,
        },
        "completion_successes": sum(row["completion_success"] for row in selected),
        "num_tracks": 9,
        "track_results": [{key: row[key] for key in (
            "entry_name", "completion_success", "failure_reason", "failure_frame",
            "rollout_seed", "trajectory_shard", "trajectory_row",
        )} for row in selected],
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {output}: {manifest['completion_successes']}/9 pickups completed")


if __name__ == "__main__":
    main()
