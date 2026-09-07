# Slides

This directory contains the Beamer source for **From Variational Inference to
Diffusion-Based Reinforcement Learning**.

The separate nine-slide DA-MDP research introduction is `intro.pdf`, built from
`intro.tex`. Its approximately 5-minute speaking script is in
`intro_notes.md`; present it immediately before `main.pdf`. The introduction
has no Beamer reveal overlays, so each slide is exactly one PDF page. Its robot
videos are embedded as autoplaying, looping frame animations.

The opening slide shows animated humanoid and StackCube research previews. The
introduction also includes two silent real-world humanoid excerpts
from `IMG_4622.MOV`: 1:40--1:52 and 0:10--0:20. PDF-compatible JPEG frames are
in `assets/humanoid_motion_a/` and `assets/humanoid_motion_b/`, respectively.
The improved silent simulation video is `assets/humanoid_sim_gallery.mp4`: it
replays nine successful pickup motions sampled from the fine-tuned
`checkpoint_step_0000364.pt` policy together in one MuJoCo scene. Its exact
render settings are recorded in `assets/humanoid_sim_gallery.json`, and the
PDF-compatible animation frames are under `assets/humanoid_sim_gallery/`.
The humanoid comparison slide places data-only training on the left, this
fine-tuned gallery in the middle, and the two real-world excerpts playing
consecutively on the right. `assets/humanoid_sim_pretrained_gallery/` contains
five successful pretrained rollouts and four selected recorded failure cases,
rendered with the same camera. After each recording ends at its failure
threshold, a passive MuJoCo continuation lets the robot and box fall and
settle. Actuation is disabled for this illustrative continuation; it does not
represent further execution of the learned tracker. The selection emphasizes
failure examples, rather than estimating a success rate. The middle gallery
retains its original nine successful pickups.
The baseline video and provenance are stored alongside its frame directory
as `.mp4` and `.json` files. To regenerate them using the existing
`humanoid_diff` environment and archived trajectories, run
`scripts/render_humanoid_pretrained_gallery.py --humanoid-root /path/to/humanoid_diff`.

## Build

From this directory, run:

```bash
make
```

or invoke Tectonic directly:

```bash
tectonic --keep-logs --synctex main.tex
```

The resulting deck is `main.pdf`. The checked-in images under `assets/`
include the robotics-motivation, Lesson 1, and Lesson 3 PDF-compatible
animation frames, so the TeX source compiles without executing the notebooks
first. The StackCube overlay is also available as
`assets/stackcube_symmetric_overlay.gif`. The merged variational-inference
lesson uses `assets/l2_training.gif`, derived from
`L2-MaxEntMin/maximum_entropy_reward_training.gif`; the diffusion-sampler GIF
is `L4-DiffusionSamplers/diffusion_sampler_training.gif`.
The Multimodal Agent benchmark includes a REPPO-only animation, an analytic
optimal-policy reference that samples both reward-maximizing turns, and a
synchronized REPPO-versus-DA:REPPO comparison. Their GIFs and PDF-compatible
frame sequences are stored under `assets/multimodal_agent_*`.
The forward-noising and learned-reverse transport illustration includes the
GMM-40 cost landscape with samples overlaid on its low-cost basins. It is
stored as `assets/diffusion_transport.gif`; its PDF-compatible frames are in
`assets/diffusion_transport/` and can be regenerated with
`scripts/generate_diffusion_transport.py`.

The TUM and ATARI Lab marks used in the deck are stored under `assets/logos/`.
They were obtained from TUM's official logo portal and the official ATARI Lab
website, respectively.

Content slides are staged with Beamer overlays: advance once to reveal the next
bullet, equation, code alternative, or conclusion. The opening title and lesson
divider frames remain static. The deck contains 48 teaching frames and 132 PDF
pages because each reveal state is stored as a PDF page. The embedded StackCube
and training animations continue to play independently.

Lessons 1 and 3 use the reference GMM-40 geometry from Midgley et al.
(2023), with PyTorch seed 0, component means in `[-40, 40]^2`, and covariance
`I`. Their contour-and-sample styling follows arXiv:2502.06685. Lesson 3's
four-panel GIF compares reparameterization and log-derivative training, each
with and without Langevin preconditioning. Every landscape animation includes
the corresponding loss history and a vertical marker for the current frame.

## Build from the SummerSchool VS Code workspace

Open `/home/sebastian/code/SummerSchool` as the VS Code folder and install the
recommended **LaTeX Workshop** extension. The workspace settings select the
bundled Tectonic recipe automatically, so opening `Slides/main.tex` and saving
it rebuilds `Slides/main.pdf`.

You can also run **Terminal → Run Build Task** (or `Ctrl+Shift+B`) and select
`Build SummerSchool slides`. This task runs from the workspace root and does
not require `latexmk` or a full TeX Live installation.

The talk follows four lessons:

1. variational inference and its zero-temperature reward-maximization limit,
   using `L2-MaxEntMin/` and `L1-CostMin/`;
2. reinforcement learning as variational inference over trajectories, including
   the TRPO trust-region motivation for REPPO;
3. variance-preserving diffusion samplers, the data-processing inequality,
   and a path-space KL upper bound in `L4-DiffusionSamplers/`; and
4. diffusion-augmented MDPs following arXiv:2512.02019.
