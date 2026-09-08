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
include the robotics-motivation, Lesson 1, and Lesson 2 PDF-compatible
animation frames, so the TeX source compiles without executing the notebooks
first. The StackCube overlay is also available as
`assets/stackcube_symmetric_overlay.gif`. Lesson 1 (variational inference)
uses `assets/l2_training.gif`, derived from
`L1-VariationalInference/maximum_entropy_reward_training.gif`; the Lesson 2
diffusion-sampler GIF is `L2-DiffusionSamplers/diffusion_sampler_training.gif`.
The Multimodal Agent benchmark includes a REPPO-only animation, an analytic
optimal-policy reference that samples both reward-maximizing turns, and a
synchronized REPPO-versus-DA:REPPO comparison. Their GIFs and PDF-compatible
frame sequences are stored under `assets/multimodal_agent_*`.
Immediately after StackCube, the main deck shows the paper's two successful
PushT rollouts side by side: clockwise (policy seed 20001) and counter-clockwise
(policy seed 20000), with the same initial block yaw of 192 degrees and
environment seed 10000. The synchronized original video is
`assets/pusht_multimodal.mp4`; provenance and rollout metadata are in
`assets/pusht_multimodal.json`. Its 117 PDF frames preserve the original 10 fps
and complete motions, holding the earlier success until both finish. The
diagnostic header/footer are cropped from the slide animation; direction
labels are supplied in TeX. Regenerate the frames from this directory using
FFmpeg (or the executable bundled with `imageio_ffmpeg`):

```bash
ffmpeg -i assets/pusht_multimodal.mp4 -vf 'crop=1024:512:0:78' \
  -q:v 2 -start_number 0 -y assets/pusht_multimodal/frame-%d.jpg
```

The research introduction's forward-noising and learned-reverse transport illustration includes the
GMM-40 cost landscape with samples overlaid on its low-cost basins. It is
stored as `assets/diffusion_transport.gif`; its PDF-compatible frames are in
`assets/diffusion_transport/` and can be regenerated with
`scripts/generate_diffusion_transport.py`.

Lesson 2 keeps the Gaussian-versus-diffusion family comparison, then introduces
discrete forward noising, reverse sampling with the optimal score and then its
learned approximation, and learning from a cost. The DPI is introduced using
a latent variable `Z`, then applied to diffusion by setting `X = X^0` and
`Z = X^{1:K}`. The forward slide places the equations from
[Section 2.3 of the paper](https://arxiv.org/html/2512.02019v3#S2.SS3) beside
`assets/vp_diffusion_1d.gif`, with PDF animation frames in
`assets/vp_diffusion_1d/`. It evolves a symmetric mixture with means -3 and 3
and component standard deviation 0.55 toward a unit-Gaussian prior using
600 discrete steps with `delta = 0.02`. The density curves are exact mixture
marginals of that discrete process, and the dots follow its Markov updates.
Finite Euler steps have stationary variance `nu^2 / (1 - delta / 4)`, so the
displayed convergence to the prior is approximate. Regenerate the animation
from the workspace root with
`.venv/bin/python Slides/scripts/generate_vp_diffusion_1d.py`; the companion
JSON records its parameters and displayed step indices.

The student notebook, solution, and slide code use those same discrete Euler
updates: forward mean `(1 - delta_{k-1}/2) * x^{k-1}`, reverse mean
`(1 + delta_k/2) * x^k + nu^2 * delta_k * u_Psi(x^k, k)`. Both kernels use
standard deviation `nu * sqrt(delta)` with their respective coefficient.
The exercise sets `Delta = 1`, so `delta_k = beta_k`, and stores `K+1`
coefficients for `K` transitions. The sampler passes the appropriate coefficient
as `delta`; reverse network indices run from `K` down to `1`. Sampling and
log-probability evaluation use identical means and variances. There is no
separate exercise discretization or continuous-time SDE section.

The sampler introduction mentions that both the prior and diffusion
coefficients can be learned. The exercise supports learning the monotone
interior coefficients; its Gaussian prior is fixed with a configurable scale.
The optional Langevin slide contains only the target-gradient identity and
the score parameterization using that gradient.

Both lessons' student and solution notebooks begin with a commented
hyperparameter panel. It controls the target geometry, model, training,
evaluation, and animation. After an edit, use **Run All**. Lesson 2 forwards
these values explicitly to its shared helpers, including the prior scale,
diffusion schedule, score width, and temperature curriculum.

Instructor checks cover the solution kernels, path indexing, gradients, and short training runs
for both estimators with fixed/learned schedules and Langevin preconditioning
on/off. They require the local `diffusion_sampler_solution.ipynb`, which is
intentionally excluded from Git; a student checkout reports an explicit skip.
Run them from the workspace root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s L2-DiffusionSamplers -v
```

Running the solution notebook executes the four default 800-update experiments
and regenerates `L2-DiffusionSamplers/diffusion_sampler_training.gif`. The deck
uses its frames under `assets/l4_training/`.

Lesson 1 starts with a cost, constructs its Boltzmann distribution, and motivates
approximating it with a tractable distribution. Beside the introductory bullets,
aligned plots show a double-well cost and normalized Boltzmann densities at
temperatures 0.3, 1, and 3. Regenerate `assets/boltzmann_cost.pdf` and
`assets/boltzmann_temperatures.pdf` from the workspace root with
`.venv/bin/python Slides/scripts/generate_boltzmann_temperatures.py`.
One compact KL slide then gives
the integral and expectation definitions, key properties, and a side-by-side
forward/reverse KL comparison. The vector plots `assets/kl_forward.pdf` and
`assets/kl_reverse.pdf` fit one Gaussian to the same two-component mixture.
Forward KL uses exact moment matching; reverse KL uses deterministic quadrature
and optimization from multiple starting points. Regenerate both from the
workspace root with `.venv/bin/python Slides/scripts/generate_kl_comparison.py`.

The reparameterization slide uses a one-dimensional Gaussian:
`X = mu + sigma * epsilon`, with standard-normal noise held fixed during
backpropagation. It gives the reward derivatives with respect to the mean and
standard deviation and the VI loss with analytic Gaussian entropy.

Lesson 3 identifies the discounted infinite-horizon return as the standard
MaxEnt RL objective and cites Soft Actor-Critic. The next slides apply the
entropy-aware policy-gradient theorem, derive the local reverse-KL actor loss,
and give the soft Bellman equations for the rollout policy's critic. The
surrogate gradient equals the negative return gradient at the rollout
parameters, with exact values and expectations. Later updates on the same
batch are local approximations: the trust region stays anchored to the
original rollout policy, and the improvement condition requires surrogate
gain to exceed the bound's error penalty. The slides distinguish population
bounds from practical sampled KL constraints.

The closing references include
[DIME (Celik et al., 2025)](https://arxiv.org/pdf/2502.02316) and
[TruDi (Le et al., 2026)](https://arxiv.org/abs/2606.15260), together with
Soft Actor-Critic and TRPO.
All references are visible together on one static slide, without reveal overlays.

The final thank-you slide invites collaboration on DA-MDPs and includes a
QR code for
[Sebastian Sanokowski's LinkedIn profile](https://www.linkedin.com/in/sebastian-sanokowski-0664451b7/).
The QR code is generated directly in TeX using the same `qrcode` package as
the research introduction.

The TUM and ATARI Lab marks used in the deck are stored under `assets/logos/`.
They were obtained from TUM's official logo portal and the official ATARI Lab
website, respectively.

Content slides are staged with Beamer overlays: advance once to reveal the next
bullet, equation, code alternative, or conclusion. The opening title and lesson
divider and closing frames remain static. The deck contains 52 teaching frames and 133 PDF
pages because each reveal state is stored as a PDF page. The embedded StackCube,
PushT, and training animations continue to play independently.

Lessons 1 and 2 use the reference GMM-40 geometry from Midgley et al.
(2023), with PyTorch seed 0, component means in `[-40, 40]^2`, and covariance
`I`. Their contour-and-sample styling follows arXiv:2502.06685. Lesson 2's
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
   using `L1-VariationalInference/`. Students implement both gradient estimators
   in this notebook, then vary `TEMPERATURE` (including `T = 0`);
2. variance-preserving diffusion samplers, the data-processing inequality,
   and a path-space KL upper bound in `L2-DiffusionSamplers/`;
3. reinforcement learning as variational inference over trajectories, starting
   with the reward-defined target and including the TRPO trust-region motivation
   for REPPO; and
4. diffusion-augmented MDPs following arXiv:2512.02019.
