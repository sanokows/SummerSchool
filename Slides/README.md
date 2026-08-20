# Slides

This directory contains the Beamer source for **From Variational Inference to
Diffusion-Based Reinforcement Learning**.

## Build

From this directory, run:

```bash
make
```

or invoke Tectonic directly:

```bash
tectonic --keep-logs --synctex main.tex
```

The resulting deck is `main.pdf`. The checked-in images under `assets/` are
still frames from the first two notebook animations, so the TeX source
compiles without executing the notebooks first. Lesson 4 creates its own slow
600-particle animation at
`L4-DiffusionSamplers/diffusion_sampler_training.gif`.

## Build from the SummerSchool VS Code workspace

Open `/home/sebastian/code/SummerSchool` as the VS Code folder and install the
recommended **LaTeX Workshop** extension. The workspace settings select the
bundled Tectonic recipe automatically, so opening `Slides/main.tex` and saving
it rebuilds `Slides/main.pdf`.

You can also run **Terminal → Run Build Task** (or `Ctrl+Shift+B`) and select
`Build SummerSchool slides`. This task runs from the workspace root and does
not require `latexmk` or a full TeX Live installation.

The talk follows the repository chronology:

1. expected-reward maximization in `L1-CostMin/`;
2. KL divergence and maximum-entropy variational inference in `L2-MaxEntMin/`;
3. reinforcement learning as variational inference over trajectories;
4. variance-preserving diffusion samplers, the data-processing inequality,
   and a path-space KL upper bound in `L4-DiffusionSamplers/`; and
5. diffusion-augmented MDPs following arXiv:2512.02019.
