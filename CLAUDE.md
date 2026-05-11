# CLAUDE.md

Project-local guidance for the `audio_ml_tau_final` repo. Overrides nothing in
the parent `CLAUDE.md`; that file is for the unrelated OLMo workspace.

## What this project is

TAU course final project: a from-scratch implementation of **CodecSlime**
(arXiv 2506.21074, Wang et al. 2025). Plugin-style dynamic-frame-rate wrapper
on a BigCodec (VQ-GAN) speech codec, with two pieces:

- **ScheDFR** — DP-based inference scheduler (`sched_dfr.py`).
- **Melt-and-Cool** — two-stage post-training recipe (`melt_manager.py` covers
  the Melt scheduler).

Paper PDF: `papers/codecslime_2506.21074.pdf`. Project page:
<https://x-lance.github.io/codecslime/>.

## Final deliverables

Two files must be submitted.

### `project.pdf`

- Maximum **5 pages** (references do not count toward the limit).
- Written in **Overleaf / LaTeX**.
- First page top must list every group member's **name and ID number**.

### `project_code.zip`

- **Python 3.10**.
- Must include `requirements.txt` with every pip-installable dependency.
  After `pip install -r requirements.txt`, the code must run as-is.
- Must include `readme.txt` with explicit run instructions for:
  - the **train** script
  - the **evaluation** script
- Must include **audio samples** from both the training set and the
  validation set.

## Coding conventions

- No emojis anywhere in code or generated text.
- No em-dashes (`—`). Use a regular hyphen, comma, or colon.
- Do not over-document. Comments only when the *why* is non-obvious; let
  identifiers carry the *what*.
- Follow existing style in the repo (numpy-only, plain functions/dataclasses,
  short docstrings). Match it before introducing new patterns.

## Paper-implementation notes

`melt_manager.py` has a `USE_PAPER_D_ENFORCE` toggle (default `False`). The
`False` branch is an intentional deviation from the paper's literal d-vector
formula because the paper version appears to invert the intended curriculum.
Do not "fix" it back without discussion.
