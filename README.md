# SLEAP Social-Behavior Lab

An interactive, hands-on course in analyzing **animal social behavior from pose-tracking data**.
You start from raw [SLEAP](https://sleap.ai) pose output and work all the way to a trained behavior
classifier, learning the standard computational-ethology toolkit along the way:

**feature extraction → dimensionality reduction → clustering → statistics → labeling → machine learning.**

Everything runs in [**marimo**](https://marimo.io) — reactive Python notebooks where dragging a
slider instantly re-runs the analysis, so you can *watch* how each modeling choice changes the
result. The material uses a small bundled dataset of mouse social interactions (three mice per cage,
recorded continuously) across **two cohorts**, each run through three phases (`pre`, `dep`, `post`),
giving you a **sex** axis, a dominance-**rank** axis (Dom / Mid / Sub), and an experimental
**condition** (food-deprivation) axis to test — with the second cohort providing real replication.

## What you'll build

The course runs as **ten** reactive notebooks over two weeks — five on behavior, five on neural
data. Each row links straight to a free [molab](https://molab.marimo.io) cloud kernel — click **Run**
to open that lesson in the browser (nothing to install; each notebook self-bootstraps its data).

**Week 1 — Behavior.**

| # | Notebook | You learn | You produce | Run |
|---|----------|-----------|-------------|-----|
| 01 | `01_pose_and_identity` | What SLEAP outputs; the keypoint tensor `(frames, mice, nodes, xy)`; why one identity error corrupts everything downstream | Load & scrub a real interaction; a labelled skeleton; a swap audit from normalized velocity | [Run](https://molab.marimo.io/github/talmolab/sleap-social-behavior-lab/blob/main/notebooks/01_pose_and_identity.py) |
| 02 | `02_body_frame_and_features` | Turning raw keypoints into **body-centered** social features; why behavior is rotationally invariant | A per-event interpretable 19-feature vector, arena-invariant | [Run](https://molab.marimo.io/github/talmolab/sleap-social-behavior-lab/blob/main/notebooks/02_body_frame_and_features.py) |
| 03 | `03_exploring_behavior_in_time` | Reading the signal in value, **time & frequency** (Morlet wavelet), and **who-leads-whom** coordination | A rhythm spectrogram + a shuffle-tested coordination estimate | [Run](https://molab.marimo.io/github/talmolab/sleap-social-behavior-lab/blob/main/notebooks/03_exploring_behavior_in_time.py) |
| 04 | `04_pca_clustering_and_stats` | **PCA** → the UMAP **objective** and a behavioral map → **honest statistics**: does sex or food deprivation really change behavior? | The behavioral manifold, data-driven syllables, and a cage-level test with a positive *and* negative control | [Run](https://molab.marimo.io/github/talmolab/sleap-social-behavior-lab/blob/main/notebooks/04_pca_clustering_and_stats.py) |
| 05 | `05_dynamics_and_decoding` | The **transition grammar** (Markov) in time, then **decoding** behavior with cross-validation and a leave-one-cohort-out test | A transition matrix + a decoder validated across cohorts | [Run](https://molab.marimo.io/github/talmolab/sleap-social-behavior-lab/blob/main/notebooks/05_dynamics_and_decoding.py) |

Each notebook defines the terms a newcomer needs, explains **why** we use each method, and asks the
question the next notebook answers.

## Week 2 — From behavior to the brain (notebooks 06–10)

Week 2 reuses Week 1's computational moves on neural recordings — calcium imaging, source
separation, spatial tuning, and neural decoding. Notebooks 06–09 build the imaging toolkit; notebook
10 closes the loop by decoding a social behavior directly off the neurons.

| # | Notebook | What it does | Run |
|---|----------|--------------|-----|
| 06 | `06_motion_correction` | Register a drifting miniscope movie (raw → rigid → piecewise-rigid) and prove it with a **motion index** | [Run](https://molab.marimo.io/github/talmolab/sleap-social-behavior-lab/blob/main/notebooks/06_motion_correction.py) |
| 07 | `07_calcium_extraction` | Background-subtract a striatal movie and pull one cell's calcium trace from a hand-placed **ROI** | [Run](https://molab.marimo.io/github/talmolab/sleap-social-behavior-lab/blob/main/notebooks/07_calcium_extraction.py) |
| 08 | `08_source_extraction` | **CNMF** demixes an optical mixture into per-cell footprints + traces; sort them into a neural sequence | [Run](https://molab.marimo.io/github/talmolab/sleap-social-behavior-lab/blob/main/notebooks/08_source_extraction.py) |
| 09 | `09_place_and_grid_cells` | Occupancy-normalized 2-D **rate maps** + spatial information, validated against a shuffle null | [Run](https://molab.marimo.io/github/talmolab/sleap-social-behavior-lab/blob/main/notebooks/09_place_and_grid_cells.py) |
| 10 | `10_neural_social_decoding` | Build `(9, T)` social-contact **ethograms**, then train a population **decoder** to read social state off calcium | [Run](https://molab.marimo.io/github/talmolab/sleap-social-behavior-lab/blob/main/notebooks/10_neural_social_decoding.py) |

## Quick start

Install [uv](https://docs.astral.sh/uv/) (a fast Python package manager), then:

```bash
uv sync                       # create the environment (Python 3.11)
uv run marimo edit notebooks/01_pose_and_identity.py
```

Work through `01 → 10` in order. In marimo, edit any cell or drag any slider and every dependent
cell updates automatically. To just *view* a notebook without editing:

```bash
uv run marimo run notebooks/04_pca_clustering_and_stats.py
```

## Run in the browser (no install for students)

Give students a browser experience with nothing to install. All the options below use a **real
Python kernel**, so `numba` / `umap-learn` / `hdbscan` work. The WebAssembly / GitHub-Pages
export does **not** — those libraries have no in-browser (Pyodide) build and lesson 04 needs
them, so a static WASM site would break at the clustering notebook.

- **One link for the whole course (recommended).** `serve.py` publishes a landing page plus all
  ten lessons in order under a single URL, each with its own isolated kernel per visitor. Try it
  with `uv run python serve.py` (→ <http://localhost:7860>), then host it free on a Hugging Face
  Docker Space or self-host behind a tunnel. See [`DEPLOY.md`](DEPLOY.md).
- **One notebook at a time (molab).** [molab](https://molab.marimo.io) runs a single notebook on
  a free cloud kernel (the **Run** links in the lesson table above point straight to each one).
  Each notebook declares its dependencies inline (a PEP&nbsp;723 `# /// script` block pinned to
  `pyproject.toml`) and **self-bootstraps**: if it can't find a local checkout it downloads
  `course_utils.py` and the bundled data straight from this repo, so there's nothing to upload.
  It's one link *per lesson* rather than one course site — students open `01`…`10` in turn. See
  [`DEPLOY.md`](DEPLOY.md).

## What's in `data/`

Small, self-contained, no video required (everything renders skeletons on a blank canvas):

- `train_events.npz` — ~2.5k social-approach events (two cohorts): short keypoint windows
  `(N, T, 3, 15, 2)` (mice ordered *approacher, approachee, bystander*), per-mouse ranks, condition,
  and a registry label where one exists.
- `heldout_events.npz` — events from a **held-out cage** (camera 16) with ground-truth aggression
  labels, for honest evaluation in notebook 05.
- `answer_key.csv` — ground-truth categories for the training events (a grading aid / fallback for
  the decoding step in notebook 05).
- `cohort_meta.csv` — per-cage metadata (cohort, sex, rank order, condition).
- `raw_slp/example_*.slp` — a few short real SLEAP clips for notebook 01.

## Provenance (instructors)

The bundle is produced from a lab pipeline by `tools/build_dataset.py` (+ `tools/trim_slp.py`),
which require access to the source data and are **not** needed to run the course. See those files
for exactly how each field was derived. `tools/decode_example_slp.py` turns a raw `.slp` into the
small `example_slp_decoded.npz` that notebook 01 loads (so students need no `sleap-io`). These
tools need `sleap-io`, which is kept out of the default install — get it with `uv sync --extra
build`.

The Week-2 neural notebooks (06–10) were remade from the original EDGE Colab sources kept in
[`2025/`](2025/) (the NEU 457 lineage) — one legacy script per neural lesson, preserved for provenance.

## Skeleton

15 nodes, star topology with two hubs (`head`=1, `TTI`=11, the tail-torso intersection):

```
0 nose   1 head   2 L_ear   3 L_shoulder   4 neck   5 R_ear   6 R_shoulder
7 L_haunch   8 R_haunch   9 tail_1   10 tail_0   11 TTI   12 tail_2   13 tail_tip   14 trunk
```

Mice are colored by dominance **rank**: 🔴 Dom, 🔵 Mid, 🟢 Sub.
