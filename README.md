# SLEAP Social-Behavior Lab

An interactive, hands-on course in analyzing **animal social behavior from pose-tracking data**.
You start from raw [SLEAP](https://sleap.ai) pose output and work all the way to a trained behavior
classifier, learning the standard computational-ethology toolkit along the way:

**feature extraction → dimensionality reduction → clustering → statistics → labeling → machine learning.**

Everything runs in [**marimo**](https://marimo.io) — reactive Python notebooks where dragging a
slider instantly re-runs the analysis, so you can *watch* how each modeling choice changes the
result. The material uses a small bundled dataset of mouse social interactions (three mice per cage,
recorded continuously) from a "despotism" experiment with three phases (`pre`, `dep`, `post`), which
gives you both a **rank** axis (Dom / Mid / Sub) and an experimental **condition** axis to test.

## What you'll build

| # | Notebook | You learn | You produce |
|---|----------|-----------|-------------|
| 01 | `load_sleap` | What SLEAP outputs; the keypoint tensor `(frames, mice, nodes, xy)`; why identity matters | Load & visualize a real `.slp`; scrub the skeleton |
| 02 | `features` | Turning raw keypoints into **allocentric** social features (center + rotate into one animal's body frame) | A per-event interpretable feature vector |
| 03 | `clustering` | PCA, covariate **residualization**, **UMAP**, **HDBSCAN**; the role of every hyperparameter | A live 2-D behavioral map you can re-cluster in real time |
| 04 | `rank_stats` | Testing clusters for **rank** and **condition** enrichment (χ², Bonferroni) | The rank-associated behavioral clusters |
| 05 | `label_exemplars` | Turning clusters into labeled training data; ethogram building | Your own **aggression / not-aggression** labels via a click grid |
| 06 | `mlp_inference` | Training an **MLP** classifier and evaluating it on a **held-out cage** | Predicted-aggression clips + ROC/PR on unseen data |

Each notebook shows the **equations** behind the method (e.g. the UMAP objective, PCA
eigendecomposition) and a short **why-we-use-this** justification, not just code.

## Quick start

Install [uv](https://docs.astral.sh/uv/) (a fast Python package manager), then:

```bash
uv sync                       # create the environment (Python 3.11)
uv run marimo edit notebooks/01_load_sleap.py
```

Work through `01 → 06` in order. In marimo, edit any cell or drag any slider and every dependent
cell updates automatically. To just *view* a notebook without editing:

```bash
uv run marimo run notebooks/03_clustering.py
```

## Run in the browser (molab)

No local install required — [molab](https://molab.marimo.io) runs these notebooks on a free
cloud kernel. It uses a **real Python** environment (`uv`-managed), so `numba` / `umap-learn` /
`hdbscan` install and run normally.

1. Push this repo to GitHub.
2. In molab, create a GitHub-synced notebook from this repo. Confirm the file tree shows
   `notebooks/`, `course/`, and `data/` together — the notebooks import `course_utils` from
   `course/` and load the bundled `data/`, so all three must be present.
3. Open `notebooks/01_load_sleap.py` and run. Each notebook declares its dependencies inline
   (a PEP&nbsp;723 `# /// script` block at the top, pinned to match `pyproject.toml`), so molab
   installs the correct versions automatically.

> The WebAssembly export (`marimo export html-wasm`) and GitHub-Pages hosting will **not** work
> for this course: `numba`, `umap-learn`, and `hdbscan` have no in-browser (Pyodide) builds, and
> notebook 03 onward depends on them. Use molab (or a self-hosted `marimo edit`) for a browser
> experience with a real kernel.

## What's in `data/`

Small, self-contained, no video required (everything renders skeletons on a blank canvas):

- `train_events.npz` — ~1.5k social-approach events: short keypoint windows
  `(N, T, 3, 15, 2)` (mice ordered *approacher, approachee, bystander*), per-mouse ranks, condition,
  and a registry label where one exists.
- `heldout_events.npz` — events from a **held-out cage** (camera 16) with ground-truth aggression
  labels, for honest evaluation in notebook 06.
- `answer_key.csv` — ground-truth categories for the training events (a grading aid / fallback for
  the labeling notebook).
- `cohort_meta.csv` — per-cage metadata (sex, rank order, condition).
- `raw_slp/example_*.slp` — a few short real SLEAP clips for notebook 01.

## Provenance (instructors)

The bundle is produced from a lab pipeline by `tools/build_dataset.py` (+ `tools/trim_slp.py`),
which require access to the source data and are **not** needed to run the course. See those files
for exactly how each field was derived.

## Skeleton

15 nodes, star topology with two hubs (`head`=1, `TTI`=11, the tail-torso intersection):

```
0 nose   1 head   2 L_ear   3 L_shoulder   4 neck   5 R_ear   6 R_shoulder
7 L_haunch   8 R_haunch   9 tail_1   10 tail_0   11 TTI   12 tail_2   13 tail_tip   14 trunk
```

Mice are colored by dominance **rank**: 🔴 Dom, 🔵 Mid, 🟢 Sub.
