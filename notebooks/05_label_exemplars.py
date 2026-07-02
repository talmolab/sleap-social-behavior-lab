# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "marimo>=0.9",
#     "numpy>=1.24,<2.1",
#     "scipy>=1.11",
#     "pandas>=2.0",
#     "scikit-learn>=1.3",
#     "numba>=0.59",
#     "umap-learn>=0.5.6",
#     "hdbscan>=0.8.36",
#     "plotly>=5.20",
#     "imageio>=2.34",
#     "pillow>=10.0",
#     "sleap-io>=0.4",
# ]
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # 05 · Labeling — build training data by hand

        Clustering *suggested* behavior types but couldn't cleanly isolate aggression. To build a
        **reliable detector** we need ground truth: a human watches clips and applies an **ethogram**
        (a defined behavior catalog). Here we use the simplest possible one — a binary decision:

        > **aggression** = a forceful, high-speed contact directed at another mouse (lunge, bite,
        > pin, aggressive chase). *Not aggression* = sniffing, following, passing, perching, resting.

        Below is a grid of real approach events. **Watch each clip and flip the switch on if it looks
        like aggression.** You'll get live feedback on how your calls compare to the curated labels —
        this is exactly how inter-rater reliability is measured in a real lab.
        """
    )
    return


@app.cell
def _():
    import os
    import sys
    import numpy as np

    def _find_root():
        p = os.getcwd()
        for _ in range(6):
            if os.path.isdir(os.path.join(p, "course")) and os.path.isdir(os.path.join(p, "data")):
                return p
            p = os.path.dirname(p)
        return os.getcwd()

    ROOT = _find_root()
    sys.path.insert(0, os.path.join(ROOT, "course"))
    import course_utils as cu

    events = cu.load_events(os.path.join(ROOT, "data", "train_events.npz"))
    SCRATCH = os.path.join(ROOT, "data", "_scratch")
    os.makedirs(SCRATCH, exist_ok=True)
    return ROOT, SCRATCH, cu, events, np, os


@app.cell
def _(mo):
    batch = mo.ui.slider(0, 9, value=0, step=1, label="example batch (drag for a fresh set of 9)",
                         full_width=True)
    batch
    return (batch,)


@app.cell
def _(batch, cu, events, np):
    # Deterministic per-batch draw of 9 events: a shuffled mix of aggression and non-aggression so
    # the labeler must actually watch. Ground truth (from the registry) is held back for feedback.
    _agg = np.where(events["agg_label"] == 1)[0]
    _non = np.where(events["agg_label"] == 0)[0]
    _rng = np.random.RandomState(1000 + batch.value)
    _pa = _rng.choice(_agg, 4, replace=False)
    _pn = _rng.choice(_non, 5, replace=False)
    pick_idx = np.concatenate([_pa, _pn])
    _rng.shuffle(pick_idx)
    pick_keys = [events["event_key"][i] for i in pick_idx]
    pick_truth = [int(events["agg_label"][i]) for i in pick_idx]
    pick_gifs = [cu.event_gif_bytes(events["kp"][i].astype("float32"), events["ranks"][i],
                                    int(events["contact_rel"][i]), cell=150, fps=16) for i in pick_idx]
    return pick_gifs, pick_keys, pick_truth


@app.cell
def _(cu, mo, pick_gifs):
    labels_ui = mo.ui.array([mo.ui.switch(label="aggression") for _ in pick_gifs])
    _tiles = [mo.vstack([mo.md(cu.gif_img_html(pick_gifs[i], 150)), labels_ui[i]], align="center")
              for i in range(len(pick_gifs))]
    _grid = mo.vstack([mo.hstack(_tiles[r * 3:(r + 1) * 3], justify="center", gap=1)
                       for r in range((len(_tiles) + 2) // 3)], gap=1)
    _grid
    return (labels_ui,)


@app.cell
def _(labels_ui, mo, np, pick_truth):
    _mine = np.array([int(bool(v)) for v in labels_ui.value])
    _truth = np.array(pick_truth)
    _agree = int((_mine == _truth).sum())
    _tp = int(((_mine == 1) & (_truth == 1)).sum())
    _fp = int(((_mine == 1) & (_truth == 0)).sum())
    _fn = int(((_mine == 0) & (_truth == 1)).sum())
    mo.md(
        f"""
        ### Your calls vs. the curated labels (this batch)
        - agreement: **{_agree}/{len(_truth)}**
        - you flagged aggression on **{int(_mine.sum())}** clips
          (✅ correct hits **{_tp}**, ❌ false alarms **{_fp}**, ⚠️ missed **{_fn}**)

        *Disagreements are normal — aggression grades continuously into rough play and fast
        sniffing. This ambiguity is exactly why the classifier we train next will never be perfect.*
        """
    )
    return


@app.cell
def _(mo):
    save_btn = mo.ui.run_button(label="💾 save my labels for this batch")
    save_btn
    return (save_btn,)


@app.cell
def _(SCRATCH, labels_ui, mo, os, pick_keys, save_btn):
    _path = os.path.join(SCRATCH, "student_labels.csv")
    if save_btn.value:
        import csv
        _existing = {}
        if os.path.exists(_path):
            for _r in csv.DictReader(open(_path)):
                _existing[_r["event_key"]] = _r["label"]
        for _k, _v in zip(pick_keys, labels_ui.value):
            _existing[_k] = str(int(bool(_v)))
        with open(_path, "w", newline="") as _f:
            _w = csv.writer(_f); _w.writerow(["event_key", "label"])
            for _k, _v in _existing.items():
                _w.writerow([_k, _v])
        _msg = mo.md(f"✅ Saved. `student_labels.csv` now holds **{len(_existing)}** labels. "
                     "Drag the batch slider for more, then go to notebook 06.")
    else:
        _msg = mo.md("*Label the grid, then click the button to append these 9 to your saved set. "
                     "Save several batches to build up a training set for notebook 06.*")
    _msg
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Why this matters
        - A classifier is only as good as its labels. **Ambiguous, inconsistent labels ⇒ a ceiling on
          accuracy** no model can beat.
        - Real ethograms have many categories and multiple raters; you'd measure agreement (Cohen's
          κ) and adjudicate disputes.
        - You just produced supervised training data. Next we feed it (or the full curated set) to a
          neural network and test it on a **cage the model has never seen**.

        **Next → `06_mlp_inference.py`.**
        """
    )
    return


if __name__ == "__main__":
    app.run()
