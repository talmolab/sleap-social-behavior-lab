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
        # 06 · Train a classifier & test it on an unseen cage

        Clustering was unsupervised; now we go **supervised**. We train a small **multi-layer
        perceptron (MLP)** to map a 19-D feature vector to $P(\text{aggression})$:

        $$\mathbf h_1=\sigma(W_1\mathbf x+\mathbf b_1),\quad
          \mathbf h_2=\sigma(W_2\mathbf h_1+\mathbf b_2),\quad
          \hat y=\mathrm{softmax}(W_3\mathbf h_2+\mathbf b_3)$$

        trained by minimizing regularized cross-entropy
        $\;\mathcal L=-\frac1N\sum_i \log \hat y_{i,y_i} + \alpha\lVert W\rVert^2$.

        The crucial test: we evaluate on a **held-out cage (camera 16)** the model never saw during
        training. Good accuracy there = the detector learned *aggression*, not the quirks of specific
        animals. This is the honest way to report a behavior classifier.
        """
    )
    return


@app.cell
def _():
    import os
    import sys
    import numpy as np
    import plotly.graph_objects as go

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

    train = cu.load_events(os.path.join(ROOT, "data", "train_events.npz"))
    held = cu.load_events(os.path.join(ROOT, "data", "heldout_events.npz"))
    X_train = cu.features_batch(train["kp"].astype("float32"))
    X_held = cu.features_batch(held["kp"].astype("float32"))
    y_held = held["agg_label"].astype(int)
    return ROOT, X_held, X_train, cu, go, held, np, os, train, y_held


@app.cell
def _(mo):
    use_mine = mo.ui.switch(value=False, label="train on MY labels from notebook 05 (if I saved ≥40)")
    use_mine
    return (use_mine,)


@app.cell
def _(ROOT, X_train, mo, np, os, train, use_mine):
    _y = train["agg_label"].astype(int).copy()
    _src = "curated labels (full training set)"
    _path = os.path.join(ROOT, "data", "_scratch", "student_labels.csv")
    _mask = np.ones(len(_y), dtype=bool)
    if use_mine.value and os.path.exists(_path):
        import csv
        _lab = {r["event_key"]: int(r["label"]) for r in csv.DictReader(open(_path))}
        if len(_lab) >= 40:
            _keep = np.array([k in _lab for k in train["event_key"]])
            _y = np.array([_lab.get(k, 0) for k in train["event_key"]])
            _mask = _keep
            _src = f"your {int(_keep.sum())} hand labels (notebook 05)"
    Xtr = X_train[_mask]
    ytr = _y[_mask]
    label_src = _src
    _msg = mo.md(f"**Training set:** {label_src} — {len(ytr)} events, {int(ytr.sum())} aggression.")
    _msg
    return Xtr, label_src, ytr


@app.cell
def _(mo):
    hidden = mo.ui.dropdown(options=["(32,)", "(64, 32)", "(128, 64)"], value="(64, 32)",
                            label="hidden layers")
    alpha = mo.ui.dropdown(options=["0.0001", "0.001", "0.01", "0.1"], value="0.001",
                           label="L2 regularization α")
    mo.hstack([hidden, alpha], justify="start", gap=2)
    return alpha, hidden


@app.cell
def _(Xtr, alpha, cu, hidden, ytr):
    import warnings
    from sklearn.model_selection import cross_val_score
    warnings.filterwarnings("ignore")
    _hidden = eval(hidden.value)
    model = cu.make_mlp(hidden=_hidden, alpha=float(alpha.value), max_iter=500)
    cv_auc = cross_val_score(model, Xtr, ytr, cv=4, scoring="roc_auc")
    model.fit(Xtr, ytr)
    return cv_auc, model


@app.cell
def _(cv_auc, mo):
    mo.md(f"**4-fold cross-validated ROC-AUC (training cages):** "
          f"**{cv_auc.mean():.3f}** ± {cv_auc.std():.3f}  ·  folds = "
          f"{', '.join(f'{a:.3f}' for a in cv_auc)}")
    return


@app.cell
def _(X_held, mo, model):
    proba_held = model.predict_proba(X_held)[:, 1]
    mo.md(r"""### The real test: the held-out cage (camera 16)
          The model has never seen these three mice. We score every event and evaluate.""")
    return (proba_held,)


@app.cell
def _(mo):
    thr = mo.ui.slider(0.05, 0.95, value=0.5, step=0.05, label="decision threshold", full_width=True)
    thr
    return (thr,)


@app.cell
def _(go, proba_held, y_held):
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    from plotly.subplots import make_subplots
    _fpr, _tpr, _ = roc_curve(y_held, proba_held)
    _pre, _rec, _ = precision_recall_curve(y_held, proba_held)
    _auc = roc_auc_score(y_held, proba_held)
    _ap = average_precision_score(y_held, proba_held)
    _fig = make_subplots(rows=1, cols=2,
                         subplot_titles=(f"ROC (AUC={_auc:.3f})", f"Precision–Recall (AP={_ap:.3f})"))
    _fig.add_scatter(x=_fpr, y=_tpr, mode="lines", line=dict(color="#d62728", width=3),
                     row=1, col=1, showlegend=False)
    _fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#bbb", dash="dash"),
                     row=1, col=1, showlegend=False)
    _fig.add_scatter(x=_rec, y=_pre, mode="lines", line=dict(color="#4c78a8", width=3),
                     row=1, col=2, showlegend=False)
    _fig.update_xaxes(title="false positive rate", row=1, col=1)
    _fig.update_yaxes(title="true positive rate", row=1, col=1)
    _fig.update_xaxes(title="recall", row=1, col=2)
    _fig.update_yaxes(title="precision", row=1, col=2)
    _fig.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=40, b=10))
    _fig
    return


@app.cell
def _(cu, mo, proba_held, thr, y_held):
    _m = cu.eval_binary(y_held, proba_held, thr=thr.value)
    _cm = _m["confusion"]
    _prec = _cm[1][1] / max(1, _cm[1][1] + _cm[0][1])
    _rec = _cm[1][1] / max(1, _cm[1][1] + _cm[1][0])
    mo.md(
        f"""
        ### Held-out performance @ threshold = {thr.value:.2f}
        |  | predicted **not** | predicted **agg** |
        |---|---|---|
        | actual **not** | {_cm[0][0]} | {_cm[0][1]} |
        | actual **agg** | {_cm[1][0]} | {_cm[1][1]} |

        held-out ROC-AUC **{_m.get('roc_auc', float('nan')):.3f}** · AP **{_m.get('avg_precision', float('nan')):.3f}**
        · precision **{_prec:.2f}** · recall **{_rec:.2f}**

        *Slide the threshold: low → catch more aggression but more false alarms (high recall); high →
        cleaner hits but misses subtle events (high precision). There is no free lunch — you pick the
        operating point for your question.*
        """
    )
    return


@app.cell
def _(cu, held, mo, np, proba_held):
    _top = np.argsort(-proba_held)[:9]
    _evs = [(held["kp"][i].astype("float32"), held["ranks"][i], int(held["contact_rel"][i]))
            for i in _top]
    _gif = cu.grid_gif_bytes(_evs, ncols=3, cell=150, fps=16)
    _hit = int(sum(held["agg_label"][i] for i in _top))
    mo.md(
        f"### The model's 9 most-confident aggression calls (held-out cage)\n\n"
        f"{cu.gif_img_html(_gif, width=470)}\n\n"
        f"**{_hit}/9** are true aggression by the curated labels. Eyeball them: confident calls "
        f"should look like real lunges/pins/chases — the detector generalized to new animals."
    )
    return


@app.cell
def _(label_src, mo):
    mo.md(
        f"""
        ### Course wrap-up
        You went the whole distance on real pose data:

        1. **load** SLEAP keypoints → 2. **allocentric features** → 3. **PCA/UMAP/HDBSCAN** clustering
        → 4. **χ² rank statistics** → 5. **hand-labeling** → 6. an **MLP** that detects aggression on a
        **cage it never saw** (trained on: *{label_src}*).

        Things worth remembering:
        - Pose is just moving points; **features** are where the biology enters.
        - Unsupervised methods **suggest** structure; they rarely hand you clean behavior categories.
        - **Held-out evaluation** (new animals/cages) is the only honest measure of a detector.
        - Every step has knobs, and **labels set the ceiling**. Judgment, not just code, makes the
          analysis good.

        Ideas to explore: add features (tail, acceleration), try a 3-way ethogram, swap the MLP for a
        gradient-boosted tree, or leave out a *different* cage and see if the score holds.
        """
    )
    return


if __name__ == "__main__":
    app.run()
