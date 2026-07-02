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
        # 03 · Finding behavior types by clustering

        We now have a 19-D feature vector per event. **Unsupervised clustering** asks: without any
        labels, do the events fall into recurring *types* (sniff, chase, fight, perch, pass-by)?

        The standard recipe — and every knob that matters — is below. **Drag any slider and the whole
        map re-computes.** Watch how the number and shape of clusters depends on your choices; there
        is no single "correct" clustering, and understanding that is the point.

        `standardize → PCA → residualize → UMAP → HDBSCAN`
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

    events = cu.load_events(os.path.join(ROOT, "data", "train_events.npz"))
    X = cu.features_batch(events["kp"].astype("float32"))
    return X, cu, events, go, np


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. PCA — decorrelate & compress

        Standardize each feature (mean 0, sd 1), then rotate onto the axes of greatest variance.
        PCA diagonalizes the covariance $C=\tfrac1n X^\top X = V\Lambda V^\top$; the first $k$
        eigenvectors give scores $Z = X V_k$. This removes redundancy (many of our 19 features are
        correlated) and denoises before the nonlinear step.
        """
    )
    return


@app.cell
def _(mo):
    pca_k = mo.ui.slider(3, 15, value=10, step=1, label="PCA components (k)", full_width=True)
    pca_k
    return (pca_k,)


@app.cell
def _(X, cu, pca_k):
    Xz, _mu, _sd = cu.standardize(X)
    scores, evr, _pca = cu.pca_scores(Xz, pca_k.value)
    return evr, scores


@app.cell
def _(evr, go, np):
    _fig = go.Figure()
    _fig.add_bar(x=[f"PC{i}" for i in range(len(evr))], y=evr, marker_color="#4c78a8",
                 name="per-PC")
    _fig.add_scatter(x=[f"PC{i}" for i in range(len(evr))], y=np.cumsum(evr), mode="lines+markers",
                     name="cumulative", yaxis="y", line=dict(color="#e45756"))
    _fig.update_layout(template="plotly_white", height=320, title="Scree plot — variance explained",
                       yaxis_title="fraction of variance", margin=dict(l=10, r=10, t=40, b=10))
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Residualization — remove nuisance axes

        In approach events the biggest variance is almost always *overall proximity / locomotor
        magnitude* — "how close, how fast". That's real, but it can dominate the map and hide finer
        social types. **Residualizing** = zeroing selected top PCs before UMAP, so structure that
        isn't just "close vs far" gets a chance to separate. (Default: drop PC0 + PC2.) Toggle them
        and watch the map reorganize.
        """
    )
    return


@app.cell
def _(mo):
    drop_pcs = mo.ui.multiselect(options=[str(i) for i in range(6)], value=["0", "2"],
                                 label="PCs to residualize (drop)")
    drop_pcs
    return (drop_pcs,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3. UMAP — a 2-D map that preserves neighborhoods

        UMAP builds a fuzzy graph of nearest neighbors in feature space and lays it out in 2-D by
        minimizing the cross-entropy between the high-D memberships $v_{ij}$ and low-D ones $w_{ij}$:

        $$\mathcal{L}=\sum_{i\neq j} v_{ij}\log\frac{v_{ij}}{w_{ij}}
          +(1-v_{ij})\log\frac{1-v_{ij}}{1-w_{ij}},\qquad
          w_{ij}=\bigl(1+a\lVert y_i-y_j\rVert^{2b}\bigr)^{-1}.$$

        - **`n_neighbors`** — small = local detail / many islands; large = global shape.
        - **`min_dist`** — how tightly points may pack (small = tight, well-separated blobs).

        *Coordinates are not meaningful distances — only who-is-near-whom is.*
        """
    )
    return


@app.cell
def _(mo):
    n_neighbors = mo.ui.slider(5, 50, value=15, step=1, label="n_neighbors")
    min_dist = mo.ui.slider(0.0, 0.99, value=0.0, step=0.05, label="min_dist")
    mo.vstack([n_neighbors, min_dist])
    return min_dist, n_neighbors


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. HDBSCAN — density clustering (no *k*)

        HDBSCAN groups dense regions and labels sparse points as **noise** (−1) instead of forcing
        every point into a cluster. It uses the *mutual-reachability* distance
        $d_{\text{mreach}}(a,b)=\max\!\bigl(\text{core}_k(a),\text{core}_k(b),d(a,b)\bigr)$
        and keeps clusters that persist across density thresholds.
        **`min_cluster_size`** sets the smallest group you'll accept.
        """
    )
    return


@app.cell
def _(mo):
    min_cluster_size = mo.ui.slider(10, 80, value=15, step=1, label="min_cluster_size", full_width=True)
    min_cluster_size
    return (min_cluster_size,)


@app.cell
def _(cu, drop_pcs, min_dist, n_neighbors, scores):
    _drop = [int(x) for x in drop_pcs.value]
    _res = cu.residualize(scores, _drop)
    emb = cu.run_umap(_res, n_neighbors=n_neighbors.value, min_dist=min_dist.value, seed=42)
    return (emb,)


@app.cell
def _(cu, emb, min_cluster_size):
    labels = cu.run_hdbscan(emb, min_cluster_size=min_cluster_size.value)
    return (labels,)


@app.cell
def _(mo):
    color_by = mo.ui.dropdown(options=["cluster", "aggression", "approacher rank", "condition"],
                              value="cluster", label="color points by")
    color_by
    return (color_by,)


@app.cell
def _(color_by, cu, emb, events, go, labels, np):
    _mode = color_by.value
    if _mode == "cluster":
        _key = labels.astype(int)
        _groups = sorted(set(_key))
        _names = {c: ("noise" if c < 0 else f"C{c}") for c in _groups}
        _pal = {c: ("#cccccc" if c < 0 else None) for c in _groups}
    elif _mode == "aggression":
        _key = events["agg_label"].astype(int)
        _groups = [0, 1]; _names = {0: "not agg", 1: "aggression"}; _pal = {0: "#7f7f7f", 1: "#d62728"}
    elif _mode == "approacher rank":
        _key = events["ranks"][:, 0].astype(int)
        _groups = [1, 2, 3, 0]
        _names = {r: cu.RANK_NAMES[r] for r in _groups}; _pal = {r: cu.RANK_HEX[r] for r in _groups}
    else:
        _cmap = {"pre": 0, "dep": 1, "post": 2}
        _key = np.array([_cmap.get(c, -1) for c in events["condition"]])
        _groups = [0, 1, 2]; _names = {0: "pre", 1: "dep", 2: "post"}
        _pal = {0: "#54a24b", 1: "#e45756", 2: "#4c78a8"}

    _fig = go.Figure()
    for g in _groups:
        m = _key == g
        if not m.any():
            continue
        _fig.add_scattergl(x=emb[m, 0], y=emb[m, 1], mode="markers", name=str(_names.get(g, g)),
                           marker=dict(size=5, opacity=0.7, color=_pal.get(g)),
                           text=[events["category"][i] or "" for i in np.where(m)[0]],
                           hovertemplate="%{text}<extra></extra>")
    _fig.update_layout(template="plotly_white", height=560, title=f"UMAP embedding — colored by {_mode}",
                       xaxis_title="UMAP-1", yaxis_title="UMAP-2", margin=dict(l=10, r=10, t=40, b=10))
    _fig
    return


@app.cell
def _(labels, mo, np):
    _n_clusters = len([c for c in set(labels) if c >= 0])
    _noise = float((labels == -1).mean())
    mo.md(
        f"""
        **{_n_clusters} clusters** · **{_noise:.0%} noise**. &nbsp; With the defaults you should see
        ~7 clusters and low noise. Recolor by **aggression** — one or two clusters are visibly
        red-enriched; recolor by **approacher rank** to eyeball whether those are Dom-driven. The
        next notebook tests that rigorously.

        **Next → `04_rank_stats.py`.**
        """
    )
    return


if __name__ == "__main__":
    app.run()
