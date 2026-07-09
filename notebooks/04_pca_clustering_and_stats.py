# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "marimo>=0.9",
#     "numpy>=1.24,<2.1",
#     "scipy>=1.11",
#     "pandas>=2.0",
#     "scikit-learn>=1.3",
#     "hdbscan>=0.8.36",
#     "plotly>=5.20",
#     "imageio>=2.34",
#     "pillow>=10.0",
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
def _():
    import os, sys, urllib.request
    import numpy as np
    import plotly.graph_objects as go
    _RAW = os.environ.get("COURSE_REPO_RAW",
        "https://raw.githubusercontent.com/talmolab/sleap-social-behavior-lab/main")
    def _find_root():
        p = os.getcwd()
        for _ in range(6):
            if os.path.isdir(os.path.join(p, "course")) and os.path.isdir(os.path.join(p, "data")):
                return p
            p = os.path.dirname(p)
        return None
    ROOT = _find_root() or os.getcwd()
    _cu = os.path.join(ROOT, "course", "course_utils.py")
    if not os.path.exists(_cu):
        os.makedirs(os.path.dirname(_cu), exist_ok=True)
        urllib.request.urlretrieve(_RAW + "/course/course_utils.py", _cu)
    sys.path.insert(0, os.path.join(ROOT, "course"))
    import course_utils as cu
    ROOT, DATA, SCRATCH = cu.bootstrap()
    return ROOT, cu, go, np


@app.cell
def _(ROOT, cu, np):
    # ---- canonical data loads (use the loaders; they fetch on a bare kernel) ----------------------
    ev    = cu.load_events(cu.data_path("data/train_events.npz", ROOT))     # kp, ranks, condition, agg
    der   = cu.load_derived("train", ROOT)                                  # X, cohort, cage, sex, PCA
    sweep = cu.load_umap_sweep(ROOT)                                        # precomputed embeddings + labels
    hod   = cu.load_derived("heldout", ROOT)                                # held-out cam16 (count only)

    kp    = ev["kp"]                                                        # (N,T,3,15,2) for GIFs
    ranks = ev["ranks"]                                                     # (N,3) rank per ordered mouse
    cond  = ev["condition"].astype(str)                                     # 'pre'|'dep'|'post'
    yagg  = ev["agg_label"].astype(int)                                     # 1 = aggression

    X      = der["X"]                                                       # (N,19) allocentric features
    fn     = [str(f) for f in der["feature_names"]]                         # 19 feature names
    cage   = der["cage"]                                                    # cohort-unique cage id
    sex    = der["sex"].astype(str)                                         # 'M'/'F' (fixed per cage)
    cohort = der["cohort"].astype(str)                                      # date-tag cohort id

    # Standardize the 19 features, then refit a FULL-RANK PCA so the scree curve can reach 90%.
    # The shipped der['pca_scores'] keeps only 10 components (caps at 0.889) — not enough to read 90%.
    Xz, _mu, _sd = cu.standardize(X)                                        # z-score each feature
    sc, evr, _pca = cu.pca_scores(Xz, 19)                                   # sc (N,19), evr (19,)
    comp = _pca.components_                                                 # (19,19) loadings
    cumvar = np.cumsum(evr)
    dim90  = int(np.searchsorted(cumvar, 0.90) + 1)                         # smallest k with >=90% var
    cum6   = float(cumvar[5])                                               # variance kept by first 6 PCs

    # The canonical behavioral map + syllables (every downstream analysis agrees on THESE).
    emb0    = sweep["emb_grid"][tuple(sweep["default_ij"])]                 # pinned 2-D embedding (N,2)
    clabels = sweep["default_labels"].astype(int)                          # canonical clusters (-1=noise)
    base_rate = float(yagg.mean())                                         # 0.320

    # per-cluster aggression fraction -> the purest (highest-fraction) cluster and its lift
    _cs = sorted(c for c in set(clabels.tolist()) if c >= 0)
    _fr = {c: float(yagg[clabels == c].mean()) for c in _cs}
    best_cluster = max(_fr, key=_fr.get)                                   # 3
    best_frac = _fr[best_cluster]                                          # 0.381
    best_lift = best_frac / base_rate                                      # 1.19x

    # feature-name -> column index, used all over Part C
    HI  = fn.index("heading_alignment")        # 15 — the sex readout (positive control)
    BLI = fn.index("appr_body_len")            # 4  — body size (negative control)
    BDI = fn.index("bystander_dist_mean")      # 16 — the food-deprivation readout

    N    = int(len(X))                                                     # 2499 training events
    n_ho = int(len(hod["cage"]))                                           # 780 held-out events
    EXAMPLE = cu.event_index_by_key(ev, "12192025_pre|cam.10.00046-2025-12-18T16|m0-m2|83141")  # running example (cage 110, 'pre', NON-aggression)

    # canonical cluster palette (c3 = aggression-enriched -> red); pinned per-cluster exemplar indices
    CPAL = {-1: "#d5d5d5", 0: "#8c9196", 1: "#f2b134", 2: "#4c78a8", 3: "#e45756"}
    CLUSTER_EXEMPLARS = {0: [1653, 144, 1599, 912, 2212],
                         1: [769, 2395, 1627, 1150, 410],
                         2: [2142, 405, 163, 892, 94],
                         3: [436, 969, 2023, 1182, 254]}
    return (BDI, BLI, CLUSTER_EXEMPLARS, CPAL, EXAMPLE, HI, N, base_rate,
            best_cluster, best_frac, best_lift, cage, clabels, cohort, comp,
            cond, cum6, cumvar, der, dim90, emb0, ev, evr, fn, kp, n_ho, np,
            ranks, sc, sex, sweep, yagg)


@app.cell(hide_code=True)
def _(N, base_rate, mo, n_ho):
    mo.md(
        rf"""
        # NB04 · The map, its meaning, and what it reveals

        ## Where we are in the argument

        In the previous notebook we turned every social interaction — roughly 130 frames of two mice
        moving around each other — into **19 allocentric numbers** per event: closing speed, pair
        distance, mutual facing, body length, and so on. Nineteen numbers is a workable description,
        but it raises three questions we have not yet answered, and this notebook answers all three in
        one long sitting. We study social behavior and its neural basis; both rest on an objective
        account of what animals actually do. To do that we need a description of behavior that is
        compact, honest, and interpretable.

        Our working corpus is **{N:,} approach events** across two independent food-deprivation
        cohorts, with an aggression base rate of **{base_rate:.0%}**. A separate **{n_ho}** events from
        one camera (an all-female cohort) are sealed away and never touched here; we open them only
        when we test a decoder in NB05.

        ## The three linked questions

        - **(A) How many dimensions does behavior actually have?** The 19 features are correlated, so
          the true number of independent directions is smaller. We answer this with **PCA**, and we
          learn that "how many dimensions" is a *choice*, not a single fact — and that the biggest axis
          is not a nuisance to be discarded.
        - **(B) What behavioral types exist?** Without ever telling a method which events are
          aggression, can it find recurring *kinds* of behavior on its own? We build a 2-D **map**,
          look inside the algorithm that makes it, cluster the map, and give its regions meaning by
          reading them back to the 19 features and to video.
        - **(C) Do sex or food deprivation change behavior?** Once we can measure behavior, we can ask
          whether an experimental variable moves it — and we must do so *honestly*, respecting the fact
          that events from the same cage are not independent observations.

        Each part ends by stating the answer it reached and the question it hands to the next part.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # Part A — How many dimensions does behavior have?

        ## A.0 · Why reduce dimensions at all

        The 19 features are not independent. When two mice rush toward each other, closing speed rises,
        pair distance falls, and mutual facing sharpens *together*. When several measurements move in
        lockstep, they are really reporting on a smaller number of underlying things. Finding that
        smaller number gives us a shorter description that loses almost nothing, and it makes every
        later step — mapping, clustering, decoding — faster and less noisy.

        **Definitions (read before the method).**

        - **Dimensionality reduction** — replacing many numbers per event with a few new numbers that
          keep most of the information.
        - **Principal Component Analysis (PCA)** — the standard linear method for this. It finds new
          axes through the data, ordered so the first captures the most spread, the second the most of
          what remains, and so on.
        - **Principal component (PC)** — one of those new axes. Each PC is a fixed weighted recipe of
          the original 19 features; an event's *score* on a PC is how far along that axis it sits.
        - **Variance explained** — the fraction of the total spread a component accounts for. If PC1
          explains 18% of the variance, it captures 18% of the differences between events.
        - **Residualization** — deliberately zeroing one or more PCs so the remaining description
          reflects everything *except* those axes.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## A.1 · The core idea — find the direction of most spread

        PCA rests on one question: if you could keep only a single direction through a cloud of points,
        which direction preserves the most spread? Everything else is machinery for answering that
        automatically.

        The plot uses two real, correlated features — `pair_dist_mean` and `pair_dist_min`, both
        standardized — so the cloud forms a tilted ellipse. **Drag the angle** to rotate a candidate
        axis. The title reports the *variance of the points projected onto your axis*: how spread out
        they are along it. Find the angle that maximizes it, then flip on **Reveal PCA's axis** — the
        first principal component is exactly the direction you were hunting for. The projected variance
        is small across the short width of the ellipse and largest along its long diagonal.
        """
    )
    return


@app.cell
def _(mo):
    toy_angle = mo.ui.slider(0, 180, value=20, step=1, label="candidate axis angle (deg)",
                             debounce=True, full_width=True)
    toy_reveal = mo.ui.switch(value=False, label="Reveal PCA's axis")
    return toy_angle, toy_reveal


@app.cell
def _(cu, fn, go, mo, np, toy_angle, toy_reveal, X):
    from sklearn.decomposition import PCA as _PCA
    _cols = np.array([fn.index("pair_dist_mean"), fn.index("pair_dist_min")])
    _Xz2, _, _ = cu.standardize(X[:, _cols])                       # (N,2) standardized pair
    _rng = np.random.RandomState(0)
    _sel = _rng.choice(_Xz2.shape[0], size=500, replace=False)     # subsample for a snappy plot
    _P = _Xz2[_sel]

    _th = np.deg2rad(toy_angle.value)
    _u = np.array([np.cos(_th), np.sin(_th)])
    _var = float((_P @ _u).var())

    _p2 = _PCA(n_components=2).fit(_Xz2)
    _pc1 = _p2.components_[0]
    _best_deg = np.rad2deg(np.arctan2(_pc1[1], _pc1[0])) % 180
    _max_var = float(_p2.explained_variance_[0])

    _L = 3.2
    _fig = go.Figure()
    _fig.add_scattergl(x=_P[:, 0], y=_P[:, 1], mode="markers",
                       marker=dict(size=5, color="#7f7f7f", opacity=0.45), name="events")
    _fig.add_scatter(x=[-_L*_u[0], _L*_u[0]], y=[-_L*_u[1], _L*_u[1]], mode="lines",
                     line=dict(color="#f58518", width=3), name="your axis")
    if toy_reveal.value:
        _v = _pc1 / np.linalg.norm(_pc1)
        _fig.add_scatter(x=[-_L*_v[0], _L*_v[0]], y=[-_L*_v[1], _L*_v[1]], mode="lines",
                         line=dict(color="#111111", width=3, dash="dash"),
                         name=f"PC1 (angle {_best_deg:.0f} deg)")
    _fig.update_layout(template="plotly_white", height=460,
                       title=(f"Projected variance = {_var:.2f}   "
                              f"(max {_max_var:.2f} at {_best_deg:.0f} deg)"),
                       xaxis_title="pair_dist_mean (z)", yaxis_title="pair_dist_min (z)",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(range=[-_L, _L], showgrid=False, zeroline=True)
    _fig.update_yaxes(range=[-_L, _L], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=True)
    mo.vstack([toy_angle, toy_reveal, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        You just **maximized variance over a rotation** — that is all PCA does, generalized from 2
        features to 19. The winning direction is **PC1**; the best axis perpendicular to it is **PC2**;
        and so on, each capturing the most remaining spread.

        /// details | Optional — the eigen-math behind the slider
        Standardize the data to $X$ ($n$ events x 19 features, each column mean-0). The covariance is
        $C = \tfrac1n X^\top X$. PCA solves the eigenproblem $C v = \lambda v$: each **eigenvector**
        $v_k$ is a principal direction, and its **eigenvalue** $\lambda_k$ is the variance captured
        along it. The scores are the projections $Z = XV$. "Maximize projected variance over all unit
        directions" and "take the top eigenvector of the covariance" are the *same* statement — the
        slider was solving that eigenproblem by hand. The variance-explained ratio is just
        $\lambda_k / \sum_j \lambda_j$.
        ///
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## A.2 · How many components do we need? The scree plot

        Now run PCA on all 19 standardized features. A **scree plot** shows two things at once: the
        bars are how much variance each individual component explains, and the red line is the running
        total (cumulative variance) as you add components left to right.

        **Drag `keep k`** to highlight the first k components and read the cumulative percentage in the
        title. The bars fall off quickly — the first few are tall, the rest are short — and the
        cumulative line rises steeply then flattens.
        """
    )
    return


@app.cell
def _(mo):
    keep_k = mo.ui.slider(1, 19, value=6, step=1, label="keep k PCs", debounce=True, full_width=True)
    return (keep_k,)


@app.cell
def _(cumvar, dim90, evr, go, keep_k, mo, np):
    _k = keep_k.value
    _x = np.arange(1, 20)
    _fig = go.Figure()
    _fig.add_bar(x=_x, y=evr, name="per-PC variance",
                 marker=dict(color=["#4c78a8" if i < _k else "#cfd8e3" for i in range(19)]))
    _fig.add_scatter(x=_x, y=cumvar, mode="lines+markers", name="cumulative",
                     yaxis="y2", line=dict(color="#e45756", width=2))
    _fig.add_hline(y=0.90, line=dict(color="#999", dash="dot"), yref="y2")
    _fig.add_annotation(x=dim90, y=0.90, yref="y2", text=f"90% at {dim90} PCs",
                        showarrow=True, arrowhead=2, ax=40, ay=-30)
    _fig.update_layout(template="plotly_white", height=440,
                       title=f"Scree — first {_k} PCs keep {cumvar[_k-1]*100:.1f}% of variance",
                       xaxis_title="principal component",
                       yaxis=dict(title="variance ratio", showgrid=False),
                       yaxis2=dict(title="cumulative", overlaying="y", side="right",
                                   range=[0, 1.02], showgrid=False),
                       margin=dict(l=10, r=10, t=50, b=10), legend=dict(x=0.55, y=0.25))
    mo.vstack([keep_k, _fig])
    return


@app.cell(hide_code=True)
def _(cum6, dim90, mo):
    mo.md(
        f"""
        **Read the numbers honestly.** The first **6 components keep {cum6*100:.1f}%** of the variance
        — enough for the map we build in Part B. Reaching a strict **90% threshold takes {dim90}
        components**. Behavior here is genuinely about 6-to-{dim90} dimensional: far less than 19, but
        more than one. "Dimensionality" depends on how much variance you insist on keeping. It is a
        setting, not a single fact.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## A.3 · What does each component mean? Reading the loadings

        Each PC is a weighted recipe of the 19 features, and those weights are the **loadings**. The
        heatmap shows the loadings of the top components: **red** pushes an event's score *up* along
        that PC, **blue** pushes it *down*. Reading across a row tells you which real behaviors that
        component combines — it translates an abstract axis back into behavior.
        """
    )
    return


@app.cell
def _(comp, cu, fn):
    cu.pca_loadings_fig(comp, fn, k=6)
    return


@app.cell(hide_code=True)
def _(comp, fn, mo, np):
    _order = np.argsort(-np.abs(comp[0]))
    _top = ", ".join(f"`{fn[i]}`" for i in _order[:4])
    mo.md(
        f"""
        **Naming PC1.** Its largest loadings are {_top} — mean and peak speeds, angular velocity, and
        closing. Those all describe one coherent thing: **how much motion and engagement is happening**
        between the two mice. It is also the highest-variance axis, which is why PCA ranks it first. A
        natural temptation is to call this axis a nuisance ("just overall activity") and remove it. The
        next two sections show, in video and in numbers, why that removal is a *choice with a cost*.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### See it — behavior at the two ends of a component

        Loadings describe a component in words; GIFs let you see it. Pick a component and watch the
        events that score **lowest** versus **highest** on it. For PC1 you should see calm, mostly
        stationary pairs at the low end and fast, actively engaging pairs at the high end — the amount
        of activity made visible.

        Each mouse is colored by social **rank**: <span style="color:#d62728"><b>Dom = red</b></span>,
        <span style="color:#1f77b4"><b>Mid = blue</b></span>,
        <span style="color:#2ca02c"><b>Sub = green</b></span>. The white arrow points approacher to
        approachee; the red dot appears at contact.
        """
    )
    return


@app.cell
def _(mo):
    pc_pick = mo.ui.dropdown(options=[f"PC{i+1}" for i in range(6)], value="PC1",
                             label="show behavior at the extremes of")
    return (pc_pick,)


@app.cell
def _(cu, kp, mo, np, pc_pick, ranks, sc):
    # Sort every event by its score on the chosen PC, then render the 3 lowest and 3 highest as
    # skeleton GIF grids. This makes the axis concrete: what behavior sits at each end.
    _pc = int(pc_pick.value[2:]) - 1
    _order = np.argsort(sc[:, _pc])                       # ascending score on this PC
    _low = _order[:3]                                     # 3 events lowest on this axis
    _high = _order[-3:]                                   # 3 events highest on this axis
    _lo = cu.grid_gif_bytes([(kp[i], ranks[i], 40) for i in _low], ncols=3, cell=150)
    _hi = cu.grid_gif_bytes([(kp[i], ranks[i], 40) for i in _high], ncols=3, cell=150)
    _html = (
        "<div style='display:flex;gap:24px;flex-wrap:wrap'>"
        f"<div><div style='margin-bottom:4px'><b>Low {pc_pick.value}</b> — bottom of the axis</div>"
        f"{cu.gif_img_html(_lo, width=470)}</div>"
        f"<div><div style='margin-bottom:4px'><b>High {pc_pick.value}</b> — top of the axis</div>"
        f"{cu.gif_img_html(_hi, width=470)}</div>"
        "</div>")
    mo.vstack([pc_pick, mo.md(_html)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## A.4 · Removing an axis is a choice with a cost

        Before we build the map (Part B) we will deliberately remove the large "how fast / how close"
        axis so the map reflects finer differences instead of raw activity level. The helper
        `cu.residualize(scores, drop_pcs)` does exactly that:

        - **Purpose:** set chosen components aside so the rest of the pipeline behaves as if those axes
          never existed. **Inputs:** the PCA scores and a list of component indices to drop.
          **Output:** the same scores with those columns zeroed.

        But aggression is partly a high-motion behavior, so it lives partly on PC1. Choose which
        components to drop and read the aggression-decoding score on the axes that remain. That score
        is the **AUROC** (area under the ROC curve): how well a value separates aggression from
        non-aggression, where **1.0 is perfect and 0.5 is chance**. It is 5-fold cross-validated, so
        the number reflects genuine prediction, not memorization.

        Watch what happens when you drop PC1: the score **drops but does not collapse**. That is the
        whole lesson. PC1 carries real aggression signal, so **we drop it only to shape the map, never
        when we actually decode.**
        """
    )
    return


@app.cell
def _(mo):
    drop_sel = mo.ui.multiselect(options=[f"PC{i+1}" for i in range(6)], value=["PC1"],
                                 label="drop these PCs (zeroed before decoding)", full_width=True)
    return (drop_sel,)


@app.cell
def _(cu, drop_sel, go, mo, sc, yagg):
    from sklearn.model_selection import cross_val_score as _cvs
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.pipeline import make_pipeline as _mkp
    from sklearn.preprocessing import StandardScaler as _SS

    _drop = [int(s[2:]) - 1 for s in drop_sel.value]                  # "PC1" -> index 0
    _res = cu.residualize(sc, _drop)
    def _auc(S):
        return float(_cvs(_mkp(_SS(), _LR(max_iter=1000)), S, yagg, cv=5, scoring="roc_auc").mean())
    _full = _auc(sc)
    _kept = _auc(_res)

    _msg = (f"<div style='font-size:1.05em'>Aggression AUROC — all 19 PCs: <b>{_full:.3f}</b> "
            f"&nbsp;-&gt;&nbsp; after dropping {', '.join(drop_sel.value) or 'nothing'}: "
            f"<b>{_kept:.3f}</b></div>"
            f"<div style='color:#666;font-size:.85em;margin-top:4px'>Chance = 0.500. Dropping PC1 "
            f"weakens but does not erase the signal — a decision with a real trade-off, not a free "
            f"cleanup.</div>")

    # Surviving structure on two remaining axes (PC2 vs PC3), colored by aggression (not by rank).
    _fig = go.Figure()
    for _g, _c, _n in [(0, "#9aa0a6", "not agg"), (1, "#d62728", "aggression")]:
        _m = yagg == _g
        _fig.add_scattergl(x=_res[_m, 1], y=_res[_m, 2], mode="markers", name=_n,
                           marker=dict(size=4, opacity=0.5, color=_c))
    _fig.update_layout(template="plotly_white", height=420,
                       title="Surviving axes (PC2 vs PC3) still separate aggression",
                       xaxis_title="PC2", yaxis_title="PC3", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    _fig.update_yaxes(showgrid=False)
    mo.vstack([drop_sel, mo.md(_msg), _fig])
    return


@app.cell(hide_code=True)
def _(cum6, dim90, mo):
    mo.md(
        f"""
        ### Answer to Question A

        Behavior is roughly **6-to-{dim90} dimensional**: 6 PCs keep {cum6*100:.0f}% of the variance,
        {dim90} PCs reach 90%. PC1 is the overall-activity axis, and it carries genuine aggression
        signal — so "dimensionality" is a choice, and the biggest axis is not automatically a nuisance.

        > **Next question (Part B):** PCA imposed straight axes we chose by variance. If we instead let
        > the data arrange itself, do recurring *kinds* of behavior — including aggression — separate
        > out on their own, without any labels?
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # Part B — What behavioral types exist?

        ## B.0 · Why build a map, and the words we need

        PCA found directions of greatest variance in the numbers we handed it; it does not know what a
        mouse is doing. Here we take a different stance: instead of imposing axes, we let the data show
        us which **kinds of behavior recur**, without deciding in advance what to look for. We lay every
        event out as one point on a 2-D map, group nearby points into types, and check whether one type
        corresponds to aggression — a category we never named.

        **Definitions.**

        - **Unsupervised** — a method that looks only at the features and groups by similarity, never
          shown the labels. (Supervised learning, trained on labels, comes in NB05.)
        - **Embedding / 2-D map** — a procedure that places each high-dimensional event at an (x, y)
          position so similar events land near each other. Our tool is **UMAP**. One dot is one whole
          interaction.
        - **Clustering** — grouping the dots so dense pockets become named groups; sparse points can be
          left as "noise." Our tool is **HDBSCAN**.
        - **Behavioral type ("syllable")** — one recurring group the clustering finds. "Syllable" is
          borrowed from the behavior literature; treat it as a synonym for a data-driven cluster of
          similar events.

        Before we trust the map, we open the algorithm that makes it — because a map you cannot explain
        is a map you cannot defend.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## B.1 · What UMAP actually optimizes (a live, tiny demonstration)

        UMAP is often treated as a black box that "just makes a picture." It is not. It optimizes a
        concrete objective, and we can watch it do so on a small toy that runs live in a fraction of a
        second (this is a ~90-point pure-numpy teaching toy — it does **not** violate the rule against
        live UMAP on the real 2,499-point data, which stays precomputed).

        **The recipe UMAP follows.**

        1. In the original high-dimensional space, convert distances into **fuzzy neighbor
           memberships**: for each point, its true nearest neighbors get a membership near 1, and the
           membership falls off smoothly with distance. This is the graph UMAP wants to reproduce.
        2. In the 2-D layout, define a **low-D similarity** between points, $q_{ij} = 1/(1+d_{ij}^2)$,
           that is near 1 when two points are close and near 0 when far.
        3. Move the 2-D points to make $q$ match the high-D memberships $P$, minimizing a **fuzzy
           cross-entropy**. The gradient splits into an **attractive** force (true neighbors pull
           together) and a **repulsive** force (everything else pushes apart).

        Below, `cu.umap_objective_toy(...)` builds 3 Gaussian blobs in 8-D and runs this optimization.
        Step the slider through the saved snapshots and watch a random cloud organize into the three
        blobs it should recover.
        """
    )
    return


@app.cell
def _(cu):
    toy = cu.umap_objective_toy(seed=1)          # ~90 points, 3 blobs in 8-D; runs in < 1 s
    return (toy,)


@app.cell
def _(mo, toy):
    toy_step = mo.ui.slider(0, len(toy["snapshots"]) - 1, value=0, step=1,
                            label="optimization snapshot (drag right to watch it organize)",
                            debounce=True, full_width=True)
    return (toy_step,)


@app.cell
def _(cu, mo, toy, toy_step):
    _fig = cu.umap_objective_layout_fig(toy, snapshot=int(toy_step.value),
                                        title="UMAP toy — the 2-D layout organizing over epochs")
    mo.vstack([toy_step, _fig])
    return


@app.cell(hide_code=True)
def _(go, mo, toy):
    _lh = toy["loss_history"]
    _fig = go.Figure(go.Scatter(x=list(range(1, len(_lh) + 1)), y=_lh, mode="lines",
                                line=dict(color="#4c78a8", width=2)))
    _fig.update_layout(template="plotly_white", height=320,
                       title=f"The objective falling — fuzzy cross-entropy "
                             f"{_lh[0]:.0f} -> {_lh[-1]:.0f}",
                       xaxis_title="epoch", yaxis_title="cross-entropy loss",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    mo.vstack([mo.md("As the layout organizes, the cross-entropy between the high-D memberships and "
                     "the low-D similarities drops sharply and then flattens — the same shape as the "
                     "scree curve's cumulative line, for the same reason: most of the work is done "
                     "early."),
               _fig])
    return


@app.cell(hide_code=True)
def _(go, mo, np, toy):
    # (1) the high-D fuzzy membership curve UMAP FITS: membership vs high-D distance.
    _hd, _hm = toy["high_dist"], toy["high_membership"]
    _fig = go.Figure(go.Scattergl(x=_hd, y=_hm, mode="markers",
                                  marker=dict(size=3, color="#8c564b", opacity=0.35)))
    _fig.update_layout(template="plotly_white", height=340,
                       title="High-D fuzzy membership vs distance — near neighbors ~1, far pairs ~0",
                       xaxis_title="high-D distance between a pair", yaxis_title="membership P",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    _fig.update_yaxes(showgrid=False)
    mo.vstack([mo.md("**The target graph.** Each dot is one pair of toy points. Close pairs have "
                     "membership near 1 (they are neighbors); distant pairs decay toward 0. This "
                     "smooth curve — not a hard cutoff — is the *fuzzy* graph UMAP tries to reproduce "
                     "in 2-D."),
               _fig])
    return


@app.cell(hide_code=True)
def _(go, mo, toy):
    # (2) attractive vs repulsive force as a function of the FINAL low-D distance.
    _ld = toy["low_dist"]
    _fig = go.Figure()
    _fig.add_scattergl(x=_ld, y=toy["attractive"], mode="markers", name="attractive (P q)",
                       marker=dict(size=3, color="#2ca02c", opacity=0.35))
    _fig.add_scattergl(x=_ld, y=toy["repulsive"], mode="markers", name="repulsive ((1-P) q^2/(1-q))",
                       marker=dict(size=3, color="#d62728", opacity=0.35))
    _fig.update_layout(template="plotly_white", height=360,
                       title="Two forces on the layout — neighbors pull, non-neighbors push",
                       xaxis_title="low-D distance between a pair", yaxis_title="force magnitude",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    _fig.update_yaxes(showgrid=False)
    mo.vstack([mo.md("**The two forces.** Green is the **attractive** force — strong only for true "
                     "neighbors (high $P$) that are still close in 2-D, pulling them tighter. Red is "
                     "the **repulsive** force — it acts on non-neighbors and grows as points sit too "
                     "close, pushing them apart. The final layout is the balance point of these two."),
               _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Two things follow directly from this objective, and they are exactly the limits we respect
        later:

        - UMAP only tries to preserve **who is near whom** (local neighborhoods). It has no term that
          preserves *global* distances, so distance *between* clusters on the map is not meaningful.
        - The apparent size and tightness of a blob depends on the balance of the two forces, which the
          settings (`n_neighbors`, `min_dist`) tune. So cluster area is not "how common" a behavior is.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## B.2 · The real map is precomputed (live UMAP is disabled here)

        On the real {N}-event data we do **not** run UMAP live. On a fresh cloud kernel UMAP's first
        call spends ~30 seconds compiling specialized numerical code — long enough that the notebook's
        connection times out, and every slider change would recompile. So the course ships a
        precomputed **5x5 sweep**: the same events already embedded at every combination of the two
        settings. We *select* a precomputed map; we never recompute one. Only HDBSCAN runs live (it is
        fast).

        The figure shows the full sweep as a grid of small maps. Each panel is the same events at a
        different `n_neighbors` (rows) and `min_dist` (columns); a point is
        <span style="color:#d62728">red</span> if that event is aggression, gray otherwise. Scan the
        rows and columns to see how the two settings reshape the map.
        """
    )
    return


@app.cell
def _(cu, sweep, yagg):
    cu.sweep_grid_fig(
        sweep["emb_grid"], sweep["nn_values"], sweep["md_values"],
        color_key=yagg, palette={0: "#b6bac1", 1: "#d62728"},
        names={0: "not agg", 1: "aggression"}, height=680)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## B.3 · Select one map from the sweep

        The two settings control the shape:

        - **`n_neighbors`** — how many neighbors each point is tied to. Small values emphasize local
          detail and break the data into many islands; large values emphasize global shape and merge
          everything into one continent.
        - **`min_dist`** — how tightly points may pack. Small values give tight, separated blobs; large
          values spread points out.

        Pick any cell. The map redraws instantly from the precomputed grid, now colored by the
        **canonical syllables** (`sweep["default_labels"]`) that the rest of the analysis uses. The gold
        star marks our running example event. The default cell (`n_neighbors=15, min_dist=0.0`) is the
        one everything is pinned to.
        """
    )
    return


@app.cell
def _(mo, sweep):
    nn_pick = mo.ui.dropdown(options={f"{v}": i for i, v in enumerate(sweep["nn_values"])},
                             value=str(int(sweep["nn_values"][int(sweep["default_ij"][0])])),
                             label="n_neighbors")
    md_pick = mo.ui.dropdown(options={f"{v:g}": j for j, v in enumerate(sweep["md_values"])},
                             value=f"{float(sweep['md_values'][int(sweep['default_ij'][1])]):g}",
                             label="min_dist")
    return md_pick, nn_pick


@app.cell
def _(CPAL, EXAMPLE, clabels, go, md_pick, mo, nn_pick, sweep):
    _i, _j = int(nn_pick.value), int(md_pick.value)
    _emb = sweep["emb_grid"][_i, _j]
    _fig = go.Figure()
    for _c in sorted(set(clabels.tolist())):
        _m = clabels == _c
        _nm = "noise" if _c < 0 else f"C{_c}"
        _fig.add_scattergl(x=_emb[_m, 0], y=_emb[_m, 1], mode="markers", name=_nm,
                           marker=dict(size=5, opacity=0.7, color=CPAL[_c]))
    _fig.add_scatter(x=[_emb[EXAMPLE, 0]], y=[_emb[EXAMPLE, 1]], mode="markers", name=f"example #{EXAMPLE}",
                     marker=dict(symbol="star", size=17, color="#f5b400",
                                 line=dict(color="#333", width=1)))
    _fig.update_layout(template="plotly_white", height=520,
                       title=(f"Selected map — n_neighbors={int(sweep['nn_values'][_i])}, "
                              f"min_dist={float(sweep['md_values'][_j]):g} — colored by canonical syllable"),
                       xaxis_title="UMAP-1", yaxis_title="UMAP-2", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False).update_yaxes(showgrid=False)
    mo.vstack([mo.hstack([nn_pick, md_pick], justify="start"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### What one point represents

        Every dot is **one entire interaction event** — reduced from a skeleton, to 19 features, to a
        few PC scores, to one (x, y) location. When two dots sit close, the two events had similar
        posture and motion. Before we cluster, look at a real event so the dots feel concrete. Our
        running example is a **non-aggression** close approach in cage 110 (an event the earlier
        automatic detector had wrongly flagged, which is why we keep it as an honest "near miss"): the
        <span style="color:#2ca02c">Sub mouse (green)</span> approaches the
        <span style="color:#1f77b4">Mid mouse (blue)</span>, with the
        <span style="color:#d62728">Dom mouse (red)</span> nearby as bystander.
        """
    )
    return


@app.cell
def _(EXAMPLE, cu, ev, mo):
    _gif = cu.event_gif_bytes(ev["kp"][EXAMPLE], ev["ranks"][EXAMPLE],
                              int(ev["contact_rel"][EXAMPLE]), cell=240)
    mo.md(f"<b>Example event #{EXAMPLE}</b> — a calm, non-aggressive approach. Mice colored by rank; "
          "the white arrow points approacher -> approachee; the red dot marks contact.<br>"
          + cu.gif_img_html(_gif, width=260))
    return


@app.cell(hide_code=True)
def _(base_rate, mo):
    mo.md(
        rf"""
        ### And what aggression looks like

        For contrast, here are four **aggression** events (left) and four **non-aggression** events
        (right), so the category we are hunting for is concrete before we ask an algorithm to find it.
        Aggression tends to be close, contact-heavy, and fast; non-aggression is looser and calmer.
        Aggression is only about {base_rate:.0%} of these already-filtered approach events, so it is a
        minority we are trying to recover, not the default.
        """
    )
    return


@app.cell
def _(cu, kp, mo, ranks):
    _agg = [969, 560, 900, 53]
    _non = [161, 341, 376, 345]
    _ga = cu.grid_gif_bytes([(kp[i], ranks[i], 40) for i in _agg], ncols=2, cell=140)
    _gn = cu.grid_gif_bytes([(kp[i], ranks[i], 40) for i in _non], ncols=2, cell=140)
    _html = ("<div style='display:flex;gap:24px;flex-wrap:wrap'>"
             f"<div><div style='margin-bottom:4px'><b>Aggression</b></div>"
             f"{cu.gif_img_html(_ga, width=320)}</div>"
             f"<div><div style='margin-bottom:4px'><b>Non-aggression</b></div>"
             f"{cu.gif_img_html(_gn, width=320)}</div></div>")
    mo.md(_html)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## B.4 · Group the points — HDBSCAN (live)

        **What clustering does here.** HDBSCAN scans the map for dense pockets and calls each pocket a
        cluster. Points in sparse regions are labelled **noise (-1)** rather than forced into a group.
        Its input is the (x, y) map plus `min_cluster_size` (the smallest group it will accept); its
        output is a label per event. Unlike k-means, you never say how many clusters to find.

        This is the one step we run live. At the canonical `min_cluster_size = 15` it reproduces the
        shared `default_labels` exactly. Larger values merge syllables; smaller values fracture the map
        and push more points into noise.
        """
    )
    return


@app.cell
def _(mo):
    mcs = mo.ui.slider(8, 80, value=15, step=1, label="min_cluster_size (live)",
                       debounce=True, full_width=True)
    return (mcs,)


@app.cell
def _(cu, emb0, go, mcs, mo):
    _lab = cu.run_hdbscan(emb0, min_cluster_size=int(mcs.value))
    _nc = len([c for c in set(_lab.tolist()) if c >= 0])
    _noise = float((_lab == -1).mean())
    _fig = go.Figure()
    for _c in sorted(set(_lab.tolist())):
        _m = _lab == _c
        _fig.add_scattergl(x=emb0[_m, 0], y=emb0[_m, 1], mode="markers",
                           name=("noise" if _c < 0 else f"C{_c}"),
                           marker=dict(size=5, opacity=0.7,
                                       color=("#cfd2d8" if _c < 0 else None)))
    _fig.update_layout(template="plotly_white", height=470,
                       title=f"Live HDBSCAN — {_nc} clusters, {_noise:.0%} noise "
                             f"(min_cluster_size={int(mcs.value)})",
                       xaxis_title="UMAP-1", yaxis_title="UMAP-2", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False).update_yaxes(showgrid=False)
    mo.vstack([mcs, _fig,
               mo.md("*Set the slider to 15 to reproduce the canonical result (4 clusters). This "
                     "slider is for exploration; the shared syllables stay fixed at 15.*")])
    return


@app.cell(hide_code=True)
def _(base_rate, best_cluster, best_frac, best_lift, clabels, mo):
    _sizes = {f"C{c}": int((clabels == c).sum())
              for c in sorted(set(clabels.tolist())) if c >= 0}
    mo.md(
        f"""
        ## B.5 · The syllables, and where aggression concentrates

        The canonical clustering produces **{len(_sizes)} syllables** plus noise, with sizes
        {_sizes}. Ranking them by aggression fraction, the purest is **C{best_cluster}** at
        **{best_frac:.0%} aggression** — a **{best_lift:.2f}x lift** over the {base_rate:.0%} base
        rate. That is a modest but real enrichment: an unsupervised method, never shown a single
        label, still concentrated aggression into one region of the map.

        Pick a syllable and render five of its member events (nearest the cluster centroid) as skeleton
        GIFs. Watching the members is the by-eye half of validating a cluster; the enrichment number is
        the by-number half. C3 should look contact-heavy; the large C2 should look like a mixture.
        """
    )
    return


@app.cell
def _(best_cluster, clabels, mo):
    _opts = {f"C{c}": c for c in sorted(set(clabels.tolist())) if c >= 0}
    clus_pick = mo.ui.dropdown(options=_opts, value=f"C{best_cluster}", label="syllable to render")
    return (clus_pick,)


@app.cell
def _(CLUSTER_EXEMPLARS, clabels, clus_pick, cu, kp, mo, np, ranks, yagg):
    _c = int(clus_pick.value)
    _pick = CLUSTER_EXEMPLARS[_c]
    _gif = cu.grid_gif_bytes([(kp[i], ranks[i], 40) for i in _pick], ncols=5, cell=130)
    _n = int((clabels == _c).sum())
    _frac = float(yagg[clabels == _c].mean())
    _cap = (f"**C{_c}** · {_n} events · {_frac:.0%} aggression · showing 5 exemplars "
            f"nearest the cluster centroid")
    mo.vstack([clus_pick, mo.md(_cap), mo.md(cu.gif_img_html(_gif, width=640))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## B.6 · Give the map meaning — color it by each feature

        A cluster label alone is opaque. To make the map's regions mean something, we paint the *same*
        precomputed layout by each of the 19 features in turn (`cu.umap_colored_by_feature_fig` — it
        never runs UMAP, it only recolors the points). Where a feature forms a smooth gradient across
        the map, that feature is one of the axes the map is organized along.

        Pick a feature. Good ones to start with: `heading_alignment`, `pair_dist_mean`,
        `appr_speed_mean`, `closing_speed`, `bystander_dist_mean`. Watch which regions light up.
        """
    )
    return


@app.cell
def _(fn, mo):
    feat_pick = mo.ui.dropdown(options={f: i for i, f in enumerate(fn)},
                               value="closing_speed" if "closing_speed" in fn else fn[0],
                               label="color the map by feature")
    return (feat_pick,)


@app.cell
def _(X, cu, emb0, feat_pick, fn, mo, np):
    _idx = int(feat_pick.value)
    _name = fn[_idx]
    _fig = cu.umap_colored_by_feature_fig(emb0, X[:, _idx], name=_name,
                                          hover=np.arange(len(X)),
                                          title=f"Canonical map colored by {_name}")
    mo.vstack([feat_pick, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Per-cluster feature profile

        Coloring shows *which* features vary across the map; a **feature profile** summarizes *how each
        cluster differs*. Below, each cluster is collapsed to the **z-scored mean of every feature** —
        red means "this cluster runs high on that feature," blue "low." Reading a column tells you what
        a syllable *is*, in the vocabulary of the 19 features. The aggression-enriched C3 should run
        high on closing/contact features and low on pair distance.
        """
    )
    return


@app.cell
def _(Xz, clabels, fn, go, np):
    _cs = sorted(c for c in set(clabels.tolist()) if c >= 0)
    _prof = np.vstack([np.nanmean(Xz[clabels == c], axis=0) for c in _cs])   # (n_clusters, 19)
    _fig = go.Figure(go.Heatmap(
        z=_prof.T, x=[f"C{c}" for c in _cs], y=fn,
        colorscale="RdBu_r", zmid=0, zmin=-1.2, zmax=1.2,
        colorbar=dict(title="z-mean")))
    _fig.update_layout(template="plotly_white", height=560,
                       title="Per-cluster feature profile (z-scored feature means)",
                       xaxis_title="syllable", margin=dict(l=10, r=10, t=50, b=10))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The profile is a table; the raw points tell you how reliable each cell is. Pick a feature and
        see its full per-cluster distribution as a box-with-points display — quartiles plus every event
        — so a "high" cluster mean backed by tight points reads very differently from one dragged up by
        a few outliers.
        """
    )
    return


@app.cell
def _(fn, mo):
    profile_pick = mo.ui.dropdown(options={f: i for i, f in enumerate(fn)},
                                  value="closing_speed" if "closing_speed" in fn else fn[0],
                                  label="feature to break down by cluster")
    return (profile_pick,)


@app.cell
def _(X, clabels, cu, mo, np, profile_pick, fn):
    _idx = int(profile_pick.value)
    _keep = clabels >= 0
    _groups = np.array([f"C{c}" for c in clabels[_keep]])
    _order = [f"C{c}" for c in sorted(c for c in set(clabels.tolist()) if c >= 0)]
    _fig = cu.box_points_fig(X[_keep, _idx], _groups, group_order=_order,
                             ylabel=fn[_idx], xlabel="syllable",
                             title=f"{fn[_idx]} by syllable — quartiles + every event")
    mo.vstack([profile_pick, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## B.7 · Looking inside a cluster — subclustering (L1 -> L2)

        A single clustering pass gives coarse groups, and a large group may hide several behaviors that
        look similar at low resolution. The standard response is to zoom in: cluster once (level 1),
        then take one large group and cluster *its members again* (level 2).

        Take the large **C2** and re-run HDBSCAN on just its coordinates at a finer `min_cluster_size`.
        A group that looked homogeneous can split into sub-types with **different aggression rates** —
        one nearly aggression-free (a quiet co-presence mode) and another carrying the fights the coarse
        pass absorbed. This is why "one cluster" is rarely the final answer.
        """
    )
    return


@app.cell
def _(clabels, mo):
    _opts = {f"C{c}": c for c in sorted(set(clabels.tolist())) if c >= 0}
    parent_pick = mo.ui.dropdown(options=_opts, value="C2", label="parent syllable to split")
    sub_mcs = mo.ui.slider(15, 60, value=25, step=1, label="sub min_cluster_size", debounce=True)
    return parent_pick, sub_mcs


@app.cell
def _(clabels, cu, emb0, go, mo, np, parent_pick, sub_mcs, yagg):
    _p = int(parent_pick.value)
    _mask = clabels == _p
    _sub = cu.run_hdbscan(emb0[_mask], min_cluster_size=int(sub_mcs.value))
    _e = emb0[_mask]
    _a = yagg[_mask]
    _fig = go.Figure()
    _lines = []
    for _c in sorted(set(_sub.tolist())):
        _m = _sub == _c
        _nm = "noise" if _c < 0 else f"C{_p}.{_c}"
        _fig.add_scattergl(x=_e[_m, 0], y=_e[_m, 1], mode="markers", name=_nm,
                           marker=dict(size=6, opacity=0.75,
                                       color=("#cfd2d8" if _c < 0 else None)))
        if _c >= 0:
            _lines.append(f"**{_nm}**: n={int(_m.sum())}, aggression={_a[_m].mean():.0%}")
    _nsub = len([c for c in set(_sub.tolist()) if c >= 0])
    _fig.update_layout(template="plotly_white", height=460,
                       title=f"Sub-types inside C{_p} — {_nsub} level-2 clusters "
                             f"(sub min_cluster_size={int(sub_mcs.value)})",
                       xaxis_title="UMAP-1", yaxis_title="UMAP-2", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False).update_yaxes(showgrid=False)
    mo.vstack([mo.hstack([parent_pick, sub_mcs], justify="start"), _fig,
               mo.md(f"C{_p} overall aggression rate is about {_a.mean():.0%}. Sub-types: "
                     + " · ".join(_lines))])
    return


@app.cell(hide_code=True)
def _(base_rate, mo):
    mo.md(
        rf"""
        ## B.8 · Exercise — did the map rediscover aggression?

        **The question.** Is at least one data-driven syllable enriched for aggression above the
        {base_rate:.0%} base rate? If so, the unsupervised map found aggression on its own.

        **Python skill practiced.** *Looping over groups and summarizing each with a boolean mask* — the
        pattern behind every "group-by" you will ever write. You will iterate over clusters and, for
        each, compute the fraction of its events that are aggression.

        **What you have.**

        - `clabels : (N,) int` — the canonical syllable of each event (-1 = noise).
        - `yagg : (N,) int` — 1 if the event is aggression, else 0.
        - `base_rate : float` — the corpus-wide aggression rate.

        **Your task.** `purest_agg_cluster` should return the syllable with the highest **aggression
        fraction**, but it currently ranks clusters by **size**. Fix the one flagged line so it ranks
        by aggression fraction. Everything else — computing `frac` and `lift` for whichever cluster you
        pick — is done for you.
        """
    )
    return


@app.cell
def _(base_rate, clabels, np, yagg):
    def purest_agg_cluster(clabels, yagg, base_rate):
        clabels = np.asarray(clabels); yagg = np.asarray(yagg)
        clusters = [c for c in sorted(set(clabels.tolist())) if c >= 0]
        # -------------------- EDIT THE NEXT LINE ONLY --------------------
        # `max(clusters, key=...)` returns the cluster with the largest value of `key`.
        # Right now key = (clabels == c).sum(), which is the cluster's SIZE (number of events) —
        # so it picks the BIGGEST cluster, which is mostly non-aggression and misses the point.
        # You want the cluster with the highest AGGRESSION FRACTION instead. That fraction is the
        # MEAN of the 0/1 aggression labels over the events in cluster c:
        #     yagg[clabels == c].mean()   -> fraction of cluster c that is aggression.
        # Replace  (clabels == c).sum()  with  yagg[clabels == c].mean().
        chosen = max(clusters, key=lambda c: (clabels == c).sum())     # <-- EDIT THIS LINE
        # -----------------------------------------------------------------
        frac = float(yagg[clabels == chosen].mean())     # aggression fraction of the chosen cluster
        lift = frac / base_rate                          # how many times the base rate it reaches
        return int(chosen), frac, lift

    student_cluster, student_frac, student_lift = purest_agg_cluster(clabels, yagg, base_rate)
    return purest_agg_cluster, student_cluster, student_frac, student_lift


@app.cell(hide_code=True)
def _(base_rate, clabels, go, mo, np, yagg):
    # The expected picture: aggression fraction per cluster, base rate as a reference line.
    _cs = sorted(c for c in set(clabels.tolist()) if c >= 0)
    _fr = [float(yagg[clabels == c].mean()) for c in _cs]
    _fig = go.Figure(go.Bar(x=[f"C{c}" for c in _cs], y=_fr,
                            marker_color=["#e45756" if f == max(_fr) else "#9aa0a6" for f in _fr],
                            text=[f"{f:.0%}" for f in _fr], textposition="outside"))
    _fig.add_hline(y=base_rate, line=dict(color="#333", dash="dash"),
                   annotation_text=f"base rate {base_rate:.0%}")
    _fig.update_layout(template="plotly_white", height=360,
                       title="Aggression fraction per syllable (red = purest)",
                       yaxis_title="fraction aggression", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    mo.md("**Expected picture after the fix:** the purest syllable (red) sits clearly above the dashed "
          "base-rate line, while the largest cluster sits near or below it — the aggression signal "
          "lives in a small dense pocket, not the big blob.")
    return _fig,


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Reveal solution": mo.md(
            r"""
            Rank the clusters by **aggression fraction**, not size:

            ```python
            chosen = max(clusters, key=lambda c: yagg[clabels == c].mean())
            ```

            On the canonical labels this returns **C3**, about 38% aggression, a **1.19x lift**. The
            largest cluster (C2, ~1,423 events) sits near the base rate — which is why ranking by size
            gives the wrong answer.
            """)
    })
    return


@app.cell(hide_code=True)
def _(best_cluster, best_lift, mo, student_cluster, student_frac, student_lift):
    _PIN, _TOL = best_lift, 0.10          # accept the pinned lift +/- 0.10
    _ok = abs(student_lift - _PIN) <= _TOL and student_cluster == best_cluster
    if _ok:
        _bg, _fg, _icon = "#e7f6ec", "#166534", "PASS"
        _msg = (f"C{student_cluster} is {student_frac:.0%} aggression, a **{student_lift:.2f}x lift** "
                f"(pinned {_PIN:.2f}). The map recovered aggression **without any labels** — but read "
                f"it honestly: {student_frac:.0%} is a modest enrichment, not a pure fighting island. "
                f"Unsupervised recovery is real here, not clean.")
    else:
        _bg, _fg, _icon = "#fdecec", "#9b1c1c", "NOT YET"
        _msg = (f"Got C{student_cluster} at **{student_lift:.2f}x** — expected C{best_cluster} at about "
                f"{_PIN:.2f}x. If your lift is near 1.0x you are still ranking by *size* (the biggest "
                f"cluster). Rank by `yagg[clabels == c].mean()` instead.")
    mo.md(f"<div style='border:1px solid #ccc;border-radius:8px;padding:10px 14px;background:{_bg};"
          f"color:{_fg}'><b>{_icon}</b> &nbsp; {_msg}</div>")
    return


@app.cell(hide_code=True)
def _(best_cluster, best_lift, mo):
    mo.md(
        f"""
        ### Answer to Question B

        Distinct behavioral types **do** exist, and aggression is one of them: an unsupervised map,
        built and clustered without labels, concentrates aggression into **C{best_cluster}** at a
        **{best_lift:.2f}x** lift, and the per-cluster feature profile tells us *why* — that syllable
        runs high on closing and contact features. The recovery is real but impure (~38% aggression),
        which is exactly why NB05 will train a supervised decoder to sharpen it.

        > **Next question (Part C):** now that we can measure and cluster behavior, does an experimental
        > variable — the animals' **sex**, or **food deprivation** — actually change it? And how do we
        > test that without fooling ourselves, when thousands of events come from only a handful of
        > cages?
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # Part C — Do sex or food deprivation change behavior?

        ## C.0 · Why the unit of analysis is everything

        We now have behavioral readouts we trust. The scientific payoff is comparing them across
        conditions: are males and females different? Does hunger change how mice interact? These are
        exactly the questions a later neural manipulation will also ask — *did it move behavior?* — so
        getting the statistics right here is not busywork.

        The central danger is **pseudoreplication**: treating measurements that are not independent as
        if they were. Sex is a fixed property of a **cage**, and we have only 14 cages. So 2,499 events
        give us 14 independent observations of sex, not 2,499. If we ignore that, a weak, cage-driven
        pattern can masquerade as an overwhelming effect.

        **Definitions.**

        - **Event level** — treat every event as an independent sample. Valid for a variable that
          *varies within* a cage (like food-deprivation phase, which each cage went through), invalid
          for one *fixed across* a cage (like sex).
        - **Cage level** — collapse each cage to one summary value, then compare cages. The honest unit
          for a between-cage variable.
        - **Cohort-unique cage** — each cage id here encodes its cohort (`cohort_index*100 + cam`), so
          grouping by cage never mixes the two cohorts. There are 14 cages, 7 male and 7 female.
        - **Mann-Whitney U test** — `scipy.stats.mannwhitneyu(a, b)` asks whether two groups differ in
          typical value without assuming a bell shape. A small p means they differ.
        - **Permutation test** — build the null distribution by shuffling the label *at the correct
          unit* (here, shuffling sex across the 14 cages) many times and seeing where the real value
          falls. No distributional assumption at all.
        - **Paired Wilcoxon test** — `scipy.stats.wilcoxon(a, b)` compares two *paired* measurements
          (here, each cage's pre value vs its own dep value).

        We work through a **positive control** (a real sex effect that survives the honest test), a
        **negative control** (an apparent sex effect that does not), and a **food-deprivation effect**.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## C.1 · The positive control — heading alignment differs by sex

        `heading_alignment` measures whether the two mice point the same way during the interaction
        (+1 = aligned, -1 = opposed). Split every event by the sex of its cage and look at the raw
        distributions — not a bar of means, but every point.
        """
    )
    return


@app.cell
def _(HI, X, cu, mo, np, sex):
    _fig = cu.violin_points_fig(
        X[:, HI], sex, group_order=["M", "F"],
        colors={"M": "#4c78a8", "F": "#e45756"},
        ylabel="heading_alignment", xlabel="cage sex",
        title="heading_alignment by sex — every event, both cohorts pooled")
    mo.vstack([mo.md("Each dot is one event; the violin is the smoothed distribution, the inner box "
                     "the quartiles. Males sit lower (more opposed headings) than females."),
               _fig])
    return


@app.cell
def _(HI, X, cu, mo, sex):
    _fig = cu.ecdf_fig(X[:, HI], sex, group_order=["M", "F"],
                       colors={"M": "#4c78a8", "F": "#e45756"},
                       xlabel="heading_alignment", title="Same comparison as cumulative curves (ECDF)")
    mo.vstack([mo.md("The two cumulative curves are cleanly separated: at almost every value, a larger "
                     "fraction of male events falls below it. This is the event-level picture, and it "
                     "looks decisive — which is exactly when we should slow down."),
               _fig])
    return


@app.cell(hide_code=True)
def _(HI, X, mo, np, sex):
    from scipy.stats import mannwhitneyu as _mwu
    _p = float(_mwu(X[sex == "M", HI], X[sex == "F", HI])[1])
    _mM, _mF = float(np.median(X[sex == "M", HI])), float(np.median(X[sex == "F", HI]))
    mo.md(
        f"""
        **Event-level test.** Mann-Whitney U on `heading_alignment`, males vs females, gives
        **p = {_p:.1e}** (median M = {_mM:.3f}, F = {_mF:.3f}; Cohen d about 0.23, M &lt; F). Taken at
        face value that is overwhelming. But sex is fixed per cage, so before believing it we must ask
        whether the *cage* — not the event — is the honest unit.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Look at the cages

        Collapse each of the 14 cages to its **mean `heading_alignment`** and color by sex. If the
        effect is genuinely about sex, the seven male cages should sit systematically below the seven
        female cages — not because two extreme cages drag the average, but as a consistent shift.
        """
    )
    return


@app.cell
def _(HI, X, cage, cu, mo, np, sex):
    _ucg = np.unique(cage)
    _cm = np.array([np.nanmean(X[cage == c, HI]) for c in _ucg])       # one mean per cage
    _cs = np.array([sex[cage == c][0] for c in _ucg])                  # one sex per cage
    _fig = cu.strip_points_fig(
        _cm, _cs, group_order=["M", "F"],
        colors={"M": "#4c78a8", "F": "#e45756"}, jitter=0.12, point_size=13,
        hover=[f"cage {c}" for c in _ucg],
        ylabel="cage-mean heading_alignment", xlabel="cage sex",
        title="14 cohort-unique cages, one point each — the honest unit of analysis")
    mo.vstack([mo.md("Now there are only **14 points**. The male cages do sit lower on the whole — but "
                     "with 7 versus 7, is that shift larger than random cage-labelings would produce? "
                     "That is what the permutation test answers."),
               _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The honest test — permute sex across the 14 cages

        The exchangeable unit is the cage. So we build the null by **shuffling the 7 M / 7 F labels
        across the 14 cage means** many times, each time recomputing the male-minus-female gap. If sex
        genuinely drives `heading_alignment`, the real gap sits far in the tail. Here it does: the
        cage-level p is about **0.0094 — it survives**.
        """
    )
    return


@app.cell
def _(HI, X, cage, go, mo, np, sex):
    def _cage_perm(values, sex, cage, n=10000, seed=0):
        rng = np.random.RandomState(seed)
        ucg = np.unique(cage)
        cm = np.array([np.nanmean(values[cage == cg]) for cg in ucg])   # 14 cage means
        cs = np.array([sex[cage == cg][0] for cg in ucg])               # 14 sex labels
        def gap(lab):
            return abs(np.nanmean(cm[lab == "M"]) - np.nanmean(cm[lab == "F"]))
        obs = gap(cs)
        null = np.array([gap(rng.permutation(cs)) for _ in range(n)])
        p = (np.sum(null >= obs - 1e-12) + 1) / (n + 1)
        return obs, float(p), null

    _obs, _p, _null = _cage_perm(X[:, HI], sex, cage)
    _fig = go.Figure()
    _fig.add_histogram(x=_null, nbinsx=30, marker_color="#c7c7c7", name="cage-shuffled null")
    _fig.add_vline(x=_obs, line=dict(color="#e45756", width=3),
                   annotation_text="observed", annotation_position="top")
    _fig.update_layout(template="plotly_white", height=360, showlegend=False,
                       title=f"Cage-level permutation null — observed gap is out in the tail "
                             f"(p = {_p:.4f}, SURVIVES)",
                       xaxis_title="|male cage-mean - female cage-mean|",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    mo.vstack([mo.md(f"**Cage-level p = {_p:.4f}.** The observed gap sits beyond almost all random "
                     f"relabelings. Unlike the pseudoreplicated event-level number, this one respects "
                     f"the 14-cage design — and the sex effect **holds**."),
               _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Does it replicate across the two cohorts?

        A single significant test can still be a fluke of one cohort. The strongest evidence is
        **independent replication**. Split the data by cohort and re-run the event-level Mann-Whitney
        in each: the sex difference in `heading_alignment` shows up in **both**.
        """
    )
    return


@app.cell
def _(HI, X, cohort, cu, mo, np, sex):
    from scipy.stats import mannwhitneyu as _mwu
    _rows = []
    for _co in np.unique(cohort):
        _m = cohort == _co
        _p = float(_mwu(X[_m & (sex == "M"), HI], X[_m & (sex == "F"), HI])[1])
        _rows.append((f"cohort {_co}", _p))
    _fig = cu.strip_points_fig(
        X[:, HI], np.array([f"{c}·{s}" for c, s in zip(cohort, sex)]),
        group_order=[f"{co}·{s}" for co in np.unique(cohort) for s in ["M", "F"]],
        colors={f"{co}·M": "#4c78a8" for co in np.unique(cohort)} |
               {f"{co}·F": "#e45756" for co in np.unique(cohort)},
        point_size=4, opacity=0.5, ylabel="heading_alignment", xlabel="cohort · sex",
        title="heading_alignment by sex, split by cohort — the M<F gap repeats in both")
    mo.vstack([mo.md("Event-level Mann-Whitney within each cohort: "
                     + "; ".join(f"**{name}** p = {p:.1e}" for name, p in _rows)
                     + ". The effect is present in both independent cohorts — replication, not a "
                       "one-cohort artifact."),
               _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## C.2 · The negative control — body size does *not* survive

        Now a feature that should make us suspicious: `appr_body_len`, the approacher's body length.
        Males are larger, so at the event level the sex difference is astronomically significant
        (p on the order of 1e-22). If we stopped at the event level we would "discover" a sex effect on
        body size that is really just **the same 14 cages counted thousands of times**. Run the *same*
        cage-level permutation and it does **not** survive.
        """
    )
    return


@app.cell
def _(BLI, X, cage, cu, mo, np, sex):
    # Left: cage-mean body length by sex (14 points). Compare to the heading-alignment strip above.
    _ucg = np.unique(cage)
    _cm = np.array([np.nanmean(X[cage == c, BLI]) for c in _ucg])
    _cs = np.array([sex[cage == c][0] for c in _ucg])
    _fig = cu.strip_points_fig(
        _cm, _cs, group_order=["M", "F"],
        colors={"M": "#4c78a8", "F": "#e45756"}, jitter=0.12, point_size=13,
        hover=[f"cage {c}" for c in _ucg],
        ylabel="cage-mean appr_body_len", xlabel="cage sex",
        title="Body length per cage — M and F cages overlap heavily")
    mo.vstack([mo.md("At the cage level the male and female body-length means **overlap**: the huge "
                     "event-level significance came from sample size, not from a clean 7-vs-7 "
                     "separation."),
               _fig])
    return


@app.cell
def _(BLI, X, cage, go, mo, np, sex):
    from scipy.stats import mannwhitneyu as _mwu
    def _cage_perm(values, sex, cage, n=10000, seed=0):
        rng = np.random.RandomState(seed)
        ucg = np.unique(cage)
        cm = np.array([np.nanmean(values[cage == cg]) for cg in ucg])
        cs = np.array([sex[cage == cg][0] for cg in ucg])
        def gap(lab):
            return abs(np.nanmean(cm[lab == "M"]) - np.nanmean(cm[lab == "F"]))
        obs = gap(cs)
        null = np.array([gap(rng.permutation(cs)) for _ in range(n)])
        return obs, float((np.sum(null >= obs - 1e-12) + 1) / (n + 1)), null

    _ep = float(_mwu(X[sex == "M", BLI], X[sex == "F", BLI])[1])
    _obs, _p, _null = _cage_perm(X[:, BLI], sex, cage)
    _fig = go.Figure()
    _fig.add_histogram(x=_null, nbinsx=30, marker_color="#c7c7c7", name="null")
    _fig.add_vline(x=_obs, line=dict(color="#e45756", width=3),
                   annotation_text="observed", annotation_position="top")
    _fig.update_layout(template="plotly_white", height=360, showlegend=False,
                       title=f"appr_body_len cage-level null — observed gap is ordinary "
                             f"(p = {_p:.3f}, does NOT survive)",
                       xaxis_title="|male cage-mean - female cage-mean|",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    mo.vstack([mo.md(f"**Event-level p = {_ep:.1e}** (looks overwhelming) but **cage-level p = "
                     f"{_p:.3f}** (does not clear 0.05). Same data, honest unit, opposite verdict."),
               _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The two controls, side by side

        This is the whole lesson of pseudoreplication in one comparison. Both features look
        wildly significant at the event level. Only one survives when the cage is the unit.

        | Feature | Event-level p | Cage-level p | Verdict |
        |---|---|---|---|
        | `heading_alignment` (positive control) | ~6.5e-9 | **~0.009** | **survives** — a real, replicated sex effect |
        | `appr_body_len` (negative control) | ~5.4e-22 | ~0.078 | does **not** survive — pseudoreplication |

        The event-level p-value tells you almost nothing about which is real; a *smaller* event-level p
        (body size) is the one that fails. What matters is whether the effect is consistent across the
        14 independent cages. And note the reassuring null: aggression *rate* by sex is not significant
        at either level (event chi-square p about 0.60) — we are not claiming aggression is a male
        behavior.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## C.3 · Food deprivation increases bystander distance

        Food-deprivation **phase** (`pre`/`dep`/`post`) is different from sex: every cage was recorded
        in all three phases, so phase *varies within* a cage and the event level is not automatically
        pseudoreplicated. Still, the honest confirmation is a **paired** cage-level test — each cage
        compared to itself.

        The readout is `bystander_dist_mean`: how far the third (bystander) mouse sits from the
        interacting pair. Under deprivation the bystander keeps its distance.
        """
    )
    return


@app.cell
def _(BDI, X, cond, cu, mo):
    _fig = cu.violin_points_fig(
        X[:, BDI], cond, group_order=["pre", "dep", "post"],
        colors={"pre": "#54a24b", "dep": "#e45756", "post": "#4c78a8"},
        ylabel="bystander_dist_mean (px)", xlabel="deprivation phase",
        title="bystander_dist_mean by phase — every event")
    mo.vstack([mo.md("The `dep` distribution is shifted upward relative to `pre`: the bystander mouse "
                     "sits farther from the interacting pair when the cage is food-deprived."),
               _fig])
    return


@app.cell(hide_code=True)
def _(BDI, X, cage, cond, go, mo, np):
    from scipy.stats import mannwhitneyu as _mwu, wilcoxon as _wil
    _ep = float(_mwu(X[cond == "dep", BDI], X[cond == "pre", BDI])[1])
    _mpre, _mdep = float(np.median(X[cond == "pre", BDI])), float(np.median(X[cond == "dep", BDI]))
    _ucg = np.unique(cage)
    _pre = np.array([np.nanmean(X[(cage == c) & (cond == "pre"), BDI]) for c in _ucg])
    _dep = np.array([np.nanmean(X[(cage == c) & (cond == "dep"), BDI]) for c in _ucg])
    _wp = float(_wil(_dep, _pre)[1])
    # paired dumbbell: each cage's pre -> dep
    _fig = go.Figure()
    for _k, _c in enumerate(_ucg):
        _fig.add_scatter(x=["pre", "dep"], y=[_pre[_k], _dep[_k]], mode="lines+markers",
                         line=dict(color="#bbb", width=1), marker=dict(size=8, color="#4c78a8"),
                         showlegend=False, hovertemplate=f"cage {_c}: %{{y:.0f}} px<extra></extra>")
    _fig.update_layout(template="plotly_white", height=400,
                       title=f"Each cage vs itself, pre -> dep (paired Wilcoxon p = {_wp:.4f})",
                       yaxis_title="cage-mean bystander_dist_mean (px)",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    mo.vstack([mo.md(f"**Event-level Mann-Whitney p = {_ep:.1e}** (median {_mpre:.0f} -> {_mdep:.0f} "
                     f"px). **Cage-level paired Wilcoxon p = {_wp:.4f}**, mean shift "
                     f"+{np.mean(_dep - _pre):.0f} px. Most cages move the same direction — the "
                     f"deprivation effect is real at the honest, paired unit, and it replicates in "
                     f"both cohorts. (Aggression *rate* pre vs dep is null, p about 0.29.)"),
               _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## C.4 · Exercise — write the cage-level permutation test yourself

        **Python skill practiced.** *Writing a permutation-test loop* — the single most transferable
        statistical tool in this course. You will collapse events to cages, compute an observed
        statistic, and build a null by shuffling labels **at the cage level** inside a loop.

        **Why the shuffle level matters (the scientific point).** If you shuffle sex across *events*,
        you destroy nothing — every event in a male cage is still male — and you reproduce the
        pseudoreplicated, wildly-significant event-level p. Shuffling across the **14 cages** is what
        makes the test honest: it asks "could a random 7-vs-7 split of cages produce a gap this big?"

        **Your task.** One line is blanked. Fill it so the loop shuffles the **cage** sex labels.
        """
    )
    return


@app.cell
def _(np):
    def cage_perm_p(values, sex, cage, n=10000, seed=0):
        rng = np.random.RandomState(seed)
        ucg = np.unique(cage)
        # Collapse each cage to ONE mean value and ONE sex label — the honest 14-unit view.
        cage_mean = np.array([np.nanmean(values[cage == cg]) for cg in ucg])   # (14,)
        cage_sex  = np.array([sex[cage == cg][0] for cg in ucg])               # (14,)
        # The statistic: absolute gap between the male-cage mean and the female-cage mean.
        def gap(labels):
            return abs(np.nanmean(cage_mean[labels == "M"]) - np.nanmean(cage_mean[labels == "F"]))
        obs = gap(cage_sex)                              # the REAL, observed gap
        hits = 0
        for _ in range(n):
            # -------------------- FILL IN THE BLANK --------------------
            # Build one random relabeling of the 14 cages. We must shuffle the CAGE sex labels
            # (`cage_sex`), NOT the per-event sex — shuffling events would leave every cage's sex
            # unchanged and give back the dishonest event-level answer. `np.random.RandomState`'s
            # `.permutation(x)` returns a shuffled COPY of the array x, keeping 7 M / 7 F fixed.
            # Replace ____ with:  rng.permutation(cage_sex)
            perm = ____                                  # <-- EDIT THIS LINE
            # -----------------------------------------------------------
            if gap(perm) >= obs - 1e-12:                 # count relabelings at least as extreme
                hits += 1
        p_emp = (hits + 1) / (n + 1)                     # empirical p-value (+1 smoothing)
        return float(obs), float(p_emp)

    return (cage_perm_p,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Reveal solution": mo.md(
            r"""
            Shuffle the **cage** labels, not the events:

            ```python
            perm = rng.permutation(cage_sex)
            ```

            Then `heading_alignment` gives `p_emp` about **0.0094** (survives) and `appr_body_len`
            gives about **0.078** (does not). If you had instead written `rng.permutation(sex)` (event
            level) both would come back near zero — the pseudoreplication trap this whole part is about.
            """)
    })
    return


@app.cell(hide_code=True)
def _(HI, BLI, X, cage, cage_perm_p, mo, sex):
    try:
        _h_obs, _h_p = cage_perm_p(X[:, HI], sex, cage)      # positive control -> should SURVIVE
        _b_obs, _b_p = cage_perm_p(X[:, BLI], sex, cage)     # negative control -> should NOT survive
        _c1 = 0.003 <= _h_p <= 0.02                          # heading survives (~0.009)
        _c2 = 0.04 <= _b_p <= 0.12                           # body size fails (~0.078)
        _ok = _c1 and _c2
    except Exception as _e:
        _ok = False
        _h_p = _b_p = float("nan")
        _c1 = _c2 = False
    def _row(ok, txt):
        return f"<div>{'✅' if ok else '❌'} {txt}</div>"
    if _ok:
        _bg, _icon, _verdict = "#eafaef", "PASS", (
            "You reproduced the real result: <b>heading_alignment survives</b> the cage-level test "
            "(a genuine, replicated sex effect) while <b>body size does not</b> (pseudoreplication). "
            "The event level could not tell these apart — the honest unit could.")
    else:
        _bg, _icon, _verdict = "#fdeaea", "NOT YET", (
            "If both p-values came back near 0, you are still shuffling events — set "
            "<code>perm = rng.permutation(cage_sex)</code> to shuffle the 14 cages.")
    _body = (_row(_c1, f"heading_alignment cage-level p = {_h_p:.4f} (survives; band 0.003-0.02)")
             + _row(_c2, f"appr_body_len cage-level p = {_b_p:.3f} (does NOT survive; band 0.04-0.12)"))
    mo.md(f"<div style='border:1px solid #ccc;border-radius:8px;padding:10px 12px;background:{_bg}'>"
          f"<b>Self-check — {_icon}</b>{_body}<div style='margin-top:6px'>{_verdict}</div></div>")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Conceptual questions

        1. **Why does shuffling events give the wrong p?** Every event in a male cage stays male under
           an event shuffle, so the "null" still has the real cage structure baked in. You are testing
           against a null that already contains the effect, which is why it looks certain.
        2. **A feature has event-level p = 1e-30 but cage-level p = 0.4. Real or not?** Not
           demonstrated. A tiny event-level p only says the *events* differ; if the split is really
           7-vs-7 cages, 14 is your sample size and 0.4 is the honest answer.
        3. **Why is food-deprivation phase testable within a cage but sex is not?** Every cage
           contributes pre, dep, and post events, so phase is not confounded with cage identity. Sex is
           a fixed cage property, so a "sex" test is a 7-vs-7 cage comparison in disguise.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Limits of these methods

        - **PCA is linear and variance-greedy.** It can rotate and stretch but not unbend a curved
          shape; a behavior that curls through feature space is smeared across many PCs. Aggression is
          partly on PC1 but never cleanly isolated by any single component.
        - **The map discards time and rare behaviors.** Each event becomes one frozen dot, and any
          behavior too rare to reach `min_cluster_size` dissolves into noise. The map also inherits
          every earlier modeling choice (which features, how scaled).
        - **The map's geometry is a setting, not a fact.** Distance between clusters, cluster area, and
          even the number of clusters all move with `n_neighbors`, `min_dist`, and `min_cluster_size`.
          Trust local neighborhoods, plus a statistical test, not the picture's global shape.
        - **Small samples cap what we can claim.** With 14 cages (7 vs 7), a real sex effect must be
          fairly strong to clear the permutation test; a true weak effect could hide. Surviving here is
          strong evidence; failing to survive (body size) is not proof of no effect, only of no effect
          *at this power*.
        """
    )
    return


@app.cell(hide_code=True)
def _(best_cluster, best_lift, dim90, mo):
    mo.md(
        f"""
        ## What we reached, and what comes next

        In one long analysis we answered all three questions we opened with.

        - **(A) How many dimensions?** About 6-to-{dim90}: 6 PCs keep ~71% of the variance, {dim90}
          reach 90%. PC1 is the activity axis and carries real aggression signal, so we drop it only to
          shape the map, never to decode.
        - **(B) What types exist?** An unsupervised map concentrates aggression into
          **C{best_cluster}** at a **{best_lift:.2f}x** lift, and the feature profile tells us why —
          real, recurring, but impure.
        - **(C) What changes behavior?** `heading_alignment` differs by sex and **survives** the
          cage-level permutation test in both cohorts (positive control); body size does **not**
          survive (negative control, pseudoreplication); and food deprivation reliably increases
          `bystander_dist_mean` (paired cage-level Wilcoxon). We report what the 14-cage design can
          actually support.

        > **Next question (NB05):** the map's aggression recovery was real but only ~38% pure. Can a
        > *supervised* decoder, trained on labels, read aggression far more sharply — and, crucially,
        > does it generalize to a **cohort it never trained on**? That leave-one-cohort-out test is the
        > honest measure of whether we have learned behavior or just memorized cages.
        """
    )
    return


if __name__ == "__main__":
    app.run()
