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
    # ---- canonical data loads (the loaders fetch on a bare kernel) --------------------------------
    ev    = cu.load_events(cu.data_path("data/train_events.npz", ROOT))     # kp, ranks, condition, agg
    der   = cu.load_derived("train", ROOT)                                  # X, PCA, cage/sex/cohort
    sweep = cu.load_umap_sweep(ROOT)                                        # precomputed embeddings + labels

    kp     = ev["kp"]                                                       # (N,T,3,15,2) for GIFs
    ranks  = ev["ranks"]                                                    # (N,3) rank per ordered mouse
    crel   = ev["contact_rel"].astype(int)                                  # contact frame per event
    yagg   = ev["agg_label"].astype(int)                                    # 1 = aggression (ground truth)
    cats   = ev["category"].astype(str)                                     # registry label if any

    X      = der["X"]                                                       # (N,19) allocentric features
    fn     = [str(f) for f in der["feature_names"]]                         # 19 feature names

    # Standardize the 19 features, then refit a FULL-RANK PCA so the scree curve can reach 90% and
    # so we can look at every principal component, not just the shipped 10.
    Xz, _mu, _sd = cu.standardize(X)                                        # z-score each feature
    sc, evr, _pca = cu.pca_scores(Xz, 19)                                   # sc (N,19), evr (19,)
    comp   = _pca.components_                                               # (19,19) loadings
    cumvar = np.cumsum(evr)
    dim90  = int(np.searchsorted(cumvar, 0.90) + 1)                         # smallest k with >=90% var
    cum6   = float(cumvar[5])                                               # variance kept by first 6 PCs

    # per-PC standardized scores -> the |z| of every event on every axis (used to hunt the tails)
    zsc    = (sc - sc.mean(0)) / sc.std(0)                                  # (N,19)

    # A speed-spike detector: the larger of the two "peak speed" features per event. Frame-to-frame
    # displacement is exactly what a teleport / dropout inflates, so this flags tracking artifacts.
    smax   = np.maximum(X[:, fn.index("appr_speed_max")], X[:, fn.index("appe_speed_max")])
    gate_idx = np.where(smax > 250.0)[0]                                    # the ~9 speed-spike events

    # The canonical behavioral map + syllables (every downstream analysis agrees on THESE).
    emb0    = sweep["emb_grid"][tuple(sweep["default_ij"])]                 # pinned 2-D embedding (N,2)
    clabels = sweep["default_labels"].astype(int)                          # canonical clusters (-1=noise)
    base_rate = float(yagg.mean())                                         # 0.320

    # per-cluster aggression fraction -> the purest (highest-fraction) cluster and its lift.
    # Gate out tiny clusters first: a size-50 cluster's fraction is so noisy that it can top the
    # ranking by luck, so the "purest" pick flips between rebuilds. Only clusters with >=100 events
    # are eligible to be named (here that drops C0, n=53; C3 stays the purest either way).
    MIN_CLUSTER_N = 100
    _cs = sorted(c for c in set(clabels.tolist()) if c >= 0)
    _gated = [c for c in _cs if (clabels == c).sum() >= MIN_CLUSTER_N]
    _fr = {c: float(yagg[clabels == c].mean()) for c in _gated}
    best_cluster = max(_fr, key=_fr.get)                                   # 3 (stable under the gate)
    best_frac = _fr[best_cluster]                                          # 0.381
    best_lift = best_frac / base_rate                                      # 1.19x

    N       = int(len(X))                                                  # 2499 training events
    EXAMPLE = cu.event_index_by_key(
        ev, "12192025_pre|cam.10.00046-2025-12-18T16|m0-m2|83141")        # example: cage 110 'pre', NON-agg

    # canonical cluster palette (c3 = aggression-enriched -> red)
    CPAL = {-1: "#d5d5d5", 0: "#8c9196", 1: "#f2b134", 2: "#4c78a8", 3: "#e45756"}
    return (CPAL, EXAMPLE, MIN_CLUSTER_N, N, Xz, base_rate, best_cluster, best_frac, best_lift,
            cats, clabels, comp, crel, cum6, cumvar, der, dim90, emb0, ev, evr,
            fn, gate_idx, kp, np, ranks, sc, smax, sweep, yagg, zsc)


@app.cell(hide_code=True)
def _(N, base_rate, mo):
    mo.md(
        rf"""
        # NB04 · Collapsing behavior to a manifold

        ## Where we are, and what we ask here

        In the previous notebook we treated a single interaction as a signal over time and asked which
        of our measurements move together. This notebook zooms out from one event to the whole corpus:
        **{N:,} approach events**, each already reduced to **19 allocentric numbers** — closing speed,
        pair distance, mutual facing, body length, and so on. Nineteen numbers per event is workable,
        but it raises two questions we have not yet answered.

        - **(A) How many dimensions does behavior actually have?** The 19 features are correlated, so
          the true number of independent directions is smaller. We answer this with **PCA**, and we
          learn that "how many dimensions" is a *choice*, not a single fact. Along the way we meet a
          data-hygiene trap: the most extreme events on the higher components turn out to be tracking
          failures, not rare behaviors.
        - **(B) What behavioral types exist?** Without ever telling a method which events are
          aggression, can it find recurring *kinds* of behavior on its own? We build a 2-D **map** with
          UMAP, look inside the algorithm that makes it, cluster the map, and give its regions meaning
          by reading them back to the 19 features and to video.

        Our corpus has an aggression base rate of **{base_rate:.0%}** — aggression is a minority we are
        trying to recover, not the default. A separate 780 events from one held-out camera are sealed
        away and never touched here.

        **A word on terms.** A *manifold* is the low-dimensional surface that the data actually lives
        on, even when each point is written with many numbers: a sheet of paper is two-dimensional even
        after you crumple it into three-dimensional space. Most of this notebook is about finding that
        surface — first with a straight (linear) tool, then with a bendable (nonlinear) one.

        Each part ends by stating the answer it reached and the question it hands forward.
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
          the original 19 features; an event's *score* on a PC is how far along that axis it sits. We
          number them from **PC1** (most spread) upward.
        - **Variance explained** — the fraction of the total spread a component accounts for. If PC1
          explains 18% of the variance, it captures 18% of the differences between events.
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
        first principal component is exactly the direction you were hunting for. Projected variance is
        small across the short width of the ellipse and largest along its long diagonal.

        **The lesson this control teaches.** The angle is a knob you are turning *by hand* to maximize a
        real number (projected variance). PCA turns exactly this knob automatically. The answer is not
        arbitrary and not a setting you choose — it is *learned from the data*: the covariance of the
        cloud fixes the winning angle.
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
        Standardize the data to $X$ ($n$ events × 19 features, each column mean-0). The covariance is
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
        total (cumulative variance) as you add components left to right. ("Scree" is the loose rock at
        the base of a cliff — the bars fall off like a talus slope.)

        **Drag `keep k`** to highlight the first *k* components and read the cumulative percentage in the
        title. The bars fall off quickly — the first few are tall, the rest are short — and the
        cumulative line rises steeply then flattens.

        **The lesson this control teaches.** There is no single "right" *k*. Two common rules disagree:
        the **elbow** (stop where the bars stop falling steeply, ~6 here) keeps far fewer components
        than a **variance threshold** (keep enough for 90%, which needs more). *k* is a choice with a
        stated criterion, not a fact read off the data.
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
    _fig.update_xaxes(showgrid=False)
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
        heatmap shows the loadings of the top 6 components: **red** pushes an event's score *up* along
        that PC, **blue** pushes it *down*. Reading across a row tells you which real behaviors that
        component combines — it translates an abstract axis back into behavior. (The rows are labelled
        PC1…PC6 from the top, matching the numbering we use everywhere.)
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
        next section shows, in numbers, why that removal is a *choice with a cost*.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### See it — behavior at the two ends of a component

        Loadings describe a component in words; GIFs let you see it. Pick a component and watch the
        events that score **lowest** versus **highest** on it. PC1 is the *amount-of-activity* axis, so
        its two ends are low-motion versus high-motion pairs: one end collects calm, mostly stationary
        pairs and the other collects fast, actively engaging ones. In this run PC1 loads *negative* on
        the speed features, so the **low** end is the fast, engaging pairs and the **high** end is the
        calm, stationary ones — read the speeds off the GIFs to confirm which way this fit landed (the
        overall sign of a principal component is arbitrary, so a rebuild can swap the two ends without
        changing the axis).

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
def _(crel, cu, kp, mo, np, pc_pick, ranks, sc):
    # Sort every event by its score on the chosen PC, then render the 3 lowest and 3 highest as
    # skeleton GIF grids. This makes the axis concrete: what behavior sits at each end.
    _pc = int(pc_pick.value[2:]) - 1
    _order = np.argsort(sc[:, _pc])                       # ascending score on this PC
    _low = _order[:3]                                     # 3 events lowest on this axis
    _high = _order[-3:]                                   # 3 events highest on this axis
    _lo = cu.grid_gif_bytes([(kp[i], ranks[i], int(crel[i])) for i in _low], ncols=3, cell=150)
    _hi = cu.grid_gif_bytes([(kp[i], ranks[i], int(crel[i])) for i in _high], ncols=3, cell=150)
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
        ## A.4 · Which components carry aggression? Leave-one-PC-out

        Before we build the map (Part B) we will deliberately set aside the large "how fast / how close"
        axis so the map reflects finer differences instead of raw activity level. But is that axis
        really disposable for reading behavior? The honest way to ask is: *how much does the ability to
        tell aggression from non-aggression fall if we delete one PC and keep the other 9?*

        We answer with a **leave-one-PC-out** experiment. For each component we remove it, fit a simple
        logistic decoder on the remaining components, and score how well it separates aggression from
        non-aggression. The score we use is the **AUROC**, defined next.

        **What AUROC means (read before the numbers).** The decoder does not output a hard yes/no; it
        outputs a **score** per event — a number, higher when the event looks more like aggression. To
        turn scores into calls you pick a **threshold**: events scoring above it are called
        "aggression," the rest "not." A strict (high) threshold flags few events; a lax (low) one flags
        many. Every threshold has two consequences worth tracking:

        - the **true-positive rate (TPR)** — the fraction of the real aggression events the decoder
          correctly flags (also called recall);
        - the **false-positive rate (FPR)** — the fraction of the non-aggression events it wrongly
          flags.

        Sweeping the threshold from strict to lax and plotting TPR (y) against FPR (x) traces the
        **ROC curve** (receiver operating characteristic). The **AUROC is the area under that curve**.
        It has a direct reading: it is the probability that a randomly chosen aggression event is given
        a higher score than a randomly chosen non-aggression event. A value of **0.5 means the scores
        carry no information** (the coin-flip diagonal), and **1.0 means perfect ranking** (every
        aggression event outscores every non-aggression event).

        **Why AUROC and not plain accuracy here.** Aggression is a minority — about a third of these
        events. A lazy decoder that labels *everything* "not aggression" would still be right on the
        roughly two-thirds that are not aggression, scoring near 68% accuracy while detecting nothing.
        AUROC is immune to that trap: it depends only on how scores are *ranked*, not on the class
        balance and not on any single threshold choice. That threshold-independence and robustness to
        imbalance is why AUROC is the standard measure for a minority-class detection problem like this.

        The AUROC below is 5-fold cross-validated, so it reflects genuine prediction, not memorization.
        A bar that dips **below** the dashed all-PCs line marks a component that was carrying signal.
        """
    )
    return


@app.cell
def _(cu, sc, yagg):
    cu.loo_pc_auroc_fig(sc[:, :10], yagg,
                        title="Leave-one-PC-out aggression AUROC — which axes carry the signal")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **PC1 is the one that matters.** Removing it drops the AUROC by roughly 0.19 — from about 0.84
        to about 0.65 — while removing any single higher component barely moves the number. Aggression
        is partly a high-motion behavior, so it lives substantially on the activity axis. That is the
        whole point: **PC1 carries real aggression signal, so we drop it only to shape the map in Part
        B, never when we actually decode** (that decoder is NB05's job).

        You can confirm the trade-off directly. Choose PCs to zero out and read the cross-validated
        AUROC on whatever remains.
        """
    )
    return


@app.cell
def _(mo):
    drop_sel = mo.ui.multiselect(options=[f"PC{i+1}" for i in range(10)], value=["PC1"],
                                 label="drop these PCs, then decode on the rest", full_width=True)
    return (drop_sel,)


@app.cell
def _(cu, drop_sel, mo, sc, yagg):
    from sklearn.model_selection import cross_val_score as _cvs
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.pipeline import make_pipeline as _mkp
    from sklearn.preprocessing import StandardScaler as _SS

    _drop = [int(s[2:]) - 1 for s in drop_sel.value]                  # "PC1" -> index 0
    _res = cu.residualize(sc[:, :10], _drop)
    def _auc(S):
        return float(_cvs(_mkp(_SS(), _LR(max_iter=1000)), S, yagg, cv=5, scoring="roc_auc").mean())
    _full = _auc(sc[:, :10])
    _kept = _auc(_res)
    _msg = (f"<div style='font-size:1.05em'>Aggression AUROC — all 10 PCs: <b>{_full:.3f}</b> "
            f"&nbsp;-&gt;&nbsp; after dropping {', '.join(drop_sel.value) or 'nothing'}: "
            f"<b>{_kept:.3f}</b></div>"
            f"<div style='color:#666;font-size:.9em;margin-top:6px'>Chance = 0.500. Dropping PC1 "
            f"weakens but does not erase the signal — a decision with a real trade-off, not a free "
            f"cleanup. Dropping a higher PC costs almost nothing.</div>")
    mo.vstack([drop_sel, mo.md(_msg)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## A.5 · Looking past PC1–PC2 — the higher components

        PC1 and PC2 are the axes everyone plots, because they hold the most variance. But we just saw
        that aggression signal is spread thinly across several components, so it is worth *looking* at
        the higher ones. Do PC3–PC6 carve the data into separable pockets, or are they featureless?

        Below, pick a pair of higher components and scatter every event, colored by whether it is
        aggression. The axes are **robust-clipped** by default (trimmed to the 1st–99th percentile) —
        for a reason we are about to expose. Read the marginal histograms on the top and right as the
        one-dimensional shape of each axis.
        """
    )
    return


@app.cell
def _(mo):
    pcpair_pick = mo.ui.dropdown(options={"PC3 vs PC4": "2,3", "PC5 vs PC6": "4,5",
                                          "PC7 vs PC8": "6,7"},
                                 value="PC5 vs PC6", label="which higher-PC pair to view")
    outlier_reveal = mo.ui.switch(value=False,
                                  label="Reveal the raw auto-ranged view (expose the outliers)")
    return outlier_reveal, pcpair_pick


@app.cell
def _(cu, gate_idx, go, mo, np, outlier_reveal, pcpair_pick, sc, smax, yagg):
    _a, _b = (int(s) for s in pcpair_pick.value.split(","))
    _x, _y = sc[:, _a], sc[:, _b]
    _lab = np.where(yagg == 1, "aggression", "not agg")
    _fig = go.Figure()
    for _g, _c in [("not agg", "#b6bac1"), ("aggression", "#d62728")]:
        _m = _lab == _g
        _fig.add_scattergl(x=_x[_m], y=_y[_m], mode="markers", name=_g,
                           marker=dict(size=5, opacity=0.55, color=_c, line=dict(width=0)))
    if outlier_reveal.value:
        # auto-range (no clip) + ring the speed-spike events so the stretch is traceable to them
        _fig.add_scattergl(x=_x[gate_idx], y=_y[gate_idx], mode="markers",
                           name="speed-spike events",
                           marker=dict(size=13, color="rgba(0,0,0,0)",
                                       line=dict(color="#111", width=2.2)))
        _worst = gate_idx[int(np.argmax(smax[gate_idx]))]
        _fig.add_annotation(x=_x[_worst], y=_y[_worst], text="teleport", showarrow=True,
                            arrowhead=2, ax=-40, ay=-30, font=dict(color="#111"))
        _title = (f"PC{_a+1} vs PC{_b+1} — RAW auto-range: a few speed-spike events stretch the axes "
                  f"and squash the bulk")
    else:
        _rx, _ry = cu.robust_range(_x), cu.robust_range(_y)
        if _rx:
            _fig.update_xaxes(range=_rx)
        if _ry:
            _fig.update_yaxes(range=_ry)
        _title = f"PC{_a+1} vs PC{_b+1} — robust axes: a round blob, aggression fully intermixed"
    _fig.update_layout(template="plotly_white", height=520, title=_title,
                       xaxis_title=f"PC{_a+1}", yaxis_title=f"PC{_b+1}",
                       margin=dict(l=10, r=10, t=60, b=10))
    _fig.update_xaxes(showgrid=False)
    _fig.update_yaxes(showgrid=False)
    mo.vstack([mo.hstack([pcpair_pick, outlier_reveal], justify="start"), _fig])
    return


@app.cell(hide_code=True)
def _(gate_idx, mo, smax):
    mo.md(
        f"""
        **What the toggle reveals.** With robust axes the higher-PC scatters are round, roughly Gaussian
        blobs, and the red aggression points are sprinkled evenly through them — no separated island.
        Flip the reveal switch and the picture on `PC5 vs PC6` changes completely: a **single event**
        near PC5 ≈ 13 (and its companion near PC6 ≈ 8) drags the axes out to those values, compressing
        the entire bulk — which lives inside ±3 — into a tiny smear in the corner. The ringed markers
        are the **{len(gate_idx)} events** (out of ~2,500, about 0.4%) whose peak speed exceeds 250
        px/frame. That handful, not the behavior, is what defines the tails of the higher components.

        The largest, labelled *teleport*, jumps at **{smax[gate_idx].max():.0f} px/frame** — most of an
        arena width in a single frame. No mouse moves that fast. This is a tracking failure, and the
        next cell traces it back to the raw keypoints.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## A.6 · Who are the outliers? Tracing the tails back to the video

        An extreme score on a principal component is tempting — it *looks* like a rare, distinctive
        behavior. Far more often it is bad data. Here we take the speed-spike events flagged above and
        render their skeletons. Watch for two failure modes:

        - **Teleport** — a mouse's whole body jumps hundreds of pixels in one frame (the tracker
          latched onto the wrong animal or a reflection), then snaps back.
        - **Dropout** — keypoints vanish for several frames (occlusion or a missed detection); the
          centroid, averaged over whatever nodes survive, whips across the arena and back.

        Both inflate frame-to-frame displacement, which is exactly how `appr_speed_max` / `appe_speed_max`
        are computed — so both masquerade as "very fast" events and land in the tails of the higher PCs.
        """
    )
    return


@app.cell
def _(cats, crel, cu, gate_idx, kp, mo, np, ranks, smax, yagg):
    # Render the worst speed-spike events (data-derived, not hardcoded indices) so students SEE the
    # teleports/dropouts that define the higher-PC tails.
    _order = gate_idx[np.argsort(-smax[gate_idx])][:6]              # up to 6 worst, highest speed first
    _nanf = {i: float(np.isnan(kp[i]).any(-1).mean()) for i in _order}
    _gif = cu.grid_gif_bytes([(kp[i], ranks[i], int(crel[i])) for i in _order], ncols=3, cell=150)
    _rows = " · ".join(
        f"peak {smax[i]:.0f}px/fr, {_nanf[i]:.0%} missing"
        + (f", label “{cats[i]}”" if cats[i] else "")
        for i in _order)
    _cap = (f"**The {len(_order)} worst speed-spike events**, highest peak speed first. "
            f"Corpus-wide, a mouse's centroid moves ~70 px between frames; these hit hundreds. "
            f"Per event — {_rows}.")
    mo.vstack([mo.md(_cap), mo.md(cu.gif_img_html(_gif, width=490))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The fix is not just a better axis range — it is better data

        Robust-clipping the *view* makes the scatter legible, but it leaves the bad events in the PCA
        that produced the axes. The honest fix is to handle the artifacts before fitting: either
        **drop** the flagged events or **winsorize** (cap) the speed features at a high percentile so a
        teleport counts as "fast" rather than "impossibly fast." Below we refit PCA both ways and read
        off how far the extreme tail shrinks.
        """
    )
    return


@app.cell
def _(X, cu, fn, go, mo, np, smax, zsc):
    # Refit PCA after gating (drop speed-spike events) and after winsorizing the two speed_max columns.
    _keep = smax <= 250.0
    _Xz_g, _, _ = cu.standardize(X[_keep])
    _sc_g, _, _ = cu.pca_scores(_Xz_g, 19)
    _zg = (_sc_g - _sc_g.mean(0)) / _sc_g.std(0)

    _Xw = X.copy()
    for _si in (fn.index("appr_speed_max"), fn.index("appe_speed_max")):
        _Xw[:, _si] = np.minimum(_Xw[:, _si], np.nanpercentile(_Xw[:, _si], 99))
    _Xz_w, _, _ = cu.standardize(_Xw)
    _sc_w, _, _ = cu.pca_scores(_Xz_w, 19)
    _zw = (_sc_w - _sc_w.mean(0)) / _sc_w.std(0)

    _x = [f"PC{i+1}" for i in range(10)]
    _fig = go.Figure()
    _fig.add_bar(x=_x, y=np.abs(zsc).max(0)[:10], name="original (with artifacts)",
                 marker_color="#e45756")
    _fig.add_bar(x=_x, y=np.abs(_zg).max(0)[:10], name="after dropping the spikes",
                 marker_color="#4c78a8")
    _fig.add_bar(x=_x, y=np.abs(_zw).max(0)[:10], name="after winsorizing speed",
                 marker_color="#54a24b")
    _fig.update_layout(template="plotly_white", height=420, barmode="group",
                       title="Most extreme event on each PC (|z| of the top score) — before vs after cleanup",
                       yaxis_title="max |z| score on that PC",
                       margin=dict(l=10, r=10, t=60, b=10),
                       legend=dict(orientation="h", y=-0.18))
    _fig.update_xaxes(showgrid=False)
    mo.vstack([
        mo.md("The red bars are the original PCA: **PC5 has an event at |z| ≈ 10.5 and PC9 one at "
              "|z| ≈ 13** — grotesque tails for standardized scores, which should rarely pass |z| ≈ 4. "
              "Both cleanups pull the artifact-driven PC5 and PC9 tails back under ≈ 5, and the "
              "variance explained barely changes (the spikes added spurious spread without adding "
              "behavior). PC6 keeps a genuinely heavy tail (|z| ≈ 7.7 either way) — a real, if "
              "extreme, event, not a tracking failure; the cleanup targets the artifacts and leaves it "
              "alone, which is exactly what we want."),
        _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This is the same data-quality thread that runs through the whole course: NB01 measured that the
        tracker mislabels identity on roughly one crop in six, and NB05 will show how leakage and
        double-dipping inflate results. Here the lesson is concrete and visual: **an extreme principal-
        component score is more often a tracking failure than a rare behavior.** Gate or winsorize
        before you trust a tail.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## A.7 · Why PCA runs out — and what we need next

        Step back and look at what the higher components gave us. With robust axes (and even after
        cleaning the artifacts), **PC3-vs-PC4 and PC5-vs-PC6 are round, structureless blobs with
        aggression fully intermixed.** There is no linear 2-D slice past PC1–PC2 that pulls behavioral
        types apart.

        That is not a failure of effort; it is a property of the method. PCA can only **rotate and
        stretch** the cloud — it draws straight axes ranked by variance. If the interesting structure is
        a curved, thin sheet folded through the 19-dimensional space, or a web of weak nonlinear
        relationships, no single straight axis will expose it. The variance-ranked linear directions
        simply run out of things to say.

        ### Answer to Question A

        Behavior is roughly **6-to-11 dimensional**: 6 PCs keep ~71% of the variance, 11 reach 90%. PC1
        is the overall-activity axis and it carries genuine aggression signal, so "dimensionality" is a
        choice and the biggest axis is not a nuisance. And the higher components, once we strip out
        tracking artifacts, hold no visible clusters under linear projection.

        > **Next question (Part B):** if a straight tool cannot bend to the shape of behavior, can a
        > *nonlinear* one? Part B builds a map that is allowed to curve — and asks whether recurring
        > *kinds* of behavior, including aggression, separate out on their own, without any labels.
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
        mouse is doing, and its straight axes could not separate behavioral types. Here we take a
        different stance: instead of imposing axes, we let the data show us which **kinds of behavior
        recur**, without deciding in advance what to look for. We lay every event out as one point on a
        2-D map, group nearby points into types, and check whether one type corresponds to aggression —
        a category we never named.

        **Definitions.**

        - **Unsupervised** — a method that looks only at the features and groups by similarity, never
          shown the labels. (Supervised learning, trained on labels, comes in NB05.)
        - **Embedding / 2-D map** — a procedure that places each high-dimensional event at an (x, y)
          position so similar events land near each other. Our tool is **UMAP** (Uniform Manifold
          Approximation and Projection). One dot is one whole interaction.
        - **Nonlinear** — unlike PCA, UMAP may bend and stretch different regions of the space by
          different amounts, so it can flatten a curved manifold that no straight axis could.
        - **Clustering** — grouping the dots so dense pockets become named groups; sparse points can be
          left as "noise." Our tool is **HDBSCAN**.
        - **Behavioral type ("syllable")** — one recurring group the clustering finds. "Syllable" is
          borrowed from the behavior literature; treat it as a synonym for a data-driven cluster of
          similar events.

        Before we trust the map, we open the algorithm that makes it, so that every feature of the
        picture can be traced to something the method does.
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
        second.

        **The recipe UMAP follows.**

        1. In the original high-dimensional space, convert distances into **fuzzy neighbor
           memberships**: for each point, its true nearest neighbors get a membership near 1, and the
           membership falls off smoothly with distance. This is the graph UMAP wants to reproduce.
        2. In the 2-D layout, define a **low-D similarity** between points, $q_{ij} = 1/(1+d_{ij}^2)$,
           that is near 1 when two points are close and near 0 when far.
        3. Move the 2-D points to make $q$ match the high-D memberships $P$, minimizing a **fuzzy
           cross-entropy**. The gradient splits into an **attractive** force (true neighbors pull
           together) and a **repulsive** force (everything else pushes apart).

        To make each step concrete, we first work the smallest possible case by hand — three points —
        and then scale up to a cloud of ninety.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The smallest possible case — three points, ten dimensions each

        Take exactly **three points**, each described by **ten numbers** (a 10-dimensional space).
        Points 0 and 1 are placed close together (a distance of 1 apart); point 2 is placed far away (a
        distance of 8). This is the whole story UMAP has to reproduce: two neighbors and one outlier.

        Step 1 turns those high-dimensional distances into **membership** values $P$ — a number near 1
        for a close pair and near 0 for a far pair. Steps 2 and 3 then move the three points around a
        2-D layout to reproduce those memberships. With so few points we can print every number and
        watch the objective fall.
        """
    )
    return


@app.cell
def _(cu, mo, np):
    toy3 = cu.umap_toy_3point(seed=0)                      # 3 points, 10-D; every number is exact
    _P = toy3["P"]
    _d = toy3["high_dist"]
    _lo = toy3["Y_final"]
    _ld = toy3["low_dist_final"]
    _lh = toy3["loss_history"]
    _tbl = (
        "<table style='border-collapse:collapse;font-size:0.95em'>"
        "<tr><th style='text-align:left;padding:2px 14px 2px 0'>pair</th>"
        "<th style='padding:2px 14px'>high-D distance</th>"
        "<th style='padding:2px 14px'>membership P</th>"
        "<th style='padding:2px 14px'>final 2-D distance</th></tr>"
        f"<tr><td style='padding:2px 14px 2px 0'>points 0 &amp; 1 (neighbors)</td>"
        f"<td style='text-align:center'>{_d[0,1]:.3f}</td>"
        f"<td style='text-align:center'><b>{_P[0,1]:.3f}</b></td>"
        f"<td style='text-align:center'><b>{_ld[0,1]:.3f}</b></td></tr>"
        f"<tr><td style='padding:2px 14px 2px 0'>points 0 &amp; 2 (far)</td>"
        f"<td style='text-align:center'>{_d[0,2]:.3f}</td>"
        f"<td style='text-align:center'>{_P[0,2]:.3f}</td>"
        f"<td style='text-align:center'>{_ld[0,2]:.3f}</td></tr>"
        "</table>")
    _fig = cu.umap_toy_fig(toy3, title="Three points — the neighbor pair converges, the far point departs")
    mo.vstack([
        mo.md(
            f"The close pair earns a high membership (**P = {_P[0,1]:.3f}**) while the far point's "
            f"membership collapses to about **{_P[0,2]:.3f}**. Optimizing the objective then pulls the "
            f"two neighbors from a random start down to a 2-D distance of **{_ld[0,1]:.3f}**, while the "
            f"far point is pushed out to **{_ld[0,2]:.3f}**. The cross-entropy objective falls from "
            f"**{_lh[0]:.2f}** to **{_lh[-1]:.2f}** over the run."
            + _tbl),
        _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The same three steps scale directly to a cloud of points. Below, `cu.umap_objective_toy(...)`
        builds three Gaussian blobs in 8-D (about ninety points) and runs the identical optimization.
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
def _(go, mo, np, toy):
    _lh = toy["loss_history"]
    _ep = np.arange(1, len(_lh) + 1)
    _fig = go.Figure(go.Scatter(x=_ep, y=_lh, mode="lines", line=dict(color="#4c78a8", width=2)))
    _fig.add_vrect(x0=1, x1=10, fillcolor="#f0a500", opacity=0.10, line_width=0)
    _fig.add_annotation(x=6, y=np.log10(6000), showarrow=False, xanchor="left",
                        text="coarse layout:<br>the random cloud pulls apart into rough blobs",
                        font=dict(size=11, color="#8a6d00"))
    _fig.add_annotation(x=150, y=np.log10(1500), showarrow=False,
                        text="refinement: positions fine-tune", font=dict(size=11, color="#555"))
    _fig.update_layout(template="plotly_white", height=340,
                       title=f"The objective falling — fuzzy cross-entropy "
                             f"{_lh[0]:.0f} -> {_lh[-1]:.0f}",
                       xaxis_title="epoch", yaxis_title="cross-entropy loss (log scale)",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_yaxes(type="log", showgrid=False)
    _fig.update_xaxes(showgrid=False)
    mo.vstack([mo.md("The objective falls in two regimes, made visible here by the logarithmic vertical "
                     "axis (each gridline is a factor of ten, so a slow decline near the bottom is not "
                     "flattened to a false zero line). In the first few epochs (shaded) the cross-"
                     "entropy plunges by more than an order of magnitude: the random cloud is pulling "
                     "apart into roughly the right blobs — the **coarse-layout** phase. After that the "
                     "curve keeps dropping, gently but steadily, as points settle into their final "
                     "positions — the **refinement** phase. The coarse structure is fixed early; the "
                     "later epochs only fine-tune it."),
               _fig])
    return


@app.cell(hide_code=True)
def _(go, mo, toy):
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
def _(go, mo, np, toy):
    # (2) attractive vs repulsive force as a function of the FINAL low-D distance. The raw per-pair
    # scatter is unreadable — the repulsive force diverges as a pair's 2-D distance approaches zero, so
    # one near-overlapping pair sets the scale and flattens everything else. We instead bin pairs by
    # their 2-D distance and draw the MEDIAN force per bin as two curves, on a log vertical axis.
    _ld = toy["low_dist"]
    _att = toy["attractive"]
    _rep = toy["repulsive"]
    _XMAX = 8.0                                            # the range where the bulk of pairs live
    _edges = np.linspace(0.0, _XMAX, 17)
    _ctr = 0.5 * (_edges[:-1] + _edges[1:])
    _bin = np.digitize(_ld, _edges) - 1
    _med_att = np.array([np.median(_att[_bin == b]) if (_bin == b).any() else np.nan
                         for b in range(len(_ctr))])
    _med_rep = np.array([np.median(_rep[_bin == b]) if (_bin == b).any() else np.nan
                         for b in range(len(_ctr))])
    _fig = go.Figure()
    _fig.add_scatter(x=_ctr, y=_med_att, mode="lines+markers",
                     name="attractive  P·q  (pulls true neighbors in)",
                     line=dict(color="#2ca02c", width=3), marker=dict(size=6))
    _fig.add_scatter(x=_ctr, y=_med_rep, mode="lines+markers",
                     name="repulsive  (1-P)·q²/(1-q)  (pushes overlaps apart)",
                     line=dict(color="#d62728", width=3), marker=dict(size=6))
    _fig.add_annotation(x=_ctr[0], y=np.log10(_med_rep[0]), text="repulsion peaks as a pair's<br>2-D "
                        "distance approaches zero", showarrow=True, arrowhead=2, ax=70, ay=-20,
                        font=dict(size=11, color="#a02020"))
    _fig.update_layout(template="plotly_white", height=380,
                       title="The two forces vs 2-D distance — median force among pairs in each distance bin",
                       xaxis_title="distance between a pair in the 2-D layout",
                       yaxis_title="force magnitude (log scale)",
                       margin=dict(l=10, r=10, t=50, b=10),
                       legend=dict(x=0.36, y=0.97))
    _fig.update_yaxes(type="log", showgrid=False)
    _fig.update_xaxes(range=[0, _XMAX], showgrid=False)
    mo.vstack([mo.md("**The two forces.** Green is the **attractive** force $P\\cdot q$. It is "
                     "appreciable only for **true neighbors** (high membership $P$) and only while they "
                     "are still close, so it exists only at short distance and vanishes past the "
                     "neighbor range — it exerts no pull between unrelated points. Red is the "
                     "**repulsive** force $(1-P)\\,q^2/(1-q)$. It acts on **every** non-neighbor pair "
                     "and grows without bound as two points overlap, which is why it dominates at very "
                     "short distance and pushes crowded points apart. The layout comes to rest where, "
                     "for each pair, these two forces balance: neighbors settle at a short distance "
                     "where their attraction offsets the residual repulsion; everyone else is spread "
                     "out until repulsion fades. (Each point is the median force among all pairs at "
                     "that distance; a single near-overlapping pair would otherwise set the scale.)"),
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

        **PCA vs UMAP (and t-SNE).** PCA gives you *global, linear, interpretable* axes
        (each PC is a fixed recipe of features) but cannot bend, so a plain PC1-vs-PC2 scatter — the
        third option — could not pull the types apart (Part A). UMAP gives you a *local, nonlinear*
        layout that can flatten a curved manifold, but its axes have no formula and its global geometry
        is not to be read. **t-SNE** optimizes a similar neighbor-preserving objective and makes
        comparable pictures, but it preserves *even less* global structure (distances between clusters
        are essentially meaningless) and does not scale as gracefully, which is why the field has largely
        moved to UMAP for maps like this. We used PCA to count dimensions and denoise; we use UMAP to
        expose types. Different tools for different questions.
        """
    )
    return


@app.cell(hide_code=True)
def _(N, mo):
    mo.md(
        rf"""
        ## B.2 · The real map, across two settings

        On the real {N:,}-event data we work from a **5×5 sweep**: the same events already
        embedded at every combination of the two UMAP settings that shape the map. Selecting a cell
        below shows that setting's map; the clustering step that follows runs live.

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

        **The lesson these controls teach.** `n_neighbors` and `min_dist` are *not* learned from the
        data — they are choices, and they change the picture a lot. That is why we never read the map's
        global geometry as fact and always back a cluster claim with a statistical test (B.8).
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
    _fig.add_scatter(x=[_emb[EXAMPLE, 0]], y=[_emb[EXAMPLE, 1]], mode="markers", name="example event",
                     marker=dict(symbol="star", size=17, color="#f5b400",
                                 line=dict(color="#333", width=1)))
    _fig.update_layout(template="plotly_white", height=520,
                       title=(f"Selected map — n_neighbors={int(sweep['nn_values'][_i])}, "
                              f"min_dist={float(sweep['md_values'][_j]):g} — colored by canonical syllable"),
                       xaxis_title="UMAP-1", yaxis_title="UMAP-2", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    _fig.update_yaxes(showgrid=False)
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
def _(EXAMPLE, crel, cu, ev, mo):
    _gif = cu.event_gif_bytes(ev["kp"][EXAMPLE], ev["ranks"][EXAMPLE],
                              int(crel[EXAMPLE]), cell=240)
    mo.md("<b>The example event</b> — a calm, non-aggressive approach. Mice colored by rank; "
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
def _(clabels, crel, cu, kp, mo, np, ranks, yagg):
    # Pick clear exemplars LIVE (robust to bundle rebuilds): aggression events inside the purest
    # cluster, non-aggression events inside the large mixed cluster.
    _agg = np.where((yagg == 1) & (clabels == 3))[0][:4]
    _non = np.where((yagg == 0) & (clabels == 2))[0][:4]
    _ga = cu.grid_gif_bytes([(kp[i], ranks[i], int(crel[i])) for i in _agg], ncols=2, cell=140)
    _gn = cu.grid_gif_bytes([(kp[i], ranks[i], int(crel[i])) for i in _non], ncols=2, cell=140)
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
        output is a label per event.

        **Why HDBSCAN and not k-means?** k-means makes you *state the number of clusters in advance* and
        assumes each is a round ball of similar size — assumptions our folded map violates. HDBSCAN
        instead finds however many dense regions exist and refuses to assign genuinely sparse points.
        You never say how many clusters to find; `min_cluster_size` is the one knob.

        This is the one step we run live. At the canonical `min_cluster_size = 15` it reproduces the
        shared `default_labels` exactly. Larger values merge syllables; smaller values fracture the map
        and push more points into noise.

        **The lesson this control teaches.** `min_cluster_size` sets the *resolution* of the answer, and
        the number of clusters is a direct consequence of it — a modeling choice, not a discovered fact.
        Watch the cluster count and noise fraction change as you drag.
        """
    )
    return


@app.cell
def _(mo):
    mcs = mo.ui.slider(8, 35, value=15, step=1, label="min_cluster_size (live)",
                       debounce=True, full_width=True)
    return (mcs,)


@app.cell
def _(cu, emb0, go, mcs, mo):
    _lab = cu.run_hdbscan(emb0, min_cluster_size=int(mcs.value))
    _nc = len([c for c in set(_lab.tolist()) if c >= 0])
    _noise = float((_lab == -1).mean())
    _show_leg = _nc <= 12                                 # suppress a useless 20+ entry legend
    _fig = go.Figure()
    for _c in sorted(set(_lab.tolist())):
        _m = _lab == _c
        _fig.add_scattergl(x=emb0[_m, 0], y=emb0[_m, 1], mode="markers",
                           name=("noise" if _c < 0 else f"C{_c}"), showlegend=_show_leg,
                           marker=dict(size=5, opacity=0.7,
                                       color=("#cfd2d8" if _c < 0 else None)))
    _fig.update_layout(template="plotly_white", height=470, showlegend=_show_leg,
                       title=f"Live HDBSCAN — {_nc} clusters, {_noise:.0%} noise "
                             f"(min_cluster_size={int(mcs.value)})",
                       xaxis_title="UMAP-1", yaxis_title="UMAP-2", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    _fig.update_yaxes(showgrid=False)
    mo.vstack([mcs, _fig,
               mo.md("*Set the slider to 15 to reproduce the canonical result (4 clusters). This "
                     "slider is for exploration; the shared syllables stay fixed at 15.*")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Crisp labels are not the same as real structure.** HDBSCAN returns a definite integer label
        for every dense point, and the map looks like clean, separated islands. That confidence is a
        property of the algorithm, not evidence about behavior: it will partition any dense cloud into
        tidy-looking groups whether or not those groups correspond to distinct behaviors. The crispness
        of the picture is not a measure of whether the grouping is real. Keeping that separation in mind
        — how sure a method *looks* versus how much it has actually established — is what the rest of
        Part B is about.
        """
    )
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
        **{best_frac:.0%} aggression** — a **{best_lift:.2f}× lift** over the {base_rate:.0%} base
        rate. An unsupervised method, never shown a single label, has concentrated aggression somewhat
        into one region of the map: a real, if modest, tendency (about 38% aggression against a 32% base
        rate). Because we picked this cluster precisely for being the purest of several, the honest
        question is whether that lift is larger than the best-of-several would reach by chance. We put it
        to that test in B.8.

        Pick a syllable and render five of its member events (nearest the cluster centroid) as skeleton
        GIFs. Watching the members is the by-eye half of validating a cluster; the enrichment number is
        the by-number half.
        """
    )
    return


@app.cell
def _(best_cluster, clabels, mo):
    _opts = {f"C{c}": c for c in sorted(set(clabels.tolist())) if c >= 0}
    clus_pick = mo.ui.dropdown(options=_opts, value=f"C{best_cluster}", label="syllable to render")
    return (clus_pick,)


@app.cell
def _(clabels, clus_pick, crel, cu, emb0, kp, mo, np, ranks, yagg):
    # Exemplars chosen LIVE = the 5 members nearest the cluster centroid (robust to bundle rebuilds).
    _c = int(clus_pick.value)
    _m = np.where(clabels == _c)[0]
    _ctr = emb0[_m].mean(0)
    _pick = _m[np.argsort(np.linalg.norm(emb0[_m] - _ctr, axis=1))[:5]]
    _gif = cu.grid_gif_bytes([(kp[i], ranks[i], int(crel[i])) for i in _pick], ncols=5, cell=130)
    _frac = float(yagg[_m].mean())
    _cap = (f"**C{_c}** · {len(_m)} events · {_frac:.0%} aggression · showing 5 exemplars "
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

        Pick a feature. Good ones to start with: `bystander_dist_mean`, `heading_alignment`,
        `pair_dist_mean`, `appr_speed_mean`, `closing_speed`. Watch which regions light up.
        """
    )
    return


@app.cell
def _(fn, mo):
    feat_pick = mo.ui.dropdown(options={f: i for i, f in enumerate(fn)},
                               value="bystander_dist_mean" if "bystander_dist_mean" in fn else fn[0],
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
        red means "this cluster runs high on that feature," blue "low" (same red-is-up convention as the
        loadings heatmap). Reading a column tells you what a syllable *is*, in the vocabulary of the 19
        features.
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
    _fig.update_xaxes(showgrid=False)
    _fig.update_yaxes(showgrid=False)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Reading C3 honestly — it is *not* a "contact-heavy fighting" cluster.** The
        aggression-enriched syllable C3 runs **high on bystander distance and triangle area** (the third
        mouse is far away and the three animals are spread out) and **low on heading alignment and mutual
        facing** (the two interacting mice point in opposition, not the same way). It is *not* especially
        high on closing speed or contact. So the honest description is: the map's aggression pocket is
        events where the pair squares off with opposed headings while the bystander keeps its distance —
        a plausible fighting geometry, but read it from the data, not from the assumption that
        "aggression = contact."

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
                                  value="bystander_dist_mean" if "bystander_dist_mean" in fn else fn[0],
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
        A group that looked homogeneous can split into sub-types with **different aggression rates**.
        This is why "one cluster" is rarely the final answer.

        **The lesson this control teaches.** `sub min_cluster_size` sets how finely you split the parent;
        too small and you shatter it into noise, too large and it stays whole. Watch the level-2 count
        and the per-sub-type aggression rates respond.
        """
    )
    return


@app.cell
def _(clabels, mo):
    _opts = {f"C{c}": c for c in sorted(set(clabels.tolist())) if c >= 0}
    parent_pick = mo.ui.dropdown(options=_opts, value="C2", label="parent syllable to split")
    sub_mcs = mo.ui.slider(20, 45, value=40, step=1, label="sub min_cluster_size", debounce=True)
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
    _fig.update_xaxes(showgrid=False)
    _fig.update_yaxes(showgrid=False)
    mo.vstack([mo.hstack([parent_pick, sub_mcs], justify="start"), _fig,
               mo.md(f"C{_p} overall aggression rate is about {_a.mean():.0%}. Sub-types: "
                     + " · ".join(_lines))])
    return


@app.cell(hide_code=True)
def _(base_rate, mo):
    mo.md(
        rf"""
        ## B.8 · Exercise — did the map rediscover aggression, and is the lift real?

        **The question.** Is at least one data-driven syllable enriched for aggression above the
        {base_rate:.0%} base rate? And is that enrichment more than we would expect by chance? If both,
        the unsupervised map genuinely found aggression on its own.

        **Python skill practiced.** *Looping over groups and summarizing each with a boolean mask* — the
        pattern behind every "group-by" you will ever write. You will iterate over clusters and, for
        each, compute the fraction of its events that are aggression.

        **What you have.**

        - `clabels : (N,) int` — the canonical syllable of each event (-1 = noise).
        - `yagg : (N,) int` — 1 if the event is aggression, else 0.
        - `base_rate : float` — the corpus-wide aggression rate.

        **Your task.** `purest_agg_cluster` should return the syllable with the highest **aggression
        fraction**, but it currently ranks clusters by **size**. Fix the one flagged line so it ranks
        by aggression fraction. Everything else — the **min-size gate** (only clusters with at least 100
        events are eligible, so a tiny high-variance cluster cannot win "purest" by luck), and `frac` /
        `lift` for whichever cluster you pick — is done for you. The expected plot puts the purest
        syllable (red) clearly above the dashed base-rate line while the biggest cluster sits near it.
        """
    )
    return


@app.cell
def _(MIN_CLUSTER_N, base_rate, clabels, np, yagg):
    def purest_agg_cluster(clabels, yagg, base_rate, min_n=MIN_CLUSTER_N):
        clabels = np.asarray(clabels); yagg = np.asarray(yagg)
        # Min-size gate (done for you): only clusters with at least `min_n` events are eligible.
        # A size-50 cluster's aggression fraction is so noisy it can top the ranking by luck, so
        # without this gate the "purest" pick flips between rebuilds — a stability guard, not cheating.
        clusters = [c for c in sorted(set(clabels.tolist()))
                    if c >= 0 and (clabels == c).sum() >= min_n]
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
    return student_cluster, student_frac, student_lift


@app.cell(hide_code=True)
def _(MIN_CLUSTER_N, base_rate, best_cluster, clabels, go, mo, np, yagg):
    # The expected picture: aggression fraction per cluster, base rate as a reference line.
    _cs = sorted(c for c in set(clabels.tolist()) if c >= 0)
    _n = {c: int((clabels == c).sum()) for c in _cs}
    _fr = [float(yagg[clabels == c].mean()) for c in _cs]
    # red = the gated purest cluster; a cluster below the min-size gate is drawn hollow (ineligible).
    _colors = ["#e45756" if c == best_cluster
               else ("#d8dbdf" if _n[c] < MIN_CLUSTER_N else "#9aa0a6") for c in _cs]
    _fig = go.Figure(go.Bar(x=[f"C{c}" for c in _cs], y=_fr, marker_color=_colors,
                            text=[f"{f:.0%}" + ("<br>(n<100)" if _n[c] < MIN_CLUSTER_N else "")
                                  for c, f in zip(_cs, _fr)], textposition="outside"))
    _fig.add_hline(y=base_rate, line=dict(color="#333", dash="dash"),
                   annotation_text=f"base rate {base_rate:.0%}")
    _fig.update_layout(template="plotly_white", height=360,
                       title="Aggression fraction per syllable (red = gated purest)",
                       yaxis_title="fraction aggression", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    mo.vstack([mo.md("**Expected picture after the fix:** the purest *eligible* syllable (red) sits "
                     "clearly above the dashed base-rate line, while the largest cluster sits near or "
                     "below it — the aggression signal lives in a small dense pocket, not the big blob. "
                     "A cluster below the 100-event gate (drawn hollow) is ignored even if its noisy "
                     "fraction looks high."),
               _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Reveal solution": mo.md(
            r"""
            Rank the clusters by **aggression fraction**, not size:

            ```python
            chosen = max(clusters, key=lambda c: yagg[clabels == c].mean())
            ```

            On the canonical labels this returns **C3**, about 38% aggression, a **1.19× lift**. The
            largest cluster (C2, ~1,423 events) sits near the base rate — which is why ranking by size
            gives the wrong answer. Note the tiny **C0** (n=53) actually has a *nearly identical*
            fraction (~38%), but the min-size gate keeps it out of the running: with only 53 events its
            fraction is too noisy to trust, and letting it compete would make the "purest" pick flip on
            the next rebuild.
            """)
    })
    return


@app.cell(hide_code=True)
def _(best_cluster, best_lift, mo, student_cluster, student_frac, student_lift):
    _PIN, _TOL = best_lift, 0.10          # accept the pinned lift +/- 0.10
    _ok = abs(student_lift - _PIN) <= _TOL and student_cluster == best_cluster
    if _ok:
        _bg, _fg, _icon = "#e7f6ec", "#166534", "PASS"
        _msg = (f"C{student_cluster} is {student_frac:.0%} aggression, a **{student_lift:.2f}× lift** "
                f"(pinned {_PIN:.2f}) — the purest *eligible* syllable. The map has concentrated "
                f"aggression somewhat, without ever seeing a label: a real, modest tendency. Because we "
                f"picked this cluster *for* being the purest of several, the next cell checks it "
                f"properly — comparing it against a null that is also allowed to pick the best of "
                f"several clusters.")
    else:
        _bg, _fg, _icon = "#fdecec", "#9b1c1c", "NOT YET"
        _msg = (f"Got C{student_cluster} at **{student_lift:.2f}×** — expected C{best_cluster} at about "
                f"{_PIN:.2f}×. If your lift is near 1.0× you are still ranking by *size* (the biggest "
                f"cluster). Rank by `yagg[clabels == c].mean()` instead.")
    mo.md(f"<div style='border:1px solid #ccc;border-radius:8px;padding:10px 14px;background:{_bg};"
          f"color:{_fg}'><b>{_icon}</b> &nbsp; {_msg}</div>")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Is a 1.19× lift more than luck? The honest test is a *null of the maximum*

        A lift ratio on its own is not evidence — **with several clusters to choose from, one will
        always look like the purest**, even under random labels. So the fair question is not "is C3
        special?" (we already picked C3 *because* it was the highest — testing it alone is circular).
        The fair question is: *if aggression labels were sprinkled at random across the clustered events,
        how often would the **purest cluster — whichever one it turned out to be** — reach the 38% we
        observed?*

        That is a **null-of-the-maximum** permutation test. Each iteration we shuffle the 0/1 aggression
        labels across all clustered events, recompute every cluster's fraction, and record only the
        **largest** of them (over all clusters the map produced — including the tiny C0, because in
        practice you *do* scan every cluster and report the best). Comparing our observed purest fraction
        to that null-of-max is the honest family-wise test the intro promised — and it gives a very
        different answer than testing C3 in isolation would.
        """
    )
    return


@app.cell
def _(best_cluster, clabels, go, mo, np, yagg):
    _clustered = clabels >= 0
    _lab = clabels[_clustered]
    _y = yagg[_clustered].astype(float)
    _masks = [_lab == c for c in sorted(set(_lab.tolist()))]      # membership per cluster (fixed)
    _obs = max(float(_y[_mk].mean()) for _mk in _masks)          # observed PUREST fraction (= C3, 38%)
    _rng = np.random.RandomState(0)
    _NIT = 5000
    _null = np.empty(_NIT)
    for _i in range(_NIT):
        _ys = _rng.permutation(_y)                               # shuffle labels across clustered events
        _null[_i] = max(float(_ys[_mk].mean()) for _mk in _masks)   # purest cluster THIS shuffle
    _p = (np.sum(_null >= _obs - 1e-12) + 1) / (_NIT + 1)
    _p95 = float(np.percentile(_null, 95))
    # For contrast: the WRONG (circular) null that only ever looks at C3's fixed membership.
    _pool = _y.copy(); _n3 = int((_lab == best_cluster).sum())
    _null_fixed = np.array([_rng.permutation(_pool)[:_n3].mean() for _ in range(_NIT)])
    _p_fixed = (np.sum(_null_fixed >= _obs - 1e-12) + 1) / (_NIT + 1)

    _fig = go.Figure()
    _fig.add_histogram(x=_null, nbinsx=40, marker_color="#c7c7c7", name="null-of-max")
    _fig.add_vline(x=_p95, line=dict(color="#8a8a8a", width=2, dash="dot"),
                   annotation_text="null 95th pct", annotation_position="top left")
    _fig.add_vline(x=_obs, line=dict(color="#e45756", width=3),
                   annotation_text="observed purest", annotation_position="top")
    _fig.update_layout(template="plotly_white", height=340, showlegend=False,
                       title=f"Purest-cluster aggression fraction vs a null-of-max (p = {_p:.3f})",
                       xaxis_title="purest cluster's aggression fraction under shuffled labels",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    mo.vstack([mo.md(f"**Observed purest fraction = {_obs:.1%}; null-of-max p = {_p:.3f}.** Across "
                     f"{_NIT:,} random relabelings the purest of the four clusters reaches "
                     f"{_null.mean():.1%} *on average* and {_p95:.1%} at its 95th percentile — so our "
                     f"observed {_obs:.1%} sits **inside** the cloud, not out in the tail. Once you "
                     f"account for having picked the purest of several clusters, the 1.19× lift **does "
                     f"not clear a family-wise null**.\n\n"
                     f"Contrast this with the *circular* test that fixes C{best_cluster}'s membership "
                     f"and only shuffles which events are aggression: that one reports p = {_p_fixed:.3f}, "
                     f"apparently strong — but it is cheating, because we selected C{best_cluster} *for* "
                     f"being the purest. Testing the thing you hand-picked against a null that cannot "
                     f"pick a different winner is exactly the double-dipping error NB05 is built to "
                     f"expose. The honest number is {_p:.3f}."),
               _fig])
    return


@app.cell(hide_code=True)
def _(best_cluster, best_lift, mo):
    mo.md(
        f"""
        ### Answer to Question B

        Distinct behavioral types **do** exist — the map's regions differ sharply in the 19 features,
        and its purest syllable, **C{best_cluster}**, is the opposed-heading, bystander-distant
        configuration, not a generic "contact" blob. Does the map lean toward aggression on its own?
        Yes, modestly: C{best_cluster} reaches a **{best_lift:.2f}×** lift over the base rate — a real
        tendency, an aggression fraction meaningfully above the corpus baseline. The lesson is
        in the gap between how that result *looks* and what it *establishes*. A crisp cluster and a lift
        ratio read as more definitive than they are: tested against a null that, like us, is allowed to
        pick the purest of several clusters, the {best_lift:.2f}× lift is not distinguishable from
        chance (null-of-max p ≈ 0.25). **The confidence of an assignment is not the validity of the
        structure behind it.** The effect is genuine and modest; proving that a specific label lives in
        a specific pocket takes more than a hand-picked cluster and a ratio — which is what the next
        notebook builds.

        > **Next question (NB05):** the map leans toward an aggression pocket, but a lift ratio on a
        > hand-picked cluster cannot settle the claim. How do we test a claim like this *honestly* —
        > choosing the right null, not double-dipping on the cluster we selected, not leaking, and not
        > counting the same cage thousands of times? And can a *supervised* decoder, trained on labels,
        > read aggression far more sharply than this map did?
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
        - **Extreme scores are often bad data.** The tails of the higher PCs were tracking artifacts,
          not rare behaviors — always gate or winsorize before you trust an outlier.
        - **The map discards time and rare behaviors.** Each event becomes one frozen dot, and any
          behavior too rare to reach `min_cluster_size` dissolves into noise. The map also inherits
          every earlier modeling choice (which features, how scaled).
        - **The map's geometry is a setting, not a fact.** Distance between clusters, cluster area, and
          even the number of clusters all move with `n_neighbors`, `min_dist`, and `min_cluster_size`.
          Trust local neighborhoods plus a statistical test, not the picture's global shape.
        """
    )
    return


if __name__ == "__main__":
    app.run()
