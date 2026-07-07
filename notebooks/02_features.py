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
# ]
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 02 · From keypoints to *allocentric* features

        Raw keypoints are in **camera pixels**: the same behavior looks completely different if it
        happens in the top-left vs. bottom-right of the cage, or facing north vs. south. A classifier
        trained on raw pixels would waste capacity learning *where* and *which way* — not *what*.

        The fix is an **egocentric / allocentric transform**: re-express every event in the
        **approacher's own body frame**. Put the approacher's tail-base at the origin and rotate so
        it faces "up". Now only the *social geometry* remains — where the other mice are **relative
        to the approacher** — which is what behavior actually is.

        $$\tilde{\mathbf{p}} \;=\; \mathbf{R}(\alpha)\,(\mathbf{p} - \mathbf{c}),
        \qquad
        \mathbf{R}(\alpha)=\begin{bmatrix}\cos\alpha & -\sin\alpha\\ \sin\alpha & \cos\alpha\end{bmatrix}$$

        where $\mathbf{c}$ is the approacher's tail-base (TTI) and $\alpha$ rotates its heading
        (TTI→head) onto $+y$. The **same** $\mathbf{R}$ and $\mathbf{c}$ are applied to all three
        mice, so their *relative* configuration is preserved while the arena pose is removed.
        """
    )
    return


@app.cell
def _():
    import os
    import sys
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    import urllib.request

    _RAW = os.environ.get(
        "COURSE_REPO_RAW",
        "https://raw.githubusercontent.com/Elmaestrotango/sleap-social-behavior-lab/main",
    )

    def _find_root():
        p = os.getcwd()
        for _ in range(6):
            if os.path.isdir(os.path.join(p, "course")) and os.path.isdir(os.path.join(p, "data")):
                return p
            p = os.path.dirname(p)
        return None

    # On a bare cloud notebook (e.g. molab) there is no repo checkout: fetch course_utils.py, then
    # let cu.bootstrap() download the bundled data on first use.
    ROOT = _find_root() or os.getcwd()
    _cu = os.path.join(ROOT, "course", "course_utils.py")
    if not os.path.exists(_cu):
        os.makedirs(os.path.dirname(_cu), exist_ok=True)
        urllib.request.urlretrieve(_RAW + "/course/course_utils.py", _cu)
    sys.path.insert(0, os.path.join(ROOT, "course"))
    import course_utils as cu

    ROOT, DATA, SCRATCH = cu.bootstrap()

    events = cu.load_events(os.path.join(ROOT, "data", "train_events.npz"))
    return cu, events, go, make_subplots, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## See the transform

        Pick an event and a frame. **Left** = raw camera coordinates. **Right** = the same three
        skeletons after centering + rotating into the approacher's frame (🔴 approacher's rank color
        varies; the approacher always ends up at the origin facing up). Notice the right panel looks
        the same regardless of where in the cage the event happened.
        """
    )
    return


@app.cell
def _(events, mo):
    ev_sel = mo.ui.slider(0, len(events["kp"]) - 1, value=0, step=1, label="event #", full_width=True)
    ev_sel
    return (ev_sel,)


@app.cell
def _(cu, ev_sel, events, mo):
    fr_sel = mo.ui.slider(0, events["kp"].shape[1] - 1, value=int(events["contact_rel"][ev_sel.value]),
                          step=1, label="frame", full_width=True)
    fr_sel
    return (fr_sel,)


@app.cell
def _(cu, ev_sel, events, fr_sel, go, make_subplots, np):
    _kp = events["kp"][ev_sel.value].astype("float32")            # (T,3,15,2) world
    _kp_allo = cu.allocentricize(_kp)                             # (T,3,15,2) approacher frame
    _rk = events["ranks"][ev_sel.value]
    _cols = [cu.RANK_HEX.get(int(r), "#888") for r in _rk]
    _names = [f"{cu.RANK_NAMES[int(_rk[0])]} (appr)", f"{cu.RANK_NAMES[int(_rk[1])]} (appe)",
              f"{cu.RANK_NAMES[int(_rk[2])]} (byst)"]

    def _add(fig, kp_frame, col, rev_y):
        for m in range(3):
            kp = kp_frame[m]
            ok = np.isfinite(kp).all(1)
            ex, ey = [], []
            for u, v in cu.SKELETON_EDGES:
                if ok[u] and ok[v]:
                    ex += [kp[u, 0], kp[v, 0], None]; ey += [kp[u, 1], kp[v, 1], None]
            fig.add_scatter(x=ex, y=ey, mode="lines", line=dict(color=_cols[m], width=2),
                            row=1, col=col, name=_names[m], showlegend=(col == 1), hoverinfo="skip")
            fig.add_scatter(x=kp[ok, 0], y=kp[ok, 1], mode="markers",
                            marker=dict(color=_cols[m], size=6), row=1, col=col,
                            showlegend=False, hoverinfo="skip")

    _fig = make_subplots(rows=1, cols=2, subplot_titles=("raw (camera pixels)",
                                                         "allocentric (approacher frame)"))
    _add(_fig, _kp[fr_sel.value], 1, True)
    _add(_fig, _kp_allo[fr_sel.value], 2, False)
    _fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, row=1, col=1)
    _fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
    _fig.add_scatter(x=[0], y=[0], mode="markers", marker=dict(color="black", symbol="x", size=10),
                     row=1, col=2, name="origin", showlegend=False)
    _fig.update_layout(height=460, template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The feature vector

        From the allocentric window we compute **19 interpretable features** — each a number you
        could explain to a biologist. They summarize *speed*, *body posture*, *inter-mouse geometry*,
        and *the bystander*. This is deliberately small and readable (real pipelines use hundreds);
        the point is to see that a few well-chosen numbers already capture behavior.
        """
    )
    return


@app.cell
def _(cu, mo):
    _desc = {
        "appr_speed_mean": "approacher mean centroid speed (px/frame)",
        "appr_speed_max": "approacher peak speed — lunges spike this",
        "appe_speed_mean": "approachee mean speed",
        "appe_speed_max": "approachee peak speed — fleeing spikes this",
        "appr_body_len": "approacher nose→tail-base length (stretched vs hunched)",
        "appe_body_len": "approachee body length",
        "appr_angvel": "approacher turning rate (rad/frame)",
        "appe_angvel": "approachee turning rate",
        "pair_dist_mean": "mean approacher–approachee distance",
        "pair_dist_min": "closest approach (contact ⇒ small)",
        "appr_nose_to_appe_tti_min": "approacher nose → approachee tail-base (anogenital sniff)",
        "appe_nose_to_appr_tti_min": "approachee nose → approacher tail-base",
        "appr_faces_appe": "how directly the approacher faces the approachee (cosine)",
        "appe_faces_appr": "how directly the approachee faces back",
        "closing_speed": "rate the gap closes (+ = approaching)",
        "heading_alignment": "cosine of the two headings (+1 same dir, −1 head-on)",
        "bystander_dist_mean": "mean approacher–bystander distance",
        "bystander_dist_min": "closest the bystander gets",
        "triangle_area_mean": "area of the 3-mouse triangle (group spread)",
    }
    mo.md("| # | feature | meaning |\n|---|---|---|\n" +
          "\n".join(f"| {i} | `{n}` | {_desc[n]} |" for i, n in enumerate(cu.FEATURE_NAMES)))
    return


@app.cell
def _(cu, events):
    # compute all features once (fast: ~1 s for 1500 events)
    X = cu.features_batch(events["kp"].astype("float32"))
    return (X,)


@app.cell
def _(cu, mo):
    feat_pick = mo.ui.dropdown(options=cu.FEATURE_NAMES, value="pair_dist_min",
                               label="feature to inspect")
    feat_pick
    return (feat_pick,)


@app.cell
def _(X, cu, events, feat_pick, go):
    _j = cu.FEATURE_NAMES.index(feat_pick.value)
    _v = X[:, _j]
    _agg = events["agg_label"] == 1
    _fig = go.Figure()
    _fig.add_histogram(x=_v[~_agg], name="not aggression", opacity=0.65, nbinsx=45,
                       marker_color="#7f7f7f")
    _fig.add_histogram(x=_v[_agg], name="aggression", opacity=0.65, nbinsx=45,
                       marker_color="#d62728")
    _fig.update_layout(barmode="overlay", template="plotly_white", height=380,
                       title=f"'{feat_pick.value}' — aggression vs. not",
                       xaxis_title=feat_pick.value, yaxis_title="count",
                       margin=dict(l=10, r=10, t=40, b=10))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Try `pair_dist_min`, `appr_speed_max`, `closing_speed`, `appr_nose_to_appe_tti_min`: the
        aggression (red) and non-aggression (grey) distributions clearly differ — the features carry
        real signal. **No single feature is enough**, though; that's why we next look at all 19 at
        once with dimensionality reduction and clustering.

        **Next → `03_clustering.py`.**
        """
    )
    return


if __name__ == "__main__":
    app.run()
