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
        # 01 · Loading SLEAP data

        **Pose estimation** turns a video of an animal into a time series of *keypoints* — the
        pixel coordinates of body landmarks (nose, ears, tail base, …) in every frame.
        [SLEAP](https://sleap.ai) is a deep-learning tool that does this for multiple interacting
        animals at once and saves the result in a `.slp` file.

        This course analyzes **three mice living together**, filmed continuously. A single frame of
        SLEAP output is a small tensor:

        $$\text{frame} \;\in\; \mathbb{R}^{\,M \times N \times 2}
        \qquad M=\text{mice}=3,\; N=\text{nodes}=15,\; 2=(x,y)\text{ pixels}.$$

        Stack every frame and you get the full recording, $\mathbb{R}^{F\times M\times N\times 2}$.
        Our job for the rest of the course: turn this raw geometry into a description of *social
        behavior* — who is doing what to whom.

        **The road ahead:** `01 load` → `02 features` → `03 clustering` → `04 statistics` →
        `05 labeling` → `06 classifier`.
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

    DATA = os.path.join(ROOT, "data")
    return DATA, cu, go, np, os


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Read a real `.slp` file

        We load a short clip with [`sleap-io`](https://io.sleap.ai). The `.slp` carries three
        things we care about: the **skeleton** (which nodes exist and how they connect), the
        **tracks** (one per animal identity), and the per-frame **keypoints**.
        """
    )
    return


@app.cell
def _(DATA, np, os):
    import sleap_io as sio

    _cands = [os.path.join(DATA, "raw_slp", f) for f in
              ("example_dep.slp", "example_pre.slp", "example_post.slp", "example_heldout.slp")]
    slp_path = next((p for p in _cands if os.path.exists(p)), None)

    labels = sio.load_slp(slp_path) if slp_path else None
    if labels is not None:
        # build a compact array directly from the labeled frames (frame, mouse, node, xy)
        slp_kp = np.stack([lf.numpy() for lf in labels.labeled_frames]).astype(np.float32)
        skel = labels.skeletons[0]
        node_names = [n.name for n in skel.nodes]
        slp_edges = [(node_names.index(e.source.name), node_names.index(e.destination.name))
                     for e in skel.edges]
    else:
        slp_kp, skel, node_names, slp_edges = None, None, [], []
    return labels, node_names, skel, slp_edges, slp_kp, slp_path


@app.cell
def _(labels, mo, node_names, slp_edges, slp_kp, slp_path):
    if labels is None:
        _out = mo.md("⚠️ No `.slp` found in `data/raw_slp/` — run `tools/trim_slp.py` (instructors).")
    else:
        _out = mo.md(
            f"""
            **Loaded** `{slp_path.split('/')[-1]}`

            | property | value |
            |---|---|
            | frames in clip | {slp_kp.shape[0]} |
            | mice (tracks) | {slp_kp.shape[1]} |
            | nodes | {slp_kp.shape[2]} |
            | array shape | `{tuple(slp_kp.shape)}` = (frames, mice, nodes, xy) |

            Nodes: {", ".join(f"`{i}:{n}`" for i, n in enumerate(node_names))}

            Each animal is one **track**. Notice we have *identities* here — track 0 stays track 0
            across frames — but SLEAP's raw tracks can swap when animals huddle or cross. Resolving
            identity reliably (here, to a dominance **rank**) is its own hard problem; the bundled
            dataset has already been identity-corrected for you.
            """
        )
    _out
    return


@app.cell
def _(mo, slp_kp):
    frame_idx = mo.ui.slider(
        0, (slp_kp.shape[0] - 1 if slp_kp is not None else 1), value=0, step=1,
        label="frame", full_width=True,
    )
    frame_idx
    return (frame_idx,)


@app.cell
def _(cu, frame_idx, go, np, slp_edges, slp_kp):
    def _skeleton_fig(kp_frame, edges):
        # kp_frame: (mice, nodes, 2) in image pixels (y grows downward)
        _colors = ["#d62728", "#1f77b4", "#2ca02c"]
        _traces = []
        for m in range(kp_frame.shape[0]):
            kp = kp_frame[m]
            ok = np.isfinite(kp).all(1)
            ex, ey = [], []
            for u, v in edges:
                if ok[u] and ok[v]:
                    ex += [kp[u, 0], kp[v, 0], None]
                    ey += [kp[u, 1], kp[v, 1], None]
            _traces.append(go.Scatter(x=ex, y=ey, mode="lines",
                                      line=dict(color=_colors[m], width=2),
                                      name=f"mouse {m}", hoverinfo="skip"))
            _traces.append(go.Scatter(x=kp[ok, 0], y=kp[ok, 1], mode="markers",
                                      marker=dict(color=_colors[m], size=7),
                                      showlegend=False, hoverinfo="skip"))
        f = go.Figure(_traces)
        f.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
        f.update_layout(height=520, title="SLEAP skeletons — drag the frame slider",
                        margin=dict(l=10, r=10, t=40, b=10), template="plotly_white")
        return f

    _skeleton_fig(slp_kp[frame_idx.value], slp_edges) if slp_kp is not None else None
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Scrub the slider and watch the three skeletons move. That is *all* SLEAP gives us: moving
        points. Everything else — "mouse A is chasing mouse B", "this is aggression" — we have to
        **compute**. The next notebook turns these points into behavioral features.

        ## The unit of analysis: an *approach event*

        Rather than analyze all 90,000 frames of every video, the dataset is pre-segmented into
        **approach events**: moments when two mice go from far apart to within ~200 px while at
        least one is moving — i.e. candidate social interactions. Each event is a short window with
        the three mice already **ordered by role** and **colored by dominance rank**
        (🔴 Dom, 🔵 Mid, 🟢 Sub):
        """
    )
    return


@app.cell
def _(DATA, cu, os):
    events = cu.load_events(os.path.join(DATA, "train_events.npz"))
    n_events = len(events["kp"])
    return events, n_events


@app.cell
def _(mo, n_events):
    ev_pick = mo.ui.slider(0, n_events - 1, value=0, step=1, label="event #", full_width=True)
    ev_pick
    return (ev_pick,)


@app.cell
def _(cu, ev_pick, events, mo):
    _i = ev_pick.value
    _rk = events["ranks"][_i]
    _gif = cu.event_gif_bytes(events["kp"][_i].astype("float32"), _rk,
                              int(events["contact_rel"][_i]), cell=240, fps=18)
    _cat = events["category"][_i] or "(unlabeled)"
    mo.md(
        f"""
        {cu.gif_img_html(_gif, width=280)}

        **event {_i}** &nbsp;·&nbsp; condition **{events['condition'][_i]}** &nbsp;·&nbsp;
        registry label: **{_cat}** &nbsp;·&nbsp;
        ranks (approacher→approachee→bystander): {[cu.RANK_NAMES[int(r)] for r in _rk]}

        *The white arrow points approacher → approachee; the red dot marks contact onset.*
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Recap
        - SLEAP output = keypoint tensor `(frames, mice, nodes, xy)`; nothing more.
        - Identity/tracking is a real problem; our data is identity-resolved to **rank**.
        - We analyze short **approach events**, not raw video.

        **Next → `02_features.py`:** convert each event's raw keypoints into an *allocentric*
        feature vector that describes the social configuration independent of where in the cage it
        happened.
        """
    )
    return


if __name__ == "__main__":
    app.run()
