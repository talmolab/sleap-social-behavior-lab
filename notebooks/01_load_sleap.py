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

    DATA = os.path.join(ROOT, "data")
    return DATA, cu, go, np, os


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Read a real `.slp` file

        A `.slp` carries three things we care about: the **skeleton** (which nodes exist and how
        they connect), the **tracks** (one per animal identity), and the per-frame **keypoints**.
        You read one with [`sleap-io`](https://io.sleap.ai):

        ```python
        import sleap_io as sio
        labels = sio.load_slp("example_dep.slp")
        skeleton = labels.skeletons[0]          # nodes + edges
        # put each instance in its FIXED track slot so slot m is always the same animal:
        kp = labels.numpy()                     # (frames, tracks, nodes, xy)
        ```

        To keep this course installable on any cloud kernel, we ran exactly that offline
        (`tools/decode_example_slp.py`) and saved the resulting arrays to a tiny `.npz`, so the
        notebook needs no heavy pose-IO dependency. The keypoints below are the real decoded clip.
        """
    )
    return


@app.cell
def _(cu):
    # The real decoded clip: (frames, tracks, nodes, xy) with each animal in a FIXED track slot.
    _slp = cu.load_slp_demo()
    slp_kp = _slp["kp"]
    node_names = _slp["node_names"]
    slp_edges = _slp["edges"]
    slp_source = _slp["source"]
    return node_names, slp_edges, slp_kp, slp_source


@app.cell
def _(mo, node_names, slp_kp, slp_source):
    mo.md(
        f"""
        **Loaded** `{slp_source}`

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
    return


@app.cell
def _(mo, slp_kp):
    frame_idx = mo.ui.slider(0, slp_kp.shape[0] - 1, value=0, step=1, label="frame",
                             full_width=True)
    return (frame_idx,)


@app.cell
def _(cu, frame_idx, mo, slp_edges, slp_kp):
    # Slider stacked directly above the figure, so the control sits next to what it drives (in the
    # editor *and* in app mode). cu.skeleton_fig colors each mouse by its fixed track slot.
    mo.vstack([frame_idx, cu.skeleton_fig(slp_kp[frame_idx.value], slp_edges)])
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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
