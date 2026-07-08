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
        "https://raw.githubusercontent.com/Elmaestrotango/sleap-social-behavior-lab/main")

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
    return cu, go, np


@app.cell
def _(cu, np):
    # --- the training corpus, and the held-out cage (Camera 16), which we keep sealed until Week 2 ---
    ev = cu.load_events("data/train_events.npz")     # kp, ranks, condition, contact_rel, agg_label, ...
    der = cu.load_derived("train")                   # cohort, cage, sex, X (features), pca_scores, ...
    ho = cu.load_events("data/heldout_events.npz")   # Camera 16 — never touched by any Week-1 analysis

    # --- our example approach event: one real event we return to throughout the notebook ---
    # The three mice are stored in fixed slots, ordered [approacher, approachee, bystander].
    # In this event the approacher is the Sub mouse (green), the approachee is the Mid mouse (blue),
    # and the bystander is the Dom mouse (red). It is a NON-aggressive approach (agg_label = 0), from a
    # female cage (cohort 12192025, cohort-unique cage 110). We chose it because every keypoint is
    # tracked cleanly, which makes it a good object to learn the data structure on.
    EX_IDX = cu.event_index_by_key(ev, "12192025_pre|cam.10.00046-2025-12-18T16|m0-m2|83141")
    ex_kp = ev["kp"][EX_IDX]                          # (130, 3, 15, 2) pose over time
    ex_ranks = ev["ranks"][EX_IDX]                    # (3,) rank of each ordered mouse -> [3, 2, 1]
    ex_cr = int(ev["contact_rel"][EX_IDX])            # frame at which contact begins

    # Mouse colors are ALWAYS by rank: Dom=red, Mid=blue, Sub=green, unknown=gray. We map each of the
    # three ordered mice to its rank color so the colors stay consistent with every later notebook.
    ex_cols = tuple(cu.RANK_HEX.get(int(r), cu.RANK_HEX[0]) for r in ex_ranks)
    return der,ev, ex_cols, ex_cr, ex_kp, ex_ranks, ho


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # NB01 · Keypoints and identity
        ### *Reading the Social Brain — Week 1*

        We study social behavior and its neural basis. Our aim is to relate what a mouse *does* to
        what its brain is *doing*, and in Week 2 we work directly with neural recordings. Behavior and
        neural activity are two views of the same system, and each is only as useful as our ability to
        measure it; this week we build the behavioral side of that measurement. Everything this week
        turns raw tracking data into a clean, comparable readout of what happens when two mice meet.

        **The question for this notebook.** *What is behavior, and can we trust how we measure it?*
        We never see the mice directly. We see the output of a pose tracker: a table of body-landmark
        coordinates, one row per video frame. Before we build anything on top of that table, we have to
        know what is in it, where it is reliable, and where it quietly fails. Concretely, the data
        asserts that one specific mouse is "the approacher." Is that a real fact about the animals, or
        just bookkeeping? And what happens downstream if the tracker confuses two mice? By the end you
        will have read the raw pose axis by axis, seen exactly where it breaks, and tested its central
        assumption yourself.

        **How to work through it.** Read each section top to bottom. Prose cells explain *why* a step
        matters and *define* every term before it is used. Code cells render a figure or a short
        animation. Three exercises ask you to fill a single blank line and check your answer against a
        pinned value. The Python skill this notebook builds is **array indexing and slicing** — pulling
        the exact numbers you want out of a multi-dimensional table.
        """
    )
    return


@app.cell(hide_code=True)
def _(ho, mo):
    _n = len(ho["kp"])
    mo.md(
        f"""
        <div style="border:2px solid #999; border-radius:10px; padding:14px 18px;
        background:#f6f6f8; color:#333;">
        <div style="font-size:1.15em; font-weight:600; color:#555;">Held-out data · Camera 16
        (sealed)</div>
        <div style="margin-top:6px;">
        <b>{_n} events</b> from one cage that no Week-1 analysis will look at.
        </div>
        <div style="margin-top:8px; font-size:0.95em; color:#555;">
        A measurement or a decoder is only convincing if it works on data it was never tuned on. We
        therefore set Camera 16 aside now and open it only in Week 2, to test the finished pipeline on
        a cage it has never seen. This cage happens to be all female, which matters for the sex
        comparison we make later — but that is a Week-2 concern; here we simply leave it closed.
        </div>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 1 · The pose tensor, axis by axis

        **Why this matters.** Everything downstream rests on one array of numbers per event. If we do
        not understand its shape and its quirks, every later step inherits our confusion. So we start
        by reading that array carefully.

        **Definitions.**

        - A **keypoint** (or *node*) is one tracked body landmark — the nose, an ear, the base of the
          tail. Each keypoint has an (x, y) position in the video image, measured in pixels.
        - A **pose** is the full set of keypoints for one mouse in one frame. Our skeleton has
          **15 keypoints**.
        - A **frame** is one still image from the video. Our clips run at **50 frames per second (fps)**.
        - A **tensor** is a multi-dimensional array of numbers — a table with more than two axes.
        - **SLEAP** is the pose-tracking program that produced this data (Pereira et al., 2022). It
          finds the 15 landmarks on each mouse in each frame. Its output is not a picture; it is the
          coordinate table we work with.

        SLEAP gives us, per event, one array `kp` with shape **`(T, mice, nodes, xy)`**. Read it left
        to right:

        | axis | size | meaning |
        |---|---|---|
        | `T` | 130 | frames (about 2.6 s at 50 fps) |
        | `mice` | 3 | the three mice, stored in fixed slots `[approacher, approachee, bystander]` |
        | `nodes` | 15 | body keypoints (nose, head, tail base, tail tip, …) |
        | `xy` | 2 | pixel coordinates (note: y grows *downward* in image space) |

        Two properties matter downstream and are worth stating now.

        - **A missing keypoint is stored as `NaN`, not 0.** `NaN` means "not a number" — here it marks
          a keypoint the tracker could not place on that frame. Storing 0 would falsely pin that body
          part to the top-left corner of the image; `NaN` honestly records "unknown," and our code
          skips it.
        - **A slot in the array is not a guaranteed identity.** The mice live in fixed slots, and
          slot 0 is *labelled* "the approacher." But that label is only as good as the tracker's
          ability to keep each mouse in its own slot on every frame. When two mice are close, the
          tracker can swap them, and then slot 0 silently holds the other animal. We test this in
          Section 3.

        The 15 keypoints connect through two hubs — **head (node 1)** and **TTI (node 11**, the
        junction where the tail meets the torso) — in a star-shaped skeleton we draw below.
        """
    )
    return


@app.cell(hide_code=True)
def _(ev, ex_kp, mo):
    # Read the array's shape and axes back out of the real data, in plain language.
    _N, _T, _M, _Nn, _xy = ev["kp"].shape
    mo.md(
        f"""
        **Reading the shape directly from the data.** The full training array `ev["kp"]` has shape
        **`{tuple(ev["kp"].shape)}`** — that is **{_N} events**, each **{_T} frames** long, with
        **{_M} mice**, **{_Nn} keypoints** per mouse, and **{_xy} numbers (x, y)** per keypoint.

        One event, `ex_kp = ev["kp"][EX_IDX]`, has shape **`{tuple(ex_kp.shape)}`**. To pull out a single
        mouse's single keypoint over the whole event, we index the two middle axes and keep `:` (which
        means "everything") on the first axis (time) and the last axis (x, y). For example the
        approacher's nose track is `ex_kp[:, 0, cu.NOSE, :]`, an array of shape
        **`{tuple(ex_kp[:, 0, 0, :].shape)}`** — one (x, y) pair per frame. Indexing like this is the
        core skill of the notebook: you name a position on each axis (or `:` to keep the whole axis),
        and you get back exactly the sub-array you asked for.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercise 1 — plot one keypoint's position over time

        **Python skill: array indexing and slicing.** You will pull a single keypoint out of the
        four-axis pose tensor and plot how it moves. The output is a line plot you can check by eye.

        **What to edit.** Exactly one line below is blank. Replace `____` so that `nose_xy` becomes the
        approacher's nose track: slot `0`, node `cu.NOSE`, **all** frames, **both** (x, y) columns.

        ```python
        # ex_kp has shape (T, mice, nodes, xy) = (130, 3, 15, 2).
        # Axis 0 = frame, axis 1 = which mouse (slot), axis 2 = which keypoint, axis 3 = x or y.
        #
        # TODO: select slot 0 (the approacher), node cu.NOSE, ALL frames, and both (x, y) columns.
        #   - keep ':' on axis 0 so you get every frame (not just one),
        #   - put 0 on axis 1 to pick the approacher's slot,
        #   - put cu.NOSE on axis 2 to pick the nose keypoint,
        #   - keep ':' on axis 3 so you keep BOTH x and y.
        # Why it matters: if you drop a ':' you collapse an axis you meant to keep, and the plot below
        # will be flat or throw a shape error. Getting the axes right is the whole game with pose data.
        nose_xy = ____                      # want shape (130, 2): one (x, y) pair per frame
        ```

        **What you should see.** Two **green** curves against frame number — the approacher here is the
        Sub mouse, and we always color mice by rank (Sub = green). A **solid** line traces x
        (horizontal position) and a **dashed** line traces y (vertical position). Both wander up and
        down as the mouse moves over the ~2.6 s. Steep short segments are fast movement; flat segments
        are moments the mouse holds still. If a curve is flat at a single value, or you get a shape
        error, you indexed the wrong axis — check that `:` is on the first and last axes.
        """
    )
    return


@app.cell
def _(cu, ex_cols, ex_kp, go, np):
    # Reference solution (runs on load so the figure always renders). The blank line the student fills
    # is the first one below; everything else is provided.
    _nose_xy = ex_kp[:, 0, cu.NOSE, :]                 # (130, 2): approacher nose (x, y) per frame
    _t = np.arange(ex_kp.shape[0])
    _col = ex_cols[0]                                  # rank color of the approacher (Sub -> green)
    _fig = go.Figure()
    _fig.add_scatter(x=_t, y=_nose_xy[:, 0], mode="lines", name="x (px)",
                     line=dict(color=_col, width=2))
    _fig.add_scatter(x=_t, y=_nose_xy[:, 1], mode="lines", name="y (px)",
                     line=dict(color=_col, width=2, dash="dash"))
    _fig.update_layout(template="plotly_white", height=320,
                       title="Approacher nose position over time (Sub mouse, green)",
                       xaxis_title="frame", yaxis_title="pixel coordinate",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    _fig
    return


@app.cell(hide_code=True)
def _(cu, ex_kp, mo):
    mo.accordion({
        "Reveal solution — Exercise 1": mo.md(
            r"""
            ```python
            nose_xy = ex_kp[:, 0, cu.NOSE, :]     # slot 0, nose node, all frames, (x, y)
            #             |   |     |      |
            #          frames slot0 nose   x and y
            ```
            `ex_kp[:, 0, cu.NOSE, :]` keeps every frame (`:` on axis 0), selects slot `0` on the mice
            axis, selects the nose on the nodes axis (`cu.NOSE` is 0), and keeps both `x` and `y` on the
            last axis. The result has shape """
            + f"`{tuple(ex_kp[:, 0, cu.NOSE, :].shape)}`" + r""" — 130 rows (frames), 2 columns
            (x, y). Swapping the order of the indices, or replacing a `:` with a single number, would
            hand you the wrong slice.
            """)
    })
    return


@app.cell
def _(cu, ex_cols, ex_kp, np):
    # A labelled diagram of the skeleton so you know what the 15 keypoints are and how they connect.
    # There are no raw video frames in this teaching bundle, so instead of drawing on a photo we draw
    # one real pose on a blank canvas and print every keypoint's index and name. We take the
    # approacher's most-complete frame and color it by the approacher's rank (Sub = green).
    _nfin = np.isfinite(ex_kp[:, 0]).all(axis=2).sum(axis=1)   # tracked-node count per frame
    _best = int(np.argmax(_nfin))
    _pose = ex_kp[_best, 0]                                    # (15, 2) one mouse, one frame
    cu.labelled_skeleton_fig(_pose, color=ex_cols[0],
                             title="The 15-keypoint SLEAP skeleton (approacher, Sub = green)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Reading the diagram.** Each dot is one keypoint; each line is a skeleton *edge* connecting
        two keypoints. The layout is a **star** with two hubs. The **head (node 1)** anchors the front
        — nose, ears, shoulders, neck. The **TTI (node 11)**, the tail–torso junction, anchors the back
        — haunches, trunk, and the four tail keypoints (`tail_1`, `tail_0`, `tail_2`, `tail_tip`). Keep
        this picture in mind: when we later talk about a keypoint dropping out, or two mice being
        confused, this is the object it happens to. The head-to-TTI axis is also the mouse's facing
        direction, which becomes central in NB02.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    frame_slider = mo.ui.slider(0, 129, value=40, step=1, label="frame", debounce=True,
                                full_width=True)
    return (frame_slider,)


@app.cell
def _(cu, ex_cols, ex_cr, ex_kp, frame_slider, mo):
    # Scrub through the example event one frame at a time. Colors are by rank (Dom=red, Mid=blue,
    # Sub=green). As long as each color stays attached to one animal, identities are intact.
    _t = frame_slider.value
    _tag = "  ·  contact" if _t >= ex_cr else ""
    _fig = cu.skeleton_fig(ex_kp[_t], cu.SKELETON_EDGES, colors=ex_cols,
                           title=f"Example approach event — frame {_t}/129{_tag}", height=480)
    mo.vstack([
        mo.md(f"**Frame scrubber — the example approach event.** Contact begins at frame "
              f"**{ex_cr}**. Drag the slider and watch the three skeletons: each rank color should stay "
              "glued to one animal for the whole clip. The white line connects the approacher (Sub, "
              "green) to the approachee (Mid, blue); the third mouse is the bystander (Dom, red)."),
        frame_slider,
        _fig,
    ])
    return


@app.cell
def _(cu, ex_cr, ex_kp, ex_ranks, mo):
    # The same event as a short looping GIF (embedded as a data-URI so it animates in the notebook).
    _gif = cu.event_gif_bytes(ex_kp, ex_ranks, contact_rel=ex_cr, cell=200, fps=20)
    mo.vstack([mo.md("*The whole 2.6 s at a glance: the approacher (Sub, green) closes on the "
                     "approachee (Mid, blue) while the bystander (Dom, red) sits apart; the small red "
                     "dot in the corner marks the contact frames. This is a calm, non-aggressive "
                     "approach — a reminder that most encounters are not fights.*"),
               mo.Html(cu.gif_img_html(_gif, width=240))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Not every approach looks the same.** "Behavior" is not one thing, and a good first step is
        always to *look*. The clips below are real homecage recordings — the actual video the tracker
        ran on — with the tracked skeleton drawn on top and each mouse colored by rank (Dom = red,
        Mid = blue, Sub = green). Seeing the raw video alongside the skeleton is the honest way to
        judge what a coarse label like "aggression" really refers to. We are not measuring anything
        yet, just training the eye on the range of what we will later have to quantify.
        """
    )
    return


@app.cell
def _(cu, mo):
    # Video-backed exemplars: real homecage frames with the rank-colored skeleton overlaid, loaded
    # from data/exemplar_gifs/ (cu.load_asset_gif downloads them on a bare molab kernel). These five
    # are the genuinely fast, close, and directed approaches the lab labelled aggression — the
    # approacher drives straight at the partner and the gap collapses within a few frames.
    _fast = ["fast_directed_1.gif", "fast_directed_2.gif", "fast_directed_3.gif",
             "fast_directed_4.gif", "fast_directed_5.gif"]
    _html = "".join(cu.gif_img_html(cu.load_asset_gif(_n), width=195) for _n in _fast)
    mo.vstack([
        mo.md("**Fast, close, directed approaches (labelled aggression).** In every clip the "
              "approacher moves fast and straight at the partner and the distance between them "
              "collapses within a few frames — the closing is sustained and purposeful, not a slow "
              "drift. This is what the coarse label \"aggression\" looks like on the raw video."),
        mo.Html(_html),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Finer than "aggression."** "Aggression" is a coarse bucket. Within it the lab distinguishes
        finer bouts that look different on the video and mean different things:

        - **tail bite** — the aggressor's nose reaches the partner's tail base and bites; the two
          bodies line up nose-to-tail and the minimum inter-mouse distance goes to zero.
        - **mounting** — one mouse climbs over the other's rear, a distinct posture rather than a
          straight charge.
        - **pursuit (a chasing proxy)** — both mice move fast at once, roughly two to three times the
          speed of an ordinary approach: one flees and the other follows.

        Seeing these side by side shows why a single "aggression" label is not enough — the same word
        covers a bite, a mount, and a chase.
        """
    )
    return


@app.cell
def _(cu, mo):
    # Fine-grained aggressive bouts, all video-backed. tail_bite (nose-to-tail-base bite), mounting
    # (climbing posture), and pursuit (both mice fast = chase/flight). "chasing" is a proxy: the
    # bundle has no explicit chasing category, so these are the fastest two-mouse aggression events.
    _row = lambda _names: mo.Html("".join(cu.gif_img_html(cu.load_asset_gif(_x), width=200)
                                          for _x in _names))
    mo.vstack([
        mo.md("**Tail bite.** The aggressor's nose reaches the partner's tail base; the bodies align "
              "nose-to-tail and the minimum inter-mouse distance drops to zero."),
        _row(["tail_bite_1.gif", "tail_bite_2.gif"]),
        mo.md("**Mounting.** One mouse climbs over the other's rear — a posture, not a charge. Only a "
              "few clean mounting events exist in this data, so treat these two as illustrations."),
        _row(["mounting_1.gif", "mounting_2.gif"]),
        mo.md("**Pursuit (chasing proxy).** Both mice move fast at once — roughly 2-3x an ordinary "
              "approach — as one flees and the other follows. The bundle has no explicit chasing "
              "label, so these high-speed aggression events stand in for pursuit."),
        _row(["chasing_1.gif", "chasing_2.gif"]),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **And most contact is not a fight.** For contrast, here is the same rendering on
        non-aggressive social contact — investigation and grooming. The movement is slower and looser
        and the bodies do not drive together the way the aggressive bouts do. Keeping these in view
        guards against reading every approach as a fight.
        """
    )
    return


@app.cell
def _(cu, mo):
    # Non-aggressive social contact, video-backed: anogenital investigation, side-by-side contact,
    # and allogrooming. Slower and looser, with no sustained drive-in. Colors are by rank as above.
    _affil = ["anogenital_1.gif", "side_kissing_1.gif", "grooming_1.gif"]
    _html = "".join(cu.gif_img_html(cu.load_asset_gif(_n), width=200) for _n in _affil)
    mo.vstack([
        mo.md("**Non-aggressive social contact.** Left to right: anogenital investigation, "
              "side-by-side contact, and grooming. Colors are by rank, exactly as above."),
        mo.Html(_html),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 2 · Checking the signal: what is missing, and where

        **Why this matters.** Pose tracking is not perfect. Some keypoints are found more reliably than
        others, and which ones fail is not random — it follows the anatomy and the camera view. Before
        trusting any measurement, we ask which keypoints we can rely on.

        **Definition.** The *tracked fraction* of a keypoint is the fraction of frames (across all
        events and all mice) on which that keypoint is present — finite, not `NaN`. A value of 1.0
        means "found on every frame"; 0.5 means "found half the time."

        **The function we use.** `cu.node_reliability(kp)` takes a pose array (shape `(…, 15, 2)`) and
        returns a length-15 array giving each keypoint's tracked fraction over all frames it sees.
        """
    )
    return


@app.cell
def _(cu, ev, np):
    # Per-keypoint tracked fraction across the whole corpus, and the single least-reliable keypoint.
    nr = cu.node_reliability(ev["kp"])
    least_node = cu.NODE_NAMES[int(np.argmin(nr))]
    return least_node, nr


@app.cell
def _(cu, go, nr):
    # A lollipop chart (marker + stem), not a bar chart, so each keypoint reads as one data point.
    # Tail keypoints are highlighted because they are the weak link.
    _tail = {9, 10, 12, 13}
    _tip = cu.NODE_NAMES.index("tail_tip")
    _cols = ["#e45756" if i == _tip else ("#f2a25c" if i in _tail else "#4c78a8")
             for i in range(len(nr))]
    _fig = go.Figure()
    for _i, (_name, _v, _c) in enumerate(zip(cu.NODE_NAMES, nr, _cols)):
        _fig.add_scatter(x=[_name, _name], y=[0, _v], mode="lines",
                         line=dict(color=_c, width=2), showlegend=False, hoverinfo="skip")
    _fig.add_scatter(x=cu.NODE_NAMES, y=nr, mode="markers",
                     marker=dict(color=_cols, size=11, line=dict(width=0.5, color="white")),
                     hovertemplate="%{x}: %{y:.3f}<extra></extra>", showlegend=False)
    _fig.update_layout(template="plotly_white", height=360,
                       title="Per-keypoint tracked fraction — the tail keypoints drop out most",
                       yaxis_title="fraction of frames tracked", yaxis_range=[0, 1.02],
                       margin=dict(l=10, r=10, t=50, b=90))
    _fig.update_xaxes(tickangle=-45, showgrid=False)
    _fig
    return


@app.cell
def _(cu, ev, np):
    # Per-EVENT reliability: for each event, what fraction of (frame x mouse) slots is tracked, split
    # into the compact body (nodes 0-8) versus the tail chain (9, 10, 12, 13). This gives one number
    # per event per group -> a real distribution we can show point by point rather than a single mean.
    _ok = np.isfinite(ev["kp"]).all(axis=-1)                 # (N, T, 3, 15) True where tracked
    _body_nodes = list(range(9)); _tail_nodes = [9, 10, 12, 13]
    body_rel = _ok[:, :, :, _body_nodes].reshape(len(ev["kp"]), -1).mean(axis=1)   # (N,)
    tail_rel = _ok[:, :, :, _tail_nodes].reshape(len(ev["kp"]), -1).mean(axis=1)   # (N,)
    return body_rel, tail_rel


@app.cell
def _(body_rel, cu, np, tail_rel):
    # An ECDF is a clean way to compare two whole distributions: F(v) = fraction of events at or below
    # reliability v. The body curve hugs the right edge (almost always fully tracked); the tail curve
    # is shifted far left (routinely half-missing). Both curves use all 2499 events, not a summary bar.
    _vals = np.concatenate([body_rel, tail_rel])
    _grp = np.array(["body (nodes 0-8)"] * len(body_rel) + ["tail (9,10,12,13)"] * len(tail_rel))
    cu.ecdf_fig(_vals, _grp,
                group_order=["body (nodes 0-8)", "tail (9,10,12,13)"],
                colors={"body (nodes 0-8)": "#4c78a8", "tail (9,10,12,13)": "#e45756"},
                xlabel="per-event tracked fraction", title="Body vs tail reliability across all events")
    return


@app.cell(hide_code=True)
def _(least_node, mo, nr):
    mo.md(
        f"""
        The body keypoints are tracked about **0.96–0.98** of frames — nearly always present. The
        **tail keypoints** (`tail_1`, `tail_0`, `tail_2`, `tail_tip`) drop to roughly **0.73–0.78**,
        and the single least-reliable keypoint is **`{least_node}`** at **{nr.min():.3f}**. The ECDF
        makes the gap concrete: for the body, essentially every event is tracked above 0.95 (its curve
        stays flat at the bottom until the far right), while the tail curve rises across the whole range
        — a typical event has the tail present only about **0.80** of the time, and a long lower tail of
        events have it much worse.

        This is not a cosmetic problem. The lab identifies and ranks each mouse from marks painted on
        the **tail**, so an unreliable tail directly weakens those identity and rank labels — a known
        source of roughly **16% labelling error**. We carry that caveat forward: any result we later
        split by rank inherits this uncertainty, so we will prefer rank-free analyses where we can.
        """
    )
    return


@app.cell
def _(cu, ev, mo):
    # Show the tail dropout frame by frame so it is something you SEE, not just a statistic. In these
    # events the body stays intact while the tail chain blinks on and off — this is what a 0.73-0.78
    # tail tracked-fraction looks like in motion.
    _tail_events = [504, 954, 706]
    _events = [(ev["kp"][i], ev["ranks"][i], 40) for i in _tail_events]
    _gif = cu.grid_gif_bytes(_events, ncols=3, cell=180, fps=18)
    mo.vstack([
        mo.md(f"**Tail keypoints flickering (events {_tail_events}).** The body skeleton stays solid "
              "while the four tail dots repeatedly vanish and reappear. That flicker is exactly the "
              "reliability gap the ECDF above summarizes, and it is why tail-mark identity is the "
              "weakest link in the pipeline. Colors are by rank."),
        mo.Html(cu.gif_img_html(_gif, width=560)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 2b · Who is in the sample: two cohorts

        **Why this matters.** A measurement means little without knowing the population it came from.
        Our data is not one homogeneous pile of events — it was collected in **two separate cohorts**,
        recorded on different dates, and each cohort holds several cages of mice. Knowing that structure
        now saves us from a serious mistake later: events from the same cage are not independent of one
        another, so the *cage* — not the individual event — is often the right unit of analysis. We set
        that idea up here and use it in earnest in NB04.

        **Definitions.**

        - A **cohort** is one full run of the experiment — a batch of cages recorded together over the
          same span. We have two, identified only by their date tags `20260222` and `12192025`.
        - A **cage** holds three mice living together. The derived data gives each event a
          **cohort-unique** cage id, computed as `cohort_index * 100 + camera`. So cages `9–15` are the
          first cohort and cages `109–115` are the second, and a cage id never collides across cohorts.
        - Each cage is a single **sex** (all-male or all-female), which is why sex and cage are tied
          together — a fact that will matter a great deal when we test for sex differences in NB04.
        """
    )
    return


@app.cell
def _(der, ev, np):
    # Build the sample-composition summary from the cohort-unique fields in the derived bundle.
    cages = np.unique(der["cage"])
    cage_n = np.array([int((der["cage"] == c).sum()) for c in cages])       # events per cage
    cage_sex = np.array([der["sex"][der["cage"] == c][0] for c in cages])   # each cage's single sex
    cage_cohort = np.array([str(der["cohort"][der["cage"] == c][0]) for c in cages])
    cond_names, cond_counts = np.unique(ev["condition"], return_counts=True)
    return cage_cohort, cage_n, cage_sex, cages, cond_counts, cond_names


@app.cell
def _(cage_cohort, cage_n, cages, cu):
    # Events per cage, one point per cage, grouped by cohort. A strip plot (individual points) instead
    # of a bar chart, so you see that cages differ in how many events they contributed. Hover shows the
    # cage id.
    cu.strip_points_fig(cage_n, cage_cohort,
                        group_order=sorted(set(cage_cohort.tolist())),
                        hover=[f"cage {c}" for c in cages], jitter=0.12, point_size=12,
                        ylabel="events contributed", xlabel="cohort (date tag)",
                        title="Events per cohort-unique cage (14 cages, 7 per cohort)")
    return


@app.cell(hide_code=True)
def _(cage_cohort, cage_n, cage_sex, cond_counts, cond_names, mo):
    # Plain-language composition summary computed from the data.
    _by_cohort = {}
    for _co, _n in zip(cage_cohort, cage_n):
        _by_cohort[_co] = _by_cohort.get(_co, 0) + int(_n)
    _nM = int(sum(int(n) for s, n in zip(cage_sex, cage_n) if s == "M"))
    _nF = int(sum(int(n) for s, n in zip(cage_sex, cage_n) if s == "F"))
    _cohorts = ", ".join(f"`{k}` ({v} events)" for k, v in sorted(_by_cohort.items()))
    _conds = " / ".join(f"{c}={n}" for c, n in zip(cond_names, cond_counts))
    mo.md(
        f"""
        **The sample in numbers.** Two cohorts — {_cohorts} — for **{int(sum(cage_n))} training
        events** across **{len(cage_n)} cohort-unique cages**, balanced **7 male / 7 female** cages
        (**{_nM}** male-cage events and **{_nF}** female-cage events). Each cage sits in exactly one
        cohort and is one sex, so grouping by cage never mixes cohorts or sexes — that is what makes the
        cage a safe unit of analysis.

        The events also carry a **condition** label from the experiment's manipulation
        (**{_conds}**), which we set aside for now and return to when we ask what actually changes
        behavior in NB04. For this notebook the point is simply that our data has structure, and we
        will respect it.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 2c · What "contact" physically means

        **Why this matters.** These events were extracted as *approaches*: two mice start apart and one
        closes on the other until they touch. If we are going to build measurements around the moment
        of contact, we should first confirm that the geometry matches the story. We do that with the
        simplest possible summary of an interaction — the distance between the two mice.

        **Definition.** A mouse's **centroid** is the average position of its body keypoints — a single
        point standing in for where the whole body is. `cu._centroids(mouse_kp)` computes it per frame,
        ignoring missing keypoints. The **inter-mouse distance** is the pixel distance between the
        approacher's and approachee's centroids.
        """
    )
    return


@app.cell
def _(cu, ex_cr, ex_kp, go, np):
    # Inter-mouse distance over the example event: it falls as the approach happens.
    _c0 = cu._centroids(ex_kp[:, 0])                    # approacher centroid per frame (T,2)
    _c1 = cu._centroids(ex_kp[:, 1])                    # approachee centroid per frame (T,2)
    _d01 = np.linalg.norm(_c0 - _c1, axis=1)            # centroid-to-centroid distance (px)
    _fig = go.Figure()
    _fig.add_scatter(y=_d01, mode="lines", line=dict(color="#333", width=2),
                     name="approacher–approachee distance")
    _fig.add_hline(y=150, line=dict(color="#e45756", dash="dash"),
                   annotation_text="≈ contact distance (150 px)", annotation_position="top left")
    _fig.add_vline(x=ex_cr, line=dict(color="#888", dash="dot"),
                   annotation_text="contact frame", annotation_position="top right")
    _fig.update_layout(template="plotly_white", height=320,
                       title="Example event — inter-mouse distance falls as the approach happens",
                       xaxis_title="frame", yaxis_title="pixels", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    _fig
    return


@app.cell
def _(cu, ev, np):
    # Corpus-wide closest-approach distance: for each event, the smallest inter-mouse centroid distance
    # it reaches. Shown as an ECDF so the whole distribution is visible.
    _kp = ev["kp"]
    _closest = np.empty(len(_kp))
    for _i in range(len(_kp)):
        _c0 = cu._centroids(_kp[_i, :, 0]); _c1 = cu._centroids(_kp[_i, :, 1])
        _d = np.linalg.norm(_c0 - _c1, axis=1)
        _closest[_i] = np.nanmin(_d) if np.isfinite(_d).any() else np.nan
    closest_dist = _closest
    return (closest_dist,)


@app.cell
def _(closest_dist, cu):
    cu.ecdf_fig(closest_dist, xlabel="closest-approach distance (px)",
                title="How close the two mice actually get (all events)")
    return


@app.cell(hide_code=True)
def _(closest_dist, mo, np):
    _med = float(np.nanmedian(closest_dist))
    _q1, _q3 = [float(x) for x in np.nanpercentile(closest_dist, [25, 75])]
    mo.md(
        f"""
        **What the plots show.** In the example event the two centroids start about **289 px** apart,
        fall to **151 px** at the contact frame, and reach a closest approach of **136 px** — a
        gentle, non-aggressive approach that never gets very tight. Across the whole corpus the median
        closest approach is about **{_med:.0f} px** (middle half of events between **{_q1:.0f}** and
        **{_q3:.0f} px**), and the median event starts near **222 px** and reaches about **155 px** at
        contact. The geometry matches the label: these really are approaches, with the bodies drawing
        together over the clip. "Contact" is not a mystical event — it is simply the mice getting close,
        and the dashed line marks the distance where we start calling it that.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 3 · A slot is not an identity: the swap problem

        **Why this matters.** Every label we build this week — who approached whom, who was aggressive —
        assumes the mouse in slot 0 stays the same animal for all 130 frames. If the tracker ever swaps
        two mice, that assumption breaks and the label can invert. So we need to understand when swaps
        happen and whether we can catch them.

        **Definition.** An **identity swap** is when the tracker exchanges two mice between slots at
        some frame. From that frame on, "the approacher" (slot 0) is really the other mouse, and any
        directional claim about the event points the wrong way.

        **How a swap detector works.** A mouse cannot teleport, so one way to flag a swap is to watch
        each track's **centroid velocity** — how far its body centroid moves from one frame to the
        next, in pixels per frame. (This is a *normalized* velocity: displacement divided by the time
        step, so it is a per-frame rate that we can compare across events regardless of clip length.) A
        sudden large jump in centroid velocity is suspicious, because real mice move smoothly.

        **The catch.** If you swap two tracks at a frame, the jump this creates is exactly *how far
        apart the two mice were* at that frame. So a swap that happens **at contact** — where the mice
        are nearly on top of each other — produces only a **tiny** jump that slips under any threshold.
        Swaps are therefore hardest to detect precisely where the mice are closest, which is exactly
        where aggression happens. Drag the threshold below to see the blind spot form.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    thr_slider = mo.ui.slider(10, 300, value=80, step=5, label="swap-detector threshold (px)",
                              debounce=True, full_width=True)
    return (thr_slider,)


@app.cell
def _(cu, ex_cr, ex_kp, go, mo, np, thr_slider):
    # For the example event: at every frame, the jump a slot 0<->1 swap would create equals the
    # distance between those two tracks' centroids at that frame. Frames where that jump is below the
    # detector threshold are blind spots — a swap there would go unnoticed.
    _c0 = cu._centroids(ex_kp[:, 0]); _c1 = cu._centroids(ex_kp[:, 1])
    _induced = np.linalg.norm(_c0 - _c1, axis=1)          # jump a 0<->1 swap would create
    _thr = thr_slider.value
    _undetectable = _induced < _thr
    _fig = go.Figure()
    _fig.add_scatter(y=_induced, mode="lines", line=dict(color="#1f77b4", width=2),
                     name="jump a swap would create")
    _fig.add_hline(y=_thr, line=dict(color="#e45756", dash="dash"),
                   annotation_text=f"detector threshold = {_thr} px", annotation_position="top left")
    _frames = np.arange(len(_induced))
    _fig.add_scatter(x=_frames[_undetectable], y=_induced[_undetectable], mode="markers",
                     marker=dict(color="#e45756", size=6), name="swap here would be missed")
    _fig.add_vline(x=ex_cr, line=dict(color="#888", dash="dot"),
                   annotation_text="contact", annotation_position="bottom right")
    _fig.update_layout(template="plotly_white", height=360,
                       title="Where would a swap slip past the detector? (example event)",
                       xaxis_title="frame", yaxis_title="induced jump (px)",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    _n_blind = int(_undetectable.sum())
    mo.vstack([
        thr_slider,
        _fig,
        mo.md(f"At this threshold, **{_n_blind}/{len(_induced)} frames** are blind spots (red), and "
              "they cluster around **contact**, where the two mice are closest. Lower the threshold and "
              "you catch more swaps but also flag honest fast movement as a swap (false alarms); raise "
              "it and swaps at contact disappear entirely. No single threshold avoids both problems — "
              "which is the whole difficulty of keeping identities straight."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Exercise 2 — which events contain an identity swap?

        **Python skill: calling a helper, then array arithmetic and a boolean mask.** You will run a
        provided function over five events, then use a comparison to decide which events swapped. This
        is the next rung up from Exercise 1: you still index, but now you also *compute on* the result
        and *filter* with a condition.

        **The setup.** You are given five events. Three are clean; in two of them we deliberately
        introduced a swap — we exchanged slots 0 and 1 for a few frames to simulate a tracking error.
        Your job is to plot each event's centroid velocities and decide which two swapped.

        **The tool.** `cu.centroid_jumps(clip)` takes one event's pose array (shape `(T, 3, 15, 2)`)
        and returns an array of shape **`(3, T-1)`**: one row per mouse, each value the distance that
        mouse's centroid moved since the previous frame (pixels per frame). A clean event has three low
        traces. A swap shows up as a **single tall spike shared by two mice at the same frame** —
        because both tracks jump by the inter-mouse distance at the instant they are exchanged.

        **What to edit.** One line below is blank:

        ```python
        # clip is one event's pose array, shape (T, 3, 15, 2).
        # TODO: compute the per-mouse centroid velocity for this clip by CALLING the helper.
        #   Replace ____ with cu.centroid_jumps(clip).
        # Why it matters: this single call turns raw coordinates into the per-frame movement signal a
        #   swap detector actually watches. The plotting code below expects `jumps` to have shape
        #   (3, 129) — three rows, one per mouse. If you index or transpose it, the panels break.
        jumps = ____
        ```

        **What you should see.** Five small panels, one per event. **Three** show three low traces
        (ordinary movement peaking well under ~90 px/frame). **Two** show a single very tall spike
        (roughly 130–170 px/frame) shared by two of the traces at the same frame — those are the
        swapped events. Note the two event numbers with the tall spike, then open the reveal to check
        your answer and watch the swap happen in the rendered GIFs.
        """
    )
    return


@app.cell
def _(cu, ev, np):
    # Build the five-event exercise set. We deliberately induce a swap in two of them by exchanging
    # track slots 0 and 1 for frames [40, 48) (near contact, so the induced jump is modest — the very
    # blind spot Section 3 warned about). The reference computation runs on load so the plot and
    # self-check always work.
    swap_events = [1496, 95, 77, 1121, 110]

    def _induce_swap(kp_event, f0, f1):
        """Swap track slots 0<->1 for frames [f0, f1) to simulate an identity swap."""
        out = kp_event.copy()
        out[f0:f1, [0, 1]] = out[f0:f1, [1, 0]]
        return out

    swap_clips = []
    for _pos, _idx in enumerate(swap_events):
        _k = ev["kp"][_idx].copy()
        if _pos in (1, 3):                                 # events 95 and 1121 get a swap
            _k = _induce_swap(_k, 40, 48)
        swap_clips.append(_k)

    # cu.centroid_jumps(clip) -> (3, T-1) per-mouse centroid velocity (px/frame).
    swap_maxjump = np.array([float(np.nanmax(cu.centroid_jumps(c))) for c in swap_clips])
    SWAP_THR = 100.0
    swap_detected = [swap_events[i] for i in range(5) if swap_maxjump[i] > SWAP_THR]
    return SWAP_THR, swap_clips, swap_detected, swap_events, swap_maxjump


@app.cell
def _(cu, ev, go, swap_clips, swap_events):
    # Plot the three centroid-velocity traces for each of the five events. Lines colored by rank.
    from plotly.subplots import make_subplots as _make_subplots
    _fig = _make_subplots(rows=1, cols=5, shared_yaxes=True,
                          subplot_titles=[f"event {i}" for i in swap_events])
    for _col, (_idx, _clip) in enumerate(zip(swap_events, swap_clips), start=1):
        _jumps = cu.centroid_jumps(_clip)                 # (3, T-1) — the line the student fills
        _ranks = ev["ranks"][_idx]
        for _m in range(3):
            _c = cu.RANK_HEX.get(int(_ranks[_m]), cu.RANK_HEX[0])
            _fig.add_scatter(y=_jumps[_m], mode="lines", line=dict(color=_c, width=1.5),
                             showlegend=False, row=1, col=_col)
    _fig.update_layout(template="plotly_white", height=300,
                       title="Centroid velocity per mouse — spot the two events with a tall spike",
                       margin=dict(l=10, r=10, t=60, b=30))
    _fig.update_yaxes(title_text="px / frame", row=1, col=1)
    _fig.update_xaxes(showgrid=False)
    _fig
    return


@app.cell(hide_code=True)
def _(cu, ev, mo, swap_clips):
    # Reveal: render the two swapped events so the swap is visible. During frames 40-47 the two
    # skeletons jump to each other's positions, then jump back.
    _g95 = cu.event_gif_bytes(swap_clips[1], ev["ranks"][95], contact_rel=40, cell=200, fps=18)
    _g1121 = cu.event_gif_bytes(swap_clips[3], ev["ranks"][1121], contact_rel=40, cell=200, fps=18)
    mo.accordion({
        "Reveal answer — Exercise 2": mo.vstack([
            mo.md(
                r"""
                **The swapped events are 95 and 1121.** The three clean events (1496, 77, 110) top out
                at ordinary movement — their maximum centroid velocity stays below the detector
                threshold. The two swapped events spike above it: event **95** jumps to about
                **169 px/frame** and event **1121** to about **131 px/frame**, each spike equal to how
                far apart the two mice were at the swapped frame. Note how modest those spikes are —
                because we swapped *near contact*, the jump is smaller than it would be if the mice were
                far apart, exactly the blind spot from Section 3. A threshold near 100 px separates
                these five cleanly, but the margin is narrow, which is the real lesson.

                In the GIFs below (the two swapped events), watch frames 40–47: the skeletons jump to
                each other's positions and then snap back. That jump is the swap.
                """),
            mo.Html(cu.gif_img_html(_g95, width=220) + cu.gif_img_html(_g1121, width=220)),
        ])
    })
    return


@app.cell(hide_code=True)
def _(SWAP_THR, mo, swap_detected, swap_events, swap_maxjump):
    # Self-check: the events flagged by the threshold should be exactly {95, 1121}.
    _ok = set(swap_detected) == {95, 1121}
    _bg = "#e6f4ea" if _ok else "#fce8e6"
    _bd = "#34a853" if _ok else "#ea4335"
    _mark = "PASS" if _ok else "CHECK"
    _rows = "".join(f"<li>event {e}: max {m:.0f} px/frame</li>"
                    for e, m in zip(swap_events, swap_maxjump))
    mo.md(
        f"""
        <div style="background:{_bg}; border-left:6px solid {_bd}; padding:12px 16px;
        border-radius:6px;">
        <b>{_mark}</b><br>
        Maximum centroid velocity per event:
        <ul style="margin:6px 0;">{_rows}</ul>
        Events above the {SWAP_THR:.0f} px/frame threshold: <b>{sorted(swap_detected)}</b>
        &nbsp;→ {"matches the answer key {95, 1121}." if _ok else "expected {95, 1121}."}<br>
        <span style="color:#555;">The two swapped events clear the threshold while the three clean
        events stay below it. Graded against the pinned answer key.</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Exercise 3 — is "slot 0 = the approacher" a real label?

        **Python skill: writing a small function.** So far you have indexed an array and called a
        helper. Now you write a two-line computation of your own inside a function. This is the rung
        the next notebook builds on.

        **The idea.** The files *assert* that slot 0 is the approacher. Rather than trust the label, we
        test a prediction that follows from it: if slot 0 really is the mouse doing the approaching, it
        should **move more than slot 1 in the frames just before contact**. We check this across all
        events at once.

        **Definitions.** *Speed* here is the distance a keypoint moves between consecutive frames
        (pixels per frame). We use the **TTI** keypoint (the tail–torso junction, node 11) as a stable
        stand-in for body position. `np.diff` takes frame-to-frame differences; `np.linalg.norm` turns
        an (x, y) difference into a single distance; `np.nanmean` averages while ignoring untracked
        (`NaN`) frames.

        **What to edit.** One line inside the function is blank:

        ```python
        def mean_pre_speed(k, cr, m):
            t0 = max(0, cr - 50)                 # start 50 frames before contact
            tti = k[t0:cr, m, cu.TTI, :]         # that mouse's TTI track over the window, shape (win, 2)
            if len(tti) < 2:
                return np.nan
            # TODO: per-frame speed = the length of the frame-to-frame change in position.
            #   Replace ____ with:  np.linalg.norm(np.diff(tti, axis=0), axis=1)
            #   - np.diff(tti, axis=0) gives the (x, y) step between consecutive frames, shape (win-1, 2)
            #   - np.linalg.norm(..., axis=1) collapses each (x, y) step to one distance, shape (win-1,)
            # Why it matters: this is how a raw position track becomes a speed. Get the axes wrong and
            #   you average positions instead of movements, and the test becomes meaningless.
            step = ____
            return np.nanmean(step)
        ```

        **What you should see.** A scatter with one point per event: the approacher's pre-contact speed
        on the x-axis, the approachee's on the y-axis, with the diagonal drawn in. Most points fall
        **below** the diagonal (approacher faster). The green PASS box reports the fraction of events
        below the diagonal; it should land near **0.68**, well above the 0.5 you would get by chance,
        with a tiny p-value, and confirm `tail_tip` as the least-reliable keypoint.
        """
    )
    return


@app.cell
def _(cu, ev, np):
    # Reference solution (runs on load so the self-check can grade; a ~2500-event loop, well under 1s).
    def _mean_pre_speed(k, cr, m):
        _t0 = max(0, cr - 50)
        _tti = k[_t0:cr, m, cu.TTI, :]                    # (window, 2)
        if len(_tti) < 2:
            return np.nan
        _step = np.linalg.norm(np.diff(_tti, axis=0), axis=1)   # px/frame
        return np.nanmean(_step) if np.isfinite(_step).any() else np.nan

    _kp = ev["kp"]; _cr = ev["contact_rel"].astype(int)
    s0_pre = np.array([_mean_pre_speed(_kp[i], _cr[i], 0) for i in range(len(_kp))])
    s1_pre = np.array([_mean_pre_speed(_kp[i], _cr[i], 1) for i in range(len(_kp))])
    valid_pre = np.isfinite(s0_pre) & np.isfinite(s1_pre)
    n_valid = int(valid_pre.sum())
    n_more = int(np.sum(s0_pre[valid_pre] > s1_pre[valid_pre]))
    frac_more = n_more / n_valid

    from scipy.stats import binomtest
    p_more = binomtest(n_more, n_valid, 0.5).pvalue
    return frac_more, n_valid, p_more, s0_pre, s1_pre, valid_pre


@app.cell
def _(go, np, s0_pre, s1_pre, valid_pre):
    # Scatter of approacher vs approachee pre-contact speed, one point per event. Points below the
    # diagonal are events where the approacher moved more. Coloring the two sides makes the majority
    # visible; hover shows each event's coordinates.
    _x = s0_pre[valid_pre]; _y = s1_pre[valid_pre]
    _below = _x > _y
    _hi = float(np.nanpercentile(np.concatenate([_x, _y]), 99))
    _fig = go.Figure()
    _fig.add_scatter(x=_x[_below], y=_y[_below], mode="markers", name="approacher faster",
                     marker=dict(size=4, color="#4c78a8", opacity=0.5))
    _fig.add_scatter(x=_x[~_below], y=_y[~_below], mode="markers", name="approachee faster",
                     marker=dict(size=4, color="#e45756", opacity=0.5))
    _fig.add_scatter(x=[0, _hi], y=[0, _hi], mode="lines", line=dict(color="#333", dash="dot"),
                     name="equal speed", showlegend=True)
    _fig.update_layout(template="plotly_white", height=430,
                       title="Pre-contact speed: approacher (slot 0) vs approachee (slot 1)",
                       xaxis_title="approacher speed (px/frame)", yaxis_title="approachee speed (px/frame)",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(range=[0, _hi], showgrid=False); _fig.update_yaxes(range=[0, _hi], showgrid=False)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Reveal solution — Exercise 3": mo.md(
            r"""
            ```python
            def mean_pre_speed(k, cr, m):
                t0 = max(0, cr - 50)
                tti = k[t0:cr, m, cu.TTI, :]                        # (window, 2)
                if len(tti) < 2:
                    return np.nan
                step = np.linalg.norm(np.diff(tti, axis=0), axis=1)   # px/frame
                return np.nanmean(step)

            s0 = np.array([mean_pre_speed(kp[i], cr[i], 0) for i in range(len(kp))])
            s1 = np.array([mean_pre_speed(kp[i], cr[i], 1) for i in range(len(kp))])
            v  = np.isfinite(s0) & np.isfinite(s1)
            frac = np.mean(s0[v] > s1[v])                          # about 0.68

            least = cu.NODE_NAMES[np.argmin(cu.node_reliability(kp))]   # 'tail_tip'
            ```
            The approacher out-moving the approachee about **68%** of the time confirms that slot 0 is
            a real role, not a coin flip. It is not 100%: passive co-approaches, mutual approaches, and
            the occasional identity swap live in the other third — the label noise we spend the rest of
            the week working to reduce.
            """)
    })
    return


@app.cell(hide_code=True)
def _(cu, ev, frac_more, mo, n_valid, np, p_more):
    # Tolerance-band self-check: the fraction is well above 0.5 with a tiny p-value, and the
    # least-reliable keypoint is tail_tip.
    _least = cu.NODE_NAMES[int(np.argmin(cu.node_reliability(ev["kp"])))]
    _ok_frac = (0.62 <= frac_more <= 0.75) and (p_more < 1e-6)
    _ok_node = (_least == "tail_tip")
    _ok = _ok_frac and _ok_node
    _bg = "#e6f4ea" if _ok else "#fce8e6"
    _bd = "#34a853" if _ok else "#ea4335"
    _mark = "PASS" if _ok else "CHECK"
    mo.md(
        f"""
        <div style="background:{_bg}; border-left:6px solid {_bd}; padding:12px 16px;
        border-radius:6px;">
        <b>{_mark}</b><br>
        Fraction of events with <b>approacher &gt; approachee</b> (pre-contact TTI speed):
        <b>{frac_more:.3f}</b> over {n_valid} events &nbsp;(sign test p = {p_more:.1e}).
        &nbsp;Target band: <b>0.62–0.75</b>, p &lt; 1e-6 → {"met." if _ok_frac else "not met."}<br>
        Least-reliable keypoint found: <b>{_least}</b> →
        {"tail_tip, as predicted." if _ok_node else "expected tail_tip."}<br>
        <span style="color:#555;">The approacher really does move more, so the role label is honest at
        the population level, and the tail tip really is the weakest keypoint. Graded against values
        pinned at build time.</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Review questions": mo.md(
            r"""
            **1. Why does a single identity swap corrupt a "who approached whom" label more than a few
            dropped keypoints do?** A dropped keypoint removes one measurement on one frame — a small,
            local gap we can average over with `nanmean`. A swap relabels the whole event: every frame
            after the swap assigns the behavior to the wrong mouse, so the *direction* of the behavior
            flips. Missing data adds noise, which averages out; a swap adds a systematic error, which
            does not.

            **2. Why are swaps hardest to catch exactly when they matter most?** A velocity-based
            detector flags a swap by the size of the jump it creates, and that jump equals the distance
            between the two mice. During contact the mice are closest, so the jump is smallest — right
            where aggression happens, the swap is nearly invisible (Section 3, Exercise 2).

            **3. Why is the cage, not the event, often the right unit of analysis?** Events from one
            cage share three specific animals, one arena, and one recording session, so they are not
            independent. Treating hundreds of events from a handful of cages as independent samples
            (pseudoreplication) inflates our confidence. Because each cage here is cohort-unique and
            single-sex, grouping by cage gives us clean, independent units — the tool we rely on in
            NB04.

            **4. Further reading.** The pose tracking used here is SLEAP (Pereira et al., 2022,
            *Nature Methods*); a related tool is DeepLabCut (Mathis et al., 2018, *Nature
            Neuroscience*). Both estimate body keypoints from video; neither, on its own, solves the
            identity-over-time problem we examined today.
            """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## What we answered, and what comes next

        **The question was:** what is behavior, and can we trust how we measure it? We answered it in
        three parts.

        1. **What behavior is, to us:** a `(frames, mice, keypoints, xy)` pose tensor — a table of body
           landmarks over time — that we can index, measure, and compare across animals and across two
           cohorts.
        2. **Where it is trustworthy:** the body keypoints are tracked ~97% of frames, and the
           "approacher" role is real at the population level (the approacher out-moves the approachee
           ~68% of the time). We can build on both.
        3. **Where it is not:** the tail keypoints are missing ~20–25% of the time, which weakens the
           tail-mark identity and rank labels (~16% error), and identity swaps are nearly undetectable
           at contact — exactly where the interesting behavior happens. We carry both caveats forward.

        We also met the shape of the data: two cohorts, fourteen cohort-unique single-sex cages, and a
        condition manipulation we have not yet touched.

        **The next question.** Right now an event is described by where each mouse sits in the camera's
        pixel grid, so the same behavior looks completely different depending on which corner of the
        cage it happens in. That is no way to compare interactions. **NB02 asks: how do we describe an
        interaction regardless of where in the cage it happens?** The answer is to re-express every
        event from the approacher's own body frame — an allocentric transform — and reduce it to a
        handful of interpretable, location-free features.
        """
    )
    return


if __name__ == "__main__":
    app.run()
