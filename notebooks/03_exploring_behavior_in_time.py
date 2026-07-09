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
def _():
    from plotly.subplots import make_subplots
    return (make_subplots,)


# ============================================================ 1. Throughline: where we are
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # NB03 · Exploring behavior in time

        ## The question so far, and the question now

        We study social behavior and its neural basis. In the first notebooks we turned raw video
        into **pose**: for each mouse, in every video frame, the pixel position of 15 body
        **keypoints** (a keypoint is a labelled dot on the body — nose, ears, shoulders, tail base).
        We then rebuilt each interaction from a **body's-eye view** and reduced it to **19 numbers per
        event** — allocentric features such as how fast the mice close the gap or how directly one
        faces the other.

        Those notebooks answered: *how do we turn a video of two mice into numbers we can compute
        with?* This notebook asks the next question:

        > **What does a single interaction actually look like as a signal over time?**

        This is **exploratory data analysis (EDA)** — the unglamorous, essential first pass where we
        *look* at the data before we model it. We take our time here. Two notebooks from now we will
        **compress** each event into a handful of numbers and, in doing so, average away *time* — the
        moment-to-moment unfolding of the encounter. Before we throw time away, we should understand
        what it holds. We will ask three concrete questions about an interaction:

        1. **What is the shape of each measurement** across many events? (its *distribution*)
        2. **How does a mouse move through time, and at what rhythm?** (a *time–frequency* view)
        3. **Do two mice move together, and can that tell us who is interacting with whom?**
           (*coordination* between animals)

        Along the way we practice a specific Python skill: **writing a small function** and using
        **`np.diff`** to turn positions into motion, the vectorized way.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Definitions you need first

        **An approach event.** The dataset is a collection of short clips, each about 2.6 seconds long
        (130 frames at 50 frames per second). Each clip was cut out by the tracking pipeline as an
        *approach*: two mice **start far apart** (body centers roughly 200 px apart) and **close the
        distance** until they are in contact (centers about 150 px apart, noses nearly touching). Every
        clip is therefore centered on the same kind of moment — one animal coming up to another — with
        **contact** at a fixed frame (frame 40, 0.8 s in). There are **2,499** such training clips.

        **Three mice, named by role.** Each clip contains three animals, stored in a fixed order:

        - the **approacher** — the mouse that closes the distance (mouse 0),
        - the **approachee** — the mouse being approached (mouse 1),
        - the **bystander** — the third mouse, present but not the focus (mouse 2).

        **Rank, and how we color mice.** Each mouse has a social **rank** measured separately by the
        lab. Throughout every notebook, mice are colored **only by rank**: **Dom = red**, **Int/Mid =
        blue**, **Sub = green** (gray = unknown). Color never means anything else about a mouse.

        **The aggression label is human-scored ground truth.** Some approaches escalate into aggression
        and some do not. A member of the lab **watched each clip and hand-scored it** as aggression or
        not. These labels are **ground truth produced by a human observer**, not a guess from a model.
        About **32%** of the training clips (base rate 0.320) are labelled aggression. We will often
        split plots by this label to see where aggression differs from ordinary approach.

        **Two cohorts.** The clips come from **two independent groups of animals** recorded on separate
        dates (we will call them cohort A and cohort B; 1,282 + 1,217 events). Each home cage belongs
        to exactly one cohort and one sex, so a cage is the natural independent unit — a fact that
        becomes important when we test claims later. One additional cage (`cam16`, all female, 780
        events) is **held out entirely** and never touched until a later notebook, so we can check our
        methods on animals they were never tuned on.

        **Coordination previews "who interacts with whom."** With three animals in frame, a natural
        question is: which two are actually engaged? A simple physical clue is **coordination** — if
        two mice speed up and slow down *together*, their movements are **correlated**, and correlation
        hints that they are interacting rather than moving independently. We make this precise in
        Section 4 and test exactly how far the hint can be trusted.
        """
    )
    return


# ============================================================ data + shared helpers
@app.cell
def _(ROOT, cu):
    ev = cu.load_events(cu.data_path("data/train_events.npz", ROOT))
    der = cu.load_derived("train", ROOT)
    return der, ev


@app.cell
def _(der, ev):
    agg = ev["agg_label"].astype(int)
    cage = der["cage"]
    sexv = der["sex"].astype(str)
    cohort = der["cohort"].astype(str)
    cond = ev["condition"].astype(str)
    cr = ev["contact_rel"].astype(int)
    kp = ev["kp"]
    ranks = ev["ranks"]
    X = der["X"]
    fnames = [str(f) for f in der["feature_names"]]
    return X, agg, cage, cohort, cond, cr, fnames, kp, ranks, sexv


@app.cell
def _(cu, np):
    # Shared helpers, defined once so the reactive graph stays clean. Each one is a small function
    # built on np.diff — exactly the pattern the exercise asks you to reproduce yourself.

    def mouse_speed(kp_event, m):
        """Per-frame speed (px/frame) of mouse m in world coordinates.
        kp_event (T,3,15,2); m in {0,1,2}. Returns (T-1,).
        Speed = length of the change in body-center position from one frame to the next.
        `np.diff(cen, axis=0)` is the frame-to-frame displacement vector; its norm is the speed."""
        cen = cu._centroids(kp_event[:, m])                      # (T,2) body-center track
        return np.nan_to_num(np.linalg.norm(np.diff(cen, axis=0), axis=1))

    def appr_appe_speed(kp, i):
        """Speed traces of the approacher (mouse 0) and approachee (mouse 1) for event i."""
        return mouse_speed(kp[i], 0), mouse_speed(kp[i], 1)

    def pair_peak_corr(kp_event, a, b, max_lag=15):
        """Coordination of two mice: the peak (best over small time-shifts) cross-correlation of
        their speed traces. Near +1 = move together; near 0 or negative = independent."""
        _, c, _ = cu.cross_corr_lag(mouse_speed(kp_event, a), mouse_speed(kp_event, b), max_lag)
        return float(c.max())

    def pre_speeds(kp, cr, i, win=50):
        """The two speed traces over the `win` frames just BEFORE contact — the window where the
        approacher and the true first-mover can dissociate."""
        s0, s1 = appr_appe_speed(kp, i)
        c = int(cr[i]); a = max(1, c - win)
        return s0[a - 1:c - 1], s1[a - 1:c - 1]

    def leader_fraction(kp, cr, idxs, max_lag=10):
        """Fraction of events where the approacher LEADS (peak cross-corr lag > 0). ~0.5 = no
        consistent leader. Skips traces too short/flat for the requested lag. Returns (fraction, n)."""
        leads = []
        for i in idxs:
            x, y = pre_speeds(kp, cr, i)
            if len(x) < 2 * max_lag + 4 or x.std() < 1e-6 or y.std() < 1e-6:
                continue
            _, _, pk = cu.cross_corr_lag(x, y, max_lag)
            if pk != 0:
                leads.append(1 if pk > 0 else 0)
        return (float(np.mean(leads)) if leads else float("nan"), len(leads))

    def padded_wavelet(sig, freqs, fps, padlen=600):
        """Reflect-pad a short signal so low-frequency Morlet kernels fit, run cu.wavelet_power, then
        crop back to the original span. The padded flanks are the EDGE-EFFECT region we warn about."""
        sig = np.nan_to_num(np.asarray(sig, float)); T = len(sig)
        pad = max(0, (padlen - T) // 2)
        sp = np.pad(sig, pad, mode="reflect")
        P = cu.wavelet_power(sp, freqs, fps)
        return P[:, pad:pad + T]
    return (appr_appe_speed, leader_fraction, mouse_speed, padded_wavelet,
            pair_peak_corr, pre_speeds)


@app.cell
def _(cu, ev):
    # Pinned build-time constants (verified against the committed 2-cohort bundle).
    EXAMPLE = cu.event_index_by_key(ev, "12192025_pre|cam.10.00046-2025-12-18T16|m0-m2|83141")  # example event: cage 110 (cohort B), female, NON-aggression, contact @40
    CLEAN_AGG = [969, 560, 900, 53]        # clean, well-tracked aggression approaches
    CLEAN_NON = [161, 341, 376, 345]       # clean, well-tracked non-aggression approaches
    DOM_FREQ = 1.0             # example-event dominant speed rhythm (Hz), low edge of 1-12 Hz band
    HIFREQ = [75, 912, 1735, 6, 1428]      # high-frequency speed content (~7-13 Hz), jittery motion
    LOFREQ = [889, 1498, 1143, 39, 710]    # low-frequency speed content (~1.5-2 Hz), smooth motion
    CC_STRONG = [1097, 667, 43, 2008, 649] # interacting pair strongly coordinated (peak r 0.81-0.85)
    CC_WEAK = [1514, 925, 979, 1933, 1972] # labelled pair weakly / anti-correlated (peak r -0.19..-0.29)
    CC_HI = 1097               # single strongly-coordinated example
    CC_LO = 1514               # single weakly-coordinated example
    WHO_CHANCE = 1 / 3         # chance that a given pair is the most-correlated of 3
    WHO_FRAC_ALL = 0.41        # full corpus: interacting pair IS the most-correlated pair
    WHO_FRAC_STRONG = 0.90     # among strongly-coordinated events (peak r > 0.5): rises to 90%
    N_STRONG = 285             # count of strongly-coordinated events
    WHO_FRAC_SUB = 0.406       # exercise subsample (seed 0, n=399): win fraction to reproduce
    N_SUB = 399
    FULL_FRAC = 0.464          # pre-contact: fraction of aggression events where approacher LEADS
    FULL_N = 750
    NULL_MEAN = 0.506          # within-event shuffle null: mean approacher-leads fraction
    NULL_HI = 0.577            # shuffle null 97.5th percentile
    return (CC_HI, CC_LO, CC_STRONG, CC_WEAK, CLEAN_AGG, CLEAN_NON, DOM_FREQ, EXAMPLE,
            FULL_FRAC, FULL_N, HIFREQ, LOFREQ, NULL_HI, NULL_MEAN, N_STRONG, N_SUB,
            WHO_CHANCE, WHO_FRAC_ALL, WHO_FRAC_STRONG, WHO_FRAC_SUB)


# ============================================================ 2. The example event
@app.cell(hide_code=True)
def _(EXAMPLE, mo):
    mo.md(
        rf"""
        ## An example interaction to follow throughout

        We will keep one clip in view for the whole notebook so every method has a concrete picture
        attached. Our example is **event #{EXAMPLE}** (cohort B, a female cage; a hand-scored
        **non-aggression** approach with clean tracking). By rank, the **approacher is the Sub (green)**
        mouse, the **approachee is the Int/Mid (blue)** mouse, and the **bystander is the Dom (red)**
        mouse. Contact occurs at frame 40.

        The animation below shows the three skeletons over the 2.6-second clip. Watch the green mouse
        close in on the blue one; the small marker shows the moment of contact. Keep this picture in
        mind: each number, spectrogram, and correlation we compute later is a *summary* of a movie like
        this one.
        """
    )
    return


@app.cell
def _(EXAMPLE, cr, cu, kp, mo, ranks):
    _gif = cu.event_gif_bytes(kp[EXAMPLE], ranks[EXAMPLE], contact_rel=int(cr[EXAMPLE]), cell=210, fps=20)
    mo.vstack([
        mo.md(f"**Example approach event #{EXAMPLE}** — skeletons colored by rank "
              "(red = Dom, blue = Int, green = Sub); the white arrow points approacher → approachee; "
              "the marker shows contact onset."),
        mo.Html(cu.gif_img_html(_gif, width=240)),
    ])
    return


@app.cell(hide_code=True)
def _(CLEAN_AGG, CLEAN_NON, mo):
    mo.md(
        rf"""
        ### Aggression versus ordinary approach, side by side

        The single most important split in this dataset is the human aggression label. Before we
        quantify anything, it helps to *see* both categories. Below are four clean **aggression**
        approaches (top) and four clean **non-aggression** approaches (bottom). All are colored by
        rank, so any difference you notice is a difference in *movement*, not in color. Aggression
        clips tend to look faster and more entangled; ordinary approaches look calmer. Everything we do
        next is an attempt to turn "looks faster and more entangled" into numbers.

        Aggression exemplars: {CLEAN_AGG} &nbsp;·&nbsp; Non-aggression exemplars: {CLEAN_NON}.
        """
    )
    return


@app.cell
def _(CLEAN_AGG, CLEAN_NON, cr, cu, kp, mo, ranks):
    _agg = cu.grid_gif_bytes([(kp[i], ranks[i], int(cr[i])) for i in CLEAN_AGG], ncols=4, cell=135, fps=18)
    _non = cu.grid_gif_bytes([(kp[i], ranks[i], int(cr[i])) for i in CLEAN_NON], ncols=4, cell=135, fps=18)
    mo.vstack([
        mo.md("**Aggression (human-scored):**"),
        mo.Html(cu.gif_img_html(_agg, width=560)),
        mo.md("**Non-aggression (human-scored):**"),
        mo.Html(cu.gif_img_html(_non, width=560)),
    ])
    return


# ============================================================ 3. Distributions
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1 · Distributions — look at each measurement first

        **Why.** Before fitting any model we look at the raw numbers, one measurement at a time. From
        the pose, the pipeline computed **19 features** per event — single numbers summarizing the
        interaction, such as `closing_speed` (how fast the gap shrinks) or `appr_faces_appe` (how
        directly the approacher faces the approachee). Each feature has a **distribution**: the spread
        of its values across all 2,499 events. Crucially, aggression often shows up not as a shift in
        the *average* but as a heavier **tail** — a subset of events with unusually large values. A bar
        of group means would hide that entirely, which is why we plot **every individual event**.

        **Definition — distribution.** The distribution of a feature is simply how often each value
        occurs across events. We display it with **strip plots** (every event as one jittered point),
        **violins** (the smoothed shape with the points overlaid), and **ECDFs** (empirical cumulative
        distribution functions — for each value `v`, the fraction of events at or below `v`; a clean
        way to compare whole distributions).

        **Method.** Pick a feature, a way to split the events into groups, and a plot style.

        - **Functions:** `cu.strip_points_fig`, `cu.violin_points_fig`, `cu.ecdf_fig`.
        - **Inputs:** one feature column (values), a grouping label per event, and a plot style.
        - **Output:** a seaborn-style figure showing the raw points, hoverable by event index.

        Splitting by **sex**, **condition** (pre / dep / post), and **cohort** is a habit worth
        keeping, because those are the variables a hidden confound could ride in on later.
        """
    )
    return


@app.cell
def _(mo):
    feat_sel = mo.ui.dropdown(
        options=["closing_speed", "appr_speed_mean", "appr_speed_max", "pair_dist_min",
                 "appr_faces_appe", "appe_faces_appr", "heading_alignment", "appr_angvel",
                 "triangle_area_mean", "bystander_dist_mean", "appr_body_len"],
        value="closing_speed", label="feature", full_width=True)
    split_sel = mo.ui.dropdown(options=["aggression", "sex", "condition", "cohort"],
                               value="aggression", label="split by", full_width=True)
    style_sel = mo.ui.dropdown(options=["strip", "violin", "ecdf"], value="strip",
                               label="plot style", full_width=True)
    return feat_sel, split_sel, style_sel


@app.cell
def _(X, agg, cohort, cond, cu, feat_sel, fnames, mo, sexv, split_sel, style_sel):
    _fi = fnames.index(feat_sel.value)
    _v = X[:, _fi]
    _idx = list(range(len(_v)))
    if split_sel.value == "aggression":
        _g = ["aggression" if a else "non-aggression" for a in agg]
        _order = ["non-aggression", "aggression"]
        _colors = {"aggression": "#c1272d", "non-aggression": "#8899aa"}
    elif split_sel.value == "sex":
        _g = ["male" if s == "M" else "female" for s in sexv]
        _order = ["male", "female"]; _colors = {"male": "#4c78a8", "female": "#e45756"}
    elif split_sel.value == "condition":
        _g = list(cond); _order = ["pre", "dep", "post"]
        _colors = {"pre": "#4c78a8", "dep": "#e45756", "post": "#54a24b"}
    else:
        _g = list(cohort); _order = sorted(set(cohort.tolist()))
        _colors = {_order[0]: "#4c78a8", _order[1]: "#f58518"}
    _ttl = f"{feat_sel.value} — split by {split_sel.value}"
    if style_sel.value == "strip":
        _fig = cu.strip_points_fig(_v, _g, group_order=_order, colors=_colors, hover=_idx,
                                   ylabel=feat_sel.value, title=_ttl, height=440)
    elif style_sel.value == "violin":
        _fig = cu.violin_points_fig(_v, _g, group_order=_order, colors=_colors,
                                    ylabel=feat_sel.value, title=_ttl, height=460)
    else:
        _fig = cu.ecdf_fig(_v, _g, group_order=_order, colors=_colors,
                           xlabel=feat_sel.value, title=_ttl, height=440)
    mo.vstack([mo.hstack([feat_sel, split_sel, style_sel]), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **What to notice.** Split `closing_speed` or `appr_speed_max` by **aggression** and the
        aggression group's points stretch further up — a heavier upper tail, not a wholly different
        cloud. Now switch the split to **sex** and pick `heading_alignment` (how aligned the two mice's
        body orientations are): the male and female clouds sit at visibly different heights. That is a
        real hint, but a hint is not a finding — two mice in the same cage are not independent
        observations, so a difference between many male and many female *events* can be an illusion
        created by a few cages. We do not resolve that here; we resolve it carefully, with cage-level
        tests and a negative control, in the next notebook. For now the habit is simply: **look at the
        points, notice the tails and the shifts, and stay skeptical.**
        """
    )
    return


# ============================================================ 4. Correlation heatmap
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2 · The 19×19 feature-correlation heatmap

        **Why.** The 19 features are **not independent** measurements. Many rise and fall together
        across events, which means the representation carries **redundant** information. This matters
        because the next notebook uses **PCA** to replace many redundant features with a few
        independent ones — and this heatmap is the picture that motivates it.

        **Definition — Pearson correlation.** The Pearson correlation `r` between two features measures
        how linearly they move together across events: `r = +1` (rise together), `r = -1` (one rises
        as the other falls), `r = 0` (unrelated). We compute `r` for every pair of the 19 features and
        display the 19×19 matrix as a colored grid.

        **Method.** `np.corrcoef` takes the standardized feature matrix (events × features) and returns
        the 19×19 matrix of pairwise correlations. Bright off-diagonal blocks mark groups of features
        that are essentially measuring the same thing — for example the four speed features, or the two
        facing features, tend to move as a block.
        """
    )
    return


@app.cell
def _(X, cu, fnames, go, mo, np):
    _Xz, _, _ = cu.standardize(X)
    _C = np.corrcoef(_Xz.T)
    _fig = go.Figure(go.Heatmap(z=_C, x=fnames, y=fnames, colorscale="RdBu", zmid=0,
                                zmin=-1, zmax=1, colorbar=dict(title="r")))
    _fig.update_layout(template="plotly_white", height=640,
                       title="Feature–feature correlation (Pearson r) — off-diagonal blocks = redundancy",
                       margin=dict(l=10, r=10, t=50, b=130), font=dict(size=12))
    _fig.update_xaxes(tickangle=45)
    _absC = np.abs(_C - np.eye(19))
    _i, _j = np.unravel_index(np.argmax(_absC), _C.shape)
    mo.vstack([_fig, mo.md(
        f"**Most-correlated pair:** `{fnames[_i]}` ↔ `{fnames[_j]}` (r = {_C[_i, _j]:.2f}). "
        "Several features are this redundant, which is why roughly **6 combined axes** will capture "
        "most of the variation once we run PCA — that is the whole reason the next notebook exists.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Seeing redundancy as a joint distribution

        A single number in the heatmap (say r = 0.8 between two features) is easier to trust once you
        see the point cloud behind it. Pick any two features below and we draw their **joint density** —
        a 2-D histogram smoothed into contours — with every event overlaid as a point (hover for the
        event index). Two highly correlated features produce a tight diagonal ridge; two independent
        features produce a round blob. This is redundancy made visible: when the cloud is a thin
        diagonal, one feature is nearly predictable from the other, so keeping both wastes a dimension.
        """
    )
    return


@app.cell
def _(fnames, mo):
    kx = mo.ui.dropdown(options=fnames, value="appr_speed_mean", label="x feature", full_width=True)
    ky = mo.ui.dropdown(options=fnames, value="appr_speed_max", label="y feature", full_width=True)
    return kx, ky


@app.cell
def _(X, cu, fnames, kx, ky, mo, np):
    _ix = fnames.index(kx.value); _iy = fnames.index(ky.value)
    _x = X[:, _ix]; _y = X[:, _iy]
    _r = float(np.corrcoef(_x, _y)[0, 1])
    _fig = cu.kde2d_fig(_x, _y, hover=list(range(len(_x))), xlabel=kx.value, ylabel=ky.value,
                        title=f"joint density: {kx.value} vs {ky.value}   (r = {_r:.2f})", height=480)
    mo.vstack([mo.hstack([kx, ky]), _fig])
    return


# ============================================================ 5. Rhythm in time & frequency
@app.cell(hide_code=True)
def _(EXAMPLE, mo):
    mo.md(
        rf"""
        ## 3 · Rhythm — one mouse in time and frequency

        **Why.** A single number like "average speed" hides *how* a mouse moved. Two animals can share
        the same average speed while one glides smoothly and the other darts in quick bursts. The
        **rhythm** of movement — how quickly speed rises and falls — is itself informative, and it is
        exactly the kind of within-event structure the later compression discards.

        **Method, part 1 — the raw traces.** First we plot, for event #{EXAMPLE}, the distance between
        the two interacting mice and each mouse's speed over time. This uses our `mouse_speed` helper,
        which is just `np.diff` of the body-center track followed by a length — the same small function
        you will write in the exercise. We expect the distance to collapse and the speeds to rise as
        contact approaches.
        """
    )
    return


@app.cell
def _(EXAMPLE, appr_appe_speed, cr, cu, kp, make_subplots, mo, np, ranks):
    _k = kp[EXAMPLE]
    _c0 = cu._centroids(_k[:, 0]); _c1 = cu._centroids(_k[:, 1])
    _dist = np.linalg.norm(_c0 - _c1, axis=1)                  # (T,) closing distance
    _s0, _s1 = appr_appe_speed(kp, EXAMPLE)                    # (T-1,)
    _t = np.arange(len(_dist)) / cu.FPS
    _te = np.arange(len(_s0)) / cu.FPS
    _cf = int(cr[EXAMPLE]) / cu.FPS
    _c_appr = cu.RANK_HEX[int(ranks[EXAMPLE][0])]              # approacher colored by ITS rank (Sub)
    _c_appe = cu.RANK_HEX[int(ranks[EXAMPLE][1])]              # approachee colored by ITS rank (Int)
    _fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("pair distance (px)", "per-mouse speed (px/frame)"))
    _fig.add_scatter(x=_t, y=_dist, mode="lines", line=dict(color="#333"), name="pair distance",
                     row=1, col=1)
    _fig.add_scatter(x=_te, y=_s0, mode="lines", line=dict(color=_c_appr), name="approacher (Sub)",
                     row=2, col=1)
    _fig.add_scatter(x=_te, y=_s1, mode="lines", line=dict(color=_c_appe), name="approachee (Int)",
                     row=2, col=1)
    for _r in (1, 2):
        _fig.add_vline(x=_cf, line=dict(color="#888", dash="dot"), row=_r, col=1)
    _fig.update_layout(template="plotly_white", height=440, font=dict(size=14),
                       margin=dict(l=10, r=10, t=40, b=10))
    _fig.update_xaxes(title_text="time (s)", row=2, col=1, showgrid=False)
    _fig.update_yaxes(showgrid=False)
    mo.vstack([_fig, mo.md("The dotted line marks contact. The distance falls and both speeds rise "
                           "into the meeting. Next we ask *at what rhythm* the speed changes.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Method, part 2 — the wavelet spectrogram.** To read rhythm we use a **Morlet wavelet
        transform**. A wavelet is a short wave-shaped template. We slide a template of a given
        frequency along the speed signal and measure how strongly the signal matches it at each moment.
        Repeating this across many frequencies produces a **spectrogram** — a picture with **time** on
        the horizontal axis, **frequency** (rhythm, in cycles per second, Hz) on the vertical axis, and
        brightness showing how much of that rhythm is present at that moment.

        - **Function:** `cu.wavelet_power(signal, freqs, fps)` (wrapped by `padded_wavelet`, which pads
          the short clip so low-frequency templates fit).
        - **Inputs:** a 1-D signal (here the approacher's speed), a list of frequencies to test, and
          the sampling rate (50 fps).
        - **Output:** a `frequency × time` grid of power — bright where that rhythm is present.

        Slide the control to change the top frequency shown. The dotted white line marks the strongest
        (dominant) rhythm.
        """
    )
    return


@app.cell
def _(mo):
    fmax_slider = mo.ui.slider(6, 20, value=12, step=1, label="wavelet upper frequency (Hz)",
                               debounce=True, full_width=True)
    return (fmax_slider,)


@app.cell
def _(DOM_FREQ, EXAMPLE, appr_appe_speed, cu, fmax_slider, go, kp, mo, np, padded_wavelet):
    _s0, _ = appr_appe_speed(kp, EXAMPLE)
    _freqs = np.linspace(1.0, float(fmax_slider.value), 45)
    _P = padded_wavelet(_s0, _freqs, cu.FPS, padlen=600)
    _t = np.arange(_P.shape[1]) / cu.FPS
    _dom = float(_freqs[np.argmax(_P.mean(axis=1))])
    _fig = go.Figure(go.Heatmap(z=_P, x=_t, y=_freqs, colorscale="Viridis",
                                colorbar=dict(title="power")))
    _fig.add_hline(y=_dom, line=dict(color="white", dash="dot"))
    _fig.update_layout(template="plotly_white", height=380, font=dict(size=14),
                       title=f"Morlet spectrogram of approacher speed — dominant ≈ {_dom:.1f} Hz",
                       xaxis_title="time (s)", yaxis_title="frequency (Hz)",
                       margin=dict(l=10, r=10, t=50, b=10))
    mo.vstack([fmax_slider, _fig, mo.md(
        f"**Dominant rhythm ≈ {_dom:.1f} Hz** (pinned build value {DOM_FREQ} Hz over the 1–12 Hz "
        "band). This is a **slow** rhythm — the pace of ordinary locomotion, a mouse taking a step or "
        "two per second — not a fast oscillation. The power sits low on the frequency axis and near "
        "the middle in time, right around contact.")])
    return


@app.cell(hide_code=True)
def _(HIFREQ, LOFREQ, mo):
    mo.md(
        rf"""
        ### What high frequency looks like: fast, jittery movement

        The example event's speed is dominated by a slow (~1 Hz) locomotor rhythm. But some events
        carry **high-frequency** speed content: the speed rises and falls several times per second,
        which corresponds to **quick, jerky movement** — rapid darting, scrambling, repeated
        start-and-stop — rather than a smooth glide. The wavelet lets us *find* those events: we take
        the dominant frequency of each approacher's speed and keep the extremes.

        Below are two grids of skeleton animations. The **top** grid is six high-frequency events
        (their speed changes ~7–13 times per second); the **bottom** grid is smooth, low-frequency
        events (~1.5–2 Hz). Watch how the top animations look abrupt and stuttery while the bottom ones
        glide. High-frequency movement is a description of *how* the animal moved — both aggression and
        non-aggression events appear here, so it is not by itself a sign of aggression.

        High-frequency: {HIFREQ} &nbsp;·&nbsp; Low-frequency: {LOFREQ}.
        """
    )
    return


@app.cell
def _(HIFREQ, LOFREQ, cr, cu, kp, mo, ranks):
    _hi = cu.grid_gif_bytes([(kp[i], ranks[i], int(cr[i])) for i in HIFREQ], ncols=3, cell=145, fps=18)
    _lo = cu.grid_gif_bytes([(kp[i], ranks[i], int(cr[i])) for i in LOFREQ], ncols=3, cell=145, fps=18)
    mo.vstack([
        mo.md("**High-frequency movement** — short darts and abrupt stops (speed changes many times/s):"),
        mo.Html(cu.gif_img_html(_hi, width=470)),
        mo.md("**Low-frequency movement** — smooth, gliding locomotion (speed changes slowly):"),
        mo.Html(cu.gif_img_html(_lo, width=470)),
    ])
    return


@app.cell
def _(HIFREQ, appr_appe_speed, cu, go, kp, mo, np, padded_wavelet):
    # Spectrogram of one HIGH-frequency event, for direct contrast with the ~1 Hz example above.
    _i = HIFREQ[0]
    _s0, _ = appr_appe_speed(kp, _i)
    _freqs = np.linspace(1.0, 14.0, 45)
    _P = padded_wavelet(_s0, _freqs, cu.FPS, padlen=600)
    _t = np.arange(_P.shape[1]) / cu.FPS
    _dom = float(_freqs[np.argmax(_P.mean(axis=1))])
    _fig = go.Figure(go.Heatmap(z=_P, x=_t, y=_freqs, colorscale="Viridis", colorbar=dict(title="power")))
    _fig.add_hline(y=_dom, line=dict(color="white", dash="dot"))
    _fig.update_layout(template="plotly_white", height=360, font=dict(size=14),
                       title=f"High-frequency event #{_i} — dominant ≈ {_dom:.1f} Hz",
                       xaxis_title="time (s)", yaxis_title="frequency (Hz)",
                       margin=dict(l=10, r=10, t=50, b=10))
    mo.vstack([_fig, mo.md(
        f"Here the bright band sits **higher** on the frequency axis (~{_dom:.1f} Hz) than in the "
        "example event's slow spectrogram. The band moved up because the speed itself changes more "
        "times per second — exactly the jittery motion in the top grid.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Two honest limits of the wavelet on this data

        These clips are only ~2.6 s long, and that creates two limitations to keep in mind:

        1. **Time–frequency trade-off.** A wavelet narrow enough to pin down *when* a burst happened is
           blurry about *what frequency* it was, and vice versa. You cannot have sharp timing and sharp
           frequency at once.
        2. **Edge effects.** A low-frequency template is wider than the clip, so it runs off both ends.
           We pad the signal just to fit the template, but the padded flanks are fabricated, not
           measured. Trust the **middle** of each spectrogram and distrust the extreme left and right
           edges.

        This is also why a wavelet suits this data better than a single Fourier transform (FFT): an FFT
        assumes one fixed spectrum for the whole clip, whereas the wavelet allows the rhythm to
        **change** as contact approaches — which it does.
        """
    )
    return


# ============================================================ 6. Coordination: who interacts with whom
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4 · Coordination — who is interacting with whom?

        **Why.** Each clip has three mice, and we would like to recover which two are actually engaged
        from movement alone. The intuition: two interacting mice should move in a **coordinated** way —
        when one speeds up, so does the other. If that intuition holds, coordination becomes a tool for
        answering *who interacts with whom*.

        **Definition — cross-correlation.** To measure coordination we slide one mouse's speed trace
        against another's and, at each shift (**lag**), compute how well the two line up. The **peak**
        value (best alignment over small lags) is a single number between about −1 and +1: near +1 the
        two speed traces rise and fall together (**highly coordinated**), near 0 they are unrelated,
        and negative means they tend to move in opposition.

        - **Function:** `cu.cross_corr_lag(x, y, max_lag)` → `(lags, corr, peak_lag)`; our helper
          `pair_peak_corr` returns the peak correlation value directly.
        - **Inputs:** two speed traces and the largest lag to consider (in frames).
        - **Output:** the coordination of that pair, and (via the peak lag) which one tends to lead.

        Two examples make the idea concrete: a **strongly** coordinated interacting pair, and a
        **weakly / anti**-coordinated one.
        """
    )
    return


@app.cell
def _(CC_HI, cr, cu, go, kp, make_subplots, mo, mouse_speed, ranks):
    _i = CC_HI
    _k = kp[_i]
    _sp = [mouse_speed(_k, m) for m in range(3)]
    _t = [x / cu.FPS for x in range(len(_sp[0]))]
    _lags, _corr, _pk = cu.cross_corr_lag(_sp[0], _sp[1], 15)
    _peak = float(_corr.max())
    _c0 = cu.RANK_HEX[int(ranks[_i][0])]; _c1 = cu.RANK_HEX[int(ranks[_i][1])]
    _fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                         subplot_titles=("approacher & approachee speed (px/frame)",
                                         "cross-correlation vs lag"))
    _fig.add_scatter(x=_t, y=_sp[0], mode="lines", line=dict(color=_c0), name="approacher", row=1, col=1)
    _fig.add_scatter(x=_t, y=_sp[1], mode="lines", line=dict(color=_c1), name="approachee", row=1, col=1)
    _fig.add_scatter(x=_lags, y=_corr, mode="lines+markers", line=dict(color="#7b3294"),
                     showlegend=False, row=1, col=2)
    _fig.update_layout(template="plotly_white", height=340, font=dict(size=13),
                       title=f"STRONG coordination — event #{_i}: peak correlation ≈ {_peak:.2f}",
                       margin=dict(l=10, r=10, t=60, b=10))
    _fig.update_xaxes(showgrid=False); _fig.update_yaxes(showgrid=False)
    _gif = cu.event_gif_bytes(_k, ranks[_i], contact_rel=int(cr[_i]), cell=170, fps=18)
    mo.vstack([_fig, mo.Html(cu.gif_img_html(_gif, width=200)),
               mo.md("The two speed traces rise and fall together, so the peak correlation is high, and "
                     "in the animation the two mice really are moving as a coordinated pair.")])
    return


@app.cell
def _(CC_LO, cr, cu, go, kp, mo, mouse_speed, pair_peak_corr, ranks):
    _i = CC_LO
    _k = kp[_i]
    _sp = [mouse_speed(_k, m) for m in range(3)]
    _t = [x / cu.FPS for x in range(len(_sp[0]))]
    _pk01 = pair_peak_corr(_k, 0, 1)     # labelled interacting pair
    _pk12 = pair_peak_corr(_k, 1, 2)     # approachee & bystander
    _c0 = cu.RANK_HEX[int(ranks[_i][0])]; _c1 = cu.RANK_HEX[int(ranks[_i][1])]
    _c2 = cu.RANK_HEX[int(ranks[_i][2])]
    _fig = go.Figure()
    _fig.add_scatter(x=_t, y=_sp[0], mode="lines", line=dict(color=_c0), name="approacher")
    _fig.add_scatter(x=_t, y=_sp[1], mode="lines", line=dict(color=_c1), name="approachee")
    _fig.add_scatter(x=_t, y=_sp[2], mode="lines", line=dict(color=_c2, dash="dot"), name="bystander")
    _fig.update_layout(template="plotly_white", height=320, font=dict(size=13),
                       title=(f"WEAK coordination for the labelled pair — event #{_i}: "
                              f"approacher–approachee ≈ {_pk01:.2f}, approachee–bystander ≈ {_pk12:.2f}"),
                       xaxis_title="time (s)", yaxis_title="speed (px/frame)",
                       margin=dict(l=10, r=10, t=60, b=10))
    _fig.update_xaxes(showgrid=False); _fig.update_yaxes(showgrid=False)
    _gif = cu.event_gif_bytes(_k, ranks[_i], contact_rel=int(cr[_i]), cell=170, fps=18)
    mo.vstack([_fig, mo.Html(cu.gif_img_html(_gif, width=200)),
               mo.md("Here the labelled approacher and approachee are **poorly** coordinated. Two mice "
                     "can move independently for ordinary reasons — resting apart, wandering on their "
                     "own — so a low correlation is a real possibility even for the interacting pair.")])
    return


@app.cell(hide_code=True)
def _(CC_STRONG, CC_WEAK, mo):
    mo.md(
        rf"""
        The same contrast across several events at once. **Top:** five events where the interacting
        pair is strongly coordinated (peak r ≈ 0.81–0.85). **Bottom:** five where the labelled pair is
        weakly or anti-coordinated (peak r ≈ −0.19 to −0.29). Watch the top pairs move in lockstep and
        the bottom pairs do their own thing.

        Strongly coordinated: {CC_STRONG} &nbsp;·&nbsp; Weakly coordinated: {CC_WEAK}.
        """
    )
    return


@app.cell
def _(CC_STRONG, CC_WEAK, cr, cu, kp, mo, ranks):
    _hi = cu.grid_gif_bytes([(kp[i], ranks[i], int(cr[i])) for i in CC_STRONG], ncols=5, cell=125, fps=18)
    _lo = cu.grid_gif_bytes([(kp[i], ranks[i], int(cr[i])) for i in CC_WEAK], ncols=5, cell=125, fps=18)
    mo.vstack([
        mo.md("**Strongly coordinated interacting pairs:**"),
        mo.Html(cu.gif_img_html(_hi, width=600)),
        mo.md("**Weakly / anti-coordinated labelled pairs:**"),
        mo.Html(cu.gif_img_html(_lo, width=600)),
    ])
    return


@app.cell(hide_code=True)
def _(WHO_CHANCE, WHO_FRAC_ALL, WHO_FRAC_STRONG, mo):
    mo.md(
        rf"""
        ### Does the most-correlated pair identify the interacting pair?

        The examples suggest coordination is a real cue. To test it properly, for every well-tracked
        event we compute the peak speed-correlation of **all three** possible pairs
        (approacher–approachee, approacher–bystander, approachee–bystander) and ask how often the
        **labelled interacting pair** is the **most-correlated** of the three. With three pairs, pure
        chance would give **{WHO_CHANCE*100:.0f}%**.

        Across the whole corpus the interacting pair wins **~{WHO_FRAC_ALL*100:.0f}%** of the time —
        clearly above chance, so correlation genuinely carries information about who interacts, but far
        from perfect. The reason it is imperfect is the **bystander confound**: two mice can move
        together for reasons unrelated to interacting (both resting still, both walking in parallel, a
        common startle shared by all three), which can make a non-interacting pair the most correlated.

        The cue gets much sharper when we **restrict to events where the interacting pair is actually
        strongly coordinated** (peak r > 0.5). Among those, the interacting pair is the most-correlated
        pair **~{WHO_FRAC_STRONG*100:.0f}%** of the time. In other words: *when there is real
        coordination in the clip, it almost always belongs to the interacting pair.* Move the threshold
        slider below and watch the win-fraction climb from just above chance toward ~0.90.
        """
    )
    return


@app.cell
def _(cu, kp, np, pair_peak_corr):
    # Precompute the three pairwise coordinations for a fixed subsample (runs once, fast).
    _rng = np.random.RandomState(1)
    cc_idx = _rng.choice(len(kp), size=600, replace=False)
    _r01, _r02, _r12 = [], [], []
    for _i in cc_idx:
        _k = kp[_i]
        if min(cu._centroids(_k[:, m]).std() for m in range(3)) < 1e-6:
            _r01.append(np.nan); _r02.append(np.nan); _r12.append(np.nan); continue
        _r01.append(pair_peak_corr(_k, 0, 1))
        _r02.append(pair_peak_corr(_k, 0, 2))
        _r12.append(pair_peak_corr(_k, 1, 2))
    cc_r01 = np.array(_r01); cc_r02 = np.array(_r02); cc_r12 = np.array(_r12)
    return cc_r01, cc_r02, cc_r12


@app.cell
def _(mo):
    cc_thr = mo.ui.slider(-0.2, 0.8, value=0.0, step=0.05,
                          label="keep events where interacting-pair peak r >", debounce=True,
                          full_width=True)
    return (cc_thr,)


@app.cell
def _(WHO_CHANCE, WHO_FRAC_STRONG, cc_r01, cc_r02, cc_r12, cc_thr, go, mo, np):
    _keep = np.isfinite(cc_r01) & (cc_r01 > cc_thr.value)
    _wins = (cc_r01[_keep] >= np.maximum(cc_r02[_keep], cc_r12[_keep]))
    _frac = float(np.mean(_wins)) if _keep.sum() else float("nan")
    _fig = go.Figure()
    _fig.add_bar(x=["interacting pair is most-correlated"], y=[_frac], marker_color="#4c78a8",
                 width=0.5, text=[f"{_frac:.2f}<br>n={int(_keep.sum())}"], textposition="outside")
    _fig.add_hline(y=WHO_CHANCE, line=dict(color="#333", dash="dash"),
                   annotation_text=f"chance among 3 pairs ({WHO_CHANCE:.2f})")
    _fig.add_hline(y=WHO_FRAC_STRONG, line=dict(color="#137333", dash="dot"),
                   annotation_text=f"strong-coordination regime (~{WHO_FRAC_STRONG:.2f})")
    _fig.update_layout(template="plotly_white", height=380, font=dict(size=14),
                       yaxis_title="fraction of kept events", yaxis_range=[0, 1],
                       title="As the coordination threshold rises, the interacting pair wins more often",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False); _fig.update_yaxes(showgrid=False)
    mo.vstack([cc_thr, _fig, mo.md(
        "At a low threshold (all events) the interacting pair wins only a little above chance. Raise "
        "the threshold to keep only genuinely coordinated events and the fraction climbs toward 0.90. "
        "**Coordination identifies the interacting pair — but only when there is real coordination to "
        "read.**")])
    return


# ============================================================ 7. Who leads whom
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### A directed version: who moves first?

        **Why.** Cross-correlation also carries **order**: if the approacher's speed changes and the
        approachee's follows a moment later, the peak alignment sits at a positive **lag**, and we say
        the approacher **leads**. "Who moves first before contact" is a natural behavioral question —
        but we will see it is hard to answer reliably on such short clips, and reporting that honestly
        is part of the job.

        We look only at frames **before contact**. After contact both mice necessarily move together,
        so including those frames would answer the question trivially.

        First, a quick check that the estimator works when the answer is known: we build two signals
        where one is a delayed copy of the other and confirm `cross_corr_lag` recovers the delay.
        """
    )
    return


@app.cell
def _(mo):
    toy_lag = mo.ui.slider(-8, 8, value=3, step=1, label="imposed lag (B follows A by …)",
                           debounce=True, full_width=True)
    return (toy_lag,)


@app.cell
def _(cu, go, mo, np, toy_lag):
    _rng = np.random.RandomState(1)
    _base = np.cumsum(_rng.randn(120))                     # a wandering "driver" A
    _A = _base + 0.4 * _rng.randn(120)
    _lag = int(toy_lag.value)
    _B = np.roll(_base, _lag) + 0.4 * _rng.randn(120)      # B = A delayed by `lag`
    _lags, _corr, _pk = cu.cross_corr_lag(_A, _B, 12)
    _fig = go.Figure()
    _fig.add_scatter(x=_lags, y=_corr, mode="lines+markers", line=dict(color="#7b3294"))
    _fig.add_vline(x=_pk, line=dict(color="#d62728", dash="dot"))
    _fig.update_layout(template="plotly_white", height=320, font=dict(size=14),
                       title=f"recovered peak lag = {_pk}  (imposed {_lag})",
                       xaxis_title="lag (A vs B)", yaxis_title="normalized cross-correlation",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False); _fig.update_yaxes(showgrid=False)
    mo.vstack([toy_lag, _fig, mo.md(
        "A positive lag means A leads B. The estimator recovers the imposed lag cleanly on a long, "
        "clean signal — the regime the real, short, noisy mouse clips do **not** enjoy.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now the same test on real pre-contact traces. For each aggression event we ask whether the
        approacher leads, and report the **fraction of events** in which it does. With no consistent
        leader that fraction sits near **0.50**. The gray band is a **shuffle null**: we scrambled the
        traces within each event to see how far from 0.50 the fraction wanders by chance. A real leader
        effect would have to poke **outside** that band.
        """
    )
    return


@app.cell
def _(mo):
    coord_maxlag = mo.ui.slider(4, 15, value=10, step=1, label="max lag (frames)",
                                debounce=True, full_width=True)
    coord_split = mo.ui.dropdown(options=["all", "sex", "condition"], value="all",
                                 label="split by", full_width=True)
    return coord_maxlag, coord_split


@app.cell
def _(FULL_FRAC, FULL_N, NULL_HI, agg, cond, coord_maxlag, coord_split, cr, go,
      kp, leader_fraction, mo, np, sexv):
    # Live loop on a balanced <=200-event aggression subsample; precomputed full-corpus result beside.
    _rng = np.random.RandomState(0)
    _agg_idx = np.where(agg == 1)[0]
    _sub = _rng.choice(_agg_idx, size=min(200, len(_agg_idx)), replace=False)
    if coord_split.value == "all":
        _splits = [("all aggression", _sub)]
    elif coord_split.value == "sex":
        _splits = [("male", _sub[sexv[_sub] == "M"]), ("female", _sub[sexv[_sub] == "F"])]
    else:
        _splits = [(c, _sub[cond[_sub] == c]) for c in ["pre", "dep", "post"]]
    _names, _fracs, _ns = [], [], []
    for _nm, _idx in _splits:
        _f, _n = leader_fraction(kp, cr, _idx, max_lag=int(coord_maxlag.value))
        _names.append(_nm); _fracs.append(_f if _f == _f else 0.0); _ns.append(_n)
    _fig = go.Figure()
    _fig.add_bar(x=_names, y=_fracs, marker_color="#4c78a8",
                 text=[f"{f:.2f}<br>n={n}" for f, n in zip(_fracs, _ns)], textposition="outside")
    _fig.add_hline(y=0.5, line=dict(color="#333", dash="dash"),
                   annotation_text="no consistent leader (0.50)")
    _fig.add_hrect(y0=1 - NULL_HI, y1=NULL_HI, fillcolor="#bbbbbb", opacity=0.25, line_width=0,
                   annotation_text="within-event shuffle null (95%)", annotation_position="top left")
    _fig.update_layout(template="plotly_white", height=400, font=dict(size=14),
                       yaxis_title="fraction approacher LEADS", yaxis_range=[0, 1],
                       title=f"Pre-contact lead–lag — subsample, split by {coord_split.value}",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False); _fig.update_yaxes(showgrid=False)
    mo.vstack([mo.hstack([coord_maxlag, coord_split]), _fig, mo.md(
        f"**Precomputed full corpus (all {FULL_N} usable aggression events):** approacher-leads "
        f"fraction = **{FULL_FRAC:.3f}** — inside the gray shuffle band. The bars land near 0.50 too: "
        "there is **no robust leader** on these short, noisy, pre-contact windows. That is the honest "
        "result, and unlike the who-interacts question, more coordination does not rescue it.")])
    return


# ============================================================ 8. Exercise
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Exercise — write the speed function, then find who interacts

        **Python skill practiced:** *writing a small function* that turns positions into motion using
        **`np.diff`** and vectorized array math — no Python loop over frames. This is the single most
        reused operation in the whole course: every speed, every spectrogram, every correlation above
        started with this exact `np.diff`.

        **The scientific goal.** Reproduce the headline result of Section 4 with your own code: across a
        subsample of events, the labelled **interacting pair** is the most-correlated of the three
        pairs **more often than the 1/3 you'd get by chance**. You will write the one function the whole
        pipeline depends on — `my_speed` — and a provided loop uses it to build the plot and the number.
        """
    )
    return


@app.cell(hide_code=True)
def _(WHO_CHANCE, WHO_FRAC_SUB, mo):
    mo.md(
        rf"""
        ### What to do

        You will edit **one line** inside `my_speed`. Everything else is written for you.

        **Expected picture.** A strip plot with three columns — one per pair type — showing every
        event's peak coordination as an individual point. The **interacting** column's cloud should sit
        **a little higher** than the two bystander columns. The self-check then confirms the interacting
        pair wins the "most-correlated" contest about **{WHO_FRAC_SUB:.2f}** of the time — above the
        **{WHO_CHANCE:.2f}** chance line, a real but imperfect cue.
        """
    )
    return


@app.cell
def _(np):
    _rng = np.random.RandomState(0)
    ex_idx = _rng.choice(2499, size=400, replace=False)      # fixed subsample for the exercise
    return (ex_idx,)


@app.cell
def _(cu, np):
    def my_speed(kp_event, m):
        # ---- TODO (student): edit the ONE marked line -------------------------------------------
        # GOAL: return mouse m's per-frame speed (px/frame) — how far its body-center moved between
        # consecutive frames. Speed is the *length* of the frame-to-frame *change* in position.
        #
        # `cen` is already the body-center track, shape (T, 2): one (x, y) per frame.
        cen = cu._centroids(kp_event[:, m])
        #
        # EDIT THIS LINE: replace ____ with `np.diff(cen, axis=0)`.
        #   - np.diff(cen, axis=0) gives the frame-to-frame displacement vectors, shape (T-1, 2):
        #     row t is (cen[t+1] - cen[t]). This is WHY it matters — it converts *positions* into
        #     *motion*, the quantity every rhythm and correlation in this notebook is built from.
        #   - Do NOT write a for-loop over frames; np.diff does all T-1 subtractions at once
        #     (vectorized), which is both faster and the idiom the rest of the course uses.
        # The np.linalg.norm(..., axis=1) around it takes each displacement vector's length = speed.
        disp = np.diff(cen, axis=0)                          # <-- YOUR TURN: try writing this line yourself (shown filled so the notebook runs)
        # -----------------------------------------------------------------------------------------
        return np.nan_to_num(np.linalg.norm(disp, axis=1))   # (T-1,) px/frame
    return (my_speed,)


@app.cell
def _(cu, ex_idx, kp, my_speed, np):
    # Provided: uses YOUR my_speed to compute the three pairwise coordinations for each event, then
    # records which pair is the most-correlated. (If my_speed is wrong, disp has the wrong shape and
    # this cell errors — that is the feedback that your edit was not right.)
    def _peak(k, a, b):
        _, c, _ = cu.cross_corr_lag(my_speed(k, a), my_speed(k, b), 15)
        return float(c.max())
    _vals, _grp = [], []
    _wins, _tot = 0, 0
    for _i in ex_idx:
        _k = kp[_i]
        if min(my_speed(_k, m).std() for m in range(3)) < 1e-6:
            continue
        _r01 = _peak(_k, 0, 1); _r02 = _peak(_k, 0, 2); _r12 = _peak(_k, 1, 2)
        _vals += [_r01, _r02, _r12]
        _grp += ["interacting", "appr–bystander", "appe–bystander"]
        _tot += 1
        if _r01 >= max(_r02, _r12):
            _wins += 1
    ex_vals = np.array(_vals); ex_grp = np.array(_grp)
    ex_win = _wins / _tot if _tot else float("nan"); ex_n = _tot
    return ex_grp, ex_n, ex_vals, ex_win


@app.cell
def _(cu, ex_grp, ex_vals, mo):
    _fig = cu.strip_points_fig(
        ex_vals, ex_grp,
        group_order=["interacting", "appr–bystander", "appe–bystander"],
        colors={"interacting": "#c1272d", "appr–bystander": "#8899aa", "appe–bystander": "#b0b7bf"},
        ylabel="peak speed correlation", title="peak coordination by pair type (your my_speed)",
        height=430)
    mo.vstack([mo.md("**Your result** — each point is one event's peak coordination for that pair type:"),
               _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Solution (reveal)": mo.md(
                r"""
                ```python
                def my_speed(kp_event, m):
                    cen = cu._centroids(kp_event[:, m])      # (T, 2) body-center track
                    disp = np.diff(cen, axis=0)              # (T-1, 2) frame-to-frame displacement
                    return np.nan_to_num(np.linalg.norm(disp, axis=1))   # (T-1,) px/frame
                ```

                `np.diff(cen, axis=0)` subtracts each frame's position from the next, all at once. Its
                per-row length is the distance travelled between frames — the speed. The interacting
                pair's cloud sits a bit higher than the two bystander clouds, and the interacting pair
                is the most-correlated pair about **0.41** of the time: above the **0.33** chance line,
                but far from certain, because a bystander pair can be more coordinated by accident.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(WHO_CHANCE, ex_n, ex_win, mo):
    _ok = (ex_win == ex_win) and (ex_win > WHO_CHANCE + 0.01) and (0.35 <= ex_win <= 0.47)
    _color = "#e6f4ea" if _ok else "#fce8e6"
    _edge = "#137333" if _ok else "#c5221f"
    _msg = (f"**PASS.** Your `my_speed` reproduced the result: the interacting pair is the "
            f"most-correlated of the three pairs **{ex_win:.3f}** of the time (n={ex_n} usable events) "
            f"— above the {WHO_CHANCE:.2f} chance line, and consistent with the pinned **0.41**. "
            "Coordination is a real but imperfect cue to who interacts."
            if _ok else
            f"Observed win fraction = {ex_win:.3f} (n={ex_n}). Expected ~0.41, above the "
            f"{WHO_CHANCE:.2f} chance line. If you see an error or a wildly off number, check that "
            "`disp = np.diff(cen, axis=0)` — the frame-to-frame displacement — is what you filled in.")
    mo.md(
        f"""
        <div style="background:{_color}; border-left:5px solid {_edge}; border-radius:6px;
        padding:12px 16px;">

        ### Self-check
        {_msg}

        The grade is on the *win fraction* landing in the pinned band (above chance, well below
        certainty), which is the pre-verified build-time result for this exact subsample.
        </div>
        """
    )
    return


# ============================================================ 9. Granger (optional)
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Optional deeper section — Granger causality

        **Why include it.** Cross-correlation asks *when* two traces align. **Granger causality** asks a
        sharper question: does knowing the **past of mouse A** improve our prediction of **mouse B's
        next step**, beyond what B's own past already tells us? If it does, we say A "Granger-causes" B.
        It is a statistical test comparing a model that uses only B's past to one that also uses A's.

        - **Function:** `cu.granger_pair(x, y, lags=4)` (pure numpy, no `statsmodels`).
        - **Inputs:** two speed traces and how many past frames to use.
        - **Outputs:** an F-statistic and p-value for each direction (A→B and B→A). A small p-value
          suggests directed influence.
        """
    )
    return


@app.cell(hide_code=True)
def _(EXAMPLE, cr, cu, kp, mo, pre_speeds):
    _x, _y = pre_speeds(kp, cr, EXAMPLE)
    try:
        _g = cu.granger_pair(_x, _y, lags=4)
        _txt = (f"Example event #{EXAMPLE}: approacher→approachee F = {_g['f_xy']:.2f} "
                f"(p = {_g['p_xy']:.3f}); approachee→approacher F = {_g['f_yx']:.2f} "
                f"(p = {_g['p_yx']:.3f}).")
    except Exception as _e:
        _txt = f"(Granger skipped: {_e})"
    mo.accordion({
        "Granger on the example event, with the caveat": mo.md(
            rf"""
            {_txt}

            **The common-cause caveat.** Granger measures **prediction, not cause**. Both mice can be
            driven by a **shared third factor** — the bystander, or a common startle — which makes A
            look like it drives B when neither actually does. Bivariate Granger is also not
            *conditional*: to move from "A predicts B" to "A predicts B *given the bystander*," you
            would add the third mouse's trace as an extra input (a conditional, multivariate model). On
            1-second, nonstationary windows, treat any single-event Granger number as a hint, not a
            verdict — the same lesson the who-leads test taught us.
            """
        )
    })
    return


# ============================================================ 10. Review questions
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Review questions
        1. **Name a confounder.** Coordination is not proof of interaction. What shared cause on this
           rig could make two mice look coupled when neither is driving the other? (The bystander; a
           common startle or arousal spike shared by all three.)
        2. **Why does thresholding help who-interacts but not who-leads?** Restricting to strongly
           coordinated events pushed the who-interacts accuracy to ~90%, yet no threshold rescues the
           leader test. Why? (The *magnitude* of coordination is a strong, stable signal; the *sign of
           the lag* on a ~1 s noisy window is not.)
        3. **Wavelet vs FFT.** When is a wavelet more appropriate than a single FFT? (When the spectrum
           is *non-stationary* — the rhythm changes across the 2.6 s, which an FFT would average into
           one blurred spectrum.)
        4. **Distributions vs means.** Why did we plot every event as a point rather than a bar of group
           means? (Aggression shows up as a heavier *tail*, not a shifted average; a bar of means hides
           the tail and can even miss the effect entirely.)
        """
    )
    return


# ============================================================ 11. Throughline: answer + next question
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## What we found, and what it raises next

        **The answer to this notebook's question.** An interaction, seen as a signal over time, is
        highly **structured**: the pair distance collapses, both speeds rise into contact, and the
        movement carries a mostly **slow (~1–2 Hz) locomotor rhythm** with occasional high-frequency
        bursts we could see and render. Coordination between two mice is a **genuine cue to who
        interacts with whom** — imperfect overall (~41%, versus 33% chance) but nearly decisive (~90%)
        whenever real coordination is present. In contrast, *who moves first* before contact is **not**
        recoverable from these short, noisy windows: the leader estimate sits inside its shuffle null,
        and we reported that honestly.

        **What the exploration also revealed.** The 19 features are **heavily redundant** — the
        correlation heatmap is full of bright off-diagonal blocks, and whole groups of features move
        together. We are carrying 19 numbers that really live in far fewer independent directions.

        **The next question.** If the 19 features are this correlated, then:

        > What are the few **underlying dimensions** that actually vary across interactions — and once
        > we have them, what **behavioral types** exist, and do **sex** or **food deprivation** genuinely
        > change behavior, or only appear to?

        The next notebook takes the first step: **PCA**, which finds those few combined axes
        automatically and lets us measure exactly how many dimensions the 19 features are hiding.
        """
    )
    return


if __name__ == "__main__":
    app.run()
