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
        # NB03 · Behavior in time

        ## The question so far, and the question now

        We study social behavior and its neural basis. In the first two notebooks we turned raw video
        into **pose** — for each mouse, in every frame, the pixel position of 15 body **keypoints** (a
        keypoint is a labelled dot on the body: nose, ears, shoulders, tail base) — and then rebuilt
        each interaction from a **body's-eye view**, summarizing it as **19 numbers per event**
        (allocentric features such as how fast the mice close the gap, or how directly one faces the
        other).

        Those notebooks answered a preparation question: *how do we turn a video of two mice into
        numbers we can compute with?* This notebook asks the next one:

        > **What does a single interaction actually look like as a signal over time?**

        This is **exploratory data analysis (EDA)** — the first, unglamorous pass where we *look* at the
        data before we model it. We take our time here, because in the next notebook we will
        **compress** each event into a handful of numbers and, in doing so, average away *time* itself.
        Before we discard time, we should understand what it holds. We will ask three concrete questions
        about an interaction:

        1. **What is the shape of each measurement** across many events? (its *distribution*)
        2. **How does a mouse move through time, and at what rhythm?** (a *time–frequency* view)
        3. **Do two mice move together, and can that tell us who is interacting with whom?**
           (*coordination* between animals)

        A second, quieter thread runs through the notebook: **choosing the right picture for the data.**
        A distribution, a redundancy, a rhythm, and a coordination each want a different chart, and the
        wrong choice can hide a real effect or manufacture a fake one. We flag that choice each time it
        comes up.

        Along the way we practice one Python skill: **writing a small function** that uses **`np.diff`**
        to turn positions into motion, the vectorized way (no Python loop over frames).
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
        About **32%** of the training clips (base rate 0.320) are labelled aggression. We often split
        plots by this label to see where aggression differs from ordinary approach.

        **Two cohorts.** The clips come from **two independent groups of animals** recorded on separate
        dates (we call them cohort A and cohort B; 1,282 + 1,217 events). Each home cage belongs to
        exactly one cohort and one sex, so a cage is the natural independent unit — a fact that becomes
        important when we test claims later. One additional cage (`cam16`, all female, 780 events) is
        **held out entirely** and never touched until a later notebook, so we can check our methods on
        animals they were never tuned on.

        **Coordination previews "who interacts with whom."** With three animals in frame, a natural
        question is: which two are actually engaged? A simple physical clue is **coordination** — if two
        mice speed up and slow down *together*, their movements are **correlated**, and correlation
        hints they are interacting rather than moving independently. We make this precise in Section 4
        and test exactly how far the hint can be trusted.
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
    sexv = der["sex"].astype(str)
    cohort = der["cohort"].astype(str)
    cond = ev["condition"].astype(str)
    cr = ev["contact_rel"].astype(int)
    kp = ev["kp"]
    ranks = ev["ranks"]
    X = der["X"]
    fnames = [str(f) for f in der["feature_names"]]
    return X, agg, cohort, cond, cr, fnames, kp, ranks, sexv


@app.cell
def _(cu, np):
    # Shared helpers, defined once so the reactive graph stays clean. Each is a small function built on
    # np.diff — exactly the pattern the exercise asks you to reproduce yourself.

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
        """Coordination of two mice: the peak (best over small time-shifts) cross-correlation of their
        speed traces. Near +1 = move together; near 0 or negative = independent."""
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

    def padded_wavelet(sig, freqs, fps, w0=6.0, padlen=600):
        """Reflect-pad a short signal so low-frequency Morlet kernels fit, run cu.wavelet_power at
        wavelet width `w0`, then crop back to the original span. The padded flanks are the EDGE-EFFECT
        region we warn about; `w0` sets the time-vs-frequency resolution tradeoff.

        The pad is sized so the WIDEST kernel fits: cu.wavelet_power builds a Morlet template of length
        ~2*ceil(6*s)+1 samples, where the scale s grows with `w0` and with 1/f. At large `w0` and the
        lowest frequency that template can exceed a fixed 600-sample pad, and np.convolve(mode="same")
        would then return a longer array than the crop expects. We therefore expand `padlen` on demand so
        the padded signal is always at least as long as the widest kernel (any `w0`, any freq range)."""
        sig = np.nan_to_num(np.asarray(sig, float)); T = len(sig)
        _s = w0 / (2 * np.pi * float(np.min(freqs))) / (1.0 / fps)   # widest kernel scale (samples)
        _klen = 2 * int(np.ceil(_s * 6)) + 1                          # its full length
        padlen = max(int(padlen), _klen + 2, T)                       # padded signal must exceed it
        pad = max(0, (padlen - T) // 2)
        sp = np.pad(sig, pad, mode="reflect")
        P = cu.wavelet_power(sp, freqs, fps, w0=w0)
        return P[:, pad:pad + T]
    return (appr_appe_speed, leader_fraction, mouse_speed, padded_wavelet,
            pair_peak_corr, pre_speeds)


@app.cell
def _(cu, ev):
    # Pinned build-time constants (verified against the committed 2-cohort bundle 4d79758).
    EXAMPLE = cu.event_index_by_key(ev, "12192025_pre|cam.10.00046-2025-12-18T16|m0-m2|83141")
    # example event #909: cohort B, a female cage, NON-aggression, clean tracking, contact @40.
    CLEAN_AGG = [969, 560, 900, 53]        # clean, well-tracked aggression approaches (nan-frac 0.00)
    CLEAN_NON = [161, 341, 376, 345]       # clean, well-tracked non-aggression approaches
    # High- vs low-frequency speed content. Re-picked so the reported rhythm is the GENUINE interior
    # peak, not a lowest-bin/drift artifact: each HIFREQ event has most of its interior power above 3 Hz
    # (verified on an axis extended down to 0.3 Hz), and each LOFREQ event genuinely PEAKS near 1 Hz with
    # power collapsing below it (not a red-noise/drift ramp that merely floors at the lowest bin).
    HIFREQ = [234, 362, 2071]              # interior dominant ~4.8-5.2 Hz; sustained/burst fast motion
    LOFREQ = [540, 871, 1040, 629, 1003]   # genuine ~1 Hz peak (falls off below 1 Hz), gliding locomotion
    CC_STRONG = [1097, 667, 43, 2008, 649] # interacting pair strongly coordinated (peak r 0.81-0.85)
    CC_WEAK = [1514, 925, 979, 1933, 1972] # labelled pair weakly / anti-correlated (peak r -0.19..-0.29)
    CC_HI = 1097               # single strongly-coordinated example (peak r 0.85)
    CC_LO = 1514               # single weakly-coordinated example (pair01 -0.29)
    WHO_CHANCE = 1 / 3         # chance that a given pair is the most-correlated of 3
    WHO_FRAC_ALL = 0.41        # full corpus: interacting pair IS the most-correlated pair
    WHO_FRAC_STRONG = 0.90     # among strongly-coordinated events (peak r > 0.5): rises to ~0.90
    N_STRONG = 285             # count of strongly-coordinated events (peak r > 0.5)
    WHO_FRAC_SUB = 0.406       # exercise subsample (seed 0, n=399): win fraction to reproduce
    N_SUB = 399
    FULL_FRAC = 0.464          # pre-contact: fraction of aggression events where approacher LEADS
    FULL_N = 750
    NULL_MEAN = 0.506          # within-event shuffle null: mean approacher-leads fraction
    NULL_HI = 0.577            # shuffle null 97.5th percentile (band = 1-NULL_HI .. NULL_HI)
    return (CC_HI, CC_LO, CC_STRONG, CC_WEAK, CLEAN_AGG, CLEAN_NON, EXAMPLE,
            FULL_FRAC, FULL_N, HIFREQ, LOFREQ, NULL_HI, N_STRONG, N_SUB,
            WHO_CHANCE, WHO_FRAC_ALL, WHO_FRAC_STRONG, WHO_FRAC_SUB)


# ============================================================ 2. The example event
@app.cell(hide_code=True)
def _(EXAMPLE, mo):
    mo.md(
        rf"""
        ## An example interaction to follow throughout

        We keep one clip in view for the whole notebook, so every method has a concrete picture
        attached. Our example is **event #{EXAMPLE}** (cohort B, a female cage; a hand-scored
        **non-aggression** approach with clean tracking). By rank, the **approacher is the Sub (green)**
        mouse, the **approachee is the Int/Mid (blue)** mouse, and the **bystander is the Dom (red)**
        mouse. Contact occurs at frame 40.

        The animation below shows the three skeletons over the 2.6-second clip. Watch the green mouse
        close in on the blue one; the small marker shows the moment of contact. Keep this picture in
        mind: every number, spectrogram, and correlation we compute later is a *summary* of a movie like
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
        approaches (top) and four clean **non-aggression** approaches (bottom), all colored by rank so
        any difference you notice is a difference in *movement*, not color. Aggression clips tend to look
        faster and more entangled; ordinary approaches look calmer. Everything we do next is an attempt
        to turn "looks faster and more entangled" into numbers.

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
        interaction, such as `appr_speed_mean` (the approacher's average speed) or `appr_faces_appe`
        (how directly the approacher faces the approachee). Each feature has a **distribution**: the
        spread of its values across all 2,499 events. Aggression often shows up not as a shift in the
        *average* but as a heavier **tail** — a subset of events with unusually large values. A bar of
        group means would hide that entirely, which is why we plot **every individual event**.

        **Definition — distribution.** The distribution of a feature is how often each value occurs
        across events. We display it three ways:

        - a **strip plot** — every event as one jittered point (you see the raw cloud and the tails),
        - a **violin** — a smoothed silhouette of that cloud with the points beside it,
        - an **ECDF** (empirical cumulative distribution function) — for each value `v`, the fraction of
          events at or below `v`. Two ECDFs that separate vertically are two distributions that differ;
          it is the cleanest way to compare whole distributions, especially in the tails.

        **Method.** Pick a feature, a way to split the events into groups, and a plot style.

        - **Functions:** `cu.strip_points_fig`, `cu.violin_points_fig`, `cu.ecdf_fig`.
        - **Inputs:** one feature column (values), a grouping label per event, and a plot style.
        - **Output:** a figure showing the raw points, hoverable by event index. When the split has
          exactly two groups we also print a **Cohen's d** (how many pooled standard deviations apart
          the two group means are) and a **Mann–Whitney p** (a rank test of whether the two clouds
          differ) beneath the plot — a *hint*, not a finding, because the events are not independent.

        Try the **default**: `appr_speed_mean`, split by **aggression**, shown as an **ECDF**.
        """
    )
    return


@app.cell
def _(mo):
    feat_sel = mo.ui.dropdown(
        options=["appr_speed_mean", "appr_speed_max", "closing_speed", "pair_dist_min",
                 "appr_faces_appe", "appe_faces_appr", "heading_alignment", "appr_angvel",
                 "triangle_area_mean", "bystander_dist_mean", "appr_body_len"],
        value="appr_speed_mean", label="feature", full_width=True)
    split_sel = mo.ui.dropdown(options=["aggression", "sex", "condition", "cohort"],
                               value="aggression", label="split by", full_width=True)
    style_sel = mo.ui.dropdown(options=["ecdf", "strip", "violin"], value="ecdf",
                               label="plot style", full_width=True)
    return feat_sel, split_sel, style_sel


@app.cell
def _(X, agg, cohort, cond, cu, feat_sel, fnames, mo, np, sexv, split_sel, style_sel):
    _fi = fnames.index(feat_sel.value)
    _v = X[:, _fi]
    _idx = list(range(len(_v)))
    if split_sel.value == "aggression":
        _g = np.array(["aggression" if a else "non-aggression" for a in agg])
        _order = ["non-aggression", "aggression"]
        _colors = {"aggression": "#c1272d", "non-aggression": "#8899aa"}
    elif split_sel.value == "sex":
        _g = np.array(["male" if s == "M" else "female" for s in sexv])
        _order = ["male", "female"]; _colors = {"male": "#4c78a8", "female": "#e45756"}
    elif split_sel.value == "condition":
        _g = np.array(list(cond)); _order = ["pre", "dep", "post"]
        _colors = {"pre": "#4c78a8", "dep": "#e45756", "post": "#54a24b"}
    else:
        _g = np.array(list(cohort)); _order = sorted(set(cohort.tolist()))
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
    # Effect-size + rank-test hint when the split is exactly two groups.
    if len(_order) == 2:
        from scipy.stats import mannwhitneyu as _mwu
        _a = _v[_g == _order[1]]; _b = _v[_g == _order[0]]
        _p = _mwu(_a, _b, alternative="two-sided").pvalue
        _pooled = np.sqrt(((len(_a) - 1) * _a.var(ddof=1) + (len(_b) - 1) * _b.var(ddof=1)) /
                          (len(_a) + len(_b) - 2) + 1e-12)
        _d = (_a.mean() - _b.mean()) / _pooled
        _hint = (f"**Hint (not a finding):** {_order[1]} vs {_order[0]} — Cohen's d = **{_d:+.2f}** "
                 f"(|d|≈0.2 small, 0.5 medium, 0.8 large), Mann–Whitney p = **{cu.fmt_p(_p)}**. "
                 "A large-n p can be tiny for a shift too small to matter, and the events are not "
                 "independent — hold this loosely until the next notebook tests it at the cage level.")
    else:
        _hint = "**Hint:** with three groups we skip the two-group effect size; compare the clouds by eye."
    mo.vstack([mo.hstack([feat_sel, split_sel, style_sel]), _fig, mo.md(_hint)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **What to notice, and three traps.**

        - **A real shift (the default).** `appr_speed_mean` split by aggression: the aggression ECDF sits
          to the **right** — aggression events run faster on average (d ≈ +0.45, a medium effect). Good.
        - **Trap 1 — a feature that points the *wrong* way.** Switch the feature to `closing_speed` (how
          fast the gap between the two mice shrinks). It is "significant" (p ≈ 1e-5) but the aggression
          cloud sits **lower**, not higher — aggression approaches *close the gap more slowly*. If you
          had only read the p-value you would have called this evidence that aggression is fast closing;
          the sign says the opposite. Always read the direction, not just the star.
        - **Trap 2 — a feature that does nothing.** Switch to `appr_speed_max` (the single fastest frame).
          Its two clouds sit on top of each other (p ≈ 0.5, d ≈ 0). A lone fast frame is often a tracking
          jitter spike, not behavior; the *mean* speed separates the groups, the *max* does not.
        - **Trap 3 — a hint that needs a cage-level test.** Switch the split to **sex** and pick
          `heading_alignment` (how aligned the two mice's body orientations are). The male and female
          clouds sit at different heights (d ≈ 0.23). That is a real hint — but two mice in the same cage
          are not independent observations, so a difference between many male and many female *events*
          can be manufactured by a few cages. We do not resolve that here; we resolve it carefully, with
          cage-level tests and a negative control, in the next notebook.

        The habit for now: **look at the points, read the direction and the tails, and stay skeptical.**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Which chart when — strip vs violin vs ECDF

        The three styles are not interchangeable, and the wrong one can erase a real effect. A concrete
        rule of thumb:

        - **Strip plot** — best when *n* is small or you want to see every raw point and outlier. Honest,
          but crowded for thousands of events.
        - **Violin** — a smoothed silhouette. It reads beautifully for a *large, unimodal* distribution,
          but the smoothing **invents density**: for small *n* or a small shift it can hide the effect
          inside two nearly identical blobs, and it can even bulge past values the data never reach.
        - **ECDF** — no smoothing, no invented density; every event contributes one step. Two
          distributions that differ show up as two curves that pull apart, and differences **in the
          tails** (exactly where aggression lives) are visible rather than smeared.

        The panel below makes the failure concrete on a genuinely small shift: `heading_alignment` split
        by sex (d ≈ 0.23). On the **left** the two violins look like the same shape stacked twice — the
        effect is invisible. On the **right** the two ECDFs are cleanly, consistently separated. Same
        data, same effect; one chart shows it and one hides it. For small shifts and heavy tails, **default
        to the ECDF.**
        """
    )
    return


@app.cell
def _(X, cu, fnames, mo, sexv):
    _v = X[:, fnames.index("heading_alignment")]
    _g = ["male" if s == "M" else "female" for s in sexv]
    _order = ["male", "female"]; _colors = {"male": "#4c78a8", "female": "#e45756"}
    _vio = cu.violin_points_fig(_v, _g, group_order=_order, colors=_colors, points=None,
                                ylabel="heading_alignment", title="violin — effect invisible", height=380)
    _ecd = cu.ecdf_fig(_v, _g, group_order=_order, colors=_colors,
                       xlabel="heading_alignment", title="ECDF — same effect, now visible", height=380)
    mo.hstack([_vio, _ecd], widths=[1, 1])
    return


# ============================================================ 4. Correlation heatmap
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2 · The 19×19 feature-correlation heatmap

        **Why.** The 19 features are **not independent** measurements. Many rise and fall together across
        events, which means the representation carries **redundant** information. This matters because
        the next notebook uses **PCA** to replace many redundant features with a few independent ones —
        and this heatmap is the picture that motivates it.

        **Definition — Pearson correlation.** The Pearson correlation `r` between two features measures
        how linearly they move together across events: `r = +1` (rise together), `r = -1` (one rises as
        the other falls), `r = 0` (unrelated). We compute `r` for every pair of the 19 features and
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
    _fig = go.Figure(go.Heatmap(z=_C, x=fnames, y=fnames, colorscale="RdBu_r", zmid=0,
                                zmin=-1, zmax=1, colorbar=dict(title="r")))
    _fig.update_layout(template="plotly_white", height=640,
                       title="Feature–feature correlation (Pearson r) — off-diagonal blocks = redundancy",
                       margin=dict(l=10, r=10, t=50, b=130), font=dict(size=12))
    _fig.update_xaxes(tickangle=45, showgrid=False)
    _fig.update_yaxes(showgrid=False)
    _absC = np.abs(_C - np.eye(19))
    _i, _j = np.unravel_index(np.argmax(_absC), _C.shape)
    mo.vstack([_fig, mo.md(
        f"**Most-correlated pair:** `{fnames[_i]}` ↔ `{fnames[_j]}` (r = {_C[_i, _j]:.2f}). "
        "Several features are this redundant, which is why roughly **6 combined axes** will capture most "
        "of the variation once we run PCA — that is the whole reason the next notebook exists. (Note the "
        "colorscale: **red = positive** correlation, blue = negative, the red-is-up convention used "
        "everywhere in this course.)")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Seeing redundancy as a joint distribution

        A single number in the heatmap (say r = 0.92 between two features) is easier to trust once you
        see the point cloud behind it. Pick any two features and we draw their **joint distribution** as
        a plain scatter with a **histogram on each margin** — every event is one opaque point, no
        smoothing. Two highly correlated features produce a tight diagonal ridge; two independent
        features produce a round blob. Redundancy made visible: when the cloud is a thin diagonal, one
        feature is nearly predictable from the other, so keeping both wastes a dimension.

        **Why a scatter with marginals and not a 2-D density (KDE)?** Several of these features have
        **heavy tails** — a few events sit far out. A 2-D kernel density estimate would spend all its
        contrast resolving the empty region around those outliers and smear the dense core into a
        blob, and it can even paint probability where the data never go. The plain scatter with
        marginal histograms shows the core, the ridge, and the outliers honestly. The axes here are also
        **robust-clipped** to the 1st–99th percentile so a lone extreme point cannot squeeze the bulk
        into a line (every point is still plotted — only the default view is clipped).

        The default pair is the most-correlated one, `bystander_dist_mean` vs `bystander_dist_min`
        (r = 0.92): the two ways of summarizing how far the bystander sits from the interaction are
        nearly the same measurement.
        """
    )
    return


@app.cell
def _(fnames, mo):
    kx = mo.ui.dropdown(options=fnames, value="bystander_dist_mean", label="x feature", full_width=True)
    ky = mo.ui.dropdown(options=fnames, value="bystander_dist_min", label="y feature", full_width=True)
    return kx, ky


@app.cell
def _(X, agg, cu, fnames, kx, ky, mo, np):
    _ix = fnames.index(kx.value); _iy = fnames.index(ky.value)
    _x = X[:, _ix]; _y = X[:, _iy]
    _r = float(np.corrcoef(_x, _y)[0, 1])
    _grp = np.array(["aggression" if a else "non-aggression" for a in agg])
    _fig = cu.scatter_marginal_fig(
        _x, _y, groups=_grp, group_order=["non-aggression", "aggression"],
        colors={"aggression": "#c1272d", "non-aggression": "#8899aa"},
        hover=list(range(len(_x))), xlabel=kx.value, ylabel=ky.value,
        title=f"joint distribution: {kx.value} vs {ky.value}   (r = {_r:.2f})", height=520)
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

        **Method, part 1 — the raw traces.** First we plot, for event #{EXAMPLE}, the distance between the
        two interacting mice and each mouse's speed over time. This uses our `mouse_speed` helper, which
        is just `np.diff` of the body-center track followed by a length — the same small function you
        will write in the exercise. We expect the distance to collapse and the speeds to rise as contact
        approaches.
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
    mo.vstack([_fig, mo.md("The dotted line marks contact. The distance falls and both speeds rise into "
                           "the meeting. Next we ask *at what rhythm* the speed changes.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Method, part 2 — the wavelet spectrogram.** To read rhythm we use a **Morlet wavelet
        transform**. A wavelet is a short wave-shaped template. We slide a template of a given frequency
        along the speed signal and measure how strongly the signal matches it at each moment. Repeating
        this across many frequencies produces a **spectrogram** — a picture with **time** on the
        horizontal axis, **frequency** (rhythm, in cycles per second, Hz) on the vertical axis, and
        brightness showing how much of that rhythm is present at that moment.

        - **Function:** `cu.wavelet_power(signal, freqs, fps, w0)` (wrapped by `padded_wavelet`, which
          pads the short clip so low-frequency templates fit).
        - **Inputs:** a 1-D signal (here the approacher's speed), a list of frequencies to test, the
          sampling rate (50 fps), and the wavelet width `w0`.
        - **Output:** a `frequency × time` grid of power — bright where that rhythm is present.

        **Reading the dominant rhythm — and why the naive way fails.** We want one number: the mouse's
        *dominant* speed rhythm. The obvious recipe — "take the brightest frequency" — is a **trap** on
        short clips. Low-frequency templates are wider than the 2.6 s clip, so they run off both ends;
        we pad the signal to fit them, but the padded flanks are fabricated and artificially bright at
        the **bottom** of the frequency axis. A naive argmax then reports ≈1 Hz for *every* event,
        driven by the padded edges rather than the behavior. We instead use
        `cu.dominant_frequency`, which takes the brightest frequency **only over the trustworthy interior
        of the clip** (the central 60%, away from the padded edges) and reports the per-frame value; we
        summarize with its median. The shaded bands below mark the untrustworthy padded edges.

        **The `w0` control teaches a real tradeoff — the uncertainty principle of time and frequency.**
        `w0` is the number of oscillations packed into each wavelet template. It is a **chosen**
        hyperparameter, and the choice is a genuine tradeoff, not a cosmetic one:

        - **small `w0`** (a short template) pins down *when* a burst happened but is blurry about *what
          frequency* it was — sharp time, fuzzy frequency;
        - **large `w0`** (a long template) resolves the frequency precisely but smears it across time —
          sharp frequency, fuzzy time.

        You cannot have both at once. Slide `w0` and watch the same spectrogram trade vertical sharpness
        for horizontal sharpness. The default `w0 = 6` is the conventional middle ground.
        """
    )
    return


@app.cell
def _(mo):
    w0_slider = mo.ui.slider(3.0, 12.0, value=6.0, step=1.0, label="wavelet width w0 (oscillations per template)",
                             debounce=True, full_width=True)
    return (w0_slider,)


@app.cell
def _(EXAMPLE, appr_appe_speed, cu, go, kp, mo, np, padded_wavelet, w0_slider):
    _s0, _ = appr_appe_speed(kp, EXAMPLE)
    _freqs = np.linspace(1.0, 12.0, 45)
    _P = padded_wavelet(_s0, _freqs, cu.FPS, w0=float(w0_slider.value), padlen=600)
    _t = np.arange(_P.shape[1]) / cu.FPS
    _domf = cu.dominant_frequency(_P, _freqs)                  # interior-only per-frame dominant freq
    _dom = float(np.nanmedian(_domf))
    _T = _P.shape[1]; _edge = 0.2 * _T / cu.FPS; _end = _T / cu.FPS
    # A fitted number the w0 control actually MOVES: temporal spread of the dominant rhythm. Take the
    # brightest interior frequency row and measure the power-weighted std (in seconds) along time. As w0
    # grows the template smears energy across MORE time, so this spread widens — the time half of the
    # time–frequency uncertainty tradeoff, made quantitative rather than merely visual.
    _lo = int(0.2 * _T); _hi = int(0.8 * _T)
    _row = _P[int(np.argmax(_P[:, _lo:_hi].sum(1))), _lo:_hi]
    _w = _row / (_row.sum() + 1e-12); _tt = np.arange(_lo, _hi) / cu.FPS
    _tmean = float(np.sum(_tt * _w)); _tspread = float(np.sqrt(np.sum((_tt - _tmean) ** 2 * _w)))
    _fig = go.Figure(go.Heatmap(z=_P, x=_t, y=_freqs, colorscale="Viridis", colorbar=dict(title="power")))
    _fig.add_vrect(x0=0, x1=_edge, fillcolor="#000", opacity=0.18, line_width=0)
    _fig.add_vrect(x0=_end - _edge, x1=_end, fillcolor="#000", opacity=0.18, line_width=0)
    _fig.add_hline(y=_dom, line=dict(color="white", dash="dot"),
                   annotation_text=f"interior dominant ≈ {_dom:.1f} Hz", annotation_position="top left")
    _fig.update_layout(template="plotly_white", height=400, font=dict(size=14),
                       title=f"Morlet spectrogram of approacher speed (w0 = {w0_slider.value:.0f}, "
                             f"temporal spread ≈ {_tspread:.2f} s)",
                       xaxis_title="time (s)", yaxis_title="frequency (Hz)",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False); _fig.update_yaxes(showgrid=False)
    mo.vstack([w0_slider, _fig, mo.md(
        f"**Interior dominant rhythm ≈ {_dom:.1f} Hz** — a **slow** rhythm, the pace of ordinary "
        "locomotion (a mouse taking a step or two per second), not a fast oscillation. Notice the "
        "dominant *frequency* barely moves as you slide `w0`: the rhythm the mouse actually has is a "
        "property of the behavior, not of the template. What `w0` **does** change is the **localization** "
        f"of that rhythm. The readout that moves is the **temporal spread ≈ {_tspread:.2f} s** — how "
        "widely the dominant band is smeared along time. Slide `w0` up from 3 to 12 and watch this number "
        "**grow** (≈0.30 s → ≈0.45 s): a wider template resolves the frequency more sharply (the blob "
        "narrows vertically) but pays for it by blurring *when* the rhythm happened (it stretches "
        "horizontally). That is the uncertainty principle, priced in seconds. The **shaded bands** are the "
        "padded edges the detector deliberately ignores.")])
    return


@app.cell(hide_code=True)
def _(HIFREQ, LOFREQ, mo):
    mo.md(
        rf"""
        ### What high frequency looks like: fast, jittery movement

        The example event's speed is dominated by a slow (~1 Hz) locomotor rhythm. But some events carry
        **high-frequency** speed content: the speed rises and falls several times per second, which
        corresponds to **quick, jerky movement** — rapid darting, scrambling, repeated start-and-stop —
        rather than a smooth glide. We *find* those events by keeping the ones whose power genuinely sits
        at high frequency in the trusted interior (not events that merely floor at the lowest bin because
        of slow drift). Genuinely sustained fast movement turns out to be **uncommon** on this rig — most
        mice move at ~1 Hz — so the high-frequency grid below is small on purpose.

        Below are two grids of skeleton animations. The **top** grid is three high-frequency events
        (interior dominant ~5 Hz — the clearest fast examples the corpus offers); the **bottom** grid is
        smooth, low-frequency events (genuine ~1 Hz peak). Watch how the top animations look abrupt and
        stuttery while the bottom ones glide. High-frequency movement is a description of *how* the animal
        moved — both aggression and non-aggression events appear here, so it is not by itself a sign of
        aggression.

        High-frequency: {HIFREQ} &nbsp;·&nbsp; Low-frequency: {LOFREQ}.
        """
    )
    return


@app.cell
def _(HIFREQ, LOFREQ, cr, cu, kp, mo, ranks):
    _hi = cu.grid_gif_bytes([(kp[i], ranks[i], int(cr[i])) for i in HIFREQ], ncols=3, cell=135, fps=18)
    _lo = cu.grid_gif_bytes([(kp[i], ranks[i], int(cr[i])) for i in LOFREQ], ncols=5, cell=135, fps=18)
    mo.vstack([
        mo.md("**High-frequency movement** — short darts and abrupt stops (speed changes many times/s):"),
        mo.Html(cu.gif_img_html(_hi, width=380)),
        mo.md("**Low-frequency movement** — smooth, gliding locomotion (speed changes slowly):"),
        mo.Html(cu.gif_img_html(_lo, width=600)),
    ])
    return


@app.cell
def _(HIFREQ, appr_appe_speed, cu, go, kp, mo, np, padded_wavelet):
    # Spectrogram of one HIGH-frequency event, for direct contrast with the ~1 Hz example above.
    _i = HIFREQ[0]
    _s0, _ = appr_appe_speed(kp, _i)
    _freqs = np.linspace(1.0, 12.0, 45)
    _P = padded_wavelet(_s0, _freqs, cu.FPS, padlen=600)
    _t = np.arange(_P.shape[1]) / cu.FPS
    _dom = float(np.nanmedian(cu.dominant_frequency(_P, _freqs)))
    _T = _P.shape[1]; _edge = 0.2 * _T / cu.FPS; _end = _T / cu.FPS
    _fig = go.Figure(go.Heatmap(z=_P, x=_t, y=_freqs, colorscale="Viridis", colorbar=dict(title="power")))
    _fig.add_vrect(x0=0, x1=_edge, fillcolor="#000", opacity=0.18, line_width=0)
    _fig.add_vrect(x0=_end - _edge, x1=_end, fillcolor="#000", opacity=0.18, line_width=0)
    _fig.add_hline(y=_dom, line=dict(color="white", dash="dot"),
                   annotation_text=f"interior dominant ≈ {_dom:.1f} Hz", annotation_position="top left")
    _fig.update_layout(template="plotly_white", height=380, font=dict(size=14),
                       title=f"High-frequency event #{_i} — bright band in the trusted interior",
                       xaxis_title="time (s)", yaxis_title="frequency (Hz)",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False); _fig.update_yaxes(showgrid=False)
    mo.vstack([_fig, mo.md(
        f"Here the bright band sits **higher** on the frequency axis (~{_dom:.1f} Hz) than in the "
        "example event's slow spectrogram, and — crucially — it is bright in the **trusted interior**, "
        "not just in the padded edges. The band moved up because the speed itself changes more times per "
        "second, exactly the jittery motion in the top grid.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Why a wavelet and not a single Fourier transform (FFT)

        A natural alternative is the **Fourier transform**: decompose the whole clip into a fixed set of
        pure sine waves and read off which frequencies are present. The FFT gives one **global** spectrum
        for the entire 2.6 s. The problem is that a mouse's rhythm is **non-stationary** — it is slow
        while the animals are far apart and can burst faster near contact — and the FFT averages all of
        that into a single blurred spectrum with **no sense of *when*** each rhythm occurred.

        The panel below shows both for the example event. On the **left**, the FFT power spectrum: one
        curve, one summary for the whole clip. On the **right**, the wavelet spectrogram from above:
        frequency *and* time. The wavelet keeps the "when," which is exactly what we need for a signal
        whose rhythm changes across the encounter. That is the whole reason we reach for a wavelet here
        rather than an FFT.
        """
    )
    return


@app.cell
def _(EXAMPLE, appr_appe_speed, cu, go, kp, make_subplots, mo, np, padded_wavelet):
    _s0, _ = appr_appe_speed(kp, EXAMPLE)
    _sig = np.nan_to_num(_s0 - np.nanmean(_s0))
    _F = np.abs(np.fft.rfft(_sig)) ** 2
    _ff = np.fft.rfftfreq(len(_sig), d=1.0 / cu.FPS)
    _keep = _ff <= 12.0
    _freqs = np.linspace(1.0, 12.0, 45)
    _P = padded_wavelet(_s0, _freqs, cu.FPS, padlen=600)
    _t = np.arange(_P.shape[1]) / cu.FPS
    _fig = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58],
                         subplot_titles=("FFT — one spectrum for the whole clip",
                                         "wavelet — frequency AND time"))
    _fig.add_scatter(x=_ff[_keep], y=_F[_keep], mode="lines", line=dict(color="#7b3294"),
                     showlegend=False, row=1, col=1)
    _fig.add_trace(go.Heatmap(z=_P, x=_t, y=_freqs, colorscale="Viridis", showscale=False),
                   row=1, col=2)
    _fig.update_xaxes(title_text="frequency (Hz)", row=1, col=1, showgrid=False)
    _fig.update_yaxes(title_text="power", row=1, col=1, showgrid=False)
    _fig.update_xaxes(title_text="time (s)", row=1, col=2, showgrid=False)
    _fig.update_yaxes(title_text="frequency (Hz)", row=1, col=2, showgrid=False)
    _fig.update_layout(template="plotly_white", height=360, font=dict(size=13),
                       margin=dict(l=10, r=10, t=50, b=10))
    mo.vstack([_fig, mo.md("The FFT tells you *which* rhythms are in the clip; the wavelet also tells you "
                           "*when* — indispensable when the rhythm changes as the mice meet.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Two honest limits of the wavelet on this data

        1. **Time–frequency trade-off.** As the `w0` slider made concrete: a template narrow enough to
           pin down *when* a burst happened is blurry about *what frequency* it was, and vice versa. You
           cannot have sharp timing and sharp frequency at once — that is a mathematical fact, not a
           software limitation.
        2. **Edge effects.** A low-frequency template is wider than the 2.6 s clip, so it runs off both
           ends. We pad the signal just to fit the template, but the padded flanks are fabricated, not
           measured (the shaded bands above). Trust the **middle** of each spectrogram and distrust the
           extreme left and right edges — which is exactly why the dominant-frequency detector reads the
           interior only.
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
        two speed traces rise and fall together (**highly coordinated**), near 0 they are unrelated, and
        negative means they tend to move in opposition.

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
def _(CC_HI, cr, cu, kp, make_subplots, mo, mouse_speed, ranks):
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
    _fig.update_layout(template="plotly_white", height=360, font=dict(size=13),
                       title=f"STRONG coordination — event #{_i}: peak correlation ≈ {_peak:.2f}",
                       margin=dict(l=10, r=10, t=60, b=10))
    _fig.update_xaxes(title_text="time (s)", row=1, col=1, showgrid=False)
    _fig.update_yaxes(title_text="speed (px/frame)", row=1, col=1, showgrid=False)
    _fig.update_xaxes(title_text="lag (frames)", row=1, col=2, showgrid=False)
    _fig.update_yaxes(title_text="correlation", row=1, col=2, showgrid=False)
    _gif = cu.event_gif_bytes(_k, ranks[_i], contact_rel=int(cr[_i]), cell=170, fps=18)
    mo.vstack([_fig, mo.Html(cu.gif_img_html(_gif, width=200)),
               mo.md("The two speed traces rise and fall together, so the peak correlation is high, and in "
                     "the animation the two mice really are moving as a coordinated pair.")])
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
    _fig.update_layout(template="plotly_white", height=340, font=dict(size=13),
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
        The same contrast across several events at once. **Top:** five events where the interacting pair
        is strongly coordinated (peak r ≈ 0.81–0.85). **Bottom:** five where the labelled pair is weakly
        or anti-coordinated (peak r ≈ −0.19 to −0.29). Watch the top pairs move in lockstep and the
        bottom pairs do their own thing.

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

        Rather than report a single number at one threshold, we plot the whole **trend**: the curve below
        is the win-fraction as a function of a **coordination threshold** — we keep only events where the
        interacting pair's peak correlation exceeds the threshold, then ask how often it is the
        most-correlated pair. Each point is annotated with `n`, the number of events surviving that
        threshold, because the fraction gets noisy as `n` shrinks.

        - At the **left** (keep everything), the interacting pair wins ~**{WHO_FRAC_ALL*100:.0f}%** of
          the time — clearly above chance, so correlation genuinely carries information about who
          interacts, but far from perfect. The reason it is imperfect is the **bystander confound**: two
          mice can move together for reasons unrelated to interacting (both resting, both walking in
          parallel, a shared startle), which can make a non-interacting pair the most correlated.
        - As the threshold **rises**, the win-fraction climbs toward ~**{WHO_FRAC_STRONG*100:.0f}%**: when
          there is *real* coordination in the clip, it almost always belongs to the interacting pair.

        This is the same **precision-vs-recall** tradeoff you meet everywhere in classification: a
        stricter threshold keeps fewer events (`n` falls) but the ones it keeps are cleaner (the fraction
        rises). The slider moves a reader line along the static curve so you can read off both numbers at
        any threshold. We stop the slider at **0.5**, because beyond it `n` becomes too small to trust the
        fraction — reading a rate off a dozen events is how people fool themselves.
        """
    )
    return


@app.cell
def _(cu, kp, np, pair_peak_corr):
    # Precompute all three pairwise coordinations for the WHOLE corpus (one cached pass, ~9 s).
    _r01 = np.full(len(kp), np.nan); _r02 = _r01.copy(); _r12 = _r01.copy()
    for _i in range(len(kp)):
        _k = kp[_i]
        if min(cu._centroids(_k[:, m]).std() for m in range(3)) < 1e-6:
            continue
        _r01[_i] = pair_peak_corr(_k, 0, 1)
        _r02[_i] = pair_peak_corr(_k, 0, 2)
        _r12[_i] = pair_peak_corr(_k, 1, 2)
    cc_r01, cc_r02, cc_r12 = _r01, _r02, _r12
    # Static win-fraction-vs-threshold curve (guard n>=20 so a noisy tail never plots).
    _thr = np.round(np.arange(-0.2, 0.55, 0.05), 2)
    _fin = np.isfinite(cc_r01)
    _tx, _wy, _wn = [], [], []
    for _th in _thr:
        _keep = _fin & (cc_r01 > _th)
        if _keep.sum() >= 20:
            _tx.append(float(_th))
            _wy.append(float((cc_r01[_keep] >= np.maximum(cc_r02[_keep], cc_r12[_keep])).mean()))
            _wn.append(int(_keep.sum()))
    cc_thr_x = np.array(_tx); cc_win_y = np.array(_wy); cc_win_n = np.array(_wn)
    return cc_r01, cc_r02, cc_r12, cc_thr_x, cc_win_n, cc_win_y


@app.cell
def _(mo):
    cc_thr = mo.ui.slider(-0.2, 0.5, value=0.0, step=0.05,
                          label="coordination threshold (keep interacting-pair peak r >)",
                          debounce=True, full_width=True)
    return (cc_thr,)


@app.cell
def _(WHO_CHANCE, cc_thr, cc_thr_x, cc_win_n, cc_win_y, go, mo, np):
    _th = float(cc_thr.value)
    _k = int(np.argmin(np.abs(cc_thr_x - _th)))
    _frac = float(cc_win_y[_k]); _n = int(cc_win_n[_k])
    _fig = go.Figure()
    _fig.add_scatter(x=cc_thr_x, y=cc_win_y, mode="lines+markers", line=dict(color="#4c78a8"),
                     text=[f"n={n}" for n in cc_win_n], name="win fraction",
                     hovertemplate="thr>%{x:.2f}<br>win=%{y:.2f}<br>%{text}<extra></extra>")
    _fig.add_hline(y=WHO_CHANCE, line=dict(color="#333", dash="dash"),
                   annotation_text=f"chance among 3 pairs ({WHO_CHANCE:.2f})", annotation_position="bottom right")
    _fig.add_vline(x=_th, line=dict(color="#d62728", dash="dot"))
    _fig.add_scatter(x=[_th], y=[_frac], mode="markers", marker=dict(size=13, color="#d62728"),
                     showlegend=False)
    _fig.update_layout(template="plotly_white", height=420, font=dict(size=14),
                       yaxis_title="fraction: interacting pair is most-correlated", yaxis_range=[0, 1],
                       xaxis_title="coordination threshold (interacting-pair peak r)",
                       title="Win-fraction climbs with the coordination threshold (n falls)",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False); _fig.update_yaxes(showgrid=False)
    mo.vstack([cc_thr, _fig, mo.md(
        f"**At threshold r > {_th:.2f}:** the interacting pair is the most-correlated pair "
        f"**{_frac:.2f}** of the time, over **n = {_n}** surviving events. Slide right and the fraction "
        "climbs toward ~0.90 while `n` shrinks — **coordination identifies the interacting pair, but "
        "only when there is real coordination to read.**")])
    return


# ============================================================ 7. Who leads whom
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### A directed version: who moves first?

        **Why.** Cross-correlation also carries **order**: if the approacher's speed changes and the
        approachee's follows a moment later, the peak alignment sits at a positive **lag**, and we say the
        approacher **leads**. "Who moves first before contact" is a natural behavioral question — but we
        will see it is hard to answer reliably on such short clips, and reporting that honestly is part of
        the job.

        We look only at frames **before contact**. After contact both mice necessarily move together, so
        including those frames would answer the question trivially.

        First, a quick check that the estimator works when the answer is known: we build two signals where
        one is a delayed copy of the other and confirm `cross_corr_lag` recovers the delay.
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
        "A positive lag means A leads B. The estimator recovers the imposed lag cleanly on a long, clean "
        "signal — the regime the real, short, noisy mouse clips do **not** enjoy.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now the same test on real pre-contact traces. For each aggression event we ask whether the
        approacher leads, and report the **fraction of events** in which it does, as a **point with an
        error bar** (not a full-height bar, which would exaggerate a fraction that is really sitting near
        one-half). With no consistent leader that fraction sits near **0.50**. The gray band is a
        **shuffle null**: we scrambled the traces within each event to see how far from 0.50 the fraction
        wanders by chance. A real leader effect would have to poke **outside** that band.
        """
    )
    return


@app.cell
def _(mo):
    coord_maxlag = mo.ui.slider(4, 12, value=10, step=1, label="max lag (frames)",
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
        _names.append(_nm); _fracs.append(_f if _f == _f else np.nan); _ns.append(_n)
    # Point + error bar (Wilson-style ±1/(2 sqrt n) sampling spread on a fraction).
    _err = [0.5 / np.sqrt(max(n, 1)) for n in _ns]
    _fig = go.Figure()
    _fig.add_hrect(y0=1 - NULL_HI, y1=NULL_HI, fillcolor="#bbbbbb", opacity=0.25, line_width=0,
                   annotation_text="within-event shuffle null (95%)", annotation_position="top left")
    _fig.add_hline(y=0.5, line=dict(color="#333", dash="dash"),
                   annotation_text="no consistent leader (0.50)")
    _fig.add_scatter(x=_names, y=_fracs, mode="markers",
                     error_y=dict(type="data", array=_err, visible=True),
                     marker=dict(size=13, color="#4c78a8"),
                     text=[f"n={n}" for n in _ns], hovertemplate="%{y:.3f}<br>%{text}<extra></extra>")
    _fig.update_layout(template="plotly_white", height=400, font=dict(size=14),
                       yaxis_title="fraction approacher LEADS", yaxis_range=[0, 1],
                       title=f"Pre-contact lead–lag — subsample, split by {coord_split.value}",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False); _fig.update_yaxes(showgrid=False)
    mo.vstack([mo.hstack([coord_maxlag, coord_split]), _fig, mo.md(
        f"**Precomputed full corpus (all {FULL_N} usable aggression events):** approacher-leads fraction "
        f"= **{FULL_FRAC:.3f}** — inside the gray shuffle band. The subsample points land near 0.50 too: "
        "there is **no robust leader** on these short, noisy, pre-contact windows. That is the honest "
        "result, and — unlike the who-interacts question — no threshold rescues it, because the *sign of "
        "a lag* on a ~1 s noisy trace is far less stable than the *magnitude* of a correlation.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Why cross-correlation, and what it cannot see

        Cross-correlation is the right first tool because it is simple, fast, and directly encodes the two
        things we want — *strength* (the peak height) and *order* (the peak lag). But it is worth naming
        what it misses, so you know when to reach for something heavier:

        - **It only sees *linear*, time-locked coupling.** Two mice could interact through a nonlinear or
          variable-delay relationship that a single peak correlation blurs. **Mutual information** or
          **coherence** (correlation resolved frequency-by-frequency) would catch some of that, at the
          cost of needing far more data than a 130-frame clip provides.
        - **It cannot separate coupling from a shared cause.** A high correlation between two mice can come
          from them driving each other *or* from both responding to the bystander or a common startle. The
          who-interacts confound above is exactly this. The optional Granger section below sharpens the
          question toward *prediction*, but even that cannot, on its own, prove *cause*.

        On short, noisy clips the simple, robust tool usually wins; the fancier estimators mostly buy
        precision we do not have the data to spend.
        """
    )
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
        subsample of events, the labelled **interacting pair** is the most-correlated of the three pairs
        **more often than the 1/3 you'd get by chance**. You write the one function the whole pipeline
        depends on — `my_speed` — and a provided loop uses it to build the plot and the number.
        """
    )
    return


@app.cell(hide_code=True)
def _(WHO_CHANCE, WHO_FRAC_SUB, mo):
    mo.md(
        rf"""
        ### What to do

        You will edit **one line** inside `my_speed`. Everything else is written for you.

        **Expected picture.** A strip plot with three columns — one per pair type — showing every event's
        peak coordination as an individual point. The **interacting** column's cloud should sit **a little
        higher** than the two bystander columns. The self-check then confirms the interacting pair wins the
        "most-correlated" contest about **{WHO_FRAC_SUB:.2f}** of the time — above the **{WHO_CHANCE:.2f}**
        chance line, a real but imperfect cue.
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
        disp = np.diff(cen, axis=0)                          # <-- YOUR TURN (shown filled so it runs)
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
                pair's cloud sits a bit higher than the two bystander clouds, and the interacting pair is
                the most-correlated pair about **0.41** of the time: above the **0.33** chance line, but
                far from certain, because a bystander pair can be more coordinated by accident.
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
            f"most-correlated of the three pairs **{ex_win:.3f}** of the time (n={ex_n} usable events) — "
            f"above the {WHO_CHANCE:.2f} chance line, and consistent with the pinned **0.41**. "
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
        sharper question: does knowing the **past of mouse A** improve our prediction of **mouse B's next
        step**, beyond what B's own past already tells us? If it does, we say A "Granger-causes" B.

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
                f"(p = {cu.fmt_p(_g['p_xy'])}); approachee→approacher F = {_g['f_yx']:.2f} "
                f"(p = {cu.fmt_p(_g['p_yx'])}).")
    except Exception as _e:
        _txt = f"(Granger skipped: {_e})"
    mo.accordion({
        "Granger on the example event, with the caveat": mo.md(
            rf"""
            {_txt}

            **The common-cause caveat.** Granger measures **prediction, not cause**. Both mice can be
            driven by a **shared third factor** — the bystander, or a common startle — which makes A look
            like it drives B when neither actually does. Bivariate Granger is also not *conditional*: to
            move from "A predicts B" to "A predicts B *given the bystander*," you would add the third
            mouse's trace as an extra input (a conditional, multivariate model). On 1-second,
            nonstationary windows, treat any single-event Granger number as a hint, not a verdict — the
            same lesson the who-leads test taught us.
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
        1. **Name a confounder.** Coordination is not proof of interaction. What shared cause on this rig
           could make two mice look coupled when neither is driving the other? (The bystander; a common
           startle or arousal spike shared by all three.)
        2. **Why does thresholding help who-interacts but not who-leads?** Restricting to strongly
           coordinated events pushed the who-interacts accuracy to ~90%, yet no threshold rescues the
           leader test. Why? (The *magnitude* of coordination is a strong, stable signal; the *sign of the
           lag* on a ~1 s noisy window is not.)
        3. **Wavelet vs FFT.** When is a wavelet more appropriate than a single FFT? (When the spectrum is
           *non-stationary* — the rhythm changes across the 2.6 s, which an FFT would average into one
           blurred spectrum.)
        4. **Chart choice.** Why default to an ECDF over a violin for the `heading_alignment`-by-sex shift?
           (The shift is small and the tails matter; a violin's smoothing hid the effect, while the ECDF
           kept it visible.)
        5. **Direction, not just the star.** `closing_speed` split by aggression has a tiny p-value. Why is
           that not evidence that aggression closes the gap faster? (The aggression cloud sits *lower* —
           the effect points the opposite way; the p-value alone does not tell you the sign.)
        """
    )
    return


# ============================================================ 11. Throughline: answer + next question
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## What we found, and what it raises next

        **The answer to this notebook's question.** An interaction, seen as a signal over time, is highly
        **structured**: the pair distance collapses, both speeds rise into contact, and the movement
        carries a mostly **slow (~1–2 Hz) locomotor rhythm** with occasional high-frequency bursts we
        could find and render. Coordination between two mice is a **genuine cue to who interacts with
        whom** — imperfect overall (~41%, versus 33% chance) but nearly decisive (~90%) whenever real
        coordination is present. In contrast, *who moves first* before contact is **not** recoverable from
        these short, noisy windows: the leader estimate sits inside its shuffle null, and we reported that
        honestly.

        **What the exploration also revealed.** The 19 features are **heavily redundant** — the
        correlation heatmap is full of bright off-diagonal blocks, and whole groups of features move
        together (the tightest pair, the two bystander-distance features, at r = 0.92). We are carrying 19
        numbers that really live in far fewer independent directions. Along the way we also collected a set
        of habits about **reading the data honestly**: plot the points, read the direction not just the
        star, prefer an ECDF for small shifts, prefer a scatter-with-marginals over a smoothed 2-D density
        for heavy tails, and never trust a rate computed from a handful of events.

        **The next question.** If the 19 features are this correlated, then:

        > What are the few **underlying dimensions** that actually vary across interactions — and once we
        > have them, what **behavioral types** exist, and do **sex** or **food deprivation** genuinely
        > change behavior, or only appear to?

        The next notebook takes the first step: **PCA**, which finds those few combined axes automatically
        and lets us measure exactly how many dimensions the 19 features are hiding.
        """
    )
    return


if __name__ == "__main__":
    app.run()
