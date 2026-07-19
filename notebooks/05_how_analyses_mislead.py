# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "marimo>=0.9",
#     "numpy>=1.24,<2.1",
#     "scipy>=1.11",
#     "pandas>=2.0",
#     "scikit-learn>=1.3",
#     "plotly>=5.20",
#     "h5py>=3.8",
#     "gdown>=5.0",
#     "openpyxl>=3.1",
#     "imageio>=2.34",
#     "imageio-ffmpeg>=0.4",
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
    import warnings as _warnings
    _warnings.filterwarnings("ignore")   # nanmean-of-all-nan and sklearn convergence chatter are expected

    # ---- the training corpus and its 19-feature / cage / sex / cohort bundle, aligned row-for-row ----
    ev     = cu.load_events(cu.data_path("data/train_events.npz", ROOT))
    der    = cu.load_derived("train", ROOT)

    X      = der["X"]                                    # (2499, 19) allocentric features
    fn     = [str(f) for f in der["feature_names"]]      # the 19 feature names
    yagg   = ev["agg_label"].astype(int)                # 1 = aggression
    cage   = der["cage"]                                # cohort-unique cage id (9-15, 109-115)
    sex    = der["sex"].astype(str)                     # 'M'/'F', fixed per cage
    cohort = der["cohort"].astype(str)                  # '12192025' / '20260222'
    cond   = ev["condition"].astype(str)                # 'pre'/'dep'/'post'

    N      = int(len(X))
    ucage  = np.unique(cage)                            # the 14 cohort-unique cages
    base_rate = float(yagg.mean())                     # 0.320

    # feature-name -> column index, used throughout
    HI  = fn.index("heading_alignment")     # positive control (sex effect that SURVIVES)
    BLI = fn.index("appr_body_len")         # negative control (sex effect that FAILS)
    BDI = fn.index("bystander_dist_mean")   # food-deprivation readout (paired, survives)
    TAI = fn.index("triangle_area_mean")    # effect-size-vs-p demo (tiny p, tiny d)

    EXAMPLE = cu.event_index_by_key(
        ev, "12192025_pre|cam.10.00046-2025-12-18T16|m0-m2|83141")
    return (BDI, BLI, EXAMPLE, HI, N, TAI, base_rate, cage, cohort, cond,
            der, ev, fn, sex, ucage, yagg, X)


@app.cell(hide_code=True)
def _(N, mo, ucage):
    mo.md(
        rf"""
        # NB05 · How analyses mislead

        ## Where we are, and what this notebook is for

        The last four notebooks built a pipeline: from raw keypoints, to a body-centered set of **19
        numbers per interaction**, to a low-dimensional map of behavioral types, and finally to the
        first hints that an experimental variable — the animals' sex, or food deprivation — moves
        behavior. Every one of those steps produced a *number*: a Cohen's d, a cluster's aggression
        lift, a p-value, a decoder's accuracy.

        This notebook is about the ways those numbers **lie to us**, and the specific habits that stop
        them from lying. It is the course's rigor spine. We are not adding a new measurement of
        behavior; we are learning to trust — or distrust — the measurements we already have.

        The plan is unusual. Each section takes one classic statistical mistake, **commits it on real
        course data where we already know the right answer**, watches it produce a confident-looking
        wrong result, and then shows the method that fixes it. Seeing an analysis fail on data you
        understand is the only way to recognize the same failure later on data you do not.

        ## The mistakes, in order

        1. **Unit of analysis / pseudoreplication** — counting {N:,} events as {N:,} independent
           observations when they come from only {len(ucage)} cages.
        2. **Choosing the wrong test** — direction versus magnitude, paired versus unpaired, and when a
           violin hides what an ECDF shows.
        3. **Multiple comparisons** — scan enough features and something is always "significant."
        4. **Effect size versus p-value** — at large n a meaningless difference has a tiny p.
        5. **Circular analysis / double-dipping** — using the same data to *choose* a hypothesis and to
           *test* it.
        6. **Cross-validation leakage** — the centerpiece: a decoder that scores 0.95 by cheating on
           time, and 0.6 when it stops.
        7. **Reading a null distribution** — what a permutation p-value actually means, and how many
           shuffles you need.
        8. **Putting it together** — effect sizes with confidence intervals, and permutation at the
           right unit, as the honest default.

        There is one throughline question behind all eight: **at what unit is my observation actually
        independent, and does my analysis respect it?** Almost every mistake below is a different mask
        on that one question.
        """
    )
    return


# ============================================================================================
# ==============  SECTION 1 — UNIT OF ANALYSIS / PSEUDOREPLICATION  ===========================
# ============================================================================================
@app.cell(hide_code=True)
def _(N, mo, ucage):
    mo.md(
        rf"""
        ---
        # 1 · The unit of analysis, and pseudoreplication

        ## Why this matters

        Every statistical test asks "could this pattern have arisen by chance?" — and to answer it, the
        test needs to know **how many independent things it is looking at**. Get that count wrong and
        every p-value it reports is wrong, usually by a huge margin, and always in the direction of
        *overconfidence*.

        ## Definitions

        - **Independent observation** — a measurement whose value is not determined, even partly, by
          another measurement in the dataset. Two events from the same cage on the same night are *not*
          independent: they share the same animals, lighting, camera, and tracking quirks.
        - **The unit of analysis** — the level at which your observations are actually independent. For
          us it is sometimes the **event**, sometimes the **cage**, depending on the question.
        - **Pseudoreplication** — treating non-independent measurements as if they were independent, so
          the test believes it has far more evidence than it does. Our corpus has **{N:,} events** but
          only **{len(ucage)} cages**. For a variable fixed within a cage, the honest sample size is
          {len(ucage)}, not {N:,}.
        - **Permutation test** — a test that builds its own null distribution by shuffling the label
          **at the unit where it is exchangeable**, then asks where the real value falls. No bell-curve
          assumption; the only thing you must get right is *what to shuffle*.

        Sex is a property of the **cage** — all three mice in a cage are the same sex, and it never
        changes. So a "does behavior differ by sex?" question is really a comparison of **7 male cages
        versus 7 female cages**, no matter how many events we recorded. We will test the same sex effect
        two ways and watch the answer change.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1.1 · The naive event-level test

        `heading_alignment` measures whether the two interacting mice point the same way (+1 aligned,
        −1 opposed). Let us do the obvious thing: pool every event, split by the sex of its cage, and
        run a **Mann–Whitney U test** (`scipy.stats.mannwhitneyu` — asks whether two groups differ in
        typical value, no bell-curve assumption). We plot it two ways so you can compare the chart
        types too.
        """
    )
    return


@app.cell
def _(HI, X, cu, mo, sex):
    _violin = cu.violin_points_fig(
        X[:, HI], sex, group_order=["M", "F"],
        colors={"M": cu.RANK_HEX[2], "F": "#e45756"},
        ylabel="heading_alignment", xlabel="cage sex",
        title="heading_alignment by sex — every event (violin)")
    _ecdf = cu.ecdf_fig(
        X[:, HI], sex, group_order=["M", "F"],
        colors={"M": cu.RANK_HEX[2], "F": "#e45756"},
        xlabel="heading_alignment", title="the same comparison as ECDFs")
    mo.vstack([
        mo.md("The violin (left) and the cumulative curves (right) tell the same story: male events sit "
              "a little lower (more opposed headings). Note already that the ECDF makes the *consistent* "
              "shift obvious while the violin's fat shapes almost hide it — hold that thought for "
              "Section 2."),
        mo.hstack([_violin, _ecdf], widths=[1, 1])])
    return


@app.cell(hide_code=True)
def _(HI, X, cu, mo, np, sex):
    from scipy.stats import mannwhitneyu as _mwu
    _p = float(_mwu(X[sex == "M", HI], X[sex == "F", HI])[1])
    _mM, _mF = float(np.median(X[sex == "M", HI])), float(np.median(X[sex == "F", HI]))
    mo.md(
        f"""
        **Event-level Mann–Whitney U: p = {cu.fmt_p(_p)}** (median M = {_mM:.3f}, F = {_mF:.3f}). That
        is a p-value with eight zeros after the decimal point. Taken at face value it is overwhelming
        proof of a sex difference. It is also *dishonest*, because it counted 2,499 non-independent
        events as 2,499 independent facts. The test thinks it has 2,499 chances to be fooled by noise;
        it really has 14. Let us give it the honest count.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1.2 · Collapse to the honest unit — one point per cage

        Reduce each of the 14 cages to a single number, its **mean `heading_alignment`**, and color by
        sex. Now there are 14 points, not 2,499. If sex genuinely drives heading, the 7 male cages
        should sit consistently below the 7 female cages — not because one or two extreme cages drag an
        average, but as a systematic 7-versus-7 separation.
        """
    )
    return


@app.cell
def _(HI, X, cage, cu, mo, np, sex, ucage):
    _cm = np.array([np.nanmean(X[cage == c, HI]) for c in ucage])   # one mean per cage
    _cs = np.array([sex[cage == c][0] for c in ucage])             # one sex per cage
    _fig = cu.strip_points_fig(
        _cm, _cs, group_order=["M", "F"],
        colors={"M": cu.RANK_HEX[2], "F": "#e45756"}, jitter=0.12, point_size=14,
        hover=[f"cage {c}" for c in ucage], show_mean=True,
        ylabel="cage-mean heading_alignment", xlabel="cage sex",
        title="14 cohort-unique cages — the honest unit of analysis")
    mo.vstack([_fig,
               mo.md("The male cages *do* sit lower on the whole. But with only 7 versus 7, is that "
                     "separation bigger than a random 7/7 relabeling of the same 14 numbers would "
                     "give? That is exactly the question a permutation test answers — and it is the "
                     "only honest way to put a p-value on 14 points.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1.3 · The honest test — permute sex across the 14 cages

        The exchangeable unit is the cage, so we build the null by **shuffling the 7 M / 7 F labels
        across the 14 cage means**, thousands of times, each time recomputing the gap between the
        male-cage mean and the female-cage mean. The observed gap's rank in that null cloud is the
        p-value. Compare this to the event-level number.
        """
    )
    return


@app.cell
def _(cage, np, sex):
    # One reusable cage-level permutation test. Collapse events to 14 cage means, then shuffle the
    # SEX labels across cages (the exchangeable unit). Returns (observed_gap, p_emp, null_array).
    def cage_perm(values, n=20000, seed=0):
        rng = np.random.RandomState(seed)
        ucg = np.unique(cage)
        cm = np.array([np.nanmean(values[cage == c]) for c in ucg])   # 14 cage means
        cs = np.array([sex[cage == c][0] for c in ucg])               # 14 sex labels
        gap = lambda lab: abs(np.nanmean(cm[lab == "M"]) - np.nanmean(cm[lab == "F"]))
        obs = gap(cs)
        null = np.array([gap(rng.permutation(cs)) for _ in range(n)])
        p = (np.sum(null >= obs - 1e-12) + 1) / (n + 1)
        return float(obs), float(p), null
    return (cage_perm,)


@app.cell
def _(HI, X, cage_perm, cu, go, mo):
    _obs, _p, _null = cage_perm(X[:, HI])
    _fig = go.Figure()
    _fig.add_histogram(x=_null, nbinsx=34, marker_color="#c7c7c7", name="cage-shuffled null")
    _fig.add_vline(x=_obs, line=dict(color="#e45756", width=3),
                   annotation_text="observed", annotation_position="top")
    cu.apply_house_style(_fig, title=f"heading_alignment · cage-level null — observed gap is in the tail "
                                     f"(p = {cu.fmt_p(_p)}, SURVIVES)", legend=None)
    _fig.update_xaxes(title="|male cage-mean − female cage-mean|", showgrid=False)
    mo.vstack([_fig,
               mo.md(f"**Cage-level p = {cu.fmt_p(_p)}.** Still significant — but now honestly so, and "
                     f"about seven orders of magnitude larger than the event-level p. This effect is "
                     f"real: it respects the 14-cage design and it survives. It also replicates "
                     f"independently in both cohorts (we check that in the exercise). Keep this "
                     f"result; it is our **positive control**.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1.4 · The negative control — the same procedure, opposite verdict

        A method that only ever says "significant" teaches nothing. So we run the identical machinery on
        a feature that *should* make us suspicious: `appr_body_len`, the approacher's body length. Males
        are larger, so at the event level the sex difference is astronomically significant. But body
        size is a fixed property of the same 14 cages — the perfect setup for pseudoreplication to
        manufacture certainty out of nothing new.
        """
    )
    return


@app.cell
def _(BLI, X, cage, cage_perm, cu, go, mo, np, sex):
    from scipy.stats import mannwhitneyu as _mwu
    _ep = float(_mwu(X[sex == "M", BLI], X[sex == "F", BLI])[1])
    _obs, _p, _null = cage_perm(X[:, BLI])
    # left: the 14 cage means; right: the permutation null
    _cm = np.array([np.nanmean(X[cage == c, BLI]) for c in np.unique(cage)])
    _cs = np.array([sex[cage == c][0] for c in np.unique(cage)])
    _strip = cu.strip_points_fig(
        _cm, _cs, group_order=["M", "F"], colors={"M": cu.RANK_HEX[2], "F": "#e45756"},
        jitter=0.12, point_size=14, ylabel="cage-mean appr_body_len", xlabel="cage sex",
        title="body length per cage — M and F cages overlap")
    _hist = go.Figure()
    _hist.add_histogram(x=_null, nbinsx=34, marker_color="#c7c7c7")
    _hist.add_vline(x=_obs, line=dict(color="#e45756", width=3),
                    annotation_text="observed", annotation_position="top")
    cu.apply_house_style(_hist, title=f"cage-level null (p = {cu.fmt_p(_p)}, does NOT survive)",
                         legend=None)
    _hist.update_xaxes(title="|M − F| cage-mean gap", showgrid=False)
    mo.vstack([
        mo.md(f"**Event-level p = {cu.fmt_p(_ep)}** (looks overwhelming — a *smaller* p than heading "
              f"alignment). But the 14 cage means overlap heavily, and the observed gap lands in the "
              f"*near tail* of the cage-shuffled null (about the 92nd percentile) without clearing "
              f"0.05: **cage-level p = {cu.fmt_p(_p)}** — a borderline miss, unlike `heading_alignment`, "
              f"whose observed gap sat far out at the 99th percentile."),
        mo.hstack([_strip, _hist], widths=[1, 1])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1.5 · The two controls, side by side — the whole lesson in one table

        | Feature | Event-level p | Cage-level p | Verdict |
        |---|---|---|---|
        | `heading_alignment` (positive control) | ~6.5e-9 | **~0.009** | **survives** — a real, replicated sex effect |
        | `appr_body_len` (negative control) | ~5.4e-22 | ~0.078 | does **not** survive — pseudoreplication |

        Read the table twice. First: **both features look wildly significant at the event level**, and
        the one that fails (body size, p ≈ 1e-22) actually has the *smaller* event-level p. The
        event-level p-value carries almost no information about which effect is real. Second: the honest
        unit tells them apart cleanly. Whether the pattern is consistent across the 14 independent cages
        is what matters, and only the cage-level test can see it.

        A reassuring aside that keeps us honest in the other direction: aggression *rate* by sex is
        **null** at every level (event χ² p ≈ 0.60). We are not claiming aggression is a male behavior;
        we are claiming males and females orient differently, and only that.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Flip the unit yourself, and watch the verdict change

        This is the interactive heart of the section. Pick a **feature** and the **unit of analysis**.
        At the *event* level we run a Mann–Whitney on all 2,499 events; at the *cage* level we run the
        cage-permutation test on the 14 cage means. Nothing about the data changes — only which unit we
        declare independent. Watch the positive control (`heading_alignment`) survive both while the
        negative control (`appr_body_len`) is "significant" at the event level and ordinary at the cage
        level. The lesson lives in that flip.
        """
    )
    return


@app.cell
def _(mo):
    feat_sel = mo.ui.dropdown(
        options={"heading_alignment (positive control)": "heading_alignment",
                 "appr_body_len (negative control)": "appr_body_len",
                 "bystander_dist_mean": "bystander_dist_mean"},
        value="appr_body_len (negative control)", label="feature")
    unit_sel = mo.ui.dropdown(options=["event", "cage"], value="event", label="unit of analysis")
    return feat_sel, unit_sel


@app.cell
def _(X, cage_perm, cu, feat_sel, fn, mo, np, sex, unit_sel):
    from scipy.stats import mannwhitneyu as _mwu
    _col = fn.index(feat_sel.value)
    if unit_sel.value == "event":
        _p = float(_mwu(X[sex == "M", _col], X[sex == "F", _col])[1])
        _n = f"{int((sex == 'M').sum())} + {int((sex == 'F').sum())} events"
        _how = "Mann–Whitney U on every event (treats each event as independent)"
    else:
        _p = cage_perm(X[:, _col])[1]
        _n = "7 + 7 cages"
        _how = "cage-level permutation (shuffles sex across the 14 cage means)"
    _sig = _p < 0.05
    _bg = "#fdecec" if (unit_sel.value == "event") else ("#e7f6ec" if _sig else "#eef2f7")
    _verdict = ("significant — but is the unit honest?" if unit_sel.value == "event"
                else ("survives the honest test" if _sig else "does NOT survive — pseudoreplication"))
    mo.vstack([
        mo.hstack([feat_sel, unit_sel], justify="start"),
        mo.md(f"<div style='border:1px solid #ccc;border-radius:8px;padding:12px 16px;background:{_bg};"
              f"font-size:1.05em'><b>{feat_sel.value}</b> · <b>{unit_sel.value}</b> level "
              f"({_n})<br>{_how}<br><b>p = {cu.fmt_p(_p)}</b> &nbsp; → &nbsp; {_verdict}</div>")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1.6 · When the event level *is* honest — the paired within-cage design

        Sex is fixed within a cage, which is what made it dangerous. **Food-deprivation phase**
        (`pre`/`dep`/`post`) is different: every cage was recorded in all three phases, so phase *varies
        within* a cage. That breaks the confound — but the cleanest test is still to compare each cage
        **to itself**, with a **paired** test (`scipy.stats.wilcoxon` on the 14 pre-vs-dep differences).

        The readout is `bystander_dist_mean`: how far the third mouse sits from the interacting pair.
        A paired plot draws one line per cage from its `pre` value to its `dep` value — green if the
        cage's bystander moved *farther* under deprivation, red if closer.
        """
    )
    return


@app.cell
def _(BDI, X, cage, cond, cu, mo, np, ucage):
    from scipy.stats import mannwhitneyu as _mwu, wilcoxon as _wil
    _pre = np.array([np.nanmean(X[(cage == c) & (cond == "pre"), BDI]) for c in ucage])
    _dep = np.array([np.nanmean(X[(cage == c) & (cond == "dep"), BDI]) for c in ucage])
    _ep = float(_mwu(X[cond == "dep", BDI], X[cond == "pre", BDI])[1])
    _wp = float(_wil(_dep, _pre)[1])
    _fig = cu.dumbbell_fig(_pre, _dep, labels=[f"cage {c}" for c in ucage],
                           before_name="pre", after_name="dep", sort_by="after",
                           title="each cage vs itself: bystander_dist_mean, pre → dep",
                           xlabel="cage-mean bystander_dist_mean (px)")
    mo.vstack([
        mo.md(f"**Event-level Mann–Whitney p = {cu.fmt_p(_ep)}**; the honest **paired cage-level "
              f"Wilcoxon p = {cu.fmt_p(_wp)}**, mean shift **+{np.mean(_dep - _pre):.0f} px**. Almost "
              f"every cage moves the same direction — the deprivation effect is real at the paired unit "
              f"and replicates in both cohorts. Because phase varies *within* a cage, here the event "
              f"level was not automatically pseudoreplicated; the paired test simply makes it "
              f"airtight."),
        _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1.7 · Exercise — write the cage-level permutation test yourself

        **Python skill practiced:** *writing a permutation-test loop* — the single most transferable
        statistical tool in this course. You collapse events to cages, compute an observed statistic,
        and build a null by shuffling labels **at the cage level**.

        **The scientific point of the one blanked line.** If you shuffle sex across *events*, you
        destroy nothing — every event in a male cage is still male — and you get back the
        pseudoreplicated, wildly-significant event-level answer. Shuffling across the **14 cages** is
        what makes the test honest.

        Replace `____` with `rng.permutation(cage_sex)`. Then run it: the self-check reports the
        positive control (`heading_alignment`, should **survive**) and the negative control
        (`appr_body_len`, should **fail**), plus the replication check within each cohort.
        """
    )
    return


@app.cell
def _(cage, np, sex):
    def cage_perm_p(values, cage_ids, sex_ids, n=20000, seed=0):
        rng = np.random.RandomState(seed)
        ucg = np.unique(cage_ids)
        # Collapse each cage to ONE mean value and ONE sex label — the honest 14-unit view.
        cage_mean = np.array([np.nanmean(values[cage_ids == cg]) for cg in ucg])   # (14,)
        cage_sex  = np.array([sex_ids[cage_ids == cg][0] for cg in ucg])           # (14,)
        gap = lambda lab: abs(np.nanmean(cage_mean[lab == "M"]) - np.nanmean(cage_mean[lab == "F"]))
        obs = gap(cage_sex)
        hits = 0
        for _ in range(n):
            # -------------------- FILL IN THE BLANK --------------------
            # Build ONE random relabeling of the 14 cages. Shuffle the CAGE sex labels (`cage_sex`),
            # NOT the per-event sex — shuffling events leaves every cage's sex unchanged and returns
            # the dishonest event-level answer. rng.permutation(x) returns a shuffled COPY of x,
            # keeping 7 M / 7 F fixed. Replace ____ with:  rng.permutation(cage_sex)
            perm = ____                                  # <-- EDIT THIS LINE
            # -----------------------------------------------------------
            if gap(perm) >= obs - 1e-12:
                hits += 1
        return float(obs), float((hits + 1) / (n + 1))
    return (cage_perm_p,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Reveal solution": mo.md(
            r"""
            ```python
            perm = rng.permutation(cage_sex)
            ```
            `heading_alignment` then gives p ≈ **0.009** (survives) and `appr_body_len` gives ≈ **0.078**
            (fails). Writing `rng.permutation(sex_ids)` instead (event level) sends both back near zero —
            the exact pseudoreplication trap this section is about.
            """)
    })
    return


@app.cell(hide_code=True)
def _(BLI, HI, X, cage, cage_perm_p, cohort, cu, mo, np, sex):
    from scipy.stats import mannwhitneyu as _mwu
    try:
        _h_obs, _h_p = cage_perm_p(X[:, HI], cage, sex)
        _b_obs, _b_p = cage_perm_p(X[:, BLI], cage, sex)
        _c1 = 0.003 <= _h_p <= 0.02
        _c2 = 0.04 <= _b_p <= 0.12
        # replication: event-level MWU within each cohort for heading
        _rep = []
        for _co in np.unique(cohort):
            _m = cohort == _co
            _rep.append((_co, float(_mwu(X[_m & (sex == "M"), HI], X[_m & (sex == "F"), HI])[1])))
        _ok = _c1 and _c2
    except Exception:
        _ok = False; _h_p = _b_p = float("nan"); _c1 = _c2 = False; _rep = []
    _bg = "#e7f6ec" if _ok else "#fdecec"
    _icon = "PASS" if _ok else "NOT YET"
    _reptxt = "; ".join(f"cohort {c}: p = {cu.fmt_p(p)}" for c, p in _rep)
    _msg = (f"heading_alignment cage-level p = {cu.fmt_p(_h_p)} (survives) · appr_body_len cage-level "
            f"p = {cu.fmt_p(_b_p)} (fails). The positive control replicates in both cohorts "
            f"({_reptxt})." if _ok else
            f"heading p = {cu.fmt_p(_h_p)}, body p = {cu.fmt_p(_b_p)}. If both came back near 0 you are "
            f"still shuffling events — set perm = rng.permutation(cage_sex).")
    mo.md(f"<div style='border:1px solid #ccc;border-radius:8px;padding:10px 14px;background:{_bg}'>"
          f"<b>{_icon}</b> &nbsp; {_msg}</div>")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        > **Section 1 answer.** The unit of analysis is not a technicality — it *is* the sample size.
        > `heading_alignment` survives the 14-cage test (a real, replicated sex effect); `appr_body_len`,
        > which had a smaller event-level p, does not. The event-level p-value could not tell them
        > apart. **Next:** given the honest unit, which *test* do we run on it?
        """
    )
    return


# ============================================================================================
# ==============  SECTION 2 — CHOOSING THE RIGHT TEST  ========================================
# ============================================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # 2 · Choosing the right test

        ## Why this matters

        Once you know the unit, you still have to pick a test — and the test encodes an *assumption*
        about what "different" means. A test that asks the wrong question can miss a real effect or
        invent a fake one. Three choices come up constantly, and we make each on real course data.

        ## 2.1 · Direction vs magnitude — sign test vs Wilcoxon signed-rank

        Back in NB01 we asked whether "slot 0 = the approacher" is a real label by measuring, for each
        event, the approacher's and approachee's speed in the 50 frames **before contact**. Take the
        per-event difference `d = approacher_speed − approachee_speed`. Two paired tests read `d`
        differently:

        - **Sign test** (`scipy.stats.binomtest` on the count of `d > 0`) — asks only about
          **direction**: in what fraction of events is the approacher faster? It throws away *how much*
          faster. It is the most assumption-free paired test there is.
        - **Wilcoxon signed-rank** (`scipy.stats.wilcoxon(d)`) — asks about **direction weighted by
          magnitude**: it ranks the absolute differences, so a few large gaps count more than many tiny
          ones. More powerful when magnitudes are meaningful, but it assumes the differences are
          symmetric.
        """
    )
    return


@app.cell
def _(cu, ev, np):
    # Pre-contact TTI speed for each slot (a ~2500-event loop, well under 1 s). Reused from NB01 Ex3.
    def _mean_pre_speed(k, cr, m):
        _t0 = max(0, cr - 50)
        _tti = k[_t0:cr, m, cu.TTI, :]
        if len(_tti) < 2:
            return np.nan
        _step = np.linalg.norm(np.diff(_tti, axis=0), axis=1)
        return np.nanmean(_step) if np.isfinite(_step).any() else np.nan
    _kp = ev["kp"]; _cr = ev["contact_rel"].astype(int)
    s0_pre = np.array([_mean_pre_speed(_kp[i], _cr[i], 0) for i in range(len(_kp))])
    s1_pre = np.array([_mean_pre_speed(_kp[i], _cr[i], 1) for i in range(len(_kp))])
    _v = np.isfinite(s0_pre) & np.isfinite(s1_pre)
    pre_diff = (s0_pre - s1_pre)[_v]           # approacher − approachee, one per valid event
    return (pre_diff,)


@app.cell
def _(cu, mo, np, pre_diff):
    from scipy.stats import binomtest as _bt, wilcoxon as _wil
    _n = len(pre_diff); _npos = int((pre_diff > 0).sum())
    _sign_p = float(_bt(_npos, _n, 0.5).pvalue)
    _wil_p = float(_wil(pre_diff)[1])
    _fig = cu.paired_diff_fig(pre_diff, xlabel="approacher − approachee pre-contact speed (px/frame)",
                              kind="hist",
                              title="per-event speed difference — the distribution both tests read")
    mo.vstack([
        mo.md(f"Approacher faster in **{_npos}/{_n} = {_npos/_n:.1%}** of events. **Sign test "
              f"p = {cu.fmt_p(_sign_p)}** (direction only); **Wilcoxon signed-rank "
              f"p = {cu.fmt_p(_wil_p)}** (direction + magnitude). Both are tiny, but the Wilcoxon p is "
              f"smaller because the approacher is not just *more often* faster — when it is faster, it "
              f"is faster by a larger margin (the right tail of this histogram is heavier than the "
              f"left). The two tests agree here; they disagree exactly when a majority lean is tiny in "
              f"size, or a minority lean is huge. Choosing between them is choosing what 'different' "
              f"means."),
        _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2.2 · When the chart is the test's twin — violin vs ECDF vs strip

        A test choice and a chart choice are the same decision in disguise: both encode what you think
        the signal looks like. The sex effect on `heading_alignment` is a **small, consistent shift**
        (Cohen's d ≈ 0.23). Watch three charts of the identical data and notice which one lets you
        *see* the effect the cage-level test confirmed.

        - **KDE violin** — smooths each group into a filled shape. For a small shift the two fat shapes
          look nearly identical; worse, the smoothing can bulge past the real data range. It flatters
          symmetric, well-separated groups and hides small shifts.
        - **ECDF** (cumulative curve) — plots the fraction of events below each value. A consistent
          shift shows up as one curve lying cleanly to the left of the other at *every* height. It is
          the honest default for a small-shift or heavy-tail comparison.
        - **Cage strip** — 14 points, the honest unit again, so you see the actual evidence.
        """
    )
    return


@app.cell
def _(HI, X, cage, cu, mo, np, sex, ucage):
    _violin = cu.violin_points_fig(
        X[:, HI], sex, group_order=["M", "F"], colors={"M": cu.RANK_HEX[2], "F": "#e45756"},
        points=False, ylabel="heading_alignment", xlabel="sex",
        title="violin — d≈0.23 nearly invisible")
    _ecdf = cu.ecdf_fig(
        X[:, HI], sex, group_order=["M", "F"], colors={"M": cu.RANK_HEX[2], "F": "#e45756"},
        xlabel="heading_alignment", title="ECDF — the consistent shift is legible")
    _cm = np.array([np.nanmean(X[cage == c, HI]) for c in ucage])
    _cs = np.array([sex[cage == c][0] for c in ucage])
    _strip = cu.strip_points_fig(
        _cm, _cs, group_order=["M", "F"], colors={"M": cu.RANK_HEX[2], "F": "#e45756"},
        jitter=0.12, point_size=12, ylabel="cage-mean", xlabel="sex",
        title="14 cages — the actual evidence")
    mo.vstack([mo.hstack([_violin, _ecdf, _strip], widths=[1, 1, 1]),
               mo.md("*Same numbers, three encodings. The violin is the least informative and the most "
                     "seductive — it looks like a rigorous distribution plot while hiding the very "
                     "effect that survives testing. Reach for the ECDF or a points plot when the shift "
                     "is small or the tails are heavy; save violins for genuinely well-separated, "
                     "roughly symmetric groups.*")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        > **Section 2 answer.** The test encodes the question. Sign vs Wilcoxon is direction vs
        > direction-and-magnitude; violin vs ECDF is "do the shapes look different" vs "is one
        > distribution consistently shifted." Pick the one whose assumption matches your effect — and
        > let the chart show the same thing the test claims. **Next:** what happens when you run *many*
        > tests at once?
        """
    )
    return


# ============================================================================================
# ==============  SECTION 3 — MULTIPLE COMPARISONS  ==========================================
# ============================================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # 3 · Multiple comparisons

        ## Why this matters

        A p-value of 0.05 means "if there were no effect, I'd see something this extreme 5% of the
        time." Run one test and that is a reasonable guard. Run **19 tests** — one per feature — and you
        expect about `19 × 0.05 ≈ 1` false alarm even if *nothing* is real. Scan enough and you will
        always find "significance." This is how honest people publish flukes.

        ## Definitions

        - **Family-wise error rate (FWER)** — the chance of *at least one* false positive across a whole
          family of tests. It grows fast with the number of tests.
        - **Bonferroni correction** — the strictest fix: to keep FWER at 0.05 across `m` tests, require
          each individual p below `0.05 / m`. Simple, conservative.
        - **Benjamini–Hochberg (BH)** — controls the *false-discovery rate* (the expected fraction of
          your "discoveries" that are false) instead of the FWER; less strict, more powerful.

        ## 3.1 · Scan all 19 features for a sex effect

        We test each of the 19 features for a sex difference, once at the (dishonest) **event level** and
        once at the honest **cage level**, and count how many clear α = 0.05 — before and after
        correction.
        """
    )
    return


@app.cell
def _(X, cage, cage_perm, cu, fn, go, mo, np, sex):
    from scipy.stats import mannwhitneyu as _mwu
    _ev_p = np.array([float(_mwu(X[sex == "M", i], X[sex == "F", i])[1]) for i in range(len(fn))])
    _cg_p = np.array([cage_perm(X[:, i], n=5000, seed=1)[1] for i in range(len(fn))])
    _m = len(fn)
    _bonf = 0.05 / _m
    _order = np.argsort(_cg_p)
    _fig = go.Figure()
    _fig.add_bar(x=[fn[i] for i in _order], y=-np.log10(_ev_p[_order]), name="event-level",
                 marker_color="#c7c7c7")
    _fig.add_bar(x=[fn[i] for i in _order], y=-np.log10(_cg_p[_order]), name="cage-level",
                 marker_color="#4c78a8")
    _fig.add_hline(y=-np.log10(0.05), line=dict(color="#333", dash="dash"),
                   annotation_text="α = 0.05")
    _fig.add_hline(y=-np.log10(_bonf), line=dict(color="#e45756", dash="dot"),
                   annotation_text=f"Bonferroni 0.05/{_m}")
    cu.apply_house_style(_fig, title="19 features scanned for a sex effect (−log10 p; taller = smaller p)",
                         legend="below")
    _fig.update_xaxes(showgrid=False, tickangle=-40)
    _fig.update_yaxes(title="−log10 p", showgrid=False)
    _n_ev = int((_ev_p < 0.05).sum()); _n_cg = int((_cg_p < 0.05).sum())
    _n_cg_bonf = int((_cg_p < _bonf).sum())
    # Benjamini–Hochberg (FDR) on the cage-level p-values: sort ascending, find the largest rank k
    # whose p <= (k/m)*0.05, and reject every hypothesis up to that rank. Less strict than Bonferroni.
    _srt = np.sort(_cg_p)
    _bh_thresh = 0.05 * np.arange(1, _m + 1) / _m
    _below = _srt <= _bh_thresh
    _n_cg_bh = int(np.max(np.where(_below)[0]) + 1) if _below.any() else 0
    mo.vstack([_fig,
               mo.md(f"**Event level: {_n_ev} of 19 features are 'significant'** at 0.05 — a scan that "
                     f"would let you claim almost any feature differs by sex. **Cage level: {_n_cg} of "
                     f"19** clear 0.05 uncorrected; after **Bonferroni ({_n_cg_bonf} of 19)** and the "
                     f"less-strict **Benjamini–Hochberg FDR ({_n_cg_bh} of 19)** correction, the count "
                     f"collapses under an honest unit *and* an honest correction. Bonferroni controls the "
                     f"chance of *any* false positive (FWER); BH controls the *fraction* of discoveries "
                     f"that are false (FDR) and so is usually more permissive — here even BH keeps "
                     f"nothing, because the smallest cage-level p (≈0.009) does not clear its own BH "
                     f"threshold of 0.05·1/19 ≈ 0.0026.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3.2 · Then why did we trust `heading_alignment`?

        This is the subtle and important part. In Section 1 `heading_alignment` (cage p ≈ 0.009)
        **survived**; here, as one of 19 scanned features, it does **not** clear the Bonferroni line
        (0.05 / 19 ≈ 0.0026). Both statements are true, and the difference is *not* in the data — it is
        in **how the hypothesis was chosen**.

        - `heading_alignment` was a **pre-specified** hypothesis (mice of different sex orient
          differently). A pre-specified test pays no multiple-comparisons tax: it is one test, judged at
          0.05.
        - The 19-feature scan is **exploratory**: we let the data pick the winner, so we *must* correct
          for the 19 chances we gave ourselves.

        The same p-value means different things depending on whether you went looking for it. The
        honest workflow is to **separate confirmatory from exploratory**: pre-register the handful of
        hypotheses you truly predicted, and treat everything a scan turns up as a lead to be tested on
        *new* data, not a finding.
        """
    )
    return


@app.cell(hide_code=True)
def _(cu, go, mo, np):
    # A pure-null demo of FWER: 19 features vs a RANDOM label, repeated; count "significant" hits.
    _rng = np.random.RandomState(0)
    _trials = 2000; _m = 19
    _hits = np.zeros(_trials, int)
    for _t in range(_trials):
        _p = _rng.rand(_m)                      # 19 independent p-values under a true null are ~Uniform
        _hits[_t] = int((_p < 0.05).sum())
    _any = float((_hits >= 1).mean())
    _fig = go.Figure(go.Histogram(x=_hits, xbins=dict(start=-0.5, end=8.5, size=1),
                                  marker_color="#4c78a8"))
    cu.apply_house_style(_fig, title=f"pure noise, 19 tests each: P(≥1 false 'hit') = {_any:.0%}",
                         legend=None)
    _fig.update_xaxes(title="number of 'significant' features (α=0.05) when NOTHING is real",
                      showgrid=False, dtick=1)
    _fig.update_yaxes(title="count of trials", showgrid=False)
    mo.vstack([
        mo.md("**The mechanism, on pure noise.** Simulate 19 features that are all null (random "
              "p-values), 2,000 times. Even though nothing is real, you get at least one 'significant' "
              "feature most of the time, and often two or three. This is why a scan needs a correction: "
              "the false alarms are not a bug in your data, they are arithmetic."),
        _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        > **Section 3 answer.** Testing 19 features is not 19 independent chances to be right — it is 19
        > chances to be *fooled*. Correct for it (Bonferroni/BH), and above all separate the hypotheses
        > you predicted from the ones a scan handed you. **Next:** even a correctly-tested, real effect
        > can be trivially small — so how big is it?
        """
    )
    return


# ============================================================================================
# ==============  SECTION 4 — EFFECT SIZE vs p-VALUE  ========================================
# ============================================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # 4 · Effect size versus p-value

        ## Why this matters

        A p-value answers "is the effect distinguishable from zero?" It does **not** answer "is the
        effect big enough to care about?" With enough data, a difference far too small to matter still
        gets an impressively tiny p. Significance is not importance.

        ## Definitions

        - **Cohen's d** — a standardized effect size: the difference in group means divided by the
          pooled standard deviation. Rough conventions: d ≈ 0.2 small, 0.5 medium, 0.8 large. Unlike a
          p-value, d does not grow just because you collected more data.
        - **Confidence interval on d** — a range of plausible effect sizes given sampling noise. We get
          it by **bootstrap**: resample the events with replacement many times, recompute d each time,
          and take the middle 95% of those values.

        ## 4.1 · A tiny p can hide a tiny effect

        Take `triangle_area_mean` (the area of the triangle the three mice form) and compare aggression
        vs non-aggression events across all 2,499 events.
        """
    )
    return


@app.cell
def _(TAI, X, cu, mo, np, yagg):
    from scipy.stats import mannwhitneyu as _mwu
    def _cohend(a, b):
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        ps = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        return (np.mean(a) - np.mean(b)) / ps
    _p = float(_mwu(X[yagg == 1, TAI], X[yagg == 0, TAI])[1])
    _d = _cohend(X[yagg == 1, TAI], X[yagg == 0, TAI])
    # bootstrap 95% CI on d
    _rng = np.random.RandomState(0)
    _a = X[yagg == 1, TAI]; _b = X[yagg == 0, TAI]
    _boot = np.array([_cohend(_rng.choice(_a, len(_a)), _rng.choice(_b, len(_b))) for _ in range(2000)])
    _lo, _hi = np.percentile(_boot, [2.5, 97.5])
    _fig = cu.ecdf_fig(X[:, TAI], np.where(yagg == 1, "aggression", "non-agg"),
                       group_order=["non-agg", "aggression"],
                       colors={"non-agg": "#9aa0a6", "aggression": "#d62728"},
                       xlabel="triangle_area_mean", title="triangle_area_mean — aggression vs non")
    mo.vstack([
        mo.md(f"**Mann–Whitney p = {cu.fmt_p(_p)}** — a p-value with ten zeros. **Cohen's "
              f"d = {_d:.2f}** (95% bootstrap CI [{_lo:.2f}, {_hi:.2f}]) — a *small* effect. The two "
              f"ECDF curves separate only modestly; the astronomically small p comes almost entirely "
              f"from n = 2,499, not from the size of the shift (d ≈ {_d:.2f}). A reader who saw only "
              f"'p = 3e-11' would badly overestimate how different aggression looks on this feature."),
        _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4.2 · The whole feature set — p and effect size disagree

        Plot every feature's aggression effect as a point: `−log10 p` on one axis (how significant),
        `|Cohen's d|` on the other (how big). If p and effect size measured the same thing the points
        would fall on a line. They do not.
        """
    )
    return


@app.cell
def _(X, cu, fn, go, mo, np, yagg):
    from scipy.stats import mannwhitneyu as _mwu
    def _cohend(a, b):
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        ps = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        return abs((np.mean(a) - np.mean(b)) / ps)
    _ps = np.array([float(_mwu(X[yagg == 1, i], X[yagg == 0, i])[1]) for i in range(len(fn))])
    _ds = np.array([_cohend(X[yagg == 1, i], X[yagg == 0, i]) for i in range(len(fn))])
    _neglp = np.clip(-np.log10(np.clip(_ps, 1e-300, None)), 0, 320)
    _fig = go.Figure()
    _fig.add_scatter(x=_ds, y=_neglp, mode="markers+text", text=fn, textposition="top center",
                     textfont=dict(size=9), marker=dict(size=10, color="#4c78a8"),
                     hovertext=[f"{f}: p={cu.fmt_p(p)}, |d|={d:.2f}" for f, p, d in zip(fn, _ps, _ds)],
                     hoverinfo="text")
    _fig.add_vline(x=0.2, line=dict(color="#bbb", dash="dot"), annotation_text="d=0.2 (small)")
    cu.apply_house_style(_fig, title="significance vs effect size, aggression — they are not the same axis",
                         legend=None)
    _fig.update_xaxes(title="|Cohen's d| (effect size)", showgrid=False)
    _fig.update_yaxes(title="−log10 p (significance, clipped at 320)", showgrid=False)
    mo.vstack([_fig,
               mo.md("Read the top row: a whole cluster of features has p so small it underflows below "
                     "1e-300 (clipped at 320) — yet they sit at wildly different effect sizes, from "
                     "d ≈ 0.6 to d ≈ 1.0. 'Maximally significant' says nothing about how big the effect "
                     "is. Meanwhile `triangle_area_mean` (from 4.1) sits at small d ≈ 0.17 but still "
                     "clears significance by a wide margin. Report **both** numbers, always: a p-value "
                     "to say the effect is real, an effect size with a CI to say whether it is worth "
                     "caring about.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        > **Section 4 answer.** p-value and effect size are different axes. At n = 2,499 even a d ≈ 0.17
        > difference earns p = 3e-11. Always pair a significance claim with a bootstrap CI on the effect
        > size. **Next:** the subtlest trap — using the same data to choose *and* test a hypothesis.
        """
    )
    return


# ============================================================================================
# ==============  SECTION 5 — CIRCULAR ANALYSIS / DOUBLE-DIPPING  =============================
# ============================================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # 5 · Circular analysis (double-dipping)

        ## Why this matters

        The most dangerous mistakes are the ones that feel like careful work. **Circular analysis** —
        also called *double-dipping* — is using the same data twice: once to **select** or **sort**
        something, and again to **test** it, without acknowledging that the selection already used the
        answer. The test then confirms a pattern the selection *put there*. It can manufacture
        significance out of pure noise, and it looks exactly like a real result.

        ## 5.1 · Selection on noise — the clean demonstration

        Here is the trap in its purest form, on data we *know* is meaningless. Generate a matrix of pure
        random noise: 200 "events" × 400 "features", plus a random binary label with no relationship to
        anything. Then:

        1. **Select** the 10 features that correlate most strongly with the label.
        2. **Test** those same 10 features against the same label.

        Because we picked the features *for* their correlation, the test on the same data will find them
        wildly "significant" — even though every number was noise.
        """
    )
    return


@app.cell
def _(cu, go, mo, np):
    _rng = np.random.RandomState(0)
    _n, _p = 200, 400
    _Xn = _rng.randn(_n, _p)                    # pure noise
    _yn = _rng.randint(0, 2, _n)                # random label, unrelated to _Xn
    from scipy.stats import ttest_ind as _tt
    # step 1: SELECT the 10 features most associated with the label (on ALL the data)
    _tstats = np.array([abs(_tt(_Xn[_yn == 1, j], _Xn[_yn == 0, j])[0]) for j in range(_p)])
    _sel = np.argsort(_tstats)[-10:]
    # step 2a (WRONG): re-test the SAME features on the SAME data
    _p_same = np.array([float(_tt(_Xn[_yn == 1, j], _Xn[_yn == 0, j])[1]) for j in _sel])
    # step 2b (RIGHT): test the selected features on a FRESH noise draw (held-out)
    _Xh = _rng.randn(_n, _p)
    _p_fresh = np.array([float(_tt(_Xh[_yn == 1, j], _Xh[_yn == 0, j])[1]) for j in _sel])
    _fig = go.Figure()
    _fig.add_box(y=_p_same, name="re-test SAME data\n(circular)", marker_color="#e45756",
                 boxpoints="all", pointpos=0, jitter=0.4)
    _fig.add_box(y=_p_fresh, name="test on FRESH noise\n(honest)", marker_color="#4c78a8",
                 boxpoints="all", pointpos=0, jitter=0.4)
    _fig.add_hline(y=0.05, line=dict(color="#333", dash="dash"), annotation_text="α = 0.05")
    cu.apply_house_style(_fig, title="the 10 'best' features of pure noise — tested two ways", legend=None)
    _fig.update_yaxes(title="p-value", showgrid=False)
    mo.vstack([_fig,
               mo.md(f"**Circular (red): every selected feature looks significant** (median "
                     f"p = {np.median(_p_same):.3f}) — on data that is *definitionally* noise. **Honest "
                     f"(blue): tested on a fresh noise draw, the same features are ordinary** (median "
                     f"p = {np.median(_p_fresh):.2f}), scattered across [0,1] as a true null should be. "
                     f"The selection created the effect; only held-out data reveals it was never "
                     f"there.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5.2 · A metric that is circular *by construction* — "sequenceness"

        The noise demo is deliberately obvious. Real circularity hides inside a metric that looks
        principled. A recurring example from neural analysis (Week 2): asking whether a population of
        cells fires in a repeatable **sequence**. The tempting recipe is:

        1. **Sort** the neurons by *when* each first becomes active in a time window.
        2. **Score** the sorted raster by correlating each neuron's row position with its
           first-activation time.

        But step 1 *ordered the rows by exactly the quantity step 2 correlates against.* The correlation
        is ≈ 1 **by construction**, whether or not any sequence exists. To prove it, we build a raster of
        **pure noise** — no sequence at all — sort it, and score it.
        """
    )
    return


@app.cell
def _(cu, go, mo, np):
    from scipy.stats import spearmanr as _sp
    # A raster with NO real sequence: each neuron gets two INDEPENDENT random bursts (one in each
    # half of the window) on a low-noise floor. Burst times are unrelated across neurons and across
    # halves, so there is no reproducible firing order — yet sorting will still manufacture one.
    _rng = np.random.RandomState(2)
    _nn, _T = 40, 1200
    _half = _T // 2
    def _burst(_t0, _amp=9.0, _w=12.0):
        _x = np.arange(_T)
        return _amp * np.exp(-0.5 * ((_x - _t0) / _w) ** 2)
    _noise = _rng.rand(_nn, _T) * 2.0
    _tA = _rng.randint(30, _half - 30, _nn)          # a burst somewhere in the first half
    _tB = _rng.randint(_half + 30, _T - 30, _nn)     # an independent burst in the second half
    for _i in range(_nn):
        _noise[_i] += _burst(_tA[_i]) + _burst(_tB[_i])
    def _seqness(_raster, _thr=5.0):
        _first = np.argmax(_raster > _thr, axis=1)
        _r, _ = _sp(np.arange(_raster.shape[0]), _first)
        return 0.0 if np.isnan(_r) else abs(float(_r))
    _first = np.argmax(_noise > 5.0, axis=1)
    _order = np.argsort(_first)
    _sorted = _noise[_order]
    _q_un = _seqness(_noise); _q_so = _seqness(_sorted)
    _left = go.Figure(go.Heatmap(z=_noise, colorscale="Viridis", zmin=0, zmax=6, showscale=False))
    cu.apply_house_style(_left, title=f"noise, unsorted · sequenceness = {_q_un:.2f}", legend=None,
                         spatial=False)
    _left.update_xaxes(showgrid=False, title="time"); _left.update_yaxes(showgrid=False, title="neuron")
    _right = go.Figure(go.Heatmap(z=_sorted, colorscale="Viridis", zmin=0, zmax=6, showscale=False))
    cu.apply_house_style(_right, title=f"same noise, SORTED · sequenceness = {_q_so:.2f}", legend=None)
    _right.update_xaxes(showgrid=False, title="time"); _right.update_yaxes(showgrid=False, title="neuron")
    mo.vstack([mo.hstack([_left, _right], widths=[1, 1]),
               mo.md(f"The unsorted raster scores **{_q_un:.2f}** — no visible order. After sorting by "
                     f"first-activation time, the same noise scores **{_q_so:.2f}** and the bursts snap "
                     f"onto a clean diagonal (climbing left-to-right through the first half) — a "
                     f"'sequence' conjured from data with no reproducible order at all. Any analysis "
                     f"reporting the sorted number as evidence of sequential firing is reporting an "
                     f"artifact of its own sorting step.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5.3 · The fix — score on data the selection never saw (split-half CV)

        The cure for every version of double-dipping is the same: **separate the data you select/sort on
        from the data you test on.** For the sequence question this is **split-half cross-validation**:

        1. Learn the neuron order on the **first half** of the window only.
        2. *Without re-sorting*, ask whether the **held-out second half** is still diagonal under that
           learned order.

        If the sequence is real, an order learned on one half predicts the other, beating a random-order
        null. If it was an artifact of sorting, the held-out score collapses to the random cloud. We run
        it on our noise raster — where the honest answer is "no sequence."
        """
    )
    return


@app.cell
def _(cu, mo, np):
    from scipy.stats import spearmanr as _sp
    # Rebuild the exact same no-sequence raster (two independent bursts per neuron, one per half).
    _rng = np.random.RandomState(2)
    _nn, _T = 40, 1200
    _half = _T // 2
    def _burst(_t0, _amp=9.0, _w=12.0):
        _x = np.arange(_T)
        return _amp * np.exp(-0.5 * ((_x - _t0) / _w) ** 2)
    _noise = _rng.rand(_nn, _T) * 2.0
    _tA = _rng.randint(30, _half - 30, _nn)
    _tB = _rng.randint(_half + 30, _T - 30, _nn)
    for _i in range(_nn):
        _noise[_i] += _burst(_tA[_i]) + _burst(_tB[_i])
    _A, _B = _noise[:, :_half], _noise[:, _half:]
    def _heldout(_row_order):
        _b = _B[_row_order]
        _active = _b.max(axis=1) > 5.0
        if _active.sum() < 3:
            return 0.0
        _first = np.argmax(_b > 5.0, axis=1)[_active]
        _r, _ = _sp(np.arange(int(_active.sum())), _first)
        return 0.0 if np.isnan(_r) else abs(float(_r))
    _learned = np.argsort(np.argmax(_A > 5.0, axis=1))       # order learned on first half only
    _cv_learned = _heldout(_learned)
    _rng2 = np.random.RandomState(1)
    _cv_null = np.array([_heldout(_rng2.permutation(_nn)) for _ in range(500)])
    _p = (np.sum(_cv_null >= _cv_learned) + 1) / (len(_cv_null) + 1)
    _fig = cu.strip_points_fig(_cv_null, np.array(["random order"] * len(_cv_null)),
                               colors={"random order": "#bab0ac"}, show_mean=True,
                               ylabel="held-out sequenceness",
                               title="split-half CV on noise — learned order does NOT beat random")
    _fig.add_scatter(x=[-0.35, 0.35], y=[_cv_learned, _cv_learned], mode="lines",
                     line=dict(color="#e45756", width=3), name="learned order")
    mo.vstack([_fig,
               mo.md(f"On noise, the order learned from the first half scores **{_cv_learned:.2f}** on "
                     f"the held-out half — right inside the random cloud (median "
                     f"{np.median(_cv_null):.2f}, empirical **p = {cu.fmt_p(_p)}**). The split-half test "
                     f"*could* have come out positive; it did not, because there was nothing to find. "
                     f"On the real striatal recording (NB08) the learned order does beat the null "
                     f"(≈0.5 vs ≈0.16) — modest but honest evidence, because it survived a test that "
                     f"could have failed. That is the whole difference between the sorted number and the "
                     f"held-out number.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Other double-dips you will meet in this course

        The same shape recurs, and now you can name it:

        - **Select-then-test the same cells.** In NB08 we flag "social neurons" by how differently they
          fire in social vs non-social frames, then ask whether those neurons fire differently in social
          frames. Of course they do — we chose them for it. The honest version selects on some frames
          and tests on others.
        - **Split-then-plot-the-split-variable.** Split sources into "active" and "silent" by
          `peak z > 5`, then plot `peak z` for the two groups — and marvel at the gap *at the cut you
          just made*. The gap is guaranteed; it is a picture of the threshold, not a finding. Plot the
          *outcome* instead (e.g. an aggression rate with a confidence interval), or show the whole
          continuum with the threshold drawn on it.

        > **Section 5 answer.** If the same data chose the hypothesis and tested it, the test is not a
        > test. Hold out the data you select or sort on. **Next:** the highest-stakes version of exactly
        > this idea — cross-validation, and how it silently leaks when your data has structure in time.
        """
    )
    return


# ============================================================================================
# ==============  SECTION 6 — CROSS-VALIDATION LEAKAGE (centerpiece)  =========================
# ============================================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # 6 · Cross-validation leakage — the centerpiece

        ## Why this matters

        Cross-validation (CV) is supposed to be the honest way to estimate how a model does on data it
        has never seen: split into folds, train on some, test on the rest, repeat. It is the single most
        trusted number in a modeling paper. And it is quietly, spectacularly easy to break — by letting
        information **leak** from the test fold into the training fold. When it breaks, it does not throw
        an error; it just reports a number that is too good.

        ## Definitions

        - **Temporal autocorrelation** — nearby-in-time samples are similar. A calcium trace changes
          slowly, so frame *t* and frame *t + 1* are almost the same measurement. Behavior states are
          sticky (NB06), so consecutive frames share a state.
        - **Random K-fold (shuffle) CV** — split the rows into folds *at random*. If samples are
          autocorrelated, a frame's near-twin at *t + 1* lands in the training fold while *t* is in the
          test fold. The model has effectively seen the test data. This is **leakage**.
        - **Blocked / grouped CV** — cut the timeline into a few **contiguous blocks** and hold out
          whole blocks. Neighboring frames stay together, so the test block is genuinely novel.
        - **Contiguous (forward-in-time) CV** — train on the first part of the recording, test on the
          last part. The strictest, most honest split.

        ## The demonstration

        We use the **real neural social decoder** from Week 2 (NB08): a logistic model that reads
        `is_social` off ~218 simultaneously-recorded neurons, frame by frame, in one imaging session.
        Calcium is heavily autocorrelated, so this is the perfect victim. We decode the *same data with
        the same model* under all three CV schemes and watch the number move.

        *(This cell loads a ~250 MB calcium file and fits the decoder several times, so it is behind a
        button.)*
        """
    )
    return


@app.cell
def _(ROOT):
    # Load the neural engine the same way the Week-2 notebooks do (course/ already on sys.path).
    import neural_utils as nu
    return (nu,)


@app.cell
def _(mo):
    leak_btn = mo.ui.run_button(label="▶ Load the neural decoder and run the CV-scheme comparison")
    return (leak_btn,)


@app.cell
def _(leak_btn, mo, np, nu):
    # Build the decoder's data ONCE (behind the button): one imaging session, put calcium on the
    # behavior clock, crop the 3-min post-entry window. X = (T, ~218) population vectors, y = is_social.
    if not leak_btn.value:
        sess_X = None; sess_y = None; sess_info = "not loaded"
    else:
        _d = nu.load_si()
        _beh, _img, _ent = _d["behavior"], _d["imaging"], _d["entrances"]
        _s = 6                                            # session 6 (7-day isolation), a clear signal
        _iss = _beh[_s]["is_social_sender"].astype(bool)
        _r = nu.zscore(nu.interp_resample(_img[_s], len(_iss), axis=0), axis=0)
        _e = int(_ent["Int_Entry"].iloc[_s]); _t0, _t1 = _e, int(_e + 3 * 60 * nu.BEHAVIOR_FPS)
        sess_X = _r[_t0:_t1]
        sess_y = _iss[_t0:_t1].astype(int)
        sess_info = f"session {_s}: {sess_X.shape[0]} frames × {sess_X.shape[1]} neurons, " \
                    f"{sess_y.mean():.0%} social"
    return sess_X, sess_info, sess_y


@app.cell
def _(cu, go, leak_btn, mo, np, sess_X, sess_info, sess_y):
    if not leak_btn.value or sess_X is None:
        _out = mo.md("*Click the button above to run the CV-leakage comparison (takes ~30 s).*")
    else:
        from sklearn.pipeline import make_pipeline as _mkp
        from sklearn.preprocessing import StandardScaler as _SS
        from sklearn.linear_model import LogisticRegression as _LR
        # A fast, converging decoder (liblinear) so the cell finishes quickly.
        _clf = lambda: _mkp(_SS(), _LR(max_iter=200, solver="liblinear", class_weight="balanced"))
        _order = np.arange(len(sess_y))              # true temporal order of the frames
        _schemes = [("shuffle\n(random K-fold)", "shuffle", "#e45756"),
                    ("blocked\n(contiguous blocks)", "blocked", "#4c78a8"),
                    ("contiguous\n(forward in time)", "contiguous", "#2ca02c")]
        _vals, _groups, _cols, _summ = [], [], {}, []
        for _lab, _sch, _c in _schemes:
            _a = cu.blocked_cv_auroc(sess_X, sess_y, order=_order, scheme=_sch, clf=_clf())
            _vals.extend(np.asarray(_a)[np.isfinite(_a)].tolist())
            _groups.extend([_lab] * int(np.isfinite(_a).sum()))
            _cols[_lab] = _c
            _summ.append(f"{_lab.splitlines()[0]}: {np.nanmean(_a):.3f}")
        _order_g = [s[0] for s in _schemes]
        _fig = cu.strip_points_fig(np.array(_vals), np.array(_groups), group_order=_order_g,
                                   colors=_cols, point_size=12, show_mean=True,
                                   ylabel="held-out AUROC (each dot = one fold)",
                                   title="same neurons, same model — three CV schemes")
        _fig.add_hline(y=0.5, line=dict(color="#999", dash="dot"), annotation_text="chance")
        _fig.update_yaxes(range=[0.4, 1.0])
        _out = mo.vstack([mo.md(f"*{sess_info}.*"), _fig, mo.md("**" + "  ·  ".join(_summ) + "**")])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### What just happened

        Same 218 neurons, same logistic decoder, same frames — three different ways of splitting them:

        - **Shuffle (random K-fold): AUROC ≈ 0.95.** This is the headline number a careless analysis
          would report. It is almost entirely **leakage**: for every test frame, its slow-calcium
          near-twin one frame away sat in the training set, so the model was effectively tested on data
          it had already seen.
        - **Blocked (contiguous blocks): AUROC ≈ 0.55–0.75.** Hold out whole stretches of time and the
          number drops sharply, with fold-to-fold variability that is itself honest — some minutes of
          the session are more decodable than others.
        - **Contiguous (train past, test future): AUROC ≈ 0.50.** The strictest split lands near chance.

        The 0.95 was not a strong result that got a little weaker under scrutiny. It was **~0.4 AUROC of
        pure leakage** on top of a real effect that is far more modest, and possibly near zero for a
        strict forward-in-time prediction. Nothing about the data or the model changed — only the
        honesty of the split.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Why a picture of the split makes it obvious

        The mechanism is easiest to *see*. Below, a short timeline of frames is colored by fold under
        the two schemes. Under **random K-fold** the test frames (red) are scattered one-apart among the
        training frames (gray) — every test frame is flanked by its own near-duplicates in the training
        set. Under **blocked** CV the test frames form one solid contiguous run, with no training frame
        adjacent to leak from.
        """
    )
    return


@app.cell
def _(cu, go, mo, np):
    _n = 60
    _rng = np.random.RandomState(0)
    _shuffle_fold = _rng.randint(0, 5, _n)            # random fold assignment
    _blocked_fold = np.repeat(np.arange(5), _n // 5)  # contiguous blocks
    _test_fold = 0
    _sh = (_shuffle_fold == _test_fold).astype(int)
    _bl = (_blocked_fold == _test_fold).astype(int)
    _fig = go.Figure()
    _fig.add_trace(go.Heatmap(z=[_sh], y=["random K-fold"], colorscale=[[0, "#d9d9d9"], [1, "#e45756"]],
                              showscale=False, xgap=1, ygap=6))
    _fig.add_trace(go.Heatmap(z=[_bl + 2], y=["blocked"], colorscale=[[0, "#d9d9d9"], [1, "#e45756"]],
                              zmin=2, zmax=3, showscale=False, xgap=1, ygap=6))
    cu.apply_house_style(_fig, title="which frames are the TEST fold? gray = train, red = test", legend=None)
    _fig.update_xaxes(title="time (frames) →", showgrid=False, showticklabels=False)
    _fig.update_yaxes(showgrid=False)
    _fig.update_layout(height=220)
    mo.vstack([_fig,
               mo.md("*Under random K-fold every red test frame is sandwiched between gray training "
                     "frames that are, for slow calcium, almost the same measurement — the model peeks. "
                     "Under blocked CV the test frames are one contiguous novel stretch. This picture "
                     "is the entire lesson of Section 6.*")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6.1 · Exercise — why does AUROC fall when you stop shuffling?

        No code to write here — a reasoning check, because understanding *why* the number moves is the
        point. Read each statement and decide if it is true. The answer accordion follows.

        1. The blocked number is lower because the model is worse. **(False — it is the same model; only
           the evaluation changed. The shuffle number was inflated, the blocked one is honest.)**
        2. If the frames were truly independent (no autocorrelation), shuffle and blocked CV would give
           about the same AUROC. **(True — leakage needs neighbors to be similar; shuffle a genuinely
           independent dataset and there is nothing to leak.)**
        3. Blocked CV is always the "right" answer and shuffle CV is always wrong. **(False — for
           independent rows, e.g. the 2,499 pooled behavior events, shuffle CV is fine. Blocked CV is
           the fix specifically when samples are ordered and autocorrelated. The real rule is: your CV
           split must respect whatever structure makes rows non-independent — time here, cage in
           Section 1.)**

        Notice the deep symmetry with Section 1. Pseudoreplication and CV leakage are the **same
        mistake** wearing different clothes: both come from a non-independence the analysis ignored
        (cage there, time here). The fix is identical in spirit — split/shuffle at the unit where your
        data is actually exchangeable.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        > **Section 6 answer.** A CV number is only as honest as its split. Random K-fold on
        > autocorrelated data leaks and inflated our neural decoder from an honest ~0.6 to a bogus 0.95.
        > Use blocked/forward CV whenever rows are ordered in time. **Next:** whichever number we keep,
        > how do we read the null it is compared against?
        """
    )
    return


# ============================================================================================
# ==============  SECTION 7 — READING NULLS AND POWER  =======================================
# ============================================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # 7 · Reading a null distribution, and power

        ## Why this matters

        Half of the tools above build a **null distribution** by shuffling, then ask where the real
        value falls. That is powerful and assumption-light — but only if you read the null correctly and
        build enough of it. Two habits separate an honest permutation p from a hand-wave.

        ## 7.1 · Report the count, not "it lands outside the cloud"

        A permutation p-value has an exact definition:

        $$ p = \frac{1 + \#\{\text{shuffles} \ge \text{observed}\}}{1 + n_{\text{shuffles}}} $$

        The "+1" counts the observed value as one of its own draws, so `p` can never be exactly 0 —
        the smallest it can report is `1/(n+1)`. Say "**0 of 20,000 shuffles beat the observed gap, so
        p < 5e-5**", not "the observed value lands outside the cloud." The precise statement tells the
        reader your resolution.

        ## 7.2 · How many shuffles? Enough to resolve your threshold

        If you only run **50** shuffles, the 95th percentile of your null is pinned by just **2–3**
        draws — the null's own tail is too noisy to test against. Below, we estimate the same 95th
        percentile of a standard-normal null from 50 shuffles vs 5,000, repeated many times, and show
        how much the estimate wobbles.
        """
    )
    return


@app.cell
def _(cu, go, mo, np):
    _rng = np.random.RandomState(0)
    _reps = 400
    _q_small = np.array([np.percentile(_rng.randn(50), 95) for _ in range(_reps)])
    _q_big = np.array([np.percentile(_rng.randn(5000), 95) for _ in range(_reps)])
    _truth = 1.645          # true 95th percentile of a standard normal
    _fig = go.Figure()
    _fig.add_histogram(x=_q_small, nbinsx=40, name="50 shuffles", marker_color="#e45756", opacity=0.6)
    _fig.add_histogram(x=_q_big, nbinsx=40, name="5000 shuffles", marker_color="#4c78a8", opacity=0.6)
    _fig.add_vline(x=_truth, line=dict(color="#333", dash="dash"), annotation_text="true 95th pct")
    cu.apply_house_style(_fig, title="estimated 95th percentile of the SAME null — 50 vs 5000 shuffles",
                         legend="below")
    _fig.update_layout(barmode="overlay")
    _fig.update_xaxes(title="estimated 95th percentile", showgrid=False)
    _fig.update_yaxes(title="count", showgrid=False)
    mo.vstack([_fig,
               mo.md(f"With **50 shuffles** the 95th-percentile estimate ranges roughly "
                     f"[{_q_small.min():.2f}, {_q_small.max():.2f}] — a threshold that swings by half a "
                     f"unit run to run. With **5,000** it tightens around the truth ({_truth:.2f}). If "
                     f"your decision (significant or not) sits near the null's tail, a coarse null can "
                     f"flip the verdict by luck. Use thousands of shuffles for a 0.05 threshold, more "
                     f"for smaller α.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 7.3 · Establish chance empirically, not by assertion

        It is tempting to write "chance = 0.5" for a balanced decoder or "chance = 1/3 for three
        states." Those are the *theoretical* values, but finite data and class imbalance shift the real
        chance level. The honest move is to **shuffle the labels** and let the data tell you where chance
        actually is — the label-shuffle null is the empirical chance band. Below, we shuffle the
        aggression labels and re-run the pooled behavior decoder's 5-fold AUROC many times; the cloud is
        the empirical chance band, and the real decoder sits far outside it.
        """
    )
    return


@app.cell
def _(X, cu, go, mo, np, yagg):
    from sklearn.model_selection import cross_val_score as _cvs, StratifiedKFold as _SKF
    from sklearn.pipeline import make_pipeline as _mkp
    from sklearn.preprocessing import StandardScaler as _SS
    from sklearn.impute import SimpleImputer as _SI
    from sklearn.linear_model import LogisticRegression as _LR
    _clf = _mkp(_SI(strategy="median"), _SS(), _LR(max_iter=1000))
    _cv = _SKF(5, shuffle=True, random_state=0)
    _real = float(_cvs(_clf, X, yagg, cv=_cv, scoring="roc_auc").mean())
    _rng = np.random.RandomState(0)
    _null = np.array([float(_cvs(_clf, X, _rng.permutation(yagg), cv=_cv, scoring="roc_auc").mean())
                      for _ in range(40)])
    _p = (np.sum(_null >= _real) + 1) / (len(_null) + 1)
    _fig = cu.strip_points_fig(_null, np.array(["label-shuffled null"] * len(_null)),
                               colors={"label-shuffled null": "#bab0ac"}, show_mean=True,
                               ylabel="5-fold AUROC", title="empirical chance band vs the real decoder")
    _fig.add_hline(y=_real, line=dict(color="#e45756", width=3),
                   annotation_text=f"real decoder = {_real:.3f}")
    _fig.add_hline(y=0.5, line=dict(color="#999", dash="dot"), annotation_text="theoretical 0.5")
    # strip_points_fig(robust=True) auto-clips the y-axis to the null cloud (~0.47–0.53); force a
    # range that also contains the real-decoder hline so the whole point of the cell is on-screen.
    _fig.update_yaxes(range=[0.45, 0.9])
    mo.vstack([_fig,
               mo.md(f"The 40 label-shuffled AUROCs cluster near **{_null.mean():.3f}** (not exactly "
                     f"0.500 — finite-sample wobble), and the real decoder at **{_real:.3f}** beats "
                     f"every one of them (**0 of 40**, empirical p = {cu.fmt_p(_p)}). This is the same "
                     f"shuffle-null template used for the behavior grammar (NB06) and the sequence "
                     f"(Section 5): draw the null, then report the observed value's rank in it — never "
                     f"assert chance, measure it.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        > **Section 7 answer.** A permutation p is a counting statement — report the count and use enough
        > shuffles that the null's tail is resolved. Measure chance by shuffling, do not assume it.
        > **Next:** assemble these habits into a single honest default.
        """
    )
    return


# ============================================================================================
# ==============  SECTION 8 — PUTTING IT TOGETHER  ===========================================
# ============================================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # 8 · Putting it together — the honest default

        The eight mistakes share one antidote, applied in three moves. Whenever you report a result:

        1. **Find the exchangeable unit** and test/split at it — cage for a between-cage variable, time
           for an autocorrelated series, event only when events are truly independent.
        2. **Report an effect size with a confidence interval**, not just a p-value — so significance
           and importance are separate.
        3. **Compare to a null you built** by shuffling at that unit, and report the count.

        For a **proportion** — an aggression rate, a social-neuron fraction — the natural interval is
        the **Wilson confidence interval** (`cu.wilson_ci`), which, unlike the naive ±1.96·SE, never
        spills below 0 or above 1. Below is the same aggression-by-cage question, done the honest way:
        the **outcome** (each cage's aggression rate) with a Wilson CI, never the median-split variable
        itself.
        """
    )
    return


@app.cell
def _(cage, cu, mo, np, sex, ucage, yagg):
    _k = np.array([int(yagg[cage == c].sum()) for c in ucage])
    _nn = np.array([int((cage == c).sum()) for c in ucage])
    _sx = np.array([sex[cage == c][0] for c in ucage])
    _fig = cu.proportion_ci_fig(_k, _nn, [f"cage {c}\n({s})" for c, s in zip(ucage, _sx)],
                                colors={f"cage {c}\n({s})": (cu.RANK_HEX[2] if s == "M" else "#e45756")
                                        for c, s in zip(ucage, _sx)},
                                ylabel="aggression rate", title="aggression rate per cage, with Wilson 95% CIs")
    mo.vstack([_fig,
               mo.md("Each point is one cage's aggression rate; the bars are Wilson 95% CIs, wide for "
                     "the smaller cages. The male (green) and female (red) cages interleave — consistent "
                     "with the null aggression-rate-by-sex result from Section 1. Plotting the outcome "
                     "with a CI, at the honest unit, shows both the estimate and how much to trust it — "
                     "the whole notebook in one figure.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## What we reached, and where it goes

        We committed eight classic mistakes on data we understood, watched each produce a confident
        wrong answer, and fixed each:

        - **Pseudoreplication** — the unit of analysis *is* the sample size; test at the cage, not the
          event, for a between-cage variable.
        - **Test choice** — the test encodes the question (direction vs magnitude; paired vs unpaired;
          ECDF vs violin for a small shift).
        - **Multiple comparisons** — scanning inflates false positives; correct, and separate
          confirmatory from exploratory.
        - **Effect size vs p** — at large n a trivial effect is "significant"; always report a d with a
          CI.
        - **Circular analysis** — never select and test on the same data; hold out.
        - **CV leakage** — random K-fold on autocorrelated data leaks; the neural decoder fell from a
          bogus 0.95 to an honest ~0.6 under blocked CV.
        - **Reading nulls** — a permutation p is a count; measure chance by shuffling, use enough draws.
        - **The synthesis** — effect size + CI + permutation at the right unit is the default.

        The single throughline held all the way through: **respect the unit at which your observations
        are independent.** Cage in Section 1, time in Section 6 — the same discipline.

        > **Where Week 2 goes.** Every tool here transfers directly to neural data, and Section 6 already
        > gave the preview: the same logistic decoder, the same permutation null, the same CV-leakage
        > trap — now reading behavior off populations of neurons instead of off 19 pose features. The
        > deep new wrinkle we meet there is that neural data does not come as a tidy fixed matrix: the
        > number of neurons is unknown and varies by animal, so it cannot be pooled the way we pooled
        > behavior events. Reading a population honestly starts from exactly the rigor this notebook
        > built.
        """
    )
    return


if __name__ == "__main__":
    app.run()
