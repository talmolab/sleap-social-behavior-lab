# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "marimo>=0.9",
#     "numpy>=1.24,<2.1",
#     "scipy>=1.11",
#     "pandas>=2.0",
#     "scikit-learn>=1.3",
#     "plotly>=5.20",
#     "h5py>=3.10",
#     "gdown>=5.1",
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
            if os.path.isdir(os.path.join(p, "course")):
                return p
            p = os.path.dirname(p)
        return None
    ROOT = _find_root() or os.getcwd()
    _nu = os.path.join(ROOT, "course", "neural_utils.py")
    if not os.path.exists(_nu):
        os.makedirs(os.path.dirname(_nu), exist_ok=True)
        urllib.request.urlretrieve(_RAW + "/course/neural_utils.py", _nu)
    sys.path.insert(0, os.path.join(ROOT, "course"))
    import neural_utils as nu
    CACHE = nu.cache_dir(ROOT)
    return CACHE, ROOT, go, np, nu


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # NB06 · Motion Correction — holding the image still

        ## Week 2 opens: from behavior to the brain

        Across Week 1 we built, one step at a time, an objective readout of *what a mouse does*. We
        asked what a keypoint is (NB01), what the body's motion looks like over time (NB02), how to
        compress the 19 allocentric features without throwing away structure (NB03), whether the
        low-dimensional map carries real cage-level biology (NB04), and finally whether we can read
        behavior in time and **decode aggression** from it (NB05). NB05 closed Week 1 with a working
        aggression decoder that held up across two independent cohorts. We can now read what a mouse
        does, objectively, from tracked points.

        Week 2 asks the other half of the question this course is built around: **what
        is the brain doing while the mouse behaves?** To relate neural activity to behavior we first
        have to *record* neurons and read their activity cleanly. Those recordings are movies. This
        notebook asks the very first question standing between a raw movie and a usable neural signal:

        > **To read neurons from a movie, how do we first hold the image still?**

        ## Why imaging needs motion correction

        The Week 2 recordings come from a **miniature microscope** — a "miniscope": a small camera
        mounted on a freely moving animal's head that films a patch of brain tissue. A genetically
        encoded **calcium reporter** (a fluorescent protein that brightens when calcium floods into an
        active neuron) turns each recording into a movie in which bright spots switch on and off as
        neurons fire. To measure one neuron's activity over time we look at the pixels where that neuron
        sits and track how their brightness changes frame by frame.

        **This only works if a given pixel keeps pointing at the same piece of tissue — and it does
        not.** The animal walks, rears, grooms, and turns; the brain shifts slightly under the lens.
        The camera's view — the **field of view**, or FOV — drifts. When it drifts, pixel `(80, 40)` in
        frame 500 and pixel `(80, 40)` in frame 501 are no longer the same neuron. Read the brightness
        at a fixed pixel and you get a mixture of *"the neuron fired"* and *"the neuron slid away,"* with
        no way to separate them. A neuron's trace is not a well-defined thing until the frames are
        aligned.

        **Motion correction** (also called **registration**) is the fix. We define it plainly:

        > **Motion correction** estimates, for every frame, how far the image has moved relative to a
        > fixed **reference template**, then shifts the frame back so that each pixel lines up with the
        > same tissue across the entire recording.

        This is the same class of problem we already solved for behavior. In Week 1 a pose track was
        meaningless until we stabilized identity — until "mouse 2 at frame 500" was guaranteed to be the
        same animal as "mouse 2 at frame 501." Aligning a signal across time before reading it is a
        general step, and imaging needs it just as much as pose tracking does. Behavior and its neural
        basis are two halves of the same study, and both depend on the same discipline of not reading a
        signal until it holds still.

        ## Rigid vs piecewise-rigid — two flavors of "hold it still"

        There are two levels of ambition, and it is worth defining both before we see them:

        - **Rigid registration.** Shift the *whole* frame by a single translation (one left/right and
          one up/down offset) chosen to best match the reference. This corrects **global drift**, where
          the entire FOV slides together. It cannot fix motion that differs across the frame.
        - **Piecewise-rigid registration.** Divide the frame into patches, shift *each patch on its own*,
          and blend the patch shifts back together smoothly. Because different patches can move by
          different amounts, this corrects **non-uniform motion**, where one part of the tissue slides or
          stretches differently from another — exactly what happens when soft brain tissue deforms under
          a moving lens.

        ## What this notebook does

        We work with one **real** miniscope movie that has been processed three ways and laid out side
        by side — `raw | rigid | pw-rigid` — so we can compare an uncorrected recording against both
        registration methods directly. We will:

        1. inspect the movie's metadata and its mean/variance projections (real EDA);
        2. watch a single frame across the three panels, and read a **fixed pixel's brightness over
           time** to see concretely why uncorrected pixels are corrupted;
        3. collapse the whole time axis into a **kymograph** so motion becomes a picture;
        4. reduce motion to a single number — the **motion index** — and show its per-frame distribution
           with individual data points;
        5. validate that number against **ground truth** with a synthetic-jitter experiment; and
        6. implement the motion index ourselves and confirm the honest conclusion:
           `MI(pw-rigid) < MI(rigid) < MI(raw)`.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 1. Load the movie and look at it before we touch it

        Good analysis starts with plain EDA: what is this file? We fetch the side-by-side
        motion-correction movie (cached; downloaded once), read it into memory subsampled in time so a
        modest kernel loads it quickly, and split the frame into its three equal-width panels.

        **The helpers, stated plainly.**

        - `nu.read_video(path, step=3, gray=True)` — *purpose:* read a movie into a numpy array without
          streaming the whole thing at full rate. *Input:* a file path; `step=3` keeps every third frame.
          *Output:* a `(F, H, W)` float32 grayscale array (F frames, H rows, W columns).
        - `nu.split_thirds(mov)` — *purpose:* cut the side-by-side frame into its three panels. *Input:*
          the `(F, H, W)` movie. *Output:* three `(F, H', w)` arrays `raw, rigid, pwr`, each one third
          the width, with the top and bottom cropped so the panels align.
        """
    )
    return


@app.cell
def _(CACHE, np, nu):
    # Load the side-by-side motion-correction movie (cached; downloaded once via gdown).
    # Subsample step=3 -> ~185 frames so a headless kernel loads it fast and keeps memory modest.
    MOV_PATH = nu.fetch_gdrive(nu.MOCO_GDRIVE_ID, nu.MOCO_NAME, CACHE and nu.find_root())
    mov = nu.read_video(MOV_PATH, step=3, gray=True)       # (F, H, W) grayscale float32
    raw, rigid, pwr = nu.split_thirds(mov)                 # each (F, H', w)
    F, H, W = raw.shape
    # global gray range for a shared colorscale across the three panels
    VMIN = float(np.percentile(mov, 1))
    VMAX = float(np.percentile(mov, 99))
    return F, H, MOV_PATH, VMAX, VMIN, W, mov, pwr, raw, rigid


@app.cell(hide_code=True)
def _(F, H, MOV_PATH, W, mo, mov, nu):
    # EDA: report the movie's real metadata and the loaded array shapes.
    _meta = nu.video_meta(MOV_PATH)
    _fps = _meta.get("fps", "?")
    _size = _meta.get("size", "?")
    _dur = _meta.get("duration", "?")
    _nf = _meta.get("nframes", "?")
    mo.md(
        f"""
        **What we loaded.** The raw file reports `fps = {_fps}`, frame size `{_size}` (width × height of
        the full three-panel image), duration `{_dur}` s, and roughly `{_nf}` frames. After subsampling
        (`step=3`) and splitting into thirds we hold **{F} frames** of each panel, each **{H} rows ×
        {W} columns**. The combined loaded movie is `{tuple(mov.shape)}` — the three panels sit
        left-to-right inside that width. Everything below reads from `raw`, `rigid`, and `pwr`; no
        further file loading is needed.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Where is the tissue? Two projections

        Before watching the movie play, we summarize it over time with two **projection images** — each
        pixel collapsed across all frames into a single value:

        - **Mean projection** (`raw.mean(axis=0)`): the average brightness at each pixel. This shows the
          steady anatomy — the general shape of the imaged tissue and the optics' uneven illumination.
        - **Standard-deviation projection** (`raw.std(axis=0)`): how much each pixel's brightness
          *varies* over time. Pixels that stay constant are dark here; pixels that change — because a
          neuron there fires, **or because the image slides across them** — are bright. In an
          uncorrected movie, the std projection lights up along edges precisely where motion drags
          bright structure back and forth. That bright rim of "variance from motion" is exactly what
          registration should remove.
        """
    )
    return


@app.cell
def _(mo, nu, raw):
    # Two time-projections of the RAW panel (core beat; both render on load).
    _mean_img = raw.mean(axis=0)
    _std_img = raw.std(axis=0)
    _f_mean = nu.image_fig(_mean_img, title="raw — mean projection (anatomy)",
                           colorscale="gray", height=340)
    _f_std = nu.image_fig(_std_img, title="raw — std projection (variance, incl. motion)",
                          colorscale="Viridis", colorbar_title="std", height=340)
    mo.hstack([_f_mean, _f_std])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 2. One movie, three registrations, side by side

        Comparing the three versions of the *same* recording at the same instant is the clearest way to
        see what each method does.

        - **left — raw:** straight off the sensor, no correction.
        - **middle — rigid:** one whole-frame translation per frame.
        - **right — pw-rigid:** per-patch translations blended together.

        Drag the frame slider and fix your eye on a single bright cell. In **raw** it wanders; in
        **rigid** it steadies; in **pw-rigid** it holds most still. A single frame is only a weak test —
        motion lives in how the image changes *over time* — but it builds the right intuition before we
        quantify it. (The movie is subsampled in time so it loads quickly.)
        """
    )
    return


@app.cell
def _(F, mo):
    frame_ctrl = mo.ui.slider(0, F - 1, value=F // 2, step=1,
                              label="frame", debounce=True, full_width=True)
    return (frame_ctrl,)


@app.cell
def _(VMAX, VMIN, frame_ctrl, mo, pwr, raw, rigid):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as _pgo
    _t = frame_ctrl.value
    _panels = [("raw", raw), ("rigid", rigid), ("pw-rigid", pwr)]
    _fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.02,
                         subplot_titles=[n for n, _ in _panels])
    for _j, (_name, _p) in enumerate(_panels, start=1):
        _fig.add_trace(_pgo.Heatmap(z=_p[_t], colorscale="gray", zmin=VMIN, zmax=VMAX,
                                    showscale=(_j == 3), colorbar=dict(title="gray", len=0.9)),
                       row=1, col=_j)
        _ax = "" if _j == 1 else str(_j)
        _fig.update_yaxes(autorange="reversed", scaleanchor="x" + _ax, scaleratio=1,
                          visible=False, row=1, col=_j)
        _fig.update_xaxes(visible=False, row=1, col=_j)
    _fig.update_layout(template="plotly_white", height=340,
                       margin=dict(l=10, r=10, t=40, b=10),
                       title=f"frame {_t}  —  raw | rigid | pw-rigid")
    mo.vstack([frame_ctrl, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 3. Read a fixed pixel over time — why uncorrected traces lie

        Here is the core problem made concrete. Reading a neuron means picking a small **region of
        interest** (ROI) — a patch of pixels sitting on one cell — and averaging their brightness in
        each frame to get a trace over time. We place the *same* ROI, at the *same* pixel coordinates,
        on all three panels and read out `panel[:, y0:y1, x0:x1].mean(axis=(1, 2))`.

        If the movie were perfectly still, the three traces would differ only by how the panel was
        processed. They do not. In the **raw** panel the trace jumps and dips as the image slides bright
        structure into and out of the fixed ROI — those excursions are *motion masquerading as
        activity*. In the **rigid** and especially **pw-rigid** panels the same ROI stays on the same
        tissue, so its trace is smoother and reflects real brightness changes. Move the ROI around: over
        a high-contrast edge the raw trace is wildest, because that is where a small shift changes the
        pixel value the most.
        """
    )
    return


@app.cell
def _(H, W, mo):
    roi_x = mo.ui.slider(4, W - 5, value=W // 2, step=1,
                         label="ROI center x (column)", debounce=True, full_width=True)
    roi_y = mo.ui.slider(4, H - 5, value=H // 2, step=1,
                         label="ROI center y (row)", debounce=True, full_width=True)
    return roi_x, roi_y


@app.cell
def _(go, mo, pwr, raw, rigid, roi_x, roi_y):
    # A fixed 9x9-px ROI read from all three panels; brightness over time.
    _r = 4
    _y, _x = roi_y.value, roi_x.value
    _sl = (slice(None), slice(_y - _r, _y + _r + 1), slice(_x - _r, _x + _r + 1))
    _colors = {"raw": "#e45756", "rigid": "#4c78a8", "pw-rigid": "#54a24b"}
    _fig = go.Figure()
    for _name, _panel in [("raw", raw), ("rigid", rigid), ("pw-rigid", pwr)]:
        _trace = _panel[_sl].mean(axis=(1, 2))
        _fig.add_scatter(y=_trace, mode="lines", name=_name,
                         line=dict(color=_colors[_name],
                                   width=1.6 if _name == "pw-rigid" else 1.1))
    _fig.update_layout(template="plotly_white", height=320,
                       margin=dict(l=10, r=10, t=44, b=10),
                       title=f"ROI brightness over time at (y={_y}, x={_x}) — raw wanders, "
                             f"corrected steadies",
                       xaxis_title="frame", yaxis_title="mean ROI brightness (gray)",
                       legend=dict(orientation="h", y=1.13))
    mo.vstack([roi_x, roi_y, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 4. Kymographs — turning motion into a picture

        A single frame is a weak test and a single ROI is only one location. To judge motion across the
        whole image and the whole recording at once, we use a **kymograph**.

        **Definition.** A **kymograph** takes one horizontal line of the image (a single row of pixels
        at height `y`) and stacks that line across every frame. The result is a 2-D image whose
        horizontal axis is **position along the row** and whose vertical axis is **time**. Each row of
        the kymograph is what that one image line looked like at one moment.

        **How to read it.** A feature that stays put traces a **straight vertical streak** down the
        kymograph. A feature that jitters left and right traces a **wiggly** streak. So "is the movie
        well registered?" becomes the simpler visual question "are the streaks straight?" This is the
        imaging counterpart of overlaying every frame of a pose track and asking whether the line holds
        still.

        **Controls.** Pick the row `y` (which image line to track) and a **time window** to highlight
        (the red band). Compare the panels: raw streaks wander, rigid straightens them, and pw-rigid
        straightens them the most. Try a row that runs through a bright cell — the straightening is most
        obvious where there is high-contrast structure to follow.
        """
    )
    return


@app.cell
def _(F, H, mo):
    row_ctrl = mo.ui.slider(0, H - 1, value=min(80, H - 1), step=1,
                            label="kymograph row y (which image line to track)",
                            debounce=True, full_width=True)
    win_ctrl = mo.ui.range_slider(0, F - 1, value=[int(0.55 * F), int(0.72 * F)], step=1,
                                  label="highlight time window [t0, t1] (frames)",
                                  debounce=True, full_width=True)
    return row_ctrl, win_ctrl


@app.cell
def _(mo, pwr, raw, rigid, row_ctrl, win_ctrl):
    from plotly.subplots import make_subplots as _mksub
    import plotly.graph_objects as _kgo
    _y = row_ctrl.value
    _t0, _t1 = win_ctrl.value
    _kymos = [("Raw", raw[:, _y, :]), ("Rigid", rigid[:, _y, :]), ("PW-Rigid", pwr[:, _y, :])]
    _fig = _mksub(rows=1, cols=3, horizontal_spacing=0.04,
                  subplot_titles=[n for n, _ in _kymos])
    for _j, (_name, _k) in enumerate(_kymos, start=1):
        _fig.add_trace(_kgo.Heatmap(z=_k, colorscale="gray", showscale=False), row=1, col=_j)
        _ax = "" if _j == 1 else str(_j)
        _fig.update_yaxes(autorange="reversed", title="time (frames)" if _j == 1 else None,
                          row=1, col=_j)
        _fig.update_xaxes(title="position (px)", row=1, col=_j)
        # red band marking the highlighted time window (spans the full subplot width)
        _fig.add_shape(type="rect", xref="x" + _ax + " domain", x0=0, x1=1,
                       yref="y" + _ax, y0=_t0, y1=_t1,
                       line=dict(color="red", width=1), fillcolor="red", opacity=0.12,
                       layer="above", row=1, col=_j)
    _fig.update_layout(template="plotly_white", height=460,
                       margin=dict(l=10, r=10, t=40, b=10),
                       title=f"kymographs at row y = {_y}   ·   window [{_t0}, {_t1}]")
    mo.vstack([row_ctrl, win_ctrl, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 5. The motion index — one number for how much jitter is left

        The kymograph turns "is it registered?" into "are the streaks straight?", but judging streaks by
        eye does not scale to thousands of movies, and it struggles to separate rigid from pw-rigid when
        both look reasonable. For that we need a single number.

        **Definition.** The **motion index** measures how much the picture changes from one frame to the
        next. `nu.motion_index(frames)` computes the mean absolute **frame-to-frame difference**,

        $$\text{MI} = \big\langle\,|f_{t+1} - f_{t}|\,\big\rangle,$$

        the average — over every pixel and every pair of neighboring frames — of how much a pixel's
        brightness changed. A perfectly still movie has a motion index near zero because consecutive
        frames are nearly identical. A jittery movie has a large motion index because every pixel keeps
        changing as the image slides.

        **The helpers, stated plainly.**

        - `nu.motion_index(frames)` — *purpose:* score total frame-to-frame motion. *Input:* a movie
          `(F, H, W)`. *Output:* one number.
        - `nu.motion_index_trace(frames)` — *purpose:* show that jitter *over time* instead of
          collapsing it. *Input:* a movie `(F, H, W)`. *Output:* a 1-D array of length `F - 1`, one
          motion value per adjacent frame pair.

        If registration works it should push the motion index **down**, and the more flexible method
        should push it lower still:

        $$\text{MI}(\text{raw}) \;>\; \text{MI}(\text{rigid}) \;>\; \text{MI}(\text{pw-rigid}).$$

        The per-frame trace below plots `nu.motion_index_trace` for all three versions. The pw-rigid
        line stays lowest across almost the whole recording — a well-stabilized signal has a small
        frame-to-frame change.
        """
    )
    return


@app.cell
def _(go, nu, pwr, raw, rigid):
    # Per-frame jitter trace for each registration (renders on load — this is a core beat).
    _tr_raw = nu.motion_index_trace(raw)
    _tr_rig = nu.motion_index_trace(rigid)
    _tr_pwr = nu.motion_index_trace(pwr)
    _fig = go.Figure()
    _fig.add_scatter(y=_tr_raw, mode="lines", name="raw", line=dict(color="#e45756", width=1))
    _fig.add_scatter(y=_tr_rig, mode="lines", name="rigid", line=dict(color="#4c78a8", width=1))
    _fig.add_scatter(y=_tr_pwr, mode="lines", name="pw-rigid", line=dict(color="#54a24b", width=1.4))
    _fig.update_layout(template="plotly_white", height=300,
                       margin=dict(l=10, r=10, t=40, b=10),
                       title="per-frame motion index (lower = more stable)",
                       xaxis_title="frame", yaxis_title="mean |Δ| to previous frame",
                       legend=dict(orientation="h", y=1.12))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The distribution, not just the average

        A single line per method (or a bar of means) hides *how* the frames are distributed. The honest
        way to compare the three panels is to show **every per-frame motion value as its own point**.
        Each of the `F - 1` frame pairs contributes one dot; the box marks the median and quartiles.

        Read it as three clouds. The **raw** cloud sits highest and spreads widest — some frame pairs
        jump a lot when the animal moves sharply. The **rigid** cloud shifts down. The **pw-rigid** cloud
        sits lowest and tightest: registration lowers not only the average jitter but the worst-case
        frames too. Hover any point to see its frame index and value. This is the same seaborn-style
        "show the raw points" discipline we used for behavioral distributions in Week 1 — a summary bar
        would have concealed the spread.
        """
    )
    return


@app.cell
def _(np, nu, pwr, raw, rigid):
    # Seaborn-style distribution of the PER-FRAME motion index, one dot per frame pair (core beat).
    _tr = {"raw": nu.motion_index_trace(raw),
           "rigid": nu.motion_index_trace(rigid),
           "pw-rigid": nu.motion_index_trace(pwr)}
    _vals = np.concatenate([_tr["raw"], _tr["rigid"], _tr["pw-rigid"]])
    _grp = np.array(["raw"] * len(_tr["raw"]) + ["rigid"] * len(_tr["rigid"]) +
                    ["pw-rigid"] * len(_tr["pw-rigid"]))
    _hover = np.concatenate([[f"frame {i}" for i in range(len(_tr[k]))]
                             for k in ("raw", "rigid", "pw-rigid")])
    nu.box_points_fig(
        _vals, _grp, group_order=["raw", "rigid", "pw-rigid"],
        colors={"raw": "#e45756", "rigid": "#4c78a8", "pw-rigid": "#54a24b"},
        ylabel="per-frame motion index (mean |Δ|)", xlabel="registration",
        title="per-frame motion index by method — every frame pair is one point", height=470)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 6. Does the motion index really measure motion? A ground-truth check

        Before we trust the motion index to *rank* raw / rigid / pw-rigid, we should confirm it tracks
        motion we control. We run a small synthetic experiment: take **one clean frame**, then build
        fake movies by shifting that frame by known random offsets of increasing amplitude. Since we
        injected the motion ourselves, we know the right answer.

        Two curves come out of it:

        - **jittered** — as we increase the shift amplitude (in pixels), the motion index rises
          monotonically. More injected motion, larger index. Good: the metric responds to motion.
        - **perfectly corrected** — if we roll each frame back by the *exact* known shift, every frame
          returns to the clean reference, consecutive frames become identical, and the motion index
          collapses to essentially zero at all amplitudes.

        That flat-zero "corrected" curve is registration in its idealized form: knowing the true shift
        and undoing it. Real registration does not know the true shift — it *estimates* it from the
        images — so rigid and pw-rigid land between the raw movie and this ideal. The experiment
        licenses using the motion index as a scoreboard.
        """
    )
    return


@app.cell
def _(go, np, nu, raw):
    # Ground-truth: inject known jitter into ONE clean frame; MI rises with amplitude, and undoing the
    # exact shift returns MI to ~0. Small teaching loop over 9 amplitudes (cheap; renders on load).
    _ref = raw[raw.shape[0] // 2]                 # one clean (H', w) reference frame
    _rng = np.random.RandomState(0)
    _amps = np.arange(0, 9)                       # injected jitter amplitude in pixels
    _mi_jit, _mi_fix = [], []
    for _a in _amps:
        _n = 40                                   # frames in each synthetic movie
        _dy = _rng.randint(-_a, _a + 1, size=_n) if _a > 0 else np.zeros(_n, int)
        _dx = _rng.randint(-_a, _a + 1, size=_n) if _a > 0 else np.zeros(_n, int)
        _jit = np.stack([np.roll(np.roll(_ref, int(y), 0), int(x), 1)
                         for y, x in zip(_dy, _dx)])
        _fix = np.stack([np.roll(np.roll(_j, -int(y), 0), -int(x), 1)
                         for _j, y, x in zip(_jit, _dy, _dx)])
        _mi_jit.append(nu.motion_index(_jit))
        _mi_fix.append(nu.motion_index(_fix))
    _fig = go.Figure()
    _fig.add_scatter(x=_amps, y=_mi_jit, mode="lines+markers", name="jittered",
                     line=dict(color="#e45756", width=2))
    _fig.add_scatter(x=_amps, y=_mi_fix, mode="lines+markers", name="perfectly corrected",
                     line=dict(color="#54a24b", width=2))
    _fig.update_layout(template="plotly_white", height=340,
                       margin=dict(l=10, r=10, t=44, b=10),
                       title="motion index vs injected jitter (ground truth)",
                       xaxis_title="injected shift amplitude (px)",
                       yaxis_title="motion index", legend=dict(orientation="h", y=1.12))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 7. Exercise — implement the motion index yourself

        **Python skill practiced:** *array arithmetic and axis reductions.* Earlier notebooks had you
        index and slice arrays (`kp[frame, mouse, node]`) and build boolean masks. Here you write a tiny
        function that differences an array along a chosen axis, takes an absolute value, and reduces the
        whole thing to one number. Understanding *which axis is time* is the crux — the same idea powers
        the velocity and cross-correlation features from Week 1.

        **The question.** Confirm, with a number you compute yourself, that registration reduces
        frame-to-frame motion and that piecewise-rigid beats plain rigid:
        $\text{MI}(\text{pw-rigid}) < \text{MI}(\text{rigid}) < \text{MI}(\text{raw})$.

        **What you already have.** `raw`, `rigid`, `pwr` — the three panels, each `(F, H, W)`. And
        `np` for array math. You will rebuild `nu.motion_index` from scratch so you understand what the
        one number is made of.

        **Your job.** Fill the two blanks in `my_motion_index` (next cell). Each blank has a comment
        stating exactly what to type and why it matters — and, so the notebook runs end to end, the
        answer is already written on the line (find the `# <-- the ____ line` marker, cover it, and try
        it yourself before peeking). Then the three panel scores are computed for you.

        **What to expect.** The self-check below reports your three numbers and confirms they (a) match
        the library's `nu.motion_index` to floating-point precision — proof you rebuilt it correctly —
        and (b) fall in the order `mi_pwr < mi_rigid < mi_raw`, with pw-rigid removing roughly a tenth of
        the raw movie's motion. The grade is on the *conclusion* (agreement + ordering + a tolerance
        band), not an exact decimal, because the absolute values scale with how the movie was subsampled.
        """
    )
    return


@app.cell
def _(np, pwr, raw, rigid):
    # ------------------------------------------------------------------ YOUR CODE (edit this cell)
    def my_motion_index(frames):
        # `frames` is a movie of shape (F, H, W): F frames stacked along axis 0 (the TIME axis).
        #
        # BLANK 1 — choose the axis to difference along.
        #   np.diff(a, axis=k) subtracts each slice from the next ALONG axis k. We want the change
        #   from one FRAME to the next, and frames are stacked along axis 0. So the time axis is 0.
        #   Differencing along axis 0 gives an (F-1, H, W) array of frame-to-frame changes.
        #   WHY IT MATTERS: pick the wrong axis and you would be measuring how brightness changes
        #   ACROSS the image within a single frame, not how the image moves over time.
        _diffs = np.diff(frames, axis=0)           # <-- the ____ line: replace ____ with 0 (time axis)
        #
        # BLANK 2 — reduce to one number.
        #   A pixel that brightens and one that darkens have BOTH changed, so take the absolute value
        #   first, then average over EVERY pixel and EVERY frame pair to get a single scalar.
        #   WHY IT MATTERS: without the absolute value, equal brightening and darkening would cancel
        #   and a very jittery movie could report near-zero motion.
        return np.abs(_diffs).mean()               # <-- the ____ line: replace ____ with mean
    # ---------------------------------------------------------------------------------------------
    mi_raw = my_motion_index(raw)     # motion index of the raw panel
    mi_rigid = my_motion_index(rigid)  # motion index of the rigid panel
    mi_pwr = my_motion_index(pwr)     # motion index of the pw-rigid panel
    return mi_pwr, mi_raw, mi_rigid, my_motion_index


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show solution": mo.md(
            r"""
            ```python
            def my_motion_index(frames):
                _diffs = np.diff(frames, axis=0)   # difference along the TIME axis -> (F-1, H, W)
                return np.abs(_diffs).mean()       # |change|, averaged over all pixels & frame pairs

            mi_raw   = my_motion_index(raw)     # ~5.50 at step=3  (3.24 at full resolution)
            mi_rigid = my_motion_index(rigid)   # ~5.10            (3.08)
            mi_pwr   = my_motion_index(pwr)     # ~4.93            (2.97)
            ```

            **What you should find.** Your three numbers match `nu.motion_index` exactly (you rebuilt
            the same computation) and fall in order `mi_pwr < mi_rigid < mi_raw`. Rigid registration
            removes a few percent of the raw movie's frame-to-frame motion; piecewise-rigid removes
            about 10 percent in total, because it also catches the *local* warping a single whole-frame
            shift cannot. The absolute values depend on the subsampling step (larger frame gaps make the
            apparent motion larger), which is why the **ordering** is the real result rather than any
            single decimal.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mi_pwr, mi_raw, mi_rigid, mo, nu, pwr, raw, rigid):
    # Honest self-check. Pinned conclusion from the real mmc3 movie: (a) the student's function matches
    # nu.motion_index to fp precision, and (b) registration MONOTONICALLY reduces the motion index
    # (raw > rigid > pw-rigid), with pw-rigid removing ~10% of raw's motion. Tolerance band on the
    # reduction is 3%..25% (robust to subsample step), so we grade the honest finding, not a decimal.
    _ref = (nu.motion_index(raw), nu.motion_index(rigid), nu.motion_index(pwr))
    _match = (abs(mi_raw - _ref[0]) < 1e-6 and abs(mi_rigid - _ref[1]) < 1e-6
              and abs(mi_pwr - _ref[2]) < 1e-6)
    _order = (mi_pwr < mi_rigid < mi_raw)
    _red_pwr = (mi_raw - mi_pwr) / mi_raw if mi_raw else 0.0
    _band = 0.03 <= _red_pwr <= 0.25
    _ok = bool(_match and _order and _band)
    _c = "#e8f5e9" if _ok else "#ffebee"
    _b = "#2e7d32" if _ok else "#c62828"
    _m0 = ("✅ your my_motion_index matches nu.motion_index to floating-point precision"
           if _match else
           "❌ your numbers do not match nu.motion_index — check the axis (should be 0) and the "
           "reduction (should be mean)")
    _m1 = ("✅ ordering holds: MI(pw-rigid) &lt; MI(rigid) &lt; MI(raw)  "
           f"({mi_pwr:.3f} &lt; {mi_rigid:.3f} &lt; {mi_raw:.3f})") if _order else (
           f"❌ ordering broken: raw={mi_raw:.3f}, rigid={mi_rigid:.3f}, pw-rigid={mi_pwr:.3f} — "
           "registration should lower MI")
    _m2 = (f"✅ pw-rigid removes {100 * _red_pwr:.1f}% of raw motion — a real reduction, in band"
           if _band else
           f"⚠️ pw-rigid removes {100 * _red_pwr:.1f}% of raw motion — outside the 3–25% band; "
           "check you passed the right panels")
    _head = "PASS — you rebuilt the motion index and it confirms registration works" if _ok else \
            "Not yet — fix the flagged part"
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_b};padding:12px 16px;border-radius:6px">
        <b style="color:{_b}">{_head}</b><br>
        {_m0}<br>{_m1}<br>{_m2}<br>
        <span style="font-size:0.9em;color:#555">Graded on the honest conclusion — agreement with the
        library, the ordering, and a tolerance band (3–25% reduction) — not an exact motion-index value,
        since that scales with subsampling.</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Reference: the algorithm and its limits": mo.md(
            r"""
            **Where these methods come from.** The rigid / piecewise-rigid split shown in this movie is
            the design of **NoRMCorre** (Pnevmatikakis & Giovannucci, *"NoRMCorre: An online algorithm
            for piecewise rigid motion correction of calcium imaging data,"* **J. Neurosci. Methods**
            291:83–94, 2017), the motion-correction stage of the widely used **CaImAn** pipeline
            (Giovannucci et al., *eLife* 2019). NoRMCorre estimates a reference template, aligns each
            frame to it by **phase-correlation** (which reads a global translation off the peak of the
            cross-correlation between a frame and the template — the rigid step), then refines the
            result with **per-patch** shifts that are smoothly interpolated back together (the
            piecewise-rigid step) to handle non-uniform tissue motion. That is exactly the raw → rigid →
            pw-rigid progression in this movie.

            **The shared idea with Week 1.** Both pose stabilization and miniscope motion correction are
            *registration*: estimate a transform (a single translation, or a field of local
            translations) that best maps one time sample onto a reference, then apply it so that later
            measurements are made in a stable frame.

            **The limits of the analogy and of the metric.** Pose tracking registers a handful of
            *labeled* keypoints whose identity is known. Motion correction registers *dense* pixel
            intensities with no labels, and must build its own reference template, so it can be misled by
            real brightness changes — a neuron firing looks a little like the frame moving. And a lower
            motion index is *necessary but not sufficient*: overly aggressive piecewise warping can lower
            the motion index while smearing the real signal, so full pipelines also check a correlation
            image and residual traces, not the motion index alone.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Summary — the answer, and the next question

        **The question we asked:** to read neurons from a movie, how do we first hold the image still?

        **The answer we reached:** by **registration**. We estimate how far each frame has moved
        relative to a reference template and shift it back. **Rigid** registration removes global drift
        with one translation per frame; **piecewise-rigid** registration removes non-uniform motion with
        per-patch shifts. We confirmed the fix with a single number, the **motion index**, whose ordering
        `MI(pw-rigid) < MI(rigid) < MI(raw)` we both saw in the per-frame distribution and rebuilt by
        hand — and we validated that number against injected ground-truth jitter. A registered movie is a
        stack of frames in which pixel `(y, x)` refers to the same tissue over time.

        **The next question (NB07):** now that the pixels hold still, a fixed ROI finally reads the same
        cell across the whole recording — but the movie is still just brightness, not neurons. *How do we
        turn a stabilized movie into one clean trace per neuron* — separating each cell's calcium signal
        from the background and from its overlapping neighbors? That is source extraction and demixing,
        and it is where NB07 goes next.
        """
    )
    return


if __name__ == "__main__":
    app.run()
