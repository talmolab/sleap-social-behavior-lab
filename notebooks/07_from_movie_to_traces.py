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
    return CACHE, ROOT, go, np, nu, os


@app.cell
def _(ROOT, np, os):
    # The committed bundle: every array the figures need, precomputed by tools/build_nb07_assets.py
    # (motion-correction panels, striatum scrubber frames, CNMF footprints/traces). Loading it means
    # the notebook NEVER downloads from Google Drive / eLife at runtime — exactly like the behavior
    # notebooks load data/train_events.npz. Falls back to a one-time pull of the committed file only
    # if a bare checkout is missing it (same pattern as cu.data_path); no external-host fetch.
    _p = os.path.join(ROOT, "data", "nb07_assets.npz")
    if not os.path.exists(_p):
        import urllib.request as _urlreq
        _RAW = os.environ.get("COURSE_REPO_RAW",
            "https://raw.githubusercontent.com/talmolab/sleap-social-behavior-lab/main")
        os.makedirs(os.path.dirname(_p), exist_ok=True)
        _urlreq.urlretrieve(_RAW + "/data/nb07_assets.npz", _p)
    NB07 = dict(np.load(_p, allow_pickle=False))
    return (NB07,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # NB7 · From a movie to traces — the neural twin of NB1–2

        ## Where Week 1 left us, and what Week 2 adds

        Across Week 1 we built an objective readout of *what a mouse does*. We started from a raw
        recording of two interacting mice, ran it through a pose tracker to get **keypoints** (the
        labelled body parts of each animal at each frame), audited how reliable those points are
        (NB1), rebuilt each animal's motion in its own body frame and turned it into **features**
        (NB2), and went on to compress, cluster, and decode that behaviour. The through-line was a
        single sentence: *turn a movie into numbers we can do statistics on.*

        Week 2 studies the **neural basis** of that behaviour. We record the brain while the animal
        behaves, and the recording is, once again, a movie — this time of glowing brain tissue. So
        the very first job is the same one we already solved for pose: turn a movie into numbers.
        This notebook asks:

        > **How do we go from a raw microscope movie of a behaving mouse's brain to one clean
        > activity trace per neuron?**

        We answer it in three stages, and each stage rhymes with something we already did on
        behaviour:

        1. **Hold the image still** (motion correction) — the neural echo of NB1's tracking-and-
           identity problem: a measurement is meaningless until the thing you are measuring stops
           sliding around.
        2. **Read one cell by hand** (background subtraction + a hand-drawn region of interest) — the
           minimal, honest version of extracting a signal, and a lesson in exactly why it is not
           enough.
        3. **Demix the whole population** (CNMF) — learn a *shape* and a *trace* for every neuron at
           once, the neural echo of NB2's move from raw pixels to structured features.

        ### What calcium imaging is, in plain terms

        Some terms first, because the rest of the notebook is built on them.

        - **Neuron firing → calcium.** When a neuron fires, calcium ions flood into the cell and the
          internal calcium concentration spikes, then decays over a fraction of a second. Calcium is a
          stand-in for activity: more calcium means the cell was more active, a moment ago.
        - **GCaMP.** A protein sensor expressed inside neurons that **glows brighter when it binds
          calcium**. Because calcium tracks firing, a GCaMP cell's brightness is a proxy for its
          activity. Filming that glow is **calcium imaging**.
        - **Miniscope.** A miniature microscope bolted to the head of a freely moving mouse, filming a
          patch of brain tissue as a video while the animal behaves.
        - **Calcium trace.** The thing we want at the end: one number per frame describing how bright a
          single cell is over time. Its sharp, asymmetric bumps — fast rise, slow decay — are
          **calcium transients**, each one a burst of firing.
        - **Region of interest (ROI).** A small patch of pixels we pick out of the image. Averaging the
          pixels inside it at each frame turns an image stack into a single trace.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The same mathematics as the behaviour arm — and where it stops being the same

        This notebook is a **twin** of NB1–2. It is worth being precise about where the analogy holds
        and where it breaks, because the break is the whole reason the neural side is harder.

        | | Behaviour (NB1–2) | Neural (this notebook) |
        |---|---|---|
        | The raw object | a movie of two mice | a movie of glowing brain tissue |
        | What we extract | keypoints → a 19-column feature matrix | footprints → one calcium trace per neuron |
        | The core operation | write the data as structure × time | write the movie as **space × time**, `Y ≈ A·C` |
        | Alignment problem | keep an animal's *identity* stable across frames | keep a *pixel* pointing at the same tissue across frames |
        | **What is given to you** | **15 named keypoints, fixed and known** | **an unknown number of cells, unlabelled, overlapping** |
        | Pooling across animals | one fixed 19-column matrix, poolable across cages | per-session, variable cell count, **not** poolable |

        The last two rows are the real difference. In Week 1 the tracker *hands* you 15 keypoints with
        known names; the same 19 feature columns describe every animal, so you can stack thousands of
        events into one matrix. Here nothing hands you the "nodes." The number of neurons is unknown,
        they are not labelled, and their light lands on **overlapping** pixels, so a single pixel is a
        blend of several cells. We have to **discover** the cells and **separate** them, per recording.
        That discovery-and-separation step — **demixing** — has no counterpart in the pose pipeline,
        and it is where most of this notebook goes.

        The question this notebook answered before it (NB6, in the behaviour arm) was *can we decode
        what a mouse is doing from its movement?* The question it hands forward (NB8) is *once we have
        a population of neurons, what does each one encode, and can we read behaviour back out of
        them?* First we have to build that population. Let us start where every recording starts: with
        a movie that will not hold still.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # Part A — holding the image still (motion correction)

        ## A1. Why a microscope movie needs correcting at all

        **Why.** The miniscope is bolted to the animal's head, so every time the mouse walks, rears, or
        turns, the whole **field of view** (FOV — the patch of tissue the camera sees) lurches. When it
        lurches, pixel `(80, 40)` in frame 500 and pixel `(80, 40)` in frame 501 are no longer looking
        at the same piece of tissue. Read the brightness at a fixed pixel and you get a mixture of *"a
        neuron fired here"* and *"a bright cell just slid into this pixel,"* with no way to tell them
        apart. A neuron's trace is not even well-defined until the frames are aligned.

        **Definition — motion correction (a.k.a. registration).** Estimate, for every frame, how far
        the image has moved relative to a fixed **reference template**, then shift the frame back so
        each pixel lines up with the same tissue across the whole recording. This is exactly the
        discipline from NB1, where a pose track was meaningless until we stabilised *identity* — until
        "mouse 2 at frame 500" was guaranteed to be the same animal as "mouse 2 at frame 501." Aligning
        a signal before reading it is a general move; imaging needs it just as much as tracking does.

        **Two levels of ambition, both worth naming:**

        - **Rigid registration.** Shift the *whole* frame by a single translation (one up/down and one
          left/right offset). Fixes **global drift**, where the entire FOV slides together. Cannot fix
          motion that differs across the frame.
        - **Piecewise-rigid registration.** Cut the frame into patches, shift *each patch on its own*,
          and blend the shifts back together smoothly. Because patches can move by different amounts,
          this fixes **non-uniform motion** — soft brain tissue deforming under a moving lens.

        We work with one real miniscope movie processed three ways and laid side by side —
        `raw | rigid | pw-rigid` — so we can compare an uncorrected recording against both methods
        directly. First, plain EDA: what is in this file?
        """
    )
    return


@app.cell
def _(NB07):
    # The side-by-side motion-correction movie, precomputed into the committed bundle and split into
    # the three panels raw | rigid | pw-rigid (each (F, H', w) grayscale). Stored uint8 (the source is
    # 8-bit, so this is lossless); we cast to float32 for the analysis below. No runtime download.
    raw = NB07["moco_raw"].astype("float32")               # (F, H', w) grayscale
    rigid = NB07["moco_rigid"].astype("float32")
    pwr = NB07["moco_pwr"].astype("float32")
    F_mov, H_mov, W_mov = raw.shape
    return F_mov, H_mov, W_mov, pwr, raw, rigid


@app.cell(hide_code=True)
def _(F_mov, H_mov, NB07, W_mov, mo):
    _fps = float(NB07["moco_fps"]); _size = tuple(int(v) for v in NB07["moco_size"])
    _dur = float(NB07["moco_duration"]); _movshape = tuple(int(v) for v in NB07["moco_mov_shape"])
    mo.md(
        f"""
        **What we loaded.** The raw file reports `fps = {_fps}`, frame size
        `{_size}` (the full three-panel image), duration `{_dur}`
        s. After subsampling and splitting into thirds we hold **{F_mov} frames** of each
        panel, each **{H_mov} rows × {W_mov} columns**. The combined loaded movie is `{_movshape}`;
        the three panels sit left-to-right inside that width. Everything below reads from `raw`,
        `rigid`, and `pwr` — no more file loading.

        **The helpers, stated plainly.** `nu.read_video(path, step=3, gray=True)` reads a movie into a
        `(F, H, W)` float32 array, keeping every third frame. `nu.split_thirds(mov)` cuts the
        side-by-side frame into its three equal-width panels and crops the top/bottom so they align.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Where is the tissue, and where is the motion? Three projections

        Before playing the movie, we summarise it over time with **projection images** — each pixel
        collapsed across all frames into one value.

        - **Mean projection** (`raw.mean(axis=0)`): the average brightness at each pixel — the steady
          anatomy and the optics' uneven illumination.
        - **Std projection** (`raw.std(axis=0)`): how much each pixel *varies* over time. Constant
          pixels are dark; pixels that change — because a neuron fired there, **or because the image
          slid across them** — are bright. In an uncorrected movie the std projection lights up along
          edges precisely where motion drags bright structure back and forth.
        - **Motion-variance difference** (`raw.std(0) − pwr.std(0)`): the *new* panel. Subtract the
          corrected movie's std projection from the raw one. What is left is the variance that
          registration *removed* — a direct picture of where the motion lived. Bright ridges here are
          high-contrast edges that jittered in the raw movie and steadied after correction.
        """
    )
    return


@app.cell
def _(mo, np, nu, pwr, raw):
    _mean_img = raw.mean(axis=0)
    _std_img = raw.std(axis=0)
    _diff_img = raw.std(axis=0) - pwr.std(axis=0)   # variance registration removed
    _f_mean = nu.image_fig(_mean_img, title="raw — mean projection (anatomy)",
                           colorscale="gray", height=330)
    _f_std = nu.image_fig(_std_img, title="raw — std projection (variance, incl. motion)",
                          colorscale="Viridis", colorbar_title="std", height=330)
    # symmetric diverging scale centred on 0 so 'removed' vs 'added' variance read clearly
    _m = float(np.percentile(np.abs(_diff_img), 99))
    _f_diff = nu.image_fig(_diff_img, title="std(raw) − std(pw-rigid): the motion that was removed",
                           colorscale="RdBu_r", zmin=-_m, zmax=_m, colorbar_title="Δstd", height=330)
    mo.hstack([_f_mean, _f_std, _f_diff])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## A2. One movie, three registrations, and why a fixed pixel lies

        Comparing the three versions of the *same* recording at the same instant is the clearest way to
        see what each method does. Drag the frame slider and fix your eye on a single bright cell: in
        **raw** it wanders, in **rigid** it steadies, in **pw-rigid** it holds most still.

        A single frame is a weak test — motion lives in how the image changes *over time* — so below
        the movie we read a **fixed edge ROI's brightness over time** from all three panels. We do not
        let you drag this box around: we place it on a **pre-vetted, high-contrast edge**, because that
        is where the lesson is visible. (A box on a flat, featureless patch barely moves in any panel,
        which would teach nothing.) On a bright edge, a tiny image shift changes the pixel value a lot,
        so the **raw** trace jumps and dips — those excursions are *motion masquerading as activity* —
        while **rigid** and especially **pw-rigid** stay on the same tissue and read smoother.
        """
    )
    return


@app.cell
def _(F_mov, mo):
    frame_ctrl = mo.ui.slider(0, F_mov - 1, value=F_mov // 2, step=1,
                              label="frame", debounce=True, full_width=True)
    return (frame_ctrl,)


@app.cell
def _(H_mov, W_mov, frame_ctrl, go, mo, np, nu, pwr, raw, rigid):
    from plotly.subplots import make_subplots as _mksub
    # shared gray range across panels
    _vmin = float(np.percentile(raw, 1)); _vmax = float(np.percentile(raw, 99))
    # curated high-contrast edge ROI (fixed): a bright cell edge near frame centre
    _ry, _rx, _r = H_mov // 2 - 12, W_mov // 2 + 10, 4
    _t = frame_ctrl.value
    _panels = [("raw", raw), ("rigid", rigid), ("pw-rigid", pwr)]
    _fig = _mksub(rows=1, cols=3, horizontal_spacing=0.02,
                  subplot_titles=[n for n, _ in _panels])
    for _j, (_name, _p) in enumerate(_panels, start=1):
        _fig.add_trace(go.Heatmap(z=_p[_t], colorscale="gray", zmin=_vmin, zmax=_vmax,
                                  showscale=(_j == 3), colorbar=dict(title="gray", len=0.9)),
                       row=1, col=_j)
        _ax = "" if _j == 1 else str(_j)
        _fig.update_yaxes(autorange="reversed", scaleanchor="x" + _ax, scaleratio=1,
                          visible=False, row=1, col=_j)
        _fig.update_xaxes(visible=False, row=1, col=_j)
        # mark the fixed ROI in every panel
        _fig.add_shape(type="rect", x0=_rx - _r, x1=_rx + _r, y0=_ry - _r, y1=_ry + _r,
                       line=dict(color="#00e5ff", width=2), row=1, col=_j)
    _fig.update_layout(template="plotly_white", height=320,
                       margin=dict(l=10, r=10, t=52, b=10),
                       title=f"frame {_t}  —  raw | rigid | pw-rigid  (cyan = fixed edge ROI)")

    # the fixed-ROI trace over time from all three panels
    _sl = (slice(None), slice(_ry - _r, _ry + _r + 1), slice(_rx - _r, _rx + _r + 1))
    _colors = {"raw": "#e45756", "rigid": "#4c78a8", "pw-rigid": "#54a24b"}
    _tr = go.Figure()
    for _name, _panel in [("raw", raw), ("rigid", rigid), ("pw-rigid", pwr)]:
        _tr.add_scatter(y=_panel[_sl].mean(axis=(1, 2)), mode="lines", name=_name,
                        line=dict(color=_colors[_name],
                                  width=1.8 if _name == "pw-rigid" else 1.1))
    _tr.update_xaxes(title="frame")
    _tr.update_yaxes(title="mean ROI brightness (gray)")
    nu.apply_house_style(_tr, title="fixed edge-ROI brightness over time — raw wanders, corrected steadies",
                         legend="below", height=300)
    mo.vstack([frame_ctrl, _fig, _tr])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## A3. Kymographs — turning motion into a picture

        A single ROI is one location. To judge motion across the whole image at once, we use a
        **kymograph**: take one horizontal line of the image (a single row of pixels at height `y`) and
        stack that line across every frame. The result is a 2-D image whose horizontal axis is
        **position along the row** and whose vertical axis is **time**. A feature that stays put traces
        a **straight vertical streak**; a feature that jitters traces a **wiggly** one. "Is the movie
        registered?" becomes "are the streaks straight?"

        Pick the row `y` and compare the panels: raw streaks wander, rigid straightens them, pw-rigid
        straightens them most. Try a row that runs through a bright cell — the straightening is
        clearest where there is high-contrast structure to follow. The fourth panel, **raw − pw-rigid**,
        subtracts the corrected kymograph from the raw one: it is flat grey where correction changed
        nothing and lights up in red/blue **exactly at the time-rows and edge-positions where the image
        jittered** — so as you drag `y`, this panel shows *how much* motion that particular line carried,
        which the three near-identical greyscale panels only hint at.
        """
    )
    return


@app.cell
def _(H_mov, mo):
    row_ctrl = mo.ui.slider(0, H_mov - 1, value=min(80, H_mov - 1), step=1,
                            label="kymograph row y (which image line to track)",
                            debounce=True, full_width=True)
    return (row_ctrl,)


@app.cell
def _(mo, np, nu, pwr, raw, rigid, row_ctrl):
    from plotly.subplots import make_subplots as _mksub
    import plotly.graph_objects as _kgo
    _y = row_ctrl.value
    _kymos = [("Raw", raw[:, _y, :]), ("Rigid", rigid[:, _y, :]), ("PW-Rigid", pwr[:, _y, :])]
    _fig = _mksub(rows=1, cols=4, horizontal_spacing=0.035,
                  subplot_titles=[n for n, _ in _kymos] + ["Raw − PW-Rigid (what moved)"])
    for _j, (_name, _k) in enumerate(_kymos, start=1):
        _fig.add_trace(_kgo.Heatmap(z=_k, colorscale="gray", showscale=False), row=1, col=_j)
        _fig.update_yaxes(autorange="reversed", title="time (frames)" if _j == 1 else None,
                          showgrid=False, row=1, col=_j)
        _fig.update_xaxes(title="position (px)", showgrid=False, row=1, col=_j)
    # difference panel: raw - pw-rigid, symmetric diverging scale so 'removed' motion reads clearly
    _diff = raw[:, _y, :] - pwr[:, _y, :]
    _m = float(np.percentile(np.abs(_diff), 99)) or 1.0
    _fig.add_trace(_kgo.Heatmap(z=_diff, colorscale="RdBu_r", zmin=-_m, zmax=_m,
                                showscale=True, colorbar=dict(title="Δ", len=0.9)), row=1, col=4)
    _fig.update_yaxes(autorange="reversed", showgrid=False, row=1, col=4)
    _fig.update_xaxes(title="position (px)", showgrid=False, row=1, col=4)
    _fig.update_layout(template="plotly_white", height=440, margin=dict(l=10, r=10, t=46, b=10),
                       title=f"kymographs at row y = {_y} — straight streaks = registered; "
                             "colour in panel 4 = the jitter correction removed")
    mo.vstack([row_ctrl, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## A4. The motion index — one number for how much jitter is left

        Judging streaks by eye does not scale to thousands of movies. For that we need a single number.

        **Definition — motion index.** `nu.motion_index(frames)` is the mean absolute **frame-to-frame
        difference**,

        $$\text{MI} = \big\langle\,|f_{t+1} - f_{t}|\,\big\rangle,$$

        the average, over every pixel and every neighbouring pair of frames, of how much a pixel's
        brightness changed. A perfectly still movie has MI near zero (consecutive frames are nearly
        identical); a jittery movie has a large MI (every pixel keeps changing as the image slides). If
        registration works it should push MI down, and the more flexible method should push it lower
        still:

        $$\text{MI}(\text{raw}) \;>\; \text{MI}(\text{rigid}) \;>\; \text{MI}(\text{pw-rigid}).$$

        **Why this score and not another?** Two alternatives are equally natural, and naming their blind
        spots is the point. *Correlation-to-template* — correlate every frame with the reference and
        call the movie steady when that stays high — needs a *good* template you do not yet have, and
        rewards a frame for matching on average even while it warps locally. *Crispness* — the sharpness
        of the mean projection, since motion blurs edges — is blind to slow, coherent drift that keeps
        every frame sharp yet mis-aligned. Frame-to-frame difference is **template-free** and cheap, and
        it reads exactly the residual jitter you can see wiggling in the kymographs; its own blind spot —
        a firing cell also changes a pixel, so activity can masquerade as motion — is why full pipelines
        cross-check a correlation image too (the reference note at the end of Part A unpacks all three).

        The per-frame trace below plots `nu.motion_index_trace` (a value per adjacent frame pair) for
        all three versions.
        """
    )
    return


@app.cell
def _(mo, np, nu, pwr, raw, rigid):
    _tr_raw = nu.motion_index_trace(raw)
    _tr_rig = nu.motion_index_trace(rigid)
    _tr_pwr = nu.motion_index_trace(pwr)
    _mi = {k: nu.motion_index(v) for k, v in [("raw", raw), ("rigid", rigid), ("pw-rigid", pwr)]}
    _fig = nu.trace_fig(None, np.stack([_tr_raw, _tr_rig, _tr_pwr], axis=1),
                        names=["raw", "rigid", "pw-rigid"],
                        xlabel="frame", ylabel="mean |Δ| to previous frame",
                        title="per-frame motion index (lower = more stable)", height=300)
    _cols = ["#e45756", "#4c78a8", "#54a24b"]
    for _j, _c in enumerate(_cols):
        _fig.data[_j].line.color = _c
        _fig.data[_j].line.width = 1.6 if _j == 2 else 1.0
    nu.apply_house_style(_fig, legend="below", height=300)
    _msg = mo.md(
        f"""
        Whole-movie motion index: **raw = {_mi['raw']:.2f}**, **rigid = {_mi['rigid']:.2f}**,
        **pw-rigid = {_mi['pw-rigid']:.2f}**. Rigid removes
        **{100 * (_mi['raw'] - _mi['rigid']) / _mi['raw']:.0f}%** of the raw movie's frame-to-frame
        motion; piecewise-rigid removes **{100 * (_mi['raw'] - _mi['pw-rigid']) / _mi['raw']:.0f}%**,
        because it also catches the local warping a single whole-frame shift cannot.
        """
    )
    mo.vstack([_fig, _msg])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Does the motion index really measure motion? A ground-truth check

        Before trusting the motion index to *rank* the three panels, confirm it tracks motion we
        control. Take one clean frame and build fake movies by shifting it by known random offsets of
        increasing amplitude. Two curves come out:

        - **jittered** — as the injected shift grows (in pixels), MI rises. Good: the metric responds
          to motion.
        - **perfectly corrected** — roll each frame back by the *exact* known shift and every frame
          returns to the clean reference; consecutive frames become identical and MI collapses to ~0 at
          all amplitudes.

        That flat-zero curve is registration in its idealised form: knowing the true shift and undoing
        it. Real registration *estimates* the shift from the images, so rigid and pw-rigid land between
        the raw movie and this ideal. (Note the simplification: we shift with `np.roll`, which *wraps*
        pixels around the edge rather than revealing new tissue, so this is a clean teaching model, not
        the exact operation a real pipeline performs.)
        """
    )
    return


@app.cell
def _(go, np, nu, raw):
    _ref = raw[raw.shape[0] // 2]
    _rng = np.random.RandomState(0)
    _amps = np.arange(0, 9)
    _mi_jit, _mi_fix = [], []
    for _a in _amps:
        _n = 40
        _dy = _rng.randint(-_a, _a + 1, size=_n) if _a > 0 else np.zeros(_n, int)
        _dx = _rng.randint(-_a, _a + 1, size=_n) if _a > 0 else np.zeros(_n, int)
        _jit = np.stack([np.roll(np.roll(_ref, int(y), 0), int(x), 1) for y, x in zip(_dy, _dx)])
        _fix = np.stack([np.roll(np.roll(_j, -int(y), 0), -int(x), 1)
                         for _j, y, x in zip(_jit, _dy, _dx)])
        _mi_jit.append(nu.motion_index(_jit)); _mi_fix.append(nu.motion_index(_fix))
    _fig = go.Figure()
    _fig.add_scatter(x=_amps, y=_mi_jit, mode="lines+markers", name="jittered",
                     line=dict(color="#e45756", width=2))
    _fig.add_scatter(x=_amps, y=_mi_fix, mode="lines+markers", name="perfectly corrected",
                     line=dict(color="#54a24b", width=2))
    _fig.update_xaxes(title="injected shift amplitude (px)")
    _fig.update_yaxes(title="motion index")
    nu.apply_house_style(_fig, title="motion index vs injected jitter (ground truth)",
                         legend="below", height=340)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## A5. Comparing the three panels *honestly* — the pairing matters

        It is tempting to throw the three clouds of per-frame motion values into three side-by-side
        boxes and eyeball which sits lower. That would throw away the most important fact about this
        data: **the three panels are the same frames**. Frame pair 57 in `raw`, `rigid`, and `pwr` is
        the *same instant of the recording*, processed three ways. When the animal lunges, all three
        panels spike together. So the honest question is not "is the raw cloud higher than the pwr
        cloud on average?" but **"frame by frame, does pw-rigid beat raw more often than not, and is
        the per-frame difference reliably positive?"** That is a **paired** comparison.

        Two views make the pairing visible:

        - **Slopegraph.** A handful of evenly-spaced frames, each drawn as one connected line across
          `raw → rigid → pw-rigid`. Most lines slope *down*: that same frame got steadier at each
          stage. A few tick up — correction is not free everywhere — but the trend is clear.
        - **Paired-difference histogram.** For every frame pair, the difference `MI(raw) − MI(pw-rigid)`.
          A distribution sitting to the right of the red zero line means raw was almost always jitterier
          than pw-rigid. The **Wilcoxon signed-rank test** — the paired, non-parametric test — reads the
          *signs and ranks* of those differences and returns how surprising the shift is under the null
          "correction does nothing." A boxplot could not see this; it forgets which frame is which.
        """
    )
    return


@app.cell
def _(mo, np, nu, pwr, raw, rigid):
    _tr = np.stack([nu.motion_index_trace(raw), nu.motion_index_trace(rigid),
                    nu.motion_index_trace(pwr)], axis=1)                 # (F-1, 3)
    # slopegraph on ~15 evenly-spaced frames so the pairing is readable (not a hairball)
    _idx = np.linspace(0, _tr.shape[0] - 1, 15).astype(int)
    _slope = nu.slopegraph_fig(_tr[_idx], ["raw", "rigid", "pw-rigid"],
                               ylabel="per-frame motion index",
                               title="15 sample frames, tracked across the three stages", height=430)
    # paired-difference histogram (raw - pwr) with Wilcoxon
    _pd = nu.paired_diff_fig(_tr[:, 0] - _tr[:, 2], kind="hist",
                             xlabel="MI(raw) − MI(pw-rigid), per frame pair",
                             title="per-frame improvement (right of 0 = pw-rigid steadier)", height=430)
    mo.hstack([_slope, _pd], widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(mo, np, nu, pwr, raw, rigid):
    from scipy.stats import wilcoxon as _wil
    _d = nu.motion_index_trace(raw) - nu.motion_index_trace(pwr)
    _w, _p = _wil(nu.motion_index_trace(raw), nu.motion_index_trace(pwr))
    _win = float((_d > 0).mean())
    mo.md(
        f"""
        **The numbers.** Across all **{len(_d)}** frame pairs, pw-rigid is steadier than raw on
        **{100 * _win:.0f}%** of them (median per-frame improvement **{np.median(_d):.2f}** gray
        units), Wilcoxon signed-rank **p = {nu.fmt_p(_p)}**. A boxplot would have shown three
        overlapping clouds and hidden this; the paired test sees the reliable frame-by-frame shift. You
        will reproduce this number yourself in the exercise below.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## A6. Where does the shift come from? A phase-correlation mini-demo

        The motion index *scores* how much jitter is left, but it does not say how a registration
        algorithm *finds* the shift to undo. The classic trick is **phase correlation**, and it is
        short enough to show in full.

        **The idea.** If frame `B` is frame `A` shifted by `(dy, dx)`, then in the Fourier domain that
        shift is a pure phase ramp. Compute the **cross-power spectrum**
        `R = (FFT(A) · conj(FFT(B))) / |FFT(A) · conj(FFT(B))|`, invert it back to image space, and you
        get a surface that is nearly zero everywhere except for **one sharp peak**. The location of that
        peak *is* the shift between the two frames. Find the peak, shift the frame back by it, and the
        frame is registered — exactly the rigid step.

        Below we take one clean frame, shift it by a known `(dy, dx)`, and recover the shift from the
        peak. The peak sits off-centre by precisely the shift that undoes the injected motion — the
        metric (motion index) and the mechanism (find-the-peak) are two ends of the same idea.
        """
    )
    return


@app.cell
def _(go, mo, np, nu, raw):
    _ref = raw[raw.shape[0] // 2]
    _H, _W = _ref.shape
    _dy, _dx = 6, -4                                   # the shift we inject (and will recover)
    _shift = np.roll(np.roll(_ref, _dy, 0), _dx, 1)
    # a 2-D Hann window tapers the frame edges so the FFT doesn't leak (the way real registration does)
    _win = np.hanning(_H)[:, None] * np.hanning(_W)[None, :]
    _Fa = np.fft.fft2(_ref * _win); _Fb = np.fft.fft2(_shift * _win)
    _R = _Fa * np.conj(_Fb); _R = _R / (np.abs(_R) + 1e-9)
    _corr = np.fft.ifft2(_R).real
    _peak = np.unravel_index(np.argmax(_corr), _corr.shape)
    _ry = _peak[0] if _peak[0] <= _H // 2 else _peak[0] - _H    # unwrap to signed shift
    _rx = _peak[1] if _peak[1] <= _W // 2 else _peak[1] - _W
    _peakv = float(_corr.max()); _second = float(np.sort(_corr.ravel())[-2])
    # peak & zero-shift centre in the fft-shifted surface (centre = no shift)
    _cy, _cx = _H // 2, _W // 2
    _psy = (_peak[0] + _H // 2) % _H; _psx = (_peak[1] + _W // 2) % _W
    # The peak lands only a few pixels from the centre of a 160x160 surface, so at full scale it is a
    # single bright pixel lost in black — and a filled marker on top of it would hide the ONE
    # informative pixel entirely. Zoom to a window around the centre so the bright pixel is actually
    # visible, mark it with an OPEN ring (bright pixel shows through), and draw a zero-shift crosshair.
    _wd = 20
    _y0, _y1 = _cy - _wd, _cy + _wd + 1
    _x0, _x1 = _cx - _wd, _cx + _wd + 1
    _sh = np.fft.fftshift(_corr)
    _crop = _sh[_y0:_y1, _x0:_x1]
    _fig = go.Figure(go.Heatmap(z=_crop, x=np.arange(_x0, _x1), y=np.arange(_y0, _y1),
                                colorscale="Inferno", zmin=0.0, zmax=_peakv,
                                colorbar=dict(title="corr")))
    # dotted crosshair = zero-shift centre; the peak's offset from it IS the recovered shift
    _fig.add_hline(y=_cy, line=dict(color="#00e5ff", width=1, dash="dot"))
    _fig.add_vline(x=_cx, line=dict(color="#00e5ff", width=1, dash="dot"))
    # OPEN ring beside/around the peak so the bright pixel is not occluded
    _fig.add_scatter(x=[_psx], y=[_psy], mode="markers", showlegend=False,
                     marker=dict(color="#00e5ff", size=24, symbol="circle-open", line=dict(width=2.5)))
    _fig.add_annotation(x=_psx, y=_psy, ax=32, ay=-32, text="peak",
                        font=dict(color="#00e5ff", size=13), arrowcolor="#00e5ff", arrowwidth=1.5)
    nu.apply_house_style(
        _fig, title="phase-correlation surface (zoomed) — one bright peak marks the shift",
        legend=None, spatial=True, height=460)
    _fig.update_yaxes(autorange="reversed")
    # 1-D slices through the peak: the color scale blacks out the field, so cut a line through the
    # peak's row (varies x -> recovers dx) and column (varies y -> recovers dy). Each is flat at ~0
    # with a SINGLE spike — the "one sharp peak" the heatmap can only hint at, made unmistakable.
    _xoff = np.arange(_x0, _x1) - _cx
    _yoff = np.arange(_y0, _y1) - _cy
    _slice = go.Figure()
    _slice.add_scatter(x=_xoff, y=_sh[_psy, _x0:_x1], mode="lines", name="row through peak (→ dx)",
                       line=dict(color="#e45756", width=2))
    _slice.add_scatter(x=_yoff, y=_sh[_y0:_y1, _psx], mode="lines", name="col through peak (→ dy)",
                       line=dict(color="#4c78a8", width=2))
    _slice.add_vline(x=0, line=dict(color="#888", width=1, dash="dot"))
    _slice.update_xaxes(title="offset from zero-shift centre (px)")
    _slice.update_yaxes(title="phase-correlation value")
    nu.apply_house_style(_slice, title="the same peak, sliced — one spike, flat everywhere else",
                         legend="below", height=460)
    mo.vstack([mo.hstack([_fig, _slice], widths=[1, 1]), mo.md(
        f"""**Injected shift** `(dy, dx) = ({_dy}, {_dx})`. The surface is flat (black) everywhere
        except **one sharp peak** (value **{_peakv:.2f}**, ringed in cyan on the left) — the next-highest
        pixel is only **{_second:.3f}**, so the peak is a **{_peakv / _second:.0f}× isolated spike**. It
        sits offset from the dotted zero-shift centre by `(dy, dx) = ({_ry}, {_rx})` — **exactly equal
        and opposite to the injected motion**, i.e. the shift that maps the moved frame back onto the
        reference. The 0–1 colour scale blacks out the near-zero field, so the panel on the right **cuts
        a 1-D line through the peak's row and column**: each is pinned at ~0 with a lone spike at offset
        `dx = {_rx}` (red) and `dy = {_ry}` (blue) — the sharpness the heatmap can only hint at. Find the
        peak, apply that shift, and the frame is registered. In a real pipeline this runs once per frame
        against the reference template; piecewise-rigid runs it once per patch."""
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## A7. Exercise 1 — rebuild the motion index, then test it as a pair

        **Python skill practised:** *array arithmetic and axis reductions, then calling a stats
        function.* You will difference an array along the time axis, reduce it to one number, and then
        run a paired test on the per-frame traces. Knowing *which axis is time* is the crux — the same
        idea powered every velocity feature in Week 1.

        **What you have.** `raw`, `rigid`, `pwr` — the three panels, each `(F, H, W)`; `np`; and
        `nu.motion_index_trace(panel)` which returns the `(F-1,)` per-frame motion trace. Fill the two
        blanks in `my_motion_index`, then compute the paired win-fraction and Wilcoxon p yourself.
        The answers are already written on the lines so the notebook runs end to end — cover them and
        try first.

        **What to expect.** Your three whole-movie numbers match `nu.motion_index` to floating-point
        precision and fall in order `mi_pwr < mi_rigid < mi_raw`, with pw-rigid removing roughly a tenth
        of the raw movie's motion; and your paired test recovers a win-fraction near **0.74** with a
        tiny p-value — the same result the section above reported.
        """
    )
    return


@app.cell
def _(np, pwr, raw, rigid, nu):
    from scipy.stats import wilcoxon as _wilcoxon
    # ------------------------------------------------------------------ YOUR CODE (edit this cell)
    def my_motion_index(frames):
        # `frames` is (F, H, W): F frames stacked along axis 0, the TIME axis.
        # BLANK 1 — difference along the time axis. np.diff(a, axis=k) subtracts each slice from the
        #   next ALONG axis k; frames are stacked on axis 0, so the time axis is 0. Wrong axis =
        #   measuring brightness change ACROSS the image, not motion over time.
        _diffs = np.diff(frames, axis=0)               # <-- replace ____ with 0 (the time axis)
        # BLANK 2 — reduce to one number. A pixel that brightens and one that darkens have BOTH moved,
        #   so take |·| first, then average over every pixel and frame pair. Without |·|, equal
        #   brightening and darkening cancel and a jittery movie can report ~0.
        return np.abs(_diffs).mean()                   # <-- replace ____ with mean
    # -----------------------------------------------------------------------------------------------
    mi_raw = my_motion_index(raw)
    mi_rigid = my_motion_index(rigid)
    mi_pwr = my_motion_index(pwr)

    # Now the PAIRED test: per-frame traces (given), then YOUR win-fraction + Wilcoxon.
    _d_raw = nu.motion_index_trace(raw)                 # (F-1,) per-frame motion, raw panel
    _d_pwr = nu.motion_index_trace(pwr)                 # (F-1,) per-frame motion, pw-rigid panel
    # BLANK 3 — fraction of frames where raw jitters MORE than pw-rigid: a boolean mask, then .mean().
    win_frac = (_d_raw > _d_pwr).mean()                 # <-- replace ____ with (_d_raw > _d_pwr)
    # BLANK 4 — the paired test. wilcoxon(a, b) compares the two traces frame-by-frame.
    _w, wil_p = _wilcoxon(_d_raw, _d_pwr)               # <-- replace ____ with _d_raw, _d_pwr
    return mi_pwr, mi_raw, mi_rigid, my_motion_index, win_frac, wil_p


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show solution": mo.md(
            r"""
            ```python
            def my_motion_index(frames):
                _diffs = np.diff(frames, axis=0)        # difference along the TIME axis -> (F-1,H,W)
                return np.abs(_diffs).mean()            # |change|, averaged over all pixels & pairs

            mi_raw, mi_rigid, mi_pwr = map(my_motion_index, (raw, rigid, pwr))  # ~5.50 / 5.10 / 4.93

            _d_raw, _d_pwr = nu.motion_index_trace(raw), nu.motion_index_trace(pwr)
            win_frac = (_d_raw > _d_pwr).mean()         # ~0.74
            _w, wil_p = wilcoxon(_d_raw, _d_pwr)        # p ~ 2e-17
            ```

            The three whole-movie numbers match `nu.motion_index` exactly (you rebuilt the same
            computation) and fall `mi_pwr < mi_rigid < mi_raw`. The paired test shows pw-rigid is
            steadier than raw on ~74% of frames, with a vanishing p-value. The *ordering* and the
            *pairing* are the results — the absolute motion-index values scale with the subsampling
            step, which is why we grade the conclusion, not a decimal.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mi_pwr, mi_raw, mi_rigid, mo, nu, pwr, raw, rigid, win_frac, wil_p):
    _ref = (nu.motion_index(raw), nu.motion_index(rigid), nu.motion_index(pwr))
    _match = (abs(mi_raw - _ref[0]) < 1e-6 and abs(mi_rigid - _ref[1]) < 1e-6
              and abs(mi_pwr - _ref[2]) < 1e-6)
    _order = (mi_pwr < mi_rigid < mi_raw)
    _red = (mi_raw - mi_pwr) / mi_raw if mi_raw else 0.0
    _paired = (0.60 <= float(win_frac) <= 0.90) and (float(wil_p) < 1e-6)
    _ok = bool(_match and _order and 0.03 <= _red <= 0.25 and _paired)
    _c = "#e8f5e9" if _ok else "#ffebee"; _b = "#2e7d32" if _ok else "#c62828"
    _m0 = ("✅ my_motion_index matches nu.motion_index to floating-point precision" if _match
           else "❌ numbers don't match — check the axis (0) and the reduction (mean)")
    _m1 = (f"✅ ordering holds: MI(pw-rigid) &lt; MI(rigid) &lt; MI(raw)  "
           f"({mi_pwr:.2f} &lt; {mi_rigid:.2f} &lt; {mi_raw:.2f}); pw-rigid removes {100*_red:.0f}%"
           if (_order and 0.03 <= _red <= 0.25) else
           f"❌ ordering/size off: raw={mi_raw:.2f}, rigid={mi_rigid:.2f}, pw-rigid={mi_pwr:.2f}")
    _m2 = (f"✅ paired test: pw-rigid steadier on {100*float(win_frac):.0f}% of frames, "
           f"p = {nu.fmt_p(wil_p)}" if _paired else
           f"❌ paired test off: win_frac={float(win_frac):.2f}, p={nu.fmt_p(wil_p)} — "
           "did you pass the raw and pw-rigid traces?")
    _head = "PASS — you rebuilt the metric and confirmed registration works, frame by frame" if _ok \
            else "Not yet — fix the flagged part"
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_b};padding:12px 16px;border-radius:6px">
        <b style="color:{_b}">{_head}</b><br>{_m0}<br>{_m1}<br>{_m2}<br>
        <span style="font-size:0.9em;color:#555">Graded on the honest conclusion — library agreement,
        the ordering (3–25% reduction), and the paired win-fraction ~0.74 with p &lt; 1e-6 — not an
        exact motion-index value, which scales with subsampling.</span></div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Reference — where these methods come from (NoRMCorre / CaImAn)": mo.md(
            r"""
            The rigid / piecewise-rigid split shown here is the design of **NoRMCorre** (Pnevmatikakis
            & Giovannucci, *J. Neurosci. Methods* 291:83–94, 2017), the motion-correction stage of the
            **CaImAn** pipeline (Giovannucci et al., *eLife* 2019). NoRMCorre estimates a reference
            template, aligns each frame to it by **phase correlation** (the peak-finding demo above —
            the rigid step), then refines with **per-patch** shifts smoothly interpolated back together
            (the piecewise-rigid step). That is exactly the raw → rigid → pw-rigid progression.

            **Why `mean|Δframe|` and not, say, correlation-to-template?** We score registration by how
            much the image *changes between consecutive frames*. Two natural alternatives exist.
            *Correlation-to-template*: correlate every frame with the reference and call the movie
            registered when that correlation is high. *Crispness*: measure the sharpness of the
            mean projection — motion blurs edges, so a sharper time-average means steadier frames. Each
            captures something real and each has a blind spot. Frame-to-frame difference is cheap and
            **template-free** (no reference needed) but confounds real activity with motion — a firing
            cell also changes a pixel. Correlation-to-template needs a *good* template and can reward a
            frame for matching it *on average* even while it is locally warped. Crispness sees blur but
            is blind to slow, coherent drift that keeps every frame sharp yet mis-aligned. We use
            frame-to-frame difference here because it needs no template and exposes exactly the residual
            jitter the reader can see wiggling in the kymographs.

            **A caveat on the metric.** For that same reason a lower motion index is *necessary but not
            sufficient*: overly aggressive piecewise warping can lower the motion index while smearing
            real signal, so full pipelines also check a correlation image and residual traces, not the
            motion index alone. And unlike pose tracking — which registers a handful of *labelled*
            keypoints — motion correction registers *dense, unlabelled* pixel intensities and must build
            its own template, so a firing neuron can look a little like the frame moving.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # Part B — reading one cell by hand (and why a box is not enough)

        ## B1. A raw miniscope movie, and cleaning it before we measure

        **Why.** The image is stable now, so a fixed pixel finally points at the same tissue over time.
        But the movie is still just brightness, not neurons — and before we reduce it we have to
        *clean* it. Real recordings carry burned-in junk: a **timestamp** stamped into a corner (here
        `Time = ...`) and **letterbox** bands where the sensor was padded to a square. These sit at the
        top of the intensity scale (near white), so they are the brightest things in the frame and
        compress the tissue's contrast into a narrow band. Worse, the timestamp *changes over time* (the
        clock ticks), so — unlike static anatomy — it **survives** the median-background removal we do
        next and would masquerade as a wildly active "cell." An ROI dropped on it reads a huge spurious
        trace. So we blank it *before* extracting anything.

        We use a different recording for Part B: a real **striatum** miniscope movie (the striatum is a
        deep motor/reward structure). The original stream is subsampled to a handful of evenly-spaced
        representative frames — enough to see structure, small enough for a bare kernel.

        **The hygiene helpers.** `nu.mask_region(frames, boxes)` paints rectangles to a constant
        (blanking the timestamp and letterbox; because each patch is now *static*, the later median
        subtraction erases it cleanly). `nu.crop_border(...)` trims fixed edge bands. Below: the raw
        mean projection — the timestamp and white letterbox bands are the brightest features — and the
        cleaned version with both blanked to black, so the whole colour range is spent on tissue.
        """
    )
    return


@app.cell
def _(NB07, nu):
    # Representative frames of the striatum miniscope movie, precomputed into the committed bundle
    # (stored uint8 -> lossless; cast to float32). We run the SAME hygiene + background-subtraction
    # helpers on them as before — only the source changed from a runtime download to the bundle.
    frames_raw = NB07["stri_frames_raw"].astype("float32")     # (n_sf, 500, 500) float32, 0..255
    # burned-in junk in THIS movie (found by EDA): a timestamp in the top-left and top/bottom
    # letterbox bands. Mask them to a constant so median-subtraction later erases them cleanly.
    _TS = (16, 40, 74, 150)                                     # timestamp box (y0,y1,x0,x1)
    _LB_TOP = (0, 10, 0, 500); _LB_BOT = (490, 500, 0, 500)     # letterbox bands
    frames_clean = nu.mask_region(frames_raw, [_TS, _LB_TOP, _LB_BOT])
    n_sf = int(frames_raw.shape[0])
    return frames_clean, frames_raw, n_sf


@app.cell
def _(frames_clean, frames_raw, mo, nu):
    # raw mean projection: force robust=False so the burned-in border saturates the scale (the bug)
    _f_raw = nu.image_fig(frames_raw.mean(0), title="raw mean projection — border saturates the scale",
                          colorscale="gray", robust=False, colorbar_title="intensity", height=430)
    _f_cl = nu.image_fig(frames_clean.mean(0), title="after masking timestamp + letterbox — tissue shows",
                         colorscale="gray", colorbar_title="intensity", height=430)
    mo.hstack([_f_raw, _f_cl], widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## B2. Remove the static background (background subtraction + z-score)

        **Why.** Most of what you just saw is unchanging: uneven illumination, out-of-focus tissue, the
        lens. That static pattern says nothing about *when a neuron fires*. Removing it leaves only the
        part that changes over time — the part we care about.

        **Definitions.** *Background subtraction* estimates the static image and subtracts it.
        *z-score* rescales each pixel into units of its own standard deviation (subtract the mean,
        divide by the std), so pixels with different baseline brightness become comparable; `+3` means
        "3 std above this pixel's usual value."

        **Method.** `nu.background_subtract(frames)` takes the **median over time** at each pixel as its
        background (a pixel on inactive tissue looks the same in almost every frame, so its median *is*
        its background; a pixel that occasionally flares stays dark in most frames, so the flare
        survives), subtracts it, then z-scores each pixel. Input: the cleaned `(F, 500, 500)` stack.
        Output: the `(500, 500)` background `bg` and the `(F, 500, 500)` z-scored foreground `fg`.
        Left below is the removed background; right is one foreground frame on a ±3σ scale — the flat
        glow gone, short-lived bright spots (firing cells) standing out. The slider controls both.
        """
    )
    return


@app.cell
def _(frames_clean, nu):
    _bs = nu.background_subtract(frames_clean)
    fg = _bs["fg"]; bg = _bs["bg"]
    return bg, fg


@app.cell
def _(mo, n_sf):
    movie_t = mo.ui.slider(0, n_sf - 1, value=0, step=1,
                           label=f"frame (subsampled movie, 0–{n_sf - 1})", debounce=True,
                           full_width=True)
    return (movie_t,)


@app.cell
def _(bg, fg, mo, movie_t, nu):
    _bg_fig = nu.image_fig(bg, title="background (median over time)", colorscale="gray",
                           colorbar_title="intensity", height=430)
    _fg_fig = nu.image_fig(fg[movie_t.value], title=f"foreground frame {movie_t.value} (z-scored)",
                           colorscale="RdBu", zmin=-3, zmax=3, colorbar_title="σ", height=430)
    mo.vstack([movie_t, mo.hstack([_bg_fig, _fg_fig], widths=[1, 1])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## B3. Where are the cells? The maximum projection

        A single foreground frame only shows the cells firing at that instant. To decide where to place
        an ROI we want one image showing **every** location that was active at any point.

        **Definition — maximum projection.** At each pixel, the largest value it ever reached. We take
        the max of the *absolute* foreground over time, `np.abs(fg).max(axis=0)` (so both bright rises
        and dark dips count; `axis=0` is time). A pixel that ever flared appears bright; a pixel that
        stayed at baseline stays dark. This is our **map of candidate cells**.
        """
    )
    return


@app.cell
def _(fg, np):
    maxproj = np.abs(fg).max(axis=0)                            # (500, 500) map of active pixels
    return (maxproj,)


@app.cell
def _(maxproj, nu):
    nu.image_fig(maxproj, title="max |foreground| over time — bright spots are active cells",
                 colorscale="Inferno", colorbar_title="peak σ", height=520)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## B4. Reading one cell — and the trap in a hand-drawn box

        **Method.** An ROI trace is the average of the foreground inside a box at every frame,
        `fg[:, y0:y1, x0:x1].mean(axis=(1,2))`. `roi_trace(fg, cx, cy, r)` does exactly this: inputs a
        box centre `(cx, cy)` and half-width `r`; output a `(F,)` trace. Drag the box over the
        active-cell map: on a bright spot the trace shows sharp calcium transients; on dark background
        it looks like flat noise. (Image indexing is `[y, x]`, so `cx` moves horizontally and `cy`
        vertically.) The title reports the trace's **variance** — how much it fluctuates.
        """
    )
    return


@app.cell
def _():
    def roi_trace(fg_stack, cx, cy, r=10):
        y0, y1 = int(cy - r), int(cy + r)
        x0, x1 = int(cx - r), int(cx + r)
        return fg_stack[:, y0:y1, x0:x1].mean(axis=(1, 2))     # (F,) per-frame box average
    return (roi_trace,)


@app.cell
def _(mo):
    roi_cx = mo.ui.slider(20, 480, value=243, step=1, label="ROI center x (cx)",
                          debounce=True, full_width=True)
    roi_cy = mo.ui.slider(50, 450, value=349, step=1, label="ROI center y (cy)",
                          debounce=True, full_width=True)
    return roi_cx, roi_cy


@app.cell
def _(fg, maxproj, mo, nu, roi_cx, roi_cy, roi_trace):
    _cx, _cy, _r = roi_cx.value, roi_cy.value, 10
    _img = nu.image_fig(maxproj, title="active-cell map — drag the ROI box",
                        colorscale="Inferno", colorbar_title="peak σ", height=430)
    _img.add_shape(type="rect", x0=_cx - _r, y0=_cy - _r, x1=_cx + _r, y1=_cy + _r,
                   line=dict(color="#00e5ff", width=3))
    _tr = roi_trace(fg, _cx, _cy, _r)
    _trace = nu.trace_fig(None, _tr, title=f"ROI trace at ({_cx}, {_cy}) — variance = {_tr.var():.3f}",
                          xlabel="frame", ylabel="mean foreground (σ)", height=430)
    _trace.update_traces(line=dict(color="#444", width=1.3))
    mo.vstack([mo.hstack([roi_cx, roi_cy]), mo.hstack([_img, _trace], widths=[1, 1])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Does the box actually isolate a cell? Three kinds of ROI

        Dragging one box is suggestive, not evidence. If placing the box *is* the measurement, then
        boxes on cells, boxes on the dark gaps *between* cells, and boxes on the burned-in border should
        give quantitatively different traces. We test all three across many boxes at once and plot the
        raw per-box variances — the honest way to compare groups.

        - **cell** — 14 boxes on bright blobs on the map.
        - **tissue gap** — 14 boxes on the dimmest patches *inside* the imaged tissue (the spaces
          between visible cells).
        - **border** — 8 boxes on the burned-in timestamp we blanked in B1 (the only "non-tissue"
          reference available).

        Watch which groups separate. The result is not the tidy one the naive story promises — and that
        is the point.
        """
    )
    return


@app.cell
def _(fg, maxproj, np, roi_trace):
    # CELL: hand-placed on bright blobs. TISSUE-GAP: dimmest interior patches (between visible cells).
    # BORDER: boxes sitting entirely inside the timestamp we blanked in B1 -> constant -> variance 0.
    CELL_ROIS = np.array([(243, 349), (154, 338), (306, 96), (276, 96), (197, 63), (336, 111),
                          (224, 66), (393, 325), (124, 265), (129, 135), (394, 371), (310, 63),
                          (336, 65), (429, 137)])
    _cand = [(cx, cy, maxproj[cy - 5:cy + 5, cx - 5:cx + 5].mean())
             for cy in range(90, 410, 15) for cx in range(90, 410, 15)]
    GAP_ROIS = np.array([(c[0], c[1]) for c in sorted(_cand, key=lambda t: t[2])[:14]])
    BORDER_ROIS = np.array([(88, 27), (100, 28), (112, 27), (124, 28), (136, 27), (96, 29),
                            (120, 29), (108, 28)])
    def _var(cx, cy):
        return float(roi_trace(fg, cx, cy).var())
    cell_var_all = np.array([_var(cx, cy) for cx, cy in CELL_ROIS])
    gap_var_all = np.array([_var(cx, cy) for cx, cy in GAP_ROIS])
    border_var_all = np.array([_var(cx, cy) for cx, cy in BORDER_ROIS])
    return (BORDER_ROIS, CELL_ROIS, GAP_ROIS, border_var_all, cell_var_all, gap_var_all)


@app.cell
def _(BORDER_ROIS, CELL_ROIS, GAP_ROIS, border_var_all, cell_var_all, gap_var_all,
      maxproj, mo, np, nu):
    _img = nu.image_fig(maxproj, title="the ROI boxes (red=cell, blue=tissue gap, gray=border)",
                        colorscale="Inferno", colorbar_title="peak σ", height=470)
    for _set, _col in [(CELL_ROIS, "#e45756"), (GAP_ROIS, "#4c78a8"), (BORDER_ROIS, "#bbbbbb")]:
        for _cx, _cy in _set:
            _img.add_shape(type="rect", x0=_cx - 10, y0=_cy - 10, x1=_cx + 10, y1=_cy + 10,
                           line=dict(color=_col, width=2))
    _vals = np.concatenate([cell_var_all, gap_var_all, border_var_all])
    _grp = np.array(["cell"] * len(cell_var_all) + ["tissue gap"] * len(gap_var_all) +
                    ["border"] * len(border_var_all))
    _strip = nu.strip_points_fig(
        _vals, _grp, group_order=["cell", "tissue gap", "border"],
        colors={"cell": "#e45756", "tissue gap": "#4c78a8", "border": "#969696"},
        ylabel="ROI trace variance", title="cell ≈ tissue gap; only the border reads flat", height=470)
    mo.hstack([_img, _strip], widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(border_var_all, cell_var_all, gap_var_all, mo, np):
    _c = np.median(cell_var_all); _g = np.median(gap_var_all); _b = np.median(border_var_all)
    mo.md(
        f"""
        **What the numbers actually say.** Cell boxes have median variance **{_c:.2f}**. Tissue-gap
        boxes — the dark spaces *between* the visible cells — have median variance **{_g:.2f}**: a
        cell/gap ratio of only **{_c / _g:.2f}×**. A box on empty-looking tissue fluctuates almost as
        much as a box on a cell, because the whole tissue **co-fluctuates**: surrounding neuropil, blood
        flow, and overall brightness make the field breathe together, and a 20×20 box averages in all
        of it. The *only* boxes that read genuinely flat (variance ≈ **{_b:.2f}**) are the **border**
        boxes — and those sit on the burned-in timestamp we blanked in B1, which is **not tissue at
        all**.

        This resolves a contradiction you may have seen quoted two ways: a "6× cell-vs-background
        separation" and a "~3× separation" from the *same* movie. Both are real; they just used
        different "background." Measure a cell against the burned-in border or the dead vignette outside
        the tissue and you get a flattering 6×; measure it against honest tissue *between* cells and the
        ratio collapses to ~1×. **A hand-drawn rectangle cannot separate a cell from the tissue it sits
        in.** That failure is exactly the problem Part C solves: a method that learns each cell's *own
        compact shape* and pulls it out of the shared background.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## B5. Exercise 2 — measure the three-way gap yourself

        **Python skill practised:** *array slicing, axis reductions, and comparing groups.* You index a
        3-D stack with a box, collapse it to one number, and compare medians. `stack[:, y0:y1, x0:x1]`
        selects a sub-volume; `.mean(axis=(1,2))` averages the two spatial axes to a per-frame trace;
        `.var()` reduces that trace to one fluctuation score.

        **The claim to test.** A cell box carries transients (high variance). A border box on the
        burned-in timestamp we blanked reads flat (variance ≈ 0) — but it is not tissue. A tissue-gap
        box, *inside* the field, is **not** flat. Fill the three marked lines, one box per group.

        **What you should see.** `cell_var ≈ 0.92`, `gap_var ≈ 0.91` (nearly the same — the box can't
        tell them apart), `border_var ≈ 0` (flat, but it is the blanked artifact). The self-check passes
        when cell and gap are both high and *close*, and the border is low.
        """
    )
    return


@app.cell
def _(fg, roi_trace):
    # -------------------------------------------------------------- YOUR CODE (edit the 3 marked lines)
    # roi_trace(fg, cx, cy, r=10) -> (F,) box-average trace; float(trace.var()) -> one fluctuation #.
    # LINE 1 — a BRIGHT CELL at (243, 349) (worked example; leave as-is).
    cell_var = float(roi_trace(fg, cx=243, cy=349, r=10).var())
    # LINE 2 — a TISSUE GAP inside the field, e.g. (240, 240): a dim patch BETWEEN cells. WHY: if this
    #   comes out nearly as high as the cell, the box cannot isolate a cell from co-fluctuating tissue.
    gap_var = float(roi_trace(fg, cx=240, cy=240, r=10).var())
    # LINE 3 — a BORDER box ON the timestamp we blanked in B1, e.g. (110, 28). WHY: this is the only
    #   "flat" reference, and it isn't tissue — it is the artifact we masked to zero. Use (110, 28).
    border_var = float(roi_trace(fg, cx=110, cy=28, r=10).var())
    # ------------------------------------------------------------------------------------------------
    return border_var, cell_var, gap_var


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show solution": mo.md(
            r"""
            ```python
            cell_var   = float(roi_trace(fg, cx=243, cy=349, r=10).var())  # ~0.92  bright blob
            gap_var    = float(roi_trace(fg, cx=240, cy=240, r=10).var())  # ~0.91  dim inter-cell tissue
            border_var = float(roi_trace(fg, cx=110, cy=28,  r=10).var())  # ~0     the blanked timestamp
            ```

            The cell and the gap are almost the same (~1× apart): the tissue co-fluctuates, so the box
            can't isolate the cell. Only the border reads flat — and the border is the timestamp we
            blanked, not real background. The naive "cell vs flat background = 6×" story only works if
            you (accidentally) use the burned-in overlay or the dead vignette as your "background." That
            is why we need demixing, not a rectangle.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(border_var, cell_var, gap_var, mo):
    _cell_hi = cell_var > 0.4
    _gap_hi = gap_var > 0.4
    _close = (0.5 < (gap_var / cell_var) < 2.0) if cell_var > 0 else False
    _border_lo = border_var < 0.4
    _ok = _cell_hi and _gap_hi and _close and _border_lo
    _c = "#e8f5e9" if _ok else "#ffebee"; _b = "#2e7d32" if _ok else "#c62828"
    _m1 = (f"✅ cell box fluctuates (var = {cell_var:.2f} &gt; 0.4)" if _cell_hi
           else f"❌ cell var = {cell_var:.2f} is low — is the box on a bright blob?")
    _m2 = (f"✅ tissue-gap box is nearly as high (var = {gap_var:.2f}); cell/gap = "
           f"{cell_var / gap_var:.2f}× — the box can't separate them" if (_gap_hi and _close)
           else f"❌ gap var = {gap_var:.2f} — pick a dim patch INSIDE the tissue (e.g. 240,240)")
    _m3 = (f"✅ border box reads flat (var = {border_var:.2f} &lt; 0.4) — but it is the blanked "
           "timestamp, not tissue" if _border_lo
           else f"❌ border var = {border_var:.2f} — pick a spot on the blanked timestamp (e.g. 110,28)")
    _head = ("PASS — cell ≈ tissue gap, only the border is flat: the box cannot isolate a cell"
             if _ok else "Not yet — fix the flagged line")
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_b};padding:12px 16px;border-radius:6px">
        <b style="color:{_b}">{_head}</b><br>{_m1}<br>{_m2}<br>{_m3}<br>
        <span style="font-size:0.9em;color:#555">Pinned from the real 90-frame movie: cell ≈ 0.92,
        tissue gap ≈ 0.91, border ≈ 0. The lesson is the near-tie between cell and gap.</span></div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # Part C — demixing the whole population (CNMF)

        ## C1. From one blunt box to every cell at once

        **Why.** Part B showed the box is a blunt tool: it cannot separate overlapping cells and it
        cannot tell tissue-wide co-fluctuation from a real cell. A lab does not draw hundreds of boxes
        by hand, and cannot afford to be fooled by neuropil. We need a method that finds *every* cell
        automatically and hands each one a clean trace.

        **Definitions.**

        - **Mixture.** A single pixel's brightness over time is the *sum* of every cell whose light
          lands there, plus background. You cannot read one neuron off one pixel.
        - **Source separation (demixing).** Splitting that mixture back into the individual neurons.
        - **Spatial footprint `A`.** For one neuron, the exact pixels it occupies and how strongly — a
          small image of *where* the cell is. The learned replacement for the hand-drawn rectangle.
        - **Calcium trace `C`.** For one neuron, its brightness over time — *when* it is active.
        - **Correlation image `Cn`.** A summary image colouring each pixel by how much it rises and
          falls *together with its immediate neighbours*. A lone flickering pixel is noise and stays
          dark; a blob that brightens in lockstep is a candidate cell body. It is the standard first
          look at a calcium movie and where demixing starts.

        **The method — CNMF** (constrained non-negative matrix factorization). It writes the movie
        `Y ≈ A · C`: the footprints `A` (the *where*) times the traces `C` (the *when*), plus
        background. We load a pre-computed result — one striatal session, `221007_4-0_D2` — and explore
        it. (We do not re-run CNMF live; it is compute-heavy and the point is to read its output.)
        """
    )
    return


@app.cell
def _(NB07, np, nu):
    # The precomputed CNMF-E result, read from the committed bundle (no Google-Drive download). The
    # footprint matrix A is 99.75% zeros, so it is stored as a sparse CSR triplet and rebuilt dense
    # here; C/Cn are stored float16. Identical arrays to what nu.load_cnmf() returned from the h5.
    from scipy import sparse as _sp
    A = _sp.csr_matrix((NB07["cnmf_A_data"], NB07["cnmf_A_indices"], NB07["cnmf_A_indptr"]),
                       shape=tuple(int(v) for v in NB07["cnmf_A_shape"])).toarray().astype(np.float32)
    C = NB07["cnmf_C"].astype(np.float32)      # (16773, 202) calcium traces, one column per neuron
    Cn = NB07["cnmf_Cn"].astype(np.float32)    # (600, 600) correlation image
    S = NB07["cnmf_S"]                          # (16773, 202) deconvolved spikes (see reference note)
    Fs = float(NB07["cnmf_Fs"])                 # 30.0 fps
    img_shape = tuple(int(v) for v in NB07["cnmf_img_shape"])   # (600, 600)
    n_neurons = int(NB07["cnmf_n_neurons"])     # 202
    n_frames = int(NB07["cnmf_n_frames"])       # 16773
    C_z = nu.zscore(C.T, axis=1)       # (202, 16773) z-scored raster, one row per neuron
    return A, C, C_z, Cn, Fs, S, img_shape, n_frames, n_neurons


@app.cell
def _(Cn, Fs, mo, n_frames, n_neurons, nu):
    _fig = nu.image_fig(Cn, title="Correlation image Cn — pixels that co-fluctuate with their neighbours",
                        colorscale="Viridis", colorbar_title="local corr", height=500)
    mo.vstack([_fig, mo.md(
        f"""**{n_neurons} neurons** across **{n_frames:,} frames** at **{Fs:.0f} fps**
        ({n_frames / Fs / 60:.1f} min). The bright rings and disks are the sources CNMF will separate
        into individual footprints and traces. Compare it to Part B's max-projection: it answers the
        same "where are the cells" question, but by neighbour-correlation, which suppresses the
        tissue-wide glow that fooled our boxes. Notice how many blobs **touch or overlap** — that
        overlap is the mixing problem this part exists to undo."""
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## C2. What the factorization means — the twin of PCA, made exact

        CNMF writes the whole movie as `Y ≈ A · C`. Read one pixel `p` at one time `t` and that says

        $$ Y[t, p] \;\approx\; \sum_{k} A[k, p]\; C[t, k], $$

        *the pixel's brightness is every neuron's footprint-weight at that pixel times that neuron's
        trace, summed over all neurons.* This is the **same shape of equation** as the PCA you ran on
        behaviour in NB4, where each feature was a weighted sum of a few components. Here is the twin
        table, filled in:

        | | Behaviour PCA (NB4) | Neural CNMF (here) |
        |---|---|---|
        | Data matrix | events × 19 features | frames × pixels |
        | Factorization | scores × components | `C` (traces) × `A` (footprints) |
        | A "component" is | a variance-ranked axis, can be negative | a **non-negative, compact cell** |
        | How many? | you *choose* (kept 6 PCs) | **discovered** from the data (202 here) |
        | Meaning | a statistical direction | a physical claim: "a cell is here" |

        **Same math, stronger claim.** PCA picks orthogonal directions of maximum variance; CNMF adds
        two constraints — every footprint is **non-negative** (a cell can only add light, never
        subtract it) and **spatially compact** (a cell is a small blob, not a field-wide pattern). Those
        two constraints are what turn "a statistical component" into "a neuron." We will *see* why they
        matter in C7, by running plain PCA on the pixels and watching it fail to produce anything that
        looks like a cell.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## C3. One source at a time — the footprint viewer

        After demixing, each neuron is a *where* (footprint) paired with a *when* (trace). `A` has one
        row per neuron and one column per pixel (`202 × 360000`); reshape a row to `600 × 600` and you
        recover that neuron's footprint. Its matching column `C[:, k]` is the trace.

        The slider picks a neuron `k` (it opens on source **79**). **Left:** its footprint, cropped to a
        tight box around the cell so you see the *shape* (a compact blob), not a speck lost in a big
        black field. **Right:** its calcium trace. This is the **un-denoised** trace `C` straight out of
        CNMF, so the baseline is *not* a clean flat line — it is a low, noisy wander — but on source 79 a
        couple of **sharp, asymmetric transients** (fast rise, slow decay) rise cleanly out of that
        baseline, each one a burst of firing. Step through a few and two facts stand out: **114 of the
        202 sources never cross `z = 5`** in the whole nine minutes — most cells are **near-silent** —
        while a busy minority (try source **148**) never really settle, riding a continuously
        fluctuating baseline instead of firing in a few discrete events. That sparsity is real striatal
        biology, and it matters for everything downstream.
        """
    )
    return


@app.cell
def _(mo, n_neurons):
    neuron_ind = mo.ui.slider(0, n_neurons - 1, value=79, step=1,
                              label="neuron index (source)", debounce=True, full_width=True)
    return (neuron_ind,)


@app.cell
def _(A, C, img_shape, mo, neuron_ind, np, nu):
    _k = neuron_ind.value
    _fp = nu.footprint(A, _k, img_shape)
    # crop to a tight bounding box around the footprint so the SHAPE is visible
    _ys, _xs = np.where(_fp > 0.1 * _fp.max())
    if len(_ys):
        _y0, _y1 = max(0, _ys.min() - 12), min(img_shape[0], _ys.max() + 12)
        _x0, _x1 = max(0, _xs.min() - 12), min(img_shape[1], _xs.max() + 12)
        _crop = _fp[_y0:_y1, _x0:_x1]
    else:
        _crop = _fp
    _fig_fp = nu.image_fig(_crop, title=f"Footprint A[{_k}] (cropped) — where source {_k} lives",
                           colorscale="Viridis", colorbar_title="weight", height=420)
    # robust=False here: on a single-neuron trace the transient PEAKS are the signal, so show them at
    # full height (a robust clip guillotines the tallest transient flat) — the noisy baseline still
    # reads clearly in the lower band.
    _fig_tr = nu.trace_fig(None, C[:, _k], title=f"Calcium trace C[:, {_k}] — when it fires",
                           xlabel="Time (frames)", ylabel="calcium (a.u.)", height=420, robust=False)
    _fig_tr.update_traces(line=dict(color="#444444", width=1))
    mo.vstack([neuron_ind, mo.hstack([_fig_fp, _fig_tr], widths=[1, 1])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## C4. All sources at once — the footprint montage

        Peak-normalise every footprint (divide each by its own max, so a dim cell and a bright cell
        count equally) and take the **maximum across all 202 sources** at each pixel. The result lays
        every neuron's territory over the field of view; `nu.footprint_montage(A, img_shape)` does it.
        The bright blobs in `Cn` should reappear here as separated footprints, each now assigned to one
        source.
        """
    )
    return


@app.cell
def _(A, img_shape, mo, np, nu):
    _mont = nu.footprint_montage(A, img_shape)
    _fig = nu.image_fig(_mont, title="Footprint montage — max projection of all 202 sources",
                        colorscale="Viridis", colorbar_title="peak-norm weight", height=500)
    # describe where the footprints actually are (they are NOT spread over the whole FOV)
    _An = np.asarray(A) / (np.asarray(A).max(1, keepdims=True) + 1e-9)
    _ys, _xs = [], []
    for _k in range(A.shape[0]):
        _m = _An[_k].reshape(img_shape) > 0.3
        if _m.any():
            _yy, _xx = np.where(_m); _ys.append(_yy.mean()); _xs.append(_xx.mean())
    _ys, _xs = np.array(_ys), np.array(_xs)
    mo.vstack([_fig, mo.md(
        f"""The sources are **not** spread evenly across the 600×600 field — they cluster in the
        upper-central region (footprint centres span y ≈ {_ys.min():.0f}–{_ys.max():.0f},
        x ≈ {_xs.min():.0f}–{_xs.max():.0f}; the middle half sit in
        y ≈ {np.percentile(_ys, 25):.0f}–{np.percentile(_ys, 75):.0f},
        x ≈ {np.percentile(_xs, 25):.0f}–{np.percentile(_xs, 75):.0f}). The bottom and right edges are
        essentially empty. That is normal — a miniscope images a limited, illuminated patch — and it is
        a reminder to describe what the data *shows*, not what we assume."""
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## C5. The mixture made concrete — demixing a single shared pixel

        Everything so far took CNMF's output on faith. This section shows the factorization doing its
        job on **one pixel**, so "demixing" stops being a word.

        In the montage, many footprints touch. Pick a pixel that sits inside **two** at once — it
        belongs partly to source 27 and partly to source 30. By the equation in C2, its brightness over
        time is a **mixture**: `Y[t, p] ≈ Σ_k A[k, p]·C[t, k]`, and for this pixel almost the whole sum
        comes from just those two terms. **Left:** the two footprints, cropped, coloured red (27) and
        blue (30), with the shared pixel marked. **Right:** the pixel's modelled brightness (black)
        decomposed into the red part (`A[27,p]·C[:,27]`) and the blue part (`A[30,p]·C[:,30]`). The two
        cells fire at *different* times, so the black mixture has red bumps and blue bumps interleaved.
        **Demixing is pulling the red and blue curves back out of the black one** — exactly what PCA did
        on behaviour, but with each component constrained to be one non-negative, localized cell.
        """
    )
    return


@app.cell
def _(A, C, go, img_shape, mo, np, nu):
    _H, _W = img_shape
    _pix = 50726                       # row 84, col 326 — a pixel inside two footprints
    _r, _c = divmod(_pix, _W)
    _o0, _o1 = 27, 30
    _fp0 = nu.footprint(A, _o0, img_shape); _fp1 = nu.footprint(A, _o1, img_shape)
    # crop to the pair's joint bounding box so the two cells are visible, not two specks
    _m = (_fp0 > 0.1 * _fp0.max()) | (_fp1 > 0.1 * _fp1.max())
    _ys, _xs = np.where(_m)
    _y0, _y1 = max(0, _ys.min() - 8), min(_H, _ys.max() + 8)
    _x0, _x1 = max(0, _xs.min() - 8), min(_W, _xs.max() + 8)
    # colour the two footprints distinctly: red channel = src27, blue channel = src30
    _rc = (_fp0 / _fp0.max())[_y0:_y1, _x0:_x1]
    _bc = (_fp1 / _fp1.max())[_y0:_y1, _x0:_x1]
    _rgb = np.stack([_rc, np.zeros_like(_rc), _bc], axis=-1)
    _rgb = (255 * _rgb / max(_rgb.max(), 1e-9)).astype(np.uint8)
    import plotly.graph_objects as _pg
    _fig_fp = _pg.Figure(_pg.Image(z=_rgb))
    _fig_fp.add_scatter(x=[_c - _x0], y=[_r - _y0], mode="markers",
                        marker=dict(color="white", size=12, symbol="x"), showlegend=False)
    _fig_fp.update_yaxes(visible=False, showgrid=False); _fig_fp.update_xaxes(visible=False, showgrid=False)
    nu.apply_house_style(_fig_fp, title="source 27 (red) + source 30 (blue), shared pixel ×",
                         legend=None, height=430)

    _contrib0 = A[_o0, _pix] * C[:, _o0]; _contrib1 = A[_o1, _pix] * C[:, _o1]
    _total = A[:, _pix] @ C.T
    _step = max(1, C.shape[0] // 2500); _t = np.arange(0, C.shape[0], _step)
    _fig_tr = go.Figure()
    _fig_tr.add_scatter(x=_t, y=_total[::_step], mode="lines", name="pixel mixture (all sources)",
                        line=dict(color="#222", width=1.3))
    _fig_tr.add_scatter(x=_t, y=_contrib0[::_step], mode="lines", name="from source 27",
                        line=dict(color="#e45756", width=1.1))
    _fig_tr.add_scatter(x=_t, y=_contrib1[::_step], mode="lines", name="from source 30",
                        line=dict(color="#4c78a8", width=1.1))
    _fig_tr.update_xaxes(title="Time (frames)"); _fig_tr.update_yaxes(title="modelled brightness (a.u.)")
    nu.apply_house_style(_fig_tr, title="one pixel's brightness = a mixture of two neurons",
                         legend="below", height=430)
    _corr = float(np.corrcoef(_contrib0, _contrib1)[0, 1])
    mo.vstack([mo.hstack([_fig_fp, _fig_tr], widths=[1, 1]), mo.md(
        f"""The two contributions correlate at only **{_corr:.2f}** — they are all but independent, so
        the black mixture genuinely carries two separate signals. A patch-average (Part B) would hand
        you the black curve and call it "the signal here"; CNMF returns the red and the blue, each one
        neuron. **That is the whole payoff of demixing: one clean cell where the raw movie only offered
        a blend.**"""
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## C6. Who are these 202 cells? A population profile

        Demixing hands us 202 sources, but they are not interchangeable. Before analysing them we
        characterise the population — the neural echo of the feature EDA from Week 1. For each neuron we
        count **large events**: the number of times its z-scored trace crosses `z = 5` (a rising-edge
        count) over the whole recording — how often it fires strongly.

        The **empirical CDF (ECDF)** below reads: for any event count on the x-axis, the fraction of
        neurons at or below it. (An ECDF, not a smoothed density: with heavy-tailed integer counts a
        KDE would invent curves between the integers and can spill below zero.) The curve rises steeply
        and flattens — most cells fire rarely, a minority carry most events. **Sparse coding, seen
        directly.**
        """
    )
    return


@app.cell
def _(C_z, mo, np, nu):
    _above = C_z > 5.0
    _neuron_events = (_above[:, 1:] & ~_above[:, :-1]).sum(axis=1)   # (202,) rising z>5 crossings
    _peakz = C_z.max(axis=1)
    _n_quiet = int((_neuron_events == 0).sum()); _n_active = int((_peakz > 5).sum())
    _fig = nu.ecdf_fig(_neuron_events, xlabel="large events (z>5 crossings, whole recording)",
                       ylabel="fraction of neurons at or below",
                       title="most cells fire rarely; a few carry the events", height=430)
    mo.vstack([_fig, mo.md(
        f"""**{_n_quiet} of the 202 neurons never cross z = 5** in the whole 9 minutes; the busiest
        fires **{int(_neuron_events.max())}** times. By a looser "ever peaks above z = 5" definition,
        **{_n_active} are active** and {202 - _n_active} are quiet. Keep this in mind for NB8: when we
        look for population structure, only the active minority can contribute to it."""
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## C7. Why CNMF and not plain PCA on the pixels?

        The twin table in C2 said the two constraints — non-negativity and compact footprints — are
        what make a CNMF component a *cell*. Here is the proof. We run **plain PCA** on the same kind of
        pixel×time matrix (the cleaned striatum foreground, `F frames × 250,000 pixels`), then look at
        what its top components' spatial maps actually are.

        PCA is not wrong — it is the same variance-maximising factorization from NB4. But with no
        non-negativity and no locality constraint, its components come out as **field-wide, mixed-sign**
        patterns: a bright-here / dark-there wash across the whole image. That can be a perfectly good
        summary of variance, but it is **not a cell** — a cell adds light in one small place and adds it
        everywhere it is present (never subtracts). Compare a PCA component to a CNMF footprint side by
        side and the difference is obvious.
        """
    )
    return


@app.cell
def _(A, frames_clean, img_shape, mo, np, nu):
    from sklearn.decomposition import PCA as _PCA
    _bs = nu.background_subtract(frames_clean); _fg = _bs["fg"]
    _F, _H, _W = _fg.shape
    _X = _fg.reshape(_F, _H * _W)                          # time × pixels
    _pca = _PCA(n_components=4, random_state=0).fit(_X)
    _evr = _pca.explained_variance_ratio_
    # PC2 spatial map: mixed-sign, non-localized
    _pc = _pca.components_[1].reshape(_H, _W)
    _m = float(np.percentile(np.abs(_pc), 99))
    _fig_pca = nu.image_fig(_pc, title=f"PCA on pixels — PC2 map (mixed sign, field-wide)",
                            colorscale="RdBu_r", zmin=-_m, zmax=_m, colorbar_title="loading", height=430)
    # a CNMF footprint for contrast, cropped
    _fp = nu.footprint(A, 148, img_shape)
    _ys, _xs = np.where(_fp > 0.1 * _fp.max())
    _crop = _fp[max(0, _ys.min() - 12):_ys.max() + 12, max(0, _xs.min() - 12):_xs.max() + 12]
    _fig_cnmf = nu.image_fig(_crop, title="CNMF footprint — non-negative, compact (a cell)",
                             colorscale="Viridis", colorbar_title="weight", height=430)
    _neg = (_pca.components_[1] < 0).mean()
    mo.vstack([mo.hstack([_fig_pca, _fig_cnmf], widths=[1, 1]), mo.md(
        f"""PC1 alone explains **{100 * _evr[0]:.0f}%** of the variance and is an all-positive,
        field-wide glow — the tissue-wide co-fluctuation that fooled our boxes in Part B, captured as
        one component. PC2 (shown) explains **{100 * _evr[1]:.1f}%** and is **{100 * _neg:.0f}% negative
        weights**, smeared across the whole field: a legitimate variance axis, but nothing you could
        call a neuron. The CNMF footprint beside it is 0% negative and a compact blob. Same
        factorization math; the constraints are the whole difference between "a statistical axis" and "a
        cell." This is the same PCA-vs-nonlinear tension you will meet again in NB8, where PCA on the
        *population* summarises variance but a structured method is needed to expose the manifold."""
    )])
    return


@app.cell(hide_code=True)
def _(S, mo, np):
    _s_nonzero = int((np.asarray(S) != 0).sum())
    mo.accordion({
        "Reference — CNMF, and why we do not report spikes here": mo.md(
            rf"""
            **The method.** Pnevmatikakis et al. 2016, *Neuron* 89(2):285–299, "Simultaneous Denoising,
            Deconvolution, and Demixing of Calcium Imaging Data" (**CNMF**); the one-photon miniscope
            variant is **CNMF-E** (Zhou et al. 2018, *eLife* 7:e28728 — the striatal dataset). CNMF
            factors the movie `Y ≈ A · C + b` into non-negative footprints `A` and traces `C` plus a
            background term `b`. It is a constrained matrix factorization, the same family as the PCA we
            ran on behaviour in NB4.

            **Why we show `C`, not `S`.** The file also carries `S`, meant to hold a **deconvolved spike
            estimate** (an attempt to recover the fast spikes under the slow calcium). Two caveats:
            deconvolution is calibrated for two-photon data and is **not validated** for one-photon
            miniscope recordings; and concretely, in this refined file `S` holds **{_s_nonzero} non-zero
            values** — no spike estimate was shipped at all. So we analyse the calcium trace `C`
            throughout and never report spike counts.

            **Limits of the analogy.** A PCA component is a *statistical* axis — abstract, can be
            negative, need correspond to nothing real. A CNMF footprint is a *physical claim*: "a cell
            is here." Stronger and testable — and it can be wrong. Two adjacent cells can merge into one
            source, or one cell split into two, and the variance explained will not tell you. Demixing
            is only as good as its footprints, and footprints are inferred, not observed.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## The answer, and the next question

        **The question we asked:** how do we go from a raw microscope movie of a behaving mouse's brain
        to one clean activity trace per neuron?

        **The answer, in three stages.** First we **held the image still**: registration estimates how
        far each frame drifted and shifts it back, rigid for global drift and piecewise-rigid for
        non-uniform tissue motion. We scored it with the motion index, validated that score against
        injected ground truth, confirmed the improvement with a *paired* test frame-by-frame, and saw
        the mechanism (phase correlation finds the shift). Then we tried to **read one cell by hand** —
        background subtraction, a max-projection, a box — and found the box's honest limit: inside real
        tissue a cell and the gap beside it fluctuate almost identically, because the tissue
        co-fluctuates; only the burned-in border reads flat. That failure motivated the third stage,
        **demixing with CNMF**, which writes the movie as `Y ≈ A · C` and returns, for each of 202
        sources, a compact footprint (*where*) and a calcium trace (*when*) — the same factorization
        math as behaviour's PCA, but constrained so each component is a cell, not a statistical axis. A
        movie of light is now a population of per-neuron time series.

        **What was the same, and what was not.** The math rhymed with Week 1 at every step — align
        before you read, factor structure out of a matrix. But where the pose tracker *handed* us 15
        named keypoints, here we had to **discover** an unknown number of unlabelled, overlapping cells
        and separate them, per recording. That is why the neural side needed a whole notebook to reach
        the starting line the behaviour side began from.

        **The next question (NB8).** We now hold 202 clean neurons, each a time series of *when* it
        fires — but we do not know what any of them is *about*. Does a neuron's activity track something
        specific: the animal's position, or whether it is being social? And can we read that variable
        back out of the population — with honest cross-validation this time? That is tuning and
        decoding, the neural twin of NB4 and NB6, and where NB8 goes next.
        """
    )
    return


if __name__ == "__main__":
    app.run()
