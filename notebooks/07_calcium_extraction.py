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
#     "imageio-ffmpeg>=0.4.9",
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
        # NB7 · From a movie to calcium traces

        **Week 2 — working with real neural recordings.**

        ### The question we carry in

        In the previous notebook we faced a mechanical problem. A miniature microscope is bolted to the
        mouse's head, so every time the animal turns or runs the whole field of view lurches. A pixel
        that pointed at one cell in frame 100 might point at the tissue next to it in frame 101. We
        asked: **can we hold the movie still enough that a fixed pixel keeps pointing at the same piece
        of tissue?** Motion correction answered yes — it registers each frame back onto a common
        template, and the jitter score dropped as the alignment tightened.

        So now we have a *stabilized* movie of glowing brain tissue. That raises the question this
        notebook exists to answer:

        > **How do we turn a movie of glowing cells into a signal we can analyze?**

        A movie is not yet data we can do statistics on. It is a stack of images: hundreds of thousands
        of pixel values per frame, thousands of frames. What we ultimately want is one number per frame
        for each neuron — a *time series of that cell's activity*. This notebook builds that reduction
        twice, from the simple hand version to the method a real lab uses.

        ### What calcium imaging is, in plain terms

        To study how the brain produces social behavior, we have to watch neurons work. We cannot see electrical firing directly through a microscope, so we use a
        proxy.

        - **Neuron firing → calcium.** When a neuron fires, calcium ions rush into the cell. The internal
          calcium concentration spikes briefly with each burst of activity, then decays back down over a
          fraction of a second. Calcium is therefore a stand-in for firing: more calcium means the cell
          was more active, moments ago.
        - **GCaMP.** A protein sensor we express inside neurons that **glows brighter when it binds
          calcium**. Because calcium tracks firing, the brightness of a GCaMP-labeled cell is a *proxy
          for its activity*. This is **calcium imaging**: filming activity as light.
        - **Miniscope.** The small head-mounted microscope that records this glowing tissue as a video
          while the mouse behaves freely (the same movie we just stabilized).
        - **Calcium trace.** The thing we want to extract: one number per frame describing how bright a
          single cell is over time. Its sharp asymmetric rises — fast up, slow down — are **calcium
          transients**, each one a burst of firing.
        - **Region of interest (ROI).** A small patch of pixels we select in the image, here a box placed
          over one cell. Averaging the pixels inside the box at each frame turns an image stack into a
          single trace.

        ### The plan

        We answer the question in two passes, from hand-made to learned:

        1. **Hand-placed ROI (Part I).** Take a real striatal miniscope movie, remove its static
           background, find where the active cells are, and read out one cell's trace by averaging a box
           of pixels. This is the honest minimal version of calcium extraction — and its failure points
           tell us exactly why a lab needs something better.
        2. **Demixing the whole population (Part II).** Replace the hand-drawn box with an algorithm
           (**CNMF**) that learns a spatial footprint for *every* neuron at once and separates
           overlapping cells, giving us hundreds of clean traces from one movie.

        Both passes do the same thing — turn pixels into a per-cell measurement — once by hand and once
        learned from the data.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # Part I — reading one cell by hand

        ## 1. The raw signal: pixels over time

        **Why.** Before processing anything, look at the input. The raw miniscope movie is the
        high-dimensional signal we are going to reduce, exactly as the raw skeletons were in Week 1.
        Seeing it first makes clear *why* the later steps are needed.

        **Method.** We stream a real striatum movie and **subsample** it: we keep every 100th frame
        (`step=100`), so roughly 9,000 raw frames become **90**. Subsampling keeps enough frames to see
        the structure while keeping memory and compute small enough to run on a bare cloud kernel.
        `nu.read_video(path, step=100)` reads the video file (input: the file path) and returns a
        `(90, 500, 500)` array of grayscale frames (output: the stack of frames).

        Each frame is a **500 × 500 = 250,000-value** image, and there are 90 of them. Move the slider to
        scrub through the movie. Notice that by eye it is genuinely hard to pick out individual cells: a
        fixed bright glow covers the whole field, and the cells hide inside it. Removing that glow is the
        very next step.
        """
    )
    return


@app.cell
def _(nu):
    # One sequential ffmpeg decode, subsampled to 90 frames (~7 s, cached video). This is the single
    # heavy beat of Part I; everything downstream is cheap numpy on the 90x500x500 array.
    striatum_path = nu.fetch_url(nu.STRIATUM_URL, nu.STRIATUM_NAME)
    frames = nu.read_video(striatum_path, step=100)          # (90, 500, 500) float32
    _bs = nu.background_subtract(frames)                       # median bg removed + per-pixel z-score
    fg = _bs["fg"]                                             # (90, 500, 500) z-scored foreground
    bg = _bs["bg"]                                             # (500, 500) median background
    F = int(frames.shape[0])
    return F, bg, fg, frames


@app.cell
def _(F, mo):
    movie_t = mo.ui.slider(0, F - 1, value=0, step=1,
                           label="frame (subsampled movie, 0–89)", debounce=True, full_width=True)
    return (movie_t,)


@app.cell
def _(frames, mo, movie_t, nu):
    _raw = nu.image_fig(frames[movie_t.value], title=f"RAW miniscope frame {movie_t.value}",
                        colorscale="gray", colorbar_title="intensity", height=470)
    mo.vstack([movie_t, _raw])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 2. Remove the static background (background subtraction + z-score)

        **Why.** Most of what you just saw is unchanging: uneven illumination, out-of-focus tissue, the
        lens itself. This static pattern is the same in every frame, so it carries no information about
        *when a neuron fires*. Removing it leaves only the part of the signal that changes over time,
        which is the part we care about.

        **Definitions.**

        - **Background subtraction** means estimating the static part of the image and subtracting it, so
          only the changing part remains.
        - **z-score** means rescaling a value into units of its own standard deviation — subtract the
          mean, divide by the standard deviation — so pixels with different baseline brightness become
          directly comparable. A z-score of `+3` means "three standard deviations above this pixel's
          usual value," whether the pixel is normally bright or dim.

        **Method.** `nu.background_subtract(frames)` performs three operations:

        $$
        \text{bg} = \operatorname{median}_t(\text{frames}), \qquad
        \text{fg} = \text{frames} - \text{bg}, \qquad
        \text{fg} \leftarrow \frac{\text{fg} - \mu_{\text{px}}}{\sigma_{\text{px}}}
        $$

        Its input is the `(90, 500, 500)` frame stack; its outputs are the `(500, 500)` background image
        `bg` and the `(90, 500, 500)` z-scored foreground `fg`. Taking the **median over time** at each
        pixel is the key idea: a pixel on inactive tissue looks the same in almost every frame, so its
        median value *is* its background; a pixel that occasionally flares when a cell fires is dark in
        most frames, so the flare survives the subtraction. The per-pixel z-score then puts every pixel
        on the same scale, so a dim active cell is not drowned out by a bright inactive patch.

        Below, the left panel is the background that was removed — the static glow. The right panel is
        one foreground frame on a symmetric ±3σ scale: the flat glow is gone, and short-lived bright
        spots (firing cells) stand out. The same slider controls both, so you can watch spots blink on
        and off in the foreground while the background stays fixed.
        """
    )
    return


@app.cell
def _(bg, fg, mo, movie_t, nu):
    _bg_fig = nu.image_fig(bg, title="background (median over time)", colorscale="gray",
                           colorbar_title="intensity", height=470)
    _fg_fig = nu.image_fig(fg[movie_t.value], title=f"foreground frame {movie_t.value}  (z-scored)",
                           colorscale="RdBu", zmin=-3, zmax=3, colorbar_title="σ", height=470)
    mo.vstack([movie_t, mo.hstack([_bg_fig, _fg_fig], widths=[1, 1])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 3. Where are the cells? The maximum projection

        **Why.** A single foreground frame only shows the cells that happen to be firing at that one
        instant. To decide where to place an ROI, we want a single image that shows **every** location
        that was active at any point in the movie — a map of all the candidate cells at once.

        **Definition.** A **maximum projection** collapses a movie into one image by taking, at each
        pixel, the largest value that pixel ever reached across the whole recording. Here we use the
        maximum of the *absolute* foreground over time:

        $$\text{active}(y,x) = \max_t \big|\text{fg}(t,y,x)\big|$$

        **Method.** `np.abs(fg).max(axis=0)` takes the absolute value of the foreground (so both bright
        rises and dark dips count as activity) and then takes the maximum across the time axis (`axis=0`
        is time). Its input is the `(90, 500, 500)` foreground; its output is a single `(500, 500)`
        image. A pixel that ever flared appears bright; a pixel that stayed at baseline stays dark. This
        image is our **map of candidate cells**, and it is where we will aim an ROI next.
        """
    )
    return


@app.cell
def _(fg, np):
    maxproj = np.abs(fg).max(axis=0)                           # (500, 500) map of active pixels
    return (maxproj,)


@app.cell
def _(maxproj, nu):
    nu.image_fig(maxproj, title="max |foreground| over time — bright spots are active cells",
                 colorscale="Inferno", colorbar_title="peak σ", height=560)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 4. Place an ROI to read out one cell

        **Why.** We now have a map of where cells are, but we still want a single number per frame for
        one specific cell. Choosing where to place the ROI box is what decides which cell — or which
        piece of background — the trace describes. The ROI *is* the measurement.

        **Method.** The ROI trace is the average of the foreground inside the box at every frame:

        $$\text{trace}(t) = \operatorname{mean}_{y,x \in \text{ROI}} \text{fg}(t, y, x)$$

        The function `roi_trace(fg, cx, cy, r=10)` (defined in the next cell) does exactly this. Its
        inputs are the foreground stack and the box center `(cx, cy)` with half-width `r`; its output is
        a `(90,)` trace, one value per frame. A box with `r=10` is 20 × 20 = 400 pixels.

        Use the two sliders to move the ROI center over the active-cell map (left; the cyan box marks the
        current position). The trace it extracts appears on the right and updates live. When the box sits
        on a bright spot, the trace shows sharp calcium transients; when it sits on dark background, the
        trace stays near zero and looks like flat noise. Remember that image indexing is `[y, x]`, so
        `cx` moves the box horizontally and `cy` moves it vertically. The title reports the trace's
        **variance** — how much it fluctuates — which we will use as our readout in a moment.
        """
    )
    return


@app.cell
def _():
    # The ROI reader used by the interactive panel, the EDA, and the exercise. A 2*r-wide box centered
    # on (cx, cy); image indexing is [y, x], so cy selects rows and cx selects columns.
    def roi_trace(fg_stack, cx, cy, r=10):
        y0, y1 = int(cy - r), int(cy + r)
        x0, x1 = int(cx - r), int(cx + r)
        return fg_stack[:, y0:y1, x0:x1].mean(axis=(1, 2))
    return (roi_trace,)


@app.cell
def _(mo):
    roi_cx = mo.ui.slider(15, 485, value=243, step=1, label="ROI center x (cx)",
                          debounce=True, full_width=True)
    roi_cy = mo.ui.slider(15, 485, value=349, step=1, label="ROI center y (cy)",
                          debounce=True, full_width=True)
    return roi_cx, roi_cy


@app.cell
def _(maxproj, mo, nu, roi_cx, roi_cy, roi_trace, fg):
    _cx, _cy, _r = roi_cx.value, roi_cy.value, 10
    _img = nu.image_fig(maxproj, title="active-cell map — drag the ROI box",
                        colorscale="Inferno", colorbar_title="peak σ", height=460)
    _img.add_shape(type="rect", x0=_cx - _r, y0=_cy - _r, x1=_cx + _r, y1=_cy + _r,
                   line=dict(color="#00e5ff", width=3))
    _tr = roi_trace(fg, _cx, _cy, _r)
    _trace = nu.trace_fig(None, _tr, title=f"ROI trace at ({_cx}, {_cy}) — variance = {_tr.var():.3f}",
                          xlabel="frame", ylabel="mean foreground (σ)", height=460)
    mo.vstack([mo.hstack([roi_cx, roi_cy]), mo.hstack([_img, _trace], widths=[1, 1])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 5. Does the ROI actually carry signal? Cell vs background, many boxes

        **Why.** Dragging a single box is suggestive but not evidence. If placing the ROI is what defines
        the measurement, then boxes on cells and boxes on empty space should give *quantitatively
        different* traces. We test that across many boxes at once, and we look at the raw numbers rather
        than a single summary — the honest way to compare two groups.

        **Definition — variance.** The **variance** of a trace is the average squared distance of its
        values from their mean. A trace that stays flat has low variance; a trace with big calcium
        transients has high variance. So variance is a one-number summary of *how much a box fluctuates
        over time*.

        **Method.** We placed **14 boxes on clearly visible cells** (bright blobs on the map above) and
        **14 boxes on dead space at the edge of the field** — the dark vignette outside the imaged
        tissue, where there are no cells at all. For each of the 28 boxes we extract its `roi_trace` and
        take `.var()`. The panel on the left marks every box on the active-cell map (warm = cell,
        gray = background); the panel on the right is a **strip plot**: one dot per box, its trace
        variance on the y-axis, cells and background side by side, with the group mean drawn as a bar.
        Hover any dot to see its center coordinates.

        We show every individual point rather than two bars because the *spread* is the story: are the
        two groups cleanly separated, or do they overlap?
        """
    )
    return


@app.cell
def _(fg, np, roi_trace):
    # 14 boxes centered on clearly visible cells (bright blobs on the max-projection map) and 14 boxes
    # on dead vignette at the very edge of the field, where there is no tissue. These are deliberate,
    # hand-placed ROIs -- exactly what a person does when reading a movie by eye.
    CELL_ROIS = np.array([(243, 349), (154, 338), (306, 96), (276, 96), (197, 63), (336, 111),
                          (224, 66), (393, 325), (124, 265), (129, 135), (394, 371), (310, 63),
                          (336, 65), (429, 137)])
    BG_ROIS = np.array([(15, 15), (30, 15), (45, 15), (60, 15), (405, 15), (420, 15), (435, 15),
                        (450, 15), (465, 15), (480, 15), (15, 30), (30, 30), (45, 30), (75, 30)])
    # Trace variance for every box (a small readable loop; each entry is one .var() reduction).
    cell_var_all = np.array([roi_trace(fg, cx, cy).var() for cx, cy in CELL_ROIS])
    bg_var_all = np.array([roi_trace(fg, cx, cy).var() for cx, cy in BG_ROIS])
    # Stack into the (values, groups, hover) form the seaborn-style helper wants.
    roi_var = np.concatenate([cell_var_all, bg_var_all])
    roi_group = np.array(["cell"] * len(CELL_ROIS) + ["background"] * len(BG_ROIS))
    roi_hover = [f"({cx},{cy})" for cx, cy in np.vstack([CELL_ROIS, BG_ROIS])]
    return (BG_ROIS, CELL_ROIS, bg_var_all, cell_var_all, roi_group, roi_hover, roi_var)


@app.cell
def _(BG_ROIS, CELL_ROIS, maxproj, mo, np, nu, roi_group, roi_hover, roi_var):
    # Left: mark every ROI box on the active-cell map so its placement is visible.
    _img = nu.image_fig(maxproj, title="the 28 ROI boxes (warm = cell, gray = background)",
                        colorscale="Inferno", colorbar_title="peak σ", height=470)
    for _cx, _cy in CELL_ROIS:
        _img.add_shape(type="rect", x0=_cx - 10, y0=_cy - 10, x1=_cx + 10, y1=_cy + 10,
                       line=dict(color="#e45756", width=2))
    for _cx, _cy in BG_ROIS:
        _img.add_shape(type="rect", x0=_cx - 10, y0=_cy - 10, x1=_cx + 10, y1=_cy + 10,
                       line=dict(color="#bbbbbb", width=2))
    # Right: strip plot of the raw per-box variances, cells vs background, points + group means.
    _strip = nu.strip_points_fig(roi_var, roi_group, group_order=["cell", "background"],
                                 colors={"cell": "#e45756", "background": "#969696"},
                                 hover=np.array(roi_hover), ylabel="ROI trace variance",
                                 title="cell ROIs fluctuate; dead-space ROIs do not", height=470)
    mo.hstack([_img, _strip], widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(bg_var_all, cell_var_all, mo, np):
    mo.md(
        f"""
        **What the numbers say.** The cell ROIs cluster near the top (median variance
        **{np.median(cell_var_all):.2f}**), the dead-space ROIs sit near the bottom (median
        **{np.median(bg_var_all):.2f}**) — a clean separation of roughly
        **{np.median(cell_var_all) / np.median(bg_var_all):.1f}×**, with no overlap between the groups.
        Averaging a box of pixels gives real calcium signal when the box is on a cell and flat noise when
        it is on empty space. **Where you place the box is what defines the measurement.**

        **An important caveat — and a preview of why we need more.** The clean separation above uses
        *dead vignette outside the tissue* as the background. If instead you place a box in the dark gaps
        *between* cells — still inside the imaged tissue — the trace is **not** flat: neuropil, blood
        flow, and overall brightness changes make the whole tissue breathe together, and that box has
        nearly as much variance as a cell. In dense tissue, a hand-placed box also blends neighboring
        cells and leaks surrounding signal into the average. A rectangle is a blunt instrument. Part II
        replaces it with a method that learns each cell's exact shape.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 6. Exercise — measure the cell/background gap yourself

        **Python skill practiced: array slicing and axis reductions.** You will index a 3-D image stack
        with a box of pixels and collapse it to one number. This is the core move of the whole notebook:
        `stack[:, y0:y1, x0:x1]` selects a sub-volume, `.mean(axis=(1, 2))` averages over the two spatial
        axes to leave a per-frame trace, and `.var()` reduces that trace to a single fluctuation score.

        **The claim to test.** A cell ROI carries calcium transients, so it varies a lot (high variance).
        A background ROI of the same size stays near zero, so it varies little (low variance). If ROI
        placement defines the measurement, the cell-ROI variance should be clearly larger than the
        background-ROI variance.

        **Tools you will use.**

        - `roi_trace(fg, cx, cy, r=10)` returns the `(90,)` box-average trace at center `(cx, cy)`.
        - `fg` is the `(90, 500, 500)` z-scored foreground from Section 2.
        - `trace.var()` returns one number: how much the trace fluctuates over the 90 frames.
        - Use the active-cell map and the slider in Section 4 to find a bright blob and a dark patch.

        **What to do.** In the next cell, complete the two marked lines so that `cell_var` measures a
        bright cell and `bg_var` measures dark background. The cell line is filled in for you as a worked
        example at `(243, 349)`; complete the background line with a dark patch such as `(30, 30)`.

        **What you should see.** The cell trace visibly rises and falls, giving a variance near **0.92**.
        The background trace stays close to zero, giving a variance near **0.15** — roughly a **6×**
        difference. The self-check below passes when the cell variance is well above the background
        variance.
        """
    )
    return


@app.cell
def _(fg, roi_trace):
    # -------------------------------------------------------------- YOUR CODE (edit the 2 marked lines)
    # roi_trace(fg, cx, cy, r=10) -> (90,) trace: the average foreground brightness inside a
    #     20x20-pixel box centered on pixel (cx, cy), one value per movie frame.
    # float(trace.var())          -> one number: how much that trace fluctuates over the 90 frames.
    #     Scientifically, high variance means the box saw calcium transients (a firing cell); low
    #     variance means the box saw flat noise (empty space).

    # LINE 1 (worked example): a BRIGHT CELL at (243, 349), the slider's default position. Leave as-is.
    _cell_trace = roi_trace(fg, cx=243, cy=349, r=10)
    cell_var = float(_cell_trace.var())

    # LINE 2 (you complete): a DARK BACKGROUND patch of the SAME size. Replace cx and cy below with a
    #     spot on empty tissue -- the corner (cx=30, cy=30) works. WHY it matters: if the two variances
    #     come out far apart, you have shown that the same averaging operation yields signal or noise
    #     depending ONLY on where the box sits. The interactive map in Section 4 helps you confirm the
    #     patch is dark before you trust the number.
    _bg_trace = roi_trace(fg, cx=30, cy=30, r=10)
    bg_var = float(_bg_trace.var())
    # ------------------------------------------------------------------------------------------------
    return bg_var, cell_var


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show solution": mo.md(
            r"""
            ```python
            cell_trace = roi_trace(fg, cx=243, cy=349, r=10)   # a bright blob on the max-projection
            cell_var   = float(cell_trace.var())               # ~0.918

            bg_trace   = roi_trace(fg, cx=30, cy=30, r=10)      # dark corner, no cell
            bg_var     = float(bg_trace.var())                 # ~0.149
            ```

            **What you should find.** The cell ROI has variance about **0.92** and the background ROI
            about **0.15**, a difference of roughly **6×**. The cell trace clearly rises and falls
            (calcium transients); the background trace stays near zero. The same operation — averaging
            the foreground inside a box — gives useful signal or flat noise depending only on **where**
            the box is placed. Placing the ROI is what defines the measurement. (A second cell sits at
            `(154, 338)`, with variance about 0.916; try it as well.)
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(bg_var, cell_var, mo):
    # Self-check with a tolerance band pinned from the real 90-frame movie:
    #   cell(243,349) var = 0.9181, cell(154,338) = 0.9155, background(30,30) = 0.1493  (ratio ~6.1).
    # Grade the claim: cell variance is well above background (ratio > 2.5) AND the background
    # patch really is flat (bg_var < 0.4). We do not grade the exact number, only the separation.
    _ratio = cell_var / bg_var if bg_var > 0 else float("inf")
    _p_cell = cell_var > 0.4
    _p_flat = bg_var < 0.4
    _p_sep = _ratio > 2.5
    _ok = _p_cell and _p_flat and _p_sep
    _c = "#e8f5e9" if _ok else "#ffebee"
    _b = "#2e7d32" if _ok else "#c62828"
    _m_cell = (f"✅ cell ROI fluctuates (variance = {cell_var:.3f} > 0.4)" if _p_cell
               else f"❌ cell ROI variance = {cell_var:.3f} is low — is the box on a bright blob?")
    _m_flat = (f"✅ background ROI is flat (variance = {bg_var:.3f} < 0.4)" if _p_flat
               else f"❌ background ROI variance = {bg_var:.3f} is high — that patch has a cell in it")
    _m_sep = (f"✅ cell / background variance ratio = {_ratio:.1f}× (> 2.5) — the ROI choice made the feature"
              if _p_sep else
              f"❌ ratio = {_ratio:.1f}× is too small — the two ROIs are not clearly different")
    _head = "PASS — the cell ROI variance is well above background" if _ok else "Not yet — fix the flagged line"
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_b};padding:12px 16px;border-radius:6px">
        <b style="color:{_b}">{_head}</b><br>
        {_m_cell}<br>{_m_flat}<br>{_m_sep}<br>
        <span style="font-size:0.9em;color:#555">Tolerance band (pinned from the real movie):
        cell_var &gt; 0.4, bg_var &lt; 0.4, ratio &gt; 2.5. Pinned truth: cell ≈ 0.918, background ≈ 0.149,
        ratio ≈ 6.1×.</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Dataset, method, and the limits of a hand-drawn box": mo.md(
            r"""
            **Provenance.** The movie is `elife-28728-video1.mp4`, distributed with the open-access eLife
            article **e28728** (DOI [10.7554/eLife.28728](https://doi.org/10.7554/eLife.28728)): a
            head-mounted miniature microscope recording GCaMP calcium fluorescence from **striatal**
            neurons in a freely moving mouse. The notebook streams it from eLife's server at runtime and
            caches it locally.

            **What this method is.** Median-background subtraction, a per-pixel z-score, and a hand-placed
            ROI form a simple, teachable version of calcium extraction. It is honest as far as it goes,
            and on a well-isolated cell in a dead-space background it gives a clean 6× signal.

            **Where it breaks.** A rectangular ROI assumes one cell sits inside the box and nothing else
            does. Real striatal tissue is dense: cells overlap, surrounding tissue (neuropil) leaks into
            the average, the tissue as a whole co-fluctuates, and a single box can blend two cells or half
            a cell plus background. The clean trace we extracted is a best case on a cell we could see by
            eye. When cells crowd together, hand-placed boxes are no longer adequate. Part II is the fix:
            a method that learns each cell's shape from the data.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # Part II — demixing the whole population

        ## 7. From one box to every cell at once

        **Why.** Part I gave us one trace from one hand-placed box, and showed the box is a blunt tool:
        it cannot separate overlapping cells and it cannot tell tissue-wide background from a real cell.
        A lab does not draw hundreds of boxes by hand, and it cannot afford to be fooled by co-fluctuating
        neuropil. We need a method that finds *every* cell automatically and gives each one a clean trace.

        **Definitions.**

        - **Source separation (demixing).** Splitting a signal that is a *mixture* of several things into
          the separate things that make it up. In a calcium movie the light from many neurons lands on
          overlapping pixels, so any single pixel is a mixture. Demixing recovers the individual neurons
          behind those pixels — the principled version of "which box is which cell."
        - **Spatial footprint `A`.** For one neuron, the exact set of pixels it occupies and how strongly
          it contributes to each — a small image showing *where* that neuron is. This is the learned
          replacement for the hand-drawn rectangle: instead of a box, the cell's true shape.
        - **Calcium trace `C`.** For one neuron, its brightness over time — a time series showing *when*
          that neuron is active, exactly like the trace we read by hand, but for every cell.
        - **Correlation image `Cn`.** A summary image where each pixel is colored by how much it rises and
          falls *together with its neighbors*. Cell bodies show up as bright blobs; isolated noisy pixels
          stay dark. It is the standard first look at a calcium movie and the starting point for demixing.

        **Method — CNMF.** The algorithm is **CNMF** (constrained non-negative matrix factorization). It
        factors the movie into the spatial footprints `A` (the *where*) and the temporal traces `C` (the
        *when*). This is the same shape of operation as the **PCA** you ran on behavior in NB3: both write
        a data matrix as a product of a spatial factor and a temporal factor. The difference is that a
        CNMF component is forced to be non-negative and spatially compact, so each one corresponds to a
        real cell rather than an abstract statistical axis.

        The recording we use is one striatal session, `221007_4-0_D2`: **202 demixed neurons** across
        about **16,800 frames at 30 fps** (roughly 9 minutes). It ships pre-computed — we load the result
        of CNMF, not re-run it live — and explore the footprints, the traces, and the population's
        temporal structure.
        """
    )
    return


@app.cell
def _(nu):
    _d = nu.load_cnmf()
    A = _d["A"]                       # (202, 360000) spatial footprints, one row per neuron
    C = _d["C"]                       # (16773, 202) calcium traces, one column per neuron
    Cn = _d["Cn"]                     # (600, 600) correlation image
    S = _d["S"]                       # (16773, 202) deconvolved spike estimate (see reference note)
    Fs = _d["Fs"]                     # 30.0 fps
    img_shape = _d["img_shape"]       # (600, 600)
    n_neurons = _d["n_neurons"]       # 202
    n_frames = _d["n_frames"]         # 16773
    # z-scored population raster, one row per neuron (per-neuron mean/std across time). Reused below.
    C_z = nu.zscore(C.T, axis=1)
    # The 2025 analysis's behavior-clock "arena entry" frame, converted to the imaging clock
    # (behavior 25 fps -> imaging 30 fps). Anchors the sequence window + the sequence exercise.
    ENTRY = int(7488 * (30 / 25))     # -> 8985
    WIN_LEN = 3 * 60 * 30             # 3 minutes at 30 fps -> 5400 frames
    return A, C, C_z, Cn, ENTRY, Fs, WIN_LEN, img_shape, n_frames, n_neurons


@app.cell(hide_code=True)
def _(Fs, mo, n_frames, n_neurons):
    mo.md(
        f"""
        ---
        ## 8. The correlation image — the starting point for demixing

        **Why.** Before any demixing runs, we need to see roughly where the cells are. The correlation
        image is the standard first look at a calcium movie, and it is also what CNMF uses to place its
        initial guesses.

        **Definition.** The **local correlation image `Cn`** colors each pixel by how strongly its
        brightness fluctuates *together with its immediate neighbors* over the whole recording. A single
        bright pixel on its own is usually noise. A *blob* of pixels that brighten and dim in lockstep is
        a candidate cell body, because all the pixels covering one neuron rise and fall together.

        **Method.** The figure below plots `Cn` for this recording of **{n_neurons} neurons** across
        **{n_frames:,} frames** at **{Fs:.0f} fps** ({n_frames / Fs / 60:.1f} min). The bright rings and
        disks are the sources CNMF will separate into individual footprints and traces. Compare this to
        the hand-made max-projection from Part I: it answers the same "where are the cells" question, but
        by neighbor-correlation rather than peak brightness, which suppresses the tissue-wide glow that
        fooled our background boxes.
        """
    )
    return


@app.cell
def _(Cn, nu):
    nu.image_fig(Cn, title="Correlation image Cn — pixels that co-fluctuate with their neighbors",
                 colorscale="Viridis", colorbar_title="local corr", height=520)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 9. One source at a time — the footprint viewer

        **Why.** After demixing, each neuron is described by two things: where it sits and when it fires.
        Looking at them side by side is the clearest way to understand what a single demixed "source"
        actually is, and how it improves on the hand-drawn box.

        **Definitions.** CNMF's spatial output `A` is a matrix with **one row per neuron** and **one
        column per pixel** (`202 × 360000`). Take one row, reshape it back to the `600 × 600` image, and
        you recover that neuron's **spatial footprint**: the exact pixels the demixing assigned to that
        one cell — its learned shape, not a box. Its matching **calcium trace** `C[:, k]` is the same
        source's brightness over time, the demixed analog of the trace we read by hand in Part I.

        **Method.** The slider selects a neuron index `k` (0 to 201). The left panel calls
        `nu.footprint(A, k, img_shape)`, which pulls row `k` out of `A` and reshapes it to the image
        (input: the matrix and an index; output: a `600 × 600` footprint image). The right panel plots
        the column `C[:, k]`, that neuron's trace. Step through the sources; each one is a *where* paired
        with a *when*. Notice how compact and cell-shaped the footprints are compared to a rectangle.
        """
    )
    return


@app.cell
def _(mo, n_neurons):
    neuron_ind = mo.ui.slider(0, n_neurons - 1, value=148, step=1,
                              label="neuron index (source)", debounce=True, full_width=True)
    return (neuron_ind,)


@app.cell
def _(A, C, img_shape, mo, neuron_ind, nu):
    _k = neuron_ind.value
    _fp = nu.footprint(A, _k, img_shape)
    _fig_fp = nu.image_fig(_fp, title=f"Footprint A[{_k}] — where source {_k} lives",
                           colorscale="Viridis", colorbar_title="weight", height=420)
    _fig_tr = nu.trace_fig(None, C[:, _k], title=f"Calcium trace C[:, {_k}] — when it fires",
                           xlabel="Time (frames)", ylabel="calcium (a.u.)", height=420)
    _fig_tr.update_traces(line=dict(color="#444444", width=1))
    mo.vstack([neuron_ind, mo.hstack([_fig_fp, _fig_tr], widths=[1, 1])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 10. All sources at once — the footprint montage

        **Why.** Viewing one footprint at a time shows what a source is, but not how the whole population
        tiles the tissue. A single combined image lets us check that the 202 sources are spread across
        the field of view and cleanly separated from one another — the thing a hand-drawn box could never
        guarantee.

        **Method.** Peak-normalize every footprint (divide each by its own maximum, so a dim cell and a
        bright cell count equally), then take the **maximum across all 202 sources** at each pixel. The
        result shows every neuron's territory laid over the field of view. Compare it back to `Cn` in
        Section 8: the bright blobs in the correlation image should reappear here as separated footprints.
        The helper `nu.footprint_montage(A, img_shape)` does the normalization and max-projection
        (input: the footprint matrix and image shape; output: one summary image).
        """
    )
    return


@app.cell
def _(A, img_shape, nu):
    nu.image_fig(nu.footprint_montage(A, img_shape),
                 title="Footprint montage — max projection of all 202 peak-normalized sources",
                 colorscale="Viridis", colorbar_title="peak-norm weight", height=520)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 11. Who are these 202 cells? A population profile

        **Why.** Demixing hands us 202 sources, but they are not interchangeable. Some are large, some
        tiny; some fire constantly, most are nearly silent. Before analyzing the population we should
        know its makeup — the neural version of the exploratory data analysis we did on behavioral
        features in Week 1. This also tells us which cells will drive any downstream result.

        **Definitions and method.** For each neuron we compute three simple properties, all as vectorized
        reductions over the CNMF matrices:

        - **Footprint area** — the number of pixels above 10% of the footprint's peak
          (`(A > 0.1 * peak).sum(axis=1)`): how physically large the cell is.
        - **Peak calcium** — the largest value of its raw trace (`C.max(axis=0)`): how big its strongest
          transient is.
        - **Large-event count** — how many times its z-scored trace crosses `z = 5` (a rising-edge count):
          how often it fires strongly across the whole 9-minute recording.

        The left panel is a **2-D density** (`kde2d_fig`) of footprint area against peak calcium — where
        neurons concentrate in that plane, with every neuron as a point you can hover (the label is its
        index). The right panel is an **empirical CDF** (`ecdf_fig`) of the large-event count: read off,
        for any count on the x-axis, the fraction of neurons at or below it. The curve rises steeply and
        then flattens, which tells you most cells fire rarely while a minority carry most of the events —
        sparse coding, seen directly.
        """
    )
    return


@app.cell
def _(A, C, C_z, np):
    # Three per-neuron properties, each a vectorized reduction (no Python loop over neurons).
    _peak = A.max(axis=1, keepdims=True)                       # (202, 1) each footprint's own max
    neuron_area = (A > 0.1 * _peak).sum(axis=1)                # (202,) pixels above 10% of peak
    neuron_peak = C.max(axis=0)                                # (202,) largest raw calcium value
    _above = C_z > 5.0                                         # (202, T) boolean: strongly active frames
    neuron_events = (_above[:, 1:] & ~_above[:, :-1]).sum(axis=1)   # (202,) rising z>5 crossings
    return neuron_area, neuron_events, neuron_peak


@app.cell
def _(mo, n_neurons, neuron_area, neuron_events, neuron_peak, np, nu):
    _idx = np.arange(n_neurons)
    _dens = nu.kde2d_fig(neuron_area, neuron_peak, hover=_idx,
                         xlabel="footprint area (px)", ylabel="peak calcium (a.u.)",
                         title="footprint size vs peak activity (one dot = one neuron)", height=460)
    _ecdf = nu.ecdf_fig(neuron_events, xlabel="large events (z>5 crossings, whole recording)",
                        ylabel="fraction of neurons at or below",
                        title="most cells fire rarely; a few carry the events", height=460)
    mo.hstack([_dens, _ecdf], widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(mo, neuron_area, neuron_events, neuron_peak, np):
    mo.md(
        f"""
        **Reading the profile.** Footprint area has a median of about **{int(np.median(neuron_area))} px**
        (middle-half range ≈ {int(np.percentile(neuron_area, 25))}–{int(np.percentile(neuron_area, 75))}),
        so the sources really are cell-sized and reasonably uniform — not stray specks or huge blobs.
        Peak calcium spans a wide range (median ≈ **{np.median(neuron_peak):.1f}**, up to
        {neuron_peak.max():.1f}): some cells fire hard, many barely. The event CDF makes the sparsity
        explicit — **{int((neuron_events == 0).sum())} of the 202 neurons never cross z = 5** in the
        entire recording, while the busiest fires **{int(neuron_events.max())}** times. A handful of
        active cells sit in the tail. Keep that in mind for the next section: when we look for temporal
        structure, only the minority of active cells can contribute to it.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 12. The population raster — every source, all the time

        **Why.** We now have 202 traces. To study the population as a whole we need to see all of them
        together and spot moments when many neurons are active at once.

        **Definition.** A **raster** is an image of the whole population's activity: **one row per neuron,
        one column per frame**, brightness showing how active each neuron is at each moment. Bright
        vertical smears are frames when many neurons fire together; horizontal streaks are individual
        cells that stay active for a while.

        **Method.** Stack all 202 traces into a matrix and **z-score each row** (subtract the neuron's
        mean, divide by its standard deviation) so a small quiet cell is on the same footing as a loud
        one — this is the matrix `C_z` built when we loaded the data. The **contrast** slider sets the
        color ceiling (`zmax`): turn it down to bring out weak transients, turn it up to keep only the
        largest calcium events.
        """
    )
    return


@app.cell
def _(mo):
    raster_zmax = mo.ui.slider(2.0, 12.0, value=6.0, step=0.5,
                               label="contrast ceiling (raster zmax, z-units)",
                               debounce=True, full_width=True)
    return (raster_zmax,)


@app.cell
def _(C_z, mo, np, nu, raster_zmax):
    # Downsample columns for a snappy display (compute stays full-res elsewhere).
    _T = C_z.shape[1]
    _step = max(1, _T // 1500)
    _disp = C_z[:, ::_step]
    _x = np.arange(0, _T, _step)
    _fig = nu.raster_fig(_disp, title="Population raster — z-scored C.T (all 202 sources)",
                         xlabel="Time (frames)", ylabel="Neuron", colorscale="Viridis",
                         zmin=0.0, zmax=float(raster_zmax.value), colorbar_title="z", height=460)
    _fig.data[0].x = _x
    mo.vstack([raster_zmax, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 13. Stacked traces — reading sources one line at a time

        **Why.** A raster shows *when* activity happens but hides the *shape* of each event. Plotting the
        traces as separate stacked lines lets us see the form of individual calcium transients — the
        fast-rise, slow-decay signature that tells us each bump is a burst of firing, not noise.

        **Method.** Take the first `N` sources, min-max normalize each to the range `[0, 1]`, and
        **offset every trace vertically** so they do not overlap. Each line is one demixed neuron. The
        slider sets how many sources to stack.
        """
    )
    return


@app.cell
def _(mo):
    n_stack = mo.ui.slider(10, 50, value=30, step=5,
                           label="number of sources to stack", debounce=True, full_width=True)
    return (n_stack,)


@app.cell
def _(C, go, mo, n_frames, n_stack, np):
    _n = int(n_stack.value)
    _sub = C[:, :_n].astype(float)
    _mn = _sub.min(axis=0, keepdims=True)
    _mx = _sub.max(axis=0, keepdims=True)
    _norm = (_sub - _mn) / np.where(_mx - _mn == 0, 1.0, _mx - _mn)   # (T, n) in [0, 1]
    _step = max(1, n_frames // 3000)
    _t = np.arange(0, n_frames, _step)
    _fig = go.Figure()
    for _j in range(_n):
        _fig.add_scatter(x=_t, y=_norm[::_step, _j] + _j * 0.8, mode="lines",
                         line=dict(width=1), showlegend=False, hoverinfo="skip")
    _fig.update_layout(template="plotly_white", height=560, margin=dict(l=10, r=10, t=40, b=10),
                       title=f"First {_n} sources — min-max normalized, offset by 0.8")
    _fig.update_xaxes(title="Time (frames)", range=[0, n_frames])
    _fig.update_yaxes(title="source (offset)", showticklabels=False)
    mo.vstack([n_stack, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 14. The neural sequence — sorting sources by *when* they fire

        **Why.** CNMF returns the sources in an arbitrary order, so the raw raster looks like scattered
        speckle even if the population has real temporal structure. If we reorder the neurons by *when*
        they first become active, we can test whether the population fires as a **sequence** — one cell
        after another, like a wave.

        **Definition.** A **sequence** here means the neurons activate in a consistent order in time. When
        a sequence is present and the rows are sorted by activation time, the raster's activity collapses
        onto a **diagonal**: early-firing neurons at the bottom, late-firing at the top.

        **Method.** Pick a window of the recording. Within it, order the neurons by the **time of their
        first large calcium event** (the first frame each z-scored trace crosses a threshold), using
        `nu.sequence_sort` (input: a raster window and a threshold; output: a permutation that reorders
        the rows). The **left** panel shows the window in the raw CNMF order; the **right** panel shows
        the same window after sorting. The default window is centered on the arena-entry moment from the
        2025 analysis, where the striatum becomes active. The sliders move the window, change its length,
        and set the activation threshold.
        """
    )
    return


@app.cell
def _(ENTRY, WIN_LEN, mo, n_frames):
    win_start = mo.ui.slider(0, n_frames - 600, value=ENTRY, step=30,
                             label="window start (frame)", debounce=True, full_width=True)
    win_len = mo.ui.slider(600, 8000, value=WIN_LEN, step=100,
                           label="window length (frames)", debounce=True, full_width=True)
    seq_thresh = mo.ui.slider(2.0, 8.0, value=5.0, step=0.5,
                              label="activation threshold (z)", debounce=True, full_width=True)
    return seq_thresh, win_len, win_start


@app.cell
def _(C_z, mo, n_frames, np, nu, seq_thresh, win_len, win_start):
    from scipy.stats import spearmanr

    def _seqness(raster, thr):
        # |Spearman| between row position and first-crossing time: 0 = no order, 1 = perfect diagonal
        _first = np.argmax(raster > thr, axis=1)
        _r, _ = spearmanr(np.arange(raster.shape[0]), _first)
        return 0.0 if np.isnan(_r) else abs(float(_r))

    _s = int(win_start.value)
    _e = min(_s + int(win_len.value), n_frames)
    _thr = float(seq_thresh.value)
    _win = C_z[:, _s:_e]
    _order = nu.sequence_sort(_win, thresh=_thr)
    _sorted = _win[_order]

    _q_un = _seqness(_win, _thr)
    _q_so = _seqness(_sorted, _thr)

    # downsample columns for display only
    _step = max(1, _win.shape[1] // 1200)
    _xd = np.arange(_s, _e, _step)
    _left = nu.raster_fig(_win[:, ::_step], title=f"unsorted  ·  sequenceness = {_q_un:.2f}",
                          xlabel="Time (frames)", ylabel="Neuron (CNMF order)",
                          colorscale="Viridis", zmin=0.0, zmax=6.0, colorbar_title="z", height=460)
    _left.data[0].x = _xd
    _right = nu.raster_fig(_sorted[:, ::_step],
                           title=f"sorted by first activation  ·  sequenceness = {_q_so:.2f}",
                           xlabel="Time (frames)", ylabel="Neuron (sequence order)",
                           colorscale="Viridis", zmin=0.0, zmax=6.0, colorbar_title="z", height=460)
    _right.data[0].x = _xd
    mo.vstack([mo.hstack([win_start, win_len, seq_thresh]),
               mo.hstack([_left, _right], widths=[1, 1])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Reference — CNMF, and the limits of the PCA analogy": mo.md(
            r"""
            **The method.** Pnevmatikakis et al. 2016, *Neuron* 89(2):285–299, "Simultaneous Denoising,
            Deconvolution, and Demixing of Calcium Imaging Data" (**CNMF**). The one-photon variant used
            here is **CNMF-E** (Zhou et al. 2018, *eLife* 7:e28728 — the same striatal miniscope dataset
            Part I draws from). CNMF factors the movie `Y ≈ A · C + b` into non-negative **spatial
            footprints `A`** and **temporal traces `C`** (plus a background term `b`). This is a
            constrained matrix factorization, the same family of methods as the PCA decomposition you ran
            on behavior in NB3.

            **The shared mathematics.** Both PCA and CNMF write a data matrix as a **low-rank product of a
            spatial factor and a temporal factor**. PCA picks orthogonal directions of maximum variance;
            CNMF picks **non-negative, spatially localized** factors so each component is a physically
            plausible cell. Non-negativity plus a sparse deconvolution model is what turns "a component"
            into "a neuron."

            **Note on `S`.** The file also carries `S`, CNMF's **deconvolved spike estimate**.
            Deconvolution is calibrated for two-photon data; for **one-photon** miniscope recordings like
            this one, `S` is not validated — the spike times are a model output, not ground truth. We show
            `C` (the calcium) and do not report `S` as spike counts.

            **Limits of the analogy.** A PCA component is a *statistical* axis: abstract, can be negative,
            need not correspond to anything real. A CNMF footprint is a *physical claim*: "a cell is
            here." That is a stronger, testable statement, and it can be wrong — two adjacent cells can be
            merged into one source, or one cell split into two, and the variance explained will not tell
            you. Demixing is only as good as the footprints, and footprints are inferred, not observed.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 15. Exercise — does sorting actually reveal a sequence?

        **Python skill practiced: writing a small scoring function and calling a library.** You will pass
        arguments into a function you were handed (`_sequenceness`) and read a number out — the same
        pattern as calling `scipy.stats.spearmanr` or an sklearn model. No new indexing; the focus is on
        composing a metric from pieces.

        **The question.** Around arena entry, does the striatal population fire as a temporal sequence? If
        it does, ordering the neurons by their first activation time should turn a formless raster into a
        diagonal one, and a "sequenceness" score should rise well above the unsorted baseline.

        **What you have.**

        - `C_z` — the `(202, n_frames)` z-scored population raster (built when we loaded the data).
        - `ENTRY` (= 8985) and `WIN_LEN` (= 5400) — the arena-entry window on the imaging clock.
        - `nu.sequence_sort(raster, thresh=5.0)` — returns a permutation ordering neurons by their first
          supra-threshold crossing.
        - `scipy.stats.spearmanr` and `np.argmax`.

        **Definition of the score.** *Sequenceness* is the absolute Spearman correlation between a
        neuron's **row position** and its **first-crossing frame** (`np.argmax(win > thr, axis=1)`). A
        value near 0 means no temporal order; a value near 1 means a clean diagonal.

        **Your job.** The cell below already builds the window `_win`, the threshold `_thr`, the
        `_sequenceness` helper, and the sort `_order`. You only fill in the **two marked lines** — the
        score before and after sorting.

        **What you should see.** The self-check below turns green when the sorted score lands in the band
        `[0.65, 0.95]` and clears the unsorted baseline by a wide margin. In numbers, expect
        `seq_unsorted` near `0.05` (no order in the raw CNMF order) and `seq_sorted` near `0.79` (a clear
        diagonal).
        """
    )
    return


@app.cell
def _(C_z, ENTRY, WIN_LEN, np, nu):
    # ------------------------------------------------------------------ YOUR CODE (edit the 2 marked lines)
    from scipy.stats import spearmanr as _spearmanr

    _thr = 5.0
    _win = C_z[:, ENTRY:ENTRY + WIN_LEN]              # fixed arena-entry window (202 neurons x 5400 frames)

    def _sequenceness(_raster):
        # first activation frame per neuron, then |Spearman(row position, first frame)|
        _first = np.argmax(_raster > _thr, axis=1)
        _r, _ = _spearmanr(np.arange(_raster.shape[0]), _first)
        return 0.0 if np.isnan(_r) else abs(float(_r))

    _order = nu.sequence_sort(_win, thresh=_thr)      # permutation: rows ordered by first crossing

    # LINE 1: score the window in its ORIGINAL CNMF order. Pass `_win` (unsorted) to `_sequenceness`.
    #     WHY: this is the baseline. CNMF's row order is arbitrary, so a near-zero score here means the
    #     population has no apparent temporal structure BEFORE we do anything.
    seq_unsorted = _sequenceness(_win)
    # LINE 2: score the SAME window after sorting its rows by first activation. Pass `_win[_order]`.
    #     WHY: if the score jumps up, sorting did not invent structure -- it REVEALED an ordering that was
    #     already latent in the data. That jump is the whole point of the exercise.
    seq_sorted = _sequenceness(_win[_order])
    # ---------------------------------------------------------------------------------------------
    return seq_sorted, seq_unsorted


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show solution": mo.md(
            r"""
            ```python
            from scipy.stats import spearmanr
            thr = 5.0
            win = C_z[:, ENTRY:ENTRY + WIN_LEN]

            def sequenceness(raster):
                first = np.argmax(raster > thr, axis=1)      # first activation frame per neuron
                r, _ = spearmanr(np.arange(raster.shape[0]), first)
                return abs(r)

            order = nu.sequence_sort(win, thresh=thr)         # order by first crossing
            seq_unsorted = sequenceness(win)                  # ~0.05  (no structure in CNMF order)
            seq_sorted   = sequenceness(win[order])           # ~0.79  (a clean diagonal)
            ```

            **What you should find.** The unsorted window has near-zero sequenceness (~0.05 — CNMF returns
            the sources in an arbitrary order), while after `sequence_sort` it rises to about **0.79**. The
            jump is the point: sorting did not *create* structure, it **revealed** a temporal sequence that
            was already present. The sorted score is ~0.79 rather than a perfect 1.0 because about 144 of
            the 202 neurons never cross the threshold in this window; among the ~58 that do fire, the
            ordering is essentially perfect. (This is exactly the sparsity you saw in the Section 11 event
            CDF.)
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo, seq_sorted, seq_unsorted):
    # Self-check with a tolerance band pinned from the real data:
    #   seq_unsorted ~ 0.05,  seq_sorted ~ 0.7938.
    # Pass = sorting lands in [0.65, 0.95] AND clears the unsorted baseline by a wide margin.
    _in_band = 0.65 <= seq_sorted <= 0.95
    _gain = seq_sorted - seq_unsorted > 0.4
    _ok = _in_band and _gain
    _c = "#e8f5e9" if _ok else "#ffebee"
    _b = "#2e7d32" if _ok else "#c62828"
    _m1 = (f"sorted sequenceness = {seq_sorted:.3f} — in the expected band [0.65, 0.95]"
           if _in_band else
           f"sorted sequenceness = {seq_sorted:.3f} — outside [0.65, 0.95]; check window/threshold")
    _m2 = (f"sorting beats the unsorted baseline ({seq_unsorted:.3f}) by "
           f"{seq_sorted - seq_unsorted:.3f} — a real sequence was revealed"
           if _gain else
           f"gain over baseline = {seq_sorted - seq_unsorted:.3f} is too small — did you sort the raster?")
    _head = "PASS — the sort reveals a neural sequence" if _ok else "Not yet — fix the flagged line"
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_b};padding:12px 16px;border-radius:6px">
        <b style="color:{_b}">{_head}</b><br>
        {_m1}<br>{_m2}<br>
        <span style="font-size:0.9em;color:#555">Tolerance band pinned from the real recording:
        sorted ≈ 0.79, unsorted ≈ 0.05. The score measures |Spearman(row, first-activation)| — how
        diagonal the raster is.</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## The answer, and the next question

        **What we asked.** How do we turn a movie of glowing cells into a signal we can analyze?

        **What we answered.** Twice, from hand-made to learned. In Part I we took a real striatal movie —
        250,000 pixel values per frame — removed its static background, built a max-projection map of
        active cells, and read out one cell's trace by averaging a box of pixels; the cell-vs-background
        variance comparison showed the box choice *is* the measurement. In Part II we replaced the box
        with **CNMF**, which learned a spatial footprint and a calcium trace for all **202** neurons at
        once, separating overlapping sources that a rectangle could never resolve. We profiled the
        population (cell-sized footprints, sparse firing), and by sorting the sources by *when* they fire
        we turned a shapeless raster into a diagonal one — a temporal sequence that was already latent in
        the data. A movie of light is now a set of per-neuron time series we can do statistics on.

        **The next question.** We have hundreds of clean neural signals, but we do not yet know what any
        of them *mean*. Does an individual neuron's activity correspond to something specific the animal
        is doing? The clearest place to start is space: **do individual neurons fire when the animal is in
        a particular location — and can we read the animal's position back out of the neurons?** That is
        the tuning question, and it is where the next notebook goes, with place and grid cells.
        """
    )
    return


if __name__ == "__main__":
    app.run()
