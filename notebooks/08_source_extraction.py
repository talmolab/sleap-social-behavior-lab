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
        # NB08 · Source extraction — separating overlapping cells into neurons

        **Week 2 · working with neural recordings**

        ### The question we are carrying forward

        In the previous notebook we asked: *how do we turn a raw fluorescence movie into a
        signal we can analyze?* We answered it the simplest way — draw a box over a patch of
        tissue and average the pixels inside it, frame by frame, to get one **calcium trace**
        for that patch. That works, but it hides a problem. A single patch of the field of
        view is not one cell. It is many cells, packed together, whose light lands on the
        same pixels. The trace we pulled out is a **mixture** of everything glowing in that
        box.

        So this notebook asks the next question:

        > **How do we separate overlapping cells into individual neurons?**

        We want to go from one blurry patch-average to a clean list of *single* neurons, each
        with its own location in the tissue and its own activity over time. This is called
        **source separation** or **demixing**, and it is one of the central operations in
        systems neuroscience. By the end you will have 202 individual neurons pulled out of a
        single 9-minute movie, and you will test whether they fire in an ordered sequence.

        ### This is the same idea we already used on behavior

        We study social behavior and its neural basis, and the operation in this notebook is not
        new to us — we already ran it on the body. Back
        in **notebook 4** we took a wide, redundant matrix of behavior features and reduced it
        with **PCA**, replacing many correlated measurements with a few underlying components.
        The verb there was *decompose a mixture into its sources*. That is exactly the verb
        here. The only change is the object: instead of decomposing a matrix of behavior
        features, we decompose a movie of glowing tissue. Same mathematics, different data.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Definitions (read these before the method)

        These five terms are the whole notebook. Everything after this is an illustration of
        one of them.

        - **Mixture.** A signal that is the *sum of several things at once*. In a calcium
          movie, the light from many neurons falls on overlapping pixels, so the brightness of
          any single pixel over time is a mixture: neuron A's activity **plus** neuron B's
          activity **plus** background, all added together. You cannot read one neuron off one
          pixel.
        - **Source separation (demixing).** Splitting a mixture back into the separate things
          that made it up. Given the movie — thousands of mixed pixels — recover the
          individual neurons behind them. "Source" here means "one neuron."
        - **Spatial footprint (`A`).** For one neuron, the set of pixels it occupies and how
          strongly it contributes to each. Reshaped back to an image, it is a small blob
          showing *where* that one cell sits in the field of view. It is the imaging analog of
          a PCA component's loading vector: a spatial pattern that says "this source lives
          here."
        - **Calcium trace (`C`).** For one neuron, its brightness over time — a time series
          showing *when* that neuron is active. Brightness rises sharply when the cell fires
          (calcium floods in) and decays slowly afterward. This is the honest signal we trust.
        - **Correlation image (`Cn`).** A single summary picture of the whole movie in which
          each pixel is colored by how much it rises and falls *together with its immediate
          neighbors*. A lone flickering pixel is noise and stays dark; a blob of pixels that
          brighten in lockstep is a candidate cell body and lights up. It is where demixing
          starts looking for cells.

        ### The method, named

        The algorithm that performs the separation is **CNMF** — *constrained non-negative
        matrix factorization*. It writes the movie `Y` as a product of two factors,
        `Y ≈ A · C`: the **spatial footprints** `A` (the *where*) times the **temporal traces**
        `C` (the *when*), plus a background term. PCA writes a data matrix as a product of a
        spatial factor and a temporal factor too — CNMF is the same family of method. The
        difference is the word *constrained*: CNMF forces every factor to be non-negative and
        every footprint to be a small compact blob, so each component comes out looking like a
        real cell instead of an abstract statistical axis. We unpack exactly what that
        factorization means in Section 4, on a single shared pixel.

        The recording is one striatal session, `221007_4-0_D2`: **202 demixed neurons** across
        about **16,800 frames at 30 fps** (roughly 9 minutes). Let us load it and look.
        """
    )
    return


@app.cell
def _(nu):
    _d = nu.load_cnmf()
    A = _d["A"]
    C = _d["C"]
    Cn = _d["Cn"]
    S = _d["S"]
    Fs = _d["Fs"]
    img_shape = _d["img_shape"]
    n_neurons = _d["n_neurons"]
    n_frames = _d["n_frames"]
    # z-scored population raster, one row per neuron (per-neuron mean/std across time).
    # Computed once here and reused everywhere below so a small quiet cell and a loud one
    # are on the same footing when we compare them.
    C_z = nu.zscore(C.T, axis=1)
    # The 2025 behavior-clock "arena entry" frame, converted onto the imaging clock
    # (behavior 25 fps -> imaging 30 fps). This anchors the sequence window and the exercise.
    ENTRY = int(7488 * (30 / 25))   # -> 8985
    WIN_LEN = 3 * 60 * 30           # 3 minutes at 30 fps -> 5400 frames
    return A, C, C_z, Cn, ENTRY, Fs, S, WIN_LEN, img_shape, n_frames, n_neurons


@app.cell(hide_code=True)
def _(Fs, mo, n_frames, n_neurons):
    mo.md(
        f"""
        ---
        ## 1. The correlation image — where the cells are

        **Why.** Before any separation runs, we need a rough picture of where the cells even
        are. The correlation image is the standard first look at a calcium movie, and it is
        also what CNMF uses to place its initial guesses. If you cannot see blobs here, there
        is nothing to demix.

        **Definition (recall).** The **local correlation image `Cn`** colors each pixel by how
        strongly its brightness fluctuates *together with its immediate neighbors* across the
        whole recording. One bright pixel on its own is usually electrical noise — its
        neighbors do not follow it. A *blob* of pixels that brighten and dim in lockstep is a
        candidate cell body, because the pixels covering one neuron rise and fall together.

        **Method.** The figure below is `Cn` for this recording of **{n_neurons} neurons**
        across **{n_frames:,} frames** at **{Fs:.0f} fps** ({n_frames / Fs / 60:.1f} min). The
        bright rings and disks are the sources CNMF will separate into individual footprints
        and traces. Look at how many of them overlap or touch — that overlap is exactly the
        mixing problem this notebook exists to undo.
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
        ## 2. What demixing produces — one source, a *where* paired with a *when*

        **Why.** After CNMF runs, each neuron is described by exactly two things: where it
        sits and when it fires. Looking at those two side by side is the clearest way to
        understand what a single demixed "source" actually is — and to convince yourself it is
        a cell and not a statistical artifact.

        **Definitions.** CNMF's spatial output `A` is a matrix with **one row per neuron** and
        **one column per pixel** (here `202 × 360000`, because the field of view is
        `600 × 600 = 360000` pixels). Take one row, reshape it back to the `600 × 600` image,
        and you recover that neuron's **spatial footprint** — the pixels demixing assigned to
        that one cell. Its matching column of `C` is the same source's **calcium trace**, its
        brightness over time.

        **Method.** The slider selects a neuron index `k` (0 to 201). The left panel calls
        `nu.footprint(A, k, img_shape)` — purpose: pull row `k` out of `A` and reshape it to
        the image; input: the matrix and an index; output: a `600 × 600` footprint image. The
        right panel plots the column `C[:, k]`, that neuron's trace. Drag the slider and watch
        the pairing: a compact blob in one corner of the tissue, and a trace that sits flat and
        then spikes. The spikes are **calcium transients** — a fast rise when the cell fires,
        followed by a slow decay as the calcium is cleared. Try a few values; you will notice
        some cells fire often and some barely fire at all. We quantify that split just below.
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
        ### 2b. How big is a source? — a first look at the footprints as a population

        **Why.** Before trusting the 202 footprints, we should check that they are the right
        *size* to be cells. If a "footprint" covered a quarter of the field of view it would
        be background contamination, not a neuron; if it were a single pixel it would be noise.
        A quick distribution tells us whether the whole population is physically plausible.

        **Method.** For each neuron we count the pixels its footprint actually occupies
        (`(A > 0).sum(axis=1)`), then convert that area to an **effective radius** in pixels,
        `sqrt(area / π)` — the radius of a disk with the same area. The plot is an **empirical
        cumulative distribution (ECDF)**: read it as "what fraction of sources have a radius at
        or below this value." A tight band of radii, all in the tens-of-pixels range, is what a
        clean population of cell bodies should look like.
        """
    )
    return


@app.cell
def _(A, C_z, np):
    # Per-source footprint size and per-source peak activity — computed once, reused below.
    fp_nonzero = (A > 0).sum(axis=1)                 # (202,) pixels occupied by each footprint
    fp_radius = np.sqrt(fp_nonzero / np.pi)          # (202,) effective radius in pixels
    peakz = C_z.max(axis=1)                          # (202,) tallest z-scored transient per source
    active_mask = peakz > 5.0                        # a source "fires" if it ever crosses z = 5
    return active_mask, fp_nonzero, fp_radius, peakz


@app.cell
def _(fp_radius, np, nu):
    _med = float(np.median(fp_radius))
    nu.ecdf_fig(fp_radius,
                title=f"Footprint effective radius — all 202 sources (median ≈ {_med:.0f} px)",
                xlabel="effective radius (pixels)", ylabel="cumulative fraction of sources",
                height=400)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The radii sit in a compact band around the median (roughly 15–20 px), with no source
        collapsing to a point or swelling to fill the frame. That is the size of a striatal
        cell body at this magnification — the footprints are physically plausible cells, not
        noise or background. With the *where* checked, we can lay all 202 out at once.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 3. All sources at once — the footprint montage

        **Why.** Viewing one footprint at a time shows what a source is, but not how the whole
        population tiles the tissue. A single combined image lets us check that the 202 sources
        are spread across the field of view and separated from one another, and lets us compare
        back to `Cn`.

        **Method.** Peak-normalize every footprint (divide each by its own maximum, so a dim
        cell and a bright cell count equally), then take the **maximum across all 202 sources**
        at each pixel. The result shows every neuron's territory laid over the field of view.
        The helper `nu.footprint_montage(A, img_shape)` does the normalization and
        max-projection — input: the footprint matrix and image shape; output: one summary
        image. Compare it back to `Cn` in Section 1: the bright blobs in the correlation image
        should reappear here as cleanly outlined footprints, now each assigned to exactly one
        source.
        """
    )
    return


@app.cell
def _(A, img_shape, nu):
    nu.image_fig(nu.footprint_montage(A, img_shape),
                 title="Footprint montage — max projection of all 202 peak-normalized sources",
                 colorscale="Viridis", colorbar_title="peak-norm weight", height=560)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 4. The mixture, made concrete — demixing a single shared pixel

        **Why.** Everything so far has taken CNMF's output on faith. This section shows the
        factorization actually doing its job on one pixel, so "demixing" stops being a word and
        becomes something you can watch. This is the conceptual heart of the notebook, and it is
        where the parallel to PCA is exact.

        **The setup.** In the montage above, many footprints touch or overlap. Pick a pixel
        that sits **inside two footprints at once** — it belongs partly to source 27 and partly
        to source 30. What does that pixel's brightness look like over time? By construction it
        is a **mixture**: CNMF models the movie as `Y ≈ A · C`, which means the brightness of a
        single pixel `p` at time `t` is

        $$ Y[t, p] \;\approx\; \sum_{k} A[k, p]\; C[t, k]. $$

        In words: the pixel's value is **every neuron's footprint-weight at that pixel times
        that neuron's trace, summed over all neurons.** For our shared pixel almost all of that
        sum comes from just two terms — source 27 and source 30 — because only those two have
        appreciable footprint weight there.

        **What the figure shows.** Left: the two footprints, with a white marker on the shared
        pixel. Right: the pixel's modeled brightness (black) decomposed into the part CNMF
        attributes to source 27 (`A[27, p]·C[:, 27]`) and the part it attributes to source 30
        (`A[30, p]·C[:, 30]`). The two coloured traces fire at *different* times — their traces
        correlate at only about −0.07 — so the black mixture has bumps from both. **Demixing is
        the act of pulling those two coloured curves back out of the black one.** That is
        precisely what PCA did in notebook 4: it wrote each observed variable as a weighted sum
        of a few underlying components. Here the "components" are constrained to be non-negative
        localized cells, so each one is a neuron.
        """
    )
    return


@app.cell
def _(A, C, go, img_shape, mo, np, nu):
    # A pixel shared by two footprints, found by scanning for pixels where exactly two
    # peak-normalized footprints exceed 0.5 and both owners actually fire. Fixed here for a
    # stable, readable example; the owners are sources 27 and 30.
    _H, _W = img_shape
    _pix = 50726                      # flat pixel index (row 84, col 326)
    _r, _c = divmod(_pix, _W)
    _o0, _o1 = 27, 30                 # the two neurons whose footprints cover this pixel

    # --- left: the two footprints, with the shared pixel marked ---
    _fp0 = nu.footprint(A, _o0, img_shape)
    _fp1 = nu.footprint(A, _o1, img_shape)
    _both = np.maximum(_fp0 / _fp0.max(), _fp1 / _fp1.max())   # peak-normed overlay of the pair
    _fig_fp = nu.image_fig(_both, title=f"Footprints of sources {_o0} + {_o1} (shared pixel marked)",
                           colorscale="Viridis", colorbar_title="peak-norm weight", height=430)
    _fig_fp.add_scatter(x=[_c], y=[_r], mode="markers",
                        marker=dict(color="white", size=9, symbol="x"),
                        name="shared pixel", showlegend=False)

    # --- right: the pixel's modeled brightness decomposed into the two sources ---
    _contrib0 = A[_o0, _pix] * C[:, _o0]              # source 27's contribution at this pixel
    _contrib1 = A[_o1, _pix] * C[:, _o1]              # source 30's contribution at this pixel
    _total = A[:, _pix] @ C.T                          # full modeled pixel value = sum over all sources
    _step = max(1, C.shape[0] // 2500)
    _t = np.arange(0, C.shape[0], _step)
    _fig_tr = go.Figure()
    _fig_tr.add_scatter(x=_t, y=_total[::_step], mode="lines", name="pixel mixture (all sources)",
                        line=dict(color="#222222", width=1.4))
    _fig_tr.add_scatter(x=_t, y=_contrib0[::_step], mode="lines", name=f"from source {_o0}",
                        line=dict(color="#e45756", width=1.2))
    _fig_tr.add_scatter(x=_t, y=_contrib1[::_step], mode="lines", name=f"from source {_o1}",
                        line=dict(color="#4c78a8", width=1.2))
    _fig_tr.update_layout(template="plotly_white", height=430, margin=dict(l=10, r=10, t=50, b=10),
                          title="One pixel's brightness = a mixture of two neurons",
                          legend=dict(orientation="h", y=1.08, x=0))
    _fig_tr.update_xaxes(title="Time (frames)")
    _fig_tr.update_yaxes(title="modeled brightness (a.u.)")
    mo.hstack([_fig_fp, _fig_tr], widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notice the black mixture has red bumps and blue bumps interleaved, at moments the two
        cells were not co-active. A patch-average trace — the kind we built in the previous
        notebook — would hand us exactly this black curve and call it "the signal here." CNMF
        instead returns the red and the blue separately, each a single neuron. That is the
        whole payoff of demixing: **one clean cell where the raw movie only offered a blend.**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 5. The population raster — every source, all the time

        **Why.** We now have 202 clean traces. To study the population as a whole rather than
        one cell at a time, we need to see all of them together and spot moments when many
        neurons are active at once.

        **Definition.** A **raster** is an image of the whole population's activity: **one row
        per neuron, one column per frame**, brightness showing how active each neuron is at each
        moment. Bright vertical smears are frames when many neurons fire together; horizontal
        streaks are individual cells that stay active for a while.

        **Method.** Stack all 202 traces into a matrix and **z-score each row** (subtract the
        neuron's mean, divide by its standard deviation) — this is the matrix `C_z` built at the
        top, and z-scoring is what lets a small quiet cell and a loud one share a color scale.
        The **contrast** slider sets the color ceiling (`zmax`): turn it down to bring out weak
        transients, turn it up to keep only the largest calcium events. In the raw CNMF row
        order the raster looks like scattered speckle — hold that thought, because Section 7
        reorders the rows and the speckle turns into structure.
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
    _fig = nu.raster_fig(_disp, title="Population raster — z-scored C.T (all 202 sources, CNMF order)",
                         xlabel="Time (frames)", ylabel="Neuron", colorscale="Viridis",
                         zmin=0.0, zmax=float(raster_zmax.value), colorbar_title="z", height=460)
    _fig.data[0].x = _x
    mo.vstack([raster_zmax, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 5b. Not every source is active — a look at the traces as a population

        **Why.** The raster hints that most rows are dark most of the time. Before we go
        looking for sequence structure we should know *how many* sources ever really fire,
        because a sequence can only involve the cells that are active in the window. This is
        also a habit worth keeping: characterize your population before you run the fancy
        analysis on it.

        **Method.** For each source we take its tallest z-scored transient, `peakz =
        C_z.max(axis=1)`, and split the population by whether that peak crosses `z = 5` — our
        working definition of "this cell fired at least once." The strip plot shows **every one
        of the 202 sources as an individual point** (hover to read its index and peak), split
        into *active* and *quiet* groups, with a line at each group's mean. You should see a
        clear minority of active sources sitting well above the threshold and a larger quiet
        group hugging low z-values. Those active cells are the ones a sequence can be built
        from.
        """
    )
    return


@app.cell
def _(active_mask, np, nu, peakz):
    _n_active = int(active_mask.sum())
    _n_quiet = int((~active_mask).sum())
    _groups = np.where(active_mask, f"active (peak z>5, n={_n_active})",
                       f"quiet (n={_n_quiet})")
    _order = [f"quiet (n={_n_quiet})", f"active (peak z>5, n={_n_active})"]
    nu.strip_points_fig(peakz, _groups, group_order=_order,
                        colors={_order[0]: "#bab0ac", _order[1]: "#e45756"},
                        hover=np.arange(len(peakz)), ylabel="peak z-scored transient",
                        title="Peak activity per source — a minority of cells carry the events",
                        height=440)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Only about ninety of the 202 sources ever cross `z = 5`; the rest stay quiet across the
        whole nine minutes. That is normal for striatal calcium imaging — sparse, event-driven
        activity — and it tells us the sequence we look for next lives in a subset of the
        population, not all 202 rows.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 6. Stacked traces — reading sources one line at a time

        **Why.** A raster shows *when* activity happens but flattens the *shape* of each event
        into a color. Plotting the traces as separate stacked lines lets us see the form of
        individual calcium transients — the fast rise, the slow decay — that the raster hides.

        **Method.** Take the first `N` sources, min-max normalize each one to the range
        `[0, 1]` (so tall and short cells are comparable), and **offset every trace vertically**
        so they do not overlap. Each line is one demixed neuron. The sharp asymmetric jumps are
        calcium transients: a fast rise when the cell fires, a slow tail as calcium clears. The
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
        ## 7. The neural sequence — sorting sources by *when* they fire

        **Why.** CNMF returns the sources in an arbitrary order, so the raw raster looks like
        scattered speckle even if the population has real temporal structure. If we reorder the
        neurons by *when* they first become active, we can test whether the population fires as
        a **sequence** — one cell after another, like a wave rolling across the tissue.

        **Definition.** A **sequence** here means the neurons activate in a consistent order in
        time. When a sequence is present and the rows are sorted by activation time, the
        raster's activity collapses onto a **diagonal**: early-firing neurons at the bottom,
        late-firing at the top.

        **Method.** Pick a window of the recording. Within it, order the neurons by the **time
        of their first large calcium event** (the first frame each z-scored trace crosses a
        threshold), using `nu.sequence_sort` — input: a raster window and a threshold; output: a
        permutation that reorders the rows. The **left** panel shows the window in the raw CNMF
        order; the **right** panel shows the same window after sorting. Each title reports a
        **sequenceness** score: the absolute Spearman correlation between a neuron's row position
        and its first-activation frame — 0 means no order, 1 means a perfect diagonal. The
        default window is centered on the arena-entry moment from the 2025 analysis, where the
        striatum becomes active. The sliders let you move the window, change its length, and set
        the activation threshold.
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
    from scipy.stats import spearmanr as _sp_seq

    def _seqness(_raster, _thr):
        # |Spearman| between row position and first-crossing time: 0 = no order, 1 = diagonal
        _first = np.argmax(_raster > _thr, axis=1)
        _r, _ = _sp_seq(np.arange(_raster.shape[0]), _first)
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
    mo.md(
        r"""
        ---
        ## 8. Is the sequence real, or did sorting manufacture it?

        **Why.** This is the honest question, and it is easy to get wrong. Sorting rows *by*
        their first-activation time will **always** produce a diagonal — that is what sorting
        does. A high sorted sequenceness on its own proves nothing. So we need a test that can
        come out negative: does the order the population fires in **hold up on data the sort
        never saw?**

        **Method (split-half cross-validation).** Cut the arena-entry window in half. Learn the
        neuron ordering on the **first** half only (`sequence_sort` on frames 0…half). Then,
        *without re-sorting*, ask whether the **second** half is still diagonal under that same
        order — Spearman correlation between the learned row position and each cell's
        first-activation frame in the held-out half, over the cells active in both halves. If
        the sequence is a real property of the population, an order learned on one half predicts
        the other. If sorting merely overfit noise, the held-out correlation collapses to what
        you get from a **random** order.

        **What the figure shows.** The strip plot is the held-out sequenceness under 500 random
        orderings (each a point). The red line is the held-out sequenceness under the order
        *learned from the first half*. The learned order lands near **0.5**, well above the
        random cloud (median near **0.16**). That is real but modest evidence: only a handful of
        cells are active in both halves, so we should not oversell it. The point of the section
        is the method, not a triumphant number — a sequence claim is only worth making if it
        survives a test that could have failed.
        """
    )
    return


@app.cell
def _(C_z, ENTRY, WIN_LEN, np, nu):
    from scipy.stats import spearmanr as _sp_cv

    _win = C_z[:, ENTRY:ENTRY + WIN_LEN]
    _half = WIN_LEN // 2
    _A_half = _win[:, :_half]                         # first half: learn the order here
    _B_half = _win[:, _half:]                          # second half: test the order here

    def _heldout_seqness(_row_order):
        # Under a given row order, how diagonal is the HELD-OUT half? (active-in-B cells only)
        _b = _B_half[_row_order]
        _active = _b.max(axis=1) > 5.0
        _first = np.argmax(_b > 5.0, axis=1)[_active]
        if _active.sum() < 3:
            return 0.0
        _r, _ = _sp_cv(np.arange(int(_active.sum())), _first)
        return 0.0 if np.isnan(_r) else abs(float(_r))

    _learned_order = nu.sequence_sort(_A_half, thresh=5.0)   # order from FIRST half only
    cv_learned = _heldout_seqness(_learned_order)            # held-out score under learned order

    _rng = np.random.RandomState(1)
    cv_shuffle = np.array([_heldout_seqness(_rng.permutation(_win.shape[0]))
                           for _ in range(500)])              # null: random orders
    return cv_learned, cv_shuffle


@app.cell
def _(cv_learned, cv_shuffle, np, nu):
    _groups = np.array(["random order"] * len(cv_shuffle))
    _fig = nu.strip_points_fig(cv_shuffle, _groups, colors={"random order": "#bab0ac"},
                               ylabel="held-out sequenceness",
                               title=("Does the sequence generalize? "
                                      f"learned order = {cv_learned:.2f}  vs  random median "
                                      f"= {np.median(cv_shuffle):.2f}"),
                               height=430)
    # Red line: the held-out score under the order LEARNED on the first half.
    _fig.add_scatter(x=[-0.35, 0.35], y=[cv_learned, cv_learned], mode="lines",
                     line=dict(color="#e45756", width=3), name="learned order",
                     hovertemplate=f"learned order: {cv_learned:.3f}<extra></extra>")
    _fig
    return


@app.cell(hide_code=True)
def _(S, mo, np):
    _s_nonzero = int((np.asarray(S) != 0).sum())
    mo.accordion({
        "Reference — CNMF, the honest limits, and why we do not report spikes here": mo.md(
            rf"""
            **The method.** Pnevmatikakis et al. 2016, *Neuron* 89(2):285–299, "Simultaneous
            Denoising, Deconvolution, and Demixing of Calcium Imaging Data" (**CNMF**). The
            one-photon variant used here is **CNMF-E** (Zhou et al. 2018, *eLife* 7:e28728 — the
            striatal miniscope dataset). CNMF factors the movie `Y ≈ A · C + b` into
            non-negative **spatial footprints `A`** and **temporal traces `C`** plus a background
            term `b`. This is a constrained matrix factorization — the same family as the PCA
            decomposition we ran on behavior in notebook 4.

            **The shared mathematics.** Both PCA and CNMF write a data matrix as a **low-rank
            product of a spatial factor and a temporal factor** (Section 4 showed this on one
            pixel: `Y[t, p] ≈ Σ_k A[k, p] C[t, k]`). PCA picks orthogonal directions of maximum
            variance; CNMF picks **non-negative, spatially localized** factors so each component
            is a physically plausible cell. Non-negativity plus a compact-footprint constraint
            is what turns "a statistical component" into "a neuron."

            **Why we show `C` and not `S`.** The file also carries a key `S`, meant to hold
            CNMF's **deconvolved spike estimate** — an attempt to convert the slow calcium trace
            into the fast spike times underneath it. Two honest caveats. First, deconvolution is
            calibrated for two-photon data; for **one-photon** miniscope recordings like this
            one it is **not validated**, so any `S` would be a model output, not ground truth.
            Second, and concretely, in *this* refined file `S` contains **{_s_nonzero} non-zero
            values** — it is all zeros, i.e. no spike estimate was shipped at all. For both
            reasons we analyze the calcium trace `C` throughout and never report spike counts.

            **Limits of the analogy.** A PCA component is a *statistical* axis: abstract, can be
            negative, need not correspond to anything real. A CNMF footprint is a *physical
            claim*: "a cell is here." That is a stronger, testable statement — and it can be
            wrong. Two adjacent cells can be merged into one source, or one cell split into two,
            and the variance explained will not tell you. Demixing is only as good as the
            footprints, and footprints are inferred, not observed.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 9. Exercise — does sorting actually reveal a sequence?

        **Python skill practiced.** *Calling a function you were given, and calling a library
        function (`scipy.stats.spearmanr`) from inside it.* You will not write the sort or the
        statistic from scratch — those are provided. You will call the scoring function on the
        right inputs and read the result. This is the rung between "write a small function" and
        "run a full analysis pipeline": you compose pieces that already exist.

        **The question.** Around arena entry, does the striatal population fire as a temporal
        sequence? If it does, ordering the neurons by their first activation time should turn a
        formless raster into a diagonal one, and the sequenceness score should rise well above
        the unsorted baseline.

        **What you have in the cell below (already written for you).**

        - `_win` — the fixed arena-entry window, `C_z[:, ENTRY:ENTRY+WIN_LEN]`, shape
          `(202, 5400)`.
        - `_sequenceness(raster)` — returns the absolute Spearman correlation between a
          neuron's row position and its first-crossing frame. Near 0 = no order; near 1 = clean
          diagonal.
        - `_order` — the permutation from `nu.sequence_sort`, i.e. the rows reordered by first
          activation.

        **Your job — fill in the two marked lines.** Each takes exactly one argument: the
        raster to score.

        - **Line 1** should score the window in its **original CNMF order** — pass the raster
          *as is*.
        - **Line 2** should score the window **after applying the sort** — pass the raster with
          its rows permuted by `_order`.

        **What you should see.** The self-check below turns green when the sorted score lands in
        `[0.65, 0.95]` and clears the unsorted baseline by a wide margin. In numbers: expect
        `seq_unsorted` near `0.05` (no order in the raw CNMF order) and `seq_sorted` near `0.79`
        (a clear diagonal). The jump between them is the whole result: sorting did not *create*
        structure, it **revealed** a sequence that was already in the population.
        """
    )
    return


@app.cell
def _(C_z, ENTRY, WIN_LEN, np, nu):
    # ------------------------------------------------------------------ YOUR CODE (edit this cell)
    from scipy.stats import spearmanr as _spearmanr

    _thr = 5.0
    _win = C_z[:, ENTRY:ENTRY + WIN_LEN]              # fixed arena-entry window, shape (202, 5400)

    def _sequenceness(_raster):
        # first activation frame per neuron, then |Spearman(row position, first frame)|
        _first = np.argmax(_raster > _thr, axis=1)
        _r, _ = _spearmanr(np.arange(_raster.shape[0]), _first)
        return 0.0 if np.isnan(_r) else abs(float(_r))

    _order = nu.sequence_sort(_win, thresh=_thr)      # permutation: rows ordered by first crossing

    # TODO line 1 — score the window in the ORIGINAL CNMF order.
    #   WHY: this is the baseline. CNMF numbers its sources arbitrarily, so with the rows in
    #   their native order there should be NO relationship between row index and firing time,
    #   and the score should come out near 0.05. Replace ____ with the unsorted window `_win`.
    seq_unsorted = _sequenceness(_win)
    # TODO line 2 — score the window AFTER sorting the rows by first activation.
    #   WHY: if a real sequence exists, reordering the rows by when each cell fires makes the
    #   raster diagonal, so row position now tracks firing time and the score jumps to ~0.79.
    #   Replace ____ with the SORTED window: index the rows with the permutation, `_win[_order]`.
    seq_sorted = _sequenceness(_win[_order])
    # ---------------------------------------------------------------------------------------------
    return seq_sorted, seq_unsorted


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show solution": mo.md(
            r"""
            ```python
            seq_unsorted = _sequenceness(_win)          # ~0.05  (no structure in CNMF order)
            seq_sorted   = _sequenceness(_win[_order])  # ~0.79  (a clean diagonal)
            ```

            **What you should find.** The unsorted window has near-zero sequenceness (~0.05 —
            CNMF returns the sources in an arbitrary order), while after `sequence_sort` it rises
            to about **0.79**. The jump is the point: sorting did not *create* structure, it
            **revealed** a temporal sequence already present in the population. The sorted score
            is ~0.79 rather than a perfect 1.0 because about 144 of the 202 neurons never cross
            the threshold in this window; among the ~58 that do fire, the ordering is essentially
            perfect. (And Section 8's split-half test is what tells us this ordering is a real
            property of the population, not an artifact of having sorted.)
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
        ## Where this leaves us

        **The question we asked:** how do we separate overlapping cells into individual neurons?
        **The answer:** with a constrained matrix factorization — CNMF — that writes the movie
        `Y ≈ A · C` and returns, for each of 202 sources, a compact spatial footprint (`A`, the
        *where*) and a calcium trace (`C`, the *when*). We watched it undo the mixing on a single
        shared pixel, checked that the footprints are cell-sized and that only a minority of
        sources are active, and then reordered those sources by firing time to expose a temporal
        **sequence** — a wave of activity around arena entry that survives a held-out
        cross-validation. This is the same decomposition we ran on behavior in notebook 4, now
        applied to the brain: decompose a mixture into its sources, once on the body and once on
        the tissue.

        **The next question.** We now hold 202 clean neurons, each a time series of *when* it
        fires. But we have not asked what any of them is *about*. A neuron is only interesting if
        its activity is tied to something in the world or the animal's behavior. So the next
        notebook asks: **what does each of these neurons encode?** We will hand a demixed
        population a behavioral variable — the animal's position in space — and ask which cells
        are **tuned** to it, building the neural analog of the behavioral detectors from Week 1.
        """
    )
    return


if __name__ == "__main__":
    app.run()
