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
    for _mod in ("course_utils.py", "neural_utils.py"):   # neural_utils imports course_utils
        _dst = os.path.join(ROOT, "course", _mod)
        if not os.path.exists(_dst):
            os.makedirs(os.path.dirname(_dst), exist_ok=True)
            urllib.request.urlretrieve(_RAW + "/course/" + _mod, _dst)
    sys.path.insert(0, os.path.join(ROOT, "course"))
    import neural_utils as nu
    CACHE = nu.cache_dir(ROOT)
    return CACHE, ROOT, go, np, nu


@app.cell
def _(ROOT, np):
    # Data bundle for this notebook: the striatal CNMF traces, the rat place/grid sessions, and the
    # social-isolation sessions. The neural blocks are stored as int8 time-deltas of the z-scored
    # traces (calcium is smooth, so the delta is lossless w.r.t. the int8 quantization); recon_block
    # cumsum-decodes + dequantizes back to z-units.
    import os as _os, urllib.request as _urlreq
    _p = _os.path.join(ROOT, "data", "nb08_assets.npz")
    if not _os.path.exists(_p):                       # bare checkout (e.g. molab): pull the committed file
        _RAW = _os.environ.get("COURSE_REPO_RAW",
            "https://raw.githubusercontent.com/talmolab/sleap-social-behavior-lab/main")
        _os.makedirs(_os.path.dirname(_p), exist_ok=True)
        _urlreq.urlretrieve(_RAW + "/data/nb08_assets.npz", _p)
    ASSETS = dict(np.load(_p, allow_pickle=False))

    def recon_block(delta, lo, hi, axis):
        _q = np.cumsum(delta.astype(np.int16), axis=axis)          # int8 delta -> quantized levels
        return ((_q.astype(np.float32) + 128.0) / 255.0) * (hi - lo) + lo   # dequantize to z-units
    return ASSETS, recon_block


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # NB08 · Reading the population

        ## Where we are

        In notebook 4 we took a wide, redundant table of behavior — 2499 interactions, each a
        row of 19 numbers — and collapsed it onto a **manifold**: a few underlying axes that
        carried most of the variation, found with PCA, then a nonlinear map found with UMAP. The
        question that notebook answered was *how many dimensions does behavior really live in, and
        can an unsupervised map rediscover aggression?* In notebook 6 we then trained a
        **decoder** — a supervised model that reads a behavior off the feature vector — and
        stress-tested it. Two moves: **factorize** a matrix into a small number of components, and
        **decode** a label from a population of measurements.

        This notebook applies those exact two moves to the brain. We hold, for each recording
        session, a population of neurons — one activity trace per cell, aligned frame by frame with
        the animal's behavior — and we ask:

        > **What does a population of neurons represent, and can we read a behavior off it?**

        We answer it in four steps: (1) look at the population as a matrix and name the
        factorization that produced it; (2) ask what a *single* neuron encodes — first location in
        space, then social contact; (3) ask whether the population fires in an ordered **sequence**;
        and (4) train a **population decoder** of social state and — the culminating result of the
        whole course — cross-validate it *honestly*.

        ## The same math, a different object

        The tools are the ones we already built. But the object they run on is not the same, and the
        difference is the intellectual core of this notebook.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <div style="background:#eef4fb;border-left:6px solid #1f77b4;padding:14px 18px;border-radius:6px">
        <b>Behavior vs. population — where the analogy holds, and where it breaks</b>

        <b>The same:</b> both behavior and neural activity are <b>time series</b>, and we read them
        with the same three tools — a <b>matrix factorization</b> (PCA on behavior features ↔ CNMF on
        the calcium movie: both write a data matrix as a <i>spatial factor times a temporal factor</i>),
        a <b>logistic decoder</b> with a threshold, and a <b>permutation / shuffle null</b> that decides
        whether a number is real.

        <b>Different:</b> behavior gave us a <b>fixed 2499×19 matrix</b> — the <i>same</i> 19 feature
        columns for every animal, so we could pool events across all 14 cages and test
        leave-one-cohort-out. Neural data does not cooperate. Each session extracts its <b>own</b> set
        of neurons — here <b>12 to 396 cells</b>, with <b>no identity correspondence</b> across
        sessions (cell 40 in one recording is not cell 40 in another). There is no shared 19-column
        axis. So the population must be <b>demixed per session</b>, and the decoder is trained and
        tested <b>within a single session</b>. Pooling across animals, which was free for behavior, is
        a genuine research problem here — it needs cell registration or a shared latent space. We name
        that limit again where it bites.
        </div>
        """
    )
    return


# ==================================================================== PART 1 — the population matrix
@app.cell
def _(ASSETS, np, recon_block):
    # The striatal CNMF-E session from notebook 7: 202 demixed neurons, ~16,800 frames at 30 fps.
    # C_z is the per-neuron z-scored trace matrix (n_neurons, n_frames), reconstructed from its int8
    # time-delta; Cn is the correlation image (H, W). The z-scored traces are the only CNMF product
    # any figure in this notebook consumes.
    C_z = recon_block(ASSETS["cz_delta"], float(ASSETS["cz_lo"]), float(ASSETS["cz_hi"]), axis=1)
    Cn = ASSETS["Cn"].astype(np.float32)            # correlation image (H, W)
    Fs = float(ASSETS["Fs"])
    n_neurons = int(ASSETS["n_neurons"])
    n_frames = int(ASSETS["n_frames"])
    # arena-entry frame (behavior 25 fps -> imaging 30 fps) anchors the sequence window in Part 3.
    ENTRY = int(7488 * (30 / 25))                   # -> 8985
    WIN_LEN = 3 * 60 * 30                           # 3 min at 30 fps -> 5400 frames
    return C_z, Cn, ENTRY, Fs, WIN_LEN, n_frames, n_neurons


@app.cell(hide_code=True)
def _(Fs, mo, n_frames, n_neurons):
    mo.md(
        f"""
        ---
        # Part 1 · The population as a matrix

        **Why.** Before asking what any neuron *means*, we look at the whole population at once and
        name where it came from. In notebook 7 we ran **CNMF** (constrained non-negative matrix
        factorization) on the raw calcium movie of a striatal session and pulled out
        **{n_neurons} neurons**: a spatial **footprint** for each cell (the *where*) and a calcium
        **trace** for each cell (the *when*). Those traces are the input here.

        **Definitions.**

        - **Calcium trace:** one neuron's brightness over time. It rises sharply when the cell fires
          (calcium floods in) and decays slowly afterward.
        - **Population raster:** a heatmap with **one row per neuron** and **one column per time
          frame**; color is that neuron's activity (here **z-scored** — each cell's trace has its mean
          subtracted and is divided by its standard deviation, so every cell is on the same scale, in
          units of standard deviations from its own mean). Bright vertical smears are moments when many
          cells fire together; horizontal streaks are individual cells active for a while.

        The raster below is the whole population — **{n_neurons} neurons × {n_frames:,} frames** at
        **{Fs:.0f} fps** ({n_frames / Fs / 60:.1f} min). The **contrast ceiling** slider sets the
        color top (`zmax`). This is a *display* control, but it teaches a real point about robust
        scaling: turn it too low and the image saturates so every row looks maximally active; turn it
        too high and only a couple of giant transients survive while the rest blanks. The informative
        band sits around z ≈ 4–6, which is why the slider starts at 5 and its floor is 3 — a calcium
        raster's noise floor is near z = 1, so a ceiling below 3 would show only noise.
        """
    )
    return


@app.cell
def _(mo):
    raster_zmax = mo.ui.slider(3.0, 9.0, value=5.0, step=0.5,
                               label="contrast ceiling (raster zmax, z-units)",
                               debounce=True, full_width=True)
    return (raster_zmax,)


@app.cell
def _(C_z, mo, n_frames, np, nu, raster_zmax):
    # Downsample the TIME axis for a light display (a full 202 x 16,800 heatmap exceeds marimo's
    # per-cell output cap); the analysis everywhere else stays full-resolution.
    _step = max(1, n_frames // 1500)
    _disp = C_z[:, ::_step]
    _x = np.arange(0, n_frames, _step)
    _fig = nu.raster_fig(_disp, x=_x,
                         title="Population raster · 202 z-scored striatal neurons (CNMF order)",
                         xlabel="time (frames, 30 fps)", ylabel="neuron",
                         zmin=0.0, zmax=float(raster_zmax.value), colorbar_title="z", height=460)
    mo.vstack([raster_zmax, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this raw CNMF row order the raster reads like scattered speckle. Hold that thought — in
        Part 3 we reorder the rows and the speckle turns into structure.

        **The factorization, named.** This population did not arrive as a raster; CNMF *produced* it by
        writing the movie `Y` (frames × pixels) as a product of two factors,

        $$ Y \;\approx\; A \cdot C, \qquad Y[t, p] \;\approx\; \sum_{k} A[k, p]\, C[t, k], $$

        where `A` holds the spatial **footprints** (one localized blob per neuron) and `C` holds the
        temporal **traces**. That is the *same shape* of statement PCA made about behavior in notebook
        4: a data matrix written as a **spatial factor times a temporal factor**. PCA picked orthogonal
        directions of maximum variance; CNMF picks **non-negative, spatially compact** factors, so each
        component is forced to look like a physical cell instead of an abstract statistical axis. Same
        family of method, a constraint swapped in. The correlation image below — each pixel colored by
        how much it co-fluctuates with its neighbors — is where CNMF found the cells to begin with;
        the bright blobs are the footprints it separated.
        """
    )
    return


@app.cell
def _(Cn, nu):
    nu.image_fig(Cn, title="Correlation image Cn — where the cells are (CNMF's starting point)",
                 colorscale="Viridis", colorbar_title="local corr", height=460)
    return


# ==================================================================== PART 2 — single-neuron tuning
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # Part 2 · What does a single neuron encode?

        A raster shows activity, but not *meaning*. A neuron is only interesting if its firing is tied
        to something — a variable in the world or in the animal's behavior. The word for that tie is
        **tuning**: a neuron is *tuned* to a variable when its firing rate is a reliable function of
        that variable. This is the same idea as a behavioral feature that separates aggression — a
        quantity whose value tracks a condition — now read off a single cell.

        We ask it twice, on two datasets, because the logic is identical and the second is the one this
        course cares about. First **space**: does a neuron fire only when the animal is in a particular
        place? (This is the founding question of systems neuroscience, and the cleanest illustration of
        a tuning curve.) Then **social contact**: does a neuron fire only when the animal is engaging a
        partner?

        ## 2a · Tuning to space — place cells

        **Why space first.** Location is a clean, low-dimensional physical variable a single cell can
        tile, so the tuning is easy to see and easy to test — the ideal warm-up for the messier social
        question. A **place cell** (O'Keefe & Dostrovsky, 1971) fires only when the animal occupies one
        particular region of the arena, its **place field**.

        **The dataset.** Three recording sessions from a separate rat experiment, each pairing tracked
        position with per-neuron spike counts. "Position" here is the centroid of two tracked **eye**
        positions — a gaze proxy, not a clean body-on-arena readout — so the fields are noisier than a
        textbook place cell. That imperfection is the point: it is why the final test grades each cell
        against a control rather than against the prettiest map.
        """
    )
    return


@app.cell
def _(ASSETS, np):
    # Rat place/grid sessions. Each holds centroid (T, 2) and spike counts (T, n_neurons) — the two
    # arrays every rate-map / SI / gridness plot consumes.
    rat_names = [str(x) for x in ASSETS["rat_names"]]
    rat_sessions = {name: {"centroid": ASSETS[f"rat_centroid_{i}"].astype(np.float32),
                           "spikes": ASSETS[f"rat_spikes_{i}"]}
                    for i, name in enumerate(rat_names)}
    return rat_names, rat_sessions


@app.cell(hide_code=True)
def _(mo, rat_names, rat_sessions):
    _rows = "\n".join(
        f"| `{n}` | {rat_sessions[n]['spikes'].shape[0]:,} | {rat_sessions[n]['spikes'].shape[1]} |"
        for n in rat_names)
    mo.md(
        f"""
        Pick a session below; it drives every place-cell plot. The first session has the most neurons
        (14) and the clearest fields, so it is the default.

        | session | frames T | neurons |
        |---|---|---|
        {_rows}
        """
    )
    return


@app.cell
def _(mo, rat_names):
    rat_session_pick = mo.ui.dropdown(options=rat_names, value=rat_names[0],
                                      label="rat session (drives the place-cell plots)")
    return (rat_session_pick,)


@app.cell
def _(mo, rat_session_pick, rat_sessions):
    # Neuron slider whose max tracks the selected session's neuron count, so it never clamps to the
    # last neuron on a 5- or 6-neuron session. The default lands on a known place-like cell in the
    # 14-neuron session.
    _nmax = rat_sessions[rat_session_pick.value]["spikes"].shape[1] - 1
    rat_neuron = mo.ui.slider(0, _nmax, value=min(5, _nmax), step=1,
                              label=f"neuron (0–{_nmax} in this session)",
                              debounce=True, full_width=True)
    return (rat_neuron,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The rate map — a spatial tuning curve, built from two ingredients

        **Why.** We want *firing rate as a function of location*, not distorted by where the animal
        happened to spend its time. The correct object is the **rate map**, and it is built from two
        maps we look at first, side by side.

        **Definitions.**

        - **Occupancy:** for each small square (bin) of the arena, the number of frames the animal
          spent there. Coverage, not firing. This is the biggest confound in place-cell work: a neuron
          can *look* tuned to a spot simply because the animal lingered there.
        - **Spike map:** for each bin, the total spikes fired while the animal was in it.
        - **Rate map:** the ratio, bin by bin, $\text{rate} = \text{spikes} / \text{occupancy}$.
          Dividing by occupancy removes the "spent all its time here" confound.

        The three panels lay the ingredients out for the chosen neuron so the division is visible. A
        bright spot present in *both* occupancy and spike map but that **vanishes** in the rate map was
        an occupancy artifact — the animal just spent time there. `nu.rate_map(x, y, spikes, bins)`
        does the binning and division; it returns `occupancy`, `spike_map`, and `rate`.
        """
    )
    return


@app.cell
def _(mo, np, nu, rat_neuron, rat_session_pick, rat_sessions):
    _d = rat_sessions[rat_session_pick.value]
    _ctr, _spk = _d["centroid"], _d["spikes"]
    _ni = min(rat_neuron.value, _spk.shape[1] - 1)
    _rm = nu.rate_map(_ctr[:, 0], _ctr[:, 1], _spk[:, _ni], bins=20)
    _xc = 0.5 * (_rm["xedges"][:-1] + _rm["xedges"][1:])
    _yc = 0.5 * (_rm["yedges"][:-1] + _rm["yedges"][1:])
    _f_occ = nu.heatmap_fig(_rm["occupancy"].T, x=_xc, y=_yc, title="occupancy (frames)",
                            xlabel="x", ylabel="y", colorscale="Cividis", colorbar_title="", height=330)
    _f_spk = nu.heatmap_fig(_rm["spike_map"].T, x=_xc, y=_yc, title="spike map (counts)",
                            xlabel="x", ylabel="y", colorscale="Magma", colorbar_title="", height=330)
    _f_rate = nu.heatmap_fig(_rm["rate"].T, x=_xc, y=_yc, title="rate = spikes / occupancy",
                             xlabel="x", ylabel="y", colorscale="Inferno", colorbar_title="", height=330)
    for _f in (_f_occ, _f_spk, _f_rate):
        _f.update_yaxes(scaleanchor="x", scaleratio=1)
    mo.vstack([mo.hstack([rat_session_pick, rat_neuron]),
               mo.hstack([_f_occ, _f_spk, _f_rate], widths=[1, 1, 1])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### One number per neuron, and a control that can say no

        **Why a single number.** A rate map is a picture; to *rank* neurons we collapse each to one
        score. **Skaggs spatial information (SI)**, in bits/spike, measures how much knowing the
        animal's location tells you about whether the neuron fires. A cell that fires the same rate
        everywhere scores ~0; a cell with a sharp field scores high. `nu.spatial_information(rate,
        occupancy)` computes it.

        **But a big number is not yet a result.** A neuron that fires only a few dozen times can post a
        very high SI purely because a handful of spikes happen to land in a few rarely-visited bins —
        that is sparsity, not tuning. We separate a real field from a sparsity artifact by comparing
        each neuron's SI against a control built from **its own spikes**: circularly shift the spike
        train by a random amount (`np.roll`), which keeps the exact spike count but pairs the spikes
        with the *wrong* positions. Any SI the shuffle produces is by chance. A real cell beats its own
        shuffle; an artifact does not.

        The **bins** slider below sets the rate-map resolution — and it teaches a subtle point, so
        watch the SI in the title as you drag it. Finer bins give a *larger* SI (there are more empty
        bins for lucky spikes to light up), so SI is **not** a fixed property of a neuron; it depends
        on a choice you made. That is exactly why the shuffle null is computed **at the same bin
        count** — the control absorbs the bin-count inflation, so what survives is real tuning, not a
        resolution artifact.
        """
    )
    return


@app.cell
def _(mo):
    rm_bins = mo.ui.slider(8, 40, value=20, step=2, label="spatial bins per axis (watch the SI)",
                           debounce=True, full_width=True)
    return (rm_bins,)


@app.cell
def _(mo, nu, rat_neuron, rat_session_pick, rat_sessions, rm_bins):
    _d = rat_sessions[rat_session_pick.value]
    _ctr, _spk = _d["centroid"], _d["spikes"]
    _ni = min(rat_neuron.value, _spk.shape[1] - 1)
    _rm = nu.rate_map(_ctr[:, 0], _ctr[:, 1], _spk[:, _ni], bins=int(rm_bins.value))
    _si = nu.spatial_information(_rm["rate"], _rm["occupancy"])
    _xc = 0.5 * (_rm["xedges"][:-1] + _rm["xedges"][1:])
    _yc = 0.5 * (_rm["yedges"][:-1] + _rm["yedges"][1:])
    _fig = nu.heatmap_fig(
        _rm["rate"].T, x=_xc, y=_yc,
        title=f"Rate map · neuron {_ni} · {int(rm_bins.value)} bins · SI = {_si:.3f} bits/spike",
        xlabel="x (px)", ylabel="y (px)", colorscale="Inferno", colorbar_title="rate", height=500)
    _fig.update_yaxes(scaleanchor="x", scaleratio=1)
    mo.vstack([rm_bins, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Which neurons are trustworthy? — SI vs spike count

        The scatter below plots every neuron by its **spike count** (x, log scale) and its **SI** (y),
        and colors each point by whether it **beats its own spike-matched shuffle** (green = passes,
        gray = fails). A trustworthy place cell sits to the
        **right** (many spikes) and **high** (large SI) and is green. A point that is high but far
        **left** (large SI, very few spikes) is the danger zone: a big number a few lucky spikes
        manufactured — and it comes out gray. Hover any point.
        """
    )
    return


@app.cell
def _(go, np, nu, rat_session_pick, rat_sessions):
    # SI vs spike count, colored by shuffle pass/fail (green = beats its own shuffle, gray = fails).
    # Each neuron gets a spike-matched circular-shift null; "passes" = observed SI above its own 95th
    # percentile.
    _d = rat_sessions[rat_session_pick.value]
    _ctr, _spk = _d["centroid"], _d["spikes"]
    def _si_of(_col):
        _rm = nu.rate_map(_ctr[:, 0], _ctr[:, 1], _col, bins=20)
        return nu.spatial_information(_rm["rate"], _rm["occupancy"])
    _rng = np.random.default_rng(0)
    _sis, _nsp, _pass = [], [], []
    for _i in range(_spk.shape[1]):
        _obs = _si_of(_spk[:, _i])
        _null = np.array([_si_of(np.roll(_spk[:, _i], int(_rng.integers(1000, len(_spk) - 1000))))
                          for _ in range(50)])
        _sis.append(_obs); _nsp.append(int((_spk[:, _i] > 0).sum()))
        _pass.append(_obs > np.percentile(_null, 95))
    _sis = np.array(_sis); _nsp = np.array(_nsp); _pass = np.array(_pass)
    _fig = go.Figure()
    for _m, _name, _col in [(_pass, "beats shuffle", "#2ca02c"), (~_pass, "fails shuffle", "#bab0ac")]:
        _fig.add_scatter(x=_nsp[_m], y=_sis[_m], mode="markers+text",
                         text=[f"n{_i}" for _i in np.where(_m)[0]], textposition="top center",
                         textfont=dict(size=10), name=_name,
                         marker=dict(size=14, color=_col, line=dict(width=0.5, color="#333")),
                         hovertext=[f"neuron {_i}: SI={_sis[_i]:.3f}, {_nsp[_i]:,} spike-frames"
                                    for _i in np.where(_m)[0]],
                         hovertemplate="%{hovertext}<extra></extra>")
    _fig.update_xaxes(title="spike-frames (more = more reliable)", type="log")
    _fig.update_yaxes(title="Skaggs spatial information (bits/spike)")
    nu.apply_house_style(_fig, title="SI vs spike count · green beats its own shuffle · top-LEFT is suspect",
                         legend="below", height=470)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In the 14-neuron session the scatter tells the cautionary tale at a glance. **Neuron 10** floats
        near the top with the largest *raw* SI (~2.14 bits/spike) — but it sits at the far left, having
        fired in only ~80 frames, and it comes out **gray**: its high number is exactly what
        position-scrambled spikes produce. **Neuron 5** is high *and* well to the right (~1.26
        bits/spike from ~1,100 spike-frames) and comes out **green** — a genuine field. The plot below
        makes that comparison concrete. Each of the two cells is scored against its own circular-shift
        null and reported as a **z-score** — the number of standard deviations the observed SI sits
        above the mean of that null. Reporting a z-score rather than the raw SI puts a highly active
        cell and a quiet one on one common axis, so the comparison no longer runs on the raw bits/spike
        axis where neuron 10's larger raw value would falsely look like the strongest place cell.
        """
    )
    return


@app.cell
def _(np, nu, rat_sessions):
    # Score two canonical cells — neuron 5 (a real place cell) and neuron 10 (a sparsity artifact) —
    # each against its OWN circular-shift null: cyclically slide the spike train by a random offset,
    # which keeps the spike count but pairs the spikes with the wrong positions, and re-score; 50 such
    # shifts form that cell's chance distribution. Report each observed SI as a z-score — how many
    # standard deviations it sits above the mean of its own null. Reporting each cell relative to its
    # own null puts a loud cell and a quiet cell on one common (normalized) axis, so the comparison no
    # longer runs on the raw bits/spike axis where neuron 10's larger raw SI (~2.14) would rank first.
    _d = rat_sessions["20160609T194655.mat"]
    _ctr, _spk = _d["centroid"], _d["spikes"]
    def _si_of(_col):
        _rm = nu.rate_map(_ctr[:, 0], _ctr[:, 1], _col, bins=20)
        return nu.spatial_information(_rm["rate"], _rm["occupancy"])
    _ids = [5, 10]
    _obs = np.array([_si_of(_spk[:, _i]) for _i in _ids])
    _rng = np.random.default_rng(0)
    _null = np.stack([[_si_of(np.roll(_spk[:, _i], int(_rng.integers(1000, len(_spk) - 1000))))
                       for _ in range(50)] for _i in _ids])   # (2, 50)
    _res = nu.per_neuron_normalized_shuffle(_obs, _null)
    _labels = np.array([f"neuron {_i}" for _i in _ids])
    _fig = nu.strip_points_fig(
        _res["z"], _labels, group_order=list(_labels),
        colors={_labels[0]: "#2ca02c", _labels[1]: "#bab0ac"}, show_mean=False,
        ylabel="SI, z vs the cell's OWN shuffle null",
        title=(f"neuron 5 (real): z = {_res['z'][0]:.1f}, p = {nu.fmt_p(_res['p'][0])}   ·   "
               f"neuron 10 (artifact): z = {_res['z'][1]:.1f}, p = {nu.fmt_p(_res['p'][1])}"),
        height=420)
    # Two categories: pin the x-axis so they sit as two adjacent columns instead of being auto-ranged
    # to opposite edges of the wide plot. Drop the legend — the x tick labels already name the cells.
    _fig.update_xaxes(range=[-0.6, 1.6])
    _fig.update_layout(showlegend=False)
    _fig.add_hline(y=1.64, line=dict(color="#e45756", width=2, dash="dash"),
                   annotation_text="95th-pct of own null (z≈1.64)", annotation_position="top left")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Neuron 5 sits ~15 standard deviations above its own null (it clears the dashed 95th-percentile
        line easily); neuron 10, despite a *larger* raw SI, lands **below** its own null — a negative
        z. On the raw SI axis neuron 10 would have looked like the best place cell in the session. The
        control reverses the verdict. This is the discipline the whole neural arm reuses: **a large
        number is not a result until it beats a matched control.**

        ### A cautionary tool — the autocorrelogram and the grid-cell trap

        **Why show it.** A **grid cell** (Hafting et al., 2005) fires not at one place but at *many*,
        arranged in a repeating triangular lattice. The standard tool for spotting one is the **spatial
        autocorrelogram**: slide the rate map over a copy of itself and, at each offset, measure how
        well it matches. A single place field gives one central peak; a true grid gives a central peak
        ringed by **six** satellites (the hexagonal fingerprint).

        This section is a deliberate warning about reading a picture too eagerly. The autocorrelogram
        below often shows a **ring of satellite peaks even for a plain place cell** — an artifact of the
        arena border and the single-field shape, not a grid. The figure carries a **neutral** title (it
        does not assert grid or not-grid), and it reports a quantitative **gridness score** —
        `nu.gridness_score`, the standard 60°-symmetry statistic: it correlates the ring with rotated
        copies of itself and returns
        $\min(\text{corr}@60°,@120°) - \max(@30°,@90°,@150°)$, which is $>0$ for a hexagonal lattice.
        """
    )
    return


@app.cell
def _(mo, np, nu, rat_neuron, rat_session_pick, rat_sessions):
    from scipy.signal import correlate2d as _corr2d
    _d = rat_sessions[rat_session_pick.value]
    _ctr, _spk = _d["centroid"], _d["spikes"]
    _ni = min(rat_neuron.value, _spk.shape[1] - 1)
    _rm = nu.rate_map(_ctr[:, 0], _ctr[:, 1], _spk[:, _ni], bins=20, smooth_sigma=1.0)
    _r = _rm["rate"]; _r0 = _r - _r.mean()
    _ac = _corr2d(_r0, _r0, mode="full")
    _pk = np.abs(_ac).max(); _ac = _ac / (_pk if _pk > 0 else 1.0)
    _g = nu.gridness_score(_ac)
    _lag = np.arange(-_r.shape[0] + 1, _r.shape[0])
    _fig = nu.heatmap_fig(
        _ac.T, x=_lag, y=_lag,
        title=f"Spatial autocorrelogram · neuron {_ni} · gridness = {_g:+.2f}  (>0 ⇒ hexagonal)",
        xlabel="x lag (bins)", ylabel="y lag (bins)", colorscale="RdBu_r",
        zmid=0.0, colorbar_title="corr", height=480)
    _fig.update_yaxes(scaleanchor="x", scaleratio=1)
    mo.vstack([mo.hstack([rat_session_pick, rat_neuron]), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here is the honest, uncomfortable finding. Neuron 5 — a cell we just confirmed is a genuine
        *place* cell — posts a **positive gridness of about +0.64**, and that value even clears its own
        circular-shift shuffle (empirical p ≈ 0.01). Read naively, the score says "grid." It is not one.
        These are hippocampal, gaze-proxy recordings, not the entorhinal cortex where grid cells live;
        the ring is a border-and-single-field artifact, and the score dutifully quantifies the artifact.

        The lesson is the one that runs through the statistics notebook: a metric passing its own
        shuffle is **necessary but not sufficient**. A real grid call needs converging evidence — the
        right brain region, a stable field spacing across the session, enough spikes — not one
        autocorrelogram and one score. When the whole preparation is wrong for the question, no amount
        of internal significance rescues it. We keep the tool, and we keep our skepticism.

        ### Exercise 1 — confirm the artifact cell FAILS its shuffle

        **Python skill practiced:** *writing and calling a small function*, then *reducing with numpy*
        (`np.percentile`). This repeats the score-then-control pattern on the cell that is supposed to
        fail — the negative result is the whole point.

        You are given `_si_of`, which maps one spike column to its SI, and a 50-sample shuffle-null
        loop. Fill the two blanks: (1) score neuron **10**'s real spikes, and (2) read the 95th
        percentile of its null as the chance band. **What you should see:** the observed SI (red line)
        lands *inside* — even below — the gray shuffle cloud, so `beats_band` comes out **False**. That
        is correct: neuron 10's high raw SI is a sparsity artifact, and its own spikes, scrambled,
        routinely produce SI at least that large.
        """
    )
    return


@app.cell
def _(np, nu, rat_sessions):
    # ------------------------------------------------------------------ YOUR CODE (edit this cell)
    _d = rat_sessions["20160609T194655.mat"]
    _ctr, _spk = _d["centroid"], _d["spikes"]
    _col = _spk[:, 10]                                  # the SUSPECT: neuron 10, ~80 spike-frames

    def _si_of(_s):
        # PURPOSE: spatial information of one spike column. INPUT: a (T,) spike-count array.
        # OUTPUT: its Skaggs SI in bits/spike (one float).
        _rm = nu.rate_map(_ctr[:, 0], _ctr[:, 1], _s, bins=20)
        return nu.spatial_information(_rm["rate"], _rm["occupancy"])

    # TODO 1 (blank #1): the observed SI of neuron 10's REAL (unshuffled) spikes.
    #   WHAT TO CHANGE: replace ____ with `_col` — the real spike column defined three lines up.
    #   WHY: this is the number under test. Neuron 10's raw SI is LARGE (~2.14) — the exercise shows
    #   that a large raw number can still be an artifact once you compare it to the right control.
    si10_obs = float(_si_of(_col))                     # <-- replace ____ with _col

    _rng = np.random.default_rng(0)
    _null = np.array([
        # circularly shift the SAME spikes by a random amount (breaking the spike-position link) and
        # re-score; 50 such shuffles form the chance distribution for THIS neuron.
        _si_of(np.roll(_col, int(_rng.integers(1000, len(_col) - 1000))))
        for _ in range(50)
    ])

    # TODO 2 (blank #2): the chance band = the 95th percentile of the 50 shuffled SI values.
    #   WHAT TO CHANGE: replace ____ with `95`.
    #   WHY: 95 is the one-sided significance threshold — a real cell must beat the value only 5% of
    #   position-scrambled shuffles exceed. For neuron 10 the band lands ABOVE the observed SI.
    si10_band = float(np.percentile(_null, 95))        # <-- replace ____ with 95
    beats_band = si10_obs > si10_band                  # expected: False (artifact fails its own control)
    # ---------------------------------------------------------------------------------------------
    return beats_band, si10_band, si10_obs


@app.cell
def _(go, np, nu, rat_sessions, si10_band, si10_obs):
    # Result plot: neuron 10's shuffle null (gray) with the chance band (dashed) and YOUR observed SI
    # (red). The null is recomputed here with the canonical settings so the picture is stable.
    _d = rat_sessions["20160609T194655.mat"]
    _ctr = _d["centroid"]; _col = _d["spikes"][:, 10]
    def _si_of(_s):
        _rm = nu.rate_map(_ctr[:, 0], _ctr[:, 1], _s, bins=20)
        return nu.spatial_information(_rm["rate"], _rm["occupancy"])
    _rng = np.random.default_rng(0)
    _null = np.array([_si_of(np.roll(_col, int(_rng.integers(1000, len(_col) - 1000))))
                      for _ in range(50)])
    _fig = go.Figure()
    _fig.add_histogram(x=_null, nbinsx=18, marker_color="#bab0ac", name="shuffled SI")
    _fig.add_vline(x=float(si10_band), line=dict(color="#4c78a8", width=2, dash="dash"),
                   annotation_text="95% band", annotation_position="top")
    _fig.add_vline(x=float(si10_obs), line=dict(color="#e45756", width=3),
                   annotation_text="observed", annotation_position="top right")
    nu.apply_house_style(_fig, title="Neuron 10 · observed SI sits INSIDE its own shuffle cloud",
                         legend="below", height=360)
    _fig.update_xaxes(title="spatial information (bits/spike)")
    _fig.update_yaxes(title="shuffles")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show solution": mo.md(
            r"""
            ```python
            col = spk[:, 10]
            si10_obs  = si_of(col)                 # ~2.14 bits/spike  (a large RAW number)
            null      = [si_of(np.roll(col, k)) for random k]   # 50 shuffles
            si10_band = np.percentile(null, 95)    # ~2.4  (the band sits ABOVE the observed SI)
            beats_band = si10_obs > si10_band      # False
            ```

            **What you should find.** Neuron 10's raw SI is large, but its spike-matched shuffle band is
            *larger* — so `beats_band` is **False**. High SI from ~80 spikes is a sparsity artifact, not
            a place field. Contrast neuron 5, whose observed SI (~1.26) sits far above its band (~0.66)
            and passes. The reliable conclusion is that place-like cells exist in this session (neurons
            4, 5, 6…), but SI alone, without a spike-matched control, manufactures false positives.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(beats_band, mo, si10_band, si10_obs):
    # Self-check: neuron 10 (20 bins, seed 0). Pinned: si10_obs ~ 2.14, band ~ 2.4, beats_band False.
    _p1 = abs(float(si10_obs) - 2.14) < 0.06
    _p2 = (bool(beats_band) is False) and (float(si10_band) > float(si10_obs))
    _ok = _p1 and _p2
    _c = "#e8f5e9" if _ok else "#ffebee"
    _b = "#2e7d32" if _ok else "#c62828"
    _m1 = (f"observed SI = {si10_obs:.3f} bits/spike (neuron 10)" if _p1 else
           f"si10_obs = {si10_obs:.3f} — expected ~2.14 for neuron 10 at 20 bins; pass the REAL column")
    _m2 = (f"the artifact FAILS its own control: band = {si10_band:.3f} sits above the observed SI, so "
           "beats_band is False — exactly right" if _p2 else
           f"band = {si10_band:.3f}, beats_band = {beats_band} — did you take the 95th percentile of 50 "
           "circular shuffles of the SAME column?")
    _head = "Pass — you caught the sparsity artifact" if _ok else "Not yet — fix the flagged part"
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_b};padding:12px 16px;border-radius:6px">
        <b style="color:{_b}">{_head}</b><br>
        {_m1}<br>{_m2}<br>
        <span style="font-size:0.9em;color:#555">Graded on the shuffle-corrected verdict, not the raw SI
        leaderboard. Tolerance: |si10_obs − 2.14| &lt; 0.06, band &gt; observed, beats_band False.</span>
        </div>
        """
    )
    return


# ---------------------------------------------------------------- 2b social tuning
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2b · Tuning to social contact

        **Why.** Space was the warm-up. The variable this course cares about is not a corner of the
        arena but *another animal*. Does a neuron fire more when the focal mouse is engaging a partner?
        Same question, same tuning logic — a firing rate that depends on a labeled state — now with a
        social label instead of a position.

        **The dataset.** A social-isolation experiment: a focal mouse (group-housed **control**, or
        isolated **24 h** / **7 d**) is reintroduced to a partner while its striatal calcium is imaged.
        Eighteen sessions, six per condition, each with frame-by-frame social scoring. We decode
        `is_social_sender` — the focal mouse *actively engaging* the partner — the behavior its own
        neurons are most likely to drive.

        **One alignment step.** Calcium is sampled at **30 fps**, behavior scored at **25 fps**, so the
        two arrays differ in length. Before relating a neuron to a label we resample the calcium onto
        the behavior clock (`nu.interp_resample`, linear interpolation — it re-grids existing samples,
        it does not invent peaks), z-score each neuron, and crop to the first 3 minutes after the
        partner enters. Pick a session below (session 6 — a 7-day, 218-neuron recording — is the
        default); it drives the social-tuning plots.
        """
    )
    return


@app.cell
def _(ASSETS, recon_block):
    # Social-isolation dataset. Each session's neural block is already resample->zscore->crop'd in that
    # exact order (every pinned number depends on it) and stored as an int8 time-delta; sess_neurons_all[s]
    # is (T_crop, n_cells) z-scored, sess_social_all[s] is the (T_crop,) is_social_sender label.
    # cond_labels drive the dropdown; both lists drive every SI figure and the run_all-gated 18-session
    # blocked-CV compute.
    cond_labels = [str(x) for x in ASSETS["si_cond_labels"]]
    n_sessions = int(ASSETS["si_n_sessions"])
    _lo, _hi = float(ASSETS["si_lo"]), float(ASSETS["si_hi"])
    sess_neurons_all = [recon_block(ASSETS[f"si_delta_{s}"], _lo, _hi, axis=0)
                        for s in range(n_sessions)]
    sess_social_all = [ASSETS[f"si_social_{s}"].astype(bool) for s in range(n_sessions)]
    return cond_labels, n_sessions, sess_neurons_all, sess_social_all


@app.cell
def _(cond_labels, mo, n_sessions):
    _opts = {f"session {s} · {cond_labels[s]}": s for s in range(n_sessions)}
    si_session_pick = mo.ui.dropdown(options=_opts, value="session 6 · 7d",
                                     label="social session (drives the social-tuning plots)")
    return (si_session_pick,)


@app.cell
def _(sess_neurons_all, sess_social_all, si_session_pick):
    # Select the precomputed block for the chosen session. The resample -> zscore -> crop was applied
    # in the bundle in that exact order (every pinned number depends on it). sess_neurons is
    # (T, n_neurons) z-scored; sess_social is (T,) bool.
    _s = int(si_session_pick.value)
    sess_neurons = sess_neurons_all[_s]
    sess_social = sess_social_all[_s]
    sess_ncells = sess_neurons.shape[1]
    sess_frac = float(sess_social.mean())
    return sess_frac, sess_ncells, sess_neurons, sess_social


@app.cell
def _(mo, sess_ncells):
    # Neuron slider whose max tracks THIS session's neuron count (never clamps silently).
    si_neuron = mo.ui.slider(0, sess_ncells - 1, value=min(173, sess_ncells - 1), step=1,
                             label=f"neuron (0–{sess_ncells - 1} in this session)",
                             debounce=True, full_width=True)
    return (si_neuron,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **One neuron, two distributions.** For the chosen neuron we split its z-scored activity into
        **social** and **non-social** frames and draw the two **empirical cumulative distributions
        (ECDFs)**. An ECDF at value $v$ is the fraction of frames at or below $v$; if the social curve
        sits to the *right* of the non-social curve, the neuron is more active during social frames.
        The title reports the summary the next step thresholds on: the ratio of
        mean absolute activity in social vs non-social frames. Slide through the neurons and watch which
        ones separate — most barely do, which is the point that motivates the population decoder.
        """
    )
    return


@app.cell
def _(mo, np, nu, sess_neurons, sess_social, si_neuron, si_session_pick):
    _i = int(si_neuron.value)
    _x = sess_neurons[:, _i]
    _grp = np.where(sess_social, "social", "non-social")
    _soc, _non = _x[sess_social], _x[~sess_social]
    _ratio = float(np.abs(_soc).mean() / (np.abs(_non).mean() + 1e-12))
    _fig = nu.ecdf_fig(
        _x, _grp, group_order=["non-social", "social"],
        colors={"social": "#2ca02c", "non-social": "#7f7f7f"},
        xlabel="activity (z-score)", ylabel="cumulative fraction of frames",
        title=f"neuron {_i}:  |social|/|non-social| ratio = {_ratio:.2f}  (flagged 'social' if > 1.5)",
        height=420)
    mo.vstack([mo.hstack([si_session_pick, si_neuron]), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### How many cells are "social", and how the criterion decides

        **Why.** Rather than inspect cells one at a time, apply one test to every neuron and count.
        `nu.social_neuron_mask` flags a neuron whose mean absolute activity is at least `threshold`×
        higher on social frames. Each point below is one neuron's ratio, sorted; drag the threshold and
        watch the count of flagged (green) cells change. This control teaches how a criterion is chosen:
        the ratio has no natural cutoff, so 1.5 is a **convention**, and where you put it changes the
        count — a modeling choice you should report, not hide. The range stops at 2.5 because past ~2.3
        essentially no neuron in any session clears the bar (the informative regime is 1.0–2.0).
        """
    )
    return


@app.cell
def _(mo):
    ratio_thr = mo.ui.slider(1.0, 2.5, value=1.5, step=0.1,
                             label="social-neuron ratio threshold (convention, not learned)",
                             debounce=True, full_width=True)
    return (ratio_thr,)


@app.cell
def _(go, mo, np, nu, ratio_thr, sess_neurons, sess_social):
    _soc = np.abs(sess_neurons[sess_social]).mean(axis=0)
    _non = np.abs(sess_neurons[~sess_social]).mean(axis=0)
    _ratio = np.where(_non > 0, _soc / _non, 0.0)
    _order = np.argsort(_ratio)
    _rs = _ratio[_order]
    _thr = float(ratio_thr.value)
    _is_soc = _rs > _thr
    _fig = go.Figure()
    _fig.add_scatter(x=np.arange(len(_rs)), y=_rs, mode="markers",
                     marker=dict(size=6, color=np.where(_is_soc, "#2ca02c", "#c7c7c7"),
                                 line=dict(width=0.5, color="white")),
                     text=[f"neuron {int(o)}" for o in _order],
                     hovertemplate="%{text}<br>ratio=%{y:.2f}<extra></extra>", showlegend=False)
    _fig.add_hline(y=_thr, line=dict(color="#e45756", width=2, dash="dash"),
                   annotation_text=f"threshold {_thr:.1f}", annotation_position="top left")
    nu.apply_house_style(_fig, title=f"{int(_is_soc.sum())} social neurons of {len(_rs)} (ratio > {_thr:.1f})",
                         legend=None, height=380)
    _fig.update_xaxes(title="neuron (sorted by ratio)")
    _fig.update_yaxes(title="|social| / |non-social| ratio")
    mo.vstack([ratio_thr, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <div style="background:#fff4e5;border-left:6px solid #e8871e;padding:12px 16px;border-radius:6px">
        <b>Why not just use a percentile test?</b> The helper offers a third method, <code>percentile</code>
        (flag a cell if it is unusually active on more than 1% of social frames). It looks principled,
        but with its shipped threshold it is <b>degenerate</b>: it flags roughly the <i>entire</i>
        population — 167 of 218 neurons in this session, and 202/202, 266/289, 395/396 in others. A
        criterion that calls almost every cell "social" carries no information. This is a small, real
        example of a badly-calibrated test. The honest count depends
        on the criterion (<code>ratio</code>, <code>delta</code>, and a fixed percentile disagree), and
        the right response is to state which criterion you used and why — not to pick the one that gives
        the biggest number.
        </div>
        """
    )
    return


# ==================================================================== PART 3 — the sequence
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # Part 3 · Does the population fire in a sequence?

        **Why.** Single-neuron tuning asks about one cell at a time. But a population can carry
        structure that no single cell reveals — in particular a **sequence**: the neurons activating in
        a consistent order in time, like a wave rolling across the tissue. This is the population-level
        analog of behavior having *temporal grammar* (notebook 6): structure in the ordering, not just
        in the marginal counts. We return to the striatal CNMF session for this, because it has a clean
        event — the moment the arena partner enters — around which to look.

        **Definition.** When a sequence is present and we sort the neurons by *when* they first fire,
        the raster's activity collapses onto a **diagonal**: the earliest-firing cells at the top, the
        latest at the bottom. `nu.sequence_sort` returns that ordering (each neuron's first
        supra-threshold frame, argsorted). The **left** panel shows a window in raw CNMF order; the
        **right** shows the same
        window sorted. Each title reports a **sequenceness** score — the absolute Spearman correlation
        between a neuron's row position and its first-activation frame — 0 = no order, 1 = perfect
        diagonal.

        The three sliders each teach something specific. **Window start** cannot exceed
        `n_frames − window length`, so the window always fits — a window that ran off the end would
        silently shrink and read as "no sequence" for the wrong reason. **Window length** is trimmed to
        the regime where the score responds (past ~4000 frames it saturates). And **activation
        threshold** is the trap: watch what happens when you raise it, discussed below the figure.
        """
    )
    return


@app.cell
def _(mo):
    win_len = mo.ui.slider(600, 4000, value=3600, step=100,
                           label="window length (frames)", debounce=True, full_width=True)
    return (win_len,)


@app.cell
def _(ENTRY, mo, n_frames, win_len):
    # Cap the start so [start, start+len] always fits inside the recording (never a degenerate <600
    # window silently clipped at the end).
    _hi = max(0, n_frames - int(win_len.value))
    win_start = mo.ui.slider(0, _hi, value=min(ENTRY, _hi), step=30,
                             label="window start (frame)", debounce=True, full_width=True)
    seq_thresh = mo.ui.slider(3.0, 6.0, value=5.0, step=0.5,
                              label="activation threshold (z) — watch it fabricate order",
                              debounce=True, full_width=True)
    return seq_thresh, win_start


@app.cell
def _(C_z, mo, n_frames, np, nu, seq_thresh, win_len, win_start):
    from scipy.stats import spearmanr as _sp
    def _seqness(_raster, _thr):
        _first = np.argmax(_raster > _thr, axis=1)
        _r, _ = _sp(np.arange(_raster.shape[0]), _first)
        return 0.0 if np.isnan(_r) else abs(float(_r))
    _s = int(win_start.value)
    _e = min(_s + int(win_len.value), n_frames)
    _thr = float(seq_thresh.value)
    _win = C_z[:, _s:_e]
    _order = nu.sequence_sort(_win, thresh=_thr)
    _sorted = _win[_order]
    _q_un, _q_so = _seqness(_win, _thr), _seqness(_sorted, _thr)
    _n_active = int((_win.max(axis=1) > _thr).sum())
    _step = max(1, _win.shape[1] // 1200)
    _xd = np.arange(_s, _e, _step)
    _left = nu.raster_fig(_win[:, ::_step], x=_xd, title=f"unsorted · sequenceness = {_q_un:.2f}",
                          xlabel="time (frames)", ylabel="neuron (CNMF order)",
                          zmin=0.0, zmax=6.0, colorbar_title="z", height=460)
    _right = nu.raster_fig(_sorted[:, ::_step], x=_xd,
                           title=f"sorted by first firing · sequenceness = {_q_so:.2f} · {_n_active} active",
                           xlabel="time (frames)", ylabel="neuron (sequence order)",
                           zmin=0.0, zmax=6.0, colorbar_title="z", height=460)
    mo.vstack([mo.hstack([win_start, win_len, seq_thresh]),
               mo.hstack([_left, _right], widths=[1, 1])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        At the default window (~3600 frames around arena entry) the sorted raster shows a clean
        diagonal, sequenceness ≈ 0.73, up from ≈ 0.08 unsorted (the fixed 5400-frame window used by the
        test and exercise below reaches ≈ 0.79). But now the warning. **Raise the activation
        threshold toward 6.** Fewer and fewer neurons ever cross it — the "sequence" thins from ~128
        cells at z = 3 down to only ~19 at z = 6 — so the diagonal ends up resting on a handful of
        cells. The sorted score *does* fall as you climb (≈ 0.73 at z = 5, ≈ 0.49 at z = 6), yet it
        never collapses to the unsorted baseline: even at z = 6, sorting 19 cells still manufactures a
        ≈ 0.49 diagonal, six times the ≈ 0.08 you get without sorting. That leftover 0.49 is
        meaningless — you sorted the rows *by* their first-crossing time and then measured order in
        that very same quantity, so a diagonal is guaranteed no matter how few cells remain. Sorting
        rows by their first-crossing time will **always** produce a diagonal; that is what sorting
        does. A high sorted sequenceness, on its own, proves nothing. This is a
        **circular-analysis** trap (the statistics notebook builds it into a general lesson): you sorted
        by a quantity and then measured order in that same quantity. To make an honest claim we need a
        test that can come out **negative**.

        ### The honest test — split-half cross-validation

        **Method.** Cut the arena-entry window in half. Learn the neuron ordering on the **first** half
        only. Then, *without re-sorting*, ask whether the **second** half is still diagonal under that
        learned order (Spearman between the learned row position and each cell's first-firing frame in
        the held-out half, over cells active in both halves). If the sequence is a real property of the
        population, an order learned on one half predicts the other. If sorting merely overfit noise,
        the held-out score collapses to what a **random** order gives.
        """
    )
    return


@app.cell
def _(C_z, ENTRY, WIN_LEN, np, nu):
    from scipy.stats import spearmanr as _spcv
    _win = C_z[:, ENTRY:ENTRY + WIN_LEN]
    _half = WIN_LEN // 2
    _A_half, _B_half = _win[:, :_half], _win[:, _half:]
    def _heldout(_row_order):
        _b = _B_half[_row_order]
        _active = _b.max(axis=1) > 5.0
        _first = np.argmax(_b > 5.0, axis=1)[_active]
        if _active.sum() < 3:
            return 0.0
        _r, _ = _spcv(np.arange(int(_active.sum())), _first)
        return 0.0 if np.isnan(_r) else abs(float(_r))
    _learned = nu.sequence_sort(_A_half, thresh=5.0)
    cv_learned = _heldout(_learned)
    _rng = np.random.RandomState(1)
    cv_shuffle = np.array([_heldout(_rng.permutation(_win.shape[0])) for _ in range(500)])
    cv_p = float((1 + np.sum(cv_shuffle >= cv_learned)) / (1 + len(cv_shuffle)))
    return cv_learned, cv_p, cv_shuffle


@app.cell
def _(cv_learned, cv_p, cv_shuffle, np, nu):
    _groups = np.array(["random orders"] * len(cv_shuffle))
    _fig = nu.strip_points_fig(
        cv_shuffle, _groups, colors={"random orders": "#bab0ac"}, show_mean=False,
        ylabel="held-out sequenceness",
        title=(f"Does the order generalize?  learned = {cv_learned:.2f}  vs  500 random "
               f"(median {np.median(cv_shuffle):.2f})  ·  empirical p = {nu.fmt_p(cv_p)}"),
        height=430)
    # Red line = held-out score under the order LEARNED on the first half (a horizontal reference, not
    # a mean of the random cloud).
    _fig.add_hline(y=float(cv_learned), line=dict(color="#e45756", width=3),
                   annotation_text="order learned on first half", annotation_position="top left")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The order learned on the first half lands near **0.52**, above the random cloud (median ≈ 0.16),
        with an empirical **p ≈ 0.04** — real but modest evidence. We report it as "**about 21 of 500
        random orders matched or beat the learned one**", not "it lands outside the cloud", because the
        honest quantity is the tail count. And we do not oversell: only a handful of cells are active in
        both halves, so this is suggestive, not decisive. The point is the **method** — a sequence claim
        is worth making only if it survives a test that could have failed.

        ### Exercise 2 — the sort reveals, it does not create

        **Python skill practiced:** *calling functions you were given* (compose existing pieces). Fill
        the two blanks: score the window in its original order, then after applying the sort. **What you
        should see:** `seq_unsorted` ≈ 0.05 (no order in arbitrary CNMF numbering) and `seq_sorted` ≈
        0.79 (a clean diagonal). The jump is the whole result — sorting *revealed* a sequence already in
        the population; Exercise-1's split-half test is what tells us it is real and not manufactured.
        """
    )
    return


@app.cell
def _(C_z, ENTRY, WIN_LEN, np, nu):
    # ------------------------------------------------------------------ YOUR CODE (edit this cell)
    from scipy.stats import spearmanr as _spearmanr
    _thr = 5.0
    _win = C_z[:, ENTRY:ENTRY + WIN_LEN]               # fixed arena-entry window, shape (202, 5400)

    def _sequenceness(_raster):
        # |Spearman(row position, first-crossing frame)|: near 0 = no order, near 1 = clean diagonal.
        _first = np.argmax(_raster > _thr, axis=1)
        _r, _ = _spearmanr(np.arange(_raster.shape[0]), _first)
        return 0.0 if np.isnan(_r) else abs(float(_r))

    _order = nu.sequence_sort(_win, thresh=_thr)        # permutation: rows ordered by first firing

    # TODO line 1 — score the window in its ORIGINAL CNMF order.
    #   WHY: the baseline. CNMF numbers sources arbitrarily, so row index and firing time are unrelated
    #   and the score should be near 0.05. Replace ____ with the unsorted window `_win`.
    seq_unsorted = _sequenceness(_win)
    # TODO line 2 — score the window AFTER sorting the rows by first firing.
    #   WHY: if a real sequence exists, reordering by firing time makes the raster diagonal and the
    #   score jumps to ~0.79. Replace ____ with the SORTED window: index rows with `_order`, i.e. _win[_order].
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
            The unsorted window has near-zero sequenceness (CNMF's arbitrary order); after
            `sequence_sort` it rises to ~0.79. Sorting did not *create* structure, it **revealed** a
            temporal sequence already present. It is ~0.79 rather than 1.0 because ~144 of the 202
            neurons never cross the threshold in this window; among the ~58 that fire, the ordering is
            essentially perfect.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo, seq_sorted, seq_unsorted):
    # Self-check. Pinned: seq_unsorted ~ 0.05, seq_sorted ~ 0.79.
    _in_band = 0.65 <= seq_sorted <= 0.95
    _gain = seq_sorted - seq_unsorted > 0.4
    _ok = _in_band and _gain
    _c = "#e8f5e9" if _ok else "#ffebee"
    _b = "#2e7d32" if _ok else "#c62828"
    _m1 = (f"sorted sequenceness = {seq_sorted:.3f} — in the expected band [0.65, 0.95]" if _in_band
           else f"sorted sequenceness = {seq_sorted:.3f} — outside [0.65, 0.95]; check window/threshold")
    _m2 = (f"sorting beats the unsorted baseline ({seq_unsorted:.3f}) by "
           f"{seq_sorted - seq_unsorted:.3f} — a real sequence was revealed" if _gain
           else f"gain over baseline = {seq_sorted - seq_unsorted:.3f} is too small — did you sort the raster?")
    _head = "Pass — the sort reveals a neural sequence" if _ok else "Not yet — fix the flagged line"
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_b};padding:12px 16px;border-radius:6px">
        <b style="color:{_b}">{_head}</b><br>{_m1}<br>{_m2}<br>
        <span style="font-size:0.9em;color:#555">Pinned from the recording: sorted ≈ 0.79, unsorted ≈
        0.05. The score is |Spearman(row, first-firing)| — how diagonal the raster is.</span>
        </div>
        """
    )
    return


# ==================================================================== PART 4 — decoding
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # Part 4 · Decoding social state from the population

        **Why.** The single-neuron tests in Part 2 found only weak social tuning — no one cell cleanly
        separated social from non-social frames. But a behavior can be written **across** the
        population, a little in each of many cells, in a pattern no single neuron makes obvious. A
        **population decoder** asks whether the whole population, taken together, carries enough
        information to predict social state. This is the exact neural analog of the behavior decoder in
        notebook 6 — same estimator, same cross-validation, same AUROC — with one substitution: 19 pose
        features became a population of real neurons.

        **Definitions.**

        - **Population vector:** every neuron's activity at one frame. For a 218-neuron session, each
          frame is a vector of 218 numbers — one input to the decoder.
        - **Decode:** fit a model mapping the population vector to a label (`is_social_sender`). If it
          predicts above chance on frames it never trained on, the behavior is *decodable*.
        - **AUROC** (area under the ROC curve): one number for how well the predicted probabilities
          separate the two classes — 0.5 is chance, 1.0 perfect. Because only ~20% of frames are
          social, plain accuracy is misleading (always-guess-"no" scores ~80%), so AUROC is the
          yardstick.
        - **Cross-validation (CV):** score frames with a model that never saw them, so the number
          reflects generalization, not memorization.

        And here is where the whole course comes to a head, because **how you cross-validate decides the
        answer.**
        """
    )
    return


@app.cell
def _():
    # A fast, standardized logistic decoder (liblinear, C=0.1 keeps the many-collinear-neuron fits
    # quick AND is the honest regularized model). Returned as a factory so every decoder cell builds a
    # fresh clone. This IS the notebook 6 estimator, only the input changed (neurons, not pose feats).
    def mk_decoder():
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        return Pipeline([("scale", StandardScaler()),
                         ("lr", LogisticRegression(solver="liblinear", C=0.1,
                                                   class_weight="balanced", max_iter=1000))])
    return (mk_decoder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### The leakage trap — the culminating lesson of the course

        Calcium is **temporally autocorrelated**: a neuron's value in one frame is nearly its value in
        the next (the slow calcium decay guarantees it). That single fact breaks the most common
        cross-validation choice. If we split frames into folds **at random** (`shuffle=True`), a frame
        lands in the test set while the frame right next to it — almost identical — sits in the training
        set. The model does not have to *generalize*; it can nearly *look up* the answer from a neighbor
        it already saw. The AUROC comes out spuriously high. This is **leakage**.

        The fix is to keep time-adjacent frames together: split into **contiguous blocks** so a test
        block is separated in time from its training data. `nu.blocked_cv_auroc` runs the same decoder
        under three schemes — `shuffle` (the leaky one), `blocked` (contiguous test blocks), and
        `contiguous` (one strict forward-in-time split) — and returns per-fold AUROC. Below, session 6,
        the same neurons and the same model under all three.
        """
    )
    return


@app.cell
def _(mk_decoder, np, nu, sess_neurons, sess_social):
    # The leakage comparison: identical decoder + data, three CV schemes. This is the headline number.
    _X = sess_neurons
    _y = sess_social.astype(int)
    cv_schemes = {}
    for _name in ["shuffle", "blocked", "contiguous"]:
        cv_schemes[_name] = nu.blocked_cv_auroc(_X, _y, scheme=_name, clf=mk_decoder())
    return (cv_schemes,)


@app.cell
def _(cv_schemes, go, np, nu, si_session_pick):
    _order = ["shuffle", "blocked", "contiguous"]
    _labels = {"shuffle": "shuffle CV<br>(random — LEAKS)",
               "blocked": "blocked CV<br>(contiguous folds)",
               "contiguous": "contiguous<br>(1 forward split)"}
    _cols = {"shuffle": "#e45756", "blocked": "#4c78a8", "contiguous": "#54a24b"}
    _fig = go.Figure()
    for _i, _k in enumerate(_order):
        _v = np.asarray(cv_schemes[_k], float)
        _v = _v[np.isfinite(_v)]
        _fig.add_scatter(x=np.full(len(_v), _i) + np.random.default_rng(_i).uniform(-0.06, 0.06, len(_v)),
                         y=_v, mode="markers", marker=dict(size=11, color=_cols[_k],
                         line=dict(width=0.6, color="white")), name=_labels[_k].replace("<br>", " "),
                         showlegend=False, hovertemplate=f"{_k}: "+"%{y:.3f}<extra></extra>")
        _fig.add_scatter(x=[_i], y=[float(np.mean(_v))], mode="markers",
                         marker=dict(size=18, color=_cols[_k], symbol="line-ew",
                                     line=dict(width=3, color=_cols[_k])), showlegend=False,
                         hovertemplate=f"{_k} mean: {np.mean(_v):.3f}<extra></extra>")
    _fig.add_hline(y=0.5, line=dict(color="#999", width=1, dash="dash"),
                   annotation_text="chance", annotation_position="right")
    _fig.update_xaxes(tickmode="array", tickvals=[0, 1, 2], ticktext=[_labels[k] for k in _order])
    _fig.update_yaxes(title="held-out AUROC", range=[0.45, 1.0])
    nu.apply_house_style(_fig, title=f"Same decoder, same neurons ({si_session_pick.selected_key}) — three CV schemes",
                         legend=None, height=440)
    _fig
    return


@app.cell(hide_code=True)
def _(cv_schemes, mo, np):
    _sh = float(np.nanmean(cv_schemes["shuffle"]))
    _bl = float(np.nanmean(cv_schemes["blocked"]))
    _co = float(np.nanmean(cv_schemes["contiguous"]))
    mo.md(
        f"""
        <div style="background:#fdecea;border-left:6px solid #e45756;padding:14px 18px;border-radius:6px">
        <b>The same data, the same model — three different answers.</b><br>
        Random (leaky) CV: AUROC ≈ <b>{_sh:.2f}</b>. Blocked CV: ≈ <b>{_bl:.2f}</b>. Single contiguous
        split: ≈ <b>{_co:.2f}</b>.<br>
        <span style="font-size:0.95em">The leaky number is the one the old analysis reported (~0.95),
        and it overstates the truth by roughly <b>{_sh - _bl:.2f} AUROC</b>. The honest, time-respecting
        estimate is far lower — a modest signal, not a near-perfect read. Nothing about the neurons
        changed between these bars; only the cross-validation did. This is the single most important
        methodological point in the course: on an autocorrelated time series, a random train/test split
        <b>leaks</b>, and the inflation can be enormous.</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Where the decoder is right and wrong — the operating point

        With the honest (blocked) predictions in hand, we can *see* the decoder behave. Below is its
        held-out probability of "social" over time, with the true social frames shaded green, and an
        **ROC curve** with a movable **decision threshold**. The decoder outputs a *probability*; a
        threshold turns it into a yes/no call. Low threshold → catch more real bouts but more false
        alarms (high recall, low precision); high threshold → stricter (high precision, low recall).
        Slide it and read the operating point, precision, recall, and confusion counts — all on
        held-out, blocked predictions.
        """
    )
    return


@app.cell
def _(mk_decoder, np, sess_neurons, sess_social):
    # Honest per-frame held-out probabilities via blocked CV (KFold with shuffle=False = contiguous
    # folds), so the timeline and ROC reflect the time-respecting number, not the leaky one.
    from sklearn.model_selection import KFold as _KFold, cross_val_predict as _cvp
    from sklearn.metrics import roc_auc_score as _auc, roc_curve as _roc
    _X = sess_neurons
    _y = sess_social.astype(int)
    dec_proba = _cvp(mk_decoder(), _X, _y, cv=_KFold(5, shuffle=False), method="predict_proba")[:, 1]
    dec_y = _y
    dec_auc = float(_auc(dec_y, dec_proba))
    dec_fpr, dec_tpr, _ = _roc(dec_y, dec_proba)
    return dec_auc, dec_fpr, dec_proba, dec_tpr, dec_y


@app.cell
def _(dec_proba, dec_y, go, np, nu):
    _t = np.arange(len(dec_proba))
    _fig = go.Figure()
    _edges = np.flatnonzero(np.diff(np.r_[0, dec_y, 0]))
    for _a, _b in zip(_edges[0::2], _edges[1::2]):
        _fig.add_vrect(x0=_a, x1=_b, fillcolor="#2ca02c", opacity=0.15, line_width=0, layer="below")
    _fig.add_scatter(x=_t, y=dec_proba, mode="lines", line=dict(color="#4c78a8", width=1),
                     name="P(social), held-out")
    _fig.add_hline(y=0.5, line=dict(color="#999", width=1, dash="dash"))
    _fig.update_yaxes(range=[-0.02, 1.02], title="P(social)")
    _fig.update_xaxes(title="time (frames, 25 fps)")
    nu.apply_house_style(_fig, title="Held-out P(social) over time · green = truly social (blocked CV)",
                         legend="below", height=320)
    _fig
    return


@app.cell
def _(mo):
    thr_slider = mo.ui.slider(0.05, 0.95, value=0.5, step=0.05,
                              label="decision threshold", debounce=True, full_width=True)
    return (thr_slider,)


@app.cell
def _(dec_auc, dec_fpr, dec_proba, dec_tpr, dec_y, go, mo, nu, thr_slider):
    _thr = float(thr_slider.value)
    _pred = (dec_proba >= _thr).astype(int)
    _tp = int(((_pred == 1) & (dec_y == 1)).sum()); _fp = int(((_pred == 1) & (dec_y == 0)).sum())
    _fn = int(((_pred == 0) & (dec_y == 1)).sum()); _tn = int(((_pred == 0) & (dec_y == 0)).sum())
    _prec = _tp / (_tp + _fp) if (_tp + _fp) else 0.0
    _rec = _tp / (_tp + _fn) if (_tp + _fn) else 0.0
    _fpr_here = _fp / (_fp + _tn) if (_fp + _tn) else 0.0
    _roc = go.Figure()
    _roc.add_scatter(x=dec_fpr, y=dec_tpr, mode="lines", line=dict(color="#4c78a8", width=2),
                     name=f"ROC (AUC {dec_auc:.3f}, blocked)")
    _roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#bbb", width=1, dash="dash"),
                     name="chance")
    _roc.add_scatter(x=[_fpr_here], y=[_rec], mode="markers",
                     marker=dict(color="#e45756", size=13, symbol="x"), name=f"threshold {_thr:.2f}")
    _roc.update_xaxes(title="false-positive rate", range=[-0.02, 1.02])
    _roc.update_yaxes(title="true-positive rate", range=[-0.02, 1.02], scaleanchor="x", scaleratio=1)
    nu.apply_house_style(_roc, title=f"@ threshold {_thr:.2f}: precision {_prec:.2f} · recall {_rec:.2f} "
                                     f"(TP {_tp} · FP {_fp} · FN {_fn} · TN {_tn})",
                         legend="below", height=460)
    mo.vstack([thr_slider, _roc])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Does the decoder work in every session, and does it track neuron count?

        **Why.** Session 6 might be lucky, and the old analysis argued "more neurons → better decoding"
        (a plausible story: a longer population vector carries more to read). But that relationship was
        measured under the **leaky** CV. Here we test it honestly. The button runs the full pipeline +
        blocked CV for all 18 sessions and a label-shuffle chance band on session 6 (a few seconds).
        """
    )
    return


@app.cell
def _(mo):
    run_all = mo.ui.run_button(label="compute honest (blocked-CV) decoders for all 18 sessions")
    return (run_all,)


@app.cell
def _(mk_decoder, mo, np, nu, run_all, sess_neurons, sess_neurons_all,
      sess_social, sess_social_all):
    mo.stop(not run_all.value,
            mo.md("*Click the button above to run all 18 blocked-CV decoders (heavy — a few seconds).*"))
    from scipy.stats import spearmanr as _sp_all

    _aucs, _ncells = [], []
    for _s in range(len(sess_neurons_all)):
        _X = sess_neurons_all[_s]; _y = sess_social_all[_s].astype(int)
        _ncells.append(_X.shape[1])
        _aucs.append(float(np.nanmean(nu.blocked_cv_auroc(_X, _y, scheme="blocked", clf=mk_decoder()))))
    all_aucs = np.array(_aucs); all_ncells = np.array(_ncells)
    _rho, _p = _sp_all(all_ncells, all_aucs)
    all_rho, all_rho_p = float(_rho), float(_p)

    # Label-shuffle empirical chance band on the selected session (blocked CV, permuted labels).
    _rng = np.random.default_rng(0)
    all_chance = np.array([float(np.nanmean(nu.blocked_cv_auroc(
        sess_neurons, _rng.permutation(sess_social.astype(int)), scheme="blocked", clf=mk_decoder())))
        for _ in range(15)])
    return all_aucs, all_chance, all_ncells, all_rho, all_rho_p


@app.cell
def _(all_aucs, all_chance, all_ncells, all_rho, all_rho_p, go, np, nu):
    _fig = go.Figure()
    # label-shuffle chance band (from the selected session) as a shaded reference
    _lo, _hi = float(np.percentile(all_chance, 5)), float(np.percentile(all_chance, 95))
    _fig.add_hrect(y0=_lo, y1=_hi, fillcolor="#bbbbbb", opacity=0.30, line_width=0,
                   annotation_text="label-shuffle chance band", annotation_position="top left")
    _fig.add_hline(y=0.5, line=dict(color="#999", width=1, dash="dash"))
    _fig.add_scatter(x=all_ncells, y=all_aucs, mode="markers",
                     marker=dict(size=13, color="#4c78a8", line=dict(width=0.6, color="white")),
                     text=[f"{n} neurons" for n in all_ncells],
                     hovertemplate="%{text}<br>AUROC %{y:.3f}<extra></extra>", showlegend=False)
    _fig.update_xaxes(title="neurons recorded in the session")
    _fig.update_yaxes(title="honest (blocked-CV) AUROC")
    nu.apply_house_style(_fig, title=(f"Blocked-CV AUROC vs neuron count · Spearman ρ = {all_rho:+.2f}, "
                                      f"p = {nu.fmt_p(all_rho_p)} — no relationship"),
                         legend=None, height=440)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Two honest findings the leaky analysis hid. First, under blocked CV the population signal is
        **modest** — the median session sits near 0.55, most above the label-shuffle chance band but not
        dramatically, and one session dips to chance. The population *does* carry social information, but
        far less cleanly than the leaky ~0.93 suggested. Second, the "more neurons → better decoding"
        story **evaporates**: honestly cross-validated AUROC is uncorrelated with neuron count
        (ρ ≈ 0, p ≈ 1). That apparent relationship was itself partly a leakage artifact — more neurons
        give the leaky model more room to overfit-then-look-up, inflating the *random-CV* number more
        for larger populations. Correcting the CV dissolves the confound. Two audit findings, one fix.

        ### Exercise 3 — recompute the AUROC under shuffle vs blocked CV

        **Python skill practiced:** *calling a library helper twice and comparing.* Everything is set
        up; fill the two blanks to score session 6 under each scheme with `nu.blocked_cv_auroc`, then
        the self-check confirms the leaky number is far higher. This is the leakage lesson in your own
        hands.
        """
    )
    return


@app.cell
def _(mk_decoder, np, nu, sess_neurons, sess_social):
    # ------------------------------------------------------------------ YOUR CODE (edit this cell)
    _X = sess_neurons
    _y = sess_social.astype(int)

    # TODO 1 — the LEAKY estimate: random StratifiedKFold. Replace ____ with the string "shuffle".
    #   WHY: shuffle scatters time-adjacent (near-identical) frames across train and test, so the model
    #   can nearly look up each answer from a neighbor — the AUROC comes out spuriously high (~0.95).
    auc_shuffle = float(np.nanmean(nu.blocked_cv_auroc(_X, _y, scheme="shuffle", clf=mk_decoder())))
    # TODO 2 — the HONEST estimate: contiguous time blocks. Replace ____ with the string "blocked".
    #   WHY: blocked keeps time-adjacent frames together, so the test fold is genuinely unseen — the
    #   AUROC drops to the honest value (~0.6), the number you would actually trust.
    auc_blocked = float(np.nanmean(nu.blocked_cv_auroc(_X, _y, scheme="blocked", clf=mk_decoder())))
    leakage_gap = auc_shuffle - auc_blocked            # how much the random split overstated things
    # ---------------------------------------------------------------------------------------------
    return auc_blocked, auc_shuffle, leakage_gap


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show solution": mo.md(
            r"""
            ```python
            auc_shuffle = np.nanmean(nu.blocked_cv_auroc(X, y, scheme="shuffle", clf=mk_decoder()))  # ~0.95
            auc_blocked = np.nanmean(nu.blocked_cv_auroc(X, y, scheme="blocked", clf=mk_decoder()))  # ~0.63
            leakage_gap = auc_shuffle - auc_blocked                                                  # ~0.32
            ```
            The random split reports ~0.95; the honest blocked split reports ~0.63. The ~0.3 gap is
            pure leakage — the same neurons, the same model, only the CV changed. Whenever your samples
            are autocorrelated in time (calcium frames, video frames, anything sampled fast), a random
            KFold overstates performance and a time-respecting split is the honest one.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(auc_blocked, auc_shuffle, leakage_gap, mo):
    _p1 = float(auc_shuffle) > 0.85
    _p2 = 0.50 < float(auc_blocked) < 0.75
    _p3 = float(leakage_gap) > 0.15
    _ok = _p1 and _p2 and _p3
    _c = "#e8f5e9" if _ok else "#ffebee"
    _b = "#2e7d32" if _ok else "#c62828"
    _m1 = (f"shuffle (leaky) AUROC = {auc_shuffle:.3f} — high, as leakage predicts" if _p1
           else f"shuffle AUROC = {auc_shuffle:.3f} — expected &gt; 0.85; did you pass scheme='shuffle'?")
    _m2 = (f"blocked (honest) AUROC = {auc_blocked:.3f} — a modest, trustworthy signal" if _p2
           else f"blocked AUROC = {auc_blocked:.3f} — expected ~0.6; did you pass scheme='blocked'?")
    _m3 = (f"leakage gap = {leakage_gap:.3f}: the random split overstated the truth by this much" if _p3
           else f"leakage gap = {leakage_gap:.3f} looks too small — recheck both schemes")
    _head = "Pass — you measured the leakage yourself" if _ok else "Not yet — fix the flagged part"
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_b};padding:12px 16px;border-radius:6px">
        <b style="color:{_b}">{_head}</b><br>{_m1}<br>{_m2}<br>{_m3}<br>
        <span style="font-size:0.9em;color:#555">Tolerance: shuffle &gt; 0.85, blocked ∈ (0.50, 0.75),
        gap &gt; 0.15. The honest number is the blocked one.</span>
        </div>
        """
    )
    return


# ==================================================================== close
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Where this leaves us — and the arc of the course

        <div style="background:#eef4fb;border-left:6px solid #1f77b4;padding:14px 18px;border-radius:6px">
        <b>Same math, different object — the ledger.</b><br>
        <b>Held:</b> we factorized the movie exactly as PCA factorized behavior (<code>Y ≈ A·C</code>);
        we asked what a single unit encodes with the same tuning-curve-plus-shuffle logic we used for
        behavioral features; we decoded a labeled state with the same logistic + threshold + permutation
        machinery as notebook 6; and the leakage lesson — random CV overstates an autocorrelated series
        — applies identically to calcium frames and to fast-sampled behavior.<br>
        <b>Broke:</b> behavior handed us a fixed 19-column matrix, poolable across all animals; neural
        data handed us a per-session, variable-N population (12–396 cells) with no identity across
        recordings, so the decoder lives <i>within</i> one session. Making it transfer — cell
        registration or a shared latent space — is a real research problem this course hands to the
        literature.
        </div>

        **The questions we asked, and the answers.** *What does a population represent, and can we read
        a behavior off it?* A single neuron can be sharply tuned to **space** (a place cell that beats
        its own shuffle) but only weakly to **social contact**. The population, however, fires in an
        ordered **sequence** around arena entry that survives a split-half test (p ≈ 0.04), and a
        **population decoder** reads social state above chance in most sessions. But the honest,
        time-respecting AUROC is **modest (~0.6)**, not the leaky ~0.95 — and the "more neurons, better
        decoding" story was itself a leakage artifact. The most important thing this notebook teaches is
        not that the brain encodes social behavior; it is *how to know* whether your number is real.

        **Closing the loop.** We opened the course with keypoints on a mouse and asked what behavior
        *is* — building it into features, a manifold, a grammar, a decoder, and a discipline for telling
        signal from artifact. We close by running those same moves on real neurons and reading a social
        state off the population, with the same skepticism. We began with behavior; we ended by reading
        it off the brain. That is the whole arc.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Background — datasets, methods, and the honest limits": mo.md(
            r"""
            **Datasets.** (1) The striatal CNMF-E session `221007_4-0_D2` (202 neurons, ~16,800 frames)
            — one-photon miniscope calcium, demixed with **CNMF-E** (Zhou et al. 2018, *eLife* 7:e28728;
            the base method is Pnevmatikakis et al. 2016, *Neuron* 89:285). (2) The rat place/grid
            sessions, adapted from the **NEU 457** (Princeton) problem set by Talmo Pereira, Andrew
            Leifer, and David Tank. (3) The **SI3_2022** social-isolation cohort (18 sessions, striatal
            miniscope + social scoring).

            **Place & grid cells.** O'Keefe & Dostrovsky 1971 (*Brain Res.* 34:171); Hafting et al. 2005
            (*Nature* 436:801). 2014 Nobel Prize: O'Keefe; May-Britt and Edvard Moser.

            **Population decoding of social variables.** Padilla-Coreano et al. 2022 (*Nature*) — mPFC
            ensembles decode competitive rank and social behavior — is the closest precedent for the
            decoder here.

            **Honest limits.** (a) *Correlation, not cause* — a decodable signal does not mean these
            cells drive behavior; causation needs perturbation. (b) *Proxy variables* — rat "position"
            is an eye-centroid gaze proxy; social decoding uses interpolated two-clock alignment. (c)
            *Within-session only* — no neuron identity across recordings. (d) *The gridness caveat* — a
            positive gridness score on hippocampal gaze-proxy data is a border artifact, not a grid
            call. (e) *Small n* — six sessions per condition; treat condition trends as descriptive. The
            through-line of the whole notebook: a number is a result only once it beats a matched
            control under an honest cross-validation.
            """
        )
    })
    return


if __name__ == "__main__":
    app.run()
