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
        # NB09 · Place and grid cells

        ## Where we are in the argument

        The previous notebook took a raw one-photon microscope movie — a grey field of flickering
        light — and pulled out of it the activity of individual neurons: one trace per cell, each
        rising when its neuron fires. The question it answered was **"how do we get a clean neural
        signal out of the movie?"** We now hold, for a recording session, a set of per-neuron signals
        aligned frame by frame with the animal's behavior.

        That raises the next question, and it is the one this notebook asks:

        > **Do individual neurons encode where the animal is?**

        This is the founding question of systems neuroscience about space. It is worth stating plainly
        why we are asking it *here*, in a course about social behavior. In Week 1 we described every
        interaction in each mouse's **own body frame**: we moved the coordinate system onto the animal
        and rotated it so the animal's heading always pointed the same way (the *egocentric* view). We
        did that because behavior toward another animal is organized around the acting animal's body —
        who it is facing, how far the partner is, which way it is turning. Choosing that reference
        frame was not a bookkeeping convenience; it was a claim about what the brain cares about.

        The brain, it turns out, also maintains a **second** reference frame: a fixed, arena-anchored
        map of the outside world that does not move when the animal turns (the *allocentric* view).
        This notebook works with real neural recordings in which that map is directly visible, so the
        claim "space is represented in the brain" stops being an assertion and becomes a quantity we
        measure — with the same tools (a tuning curve; a firing rate that depends on a variable) we
        used for social geometry in Week 1.

        ## The terms you will need

        - **Egocentric frame:** the world described relative to the animal's own body (left/right,
          front/back). This is the frame we built in Week 1.
        - **Allocentric frame:** the world described relative to fixed external landmarks (a corner
          of the arena, a wall). It does not rotate when the animal turns.
        - **Place cell** (O'Keefe & Dostrovsky, 1971): a neuron that fires only when the animal is in
          one particular location in the arena. The region it fires in is its **place field**.
        - **Grid cell** (Hafting et al., 2005): a neuron that fires at *many* locations arranged in a
          repeating triangular lattice that tiles the whole arena. We build the standard tool for
          detecting them (the spatial autocorrelogram) in Section 7.
        - **Rate map:** a **two-dimensional spatial tuning curve** — a picture of a neuron's firing
          rate as a function of the animal's (x, y) location. It is the central object of the notebook
          and is defined carefully in Section 4.
        - **Spatial information:** a single number, in bits per spike, that summarizes how much a
          neuron's firing tells you about where the animal is (Section 5).

        Place cells and grid cells, together with head-direction cells, are the neurons that carry the
        allocentric map. Their discovery earned the 2014 Nobel Prize (O'Keefe; May-Britt and Edvard
        Moser).

        ## What we will do

        We have, for each session, the animal's position at every frame and the number of spikes each
        neuron fired at every frame. We will turn *(where the animal was)* combined with *(when each
        neuron fired)* into *(where in the arena each neuron prefers to fire)* — first by hand, as
        dots on a path (Section 2), then as a smooth density (Section 3), then as the properly
        normalized rate map (Section 4). We then reduce each neuron to one spatial-information number
        (Section 5), and — crucially — we refuse to trust that number until it beats a control built
        from the neuron's own spikes (Section 6).

        ## One honest caveat, stated up front

        In this dataset the animal's "position" is **not** a clean body-on-arena readout. It is the
        average of two tracked **eye** positions, so it is a gaze / eye-in-head proxy rather than the
        body position that classic place-cell rigs record. The fields we recover will be weaker and
        noisier than a textbook place field, and some apparent tuning may reflect gaze or movement
        rather than pure location. That imperfection is not a reason to distrust the whole exercise —
        it is exactly the situation real data puts us in, and it is why the final test grades each
        cell against a matched control rather than against the prettiest-looking map.
        """
    )
    return


@app.cell
def _(ROOT, nu):
    import os as _os
    # Download + unzip the NEU 457 rat place/grid data (Dropbox), then parse all three sessions.
    # fetch_zip_dropbox is cached: it only re-downloads if the .mat files are missing.
    _rat_dir = nu.fetch_zip_dropbox(root=ROOT)
    sessions = {name: nu.load_rat_mat(_os.path.join(_rat_dir, name)) for name in nu.RAT_FILES}
    session_names = list(nu.RAT_FILES)
    return session_names, sessions


@app.cell(hide_code=True)
def _(mo, sessions, session_names):
    _rows = "\n".join(
        f"| `{n}` | {sessions[n]['spikes'].shape[0]:,} | {sessions[n]['spikes'].shape[1]} |"
        for n in session_names
    )
    mo.md(
        f"""
        ---
        ## 1. The inputs — a path and some spikes

        **Why.** To ask "where does this neuron prefer to fire?" we need exactly two things aligned in
        time: the animal's location at every frame, and the number of spikes each neuron produced at
        every frame. Everything else in the notebook is built from these two arrays. Before we compute
        anything we look carefully at both, because their quirks (how much of the arena was covered,
        how many spikes each neuron fired) determine which conclusions are trustworthy.

        **Definitions.**

        - **Session:** one continuous recording. We loaded three.
        - **Eye positions** `left`, `right`: the tracked (x, y) pixel positions of the two eyes,
          shape `(T, 2)` where `T` is the number of frames.
        - **Centroid:** `centroid = (left + right) / 2`, shape `(T, 2)`. This is our estimate of the
          animal's position at each frame (the gaze proxy described above).
        - **Spike-count matrix** `spikes`: shape `(T, n_neurons)`; entry `[t, j]` is how many times
          neuron `j` fired in frame `t`.

        | session | frames T | neurons |
        |---|---|---|
        {_rows}

        The first session (`20160609T194655`) has by far the most neurons (14) and the clearest
        fields, so we make it the default. Use the dropdown below to change which session drives every
        plot in the notebook, and watch how the number and quality of fields changes across recordings.
        """
    )
    return


@app.cell
def _(mo, session_names):
    session_pick = mo.ui.dropdown(options=session_names, value=session_names[0],
                                  label="recording session (drives all plots)")
    return (session_pick,)


@app.cell
def _(go, mo, np, session_pick, sessions):
    # Trajectory: the animal's position over the whole session, colored by time.
    _d = sessions[session_pick.value]
    _ctr = _d["centroid"]
    _ok = np.isfinite(_ctr).all(axis=1)
    _c = _ctr[_ok]
    _fig = go.Figure()
    _fig.add_scatter(x=_c[:, 0], y=_c[:, 1], mode="lines",
                     line=dict(color="#c9c9c9", width=1), name="path", showlegend=False)
    _fig.add_scatter(x=_c[:, 0], y=_c[:, 1], mode="markers",
                     marker=dict(color=np.arange(len(_c)), colorscale="Viridis", size=3,
                                 colorbar=dict(title="frame"), showscale=True),
                     name="time", showlegend=False, opacity=0.6)
    _fig.update_yaxes(scaleanchor="x", scaleratio=1, title="eye-centroid y (px)")
    _fig.update_xaxes(title="eye-centroid x (px)")
    _fig.update_layout(template="plotly_white", height=460, margin=dict(l=10, r=10, t=50, b=10),
                       title=f"Trajectory · {session_pick.value} · {len(_c):,} tracked frames")
    mo.vstack([session_pick, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The plot above shows the whole path the animal took, colored from early (dark) to late
        (yellow) frames. It answers a coverage question: which parts of the arena were visited, and
        how heavily. A region the animal returned to many times looks densely painted; a region it
        rarely entered is nearly empty.

        Coverage matters more than it first appears. A neuron cannot show a field in a place the
        animal never went, and — more subtly — a neuron can *look* tuned to a spot simply because the
        animal spent most of its time there. To see that confound directly, we next turn the path into
        an **occupancy map**: how many frames the animal spent in each small square of the arena.
        """
    )
    return


@app.cell
def _(mo, np, nu, session_pick, sessions):
    # OCCUPANCY EDA: how many frames the animal spent in each spatial bin. This is the denominator
    # that Section 4 divides by. Uneven occupancy is the single biggest confound in place-cell work.
    _d = sessions[session_pick.value]
    _ctr = _d["centroid"]
    _rm = nu.rate_map(_ctr[:, 0], _ctr[:, 1], np.zeros(len(_ctr)), bins=25)  # spikes=0 -> just occupancy
    _occ = _rm["occupancy"]
    _xc = 0.5 * (_rm["xedges"][:-1] + _rm["xedges"][1:])
    _yc = 0.5 * (_rm["yedges"][:-1] + _rm["yedges"][1:])
    _visited = float((_occ > 0).mean())
    _fig = nu.heatmap_fig(
        _occ.T, x=_xc, y=_yc,
        title=(f"Occupancy · {session_pick.value} · {_visited*100:.0f}% of bins visited · "
               "brighter = more time spent"),
        xlabel="x (px)", ylabel="y (px)", colorscale="Cividis",
        colorbar_title="frames", height=480)
    _fig.update_yaxes(scaleanchor="x", scaleratio=1)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The occupancy map is deliberately unglamorous, but it is the quantity that makes rate maps
        honest. The animal did not sample the arena uniformly: some bins hold hundreds of frames,
        others one or two. If we simply counted spikes per location we would systematically
        over-credit the heavily occupied bins. Section 4 fixes this by dividing spike counts by exactly
        this occupancy, so keep the bright and dark regions here in mind.

        Now replay the same path frame by frame. Press play. Watching the animal move makes concrete
        that "position" is a value that changes continuously across the session — the thing each
        neuron may or may not be tracking.
        """
    )
    return


@app.cell
def _(go, np, session_pick, sessions):
    # Animated replay of the trajectory: a moving dot leaves a growing trail. Subsampled and capped
    # at ~40 animation frames so the exported HTML stays light. Static export shows the first frame.
    _d = sessions[session_pick.value]
    _ctr = _d["centroid"]
    _c = _ctr[np.isfinite(_ctr).all(axis=1)]
    _step = max(1, len(_c) // 1200)
    _cs = _c[::_step]
    _cuts = np.linspace(2, len(_cs), 40).astype(int)
    _frames = [
        go.Frame(data=[
            go.Scatter(x=_cs[:k, 0], y=_cs[:k, 1], mode="lines",
                       line=dict(color="#9a9a9a", width=1)),
            go.Scatter(x=[_cs[k - 1, 0]], y=[_cs[k - 1, 1]], mode="markers",
                       marker=dict(color="#e45756", size=11)),
        ], name=str(int(k)))
        for k in _cuts
    ]
    _k0 = _cuts[0]
    _fig = go.Figure(
        data=[
            go.Scatter(x=_cs[:_k0, 0], y=_cs[:_k0, 1], mode="lines",
                       line=dict(color="#9a9a9a", width=1), name="path", showlegend=False),
            go.Scatter(x=[_cs[_k0 - 1, 0]], y=[_cs[_k0 - 1, 1]], mode="markers",
                       marker=dict(color="#e45756", size=11), name="animal", showlegend=False),
        ],
        frames=_frames,
    )
    _fig.update_yaxes(scaleanchor="x", scaleratio=1, title="y (px)",
                      range=[_cs[:, 1].min(), _cs[:, 1].max()])
    _fig.update_xaxes(title="x (px)", range=[_cs[:, 0].min(), _cs[:, 0].max()])
    _fig.update_layout(
        template="plotly_white", height=480, margin=dict(l=10, r=10, t=50, b=10),
        title=f"Trajectory replay · {session_pick.value}",
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=1.08, xanchor="left",
                          buttons=[dict(label="play", method="animate",
                                        args=[None, {"frame": {"duration": 90, "redraw": True},
                                                     "fromcurrent": True}])])])
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The other input is the spike side. Before we look at any single neuron's spatial pattern, it
        pays to see how **many** spikes each neuron fired across the session. This one number sets an
        upper bound on how reliable that neuron's map can be: a neuron that fired only a few dozen
        times cannot paint a stable field, and — as Section 6 will show — it can post a deceptively
        high spatial-information score purely by chance. The plot below shows every neuron as an
        individual point; hover to read its exact spike total.
        """
    )
    return


@app.cell
def _(go, np, session_pick, sessions):
    # SPIKE-COUNT EDA: total spike-frames per neuron, one interactive point per neuron. Individual
    # points (house style), not a bar chart. The spread here is enormous and it matters downstream.
    _d = sessions[session_pick.value]
    _spk = _d["spikes"]
    _counts = np.array([int((_spk[:, _j] > 0).sum()) for _j in range(_spk.shape[1])])
    _fig = go.Figure(go.Scatter(
        x=np.arange(len(_counts)), y=_counts, mode="markers",
        marker=dict(size=13, color=_counts, colorscale="Viridis",
                    colorbar=dict(title="spike-frames"), line=dict(width=0.5, color="white")),
        text=[f"neuron {_j}" for _j in range(len(_counts))],
        hovertemplate="%{text}<br>%{y:,} spike-frames<extra></extra>"))
    _fig.update_xaxes(title="neuron", tickmode="array", tickvals=list(range(len(_counts))),
                      ticktext=[f"n{_j}" for _j in range(len(_counts))])
    _fig.update_yaxes(title="spike-frames (frames with ≥1 spike)")
    _fig.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=50, b=10),
                       title=f"Spikes per neuron · {session_pick.value} · note the ~40× spread")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The spread is large — in the default session the busiest neuron fires in thousands of frames
        while the quietest fires in fewer than a hundred. Hold that fact: when we rank neurons by
        spatial information in Section 5, the very quietest neurons will be the ones we most need to
        double-check, because sparse spikes are exactly what fools the metric.

        ---
        ## 2. Spikes on the path

        **Why.** The simplest way to see whether a neuron is tuned to location is to mark every
        position at which it fired and look at where those marks land. This is the picture O'Keefe drew
        by hand in 1971, and it requires no statistics at all — just an overlay.

        **Definition.** In the plot below the blue line is the full trajectory (everywhere the animal
        went) and each red dot is the animal's position at a frame in which the chosen neuron fired at
        least once. If the red dots concentrate in one region while the animal clearly visited the
        whole arena, that region is a candidate place field. If the red dots are spread evenly along
        the path, the neuron is probably not spatially tuned.

        **Method.** `nu.overlay_fig(trajectory, spike_positions, ...)` draws these two layers.

        - *Purpose:* overlay event locations on a movement path.
        - *Inputs:* the full set of tracked positions `(T, 2)`, and the subset of positions where the
          neuron spiked `(K, 2)`.
        - *Output:* the figure.

        Move the `neuron` slider to change which neuron's spikes are shown. Some neurons paint a tight
        patch (a candidate field); others sprinkle everywhere (not spatial). In the default session,
        neurons 4, 5, 6, 8, and 12 are worth a close look; neuron 2 is a good example of a
        non-spatial neuron.
        """
    )
    return


@app.cell
def _(mo):
    neuron_ov = mo.ui.slider(0, 13, value=5, step=1, label="neuron",
                             debounce=True, full_width=True)
    return (neuron_ov,)


@app.cell
def _(mo, neuron_ov, np, nu, session_pick, sessions):
    _d = sessions[session_pick.value]
    _ctr = _d["centroid"]
    _spk = _d["spikes"]
    _n = _spk.shape[1]
    _ni = min(neuron_ov.value, _n - 1)                 # clamp: sessions have 14 / 6 / 5 neurons
    _spiking = (_spk[:, _ni] > 0) & np.isfinite(_ctr).all(axis=1)
    _fig = nu.overlay_fig(
        _ctr[np.isfinite(_ctr).all(axis=1)], _ctr[_spiking],
        title=f"Spikes on path · neuron {_ni} · {int(_spiking.sum())} spike-frames · {session_pick.value}",
        traj_name="path", pts_name="spikes", height=520)
    mo.vstack([neuron_ov, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Sweep the slider slowly and you are, in effect, hand-screening the population for spatial
        tuning the way the field did for its first two decades. Notice two things. First, a tight red
        patch is genuinely convincing — the animal visited the whole arena (the blue line covers it)
        yet the neuron only fired in one corner. Second, the picture is noisier than a textbook figure,
        which is the eye-position proxy showing through: even the good cells have stray dots.

        The dot cloud is suggestive but hard to *compare* across neurons — a cluster of dots and a
        slightly denser cluster look alike to the eye. The next two sections turn the dots into
        numbers: first a smooth density surface, then the properly normalized rate map.

        ---
        ## 3. From dots to a smooth field — kernel density estimation

        **Why.** Converting the dots into a smooth surface makes the shape and location of a field
        easier to read and to compare between neurons.

        **Definitions.**

        - **Kernel density estimate (KDE):** a way to turn scattered points into a smooth density.
          Place a small Gaussian bump on the arena at each spike position, then add all the bumps
          together. Where spikes are dense the bumps overlap and the surface is high; where spikes are
          sparse the surface is low.
        - **Bandwidth:** the width of each Gaussian bump — a smoothing control with a trade-off. Too
          small and the surface breaks into one lump per spike (noisy); too large and separate fields
          blur into one blob (over-smoothed).

        **Method.** `scipy.stats.gaussian_kde`

        - *Purpose:* estimate a smooth density from scattered points.
        - *Inputs:* the spike positions `(2, K)`.
        - *Output:* a callable you evaluate on a grid to get the density surface.

        We start from its automatic bandwidth and multiply it by the slider value (the same
        `bw_adjust` idea seaborn's `kdeplot` uses). Drag `bandwidth` and find a setting where a real
        field holds together while a diffuse neuron stays diffuse.

        **Limitation — and this is the whole point of Section 4.** This surface shows where the neuron
        *fired*, but not where it fires *given that the animal was there*. A neuron can look like it
        prefers a spot simply because the animal spent most of its time in that spot (recall the
        occupancy map in Section 1). The KDE says nothing about occupancy; the rate map corrects for it.
        """
    )
    return


@app.cell
def _(mo):
    kde_bw = mo.ui.slider(0.2, 2.0, value=0.6, step=0.1, label="bandwidth (× Scott default)",
                          debounce=True, full_width=True)
    return (kde_bw,)


@app.cell
def _(go, kde_bw, mo, neuron_ov, np, session_pick, sessions):
    from scipy.stats import gaussian_kde
    _d = sessions[session_pick.value]
    _ctr = _d["centroid"]
    _spk = _d["spikes"]
    _ni = min(neuron_ov.value, _spk.shape[1] - 1)
    _spiking = (_spk[:, _ni] > 0) & np.isfinite(_ctr).all(axis=1)
    _pts = _ctr[_spiking]
    _finite = _ctr[np.isfinite(_ctr).all(axis=1)]
    _x0, _x1 = _finite[:, 0].min(), _finite[:, 0].max()
    _y0, _y1 = _finite[:, 1].min(), _finite[:, 1].max()
    _xg = np.linspace(_x0, _x1, 80)
    _yg = np.linspace(_y0, _y1, 80)
    _title = f"KDE density · neuron {_ni} · bw ×{kde_bw.value:.1f} · {session_pick.value}"
    if len(_pts) >= 8 and np.ptp(_pts[:, 0]) > 0 and np.ptp(_pts[:, 1]) > 0:
        _k = gaussian_kde(_pts.T)
        _k.set_bandwidth(_k.factor * kde_bw.value)      # seaborn-style bw_adjust multiplier
        _XX, _YY = np.meshgrid(_xg, _yg)
        _Z = _k(np.vstack([_XX.ravel(), _YY.ravel()])).reshape(_XX.shape)
        _fig = go.Figure(go.Contour(x=_xg, y=_yg, z=_Z, colorscale="Inferno",
                                    contours=dict(coloring="fill"),
                                    colorbar=dict(title="density")))
        _fig.add_scatter(x=_pts[:, 0], y=_pts[:, 1], mode="markers",
                         marker=dict(color="rgba(255,255,255,0.35)", size=3),
                         name="spikes", showlegend=False)
    else:
        _fig = go.Figure()
        _fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                            text=f"neuron {_ni} has too few spikes for a KDE ({len(_pts)})")
        _title += " — too sparse"
    _fig.update_yaxes(scaleanchor="x", scaleratio=1, title="y (px)")
    _fig.update_xaxes(title="x (px)")
    _fig.update_layout(template="plotly_white", height=520, margin=dict(l=10, r=10, t=50, b=10),
                       title=_title)
    mo.vstack([kde_bw, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 4. The rate map — a spatial tuning curve

        **Why.** We want a picture of *firing rate as a function of location* that is not distorted by
        how the animal spent its time. The correct object is the **rate map**, and it is built from
        exactly the two ingredients we have already looked at separately: the spike map (Section 2) and
        the occupancy map (Section 1).

        **Definitions.**

        - **Spike map:** for each bin, the total number of spikes fired while the animal was in that
          bin.
        - **Occupancy:** for each bin, the number of frames the animal spent there (the map in
          Section 1). Coverage, not firing.
        - **Rate map:** the ratio, bin by bin,

          $$\text{rate}(x,y) = \frac{\text{spikes fired in bin }(x,y)}{\text{frames spent in bin }(x,y)}.$$

          Dividing by occupancy removes the "spent all its time here" confound: a bin the animal
          barely visited but fired in twice reads as a genuinely high rate, and a bin it sat in for a
          long time without firing reads low.
        - **Spatial tuning curve:** the general idea that a neuron's firing rate is a function of some
          variable. In Week 1 that variable was social geometry (for example, facing angle). A rate map
          is the *same idea* with the variable being the animal's (x, y) location. That is the concrete
          link back to the body-frame work: location is a variable the brain represents, and the rate
          map is how we read that representation out.

        **Method.** `nu.rate_map(x, y, spikes, bins=N)`

        - *Purpose:* build the occupancy-normalized rate map.
        - *Inputs:* the x and y position arrays `(T,)`, a neuron's spike-count column `(T,)`, and the
          grid resolution `N`.
        - *Output:* a dict with `rate`, `occupancy`, and `spike_map` on an `N × N` grid, plus the bin
          edges.

        The three panels below lay the ingredients side by side for the chosen neuron so the division
        is visible: **occupancy** (where the animal was) and **spike map** (where it fired) combine
        into the **rate map** (where it fires per unit time). A bright spot that appears in both the
        occupancy and spike maps but *vanishes* in the rate map was an occupancy artifact — the neuron
        was not tuned there, the animal just spent time there.
        """
    )
    return


@app.cell
def _(mo, neuron_ov, np, nu, session_pick, sessions):
    # Three-panel decomposition of the rate map for the chosen neuron: occupancy, spike map, rate.
    # Makes the "divide by occupancy" step visible instead of asserted.
    _d = sessions[session_pick.value]
    _ctr = _d["centroid"]
    _spk = _d["spikes"]
    _ni = min(neuron_ov.value, _spk.shape[1] - 1)
    _rm = nu.rate_map(_ctr[:, 0], _ctr[:, 1], _spk[:, _ni], bins=20)
    _xc = 0.5 * (_rm["xedges"][:-1] + _rm["xedges"][1:])
    _yc = 0.5 * (_rm["yedges"][:-1] + _rm["yedges"][1:])
    _f_occ = nu.heatmap_fig(_rm["occupancy"].T, x=_xc, y=_yc, title="occupancy (frames)",
                            xlabel="x", ylabel="y", colorscale="Cividis",
                            colorbar_title="", height=330)
    _f_spk = nu.heatmap_fig(_rm["spike_map"].T, x=_xc, y=_yc, title="spike map (counts)",
                            xlabel="x", ylabel="y", colorscale="Magma",
                            colorbar_title="", height=330)
    _f_rate = nu.heatmap_fig(_rm["rate"].T, x=_xc, y=_yc, title="rate = spikes / occupancy",
                             xlabel="x", ylabel="y", colorscale="Inferno",
                             colorbar_title="", height=330)
    for _f in (_f_occ, _f_spk, _f_rate):
        _f.update_yaxes(scaleanchor="x", scaleratio=1)
    mo.hstack([_f_occ, _f_spk, _f_rate], widths=[1, 1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With the decomposition in mind, the interactive rate map below is the object we will actually
        use. The `bins` slider sets the spatial resolution: coarse bins are stable but blurry; fine
        bins are sharp but noisy, because many bins receive only one or zero visits and their rate
        becomes wild. The title reports the map's **Skaggs spatial information**, defined next. Compare
        this rate map to the KDE from Section 3 for the same neuron — for a genuine place cell they
        agree, but for a neuron that merely fired where the animal lingered, the rate map is flatter
        than the KDE suggested.
        """
    )
    return


@app.cell
def _(mo):
    rm_bins = mo.ui.slider(8, 40, value=20, step=2, label="spatial bins (per axis)",
                           debounce=True, full_width=True)
    return (rm_bins,)


@app.cell
def _(mo, neuron_ov, nu, rm_bins, session_pick, sessions):
    _d = sessions[session_pick.value]
    _ctr = _d["centroid"]
    _spk = _d["spikes"]
    _ni = min(neuron_ov.value, _spk.shape[1] - 1)
    _rm = nu.rate_map(_ctr[:, 0], _ctr[:, 1], _spk[:, _ni], bins=int(rm_bins.value))
    _si = nu.spatial_information(_rm["rate"], _rm["occupancy"])
    # histogram2d indexes [x, y]; transpose so x is horizontal and y vertical in the image.
    _xc = 0.5 * (_rm["xedges"][:-1] + _rm["xedges"][1:])
    _yc = 0.5 * (_rm["yedges"][:-1] + _rm["yedges"][1:])
    _fig = nu.heatmap_fig(
        _rm["rate"].T, x=_xc, y=_yc,
        title=f"Rate map · neuron {_ni} · {int(rm_bins.value)} bins · SI = {_si:.3f} bits/spike · {session_pick.value}",
        xlabel="x (px)", ylabel="y (px)", colorscale="Inferno",
        colorbar_title="spikes/frame", height=520)
    _fig.update_yaxes(scaleanchor="x", scaleratio=1)
    mo.vstack([rm_bins, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 5. Which neurons are spatial? — Skaggs spatial information

        **Why.** A rate map is a picture; to compare all the neurons we want one number per neuron that
        summarizes how spatial it is. That number lets us rank the population and pick candidates.

        **Definition.** The **Skaggs spatial information** measures how much knowing the animal's
        location tells you about whether the neuron will fire. It is reported in **bits per spike**:

        $$\text{SI} = \sum_i p_i \, \frac{r_i}{\bar r}\, \log_2\!\frac{r_i}{\bar r},$$

        where $p_i$ is the fraction of time spent in bin $i$ (occupancy probability), $r_i$ is that
        bin's firing rate, and $\bar r$ is the occupancy-weighted mean rate. A neuron whose rate is the
        same everywhere gives SI $= 0$ bits/spike (location tells you nothing). A neuron with one sharp
        field gives a large positive value.

        **Method.** `nu.spatial_information(rate, occupancy)`

        - *Purpose:* collapse a rate map to a single tuning score.
        - *Inputs:* the `rate` and `occupancy` arrays from `nu.rate_map`.
        - *Output:* the SI as one float (bits/spike).

        Rather than a bar chart of SI, the plot below shows every neuron as a point in a
        **spatial-information vs spike-count** scatter — because those two axes together tell the real
        story. A trustworthy place cell sits to the **right** (many spikes) and **high** (large SI). A
        point that is high but far to the **left** (large SI, very few spikes) is the danger zone: a
        big number that a handful of lucky spikes can manufacture. Hover any point to read its neuron
        index, SI, and spike total. Section 6 turns this suspicion into a formal test.
        """
    )
    return


@app.cell
def _(go, np, nu, session_pick, sessions):
    # SI vs spike-count scatter (one interactive point per neuron). Individual points, not a bar
    # chart; the x-axis (spike count) is what exposes the sparsity artifact the metric is prone to.
    _d = sessions[session_pick.value]
    _ctr = _d["centroid"]
    _spk = _d["spikes"]
    _sis, _nsp = [], []
    for _i in range(_spk.shape[1]):
        _rm = nu.rate_map(_ctr[:, 0], _ctr[:, 1], _spk[:, _i], bins=20)
        _sis.append(nu.spatial_information(_rm["rate"], _rm["occupancy"]))
        _nsp.append(int((_spk[:, _i] > 0).sum()))
    _sis = np.array(_sis); _nsp = np.array(_nsp)
    _fig = go.Figure(go.Scatter(
        x=_nsp, y=_sis, mode="markers+text",
        text=[f"n{_i}" for _i in range(len(_sis))], textposition="top center",
        textfont=dict(size=10),
        marker=dict(size=14, color=_sis, colorscale="Inferno",
                    colorbar=dict(title="SI"), line=dict(width=0.5, color="#333")),
        hovertext=[f"neuron {_i}: SI={_sis[_i]:.3f} bits/spike, {_nsp[_i]:,} spike-frames"
                   for _i in range(len(_sis))],
        hovertemplate="%{hovertext}<extra></extra>"))
    _fig.update_xaxes(title="spike-frames (more = more reliable)", type="log")
    _fig.update_yaxes(title="Skaggs spatial information (bits/spike)")
    _fig.update_layout(template="plotly_white", height=460, margin=dict(l=10, r=10, t=50, b=10),
                       title=(f"Spatial information vs spike count · {session_pick.value} · 20 bins · "
                              "top-left points are suspect"))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In the default session the scatter tells a cautionary tale immediately. **Neuron 10** floats
        near the top with the largest raw SI (about 2.14 bits/spike) — but it sits at the far left,
        having fired in only about 80 frames. **Neuron 5**, by contrast, is high *and* well to the
        right (about 1.26 bits/spike from more than a thousand spike-frames). If we crowned the tallest
        raw score we would name neuron 10 the best place cell. That would be a mistake, and the next
        section shows exactly how to catch it.

        ---
        ## 6. The test — does the tuning beat a matched control?

        **Why a raw number is not enough.** A neuron that fires only 80 times can post a very high SI
        purely because a few spikes happen to land in a few rarely visited bins. That is sparsity, not
        spatial tuning. To separate a real field from a sparsity artifact we compare the neuron's SI
        against a control built from **its own spikes**.

        **Definitions.**

        - **Shuffle null:** a set of control values produced by breaking the link between spikes and
          position while keeping everything else identical. Here we **circularly shift** the spike
          train — slide it forward in time by a random amount and wrap the end around to the start
          (`np.roll`). This keeps the exact spike count and the train's temporal structure but pairs
          the spikes with the *wrong* positions, so any spatial information it produces is by chance.
        - **Chance band:** the 95th percentile of the shuffled SI values. A real cell should have an
          observed SI *above* this band.

        The demonstration below runs the shuffle for both neurons discussed above, side by side. Watch
        what happens: neuron 5's observed SI (solid line) sits far to the right of its own shuffle
        cloud — real tuning. Neuron 10's observed SI sits *inside*, even below, its shuffle cloud —
        its high raw number is exactly what shuffled, position-scrambled spikes produce, so it is an
        artifact. This is the picture the exercise then asks you to reproduce for one neuron.
        """
    )
    return


@app.cell
def _(go, np, nu, sessions):
    # DEMONSTRATION (not the exercise): neuron 5 (real place cell) vs neuron 10 (sparsity artifact).
    # Both observed SI + both shuffle nulls overlaid, so the test's logic is visible before the
    # student does it. Canonical settings (bins=20, seed=0, 50 shuffles) match the exercise.
    _d = sessions["20160609T194655.mat"]
    _ctr = _d["centroid"]
    _spk = _d["spikes"]

    def _si_of(_col):
        _rm = nu.rate_map(_ctr[:, 0], _ctr[:, 1], _col, bins=20)
        return nu.spatial_information(_rm["rate"], _rm["occupancy"])

    def _null_of(_col):
        _rng = np.random.default_rng(0)
        return np.array([_si_of(np.roll(_col, int(_rng.integers(1000, len(_col) - 1000))))
                         for _ in range(50)])

    _c5, _c10 = _spk[:, 5], _spk[:, 10]
    _obs5, _obs10 = _si_of(_c5), _si_of(_c10)
    _n5, _n10 = _null_of(_c5), _null_of(_c10)
    _fig = go.Figure()
    _fig.add_histogram(x=_n5, nbinsx=18, marker_color="rgba(76,120,168,0.55)",
                       name="neuron 5 shuffles")
    _fig.add_histogram(x=_n10, nbinsx=18, marker_color="rgba(228,87,86,0.45)",
                       name="neuron 10 shuffles")
    _fig.add_vline(x=float(_obs5), line=dict(color="#4c78a8", width=3),
                   annotation_text="n5 observed (real)", annotation_position="top")
    _fig.add_vline(x=float(_obs10), line=dict(color="#e45756", width=3, dash="dot"),
                   annotation_text="n10 observed (artifact)", annotation_position="top right")
    _fig.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=60, b=10),
                       barmode="overlay", title=(
                           "Shuffle nulls · neuron 5 (1,115 spikes, real) vs neuron 10 (80 spikes, "
                           "artifact) · 20160609T194655"),
                       xaxis_title="spatial information (bits/spike)", yaxis_title="shuffles")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercise — is there a genuine place cell here?

        **Python skill practised:** *writing and calling a small function*, then *calling a library
        routine* (`np.percentile`) to summarize a distribution. You are given a helper `_si_of` that
        maps one spike column to its spatial information, and a loop that builds a 50-sample shuffle
        null. You fill in the two lines that (1) score the real spikes and (2) read off the chance
        band. This is the same "define a function, apply it, summarize with numpy" pattern the neural
        arm keeps reusing.

        **Tools.**

        - `sessions["20160609T194655.mat"]` → dict with `centroid (T, 2)` and `spikes (T, n)`.
        - `nu.rate_map(x, y, spikes, bins=20)` → dict with `rate`, `occupancy`.
        - `nu.spatial_information(rate, occupancy)` → SI in bits/spike.
        - `np.roll(spike_column, shift)` → circular shift; `np.percentile(values, q)` → the qth
          percentile.

        **Your task.** For **neuron 5** of session `20160609T194655.mat`, compute the observed SI and a
        50-sample shuffle null, then take the null's 95th percentile as the chance band. The cell below
        is written for you except for **two blanks** marked in the comments. Fill them in:

        1. `si_obs` — pass the real (unshuffled) spike column `_col` into `_si_of`.
        2. `si_band` — take the `95`th percentile of the shuffled values.

        **What you should see.** The plot underneath draws the 50 shuffled SI values as a grey
        histogram, the chance band as a dashed line, and your observed SI as a solid red line. If you
        filled the blanks correctly the grey histogram sits low (roughly 0.3–0.7 bits/spike), the
        dashed band lands near 0.65–0.68, and the red observed line is far to the right near 1.26 —
        clearly past the band. Then run the self-check below it.
        """
    )
    return


@app.cell
def _(np, nu, sessions):
    # ------------------------------------------------------------------ YOUR CODE (edit this cell)
    _d = sessions["20160609T194655.mat"]
    _ctr = _d["centroid"]
    _spk = _d["spikes"]
    _col = _spk[:, 5]                                   # candidate: neuron 5

    def _si_of(_s):
        # PURPOSE: spatial information of one spike column. INPUT: a (T,) spike-count array.
        # OUTPUT: its Skaggs SI in bits/spike (a single float).
        _rm = nu.rate_map(_ctr[:, 0], _ctr[:, 1], _s, bins=20)
        return nu.spatial_information(_rm["rate"], _rm["occupancy"])

    # TODO 1 (blank #1): compute the observed SI of neuron 5's REAL spikes.
    #   WHAT TO CHANGE: replace `____` with `_col` — the unshuffled spike column defined three lines up.
    #   WHY IT MATTERS: this is the number we are testing. If you pass anything other than the real,
    #   in-order spikes, you are no longer measuring neuron 5's true spatial tuning, and the whole
    #   comparison against the shuffle null becomes meaningless.
    si_obs = float(_si_of(_col))          # <-- replace ____ with _col  (shown filled with the answer)

    _rng = np.random.default_rng(0)
    _null = np.array([
        # each entry: circularly shift the SAME spikes by a random amount (breaking the spike–position
        # link) and re-score. 50 such shuffles form the chance distribution.
        _si_of(np.roll(_col, int(_rng.integers(1000, len(_col) - 1000))))
        for _ in range(50)
    ])

    # TODO 2 (blank #2): the chance band is the 95th percentile of the 50 shuffled SI values.
    #   WHAT TO CHANGE: replace `____` with `95` — the percentile that marks the top 5% of shuffles.
    #   WHY IT MATTERS: 95 is the conventional one-sided significance threshold. A real cell should beat
    #   the value that only 5% of position-scrambled shuffles exceed; a smaller number (say 50) would
    #   pass almost any neuron, and a larger one (say 99.9) would reject even good cells.
    si_band = float(np.percentile(_null, 95))   # <-- replace ____ with 95  (shown filled with answer)
    # ---------------------------------------------------------------------------------------------
    return si_band, si_obs


@app.cell
def _(go, np, nu, sessions, si_band, si_obs):
    # Result plot for the exercise: the shuffle null (gray histogram) with your observed SI (red)
    # and chance band (dashed). The null is recomputed here with the canonical settings so the
    # picture is stable; the two vertical lines show the values YOU produced above. This plot always
    # uses the exercise session (20160609T194655), not the dropdown.
    _d = sessions["20160609T194655.mat"]
    _ctr = _d["centroid"]
    _col = _d["spikes"][:, 5]

    def _si_of(_s):
        _rm = nu.rate_map(_ctr[:, 0], _ctr[:, 1], _s, bins=20)
        return nu.spatial_information(_rm["rate"], _rm["occupancy"])

    _rng = np.random.default_rng(0)
    _null = np.array([
        _si_of(np.roll(_col, int(_rng.integers(1000, len(_col) - 1000))))
        for _ in range(50)
    ])
    _fig = go.Figure()
    _fig.add_histogram(x=_null, nbinsx=18, marker_color="#c9c9c9", name="shuffled SI")
    _fig.add_vline(x=float(si_band), line=dict(color="#4c78a8", width=2, dash="dash"),
                   annotation_text="95% chance band", annotation_position="top")
    _fig.add_vline(x=float(si_obs), line=dict(color="#e45756", width=3),
                   annotation_text="observed", annotation_position="top right")
    _fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=60, b=10),
                       title="Neuron 5 spatial information vs shuffle null · 20160609T194655",
                       xaxis_title="spatial information (bits/spike)", yaxis_title="shuffles")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show solution": mo.md(
            r"""
            ```python
            d = sessions["20160609T194655.mat"]
            ctr, spk = d["centroid"], d["spikes"]
            col = spk[:, 5]

            def si_of(s):
                rm = nu.rate_map(ctr[:, 0], ctr[:, 1], s, bins=20)
                return nu.spatial_information(rm["rate"], rm["occupancy"])

            si_obs = si_of(col)                                    # about 1.260 bits/spike
            rng = np.random.default_rng(0)
            null = np.array([si_of(np.roll(col, int(rng.integers(1000, len(col) - 1000))))
                             for _ in range(50)])
            si_band = np.percentile(null, 95)                      # about 0.65 to 0.68
            place_like = si_obs > si_band                          # True
            ```

            **Result.** Neuron 5 has `si_obs` about 1.26 bits/spike, and its shuffle band sits around
            0.65 to 0.68. The observed value is well above the band, so neuron 5 is a genuine
            place-like cell: a concrete example of the allocentric map this notebook set out to
            measure.

            **Why the shuffle matters.** If you had instead trusted the highest raw SI you would have
            been misled. In this session neuron 10 has the largest raw SI (about 2.14) but fires only
            about 80 times, and its own shuffle band is about 2.45 — so neuron 10 fails the test (its
            observed value falls *inside* its own shuffle cloud, exactly as the Section 6 demonstration
            showed). High SI from very few spikes is a sparsity artifact, not a place field. The
            reliable conclusion is that place-like cells exist here (neurons 4, 5, 6, and others), but
            SI alone, without a spike-matched control, will produce false positives.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo, si_band, si_obs):
    # Self-check. Part A: the observed SI is pinned from real data (neuron 5, 20 bins) = 1.2601.
    # Part B: the correct conclusion is that this cell clears a spike-matched shuffle band that lands
    # well below it (pinned band ~0.65-0.68 across seeds). We grade the shuffle-corrected conclusion
    # (a genuine place cell), not a raw-SI leaderboard (which would wrongly crown the sparse neuron
    # 10).
    _a = abs(float(si_obs) - 1.2601) < 0.03
    _b = (0.45 < float(si_band) < 0.95) and (float(si_obs) > float(si_band) + 0.3)
    _ok = _a and _b
    _c = "#e8f5e9" if _ok else "#ffebee"
    _bd = "#2e7d32" if _ok else "#c62828"
    _m1 = (f"observed SI = {si_obs:.3f} bits/spike (neuron 5)" if _a
           else f"si_obs = {si_obs:.3f} — expected about 1.260 for neuron 5 at 20 bins")
    _m2 = (f"chance band = {si_band:.3f}; the observed SI is well above it, so this is a genuine "
           "place-like cell"
           if _b else
           f"chance band = {si_band:.3f} looks off — did you circular-shift the SAME spike column "
           "and take the 95th percentile of 50 shuffles?")
    _head = "Pass — this is a real place cell" if _ok else "Not yet — fix the flagged part"
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_bd};padding:12px 16px;border-radius:6px">
        <b style="color:{_bd}">{_head}</b><br>
        {_m1}<br>{_m2}<br>
        <span style="font-size:0.9em;color:#555">Graded on the shuffle-corrected conclusion, not a
        raw-SI ranking. Tolerance: |si_obs − 1.260| &lt; 0.03, band ∈ (0.45, 0.95), and si_obs
        exceeds the band by &gt; 0.3.</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 7. Grid cells — and why we do not see them here

        The notebook is titled *place and grid cells*, and so far we have only measured place cells. It
        is worth understanding what a grid cell would look like, both because it completes the picture
        of the allocentric map and because the tool used to find them — the **spatial
        autocorrelogram** — is a good example of how a periodic pattern is detected.

        **Definition.** A **grid cell** fires not at one location but at many, arranged in a repeating
        triangular (hexagonal) lattice that tiles the arena. A single grid cell's rate map therefore
        looks like a regular array of bumps, not one blob.

        **How you detect the lattice.** You cannot always see the regularity by eye in a noisy rate
        map, so you compute its **spatial autocorrelogram**: slide the rate map over a copy of itself
        and, at each offset, measure how well it matches. For a place cell (one bump) the
        autocorrelogram has a single central peak and nothing else. For a grid cell it has a central
        peak surrounded by a ring of **six** satellite peaks — the fingerprint of the hexagonal
        lattice.

        **Method.** We mean-subtract the (lightly smoothed) rate map and cross-correlate it with itself
        (`scipy.signal.correlate2d`, `mode="full"`), then normalize so the central peak is 1. The
        result is shown below for the chosen neuron. In this dataset you will find single central peaks
        — place fields — but no six-fold rings. That is expected: these are hippocampal-style recordings
        with a gaze-proxy position signal, not the medial entorhinal recordings where grid cells live.
        The absence is itself informative, and knowing the tool means you would recognize a grid cell
        immediately if one appeared.
        """
    )
    return


@app.cell
def _(mo, neuron_ov, np, nu, session_pick, sessions):
    # Spatial autocorrelogram of a neuron's rate map: the standard grid-cell diagnostic. A place cell
    # -> single central peak; a grid cell -> central peak + a ring of six satellites. We show it for
    # the chosen neuron so students see the place-field signature (and the ABSENCE of grid structure).
    from scipy.signal import correlate2d
    _d = sessions[session_pick.value]
    _ctr = _d["centroid"]
    _spk = _d["spikes"]
    _ni = min(neuron_ov.value, _spk.shape[1] - 1)
    _rm = nu.rate_map(_ctr[:, 0], _ctr[:, 1], _spk[:, _ni], bins=20, smooth_sigma=1.0)
    _r = _rm["rate"]
    _r0 = _r - _r.mean()
    _ac = correlate2d(_r0, _r0, mode="full")
    _peak = np.abs(_ac).max()
    _ac = _ac / (_peak if _peak > 0 else 1.0)
    _lag = np.arange(-_r.shape[0] + 1, _r.shape[0])   # bin lags, centered at 0
    _fig = nu.heatmap_fig(
        _ac.T, x=_lag, y=_lag,
        title=(f"Spatial autocorrelogram · neuron {_ni} · {session_pick.value} · "
               "one central peak = place field, not a grid"),
        xlabel="x lag (bins)", ylabel="y lag (bins)", colorscale="RdBu",
        zmid=0.0, colorbar_title="corr", height=500)
    _fig.update_yaxes(scaleanchor="x", scaleratio=1)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "References, credit, and the limits of this dataset": mo.md(
            r"""
            **Place cells.** O'Keefe & Dostrovsky, 1971, *Brain Research* 34:171 — a hippocampal neuron
            that fires only when the animal occupies a particular location. This was the founding
            observation of the cognitive-map framework (O'Keefe & Nadel, 1978).

            **Grid cells.** Hafting, Fyhn, Molden, Moser & Moser, 2005, *Nature* 436:801 — medial
            entorhinal neurons that fire on a repeating triangular lattice tiling the environment.
            Place, grid, and head-direction cells together form the allocentric map introduced at the
            top of this notebook (2014 Nobel Prize: O'Keefe; May-Britt and Edvard Moser).

            **Credit.** This analysis is adapted from the **NEU 457** (Princeton) problem set by
            **Talmo Pereira, Andrew Leifer, and David Tank.** The data and the KDE / rate-map framing
            are theirs. We rebuilt it as an interactive notebook and added the occupancy
            normalization, the shuffle null, and the spatial autocorrelogram.

            **Limits of this dataset.** The "position" used here is the centroid of two tracked eye
            positions — a gaze / eye-in-head proxy, not the clean body-on-arena position that the
            classic place-cell rigs record. The fields are therefore weaker and noisier than a textbook
            place field, and some apparent tuning may be gaze- or movement-coupled rather than pure
            allocentric location coding. This is exactly why the exercise grades the shuffle-corrected
            conclusion rather than the prettiest map: with a proxy position signal, an honest control is
            what separates a real place cell from a sparsity artifact.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Where this leaves us

        **The question we asked:** do individual neurons encode where the animal is? **The answer:**
        yes. Building the occupancy-normalized **rate map** — a two-dimensional spatial tuning curve,
        the same firing-rate-as-a-function-of-a-variable idea we used for social geometry in Week 1 —
        we found neurons whose firing is a sharp function of the animal's (x, y) location, and at least
        one (neuron 5) survives a spike-matched shuffle control. Individual neurons carry the brain's
        allocentric map of the arena.

        We also met a discipline that recurs throughout the neural work: **a large number is not a
        result until it beats a matched control.** Sparse spikes inflate spatial information, and a
        proxy position variable adds noise, so the shuffle null comes first and belief comes second —
        which is why we did not crown the neuron with the biggest raw score.

        **The next question.** We have shown the brain represents *space* — a fixed external variable.
        But this is a course about *social* behavior, where the most important variable is not a corner
        of the arena but *another animal*. Does the brain represent the presence and actions of a
        conspecific the way it represents location, and can we read that representation out of the
        neural activity? The next notebook leaves the single-animal spatial rig for a social dataset —
        behavior bouts aligned with calcium imaging — and asks whether, and how strongly, individual
        neurons track social interaction.
        """
    )
    return


if __name__ == "__main__":
    app.run()
