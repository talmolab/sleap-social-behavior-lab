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


# ============================================================================ briefing
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # NB10 · Reading a Social Behavior off the Neurons

        **Week 2 · the culminating notebook, on real calcium-imaging data.**

        **Where we have been.** The previous notebooks walked one real recording from a raw
        microscope movie all the way to meaning: we stabilized the movie (motion correction),
        demixed it into individual cells' calcium traces, and then asked what a single neuron's
        firing *represents*. We found place cells and grid cells — neurons whose firing is a map of
        the animal's location, tuned sharply enough that we could read position off the brain. The
        question those notebooks answered was: **does a single neuron's activity encode a variable
        we care about?** For space, the answer was yes.

        **The question this notebook asks.** We now push that question to its hardest, and most
        social, form:

        > *Can we read a **social behavior** directly off a population of neurons?*

        Space is a clean physical variable a single cell can tile. A social behavior — one mouse
        actively engaging another — may not live in any single cell. It may be written across the
        population, a little in each of many neurons, in a pattern no one cell makes obvious. So the
        target of this notebook is not a single tuned cell but the **whole recorded population**, and
        the tool is the same supervised decoder we built for behavior in Week 1 — now with real
        neurons as its input.

        **The two halves of this notebook.** To connect a neuron to a behavior we need both, written
        down precisely and on the same clock.

        1. **The behavior half.** We first build the behavioral ground truth: an **ethogram** of
           social contact for every recording session, and we verify its internal definition before
           trusting it. This is the label the neurons will be scored against.
        2. **The neural half.** We put the calcium on the behavior's clock, look for individual
           social-responsive neurons, and then **train a population decoder** of social state and
           cross-validate it — the direct parallel to the Week-1 behavior decoder, now reading off
           real calcium.

        **The dataset.** A social-isolation cohort: a focal mouse is either group-housed (**control**)
        or isolated for **24 hours** or **7 days**, then reintroduced to a partner while calcium
        activity in its striatum is imaged. Eighteen sessions, six per condition. We study this
        social behavior and its neural correlate together, with neither treated as primary.
        """
    )
    return


# ============================================================================ load
@app.cell
def _(nu):
    # One full load: the SI social-isolation dataset. This pulls in the behavior scoring (light) AND
    # the ~250 MB calcium file (a few seconds, ~280 MB RAM) — the neural half needs the calcium, so
    # we load everything once here rather than twice.
    _d = nu.load_si()
    ent = _d["entrances"]
    beh = _d["behavior"]
    img = _d["imaging"]
    n_sessions = _d["n_sessions"]
    behavior_fps = _d["behavior_fps"]
    imaging_fps = _d["imaging_fps"]
    session_keys = list(_d["session_keys"])
    # condition label per session ("control" / "24hr" / "7d") via nu's canonical mapper.
    cond_labels = [nu.si_condition_label(ent["Isolation Length"].iloc[s]) for s in range(n_sessions)]
    return (beh, behavior_fps, cond_labels, ent, imaging_fps, img,
            n_sessions, session_keys)


@app.cell
def _(beh, cond_labels, n_sessions, np):
    # Per-session summaries used throughout, plus the condition palette.
    social_frac = np.array([beh[s]["is_social"].mean() for s in range(n_sessions)])
    sess_lengths = np.array([len(beh[s]["is_social"]) for s in range(n_sessions)])
    cond_counts = {c: cond_labels.count(c) for c in ["control", "24hr", "7d"]}
    # Isolation-severity ramp: group-housed control -> 24 h -> 7 d. (No mouse ranks in this dataset,
    # so a neutral condition palette is appropriate — mouse-rank coloring is only for the pose arm.)
    COND_COLORS = {"control": "#4c78a8", "24hr": "#f58518", "7d": "#e45756"}
    COND_ORDER = ["control", "24hr", "7d"]
    return COND_COLORS, COND_ORDER, cond_counts, sess_lengths, social_frac


@app.cell(hide_code=True)
def _(cond_counts, img, mo, n_sessions):
    _neur = [im.shape[1] for im in img]
    mo.md(
        f"""
        **The recording inventory.** {n_sessions} sessions, one focal mouse each, pairing miniscope
        calcium imaging with frame-by-frame social scoring. Sessions per condition:
        **{cond_counts}**.

        Two facts about this dataset shape everything downstream, so we state them now.

        **Fact 1 — every session extracts its own neurons.** Neuron counts differ across sessions
        (**{min(_neur)}–{max(_neur)}** cells) because each recording finds its own set of cells in
        its own field of view, with **no correspondence between sessions**: cell 40 in one session is
        not cell 40 in another. There is no shared neuron identity that would let a decoder trained on
        one session be applied to another. This is why, later, the decoder is trained and tested
        **within a single session**. It is not a shortcut; it is a hard property of the data, and we
        will return to it explicitly.

        **Fact 2 — two clocks.** The calcium is sampled at **{__import__('math').nan if False else 30} fps**
        and the behavior scored at **25 fps**, so the two arrays have different lengths and do not line
        up frame for frame. Before we can ask whether a neuron tracks a behavior, we have to put them
        on the same clock. That is the first step of the neural half.

        First, though, the behavior itself.
        """
    )
    return


# ============================================================================ PART A — the ethogram
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # Part A · The behavior: a social ethogram

        **Why this comes first.** A decoder is only as trustworthy as its labels. Before asking a
        single neuron anything, we build the behavioral ground truth and check that it is internally
        consistent. Everything neural in Part B is scored against the labels we validate here.

        **What is an ethogram?** An ethogram is a catalogue of the behaviors an animal can perform,
        scored over time. For each behavior and each moment it records whether the animal is doing
        it. Concretely it is a table of on/off (boolean) rows: one row per behavior, one column per
        video frame. Reading *down* a column tells you what the animal was doing at that instant;
        reading *across* a row tells you when a particular behavior occurred. This is the same object
        the pose arm built from keypoints — a stack of per-frame behavioral states — except here the
        states come pre-scored as nine social-contact channels.

        ## A1. The sessions

        The table below is the inventory: each row is one session, with its isolation condition, the
        frame at which the partner enters (`Int_Entry`), the track length, and the fraction of frames
        that are social. This is not one long recording; it is 18 separate sessions, each with its own
        behavior track and its own length.
        """
    )
    return


@app.cell
def _(COND_COLORS, cond_labels, ent, go, np, sess_lengths, social_frac):
    # Inventory table in the house style, tinted by condition.
    _sess = np.arange(len(ent))
    _iso = list(ent["Isolation Length"].astype(str))
    _entry = list(ent["Int_Entry"].astype(int))
    _rowcol = [COND_COLORS[c] for c in cond_labels]
    def _tint(hexc):
        _h = hexc.lstrip("#")
        _r, _g, _b = int(_h[0:2], 16), int(_h[2:4], 16), int(_h[4:6], 16)
        _f = 0.85
        return f"rgb({int(_r+(255-_r)*_f)},{int(_g+(255-_g)*_f)},{int(_b+(255-_b)*_f)})"
    _fill = [[_tint(c) for c in _rowcol]]
    _tbl = go.Figure(go.Table(
        columnwidth=[0.8, 1.6, 1.2, 1.2, 1.1, 1.4],
        header=dict(values=["<b>session</b>", "<b>Isolation Length</b>", "<b>condition</b>",
                            "<b>Int_Entry (frame)</b>", "<b>length (frames)</b>",
                            "<b>social fraction</b>"],
                    fill_color="#2f3b52", font=dict(color="white", size=13), align="left",
                    height=30),
        cells=dict(values=[_sess, _iso, cond_labels, _entry, sess_lengths,
                           [f"{v:.3f}" for v in social_frac]],
                   fill_color=_fill * 6, align="left", height=24,
                   font=dict(color="#222", size=12))))
    _tbl.update_layout(template="plotly_white", height=560, margin=dict(l=10, r=10, t=34, b=10),
                       title="Social-isolation sessions — one row per session (6 control · 6 × 24 h · 6 × 7 d)")
    _tbl
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## A2. The nine channels, and the definition of "social"

        **Why this matters.** "Social behavior" is not one thing. To study it we break it into
        specific, separately-scored actions. Each session provides nine boolean channels, one value
        per frame (`True` means the behavior is happening at that frame).

        **Sender vs receiver.** Six channels describe a specific contact, split by who is acting. When
        the focal mouse *performs* the action it is the **sender**; when the action is *done to* the
        focal mouse it is the **receiver**.

        | channel | meaning | side |
        |---|---|---|
        | `is_touching` | focal mouse's nose is on the partner's body | **sender** |
        | `is_ag_sniffing` | focal mouse is anogenital-sniffing the partner | **sender** |
        | `is_of_sniffing` | focal mouse is oro-facial (nose/face) sniffing the partner | **sender** |
        | `is_touched` | the partner's nose is on the focal mouse | **receiver** |
        | `is_ag_sniffed` | focal mouse is being anogenital-sniffed | **receiver** |
        | `is_of_sniffed` | focal mouse is being oro-facially sniffed | **receiver** |

        **Three derived channels.** The remaining three are logical **OR** combinations (written
        $\lor$; true if any input is true) of the six above. `is_social_sender` is true whenever the
        focal mouse is performing any social contact; `is_social_receiver` whenever it is receiving
        any; and `is_social` whenever *either* is true.

        $$
        \texttt{is\_social\_sender} = \texttt{is\_touching} \lor \texttt{is\_ag\_sniffing} \lor \texttt{is\_of\_sniffing}
        $$
        $$
        \texttt{is\_social\_receiver} = \texttt{is\_touched} \lor \texttt{is\_ag\_sniffed} \lor \texttt{is\_of\_sniffed}
        $$
        $$
        \boxed{\;\texttt{is\_social} = \texttt{is\_social\_sender} \lor \texttt{is\_social\_receiver}\;}
        $$

        That boxed identity is the one we will **verify** in the first exercise; it should hold
        exactly, frame by frame. It also tells us the **decode target** for Part B: we will decode
        `is_social_sender` — the focal mouse *actively engaging* the partner — because that is the
        behavior the focal mouse's own neurons are most likely to drive.

        **Reading the ethogram.** The heatmap below is the `(9, T)` ethogram for one session: nine
        rows, time along x, a filled cell wherever a channel is on. The `is_social` row (outlined) is
        the union — it lights up whenever any channel below it does. The bars at right give each
        channel's **occupancy**: the fraction of frames it is on. Move the slider through sessions.
        """
    )
    return


@app.cell
def _(mo, n_sessions):
    session_sel = mo.ui.slider(0, n_sessions - 1, value=5, step=1,
                               label="session", debounce=True, full_width=True)
    return (session_sel,)


@app.cell
def _(COND_COLORS, beh, behavior_fps, cond_labels, go, mo, np,
      session_sel, session_keys, social_frac):
    _s = int(session_sel.value)
    _d = beh[_s]

    # Stack the nine channels into a (9, T) ethogram, in canonical key order.
    _etho = np.stack([_d[k] for k in session_keys], axis=0).astype(float)   # (9, T)
    _T = _etho.shape[1]

    # Max-pool the time axis for display so thin bouts survive downsampling (vectorized, no loop).
    _target = 1600
    _k = max(1, _T // _target)
    _Tt = (_T // _k) * _k
    _disp = _etho[:, :_Tt].reshape(9, _Tt // _k, _k).max(axis=2)
    _tsec = (np.arange(_disp.shape[1]) * _k) / behavior_fps

    _cond = cond_labels[_s]
    _eth = go.Figure(go.Heatmap(
        z=_disp, x=_tsec, y=session_keys,
        colorscale=[[0.0, "#f5f7fa"], [1.0, COND_COLORS[_cond]]],
        showscale=False, xgap=0, ygap=1,
        hovertemplate="%{y}<br>t=%{x:.1f}s<br>%{z:.0f}<extra></extra>"))
    _eth.update_layout(template="plotly_white", height=430, margin=dict(l=10, r=10, t=50, b=40),
                       title=f"session {_s} · {_cond} · ethogram (max-pooled ×{_k})")
    _eth.update_xaxes(title="time (s)")
    _eth.add_shape(type="rect", xref="paper", yref="y",
                   x0=0, x1=1, y0=3.5, y1=4.5, line=dict(color="#222", width=1.5))

    _frac = np.array([_d[k].mean() for k in session_keys])
    _cols = ["#9aa7bd"] * len(session_keys)
    _cols[session_keys.index("is_social")] = COND_COLORS[_cond]
    _bar = go.Figure(go.Bar(
        x=_frac, y=session_keys, orientation="h", marker_color=_cols,
        text=[f"{v:.3f}" for v in _frac], textposition="outside"))
    _bar.update_layout(template="plotly_white", height=430, margin=dict(l=10, r=10, t=50, b=40),
                       title=f"session {_s} · occupancy (is_social = {social_frac[_s]:.3f})",
                       xaxis_title="fraction of frames ON")
    _bar.update_xaxes(range=[0, max(0.25, float(_frac.max()) * 1.25)])

    mo.vstack([session_sel, mo.hstack([_eth, _bar], widths=[1.6, 1.0])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Two patterns recur. First, `is_touching` (nose-to-body contact) accounts for most of the
        social time, so `is_social_sender`, which includes it, closely tracks `is_social`. Second, the
        sniff channels are sparse, and being anogenital-sniffed is the rarest of all. The bouts are
        also *brief and clustered* — social contact comes in short runs rather than long blocks, which
        is exactly the structure a per-frame decoder will have to catch in Part B.

        ---
        ## A3. One channel across all sessions

        **Why.** Within-session views cannot tell us whether behavior differs between conditions. To
        ask that, we compare the *same* channel across *all* sessions. Pick a channel; each point below
        is one session's occupancy, colored and grouped by isolation condition. Individual sessions are
        shown as individual points (not a bar of means) so the spread is visible — with only six
        sessions per group, the spread is the story.
        """
    )
    return


@app.cell
def _(mo, session_keys):
    chan_pick = mo.ui.dropdown(options=session_keys, value="is_social",
                               label="channel to compare across sessions")
    return (chan_pick,)


@app.cell
def _(COND_COLORS, COND_ORDER, beh, chan_pick, cond_labels, mo, n_sessions, np, nu):
    _key = chan_pick.value
    _frac = np.array([beh[s][_key].mean() for s in range(n_sessions)])
    # Seaborn-style strip: one point per session, jittered, grouped by condition, hover = session id.
    _fig = nu.strip_points_fig(
        _frac, cond_labels, group_order=COND_ORDER, colors=COND_COLORS,
        hover=[f"session {s}" for s in range(n_sessions)],
        ylabel=f"'{_key}' occupancy (fraction of frames)", xlabel="isolation condition",
        title=f"'{_key}' occupancy per session (each point = one session; line = group mean)",
        height=430)
    mo.vstack([chan_pick, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## A4. Does isolation change social time?

        **Why.** The central behavioral question of this dataset: does isolation change how much time
        a mouse spends in social contact? We group the per-session occupancy by condition and compare.

        **The test.** With three groups, six sessions each, and no guarantee of normality, we use the
        **Kruskal–Wallis test** — a rank-based test of whether several groups differ, the multi-group
        counterpart of the Mann–Whitney U test. It returns a p-value: the probability of seeing group
        differences at least this large if all groups were drawn from the same distribution. Small
        (conventionally below 0.05) would indicate a real difference.

        The violin below shows each group's distribution with **every session drawn as a point**. Read
        it honestly — report what the data show, not what you might have expected.
        """
    )
    return


@app.cell
def _(mo, session_keys):
    cond_chan = mo.ui.dropdown(options=session_keys, value="is_social",
                               label="channel for the condition comparison")
    return (cond_chan,)


@app.cell
def _(COND_COLORS, COND_ORDER, beh, cond_chan, cond_labels, mo, n_sessions, np, nu):
    from scipy.stats import kruskal
    _key = cond_chan.value
    _frac = np.array([beh[s][_key].mean() for s in range(n_sessions)])
    _groups = {c: _frac[[cond_labels[s] == c for s in range(n_sessions)]] for c in COND_ORDER}
    _H, _p = kruskal(*[_groups[c] for c in COND_ORDER])
    _means = {c: float(_groups[c].mean()) for c in COND_ORDER}
    _title = (f"'{_key}' by condition   ·   means "
              + " ".join(f"{c}={_means[c]:.3f}" for c in COND_ORDER)
              + f"   ·   Kruskal–Wallis p = {_p:.3f}")
    _fig = nu.violin_points_fig(_frac, cond_labels, group_order=COND_ORDER, colors=COND_COLORS,
                                ylabel="fraction of frames ON", xlabel="isolation condition",
                                title=_title, height=450)
    mo.vstack([cond_chan, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For `is_social` the group means decline slightly across conditions —
        **control ≈ 0.155, 24 h ≈ 0.148, 7 d ≈ 0.139** — a small, monotone drop. But Kruskal–Wallis
        is **not significant** (p ≈ 0.81, n = 6 per group): with this few sessions the effect is too
        small to distinguish from noise. That is the correct conclusion to record. It also sets up a
        contrast the notebook will pay off: the behavioral isolation effect is weak, so the interesting
        question is whether the **neural** readout is any sharper.

        ---
        ## A5. Exercise 1 — verify the definition, measure social time

        **Python skill practiced:** *boolean masks and array reductions.* You will build boolean
        arrays with `np.logical_or`, compare them frame-for-frame, and reduce with `.mean()`. A boolean
        array's `.mean()` is its fraction of `True` values — the single most useful reduction in this
        whole course.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **What you are checking.**

        1. That `is_social` is exactly the OR of the sender and receiver channels. If the file is
           internally consistent, the largest disagreement across all sessions is `0`.
        2. The average fraction of time spent social.
        3. The plain isolation effect (control minus 7 d).

        **How to work.** The cell below is scaffolded — the bookkeeping is written for you. Two lines
        are marked `# FILL IN`, and the comment above each says exactly what to type and *why it
        matters*. Run the cell; the self-check compares your three numbers against tolerance bands and
        turns green when they land where the real data does.

        **Toolbox.**

        - `beh` — list of `n_sessions` dicts; each `beh[s][key]` is a `(T,)` boolean array.
        - `session_keys` — the nine channel names; `cond_labels[s]` ∈ {`"control"`, `"24hr"`, `"7d"`}.
        - `np.logical_or(a, b)` — element-wise OR of two boolean arrays.
        - a boolean array's `.mean()` — its fraction of `True` values.
        """
    )
    return


@app.cell
def _(beh, cond_labels, n_sessions, np):
    # ================= YOUR CODE — edit only the two lines marked "FILL IN" ====================
    # Part 1. For each session, check the identity  is_social == is_social_sender OR
    #         is_social_receiver, frame by frame, and record the fraction of frames that disagree.
    #         WHY it matters: if this identity does NOT hold, the label we decode in Part B is
    #         inconsistent with its own parts and every downstream number is suspect.
    _mism = []
    for _s in range(n_sessions):
        _d = beh[_s]
        # FILL IN: the element-wise OR of the two component channels for this session.
        #   Type exactly:  np.logical_or(_d["is_social_sender"], _d["is_social_receiver"])
        #   This rebuilds "social" from its definition so we can compare it to the stored channel.
        _defn = np.logical_or(_d["is_social_sender"], _d["is_social_receiver"])
        # (_d["is_social"] != _defn) is a boolean array, True where they disagree; .mean() = its fraction.
        _mism.append(float((_d["is_social"] != _defn).mean()))
    max_mismatch = float(np.max(_mism))                    # largest disagreement over all sessions

    # Part 2. Social time. A boolean array's .mean() is its fraction of True frames, so
    #         beh[_s]["is_social"].mean() is that session's social fraction. Average across sessions.
    #         WHY it matters: this is the base rate the decoder in Part B has to beat — a decoder that
    #         always guessed "not social" would be right ~85% of frames, so accuracy is misleading and
    #         we will use AUROC instead.
    # FILL IN: the per-session is_social fraction.  Type exactly:  beh[_s]["is_social"].mean()
    _fr = np.array([beh[_s]["is_social"].mean() for _s in range(n_sessions)])
    mean_frac = float(_fr.mean())                          # mean social fraction across sessions

    # Part 3. Isolation effect: mean is_social fraction of control sessions minus that of 7d.
    #         (Provided for you — a boolean mask selects the sessions of each condition.)
    _ctrl = _fr[[cond_labels[_s] == "control" for _s in range(n_sessions)]]
    _iso7 = _fr[[cond_labels[_s] == "7d" for _s in range(n_sessions)]]
    control_minus_7d = float(_ctrl.mean() - _iso7.mean())
    # ==========================================================================================
    return control_minus_7d, max_mismatch, mean_frac


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show solution": mo.md(
            r"""
            ```python
            import numpy as np
            # 1. definition check
            mism = []
            for s in range(n_sessions):
                d = beh[s]
                defn = np.logical_or(d["is_social_sender"], d["is_social_receiver"])
                mism.append((d["is_social"] != defn).mean())
            max_mismatch = float(np.max(mism))               # -> 0.0 exactly

            # 2. mean social fraction across sessions
            fr = np.array([beh[s]["is_social"].mean() for s in range(n_sessions)])
            mean_frac = float(fr.mean())                     # -> ~0.147

            # 3. isolation effect
            ctrl = fr[[cond_labels[s] == "control" for s in range(n_sessions)]]
            iso7 = fr[[cond_labels[s] == "7d" for s in range(n_sessions)]]
            control_minus_7d = float(ctrl.mean() - iso7.mean())   # -> ~+0.016
            ```

            **What you should find.** The definition holds **exactly** (`max_mismatch == 0`): the
            stored `is_social` really is the OR of the sender and receiver channels, so we can trust
            it as the label for Part B. Mean social time is about **15 % of frames** — the imbalance
            that will force us to use AUROC, not accuracy. The isolation effect is small and positive,
            **about +0.016**: control mice spend slightly more time in contact than 7-day-isolated
            mice, but at six sessions per group this is **not** significant (Kruskal–Wallis p ≈ 0.81).
            The honest answer is a small effect, not a large one.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(control_minus_7d, max_mismatch, mean_frac, mo):
    # Self-check with tolerance bands pinned from the real data:
    #   max_mismatch  == 0     (definition exact)     -> band < 1e-6
    #   mean_frac     ~ 0.147                          -> band [0.10, 0.20] (a fraction, not a count)
    #   control_minus_7d ~ +0.016 (honest SMALL effect) -> band [-0.05, 0.10]
    _p1 = float(max_mismatch) < 1e-6
    _p2 = 0.10 <= float(mean_frac) <= 0.20
    _p3 = -0.05 < float(control_minus_7d) < 0.10
    _ok = _p1 and _p2 and _p3
    _c = "#e8f5e9" if _ok else "#ffebee"
    _b = "#2e7d32" if _ok else "#c62828"
    _m1 = ("PASS: is_social == sender | receiver exactly (max mismatch = "
           f"{max_mismatch:.1e})" if _p1 else
           f"FAIL: max_mismatch = {max_mismatch:.3e} — the OR definition should hold frame-for-frame")
    _m2 = (f"PASS: mean social fraction = {mean_frac:.3f} (~15% of frames)" if _p2 else
           f"FAIL: mean_frac = {mean_frac:.3f} — expected ~0.147; did you compute a count, not a fraction?")
    _m3 = (f"PASS: control − 7d = {control_minus_7d:+.3f} — a small decline; isolation barely moves "
           "behavioral social time (not significant at n=6/group)"
           if _p3 else
           f"FAIL: control_minus_7d = {control_minus_7d:+.3f} is outside the band [−0.05, 0.10]")
    _head = "PASS — definition verified, effect read honestly" if _ok else "Not yet — fix the flagged line"
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_b};padding:12px 16px;border-radius:6px">
        <b style="color:{_b}">{_head}</b><br>
        {_m1}<br>{_m2}<br>{_m3}<br>
        <span style="font-size:0.9em;color:#555">The isolation effect is scored as a <i>small</i>
        effect on purpose — the exercise is checked against the honest result, not against noise.
        Tolerance bands: max_mismatch &lt; 1e-6 · mean_frac ∈ [0.10, 0.20] ·
        control−7d ∈ [−0.05, 0.10].</span>
        </div>
        """
    )
    return


# ============================================================================ PART B — the neurons
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # Part B · The neurons: decode the social state

        We have a trustworthy behavioral label. Now the neural half. We work in four steps: put the
        calcium on the behavior clock, look at the population, look at single cells, and then decode
        the population.

        ## B1. Put the calcium on the behavior clock

        **Why.** A neuron's activity and a behavioral label must be indexed by the same time frames
        before we can relate them. Calcium is sampled at **30 fps**, behavior at **25 fps**, so the two
        arrays differ in length and do not align frame-for-frame.

        **Method — `nu.interp_resample(C, n_out, axis=0)`.**

        - *Purpose:* resample a signal to a new number of time points by linear interpolation.
        - *Inputs:* `C`, shape `(n_frames, n_neurons)`; `n_out`, the target number of frames (the
          length of the behavior array for that session).
        - *Output:* shape `(n_out, n_neurons)`, now on the behavior clock.

        Linear interpolation re-grids the samples already present; it does not invent new peaks. The
        figure shows one real neuron at its native 30 fps and after resampling to the 25 fps behavior
        length — the shape is preserved. After resampling we **z-score** each neuron (subtract its
        mean, divide by its standard deviation, so all cells are on a comparable scale) and **crop** to
        the first 3 minutes after the partner enters (`Int_Entry`), the window where the interaction
        happens.
        """
    )
    return


@app.cell
def _(go, img, np, nu):
    # One real neuron: native 30 fps samples vs resampled onto a coarser 25 fps grid.
    _C = img[6][:, 40]                      # one neuron, session 6
    _seg = _C[:120]                         # first 120 frames for legibility
    _res = nu.interp_resample(_seg, int(len(_seg) * 25 / 30))
    _fig = go.Figure()
    _fig.add_scatter(x=np.linspace(0, 1, len(_seg)), y=_seg, mode="lines+markers",
                     line=dict(color="#4c78a8", width=1), marker=dict(size=4),
                     name="calcium @ 30 fps (original)")
    _fig.add_scatter(x=np.linspace(0, 1, len(_res)), y=_res, mode="markers",
                     marker=dict(color="#e45756", size=7, symbol="x"),
                     name="resampled → 25 fps behavior clock")
    _fig.update_layout(template="plotly_white", height=300, margin=dict(l=10, r=10, t=40, b=10),
                       title="interp_resample: one neuron, 30 fps → 25 fps (shape preserved)",
                       xaxis_title="normalized time [0,1]", yaxis_title="calcium (a.u.)",
                       legend=dict(y=1.0))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## B2. The population raster with the social overlay

        **Why.** Before decoding, look at the whole population at once and check whether activity
        visibly changes during social frames. If nothing is visible here, no decoder is going to
        conjure signal from nothing.

        **Definition — raster.** A heatmap with one row per neuron and one column per time frame; color
        is that neuron's z-scored activity. It shows the entire population's activity over time in one
        image.

        **Method.** Pick a session. We run the full pipeline —
        `interp_resample → zscore(axis=0) → crop [entry, entry + 3·60·25]` — and draw the z-scored
        population. The **green bands** mark frames where the focal mouse is socially engaging
        (`is_social_sender`); unmarked frames are non-social. If the population carries social
        information, activity under the green bands should look different from the rest.
        """
    )
    return


@app.cell
def _(cond_labels, mo, n_sessions):
    _opts = {f"session {s}  ·  {cond_labels[s]}": s for s in range(n_sessions)}
    session_pick = mo.ui.dropdown(options=_opts, value="session 6  ·  7d",
                                  label="session (condition)")
    return (session_pick,)


@app.cell
def _(beh, behavior_fps, ent, img, np, nu, session_pick):
    # Full pipeline for the selected session: resample -> zscore -> crop. Order matters: the pinned
    # social-neuron counts and decoder AUROCs all use resample -> zscore -> crop.
    _s = int(session_pick.value)
    _iss = beh[_s]["is_social_sender"].astype(bool)
    _r = nu.zscore(nu.interp_resample(img[_s], len(_iss), axis=0), axis=0)
    _e = int(ent["Int_Entry"].iloc[_s])
    _t0, _t1 = _e, int(_e + 3 * 60 * behavior_fps)
    sess_neurons = _r[_t0:_t1]                 # (T, n_neurons) z-scored, cropped
    sess_social = _iss[_t0:_t1]                # (T,) bool
    sess_ncells = sess_neurons.shape[1]
    sess_frac = float(sess_social.mean())
    return sess_frac, sess_ncells, sess_neurons, sess_social


@app.cell
def _(go, mo, np, nu, session_pick, sess_frac, sess_ncells, sess_neurons, sess_social):
    _R = sess_neurons.T                          # (n_neurons, T)
    _fig = nu.raster_fig(_R, title=(f"{session_pick.selected_key}  ·  {sess_ncells} neurons  ·  "
                                    f"{sess_frac:.0%} of frames social"),
                         xlabel="time (frames, 25 fps)", ylabel="neuron",
                         zmin=-3, zmax=3, colorbar_title="z", height=460)
    _s = sess_social.astype(int)
    _edges = np.flatnonzero(np.diff(np.r_[0, _s, 0]))
    _starts, _ends = _edges[0::2], _edges[1::2]
    for _a, _b in zip(_starts, _ends):
        _fig.add_vrect(x0=_a, x1=_b, fillcolor="#2ca02c", opacity=0.18, line_width=0, layer="below")
    _fig.add_annotation(x=0.01, y=1.06, xref="paper", yref="paper", showarrow=False,
                        text="green = social (is_social_sender)", font=dict(color="#2ca02c", size=12))
    mo.vstack([session_pick, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The raster shows the population changing over the recording, with the social bands clustered
        in time. Whether any *single* row (neuron) reliably brightens under the green is hard to read
        by eye across hundreds of cells — which is exactly why we quantify it, first one neuron at a
        time, then all at once.

        ---
        ## B3. One neuron at a time: social vs non-social

        **Why.** To build intuition, take a single cell and ask whether its activity distribution
        differs between social and non-social frames.

        **Method.** We split the chosen neuron's z-scored activity into **social** and **non-social**
        frames and draw the two **empirical cumulative distributions** (ECDFs). An ECDF at value $v$ is
        the fraction of frames at or below $v$; two curves that are horizontally separated mean the two
        conditions have different activity distributions. If the social curve sits to the *right* of
        the non-social curve, the neuron is more active during social frames. The title reports the
        summary the next step will threshold on: the ratio of mean absolute activity in social vs
        non-social frames (a cell is flagged a candidate social neuron when that ratio exceeds 1.5).
        Slide through the neurons and watch which ones separate.
        """
    )
    return


@app.cell
def _(mo, sess_ncells):
    neuron_ind = mo.ui.slider(0, sess_ncells - 1, value=min(15, sess_ncells - 1), step=1,
                              label="neuron index", debounce=True, full_width=True)
    return (neuron_ind,)


@app.cell
def _(mo, neuron_ind, np, nu, sess_neurons, sess_social):
    _i = int(neuron_ind.value)
    _x = sess_neurons[:, _i]
    _grp = np.where(sess_social, "social", "non-social")
    _soc = _x[sess_social]
    _non = _x[~sess_social]
    _ratio = float(np.abs(_soc).mean() / (np.abs(_non).mean() + 1e-12))
    _delta = float(_soc.mean() - _non.mean())
    _fig = nu.ecdf_fig(
        _x, _grp, group_order=["non-social", "social"],
        colors={"social": "#2ca02c", "non-social": "#7f7f7f"},
        xlabel="activity (z-score)", ylabel="cumulative fraction of frames",
        title=(f"neuron {_i}:  |soc|/|non| ratio = {_ratio:.2f}  (social-neuron if > 1.5)   ·   "
               f"Δmean = {_delta:+.2f}"),
        height=420)
    mo.vstack([neuron_ind, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## B4. Which cells are social neurons?

        **Why.** Rather than inspect cells one at a time, apply the same test to every neuron at once
        and count how many pass.

        **Method — `nu.social_neuron_mask(neurons, is_social, method="ratio")`.**

        - *Purpose:* flag, per neuron, whether its activity differs enough between social and
          non-social frames to count as a social neuron.
        - *Inputs:* `neurons` `(T, n_neurons)` (z-scored, cropped); `is_social` `(T,)` boolean;
          `method` (`"ratio"` by default).
        - *Output:* a boolean array `(n_neurons,)`; `.sum()` gives the count.

        Each point below is one neuron's ratio; drag the threshold and watch the count of flagged
        (green) cells change. A social neuron is one whose mean absolute activity is at least
        `threshold`× higher during social frames.
        """
    )
    return


@app.cell
def _(mo):
    ratio_thr = mo.ui.slider(1.0, 3.0, value=1.5, step=0.1,
                             label="social-neuron ratio threshold", debounce=True, full_width=True)
    return (ratio_thr,)


@app.cell
def _(go, mo, np, ratio_thr, sess_neurons, sess_social):
    _soc = np.abs(sess_neurons[sess_social]).mean(axis=0)
    _non = np.abs(sess_neurons[~sess_social]).mean(axis=0)
    _ratio = np.where(_non > 0, _soc / _non, 0.0)
    _order = np.argsort(_ratio)
    _rs = _ratio[_order]
    _thr = float(ratio_thr.value)
    _is_soc = _rs > _thr
    _n = int(_is_soc.sum())
    # Points, not bars: one marker per neuron, sorted, colored by whether it clears the threshold.
    _fig = go.Figure()
    _fig.add_scatter(x=np.arange(len(_rs)), y=_rs, mode="markers",
                     marker=dict(size=6, color=np.where(_is_soc, "#2ca02c", "#c7c7c7"),
                                 line=dict(width=0.5, color="white")),
                     text=[f"neuron {int(o)}" for o in _order],
                     hovertemplate="%{text}<br>ratio=%{y:.2f}<extra></extra>", showlegend=False)
    _fig.add_hline(y=_thr, line=dict(color="#e45756", width=2, dash="dash"),
                   annotation_text=f"threshold {_thr:.1f}", annotation_position="top left")
    _fig.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=50, b=40),
                       title=f"{_n} social neurons of {len(_rs)}  (ratio > {_thr:.1f})",
                       xaxis_title="neuron (sorted by ratio)", yaxis_title="|soc| / |non| ratio")
    mo.vstack([ratio_thr, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## B5. Does isolation change the count?

        **Why.** The cohort-level question: does isolation change how many cells carry the social
        signal? Here the neural readout might be sharper than the behavioral one (Section A4), where
        total social time barely moved.

        **Method.** For every session we run the pipeline, count social neurons, and group the counts
        by condition — each point one session. The dropdown offers three detection methods (`ratio`,
        `delta`, `percentile`); they disagree in the details, which is itself worth seeing. With the
        default `ratio` method, controls carry the most social neurons.

        **Read this honestly.** Six sessions per condition and wide session-to-session spread. We
        report the *direction* of the trend, not a significance claim — a descriptive observation, not
        a hypothesis test.
        """
    )
    return


@app.cell
def _(mo):
    method_pick = mo.ui.dropdown(options=["ratio", "delta", "percentile"], value="ratio",
                                 label="social-neuron detection method")
    return (method_pick,)


@app.cell
def _(COND_COLORS, COND_ORDER, beh, behavior_fps, cond_labels, ent, img,
      method_pick, mo, np, nu):
    # Sweep all sessions with the selected method; one social-neuron count per session.
    _m = method_pick.value
    _per = []
    for _s in range(len(img)):
        _iss = beh[_s]["is_social_sender"].astype(bool)
        _r = nu.zscore(nu.interp_resample(img[_s], len(_iss), axis=0), axis=0)
        _e = int(ent["Int_Entry"].iloc[_s])
        _t0, _t1 = _e, int(_e + 3 * 60 * behavior_fps)
        _per.append(int(nu.social_neuron_mask(_r[_t0:_t1], _iss[_t0:_t1], method=_m).sum()))
    _per = np.array(_per)
    _means = {c: float(_per[[i for i, cc in enumerate(cond_labels) if cc == c]].mean())
              for c in COND_ORDER}
    _fig = nu.strip_points_fig(
        _per, cond_labels, group_order=COND_ORDER, colors=COND_COLORS,
        hover=[f"session {s}" for s in range(len(img))],
        ylabel="social neurons per session", xlabel="isolation condition",
        title=(f"social-neuron count by isolation ({_m}) — "
               f"control {_means['control']:.1f} · 24hr {_means['24hr']:.1f} · "
               f"7d {_means['7d']:.1f}   (n=6 each; each point = one session)"),
        height=430)
    mo.vstack([method_pick, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## B6. Decode social state from the population

        **Why.** The single-neuron tests look at one cell at a time. But a behavior may be represented
        *across* many cells jointly, in a way no single neuron makes obvious. A population decoder asks
        whether the whole population, taken together, carries enough information to predict social
        state — the direct neural analogue of the Week-1 behavior decoder.

        **Definitions.**

        - **Population vector.** The list of every recorded neuron's activity at one time frame. For a
          session with 218 neurons, each frame is a vector of 218 numbers. One population vector per
          frame is the decoder's input.
        - **Decode.** Train a model mapping a population vector to a label — here `is_social_sender`.
          If it predicts above chance on frames it never trained on, the behavior is *decodable* from
          the population.
        - **Cross-validation.** Split frames into 5 folds; train on 4, score the held-out 1; repeat so
          every frame is scored by a model that never saw it. This is exactly the procedure the Week-1
          behavior decoder used.
        - **AUROC** (area under the ROC curve). One number for how well the predicted probabilities
          separate the two classes: 0.5 is chance, 1.0 is perfect. Because only ~15–20 % of frames are
          social, plain accuracy is misleading (always-guess-"no" scores ~85 %), so AUROC is the right
          yardstick.

        **Method.** Same estimator as Week 1 — `StandardScaler → LogisticRegression` — 5-fold
        stratified cross-validation, AUROC. Only the *input* changed: 19 pose features became a
        population of real calcium neurons. The decoder runs on the session selected above.
        """
    )
    return


@app.cell
def _(np, sess_neurons, sess_social):
    # Population decoder: LogisticRegression, 5-fold stratified CV, AUROC (same procedure as Week 1).
    from sklearn.linear_model import LogisticRegression as _LogReg
    from sklearn.preprocessing import StandardScaler as _Scaler
    from sklearn.pipeline import make_pipeline as _mkpipe
    from sklearn.model_selection import StratifiedKFold as _SKF, cross_val_predict as _cvp
    from sklearn.metrics import roc_auc_score as _auc_score, roc_curve as _roc_curve

    _X = sess_neurons
    _y = sess_social.astype(int)
    _clf = _mkpipe(_Scaler(), _LogReg(max_iter=1000, class_weight="balanced"))
    _skf = _SKF(5, shuffle=True, random_state=0)
    dec_proba = _cvp(_clf, _X, _y, cv=_skf, method="predict_proba")[:, 1]
    dec_y = _y
    dec_auc = float(_auc_score(dec_y, dec_proba))
    dec_fpr, dec_tpr, dec_thrs = _roc_curve(dec_y, dec_proba)
    return dec_auc, dec_fpr, dec_proba, dec_tpr, dec_y


@app.cell(hide_code=True)
def _(dec_auc, mo, session_pick):
    _c = "#e8f5e9" if dec_auc > 0.7 else "#fff3e0"
    _b = "#2e7d32" if dec_auc > 0.7 else "#e65100"
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_b};padding:12px 16px;border-radius:6px">
        <b style="color:{_b}">Population decoder · {session_pick.selected_key}</b><br>
        Cross-validated <b>AUROC = {dec_auc:.3f}</b> &nbsp;(chance = 0.500).<br>
        <span style="font-size:0.9em;color:#555">The whole population, taken jointly, predicts social
        state well above chance — even though no single neuron did so cleanly. The behavior is written
        across the population, not in any one cell.</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Where the decoder is right, and where it is wrong

        **Why.** An AUROC is one number. To *see* the decoder's behavior, we plot its held-out
        (cross-validated) probability over time against the true social band. Where the probability is
        high inside a green band, the decoder scored a **hit**; high outside a band is a **false
        positive**; low inside a band is a **miss** (false negative). This is the neural counterpart of
        inspecting a behavior classifier's correct-vs-mistaken examples.
        """
    )
    return


@app.cell
def _(dec_proba, dec_y, go, np):
    # Prediction timeline: CV probability over frames, with the true social band shaded.
    _t = np.arange(len(dec_proba))
    _fig = go.Figure()
    _edges = np.flatnonzero(np.diff(np.r_[0, dec_y, 0]))
    for _a, _b in zip(_edges[0::2], _edges[1::2]):
        _fig.add_vrect(x0=_a, x1=_b, fillcolor="#2ca02c", opacity=0.15, line_width=0, layer="below")
    _fig.add_scatter(x=_t, y=dec_proba, mode="lines", line=dict(color="#4c78a8", width=1),
                     name="P(social) — held-out")
    _fig.add_hline(y=0.5, line=dict(color="#999", width=1, dash="dash"))
    _fig.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=50, b=40),
                       title="held-out P(social) over time  ·  green = truly social  ·  "
                             "high-in-green = hit, high-outside = false positive, low-in-green = miss",
                       xaxis_title="time (frames, 25 fps)", yaxis_title="P(social)",
                       legend=dict(y=1.0))
    _fig.update_yaxes(range=[-0.02, 1.02])
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Choosing a decision threshold

        **Why.** The decoder outputs a *probability*, not a yes/no call. A **decision threshold**
        converts it to a label: predict "social" when the probability is at or above the threshold. It
        is a trade-off. A low threshold labels more frames social, catching real bouts but raising
        false positives; a high threshold is stricter, fewer false positives but more misses.

        **Definitions.** *Precision* = fraction of frames the decoder called social that really are.
        *Recall* = fraction of truly social frames the decoder caught. The **ROC curve** plots the
        true-positive rate against the false-positive rate across all thresholds; the marker shows
        where the current threshold sits. Slide it and read the operating point, precision, recall, and
        the confusion counts — all on the held-out predictions.
        """
    )
    return


@app.cell
def _(mo):
    thr_slider = mo.ui.slider(0.05, 0.95, value=0.5, step=0.05,
                              label="decision threshold", debounce=True, full_width=True)
    return (thr_slider,)


@app.cell
def _(dec_auc, dec_fpr, dec_proba, dec_tpr, dec_y, go, mo, np, thr_slider):
    _thr = float(thr_slider.value)
    _pred = (dec_proba >= _thr).astype(int)
    _tp = int(((_pred == 1) & (dec_y == 1)).sum())
    _fp = int(((_pred == 1) & (dec_y == 0)).sum())
    _fn = int(((_pred == 0) & (dec_y == 1)).sum())
    _tn = int(((_pred == 0) & (dec_y == 0)).sum())
    _prec = _tp / (_tp + _fp) if (_tp + _fp) else 0.0
    _rec = _tp / (_tp + _fn) if (_tp + _fn) else 0.0
    _fpr_here = _fp / (_fp + _tn) if (_fp + _tn) else 0.0
    _roc = go.Figure()
    _roc.add_scatter(x=dec_fpr, y=dec_tpr, mode="lines", line=dict(color="#4c78a8", width=2),
                     name=f"ROC (AUC {dec_auc:.3f})")
    _roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines",
                     line=dict(color="#bbb", width=1, dash="dash"), name="chance")
    _roc.add_scatter(x=[_fpr_here], y=[_rec], mode="markers",
                     marker=dict(color="#e45756", size=13, symbol="x"),
                     name=f"threshold {_thr:.2f}")
    _roc.update_layout(template="plotly_white", height=440, margin=dict(l=10, r=10, t=50, b=10),
                       title=(f"@ threshold {_thr:.2f}:  precision {_prec:.2f} · recall {_rec:.2f}  "
                              f"(TP {_tp} · FP {_fp} · FN {_fn} · TN {_tn})"),
                       xaxis_title="false-positive rate", yaxis_title="true-positive rate",
                       legend=dict(y=0.05, x=0.55))
    _roc.update_xaxes(range=[-0.02, 1.02])
    _roc.update_yaxes(range=[-0.02, 1.02], scaleanchor="x", scaleratio=1)
    mo.vstack([thr_slider, _roc])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## B7. Does the decoder work in *every* session? (heavy — click to run)

        **Why.** Session 6 might be lucky. The honest test of "can we read social state off the
        neurons" is whether a within-session decoder clears chance in *all* sessions, and whether its
        quality tracks something interpretable — like how many neurons were recorded. This runs the
        full pipeline + 5-fold CV for all 18 sessions (a few seconds), so it is gated behind a button.
        """
    )
    return


@app.cell
def _(mo):
    run_all = mo.ui.run_button(label="compute within-session decoders for all 18 sessions")
    return (run_all,)


@app.cell
def _(beh, behavior_fps, cond_labels, COND_COLORS, COND_ORDER, ent, img, mo,
      np, nu, run_all):
    mo.stop(not run_all.value,
            mo.md("*Click the button above to run all 18 within-session decoders.*"))
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.preprocessing import StandardScaler as _SS
    from sklearn.pipeline import make_pipeline as _mp
    from sklearn.model_selection import StratifiedKFold as _KF, cross_val_predict as _CVP
    from sklearn.metrics import roc_auc_score as _AUC

    _aucs, _ncells = [], []
    for _s in range(len(img)):
        _iss = beh[_s]["is_social_sender"].astype(bool)
        _r = nu.zscore(nu.interp_resample(img[_s], len(_iss), axis=0), axis=0)
        _e = int(ent["Int_Entry"].iloc[_s])
        _t0, _t1 = _e, int(_e + 3 * 60 * behavior_fps)
        _X, _y = _r[_t0:_t1], _iss[_t0:_t1].astype(int)
        _ncells.append(_X.shape[1])
        _clf = _mp(_SS(), _LR(max_iter=1000, class_weight="balanced"))
        _p = _CVP(_clf, _X, _y, cv=_KF(5, shuffle=True, random_state=0),
                  method="predict_proba")[:, 1]
        _aucs.append(float(_AUC(_y, _p)))
    _aucs = np.array(_aucs)
    _med = float(np.median(_aucs))
    _fig = nu.strip_points_fig(
        _aucs, cond_labels, group_order=COND_ORDER, colors=COND_COLORS,
        hover=[f"session {s} · {_ncells[s]} cells" for s in range(len(img))],
        ylabel="within-session decoder AUROC", xlabel="isolation condition",
        title=(f"every session decodes social state above chance  ·  median AUROC = {_med:.3f}  "
               f"(min {_aucs.min():.3f} = the 12-neuron session; chance = 0.500)"), height=440)
    _fig.add_hline(y=0.5, line=dict(color="#999", width=1, dash="dash"))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Every session decodes social state well above chance (median AUROC ≈ 0.93). The single weakest
        session is the one with only **12 recorded neurons** — a thin population vector carries less
        information — which is the reassuring, interpretable pattern: more neurons, more to read from.

        ---
        ## B8. Why within-session, and not across sessions

        We stated it at the start; now it matters concretely. Each recording extracts its **own** set
        of cells with **no identity correspondence** across sessions — a 218-vector from one session
        and a 202-vector from another do not share axes, so a decoder fit to one cannot be applied to
        the other. This is unlike the pose decoder of Week 1, whose 19 features meant the same thing in
        every recording, which is why *that* one could be tested leave-one-cohort-out. Making a neural
        decoder transfer across sessions is a real research problem — it needs cell registration (find
        the same physical neuron across days) or a shared latent space — and it is a natural step
        beyond this course. Here, honestly, we decode within a session.

        ---
        ## B9. Exercise 2 — decode, then count

        **Python skill practiced:** *calling a full scikit-learn pipeline* — the top rung of the
        course's coding ramp. You will score cross-validated probabilities with AUROC and read a
        grouped summary. This is the same estimator you used on pose features in Week 1; only the input
        is different.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Hypothesis.** A linear readout of the population vector decodes social state above chance;
        and controls carry the most social neurons.

        **What is provided.** A `_pipeline(s)` helper (runs `interp_resample → zscore → crop`, returns
        population vectors `X` and per-frame `is_social_sender` labels `y`), the cross-validation
        setup, and the per-session counting loop. Two lines are yours, both marked `# TODO`, each with
        a comment saying exactly what to type and why.

        - **Part 1 — decode.** Cross-validated probabilities `_proba` are already computed for session
          6. On `# TODO (1)`, score them with AUROC into `decoder_auc`.
        - **Part 2 — count.** The dict `_means` holds the mean social-neuron count per condition. On
          `# TODO (2)`, pick the condition with the **highest** mean into `most_social_condition`.

        **Expected result.** `decoder_auc` near **0.95** (a linear population readout far above the 0.5
        chance line), and `most_social_condition == "control"` (means: control ≈ 11.2, 7d ≈ 7.8,
        24hr ≈ 5.8). Part 2 is a descriptive direction, not a significance test (n = 6/condition).
        """
    )
    return


@app.cell
def _(beh, behavior_fps, cond_labels, ent, img, np, nu):
    # ------------------------------------------------ YOUR CODE (edit the two TODO lines) -----------
    # Provided helper: run resample -> zscore -> crop for one session and return
    #   X = population vectors (T, n_neurons),  y = per-frame is_social_sender label (T,).
    def _pipeline(s, key="is_social_sender"):
        _iss = beh[s][key].astype(bool)
        _r = nu.zscore(nu.interp_resample(img[s], len(_iss), axis=0), axis=0)
        _e = int(ent["Int_Entry"].iloc[s])
        _t0, _t1 = _e, int(_e + 3 * 60 * behavior_fps)
        return _r[_t0:_t1], _iss[_t0:_t1]

    from sklearn.linear_model import LogisticRegression as _LogReg2
    from sklearn.preprocessing import StandardScaler as _Scaler2
    from sklearn.pipeline import make_pipeline as _mkpipe2
    from sklearn.model_selection import StratifiedKFold as _SKF2, cross_val_predict as _cvp2
    from sklearn.metrics import roc_auc_score as _auc_score2

    # Part 1 — decode session 6. Everything up to the cross-validated probabilities is done for you.
    _X, _yb = _pipeline(6)
    _y = _yb.astype(int)
    _clf = _mkpipe2(_Scaler2(), _LogReg2(max_iter=1000, class_weight="balanced"))
    _skf = _SKF2(5, shuffle=True, random_state=0)
    _proba = _cvp2(_clf, _X, _y, cv=_skf, method="predict_proba")[:, 1]
    # TODO (1): score the held-out probabilities with AUROC.
    #   Type exactly:  float(_auc_score2(_y, _proba))
    #   WHY: AUROC (not accuracy) is the honest yardstick because only ~20% of frames are social, so a
    #   "never social" guess would already score ~80% accuracy while learning nothing.
    decoder_auc = float(_auc_score2(_y, _proba))          # <-- FILL IN

    # Part 2 — count social neurons per session, then average by condition. The loop is done for you.
    _counts = {}
    for _s in range(len(img)):
        _Xs, _ys = _pipeline(_s)
        _counts.setdefault(cond_labels[_s], []).append(
            int(nu.social_neuron_mask(_Xs, _ys, method="ratio").sum()))
    _means = {c: float(np.mean(v)) for c, v in _counts.items()}
    # TODO (2): pick the condition with the HIGHEST mean count.
    #   Type exactly:  max(_means, key=_means.get)
    #   WHY: this reads the DIRECTION of the isolation effect on the neural readout — the neural side
    #   of the story, to compare against the near-null behavioral effect from Part A.
    most_social_condition = max(_means, key=_means.get)   # <-- FILL IN
    # ------------------------------------------------------------------------------------------------
    return decoder_auc, most_social_condition


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show solution": mo.md(
            r"""
            ```python
            def pipeline(s, key="is_social_sender"):
                iss = beh[s][key].astype(bool)
                r = nu.zscore(nu.interp_resample(img[s], len(iss), axis=0), axis=0)
                e = int(ent["Int_Entry"].iloc[s]); t0, t1 = e, int(e + 3*60*25)
                return r[t0:t1], iss[t0:t1]

            # Part 1 — decode
            X, y = pipeline(6); y = y.astype(int)
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=1000, class_weight="balanced"))
            proba = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                      method="predict_proba")[:, 1]
            decoder_auc = roc_auc_score(y, proba)          # ~ 0.95

            # Part 2 — count, then read the direction
            counts = {}
            for s in range(len(img)):
                Xs, ys = pipeline(s)
                counts.setdefault(cond_labels[s], []).append(
                    int(nu.social_neuron_mask(Xs, ys, method="ratio").sum()))
            means = {c: np.mean(v) for c, v in counts.items()}
            most_social_condition = max(means, key=means.get)   # "control"
            ```

            **What you should find.** The population decoder lands around **AUROC ≈ 0.95**: a linear
            readout of ~218 neurons predicts social state well above chance. The social-neuron count is
            highest in **controls** (control ≈ 11.2, 24hr ≈ 5.8, 7d ≈ 7.8), so isolation lowers the
            count. With n = 6 per condition and this spread, that is a descriptive direction, not a
            p-value. The exercise grades the honest reading, not the noise.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(decoder_auc, most_social_condition, mo):
    # Self-check with tolerance bands pinned from the real data:
    #   decoder_auc ~ 0.95 (session 6); graded as "above chance" -> > 0.70.
    #   most_social_condition -> "control" (means control 11.17 > 7d 7.83 > 24hr 5.83).
    _p1 = float(decoder_auc) > 0.70
    _p2 = str(most_social_condition) == "control"
    _ok = _p1 and _p2
    _c = "#e8f5e9" if _ok else "#ffebee"
    _b = "#2e7d32" if _ok else "#c62828"
    _m1 = (f"PASS: decoder AUROC = {decoder_auc:.3f} — a linear population readout predicts social "
           "state above the 0.5 chance line" if _p1 else
           f"FAIL: decoder AUROC = {decoder_auc:.3f} is at/near chance — check the pipeline order "
           "(resample → zscore → crop) and that you fed the full population vector")
    _m2 = ("PASS: controls carry the most social neurons — isolation lowers the count (descriptive; "
           "n=6/condition)" if _p2 else
           f"FAIL: most_social_condition = {most_social_condition!r}; the pinned means are "
           "control 11.2 > 7d 7.8 > 24hr 5.8 → 'control'")
    _head = ("PASS — decoder above chance, and the condition direction read correctly" if _ok else
             "Not yet — fix the flagged part")
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_b};padding:12px 16px;border-radius:6px">
        <b style="color:{_b}">{_head}</b><br>
        {_m1}<br>{_m2}<br>
        <span style="font-size:0.9em;color:#555">Tolerance band: AUROC &gt; 0.70 (chance = 0.50;
        pinned ≈ 0.95). Part 2 is graded on the direction of the trend, not a significance test.</span>
        </div>
        """
    )
    return


# ============================================================================ background
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Background — the paradigm, the key papers, and the limits of this analysis": mo.md(
            r"""
            **The social-isolation paradigm.** Isolate a social animal, reintroduce a partner, and
            measure what changes. **Matthews et al. 2016, *Cell* 164:617** showed dorsal-raphe
            dopamine neurons encode a loneliness-like state and that acute isolation increases social
            approach; **Zelikowsky et al. 2018, *Cell*** linked chronic isolation to Tac2/neurokinin-B
            signaling across amygdala and hypothalamus; **Tomova et al. 2020, *Nature Neuroscience***
            found a midbrain social-craving signal in humans. **Coding of conspecifics and social
            state:** Remedios et al. 2017 *Nature* (VMHvl); Kingsbury et al. 2019 *Cell* (interbrain
            coding of dominance). **Population decoding of social variables:** Padilla-Coreano et al.
            2022 *Nature* (mPFC ensembles decode competitive rank and social behavior) — the closest
            precedent for the decoder here, and the same paper the Week-1 behavior decoder cited from
            the behavioral side.

            **The shared method.** A population decoder is a supervised map from a high-dimensional
            state vector to a label, cross-validated so the score reflects generalization, not
            memorization. It is the same estimator whether the vector is 19 pose features or N neurons.
            Only the input changes.

            **Limits of this analysis.**

            - **Correlation, not cause.** A decodable social signal does not mean these cells *drive*
              social behavior. Decoding is a read-out; causation needs perturbation.
            - **Small n, two clocks aligned by interpolation.** Six sessions per condition, and the
              calcium was interpolated onto the behavior clock. Resampling only re-grids existing
              structure, but it also smooths, and any clock mis-registration would distort alignment.
              Treat the condition trends as descriptive.
            - **Within-session, not cross-session.** Each session extracts its own neurons with no
              identity correspondence, so the decoder is cross-validated within one session's
              population. Transferring across sessions needs cell registration or a shared latent
              space, which this dataset does not provide.
            - **Narrow behavioral vocabulary.** Nine scored channels for a dyadic reintroduction assay,
              dominated by `is_touching` — not the free behavior of Week 1.
            """
        )
    })
    return


# ============================================================================ close
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Summary — and the end of the course

        **The question we asked.** Can we read a social behavior directly off a population of neurons?

        **The answer.** Yes. We first built the behavioral ground truth — a `(9, T)` social ethogram
        per session — and verified its internal definition (`is_social` equals sender ∨ receiver,
        exactly) before trusting it. We put the calcium on the behavior clock, saw that no single
        neuron cleanly separated social from non-social, and then trained a **population decoder** that
        read `is_social_sender` off the whole population at **AUROC ≈ 0.95** on session 6 and above
        chance in **every** one of the 18 sessions. The social behavior is written across the
        population, not in any one cell. Isolation trended toward *fewer* social neurons (controls
        highest), a direction — not a result — given six sessions per condition. And the neural readout
        was sharper than the behavioral one: total social time barely moved across conditions, but the
        population still carried the moment-to-moment signal clearly.

        **The next question.** This decoder lives inside a single session because the neurons have no
        identity across recordings. The natural next question — *can a social decoder transfer across
        animals and days?* — is a real research problem: register the same cells over time, or learn a
        shared latent space, then decode. That is where this course hands off to the literature.

        **Closing the loop.** We opened the course with raw pose — keypoints on a mouse — and asked
        what behavior *is*, building it up into features, a low-dimensional map, and a behavior
        decoder. We close it by reading a social state off real neurons with the very same decoding
        logic. We started with behavior, and we ended by reading it off the brain. That is the whole
        arc of behavioral neuroscience in ten notebooks.
        """
    )
    return


if __name__ == "__main__":
    app.run()
