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


# ============================================================ 0. Throughline: where we are
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 05 · Behavior in time, and decoding it

        ## The question so far, and the question now

        In the previous notebook we built a **map** of behavior and asked what it *means*. We saw that
        the map is not a black box: individual features vary smoothly across it, dense regions
        correspond to recognizable kinds of interaction, and — importantly — the map carries real
        biological signal. A sex difference in how mice orient toward each other survived a strict
        cage-level test; a food-deprivation shift in how far a bystander sits survived a paired test.
        So behavior, summarized as 19 numbers per event, has **structure across events** and that
        structure reflects biology.

        This notebook closes Week 1 with the two questions that turn a description into a usable tool.

        1. **Does behavior have structure in *time*?** Every event so far has been treated as a still
           picture — one clip, summarized. But animals do not teleport between behaviors. If a cage is
           resting now, is it more likely to be resting a moment later, or is each moment an independent
           coin flip? And across the 24-hour day, when are these mice active? We are looking for
           **memory** and **rhythm**.

        2. **Can we *predict* a behavior from the features?** We have been hand-labeling aggression. If
           the 19 features contain enough information, a model should be able to read aggression out of
           them automatically — a **decoder**. The real test is not whether it works on the data it
           learned from, but whether it works on animals, and even a whole second experiment, it has
           never seen.

        We study social behavior and its neural basis, and both depend on an objective, reproducible
        readout of behavior. That is what this notebook builds and stress-tests.
        """
    )
    return


# ============================================================ Data load
@app.cell
def _(cu):
    import warnings as _warnings
    _warnings.filterwarnings("ignore", category=RuntimeWarning)   # nanmean of all-NaN frames is expected
    _warnings.filterwarnings("ignore")                            # sklearn convergence chatter on tiny fits

    # Event corpus + the precomputed 19-feature / PCA bundle, aligned row-for-row.
    ev = cu.load_events("data/train_events.npz")
    der = cu.load_derived("train")
    # The held-out cage (cam16): a whole cage set aside from the very start, used only to test.
    ho = cu.load_events("data/heldout_events.npz")
    hod = cu.load_derived("heldout")
    # The precomputed UMAP sweep, only for its canonical cluster labels (never runs UMAP live).
    sweep = cu.load_umap_sweep()
    # Three continuous 24-hour cages at 2 fps, for the grammar and the activity clock:
    # 15 = example cage (M), 10 = context (F), 13 = context (M).
    t15 = cu.load_continuous_tracks("15")
    t10 = cu.load_continuous_tracks("10")
    t13 = cu.load_continuous_tracks("13")
    return der, ev, ho, hod, sweep, t10, t13, t15


@app.cell
def _(der, ev, ho, hod):
    # Name the pieces the decoder will use, so later cells read cleanly.
    X = der["X"]                      # (2499, 19) training features
    y = ev["agg_label"].astype(int)   # (2499,) ground-truth aggression (1) vs not (0)
    cage = der["cage"]                # (2499,) cohort-unique cage id (9-15 and 109-115)
    cohort = der["cohort"]            # (2499,) date-tag of the food-deprivation cohort this cage is in
    Xh = hod["X"]                     # (780, 19) held-out cam16 features
    yh = ho["agg_label"].astype(int)  # (780,) held-out ground truth
    return X, Xh, cage, cohort, y, yh


# ============================================================ The running example event
@app.cell
def _(cu, der, ev):
    # Event #909 has been our running example all week. On the current two-cohort bundle it is a
    # fully-tracked interaction from cage 110 (cohort 12192025, a female cage). Its ranks are
    # [Sub, Mid, Dom] -> approacher=Sub (green), approachee=Mid (blue), bystander=Dom (red).
    # Its registry category is "mlp_fp": an ambiguous near-miss that an earlier model flagged as
    # aggression but a human scored as NOT aggression (ground-truth agg_label = 0). That ambiguity is
    # useful later, when we watch the decoder score it near its decision boundary.
    ex_idx = cu.event_index_by_key(ev, "12192025_pre|cam.10.00046-2025-12-18T16|m0-m2|83141")
    ex_cage = int(der["cage"][ex_idx])
    ex_sex = str(der["sex"][ex_idx])
    ex_tod = float(cu.time_of_day(str(ev["event_key"][ex_idx])))
    ex_gif = cu.gif_img_html(
        cu.event_gif_bytes(ev["kp"][ex_idx], ev["ranks"][ex_idx],
                           int(ev["contact_rel"][ex_idx]), cell=200), width=200)
    return ex_cage, ex_gif, ex_idx, ex_sex, ex_tod


@app.cell(hide_code=True)
def _(ex_cage, ex_gif, ex_idx, ex_sex, ex_tod, mo):
    mo.md(
        f"""
        ### Our example interaction

        We keep the same example event we have followed all week (event **#{ex_idx}**, cage
        **{ex_cage}**, **{"female" if ex_sex == "F" else "male"}** cage). The two interacting mice are
        the **approacher** and the **approachee**; a third mouse is a **bystander**. Skeletons are
        colored **only by social rank** (<span style="color:#d62728">Dom = red</span>,
        <span style="color:#1f77b4">Mid = blue</span>, <span style="color:#2ca02c">Sub = green</span>).
        The white arrow points from approacher to approachee; the red dot marks the moment of contact.

        {ex_gif}

        This is a deliberately hard case — a near-miss that sits right at the boundary of "aggression."
        It occurred at about **{ex_tod:.1f} h** (deep in the dark, active phase, as the activity clock
        below will show). A 130-frame clip like this is a *snapshot*. The first half of the notebook
        widens out from the snapshot to the whole day it belongs to.
        """
    )
    return


# ============================================================================================
# ============================  PART A — DOES BEHAVIOR HAVE STRUCTURE IN TIME?  ================
# ============================================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # Part A · Does behavior have structure in time?

        ## Why this matters

        Everything up to now treated behavior as a bag of independent snapshots. But an animal resting
        now tends to keep resting; a chase now makes another burst of motion likely a moment later.
        That **temporal structure** is itself a description of behavior — and measuring it on
        un-manipulated animals gives us the baseline against which any manipulation would have to show a
        change. We ask two concrete things: does behavior carry **memory** from one moment to the next,
        and does it follow a **daily rhythm**.

        ## Terms, defined before we use them

        - **State** — a coarse label for what the whole cage is doing at one moment. We use three:
          *rest*, *locomote* (moving, apart), and *huddle* (mice close together).
        - **Markov chain** — a sequence of states in which the next state depends only on the *current*
          state, not the entire history before it. Everyday analogy: weather that is "sunny" or
          "rainy", where tomorrow's odds depend only on today.
        - **Transition matrix** — a table `T` where `T[i, j]` is the probability of moving to state `j`
          next, given you are in state `i` now. Every row sums to 1.
        - **Stationary distribution** — the long-run fraction of time spent in each state if the chain
          runs forever.
        - **Entropy** — one number, in *bits*, measuring how unpredictable the next state is. Low
          entropy means the next state is easy to guess; high entropy means it is nearly random.

        The plan: label every frame of a continuous recording with a state, build the transition matrix
        by counting, read off the stationary distribution and entropy, test them against a shuffled
        null, and then look at the activity clock across the day.
        """
    )
    return


@app.cell(hide_code=True)
def _(ex_tod, mo):
    mo.md(
        f"""
        ## A.1 · From one event to the whole day

        Our example event lasted 130 frames. To ask whether behavior has **memory** and a **daily
        rhythm**, we need a *contiguous* recording, not a pile of disconnected clips. So we switch to a
        **continuous 24-hour recording of cage 15**, sampled at 2 fps (172,800 frames). Later we mark
        our example's time-of-day, **{ex_tod:.1f} h**, on that day's activity clock.

        <div style="border-left:4px solid #888; padding:6px 12px; background:rgba(0,0,0,0.03);">
        <b>Why the short event clips cannot form a Markov chain.</b> The event corpus is a set of
        <i>disconnected</i> 130-frame snippets from different days and cages. A Markov chain needs a
        <b>contiguous</b> sequence: the state at time <i>t</i> must be the state that was actually
        followed by the state at <i>t+1</i>. So the grammar is built <b>only</b> from the continuous
        recording, never from the event corpus.
        </div>
        """
    )
    return


# ============================================================ Discretize -> contiguous states
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## A.2 · Turning kinematics into a state sequence

        Before we can build a Markov chain we need a state label for every frame. `cu.discretize_states`
        does this.

        - **Purpose** — assign each moment one coarse, cage-level state.
        - **Inputs** — `speed` (T, 3), the movement speed of each mouse, and `centroids` (T, 3, 2), the
          body-center position of each mouse.
        - **Output** — `state_seq` (T,), one integer per frame, plus the list of state names.

        It applies two data-driven thresholds:

        - **rest (0)** — average mouse speed below the 40th percentile.
        - **locomote (1)** — moving, and the mice are apart.
        - **huddle (2)** — the closest pair of mice is nearer than the 25th-percentile distance
          (a proxy for social proximity).

        The result is the valid substrate for a Markov chain: a single unbroken ribbon of states, one
        frame after the next. Below we recompute the states live from the raw kinematics and confirm
        they reproduce the shipped `state_seq` exactly, then scrub a 5-minute window of the ribbon.
        """
    )
    return


@app.cell
def _(cu, np, t15):
    _state_live, STATE_NAMES = cu.discretize_states(t15["speed"], t15["centroids"])
    states_match = bool(np.array_equal(_state_live, t15["state_seq"]))
    STATE_COLORS = ["#9e9e9e", "#ff7f0e", "#6a3d9a"]   # rest / locomote / huddle (STATE colors, not mice)
    return STATE_COLORS, STATE_NAMES, states_match


@app.cell
def _(mo):
    ribbon_start = mo.ui.slider(0, 172200, value=54000, step=600,
                                label="ribbon start frame (2 fps · window = 5 min)",
                                debounce=True, full_width=True)
    return (ribbon_start,)


@app.cell
def _(STATE_COLORS, STATE_NAMES, go, mo, ribbon_start, states_match, t15):
    _s0 = int(ribbon_start.value)
    _win = 600
    _seg = t15["state_seq"][_s0:_s0 + _win]
    _tod0 = float(t15["tod_hour"][_s0])
    _fig = go.Figure(go.Heatmap(
        z=[_seg], zmin=0, zmax=2,
        colorscale=[[0.0, STATE_COLORS[0]], [0.33, STATE_COLORS[0]],
                    [0.34, STATE_COLORS[1]], [0.66, STATE_COLORS[1]],
                    [0.67, STATE_COLORS[2]], [1.0, STATE_COLORS[2]]],
        showscale=False, hovertemplate="frame +%{x}<extra></extra>"))
    _fig.update_layout(template="plotly_white", height=170,
                       title=f"Cage-15 state ribbon — 5-min window starting ~{_tod0:.1f} h "
                             f"(gray=rest · orange=locomote · purple=huddle)",
                       margin=dict(l=10, r=10, t=44, b=10))
    _fig.update_yaxes(showticklabels=False)
    _fig.update_xaxes(title="frames since window start (2 fps)")
    _check = ("The live `discretize_states` reproduces the shipped `state_seq` exactly"
              if states_match else "Mismatch — using the shipped `state_seq`")
    mo.vstack([ribbon_start, _fig,
               mo.md(f"*{_check}. The ribbon is **contiguous** — a real sequence in time, which is "
                     f"what a Markov chain requires. States: {', '.join(STATE_NAMES)}. Notice the long "
                     f"unbroken runs of one color: that stickiness is the memory we are about to "
                     f"measure.*")])
    return


# ============================================================ Per-cage grammar (compute)
@app.cell
def _(cu, np, t10, t13, t15):
    # Build the grammar (transition matrix + summary statistics) for all three cages. Nulls at n=40:
    # the shuffle-entropy of a 172,800-point sequence is very tight (std ~1e-3), so 40 draws already
    # pin it.
    def _grammar(tr):
        s = tr["state_seq"]
        T = cu.transition_matrix(s, 3)
        return dict(
            sex=tr["sex"], T=T,
            H=float(cu.transition_entropy(T)),
            self=float(np.mean(np.diag(T))),
            frac=np.bincount(s, minlength=3) / len(s),
            null_H=cu.shuffle_transition_null(s, n=40, seed=0, stat="entropy"),
            null_self=cu.shuffle_transition_null(s, n=40, seed=0, stat="self"),
        )
    grammar = {"15": _grammar(t15), "10": _grammar(t10), "13": _grammar(t13)}
    CAGE_COLORS = {"15": "#1b9e77", "10": "#d95f02", "13": "#7570b3"}   # per-cage, not per-mouse
    return CAGE_COLORS, grammar


# ============================================================ Transition matrix heatmap
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## A.3 · The transition matrix

        A transition matrix summarizes the whole day's dynamics in one small table.

        **A tiny worked example first.** Suppose a cage has only two states, *rest* and *move*. If from
        *rest* the mouse stays resting 80% of the time and starts moving 20%, and from *move* it keeps
        moving 70% and settles to rest 30%, the transition matrix is:

        $$T = \begin{bmatrix} 0.8 & 0.2 \\ 0.3 & 0.7 \end{bmatrix}$$

        Row 1 reads "given resting now, next is 0.8 rest / 0.2 move." Every row sums to 1. Large numbers
        on the **diagonal** mean behavior is *sticky* (long dwells in one state); large **off-diagonal**
        numbers mean it switches often.

        **The function.** `cu.transition_matrix(state_seq, n_states)`:

        - **Purpose** — estimate `T[i, j] = P(next = j | now = i)` from data.
        - **Input** — the contiguous `state_seq` and the number of states.
        - **Output** — a `(K, K)` matrix whose rows sum to 1.

        It works by counting every consecutive `(now, next)` pair, then dividing each row by its total.
        Pick a cage below to see its matrix.
        """
    )
    return


@app.cell
def _(grammar, mo):
    cage_pick = mo.ui.dropdown(
        options={f"Cage {c} ({grammar[c]['sex']})" + (" · example" if c == "15" else ""): c
                 for c in ["15", "10", "13"]},
        value="Cage 15 (M) · example", label="cage")
    return (cage_pick,)


@app.cell
def _(STATE_NAMES, cage_pick, go, grammar, mo, np):
    _c = cage_pick.value
    _T = grammar[_c]["T"]
    _fig = go.Figure(go.Heatmap(
        z=_T, x=STATE_NAMES, y=STATE_NAMES, colorscale="Blues", zmin=0, zmax=1,
        text=np.round(_T, 2), texttemplate="%{text}", textfont=dict(size=15),
        colorbar=dict(title="P(next|now)")))
    _fig.update_layout(template="plotly_white", height=420,
                       title=f"Cage {_c} transition matrix — rows sum to 1",
                       xaxis_title="next state", yaxis_title="current state",
                       margin=dict(l=10, r=10, t=44, b=10))
    _fig.update_yaxes(autorange="reversed")
    mo.vstack([cage_pick, _fig,
               mo.md(f"*The diagonal ({np.round(np.diag(_T),2).tolist()}) is the largest part of each "
                     f"row: behavior **stays in its current state** far more than chance (which for "
                     f"three equally likely states would be 1/3 ≈ 0.33). That stickiness is the memory "
                     f"we will test against a null.*")])
    return


# ============================================================ Stationary distribution by simulation
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## A.4 · The stationary distribution, by simulation

        The stationary distribution answers: *in the long run, what fraction of time is spent in each
        state?* There is a direct way to see it with no linear algebra. **Release a random walker** on
        the transition matrix: start in `rest`, use the current row `T[now]` as the probabilities for
        the next state, take a step, and repeat thousands of times. Tally how often the walker lands in
        each state. That tally *is* the stationary distribution.

        `cu.stationary_dist(T, method="simulate", steps=...)`:

        - **Purpose** — estimate the long-run occupancy of each state.
        - **Input** — the transition matrix `T` and the number of walker steps.
        - **Output** — a vector of fractions, one per state, summing to 1.

        Drag the walk length and watch the estimate converge to the empirical state fractions (open
        diamonds).
        """
    )
    return


@app.cell
def _(mo):
    walk_steps = mo.ui.slider(500, 30000, value=8000, step=500,
                              label="walker steps", debounce=True, full_width=True)
    return (walk_steps,)


@app.cell
def _(STATE_COLORS, STATE_NAMES, cu, go, grammar, mo, np, walk_steps):
    _T = grammar["15"]["T"]
    _pi_sim = cu.stationary_dist(_T, method="simulate", steps=int(walk_steps.value), seed=0)
    _pi_true = grammar["15"]["frac"]
    _fig = go.Figure()
    _fig.add_bar(x=STATE_NAMES, y=_pi_sim, name=f"walker ({int(walk_steps.value)} steps)",
                 marker_color=STATE_COLORS)
    _fig.add_scatter(x=STATE_NAMES, y=_pi_true, name="true long-run fraction", mode="markers",
                     marker=dict(color="black", size=13, symbol="diamond-open", line=dict(width=3)))
    _fig.update_layout(template="plotly_white", height=380,
                       title="Cage-15 stationary distribution — walker vs. truth",
                       yaxis_title="fraction of time", margin=dict(l=10, r=10, t=44, b=10),
                       legend=dict(orientation="h", y=1.12))
    _err = float(np.abs(_pi_sim - _pi_true).max())
    mo.vstack([walk_steps, _fig,
               mo.md(f"*Largest gap between walker and truth: **{_err:.3f}**. With more steps the "
                     f"walker's tally settles onto the true occupancy. A faster exact shortcut using an "
                     f"eigenvector is in the note below.*")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The eigenvector shortcut (optional math)": mo.md(
            r"""
            The walker converges to the vector $\pi$ that is unchanged by one more step:
            $$\pi^\top T = \pi^\top,\qquad \textstyle\sum_i \pi_i = 1.$$
            In words, $\pi$ is the **left eigenvector of $T$ with eigenvalue 1**.
            `cu.stationary_dist(T, method="eig")` returns exactly that: it takes `np.linalg.eig(T.T)`,
            picks the eigenvector whose eigenvalue is closest to 1, and normalizes it to sum to 1. The
            simulation and the eigenvector agree to within a few parts per thousand. We show the walker
            because it *is* the definition; the eigenvector is just the fast route to the same answer.
            (It requires the chain to be irreducible and aperiodic, which ours is.)
            """
        )
    })
    return


# ============================================================ Entropy + stickiness vs null
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## A.5 · Does the grammar beat chance? The shuffle null

        A transition matrix always *looks* structured, even for random data. To show the structure is
        real, we destroy the one thing that carries memory — the **temporal order** — and recompute.
        `cu.shuffle_transition_null` permutes the state sequence many times; each shuffle keeps the same
        overall state frequencies but removes any memory. We compare two statistics to this null:

        - **Transition entropy** (bits): the average uncertainty of the next state. Lower means a more
          predictable grammar; `log₂(3) ≈ 1.58` bits is the memoryless case. The **real** entropy
          should fall well *below* the shuffle null.
        - **Self-transition, or "stickiness"**: the mean of the diagonal, `P(stay in current state)`.
          The real value should sit well *above* the shuffle null (which hovers near chance, ~0.33).

        Below, the shuffle null for all three cages is drawn as a cloud of individual shuffled values
        (violin + points), and each cage's real value is overlaid as a large diamond. Seeing the raw
        null draws, not a single summary bar, is the honest way to judge a gap.
        """
    )
    return


@app.cell
def _(CAGE_COLORS, go, grammar, mo, np):
    from plotly.subplots import make_subplots as _msub
    _cages = ["15", "10", "13"]
    _fig = _msub(rows=1, cols=2, subplot_titles=(
        "Transition entropy (bits) — lower = more memory",
        'Self-transition "stickiness" — higher = stickier'))

    # LEFT: null entropy cloud (all cages pooled) + each cage's observed value as a diamond.
    _null_H_all = np.concatenate([grammar[c]["null_H"] for c in _cages])
    _fig.add_trace(go.Violin(y=_null_H_all, name="shuffle null", line_color="#e45756",
                             fillcolor="rgba(228,87,86,0.18)", opacity=0.6, points="all",
                             pointpos=0, jitter=0.4, marker=dict(size=3, opacity=0.4),
                             box_visible=False, meanline_visible=True, showlegend=False),
                   row=1, col=1)
    for _c in _cages:
        _fig.add_trace(go.Scatter(x=["shuffle null"], y=[grammar[_c]["H"]], mode="markers",
                                  marker=dict(color=CAGE_COLORS[_c], size=14, symbol="diamond",
                                              line=dict(width=1.5, color="black")),
                                  name=f"Cage {_c} observed", showlegend=False,
                                  hovertemplate=f"Cage {_c}: %{{y:.3f}} bits<extra></extra>"),
                       row=1, col=1)
    _fig.add_hline(y=1.585, line=dict(color="#bbb", dash="dot"), row=1, col=1,
                   annotation_text="memoryless log₂3", annotation_position="top left")

    # RIGHT: null stickiness cloud + observed diamonds.
    _null_S_all = np.concatenate([grammar[c]["null_self"] for c in _cages])
    _fig.add_trace(go.Violin(y=_null_S_all, name="shuffle null", line_color="#e45756",
                             fillcolor="rgba(228,87,86,0.18)", opacity=0.6, points="all",
                             pointpos=0, jitter=0.4, marker=dict(size=3, opacity=0.4),
                             box_visible=False, meanline_visible=True, showlegend=False),
                   row=1, col=2)
    for _c in _cages:
        _fig.add_trace(go.Scatter(x=["shuffle null"], y=[grammar[_c]["self"]], mode="markers",
                                  marker=dict(color=CAGE_COLORS[_c], size=14, symbol="diamond",
                                              line=dict(width=1.5, color="black")),
                                  showlegend=False,
                                  hovertemplate=f"Cage {_c}: %{{y:.3f}}<extra></extra>"),
                       row=1, col=2)
    _fig.update_layout(template="plotly_white", height=430, margin=dict(l=10, r=10, t=54, b=10))
    _fig.update_yaxes(range=[0, 1.7], row=1, col=1, title="bits")
    _fig.update_yaxes(range=[0, 1.0], row=1, col=2, title="P(stay)")
    _nH = float(_null_H_all.mean()); _nS = float(_null_S_all.mean())
    mo.vstack([_fig, mo.md(
        f"*Across all three cages: entropy **≈ {np.mean([grammar[c]['H'] for c in _cages]):.2f} bits** "
        f"sits well below the shuffle null **≈ {_nH:.2f}**, and stickiness **≈ "
        f"{np.mean([grammar[c]['self'] for c in _cages]):.2f}** sits well above **≈ {_nS:.2f}**. The "
        f"three colored diamonds land far outside their red null clouds, in the same direction every "
        f"time. The grammar carries real memory.*")])
    return


# ============================================================ Activity clock
@app.cell
def _(cu, t10, t13, t15):
    clocks = {c: cu.activity_by_tod(tr["speed"], tr["tod_hour"], bin_min=30, n_boot=200, seed=0)
              for c, tr in {"15": t15, "10": t10, "13": t13}.items()}
    return (clocks,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## A.6 · The activity clock

        The second question was the daily rhythm. `cu.activity_by_tod`:

        - **Purpose** — describe when across the day the animals move most.
        - **Input** — `speed` and `tod_hour` (the time of day, 0–24 h, for each frame).
        - **Output** — a curve of mean movement speed binned into 30-minute slots, with a bootstrap 95%
          confidence interval within each bin.

        The shaded band marks the **dark, active phase (09:00–21:00)**: this colony runs a *reversed*
        light cycle (lights on 21:00–09:00), so the mice are active during our daytime. The vertical
        dashed line marks our example event's time of day.

        **Read this as a description, not a circadian law.** Three cages cannot support a population
        claim about circadian rhythm; the bootstrap interval reflects within-cage sampling noise, not
        variation between animals. We are describing *these* recordings, honestly.
        """
    )
    return


@app.cell
def _(CAGE_COLORS, clocks, ex_tod, go, grammar, mo):
    _fig = go.Figure()
    _fig.add_vrect(x0=9, x1=21, fillcolor="#000", opacity=0.06, line_width=0,
                   annotation_text="dark / active phase", annotation_position="top left")
    for _c in ["15", "10", "13"]:
        _ck = clocks[_c]; _col = CAGE_COLORS[_c]
        _rgba = "rgba(%d,%d,%d,0.15)" % tuple(int(_col[i:i + 2], 16) for i in (1, 3, 5))
        _fig.add_scatter(x=list(_ck["centers"]) + list(_ck["centers"])[::-1],
                         y=list(_ck["ci_high"]) + list(_ck["ci_low"])[::-1],
                         fill="toself", fillcolor=_rgba, line=dict(color="rgba(0,0,0,0)"),
                         hoverinfo="skip", showlegend=False)
        _fig.add_scatter(x=_ck["centers"], y=_ck["curve"], mode="lines", line=dict(color=_col, width=2),
                         name=f"Cage {_c} ({grammar[_c]['sex']})")
    _fig.add_vline(x=ex_tod, line=dict(color="#d62728", dash="dash"),
                   annotation_text=f"example event @ {ex_tod:.1f}h", annotation_position="top right")
    _fig.update_layout(template="plotly_white", height=430,
                       title="Activity clock — mean speed by time of day (95% bootstrap CI)",
                       xaxis_title="time of day (h · reversed cycle)", yaxis_title="mean speed (px/s)",
                       margin=dict(l=10, r=10, t=44, b=10), legend=dict(orientation="h", y=1.12))
    _fig.update_xaxes(range=[0, 24], dtick=3)
    mo.vstack([_fig, mo.md(
        "*Activity concentrates in the dark, active window, and the example event lands squarely inside "
        "it — consistent across the three cages. With n ≈ 3 this is a description, not an inference "
        "about the population.*")])
    return


# ============================================================ Seeing time structure: GIFs
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## A.7 · Seeing structure in time

        The grammar and clock are summaries. It helps to *see* what "structure in time" looks like
        inside single events. Two illustrations, each rendered on demand:

        1. **Fast, jittery motion vs. smooth, slow motion.** For each event we can measure the dominant
           frequency of the approacher's movement (the spectral centroid of its speed). High-frequency
           events look twitchy and quick; low-frequency events glide. This is the per-event echo of the
           rhythm we saw across the day.
        2. **A coordinated pair vs. an independent pair.** For each event we can cross-correlate the two
           mice's speeds over small time lags. A strongly correlated pair moves *together* — starting
           and stopping in lockstep — the signature of an interaction. A weakly correlated pair moves
           independently, as if sharing a cage but not a moment.

        Rendering skeleton GIFs is the slow part, so each grid is behind a button.
        """
    )
    return


@app.cell
def _(mo):
    freq_btn = mo.ui.run_button(label="▶ Render fast/jittery vs smooth/slow events")
    coord_btn = mo.ui.run_button(label="▶ Render a coordinated pair vs an independent pair")
    return coord_btn, freq_btn


@app.cell
def _(cu, ev, freq_btn, mo, np):
    # Indices pinned from an offline FFT spectral-centroid ranking of approacher speed. HIGH = twitchy
    # (centroid ~13-14 Hz), LOW = smooth (centroid ~2.7 Hz). We only RENDER here; nothing is computed.
    if not freq_btn.value:
        _out = mo.md("*Click the button above to render the fast vs. slow examples.*")
    else:
        _hi = [75, 912, 1735, 6, 1428]
        _lo = [889, 1498, 1143, 39, 710]
        _hi_grid = cu.grid_gif_bytes([(ev["kp"][i], ev["ranks"][i], int(ev["contact_rel"][i]))
                                      for i in _hi], ncols=5, cell=130)
        _lo_grid = cu.grid_gif_bytes([(ev["kp"][i], ev["ranks"][i], int(ev["contact_rel"][i]))
                                      for i in _lo], ncols=5, cell=130)
        _out = mo.md(
            f"""
            **Fast, high-frequency (jittery) events** — dominant motion frequency ≈ 13–14 Hz:<br>
            {cu.gif_img_html(_hi_grid, width=680)}

            **Smooth, low-frequency events** — dominant motion frequency ≈ 2.7 Hz:<br>
            {cu.gif_img_html(_lo_grid, width=680)}

            *Same skeletons, same rank coloring — only the tempo of motion differs. The 19 features
            include speed means and maxima that capture exactly this distinction.*
            """
        )
    _out
    return


@app.cell
def _(coord_btn, cu, ev, mo):
    # Indices pinned from an offline approacher<->approachee speed cross-correlation (peak |r| over
    # +/-15-frame lags). STRONG = peak r ~0.81-0.85 (moving together); WEAK = peak r ~-0.2 (independent).
    if not coord_btn.value:
        _out = mo.md("*Click the button above to render the coordinated vs. independent pairs.*")
    else:
        _strong = [1097, 667, 43, 2008, 649]
        _weak = [1514, 925, 979, 1933, 1972]
        _s_grid = cu.grid_gif_bytes([(ev["kp"][i], ev["ranks"][i], int(ev["contact_rel"][i]))
                                     for i in _strong], ncols=5, cell=130)
        _w_grid = cu.grid_gif_bytes([(ev["kp"][i], ev["ranks"][i], int(ev["contact_rel"][i]))
                                     for i in _weak], ncols=5, cell=130)
        _out = mo.md(
            f"""
            **Strongly coordinated pairs** — the two mice's speeds correlate at peak r ≈ 0.81–0.85; they
            start and stop together:<br>
            {cu.gif_img_html(_s_grid, width=680)}

            **Weakly / independently moving pairs** — peak r ≈ −0.2; the mice move on their own
            schedules:<br>
            {cu.gif_img_html(_w_grid, width=680)}

            *Coordination in time is one of the clearest fingerprints of a genuine social interaction.
            The `closing_speed` and facing features from the earlier notebooks encode this.*
            """
        )
    _out
    return


# ============================================================ Exercise A: build transition matrix
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Exercise A — build the transition matrix by counting

        **Python skill practiced:** turning a *loop* into its *vectorized* form. Counting pairs one at a
        time in a Python loop is the obvious way; here you write the one-line vectorized equivalent that
        `cu.transition_matrix` actually uses.

        **What you have**

        - `seq15` — the `(T,)` contiguous cage-15 state sequence (0 rest · 1 locomote · 2 huddle).
        - `np.add.at(M, (rows, cols), 1)` — a "scatter-add": it adds 1 to `M[rows[k], cols[k]]` for
          every index `k`, all at once, with no Python loop. This is the vectorized counter.

        **The idea.** A transition is a pair `(state now, state next)` =
        `(seq15[t], seq15[t+1])`. Count how many times each pair occurs, then divide each row by its
        total so the row gives probabilities.

        **Your task (one line to fill in).** In the cell below, replace `____` with
        `np.add.at(M, (seq[:-1], seq[1:]), 1)`. The two lines below it already row-normalize for you.
        When you run it, the cell plots your matrix as a heatmap. **You should see a 3×3 grid with a
        strong diagonal (each diagonal cell around 0.8) and small off-diagonal values** — the same
        shape as the cage-15 matrix in A.3. If your diagonal is not dominant, your counting line is
        wrong.
        """
    )
    return


@app.cell
def _(np, t15):
    # ---- YOUR TURN (Exercise A) -----------------------------------------------------------------
    # Build a row-stochastic transition matrix by COUNTING consecutive state pairs.
    seq15 = t15["state_seq"]            # (T,) int, values in {0,1,2} — the contiguous state ribbon
    K_states = 3                        # rest, locomote, huddle

    def student_tmat(seq, K):
        M = np.zeros((K, K))            # empty count table, one row/col per state
        # TODO — replace the line below with the vectorized scatter-add:
        #        np.add.at(M, (seq[:-1], seq[1:]), 1)
        #   * seq[:-1] are the "now" states (every frame except the last).
        #   * seq[1:]  are the "next" states (every frame except the first), aligned frame-by-frame.
        #   Why it matters: counting the (now, next) pairs IS estimating the Markov chain. A plain
        #   `M[seq[:-1], seq[1:]] += 1` would silently DROP repeated coordinates and undercount the
        #   sticky diagonal; np.add.at accumulates every repeat correctly.
        np.add.at(M, (seq[:-1], seq[1:]), 1)     # <-- the ____ line (already filled with the answer)
        row = M.sum(1, keepdims=True)            # total frames spent in each "now" state
        return np.divide(M, row, out=np.zeros_like(M), where=row > 0)   # -> P(next | now)

    T_student = student_tmat(seq15, K_states)
    return T_student, seq15


@app.cell
def _(STATE_NAMES, T_student, go, mo, np):
    _fig = go.Figure(go.Heatmap(
        z=T_student, x=STATE_NAMES, y=STATE_NAMES, colorscale="Blues", zmin=0, zmax=1,
        text=np.round(T_student, 2), texttemplate="%{text}", textfont=dict(size=15),
        colorbar=dict(title="P(next|now)")))
    _fig.update_layout(template="plotly_white", height=380,
                       title="Your counted transition matrix — expect a strong diagonal",
                       xaxis_title="next state", yaxis_title="current state",
                       margin=dict(l=10, r=10, t=44, b=10))
    _fig.update_yaxes(autorange="reversed")
    mo.vstack([_fig, mo.md(
        f"*Your diagonal: {np.round(np.diag(T_student), 2).tolist()}. Each value should be near 0.8 — "
        f"behavior stays put. Compare this shape to the cage-15 matrix in A.3.*")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Reveal solution (Exercise A)": mo.md(
            r"""
            ```python
            def student_tmat(seq, K):
                M = np.zeros((K, K))
                np.add.at(M, (seq[:-1], seq[1:]), 1)   # count now->next pairs, vectorized
                row = M.sum(1, keepdims=True)          # frames spent in each 'now' state
                return np.divide(M, row, out=np.zeros_like(M), where=row > 0)  # -> P(next|now)
            ```
            `np.add.at` is an un-buffered scatter-add: it adds 1 at every `(now, next)` coordinate and
            handles repeats correctly. Dividing each row by its total turns counts into conditional
            probabilities. This is exactly what `cu.transition_matrix` computes — a loop over
            `len(seq)-1` pairs collapsed into one vectorized call.
            """
        )
    })
    return


@app.cell
def _(T_student, cu, grammar, np, seq15):
    # ---- graded self-check: exact match + robust gap vs the shuffle null ----
    _T_ref = cu.transition_matrix(seq15, 3)
    tier1_ok = bool(np.allclose(T_student, _T_ref, atol=1e-9))

    H_obs = float(cu.transition_entropy(T_student))
    self_obs = float(np.mean(np.diag(T_student)))
    _null_H = grammar["15"]["null_H"]
    _null_self = grammar["15"]["null_self"]
    entropy_gap = float(_null_H.mean() - H_obs)     # expect ~0.74 bits (real is far BELOW the null)
    self_gap = float(self_obs - _null_self.mean())  # expect ~0.50   (real is far ABOVE the null)
    tier2_ok = bool((H_obs < _null_H.min()) and (self_obs > _null_self.max()))
    # tolerance bands pinned from the bundle (cam15): H≈0.765, self≈0.832, null_H≈1.505
    bands_ok = bool((0.68 <= H_obs <= 0.86) and (0.79 <= self_obs <= 0.87)
                    and (entropy_gap >= 0.40) and (self_gap >= 0.40))
    return H_obs, bands_ok, entropy_gap, self_gap, self_obs, tier1_ok, tier2_ok


@app.cell(hide_code=True)
def _(H_obs, bands_ok, entropy_gap, mo, self_gap, self_obs, tier1_ok, tier2_ok):
    _pass = tier1_ok and tier2_ok and bands_ok
    _bg, _bd, _verdict = (
        ("rgba(40,170,80,0.12)", "#28aa50", "PASS")
        if _pass else ("rgba(228,87,86,0.12)", "#e45756", "CHECK YOUR CODE"))
    mo.md(
        f"""
        <div style="border:2px solid {_bd}; border-radius:10px; padding:12px 16px; background:{_bg};">
        <b>Self-check — {_verdict}</b><br>
        <b>Counting</b> — your matrix matches <code>cu.transition_matrix</code>: <b>{tier1_ok}</b>.<br>
        <b>Beats the null</b> — observed entropy <b>{H_obs:.3f} bits</b> is below the shuffle null by
        <b>{entropy_gap:.2f} bits</b>; stickiness <b>{self_obs:.3f}</b> is above the null by
        <b>{self_gap:.2f}</b>. Robust gap in both directions: <b>{tier2_ok}</b>; within the pinned
        tolerance bands: <b>{bands_ok}</b>.<br>
        <b>Conclusion:</b> the grammar carries <b>real memory</b> — the observed entropy and stickiness
        beat a time-shuffled null in a direction that holds across all three cages. We grade the
        <i>gap versus the null</i>, never a single noisy number.
        </div>
        """
    )
    return


# ============================================================ Part A conclusion
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Part A — the answer

        **Yes, behavior has structure in time.** On un-manipulated animals, the state sequence is
        *sticky* — behavior dwells in its current state far more than a memoryless process would — and
        that memory beats a time-shuffled null in the same direction across three cages. Activity also
        follows a clear daily rhythm, concentrated in the dark active phase. We now have a baseline
        grammar and clock: the prerequisite for ever claiming a manipulation *changed* how behavior
        moves through time.

        That answers the "does it have structure" question. But describing structure is not the same as
        being able to **read a specific behavior out** of the data automatically. That is Part B.
        """
    )
    return


# ============================================================================================
# ============================  PART B — CAN WE PREDICT A BEHAVIOR FROM FEATURES?  =============
# ============================================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # Part B · Can we predict a behavior from the features?

        ## Why this matters

        We have been hand-scoring aggression. That does not scale, and it is subjective. If the 19
        features carry the signal, a model should read aggression out of them automatically. Building
        that model — a **decoder** — is the whole point of the feature pipeline: an objective,
        reproducible readout of behavior that a human does not have to score frame by frame. But a
        decoder is only worth trusting if it works on data it never learned from. Most of Part B is
        about testing it honestly.

        ## Terms, defined before we use them

        - **Classifier / decoder.** A function that takes an event's measurements (here, the 19
          features) and outputs a guess about a category — for us, *aggression* vs *not aggression*.
          "Decoder" is the same idea from neuroscience, where the input is neural activity; we use the
          words interchangeably.
        - **Training data vs held-out data.** We *train* (fit) the decoder on events whose correct
          answer we already know. To test it fairly, we apply it to *held-out* events it never saw
          during training. Scoring a model on its own training data flatters it.
        - **Cross-validation.** A way to estimate held-out performance when data are limited: split the
          data into k parts, train on k−1, test on the part left out, and repeat until every part has
          been tested once. We use 5-fold cross-validation (k = 5).
        - **Probability score & threshold.** Rather than a hard yes/no, the decoder outputs a number
          between 0 and 1 — its estimated probability of aggression. A *threshold* turns that score into
          a decision (e.g. "call it aggression if the score ≥ 0.5").
        - **ROC-AUC (AUROC).** A single number for how well the scores separate the two classes across
          *every* threshold. 1.0 is perfect ranking; 0.5 is chance. It does not depend on the threshold
          you pick.
        - **Precision & recall.** At a fixed threshold: *precision* is, of the events called aggression,
          the fraction that really were. *Recall* is, of the events that really were aggression, the
          fraction caught. Raising the threshold usually raises precision but lowers recall.
        """
    )
    return


# ============================================================ Core on-load decoder (logistic)
@app.cell
def _(X, Xh, np):
    # The decoder we pin all our numbers to: median-impute -> standardize -> LOGISTIC REGRESSION.
    # Logistic regression is a linear classifier; it is fast, interpretable, and — as we will see —
    # essentially as good as a neural network on this problem, which tells us the classes are close to
    # linearly separable in feature space. This fits in ~1s and renders on load.
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression

    def make_lr():
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=2000))])

    return Pipeline, SimpleImputer, StandardScaler, make_lr


@app.cell
def _(X, Xh, make_lr, y):
    model_lr = make_lr()
    model_lr.fit(X, y)                                 # trained on all 14 training cages
    s_ho = model_lr.predict_proba(Xh)[:, 1]            # P(aggression) on never-seen cam16
    s_tr = model_lr.predict_proba(X)[:, 1]             # in-sample scores (for the example-event demo)
    return model_lr, s_ho, s_tr


@app.cell
def _(cu, s_ho, yh):
    res_ho = cu.eval_binary(yh, s_ho)                  # {roc_auc, avg_precision, confusion, ...} on cam16
    return (res_ho,)


@app.cell(hide_code=True)
def _(Xh, mo, yh):
    _n, _npos = len(yh), int(yh.sum())
    mo.md(
        f"""
        ### The held-out cage, and why it is held out

        <div style="border:2px solid #2ca02c;border-radius:10px;padding:14px 18px;
        background:linear-gradient(90deg,#f0fff4,#ffffff)">
        One cage — <b>cam16</b> — was set aside from the very beginning. Nothing about it was used when
        we designed the features, ran the PCA, built the map, or labeled clusters. Because the decoder
        never saw cam16 during any earlier step, its score there is an honest estimate of performance on
        genuinely new data. cam16 has <b>{_n} events</b> ({_npos} aggression, base rate
        {_npos/_n:.3f}) with <b>{Xh.shape[1]} features</b> each. It is an <b>all-female</b> cage, so it
        also serves as a cross-sex check.
        </div>
        """
    )
    return


# ============================================================ ROC / PR on cam16
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## B.1 · The held-out result on cam16

        The test runs automatically: train on the 14 training cages, evaluate on cam16. The **ROC**
        curve plots the true-positive rate against the false-positive rate as the threshold varies — a
        curve hugging the top-left corner is good, the diagonal is chance, and the area under it is the
        ROC-AUC. The **precision–recall** curve is its companion, more informative when the positive
        class is rare.
        """
    )
    return


@app.cell
def _(cu, res_ho, s_ho, yh):
    roc_fig = cu.roc_pr_fig(yh, s_ho)
    roc_fig.update_layout(title=f"cam16 held-out · ROC-AUC = {res_ho.get('roc_auc', float('nan')):.3f}")
    roc_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The **confusion matrix** (left, below) counts the four outcomes at a 0.5 threshold: correct
        rejections and detections on the diagonal, false positives and misses off it. The
        **calibration curve** (right) checks whether the probabilities mean what they say: for events
        scored around 0.7, did roughly 70% actually turn out to be aggression? A well-calibrated decoder
        follows the dotted diagonal.
        """
    )
    return


@app.cell
def _(cu, go, mo, np, res_ho, s_ho, yh):
    _cm = np.array(res_ho["confusion"])
    _fig1 = go.Figure(go.Heatmap(z=_cm, x=["pred: not", "pred: agg"], y=["true: not", "true: agg"],
                                 colorscale="Blues", showscale=False,
                                 text=_cm, texttemplate="%{text}", textfont={"size": 18}))
    _fig1.update_yaxes(autorange="reversed")
    _fig1.update_layout(template="plotly_white", height=320, title="Confusion @ 0.5",
                        margin=dict(l=10, r=10, t=40, b=10))
    _frac, _mean = cu.calibration_curve(yh, s_ho, n_bins=8)
    _fig2 = go.Figure()
    _fig2.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#bbb"),
                      showlegend=False)
    _fig2.add_scatter(x=_mean, y=_frac, mode="lines+markers", line=dict(color="#4c78a8"),
                      showlegend=False)
    _fig2.update_layout(template="plotly_white", height=320, title="Calibration (reliability curve)",
                        xaxis_title="mean predicted P", yaxis_title="observed fraction",
                        margin=dict(l=10, r=10, t=40, b=10))
    mo.hstack([_fig1, _fig2])
    return


# ============================================================ Threshold slider on cam16
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Choosing a decision threshold

        The threshold is a choice, not a fixed default, and it depends on what the readout is for. A
        false "attack" call inflates the apparent amount of aggression; a missed attack hides it. Slide
        the threshold and read **precision and recall on cam16** at each setting — there is no threshold
        that makes both errors zero.
        """
    )
    return


@app.cell
def _(mo):
    thr_slider = mo.ui.slider(0.05, 0.95, value=0.5, step=0.05, label="decision threshold (cam16)",
                              debounce=True, full_width=True)
    return (thr_slider,)


@app.cell
def _(go, mo, np, s_ho, thr_slider, yh):
    _grid = np.linspace(0.05, 0.95, 19)
    def _pr(t):
        _pred = (s_ho >= t).astype(int)
        _tp = int(((_pred == 1) & (yh == 1)).sum()); _fp = int(((_pred == 1) & (yh == 0)).sum())
        _fn = int(((_pred == 0) & (yh == 1)).sum())
        _p = _tp / (_tp + _fp) if (_tp + _fp) else 0.0
        _r = _tp / (_tp + _fn) if (_tp + _fn) else 0.0
        return _p, _r
    _P = [_pr(t)[0] for t in _grid]; _R = [_pr(t)[1] for t in _grid]
    _p_here, _r_here = _pr(thr_slider.value)
    _fig = go.Figure()
    _fig.add_scatter(x=_grid, y=_P, mode="lines", name="precision", line=dict(color="#4c78a8", width=3))
    _fig.add_scatter(x=_grid, y=_R, mode="lines", name="recall", line=dict(color="#e45756", width=3))
    _fig.add_vline(x=thr_slider.value, line=dict(color="#333", dash="dash"))
    _fig.update_layout(template="plotly_white", height=340,
                       title=f"@ threshold {thr_slider.value:.2f}:  precision {_p_here:.2f} · recall {_r_here:.2f}",
                       xaxis_title="threshold", yaxis_title="value", margin=dict(l=10, r=10, t=50, b=10))
    _fig.update_xaxes(showgrid=False)
    mo.vstack([thr_slider, _fig])
    return


# ============================================================ Logistic vs MLP + feature sets
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## B.2 · Which model, and which features?

        Two design choices deserve a check, both using 5-fold cross-validation *within the training
        cages* so nothing touches cam16.

        - **Model.** Is the linear logistic regression enough, or does a small **multi-layer perceptron
          (MLP)** — a neural network with a couple of hidden layers, which can carve nonlinear
          boundaries — do meaningfully better? `cu.make_mlp()` builds that MLP pipeline.
        - **Features.** Do we gain by feeding the decoder the raw **19 features**, the **PCA scores**
          from the dimensionality-reduction notebook, or the **19 features plus a one-hot code for
          cluster membership** from the map?

        Cross-validation gives five held-out AUROC values per configuration. Instead of a bar of means,
        we plot **all five fold values** as points so the spread is visible. This refits many models, so
        it runs on a button.
        """
    )
    return


@app.cell
def _(mo):
    featureset_btn = mo.ui.run_button(label="▶ Run the model × feature-set comparison (5-fold CV)")
    return (featureset_btn,)


@app.cell
def _(X, cu, der, featureset_btn, make_lr, mo, np, sweep, y):
    if not featureset_btn.value:
        _out = mo.md("*Click to compare logistic vs MLP across three feature sets by 5-fold "
                     "cross-validated AUROC (each dot = one fold).*")
    else:
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        # Cluster one-hot from the canonical UMAP-sweep labels (includes noise label -1).
        _lab = sweep["default_labels"].astype(int)
        _oh = np.zeros((len(_lab), int(_lab.max()) + 2))
        _oh[np.arange(len(_lab)), _lab + 1] = 1.0
        _sets = {
            "PCA scores": der["pca_scores"],
            "19 features": X,
            "19 feats + clusters": np.hstack([X, _oh]),
        }
        _skf = StratifiedKFold(5, shuffle=True, random_state=0)
        _vals, _groups = [], []
        _summary = []
        for _nm, _feat in _sets.items():
            for _mdl_nm, _mk in [("logistic", make_lr), ("MLP", cu.make_mlp)]:
                _sc = cross_val_score(_mk(), _feat, y, cv=_skf, scoring="roc_auc")
                _vals.extend(_sc.tolist())
                _groups.extend([f"{_nm}\n{_mdl_nm}"] * len(_sc))
                _summary.append(f"{_nm} · {_mdl_nm}: {_sc.mean():.3f} ± {_sc.std():.3f}")
        _order = [f"{n}\n{m}" for n in _sets for m in ("logistic", "MLP")]
        _fig = cu.strip_points_fig(np.array(_vals), np.array(_groups), group_order=_order,
                                   ylabel="5-fold CV AUROC", point_size=9, jitter=0.06,
                                   title="Model × feature-set comparison (each dot = one CV fold)",
                                   height=430)
        _fig.add_hline(y=0.851, line=dict(color="#888", dash="dot"),
                       annotation_text="pinned logistic CV ≈ 0.851")
        _fig.update_yaxes(range=[0.78, 0.90])
        _out = mo.vstack([_fig, mo.md("*" + " · ".join(_summary) + "*")])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Two results are worth stating plainly. First, the **MLP barely beats the linear logistic
        regression** — this problem is close to linearly separable in feature space, so the extra
        nonlinearity buys little. Second, **adding the cluster one-hots barely helps**: the clusters are
        a coarser summary of the same 19 features, not new information. That is the whole point of the
        feature pipeline — the useful signal was already in the features. We keep the simpler logistic
        decoder from here on.
        """
    )
    return


# ============================================================ LOCO — the honest generalization test
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## B.3 · The honest test: leave one cohort out

        Cross-validation within the training cages, and even the single held-out cam16, share one
        weakness: cam16 is one cage from **one** of our two food-deprivation cohorts. A decoder can
        quietly learn a *cohort-specific* quirk — a camera angle, a lighting difference, a tracking
        idiosyncrasy — that happens to correlate with aggression in that cohort but would not transfer.

        Because the dataset now has **two independent cohorts** (recorded weeks apart, cohort-unique
        cages), we can run the strongest generalization test available short of a new experiment:
        **leave-one-cohort-out (LOCO)**. Train on one entire cohort, test on the *other* — a genuine
        cross-dataset test. We do it both ways.

        - **Purpose** — measure whether the decoder transfers to a whole second experiment.
        - **Method** — split rows by `cohort`; fit on one, score the other; report ROC-AUC.
        - **Why it is the honest number** — the test cohort shares no cage, no session, and no animal
          with the training cohort. If performance holds, the decoder is reading *behavior*, not the
          fingerprint of one dataset.

        Below, the within-cohort 5-fold CV folds, the single cam16 held-out value, and the two LOCO
        values are shown as individual points, so you can compare them directly.
        """
    )
    return


@app.cell
def _(X, cohort, cu, make_lr, np, res_ho, y):
    # Leave-one-cohort-out: train on one date-tag, test the other. Two logistic fits, fast on load.
    from sklearn.model_selection import cross_val_score as _cvs, StratifiedKFold as _SKF
    from sklearn.metrics import roc_auc_score as _auc

    _cv_folds = _cvs(make_lr(), X, y, cv=_SKF(5, shuffle=True, random_state=0), scoring="roc_auc")
    _tags = sorted(set(cohort.tolist()))
    _loco = {}
    for _test in _tags:
        _tr = cohort != _test
        _te = cohort == _test
        _m = make_lr(); _m.fit(X[_tr], y[_tr])
        _loco[_test] = float(_auc(y[_te], _m.predict_proba(X[_te])[:, 1]))
    loco_scores = _loco                                   # {'12192025': ~0.859, '20260222': ~0.825}
    cv_folds = np.asarray(_cv_folds)
    cv_mean = float(cv_folds.mean())
    cam16_auc = float(res_ho.get("roc_auc", float("nan")))
    return cam16_auc, cv_folds, cv_mean, loco_scores


@app.cell
def _(cam16_auc, cu, cv_folds, loco_scores, mo, np):
    _tags = sorted(loco_scores)
    _vals = list(cv_folds) + [cam16_auc] + [loco_scores[t] for t in _tags]
    _groups = (["within-cohort\n5-fold CV"] * len(cv_folds)
               + ["cam16 held-out\n(1 cage)"]
               + [f"LOCO → test {t}" for t in _tags])
    _order = ["within-cohort\n5-fold CV", "cam16 held-out\n(1 cage)"] + [f"LOCO → test {t}" for t in _tags]
    _fig = cu.strip_points_fig(np.array(_vals, float), np.array(_groups), group_order=_order,
                               ylabel="ROC-AUC", point_size=12, jitter=0.05,
                               title="Within-cohort CV vs. cam16 vs. leave-one-cohort-out",
                               height=430)
    _fig.add_hline(y=0.5, line=dict(color="#bbb", dash="dot"), annotation_text="chance")
    _fig.update_yaxes(range=[0.45, 0.95])
    _loco_txt = " · ".join(f"test {t}: {loco_scores[t]:.3f}" for t in _tags)
    mo.vstack([_fig, mo.md(
        f"*Within-cohort 5-fold CV ≈ **{cv_folds.mean():.3f}**; cam16 held-out ≈ **{cam16_auc:.3f}**; "
        f"leave-one-cohort-out ({_loco_txt}). The LOCO scores drop only modestly below the "
        f"within-cohort CV and stay far above chance. The decoder is reading aggression, not a "
        f"cohort fingerprint — it transfers to a second experiment it never saw.*")])
    return


# ============================================================ Permutation importance
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## B.4 · What the decoder relies on

        A decoder that generalizes is more trustworthy if it leans on features that make biological
        sense. `permutation_importance` measures that.

        - **Purpose** — identify the features the decoder actually reads.
        - **Method** — shuffle one feature's values across events (breaking its link to the label) and
          measure how far cam16 AUROC drops; a large drop means the feature was important.
        - **Inputs** — the fitted model, `Xh`, `yh`.
        - **Output** — a mean drop (with spread over repeats) per feature.

        It re-scores many times, so it runs on a button.
        """
    )
    return


@app.cell
def _(mo):
    permimp_btn = mo.ui.run_button(label="▶ Compute permutation importance on cam16")
    return (permimp_btn,)


@app.cell
def _(Xh, cu, go, model_lr, np, permimp_btn, yh):
    if not permimp_btn.value:
        _out = go.Figure().update_layout(
            template="plotly_white", height=120,
            title="Click the button above to see which features the decoder relies on",
            margin=dict(l=10, r=10, t=40, b=10))
    else:
        from sklearn.inspection import permutation_importance
        _r = permutation_importance(model_lr, Xh, yh, n_repeats=15, random_state=0, scoring="roc_auc")
        _order = np.argsort(_r.importances_mean)
        _fig = go.Figure(go.Bar(x=_r.importances_mean[_order], y=[cu.FEATURE_NAMES[i] for i in _order],
                                orientation="h", marker_color="#4c78a8",
                                error_x=dict(type="data", array=_r.importances_std[_order])))
        _fig.update_layout(template="plotly_white", height=560,
                           title="Permutation importance on cam16 (drop in ROC-AUC when a feature is shuffled)",
                           xaxis_title="AUROC drop when shuffled", margin=dict(l=10, r=10, t=50, b=10))
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The decoder leans on the kinematic and geometric features you would expect for aggression —
        speeds, closing speed, and the nose-to-body proximity terms — rather than on any single
        idiosyncratic channel. That is reassuring: it is reading the *shape of the interaction*.
        """
    )
    return


# ============================================================ Example event as read by the decoder
@app.cell(hide_code=True)
def _(cu, ev, ex_cage, ex_idx, mo, s_tr):
    _gif = cu.event_gif_bytes(ev["kp"][ex_idx], ev["ranks"][ex_idx], int(ev["contact_rel"][ex_idx]),
                              cell=200)
    mo.md(
        f"""
        ## B.5 · Our example event, as the decoder reads it

        Event **#{ex_idx}** (cage {ex_cage}) is the ambiguous near-miss we have followed all along —
        an interaction an earlier model flagged as aggression but a human scored as *not* aggression.
        The decoder now scores it: **P(aggression) = {s_tr[ex_idx]:.2f}**.

        {cu.gif_img_html(_gif, width=200)}

        This is an in-sample check — #{ex_idx} is a training event — so the number is a demonstration,
        not evidence. But it is a nice one: a genuinely borderline clip lands near the decision
        boundary, exactly where the ambiguity you would feel scoring it by hand shows up as an
        intermediate probability. The honest evidence is the cam16 and LOCO numbers above.
        """
    )
    return


# ============================================================ Correct vs mistaken GIFs on cam16
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## B.6 · Correct calls and mistakes on the held-out cage

        A single AUROC hides *what* the decoder gets right and wrong. The clearest way to understand a
        decoder is to watch it work on the held-out cage. Below are four grids of cam16 events (the
        decoder never trained on any of them), grouped by outcome at a 0.5 threshold:

        - **Correct detections (hits)** — truly aggression, scored high.
        - **Correct rejections** — truly not aggression, scored low.
        - **False positives** — the decoder called aggression, but a human did not. These are the
          borderline near-misses.
        - **False negatives (misses)** — genuine aggression the decoder scored low. These are the most
          costly errors for a behavioral readout, because they hide real events.

        Watch the clips and judge for yourself. The mistakes are not random noise — they are the
        genuinely hard cases, the same ones that made hand-labeling difficult.
        """
    )
    return


@app.cell
def _(cu, ho, mo, s_ho):
    # Pinned cam16 indices (into heldout_events.npz), chosen offline for the logistic decoder used
    # here. HITS = true agg scored ~0.95-0.99; REJECTS = true non scored ~0.00; FALSE POS = pred agg
    # but truly non (~0.76-0.91); FALSE NEG = missed true agg (~0.02-0.03).
    _hits = [404, 123, 210, 124]
    _rejects = [589, 417, 439, 489]
    _false_pos = [637, 176, 641, 204]
    _false_neg = [2, 553, 195, 346]

    def _grid(idx):
        events = [(ho["kp"][i], ho["ranks"][i], int(ho["contact_rel"][i])) for i in idx]
        return cu.gif_img_html(cu.grid_gif_bytes(events, ncols=4, cell=135), width=560)

    def _scores(idx):
        return ", ".join(f"{s_ho[i]:.2f}" for i in idx)

    mo.md(
        f"""
        **Correct detections (hits)** — true aggression, scores {_scores(_hits)}:<br>
        {_grid(_hits)}

        **Correct rejections** — true non-aggression, scores {_scores(_rejects)}:<br>
        {_grid(_rejects)}

        **False positives** — decoder said aggression, human said no; scores {_scores(_false_pos)}:<br>
        {_grid(_false_pos)}

        **False negatives (misses)** — genuine aggression scored low; scores {_scores(_false_neg)}:<br>
        {_grid(_false_neg)}
        """
    )
    return


# ============================================================ Optional: label-noise ceiling
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## B.7 · The ceiling that labels put on any decoder (optional depth)

        A decoder can only be as reliable as the labels it learns from. If about 1 in 8 hand labels is
        wrong at the boundary (and identity from tail marks already carries roughly 16% error), then
        even a perfect model inherits that error as an upper limit. We can show this directly: flip a
        fraction of the *training* labels on purpose and watch the held-out cam16 AUROC fall. This
        refits several models, so it runs on a button.
        """
    )
    return


@app.cell
def _(mo):
    noise_btn = mo.ui.run_button(label="▶ Simulate the label-noise ceiling (refits ~5 models)")
    return (noise_btn,)


@app.cell
def _(X, Xh, cu, go, make_lr, mo, noise_btn, np, y, yh):
    if not noise_btn.value:
        _out = mo.md("*Click to corrupt a fraction of the training labels and watch held-out AUROC "
                     "fall.*")
    else:
        _levels = [0.0, 0.05, 0.10, 0.20, 0.30]
        _aucs = []
        _rng = np.random.RandomState(0)
        for _p in _levels:
            _yn = y.copy()
            _flip = _rng.rand(len(_yn)) < _p
            _yn[_flip] = 1 - _yn[_flip]
            _m = make_lr(); _m.fit(X, _yn)
            _aucs.append(cu.eval_binary(yh, _m.predict_proba(Xh)[:, 1])["roc_auc"])
        _fig = go.Figure(go.Scatter(x=_levels, y=_aucs, mode="lines+markers",
                                    line=dict(color="#e45756", width=3), marker=dict(size=9)))
        _fig.update_layout(template="plotly_white", height=340,
                           title="Held-out cam16 AUROC vs fraction of corrupted training labels",
                           xaxis_title="fraction of labels flipped", yaxis_title="cam16 ROC-AUC",
                           margin=dict(l=10, r=10, t=50, b=10))
        _out = _fig
    _out
    return


# ============================================================ Exercise B: build + check the decoder
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## Exercise B — build and check the decoder

        **Python skill practiced:** calling a machine-learning library — the scikit-learn
        `fit` / `predict_proba` pattern and `cross_val_score`. This is the top of the skill ramp: you
        drive a real model, not a hand-rolled loop.

        **Goal.** Reproduce the three headline numbers of Part B: (1) 5-fold cross-validated AUROC on
        the training cages, (2) the held-out cam16 AUROC, and (3) leave-one-cohort-out AUROC.

        **Toolbox**
        - `make_lr()` → an unfitted logistic pipeline. Call `.fit(X, y)`, then
          `.predict_proba(X)[:, 1]` for one aggression probability per event.
        - `cu.eval_binary(y_true, y_score)["roc_auc"]` → the ROC-AUC. It takes the **score vector**, not
          the model.
        - `cross_val_score(estimator, X, y, cv=..., scoring="roc_auc")` → one AUROC per fold.
        - `X, y` = training features/labels; `Xh, yh` = cam16; `cohort` = the cohort tag per training
          row.

        **Fill in the three lines marked `# <<< EDIT`.** Each already contains working code so the
        notebook runs; rewrite it yourself from the guidance in the comment, then compare against the
        self-check. **Expected:** the self-check reports 5-fold CV ≈ **0.851**, cam16 ≈ **0.873**, and
        both LOCO values above **0.80**. If your held-out AUROC comes out near 1.0, you probably scored
        `X` instead of `Xh` — that is testing on the training data.
        """
    )
    return


@app.cell
def _(X, Xh, cohort, cu, make_lr, np, y, yh):
    # ---- YOUR TURN (Exercise B): fill in the three lines marked  # <<< EDIT --------------------
    from sklearn.model_selection import cross_val_score as _cross_val_score, StratifiedKFold as _SKF
    from sklearn.metrics import roc_auc_score as _roc_auc_score

    # (1) 5-fold cross-validation on the training cages.
    #     EDIT: pass scoring="roc_auc" so each fold is scored by ROC-AUC (not accuracy). Why it
    #     matters: with a 0.32 base rate, accuracy is a misleading score — a model that never predicts
    #     aggression is already 68% "accurate". ROC-AUC is threshold-free and base-rate-robust.
    _cv = _cross_val_score(make_lr(), X, y, cv=_SKF(5, shuffle=True, random_state=0),
                           scoring="roc_auc")            # <<< EDIT: scoring="roc_auc"
    ex_cv_mean = float(_cv.mean())

    # (2) Held-out cam16 decode: train on ALL training data, then score the HELD-OUT features.
    _m = make_lr(); _m.fit(X, y)
    ex_heldout_auc = cu.eval_binary(yh, _m.predict_proba(Xh)[:, 1])["roc_auc"]   # <<< EDIT: score Xh, not X

    # (3) Leave-one-cohort-out: train on every cohort EXCEPT the test tag, then score the test cohort.
    #     EDIT: the training mask must be the ROWS NOT IN the test cohort -> (cohort != _test). Why it
    #     matters: if you accidentally train on (cohort == _test) you train and test on the SAME
    #     cohort, which defeats the entire point of a cross-dataset generalization test.
    ex_loco = {}
    for _test in sorted(set(cohort.tolist())):
        _tr = cohort != _test                            # <<< EDIT: train on the OTHER cohort
        _ml = make_lr(); _ml.fit(X[_tr], y[_tr])
        ex_loco[_test] = float(_roc_auc_score(y[cohort == _test],
                                              _ml.predict_proba(X[cohort == _test])[:, 1]))
    return ex_cv_mean, ex_heldout_auc, ex_loco


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"Reveal solution (Exercise B)": mo.md(
        r"""
        ```python
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import roc_auc_score

        # (1) within-cohort cross-validation, scored by ROC-AUC
        cv = cross_val_score(make_lr(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                             scoring="roc_auc")

        # (2) held-out cam16 — train on all training data, score Xh (NOT X)
        m = make_lr(); m.fit(X, y)
        heldout_auc = cu.eval_binary(yh, m.predict_proba(Xh)[:, 1])["roc_auc"]

        # (3) leave-one-cohort-out — train on the OTHER cohort (cohort != test)
        loco = {}
        for test in sorted(set(cohort.tolist())):
            tr = cohort != test
            ml = make_lr(); ml.fit(X[tr], y[tr])
            loco[test] = roc_auc_score(y[cohort == test], ml.predict_proba(X[cohort == test])[:, 1])
        ```
        The three edits are: score by `"roc_auc"`, score `Xh` (not `X`) for cam16, and train on the
        *complement* of the test cohort for LOCO. The whole point is that (2) and (3) measure
        generalization to data the model never saw — a single cage, and a whole second experiment.
        """)})
    return


@app.cell
def _(ex_cv_mean, ex_heldout_auc, ex_loco):
    # ---- graded self-check, pinned to the committed bundle ----
    ok_cv = bool(0.82 <= ex_cv_mean <= 0.88)                    # pinned 0.851 ± 0.006
    ok_ho = bool(0.83 <= ex_heldout_auc <= 0.91)               # pinned 0.873
    ok_loco = bool(all(v > 0.80 for v in ex_loco.values()))    # pinned 0.825 / 0.859
    all_ok = ok_cv and ok_ho and ok_loco
    return all_ok, ok_cv, ok_ho, ok_loco


@app.cell(hide_code=True)
def _(all_ok, ex_cv_mean, ex_heldout_auc, ex_loco, mo, ok_cv, ok_ho, ok_loco):
    _color = "#2ca02c" if all_ok else "#e45756"
    _msg = "PASS" if all_ok else "check your fit and splits"
    _loco_txt = ", ".join(f"{t}={v:.3f}" for t, v in sorted(ex_loco.items()))
    mo.md(
        f"""
        <div style="border:2px solid {_color};border-radius:10px;padding:12px 16px;background:#fafafa">
        <b>Self-check — {_msg}</b><br>
        5-fold CV AUROC = <b>{ex_cv_mean:.3f}</b> — in band [0.82, 0.88]? <b>{ok_cv}</b><br>
        cam16 held-out AUROC = <b>{ex_heldout_auc:.3f}</b> — in band [0.83, 0.91]? <b>{ok_ho}</b><br>
        leave-one-cohort-out AUROC = <b>{_loco_txt}</b> — both &gt; 0.80? <b>{ok_loco}</b><br>
        <b>Conclusion:</b> the decoder reproduces the pinned numbers, and — the important part — it
        holds up on a cage it never saw and on a whole second cohort. That is a readout we can trust.
        </div>
        """
    )
    return


# ============================================================ Limits
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## B.8 · What the decoder ignores, and how it can fail

        - **Time.** The decoder reads 19 window-summary features, which collapse the whole 130-frame
          trajectory into means and maxima. A fast feint and a slow approach with the same summary look
          identical to it — the very temporal structure Part A measured is thrown away here. A
          sequence-aware model could use it.
        - **Cage / cohort confounds.** A decoder can accidentally learn the *cage* or *cohort* instead
          of the behavior. Leave-one-cohort-out is the guard against this, but only for the cohorts we
          actually held out.
        - **The label-noise ceiling** is real: roughly 16% identity error plus the boundary ambiguity
          caps accuracy no matter how good the model is.
        - **Threshold sensitivity.** Whenever the readout is a single detected rate, that one number
          depends entirely on the chosen threshold; report the whole precision–recall trade-off, not one
          point on it.
        """
    )
    return


# ============================================================ Close Week 1 -> Week 2
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        # Closing Week 1 — and the question that opens Week 2

        **What we established this week.** Starting from raw SLEAP keypoints, we built a body-frame
        feature representation, reduced it, mapped it, and read biology out of it. This notebook added
        the two capstones:

        - **Behavior has structure in time.** The state grammar is sticky — it carries real memory that
          beats a shuffled null across cages — and activity follows a clear daily rhythm.
        - **Behavior is decodable.** A simple logistic decoder reads aggression from the 19 features and,
          crucially, *generalizes*: 5-fold CV ≈ 0.851, a held-out cage ≈ 0.873, and — the honest
          cross-dataset test — leave-one-cohort-out ≈ 0.825 / 0.859. We can now label behavior
          objectively and reproducibly, at scale, without a human scoring every frame.

        We can **read behavior**. That is the foundation of everything that follows.

        **The next question.** Behavior is produced by a brain. Having built an objective readout of
        *what* an animal does, Week 2 turns to *how the brain produces it*: we move from tracking bodies
        to reading neurons. The same discipline carries over — extract a clean signal, reduce it,
        find structure, and decode it honestly on held-out data — but now the raw data is a microscope
        movie of neural activity rather than a video of moving mice. How do we read the brain that
        produces the behavior we just learned to measure?
        """
    )
    return


if __name__ == "__main__":
    app.run()
