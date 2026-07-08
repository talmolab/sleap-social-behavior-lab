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
        "https://raw.githubusercontent.com/Elmaestrotango/sleap-social-behavior-lab/main")
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # NB02 · The body frame and the 19 features

        ### Where we are in the story

        In **NB01** we answered a first question: *what is the raw signal?* SLEAP gives us, for every
        frame of video, an `(x, y)` pixel location for each of **15 tracked body points** on each mouse.
        We learned to reach into that array with `kp[frame, mouse, node]`, and we saw which body points
        track reliably and which drop out.

        Those coordinates describe **where each mouse is in the arena** and **which way it faces in the
        cage**. That is a problem, because an arena position is not a behavior. This notebook asks the
        next question in the chain:

        > **How do we describe an interaction independent of where in the arena it happens and which way
        > the animals happen to be facing?**

        ### Why this matters

        We study social behavior and its neural basis; neither is treated as primary — the behavior and
        the circuits that produce it are two views of one system. A behavior — an attack, a sniff, a
        chase — is the *same behavior*
        whether it happens in the top-left corner of the cage or the bottom-right, and whether the two
        mice face north or south. Raw pixel coordinates do not capture that: the identical behavior in
        two locations produces completely different numbers, because the numbers describe *where in the
        arena* the mice are, not *what they are doing to each other*.

        The animal does not care about the arena's orientation. The social geometry that matters — who
        is in front of whom, how fast the gap is closing, who turns to face whom — is defined **relative
        to the animals themselves**, not relative to the cage walls. Behavior is, in this precise sense,
        **rotationally invariant**. So before we can compare events, describe them, or later train a
        classifier, we remove two things that are not part of the behavior: **where in the cage** the
        event happened, and **which way the animals were pointing in the arena**. What is left is the
        genuinely social part.

        ### Definitions (read these first)

        - **Keypoint** — a single tracked body point (nose, head, tail-base, …), stored as an `(x, y)`
          pixel location in the camera image. NB01 produced these.
        - **Body frame** (also called **egocentric coordinates**) — coordinates measured relative to one
          animal's own body: put the origin at that animal's tail-base, then rotate so the direction it
          faces points straight up (+y). In a body frame we describe the scene as *"the other mouse is
          ahead of me and slightly to my left"* instead of *"the other mouse is at pixel (812, 344)."*
        - **Invariant** — a number that does **not** change when we move or rotate the whole scene. The
          distance between two mice is invariant; a mouse's absolute pixel position is not.
        - **Feature** — a single interpretable number summarizing an event (a speed, a distance, a
          facing angle). We will build **19** of them.

        ### What we will do (the method)

        1. Take one event — all three mice, every frame — in raw arena coordinates.
        2. **Translate** so the approaching mouse's tail-base sits at the origin, then **rotate** so that
           mouse faces +y. This is the body-centered (egocentric) transform, `cu.allocentricize`.
        3. Summarize the transformed event into **19 interpretable numbers** — speeds, distances, and
           facing angles — that are the same no matter where or which way the event happened.

        **Deliverable of this notebook:** the feature matrix `X (2499, 19)` — one 19-number vector per
        event. Every later notebook reads `X`, not pixels.

        Colors are by rank throughout the course: **Dom = red, Int/Mid = blue, Sub = green** (gray =
        unknown).
        """
    )
    return


@app.cell
def _(ROOT, cu, np):
    # Load the two aligned bundles: the raw events (keypoints + labels) and the precomputed "derived"
    # bundle (the 19 features X, plus per-event metadata: cohort, cage, sex).
    ev = cu.load_events(cu.data_path("data/train_events.npz", ROOT))
    der = cu.load_derived("train", ROOT)

    kp = ev["kp"].astype(np.float32)          # (N, T, 3, 15, 2) raw arena keypoints
    X = der["X"]                              # (N, 19) the features we will build by hand below
    agg = ev["agg_label"].astype(int)         # (N,) 1 = aggression, 0 = not (ground truth)
    ranks = ev["ranks"]                       # (N, 3) rank of [approacher, approachee, bystander]
    contact = ev["contact_rel"].astype(int)   # (N,) frame index where contact begins
    condition = ev["condition"].astype(str)   # (N,) 'pre' | 'dep' | 'post'
    cohort = der["cohort"].astype(str)        # (N,) date-tag of the food-deprivation cohort
    cage = der["cage"].astype(int)            # (N,) cohort-unique cage id
    sex = der["sex"].astype(str)              # (N,) 'M' | 'F' (one sex per cage)
    feat_names = [str(f) for f in cu.FEATURE_NAMES]

    # Our running example (pinned by stable event_key, not a raw integer index, so it survives bundle
    # rebuilds) is cleanly tracked on every frame (all 15 nodes present the whole window), which makes
    # the coordinate transform easy to see. It is a Sub-approaches-Mid approach
    # with a Dom bystander, from a female cage; it happens to be a NON-aggression approach — we use it
    # for the geometry, and pull separate aggression clips later.
    EX = cu.event_index_by_key(ev, "12192025_pre|cam.10.00046-2025-12-18T16|m0-m2|83141")
    ex_hex = tuple(cu.RANK_HEX.get(int(r), cu.RANK_HEX[0]) for r in ranks[EX])
    return (EX, X, agg, cage, cohort, condition, contact, ev, ex_hex,
            feat_names, kp, ranks, sex)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 0. What is in the dataset

        Before any transform, it helps to know what we are holding. The events come from **two
        food-deprivation cohorts** (we refer to them only by their recording date tags, not by project
        name). Each cohort is a set of home-cages recorded across three conditions — **pre**, **dep**
        (food-deprived), and **post**. Cage identity is **cohort-unique**: a cage number never means the
        same cage in two cohorts, so grouping by cage never accidentally mixes cohorts. Each cage holds
        animals of a single sex, so cage is also the natural **unit** for any sex comparison — a point
        that becomes important when we do statistics in NB04.

        The counts below are computed live from the loaded arrays.
        """
    )
    return


@app.cell
def _(agg, cage, cohort, condition, mo, sex):
    import numpy as _np
    def _tally(a):
        vals, cnts = _np.unique(a, return_counts=True)
        return ", ".join(f"`{v}` = {c}" for v, c in zip(vals, cnts))
    _n = len(agg)
    _cages = _np.unique(cage)
    _cage_sex = {int(c): sex[cage == c][0] for c in _cages}
    _males = [c for c in _cages if _cage_sex[c] == "M"]
    _females = [c for c in _cages if _cage_sex[c] == "F"]
    mo.md(
        f"""
        | quantity | value |
        |---|---|
        | events (train) | **{_n}** |
        | aggression base rate | **{agg.mean():.3f}** ({int(agg.sum())} aggression / {_n}) |
        | cohorts | {_tally(cohort)} |
        | conditions | {_tally(condition)} |
        | sex of events | {_tally(sex)} |
        | cages | **{len(_cages)}** total — {len(_males)} male, {len(_females)} female |
        | male cages | {", ".join(str(c) for c in _males)} |
        | female cages | {", ".join(str(c) for c in _females)} |

        One more cage — **Camera 16** (780 events, a single cohort, all female, base rate 0.385) — is
        held out and never inspected here. We set it aside now and only touch it in **NB05**, where it
        measures how well a decoder generalizes to animals it has never seen. Keeping it untouched is
        what makes that later number trustworthy.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And here is one event, so "an event" is concrete rather than abstract. The clip shows all three
        mice on a blank canvas (no video needed — the skeletons are rendered from the keypoints). The
        white arrow points **approacher → approachee**, and the red dot in the corner marks the frame
        where contact begins. Because we learn behavior by seeing it, most methods in this course get
        checked against clips like this one.
        """
    )
    return


@app.cell
def _(EX, cage, cohort, contact, cu, kp, mo, ranks):
    _rn = {0: "unknown", 1: "Dom", 2: "Mid", 3: "Sub"}
    _gif = cu.event_gif_bytes(kp[EX], ranks[EX], contact_rel=int(contact[EX]), cell=220, fps=20)
    mo.vstack([
        mo.Html('<div style="text-align:center">' + cu.gif_img_html(_gif, width=260) + "</div>"),
        mo.md(
            f"""
            <div style="text-align:center;color:#555">
            <b>Example event {EX}</b> — cohort {cohort[EX]}, cage {int(cage[EX])}
            &nbsp;·&nbsp; approacher = <b>{_rn[int(ranks[EX][0])]}</b> (green),
            approachee = <b>{_rn[int(ranks[EX][1])]}</b> (blue),
            bystander = <b>{_rn[int(ranks[EX][2])]}</b> (red)
            </div>
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 1. Rotating a frame by hand

        The whole transform is two moves: **translate** so the approacher's tail-base (node `TTI`) sits
        at the origin, then **rotate** so its heading (tail-base → head) points straight up (+y). The
        second move — rotation — is worth doing by hand once, so it is not a black box.

        Below is one real frame of the example event: all three mice, in raw arena coordinates, centered
        on the field for display. Drag the slider to rotate the **entire field** by an angle β. Watch two
        things:

        - The **numbers on the right** update live: the 2×2 rotation matrix `R(β)`, and one sample
          keypoint's coordinates before and after the rotation.
        - The **shape never distorts.** Rotation is *rigid*: every distance and every angle *between* the
          mice stays exactly the same. Only the orientation of the whole picture on the page changes.
          That rigidity is exactly why the operation is safe to apply to behavior — it changes the frame
          of reference without altering the behavior.
        """
    )
    return


@app.cell
def _(mo):
    toy_angle = mo.ui.slider(-180, 180, value=0, step=5,
                             label="rotation β applied to the whole field (degrees)",
                             debounce=True, full_width=True)
    return (toy_angle,)


@app.cell
def _(EX, contact, cu, ex_hex, go, kp, mo, np, toy_angle):
    # Rotate ONE frame of the whole interaction (all three mice) by the slider angle beta, so the
    # student sees the entire field turn as a rigid body.
    _t = int(contact[EX])
    _field = kp[EX][_t].astype(float)                        # (3, 15, 2) — one frame, three mice
    _pts = _field.reshape(-1, 2)
    _c = np.nanmean(_pts[np.isfinite(_pts).all(1)], axis=0)  # field center, for in-place rotation
    _centered = _field - _c
    _beta = np.deg2rad(toy_angle.value)
    _R = np.array([[np.cos(_beta), -np.sin(_beta)], [np.sin(_beta), np.cos(_beta)]])
    _rot = np.einsum("ij,mnj->mni", _R, _centered)

    _fig = go.Figure()
    for _m in range(3):
        _mk = _rot[_m]
        _ok = np.isfinite(_mk).all(1)
        _ex, _ey = [], []
        for _u, _v in cu.SKELETON_EDGES:
            if _ok[_u] and _ok[_v]:
                _ex += [_mk[_u, 0], _mk[_v, 0], None]
                _ey += [_mk[_u, 1], _mk[_v, 1], None]
        _fig.add_scatter(x=_ex, y=_ey, mode="lines",
                         line=dict(color=ex_hex[_m], width=2), showlegend=False, hoverinfo="skip")
        _fig.add_scatter(x=_mk[_ok, 0], y=_mk[_ok, 1], mode="markers",
                         marker=dict(color=ex_hex[_m], size=6), showlegend=False, hoverinfo="skip")
    _fig.update_xaxes(range=[-260, 260], showgrid=False, zeroline=True)
    _fig.update_yaxes(range=[260, -260], showgrid=False, zeroline=True, scaleanchor="x", scaleratio=1)
    _fig.update_layout(template="plotly_white", height=460, margin=dict(l=10, r=10, t=44, b=10),
                       title=f"whole field rotated by β = {toy_angle.value}°")

    # live numeric readout beside the plot
    _in = _centered[0, cu.HEAD]                              # approacher head, centered (input)
    _out = _rot[0, cu.HEAD]                                  # approacher head, after rotation (output)
    _readout = mo.md(
        f"""
        **Rotation matrix `R(β)`, β = {toy_angle.value}°**

        |  |  |
        |---:|---:|
        | {_R[0, 0]:+.3f} | {_R[0, 1]:+.3f} |
        | {_R[1, 0]:+.3f} | {_R[1, 1]:+.3f} |

        Each keypoint `(x, y)` becomes `R(β)·(x, y)`.

        **Approacher head keypoint**

        before: ({_in[0]:+.1f}, {_in[1]:+.1f})
        after:  &nbsp;({_out[0]:+.1f}, {_out[1]:+.1f})

        Its distance to the origin is unchanged:
        {np.linalg.norm(_in):.1f} → {np.linalg.norm(_out):.1f} px. That is what "rigid" means — every
        point keeps its distance from the center; only its direction turns.
        """
    )
    mo.vstack([toy_angle, mo.hstack([_fig, _readout], widths=[2, 1])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "How the code picks β automatically": mo.md(
            r"""
            We do not have to guess β. The code reads the approacher's heading angle straight off the
            skeleton and rotates by exactly the amount that lands it on +y.

            - **Heading angle** of the approacher:
              $\varphi=\operatorname{atan2}(\text{head}_y-\text{TTI}_y,\ \text{head}_x-\text{TTI}_x)$.
            - **Rotation needed** to send that heading to straight up:
              $\alpha=\tfrac{\pi}{2}-\varphi$.
            - **Rotation matrix:**
              $R(\alpha)=\begin{bmatrix}\cos\alpha & -\sin\alpha\\ \sin\alpha & \cos\alpha\end{bmatrix}$.

            `cu._anchor_transform` returns this $R$ together with the **center** (the approacher's
            tail-base), so the full move is *translate to origin, then rotate*:
            $\mathbf{p}' = R(\alpha)\,(\mathbf{p}-\mathbf{c})$. `cu.allocentricize` then applies that
            same $(\mathbf{c}, R)$ — computed once from the approacher — to **all three** mice, so the
            whole social scene is re-expressed in the approacher's body frame.

            **`cu.allocentricize`** · *purpose:* put an event in the approacher's body frame ·
            *input:* `kp_event` of shape `(T, 3, 15, 2)` in arena pixels ·
            *output:* the same-shaped array, translated and rotated. If the approacher's head or
            tail-base is missing on every frame it cannot find a heading and returns the event
            unchanged (a failure mode we return to at the end).
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 2. Applying the transform to the example event

        Now compare the two coordinate frames side by side. On the **left** is the raw arena view — the
        mice are wherever they happened to be in the cage. On the **right** is the same frame after
        `cu.allocentricize`: the approacher's tail-base is pinned at the origin (black ✕) and the
        approacher's heading is fixed. Everything that still moves on the right is **social geometry** —
        where the other two mice sit *relative to the approacher*. Drag the frame slider and watch the
        approachee close in while the approacher stays put.
        """
    )
    return


@app.cell
def _(EX, contact, kp, mo):
    _T = kp[EX].shape[0]
    ex_frame = mo.ui.slider(0, _T - 1, value=int(contact[EX]), step=1,
                            label="frame (red dot = contact onset)", debounce=True, full_width=True)
    return (ex_frame,)


@app.cell
def _(EX, cu, ex_frame, ex_hex, kp, mo):
    _raw = kp[EX].astype(float)
    _body = cu.allocentricize(_raw)
    _t = ex_frame.value
    _fig_raw = cu.skeleton_fig(_raw[_t], cu.SKELETON_EDGES, colors=ex_hex,
                               title=f"RAW arena — frame {_t}", height=460)
    _fig_body = cu.skeleton_fig(_body[_t], cu.SKELETON_EDGES, colors=ex_hex,
                                title=f"BODY FRAME — frame {_t} (approacher pinned at origin)",
                                height=460)
    # mark the approacher origin on the body-frame panel
    _fig_body.add_scatter(x=[0], y=[0], mode="markers",
                          marker=dict(symbol="x", size=12, color="black"), showlegend=False)
    mo.vstack([ex_frame, mo.hstack([_fig_raw, _fig_body], widths=[1, 1])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 3. What the body frame keeps and what it throws away

        The function is named `allocentricize` in this codebase, and the 19 features are stored under
        that name, so our code matches everyone else's. When we reason about the science, though, we
        describe it plainly for what it is: a **body-centered (egocentric) transform** — the scene
        expressed relative to the approacher's own body.

        The point of the transform is what it *removes*. By centering on the approacher's tail-base and
        rotating its heading, we throw away the approacher's own arena pose: where it stands and which
        way it faces in the cage. Those are not part of the behavior. What survives is the **relative
        configuration** — how far apart the mice are, who faces whom, how the trio is arranged — and that
        relative configuration is identical for the same behavior in any corner of the cage. Section 5
        demonstrates that invariance directly and lets us watch a feature refuse to move while the arena
        spins.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 4. The 19 features, in plain English

        `allocentricize` gives us a body-frame movie of an event; `cu.features_one` collapses that movie
        into **19 numbers**. Each is arena-invariant (Section 5 checks this). The four **kinds** of
        feature are worth noticing, because they behave very differently later:

        - **Kinematics** — how fast each mouse moves and turns (features 0–3, 6, 7, 14).
        - **Posture** — body length, i.e. stretched vs hunched (features 4, 5).
        - **Relative geometry** — distances and facing angles *between* the two mice (features 8–13, 15).
        - **The third mouse** — where the bystander sits and how spread out the trio is (features 16–18).

        | # | name | plain meaning |
        |---|------|----------------|
        | 0 | `appr_speed_mean` | approacher's average body speed (px/frame) |
        | 1 | `appr_speed_max` | approacher's peak speed — a lunge shows up here |
        | 2 | `appe_speed_mean` | approachee's average body speed |
        | 3 | `appe_speed_max` | approachee's peak speed — a flinch or flee spikes this |
        | 4 | `appr_body_len` | approacher nose→tail-base length (stretched vs hunched posture) |
        | 5 | `appe_body_len` | approachee body length |
        | 6 | `appr_angvel` | how fast the approacher turns (heading angular velocity) |
        | 7 | `appe_angvel` | how fast the approachee turns |
        | 8 | `pair_dist_mean` | average distance between the two mice |
        | 9 | `pair_dist_min` | their closest distance during the event |
        | 10 | `appr_nose_to_appe_tti_min` | closest approacher-nose → approachee-rump distance (a rear sniff/attack) |
        | 11 | `appe_nose_to_appr_tti_min` | closest approachee-nose → approacher-rump distance |
        | 12 | `appr_faces_appe` | does the approacher point at the other? facing cosine, +1 = dead-on |
        | 13 | `appe_faces_appr` | does the approachee point back at the approacher? |
        | 14 | `closing_speed` | how fast the gap shrinks (positive = closing in) |
        | 15 | `heading_alignment` | are the two headings parallel (+1) or opposed (−1)? |
        | 16 | `bystander_dist_mean` | average distance to the third (bystander) mouse |
        | 17 | `bystander_dist_min` | closest the bystander gets |
        | 18 | `triangle_area_mean` | spread of the trio (area of the triangle of the three centroids) |

        `cu.features_one` · *purpose:* turn one event into these 19 numbers · *input:* `kp_event` of
        shape `(T, 3, 15, 2)`, mice ordered [approacher, approachee, bystander] · *output:* a length-19
        vector. Several of these are **geometric** and easiest to understand as a picture — so the next
        figure draws them on the example event's body frame.
        """
    )
    return


@app.cell
def _(EX, contact, cu, ex_hex, go, kp, mo, np):
    # A labelled geometry diagram: draw the example event's BODY-FRAME skeletons at contact and mark
    # the geometric features so they stop being names in a table. y is NOT reversed here, so "ahead of
    # the approacher" reads as UP on the page.
    _bf = cu.allocentricize(kp[EX].astype(float))
    _t = int(contact[EX])
    _frame = _bf[_t]                                         # (3, 15, 2) body-frame coords at contact
    _cen = np.array([np.nanmean(_frame[_m, cu.BODY_NODES], axis=0) for _m in range(3)])  # (3, 2)

    _fig = go.Figure()
    # skeletons
    for _m in range(3):
        _mk = _frame[_m]
        _ok = np.isfinite(_mk).all(1)
        _ex, _ey = [], []
        for _u, _v in cu.SKELETON_EDGES:
            if _ok[_u] and _ok[_v]:
                _ex += [_mk[_u, 0], _mk[_v, 0], None]
                _ey += [_mk[_u, 1], _mk[_v, 1], None]
        _fig.add_scatter(x=_ex, y=_ey, mode="lines", line=dict(color=ex_hex[_m], width=2),
                         showlegend=False, hoverinfo="skip")
        _fig.add_scatter(x=_mk[_ok, 0], y=_mk[_ok, 1], mode="markers",
                         marker=dict(color=ex_hex[_m], size=5), showlegend=False, hoverinfo="skip")
    # triangle of the three centroids (feature 18: triangle_area_mean)
    _fig.add_scatter(x=list(_cen[[0, 1, 2, 0], 0]), y=list(_cen[[0, 1, 2, 0], 1]), mode="lines",
                     line=dict(color="#999", width=1, dash="dot"), name="trio triangle (feat 18)")
    # pair distance line (features 8/9)
    _fig.add_scatter(x=[_cen[0, 0], _cen[1, 0]], y=[_cen[0, 1], _cen[1, 1]], mode="lines+markers",
                     line=dict(color="#000", width=2, dash="dash"),
                     marker=dict(size=8, color="#000"), name="pair_dist (feat 8/9)")
    # bystander distance line (features 16/17)
    _fig.add_scatter(x=[_cen[0, 0], _cen[2, 0]], y=[_cen[0, 1], _cen[2, 1]], mode="lines",
                     line=dict(color="#d62728", width=1, dash="dash"),
                     name="bystander_dist (feat 16/17)")
    # heading arrows (features 12/13/15): tail-base -> head, scaled up for visibility
    for _m, _nm in [(0, "approacher heading"), (1, "approachee heading")]:
        _tti = _frame[_m, cu.TTI]; _head = _frame[_m, cu.HEAD]
        if np.isfinite(_tti).all() and np.isfinite(_head).all():
            _d = (_head - _tti)
            _d = _d / (np.linalg.norm(_d) + 1e-9) * 55.0
            _fig.add_annotation(x=_cen[_m, 0] + _d[0], y=_cen[_m, 1] + _d[1],
                                ax=_cen[_m, 0], ay=_cen[_m, 1], xref="x", yref="y",
                                axref="x", ayref="y", showarrow=True, arrowhead=3,
                                arrowwidth=2, arrowcolor=ex_hex[_m])
    _fig.update_xaxes(showgrid=False, zeroline=True, title="x (px, body frame)")
    _fig.update_yaxes(showgrid=False, zeroline=True, scaleanchor="x", scaleratio=1,
                      title="y (px, body frame) — approacher faces up")
    _fig.update_layout(template="plotly_white", height=520, margin=dict(l=10, r=10, t=50, b=10),
                       title="The geometric features, drawn on the example event at contact",
                       legend=dict(y=0.99, x=0.01, bgcolor="rgba(255,255,255,0.6)"))
    mo.vstack([
        _fig,
        mo.md(
            """
            The **black dashed line** is the pair distance (`pair_dist_mean` / `pair_dist_min`). The
            **red dashed line** is the distance to the bystander (`bystander_dist_*`). The **dotted
            triangle** through all three centroids is what `triangle_area_mean` measures — how spread out
            the trio is. The **arrows** are each mouse's heading; the facing features (`appr_faces_appe`,
            `appe_faces_appr`) are the cosine of the angle between a mouse's arrow and the line to the
            other mouse (+1 = pointing straight at it), and `heading_alignment` is the cosine between the
            two arrows (+1 = parallel, −1 = opposed). None of these depend on where the trio sits in the
            arena — that is the whole point.
            """
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 5. Checking invariance: rotate and move the whole cage

        This is the payoff of the body-centered choice. Take the example event and apply a **rigid
        motion** to the *entire scene*: rotate the whole cage by some angle and slide it somewhere else.
        On the **left** the whole interaction visibly turns — the raw pixel coordinates swing with the
        cage. On the **right** is a single live table that reads out, side by side, two kinds of
        quantity computed on that same warped event:

        - **Body-frame features** — the 19 numbers from `features_one`, each marked **invariant**. Their
          *original* and *warped* values are identical to within rounding (the `change` column is
          essentially zero) no matter the angle, because every feature is measured *between* the mice,
          not against the arena walls.
        - **Arena-frame measurements** — the approacher's absolute heading angle and its centroid
          position in the cage, each marked **CHANGES**. These move on every turn of the slider, because
          they describe where the mouse is and which way it points in the arena.

        Drag the angle and read down the table. The `change` column stays at zero for every invariant
        feature while the three arena-frame rows swing. That contrast **is** the definition of
        invariance — far more directly than a plot of points that never move.
        """
    )
    return


@app.cell
def _(mo):
    inv_angle = mo.ui.slider(0, 350, value=0, step=10,
                             label="arena rotation applied to the whole event (degrees)",
                             debounce=True, full_width=True)
    return (inv_angle,)


@app.cell
def _(EX, cu, feat_names, go, inv_angle, kp, mo, np):
    _ev = kp[EX].astype(float)
    _f0 = cu.features_one(_ev)                              # 19 features, original
    _th = np.deg2rad(inv_angle.value)
    _R = np.array([[np.cos(_th), -np.sin(_th)], [np.sin(_th), np.cos(_th)]])
    _trans = np.array([600.0, -300.0])                     # a fixed, obvious translation
    _warp = np.einsum("ij,tmnj->tmni", _R, _ev) + _trans[None, None, None, :]
    _f1 = cu.features_one(_warp)                            # 19 features, after the rigid warp
    _maxdiff = float(np.nanmax(np.abs(_f0 - _f1)))

    # left: raw node cloud at contact, original vs warped -> the coordinates (the whole interaction)
    # clearly swing with the cage.
    _t = _ev.shape[0] // 2
    _p0 = _ev[_t].reshape(-1, 2); _p0 = _p0[np.isfinite(_p0).all(1)]
    _p1 = _warp[_t].reshape(-1, 2); _p1 = _p1[np.isfinite(_p1).all(1)]
    _left = go.Figure()
    _left.add_scatter(x=_p0[:, 0], y=_p0[:, 1], mode="markers",
                      marker=dict(color="#7f7f7f", size=6), name="original")
    _left.add_scatter(x=_p1[:, 0], y=_p1[:, 1], mode="markers",
                      marker=dict(color="#d62728", size=6), name="rotated + moved")
    _left.update_yaxes(scaleanchor="x", scaleratio=1, showgrid=False)
    _left.update_xaxes(showgrid=False)
    _left.update_layout(template="plotly_white", height=520,
                        title="RAW pixel coordinates — the whole interaction swings", legend=dict(y=1.0),
                        margin=dict(l=10, r=10, t=50, b=10))

    # arena-frame quantities that DO change: heading angle + centroid position (mid frame)
    def _arena_meas(evt):
        _mid = evt.shape[0] // 2
        _ap = evt[_mid, 0]
        _v = _ap[cu.HEAD] - _ap[cu.TTI]
        _hd = float(np.rad2deg(np.arctan2(_v[1], _v[0])))
        _cn = np.nanmean(_ap[cu.BODY_NODES], axis=0)
        return _hd, float(_cn[0]), float(_cn[1])
    _h0, _x0, _y0 = _arena_meas(_ev)
    _h1, _x1, _y1 = _arena_meas(_warp)

    # RIGHT (replaces the old "features frozen on the diagonal" scatter): one live table listing every
    # feature's value before vs after the warp, marked invariant vs changing. The 19 body-frame
    # features are invariant (change ~ 0); the three arena-frame rows CHANGE with the slider.
    _rows = [f"| `{_nm}` | invariant | {_a:.3f} | {_b:.3f} | {abs(_a - _b):.1e} |"
             for _nm, _a, _b in zip(feat_names, _f0, _f1)]
    _rows += [
        f"| approacher heading (deg) | **CHANGES** | {_h0:+.1f} | {_h1:+.1f} | {abs(_h1 - _h0):.1f} |",
        f"| approacher centroid x (px) | **CHANGES** | {_x0:.0f} | {_x1:.0f} | {abs(_x1 - _x0):.0f} |",
        f"| approacher centroid y (px) | **CHANGES** | {_y0:.0f} | {_y1:.0f} | {abs(_y1 - _y0):.0f} |",
    ]
    _table = mo.md(
        f"""
        **Feature-by-feature readout at arena rotation β = {inv_angle.value}°**
        (largest body-frame change across all 19 features: |Δ| = {_maxdiff:.2e} — numerically zero)

        | quantity | kind | original | warped | change |
        |---|---|---:|---:|---:|
        {chr(10).join("        " + _r for _r in _rows)}

        Every row marked **invariant** has an identical original and warped value (`change` ≈ 0),
        no matter the angle. The three **CHANGES** rows — heading and centroid position, defined
        against the arena — move on every turn. That is exactly what "arena-invariant" means.
        """
    )
    mo.vstack([inv_angle, mo.hstack([_left, _table], widths=[1, 1])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The body-frame features stay fixed to about **1e-4 or better** (numerically, zero) no matter how
        we spin or shift the cage, while the arena heading and centroid change with every turn. This is
        why the next several notebooks can treat one event as one point in a **19-dimensional space**
        without ever worrying about where in the arena it happened.

        ---
        ## 6. Which features carry aggression?

        We have 19 arena-invariant numbers per event. A natural first scientific question: **which of
        them actually distinguish aggression from non-aggression?** Before quantifying, look at the two
        kinds of event side by side. Below are clean **aggression** clips (top) and clean
        **non-aggression** approaches (bottom). Notice that the aggression clips are *busier* — sharp
        turns, sudden speed — while the non-aggression ones are smoother. The difference we are about to
        measure is one you can already see.
        """
    )
    return


@app.cell
def _(cu, ev, mo):
    # SHOW the difference before measuring it: two grids of exemplar clips. We pin each clip by its
    # stable event_key (never a raw integer index, which a bundle rebuild would silently re-point at a
    # different event) and resolve it to its current row; all are reliably tracked so they render
    # cleanly.
    _agg_keys = [
        "12192025_pre|cam.11.00046-2025-12-18T16|m1-m2|18193",
        "12192025_post|cam.10.00145-2025-12-23T18|m0-m2|88003",
        "12192025_pre|cam.10.00036-2025-12-18T16|m0-m1|3323",
        "12192025_dep|cam.10.00052-2025-12-19T16|m1-m2|10408",
    ]
    _non_keys = [
        "12192025_dep|cam.11.00191-2025-12-19T16|m1-m2|33859",
        "12192025_dep|cam.14.00140-2025-12-19T16|m1-m2|76967",
        "12192025_dep|cam.15.00001-2025-12-19T16|m0-m1|55075",
        "12192025_dep|cam.14.00144-2025-12-19T16|m0-m1|52961",
    ]
    _agg_idx = [cu.event_index_by_key(ev, _k) for _k in _agg_keys]
    _non_idx = [cu.event_index_by_key(ev, _k) for _k in _non_keys]
    _agg_grid = cu.grid_gif_bytes(
        [(ev["kp"][j], ev["ranks"][j], int(ev["contact_rel"][j])) for j in _agg_idx],
        ncols=4, cell=150, fps=20)
    _non_grid = cu.grid_gif_bytes(
        [(ev["kp"][j], ev["ranks"][j], int(ev["contact_rel"][j])) for j in _non_idx],
        ncols=4, cell=150, fps=20)
    mo.vstack([
        mo.md("**Aggression** — four clean exemplar clips"),
        mo.Html(cu.gif_img_html(_agg_grid, width=620)),
        mo.md("**Not aggression** — four clean exemplar clips"),
        mo.Html(cu.gif_img_html(_non_grid, width=620)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To turn that impression into a number, we use two standard tools:

        - **Cohen's d** — the difference in a feature's mean between the two groups, expressed in
          standard-deviation units. It is a plain, unitless **effect size**: |d| ≈ 0.2 is small, 0.5
          medium, 0.8 large. `cu.cohens_d(X, agg)` returns one d per feature.
        - **Mann–Whitney U** — a rank-based test of whether two groups' distributions differ, reported
          as a p-value.

        The chart below ranks all 19 features by |Cohen's d|. The pattern is the headline result of this
        notebook: the **kinematic** features (the approachee's angular velocity and speed above all)
        separate aggression far more strongly than the **geometry** features (facing, alignment,
        bystander distance). Aggression is a matter of *motion*, not of *where* the mice are.
        """
    )
    return


@app.cell
def _(X, agg, cu, feat_names, go, np):
    _d = cu.cohens_d(X, agg)                                # (19,) per-feature effect size
    _order = np.argsort(np.abs(_d))                         # ascending, so largest ends up on top
    _kin = {"appr_speed_mean", "appr_speed_max", "appe_speed_mean", "appe_speed_max",
            "appr_angvel", "appe_angvel", "closing_speed"}
    _names = [feat_names[i] for i in _order]
    _vals = _d[_order]
    _cols = ["#f58518" if feat_names[i] in _kin else "#4c78a8" for i in _order]
    _fig = go.Figure()
    # lollipop stems from 0 to d
    for _y, _v, _c in zip(range(len(_vals)), _vals, _cols):
        _fig.add_scatter(x=[0, _v], y=[_y, _y], mode="lines", line=dict(color=_c, width=2),
                         showlegend=False, hoverinfo="skip")
    _fig.add_scatter(x=_vals, y=list(range(len(_vals))), mode="markers",
                     marker=dict(size=10, color=_cols, line=dict(width=0.5, color="white")),
                     text=_names, hovertemplate="%{text}<br>Cohen's d = %{x:+.2f}<extra></extra>",
                     showlegend=False)
    _fig.add_vline(x=0, line_color="#333", line_width=1)
    _fig.update_yaxes(tickmode="array", tickvals=list(range(len(_names))), ticktext=_names)
    _fig.update_xaxes(title="Cohen's d  (aggression − not aggression, in SD units)")
    _fig.update_layout(template="plotly_white", height=560,
                       title="Effect size per feature — orange = kinematic, blue = posture/geometry",
                       margin=dict(l=10, r=10, t=50, b=10))
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Pick any feature below to see its full distribution in each group. The **violin** shows the
        shape of the distribution (a smoothed, mirrored histogram) with **every individual event drawn
        as a point** and a mean line — never a bare bar of averages. The **ECDF** beneath it (the
        empirical cumulative distribution) is a second, binning-free view: for each value on the x-axis
        it reads off the fraction of events at or below that value, so two curves that pull apart mean
        the groups genuinely differ. The header reports the Mann–Whitney p-value and Cohen's d.

        Start on `appe_angvel` (how fast the approachee turns) and watch the two clouds separate; then
        try `heading_alignment` and see them sit almost on top of each other.
        """
    )
    return


@app.cell
def _(feat_names, mo):
    feat_pick = mo.ui.dropdown(options=feat_names, value="appe_angvel",
                               label="feature to inspect")
    return (feat_pick,)


@app.cell
def _(X, agg, cu, feat_names, feat_pick, mo, np):
    from scipy.stats import mannwhitneyu
    _i = feat_names.index(feat_pick.value)
    _vals = X[:, _i]
    _grp = np.where(agg == 1, "aggression", "not aggression")
    _cols = {"aggression": cu.RANK_HEX[1], "not aggression": "#7f7f7f"}   # red vs gray

    _a = _vals[agg == 1]; _b = _vals[agg == 0]
    _u, _p = mannwhitneyu(_a[np.isfinite(_a)], _b[np.isfinite(_b)])
    _d = float(cu.cohens_d(X, agg)[_i])
    _ptxt = f"p = {_p:.1e}" if _p >= 1e-300 else "p < 1e-300"

    _violin = cu.violin_points_fig(
        _vals, _grp, group_order=["not aggression", "aggression"], colors=_cols,
        ylabel=feat_pick.value, robust=True,   # clip the value axis to [1, 99] pct so outliers don't skew it
        title=f"{feat_pick.value}   ·   Mann–Whitney {_ptxt}   ·   Cohen's d = {_d:+.2f}",
        height=440)
    _ecdf = cu.ecdf_fig(
        _vals, _grp, group_order=["not aggression", "aggression"], colors=_cols,
        xlabel=feat_pick.value, title=f"ECDF of {feat_pick.value} by group", height=320)
    mo.vstack([feat_pick, _violin, _ecdf])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        One more piece of EDA before the exercise. The 19 features are **not independent** — many carry
        overlapping information. To see a correlation honestly, plot the two features against each other
        as **individual points** (one dot per event, hover for its index) rather than a density blob:
        the scatter below shows `pair_dist_mean` vs `pair_dist_min`, with the least-squares fit line and
        the **Pearson r** annotated in the corner. `r` runs from −1 (perfect anti-correlation) through 0
        (unrelated) to +1 (perfectly linearly related). The two features track each other tightly (large
        positive `r`): events where the mice are far apart on average are also events where their closest
        approach is far. When features are this redundant, 19 numbers are really only a handful of
        independent directions — which is exactly the motivation for the dimensionality reduction we take
        up in **NB03**.
        """
    )
    return


@app.cell
def _(X, cu, feat_names, np):
    _i = feat_names.index("pair_dist_mean")
    _j = feat_names.index("pair_dist_min")
    # Individual points (NOT a density) with an annotated Pearson r — the honest way to show a
    # correlation. Robust axes clip to the 1st/99th percentile so a few far-apart events do not
    # compress the bulk of the cloud (those outliers are still plotted, just off the default view).
    cu.scatter_points_fig(
        X[:, _i], X[:, _j], hover=np.arange(len(X)), annotate_r=True, trendline=True, robust=True,
        xlabel="pair_dist_mean (px)", ylabel="pair_dist_min (px)",
        title="Two features that move together — 19 numbers are not 19 independent facts",
        height=460)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 7. Exercise — do aggressive approaches provoke a faster reaction?

        ### Python skill practiced: **array arithmetic + boolean masks**

        A *boolean mask* is an array of `True`/`False` the same length as your data, usually built by
        comparing an array to a threshold (`values > t`). Indexing another array with that mask keeps
        only the rows where it is `True`. This one idea — *compare to make a mask, then index with it* —
        is the workhorse of all data analysis, and you will use it constantly.

        ### Why ask this

        The effect-size chart said aggression is kinematic, and the single strongest feature was
        `appe_angvel` — **how fast the approachee turns**. That makes biological sense: when an approach
        is aggressive, the approached mouse *reacts* — it spins, flinches, tries to flee. We will make
        that precise: split events into those where the approachee turned a lot vs a little, and compare
        the **aggression base rate** in each half.

        ### Definitions you need

        - `appe_angvel` — feature #7, the approachee's mean heading angular velocity (turning rate).
        - **base rate** — the fraction of events that are aggression; here `agg` is a 0/1 array, so a
          fraction is just `agg[mask].mean()`.

        ### What to do

        Fill in the **two blanks** (`____`). Everything else is written for you. The plot below is driven
        by your edits, so if a blank is wrong the picture will look wrong.
        """
    )
    return


@app.cell
def _(X, agg, feat_names, np):
    # ===================== EXERCISE — edit ONLY the two lines marked  # TODO , then run ==============
    # Skill: array arithmetic + boolean masks. The two TODO lines start with deliberately-wrong
    # placeholders, so the self-check reads "Not yet" until you fix them. Nothing else needs editing.

    _i = feat_names.index("appe_angvel")     # column index of the approachee's turning-rate feature
    turn = X[:, _i]                          # (N,) one turning-rate value per event
    thr = np.nanmedian(turn)                 # median split: the middle value, so the two halves are equal-sized

    # --- TODO 1 --------------------------------------------------------------------------------------
    # Build a BOOLEAN MASK that is True for the events where the approachee turned MORE than the median.
    # WHY it matters: this mask defines the "strong reaction" half of the data; the aggression rate we
    # compute next is only meaningful if the mask correctly selects the high-turning events.
    # Replace the placeholder (currently all-False) with the array comparison  turn > thr .
    spun = np.zeros(len(turn), dtype=bool)   # TODO 1: change to  turn > thr
    calm = ~spun                             # the other half (logical NOT of your mask) — written for you

    # --- TODO 2 --------------------------------------------------------------------------------------
    # Compute the aggression BASE RATE inside the high-turning half. `agg` is a 0/1 array, so the mean
    # of the masked slice is the fraction that are aggression.
    # WHY it matters: comparing this number to the low-turning half is the whole result — if a fast
    # reaction accompanies aggression, this should be much larger than rate_calm.
    # Replace the placeholder 0.0 with  agg[spun].mean()  (index agg with your boolean mask, then mean).
    rate_spun = 0.0                          # TODO 2: change to  agg[spun].mean()
    rate_calm = agg[calm].mean()             # the low-turning half's rate — written for you
    # ================================================================================================
    return calm, rate_calm, rate_spun, spun, thr, turn


@app.cell
def _(agg, cu, np, rate_calm, rate_spun, spun, thr, turn):
    # Expected picture: two violins of the approachee's turning rate, one for each half of the split.
    # The "high-turn" violin sits well above the "low-turn" one (by construction, since we split on it),
    # and its title-reported aggression rate is far higher (~0.51 vs ~0.13). If your mask is wrong, the
    # split will not separate and the two rates will be equal.
    _grp = np.where(spun, f"high turn  (agg rate {rate_spun:.2f})",
                          f"low turn  (agg rate {rate_calm:.2f})")
    _order = [f"low turn  (agg rate {rate_calm:.2f})", f"high turn  (agg rate {rate_spun:.2f})"]
    _cols = {_order[0]: "#7f7f7f", _order[1]: cu.RANK_HEX[1]}
    cu.violin_points_fig(
        turn, _grp, group_order=_order, colors=_cols, ylabel="appe_angvel (turning rate)",
        robust=True,   # clip the value axis to [1, 99] pct so a few extreme turners don't skew the view
        title=f"Median split on approachee turning rate (threshold {thr:.3f})",
        height=440)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Show solution": mo.md(
            r"""
            Fill the two blanks like this:

            ```python
            spun = turn > thr          # TODO 1: a boolean mask, True where turning rate beats the median
            rate_spun = agg[spun].mean()   # TODO 2: aggression fraction inside that mask
            ```

            **What you should find:** aggression is roughly **0.51** in the high-turning half and only
            about **0.13** in the low-turning half — a nearly four-fold difference from a single
            kinematic feature. The high-turn violin sits far above the low-turn one. This confirms the
            effect-size chart from the other direction: an aggressive approach is one the approached
            mouse *reacts* to. The signal lives in **motion**, which is why we keep all 19 features
            rather than a geometry-only summary.
            """
        )
    })
    return


@app.cell(hide_code=True)
def _(mo, rate_calm, rate_spun):
    # Self-check with a tolerance band. Pinned full-corpus values: high-turn agg rate ~0.51,
    # low-turn ~0.13, difference ~0.38.
    _diff = float(rate_spun) - float(rate_calm)
    _p1 = abs(float(rate_spun) - 0.510) < 0.05
    _p2 = abs(float(rate_calm) - 0.130) < 0.05
    _ok = _p1 and _p2
    _c = "#e8f5e9" if _ok else "#ffebee"
    _b = "#2e7d32" if _ok else "#c62828"
    if _ok:
        _head = "PASS — both blanks correct"
        _msg = (f"High-turn aggression rate = {rate_spun:.3f}, low-turn = {rate_calm:.3f} "
                f"(difference {_diff:+.3f}). A fast reaction from the approachee accompanies aggression.")
    else:
        _head = "Not yet — check the two TODO lines"
        _msg = (f"Your rates are high-turn = {rate_spun:.3f}, low-turn = {rate_calm:.3f}. Expected about "
                "0.51 and 0.13. TODO 1 should be the mask  turn > thr ; TODO 2 should be "
                " agg[spun].mean() .")
    mo.md(
        f"""
        <div style="background:{_c};border-left:6px solid {_b};padding:12px 16px;border-radius:6px">
        <b style="color:{_b}">{_head}</b><br>
        {_msg}<br>
        <span style="font-size:0.9em;color:#555">Tolerance band: high-turn rate within 0.05 of 0.510,
        low-turn within 0.05 of 0.130.</span>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## A preview of the metadata (deferred to NB04)

        The derived bundle also carries `cohort`, `cage`, and `sex` for every event. It is tempting to
        immediately ask questions like *"do males and females differ on some feature?"* — and the naive
        event-level test will happily hand back a tiny p-value. But events from the same cage are not
        independent observations, and a cage is either all-male or all-female, so the correct **unit of
        analysis is the cage**, not the event. Doing that comparison honestly (event-level vs cage-level,
        with positive and negative controls) is the whole statistics section of **NB04**. We flag the
        metadata here so you know it exists; we deliberately do **not** draw conclusions from it yet.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## What the transform discards, and how it can break

        **Discarded on purpose.** The body-frame transform deliberately drops *where in the cage* the
        event happened and *which way the approacher faced in the arena*. That is the goal, but it means
        we can **no longer ask arena questions** from `X` alone (does aggression cluster near a wall? at
        the food hopper?). Those need the raw coordinates back — which is why we keep them on disk.

        **Failure modes on this dataset.**

        1. **Silent fallback to raw coordinates.** If the approacher's head or tail-base is missing on
           *every* frame, `allocentricize` cannot find a heading and returns the event **unchanged** —
           the features are then computed in raw arena coordinates and are *not* invariant. This is
           invisible unless we audit for it, and head/tail dropout are exactly the nodes NB01 flagged as
           least reliable.
        2. **One bad frame rotates the whole scene.** The transform reads the heading from a *single*
           anchor frame. If that frame's head or tail-base is jittery, the entire event is rotated to the
           wrong angle and every geometry feature is corrupted, with no error raised.
        3. **Effect sizes are not causes.** `appe_angvel` separates aggression cleanly, but a large
           Cohen's d is a description, not a mechanism. We have shown that aggression *co-occurs* with a
           fast reaction; we have not shown what drives what. Keeping all 19 features, rather than
           betting on one, is what lets later notebooks probe structure rather than a single ratio.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## What we answered, and what comes next

        **The question was:** how do we describe an interaction independent of where in the arena it
        happens and which way the animals face? **The answer:** re-express every event in the
        approacher's own body frame — translate its tail-base to the origin and rotate its heading to a
        fixed direction — and then summarize the result into **19 arena-invariant features**. We proved
        the invariance by spinning the whole cage and watching the features refuse to move, and we found
        a first real result: aggression is carried by **kinematics** (above all the approachee's turning
        rate), not by approach geometry.

        The deliverable, **`X (2499, 19)`**, is what every later notebook reads instead of pixels.

        But we collapsed each event to a *single* vector of summary numbers — a mean speed, a peak speed,
        a closest distance — throwing away the fact that an interaction **unfolds in time**. A lunge, a
        pause, a flee, and a chase all have shape that a mean cannot see.

        > **Next (NB03):** what does an interaction look like as a signal *over time*? We will read the
        > per-frame kinematics the way a physiologist reads a raw trace — in value, in time, and in
        > frequency — and measure who moves first between two coupled mice.
        """
    )
    return


if __name__ == "__main__":
    app.run()
