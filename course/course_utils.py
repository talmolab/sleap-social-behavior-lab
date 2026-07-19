"""
course_utils.py — the shared engine for the SLEAP social-behavior course.

Everything numerical the notebooks rely on lives here so the notebook cells stay short and
readable: the allocentric feature transform, the clustering wrappers (PCA / residualization /
UMAP / HDBSCAN), the rank-dyad statistics, skeleton-GIF rendering, and the MLP classifier.

Self-contained: numpy / scipy / scikit-learn / umap-learn / hdbscan / Pillow / imageio only.
No lab code, no video files, no GPU. Everything runs from the small bundled `data/*.npz`.

Skeleton (15 nodes, star topology with two hubs: head=1, TTI=11):
    0 nose  1 head  2 L_ear  3 L_shoulder  4 neck  5 R_ear  6 R_shoulder
    7 L_haunch  8 R_haunch  9 tail_1  10 tail_0  11 TTI  12 tail_2  13 tail_tip  14 trunk
"""
from __future__ import annotations
import io
import os
import base64
import urllib.request
import numpy as np


# ============================================================================ asset bootstrap
# So a single notebook file can run anywhere — including a bare cloud notebook (molab) with no
# repo checkout. When this module or the bundled data aren't on disk, they're fetched from the
# public GitHub repo's raw endpoint. Override the source with $COURSE_REPO_RAW.
REPO_RAW = os.environ.get(
    "COURSE_REPO_RAW",
    "https://raw.githubusercontent.com/talmolab/sleap-social-behavior-lab/main",
)
DATA_FILES = [
    "data/train_events.npz",
    "data/heldout_events.npz",
    "data/cohort_meta.csv",
    "data/answer_key.csv",
    # Notebook 01 loads this small pre-decoded clip instead of a raw .slp, so students never need
    # sleap-io (a heavy install on bare cloud kernels). Regenerate with tools/decode_example_slp.py.
    "data/raw_slp/example_slp_decoded.npz",
    # Notebook 03: a precomputed UMAP parameter sweep + default embedding, so the map renders
    # instantly instead of blocking ~30s on numba JIT. Regenerate with tools/build_umap_sweep.py.
    "data/umap_sweep.npz",
    # Precomputed metadata (cage/sex/tod) + features (X) + PCA, aligned to the event files, so no
    # student kernel does >2s compute. tools/build_derived.py.
    "data/train_derived.npz",
    "data/heldout_derived.npz",
    # One cage's continuous 24h span (2 fps) for NB07's activity clock + Markov grammar.
    "data/continuous_tracks.npz",
    # Synthetic population raster for NB08's neural payoff. tools/build_neural_demo.py.
    "data/neural_demo.npz",
]


def find_root(start=None):
    """Walk up from `start` (or cwd) for a folder holding both course/ and data/; else None."""
    p = start or os.getcwd()
    for _ in range(6):
        if os.path.isdir(os.path.join(p, "course")) and os.path.isdir(os.path.join(p, "data")):
            return p
        p = os.path.dirname(p)
    return None


def data_path(rel, root=None):
    """Local path to a repo-relative file (e.g. 'data/train_events.npz'), downloading it from
    REPO_RAW if it isn't already on disk. Lets a notebook run with no repo checkout."""
    rel = rel.replace("\\", "/").lstrip("/")
    root = root or find_root() or os.getcwd()
    dst = os.path.join(root, rel)
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        urllib.request.urlretrieve(REPO_RAW + "/" + rel, dst)
    return dst


def ensure_data(root=None):
    """Make sure every bundled data file is present under root/data (download what's missing).
    Returns the data directory path."""
    root = root or find_root() or os.getcwd()
    for rel in DATA_FILES:
        data_path(rel, root)
    return os.path.join(root, "data")


def bootstrap():
    """Make the course runnable from a bare notebook (e.g. molab): locate the repo layout (or fall
    back to the cwd), download the bundled data if absent, and return (ROOT, DATA, SCRATCH)."""
    root = find_root() or os.getcwd()
    data = ensure_data(root)
    scratch = os.path.join(data, "_scratch")
    os.makedirs(scratch, exist_ok=True)
    return root, data, scratch

# ----------------------------------------------------------------------------- skeleton
NODE_NAMES = ["nose", "head", "L_ear", "L_shoulder", "neck", "R_ear", "R_shoulder",
              "L_haunch", "R_haunch", "tail_1", "tail_0", "TTI", "tail_2", "tail_tip", "trunk"]
N_NODES = 15
NOSE, HEAD, TTI, TAILTIP = 0, 1, 11, 13
BODY_NODES = list(range(9))            # nose .. R_haunch (the compact body, used for centroids)
SKELETON_EDGES = [(0, 1), (1, 2), (1, 5), (1, 3), (1, 6), (1, 4), (1, 11),
                  (11, 14), (11, 7), (11, 8), (11, 9), (11, 10), (11, 12), (11, 13)]
FPS = 50

# rank color scheme (Dom=red, Mid=blue, Sub=green); gray = unknown
RANK_NAMES = {1: "Dom", 2: "Mid", 3: "Sub", 0: "?"}
RANK_RGB = {1: (214, 39, 40), 2: (31, 119, 180), 3: (44, 160, 44), 0: (150, 150, 150)}
RANK_HEX = {r: "#%02x%02x%02x" % c for r, c in RANK_RGB.items()}
CONDITIONS = ["pre", "dep", "post"]    # despotism manipulation (the primary independent variable)


# ============================================================================ data loading
def load_events(npz_path):
    """Load one bundled event set. Returns a dict:
       kp          (N, T, 3, 15, 2) float32 world-coordinate keypoints, mice ordered
                   [approacher, approachee, bystander] (the corrected assignment).
       ranks       (N, 3) int      rank (1=Dom,2=Mid,3=Sub, 0=unknown) of each ordered mouse.
       condition   (N,) str        'pre' | 'dep' | 'post'.
       contact_rel (N,) int        frame within the window where contact begins.
       event_key   (N,) str        stable id  cohort|stem|pair|contact_start.
       category    (N,) str        registry label if any ('aggression', 'mlp_fp', ...), else ''.
       agg_label   (N,) int        1 if category == 'aggression' else 0  (ground truth)."""
    if not os.path.exists(npz_path):                 # bare notebook: pull it from the repo
        npz_path = data_path("data/" + os.path.basename(npz_path))
    z = np.load(npz_path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["kp"] = d["kp"].astype(np.float32)
    return d


def event_index_by_key(events, key):
    """Row index of the event whose stable ``event_key`` == ``key``.

    ALWAYS select example events this way, never by a raw integer index: the bundle is periodically
    rebuilt (e.g. to add a cohort), which changes N and RE-ORDERS the rows, so a hardcoded index like
    909 silently points at a different event afterwards. An event_key is stable across rebuilds.

    ``events`` is the dict returned by ``load_events`` (uses its ``event_key`` array); an array/list
    of keys is also accepted. Raises KeyError if the key is absent, ValueError if it is duplicated."""
    keys = events["event_key"] if isinstance(events, dict) else events
    keys = np.asarray(keys).astype(str)
    hits = np.where(keys == str(key))[0]
    if len(hits) == 0:
        raise KeyError(
            f"event_key {key!r} not found in this bundle ({len(keys)} events). It may live in the "
            f"other split (train vs heldout) or predate the current build.")
    if len(hits) > 1:
        raise ValueError(f"event_key {key!r} is not unique ({len(hits)} matches at {hits.tolist()}).")
    return int(hits[0])


def find_example_index(events, derived=None, category=None, cage=None, sex=None, cohort=None,
                       tag="train"):
    """Deterministic FIRST event matching the given criteria — a robust way to pick an illustrative
    example by its PROPERTIES instead of a fragile integer index. Returns the lowest matching row
    index, or -1 if nothing matches.

    ``category`` is tested against ``events['category']`` (e.g. 'aggression', 'mounting', 'mlp_fp').
    ``cage`` / ``sex`` / ``cohort`` come from the derived bundle: pass ``derived`` (from
    ``load_derived``) or leave it None to load ``load_derived(tag)`` (defaults to the train split,
    which must be the same split as ``events``). All given criteria are AND-combined."""
    keys = np.asarray(events["event_key"]).astype(str)
    mask = np.ones(len(keys), bool)
    if category is not None:
        mask &= np.asarray(events["category"]).astype(str) == str(category)
    if any(v is not None for v in (cage, sex, cohort)):
        if derived is None:
            derived = load_derived(tag)
        if cage is not None:
            mask &= derived["cage"].astype(int) == int(cage)
        if sex is not None:
            mask &= derived["sex"].astype(str) == str(sex)
        if cohort is not None:
            mask &= derived["cohort"].astype(str) == str(cohort)
    hits = np.where(mask)[0]
    return int(hits[0]) if len(hits) else -1


def load_slp_demo(root=None):
    """Load the pre-decoded example SLEAP clip for notebook 01 (no sleap-io needed).

    Produced from a real `.slp` by tools/decode_example_slp.py, which placed each instance in its
    fixed **track slot** — so slot m is always the same animal (raw `LabeledFrame.numpy()` orders
    instances per-frame, which makes the skeleton colors flicker while scrubbing). Returns a dict:
        kp         (frames, tracks, nodes, 2) float32, NaN where a track is missing that frame
        node_names (nodes,) str
        edges      (E, 2) int   skeleton edges as (src_idx, dst_idx)
        source     str          the .slp filename it came from
    """
    path = data_path("data/raw_slp/example_slp_decoded.npz", root)
    z = np.load(path, allow_pickle=True)
    return dict(kp=z["kp"].astype(np.float32), node_names=[str(n) for n in z["node_names"]],
                edges=[(int(u), int(v)) for u, v in z["edges"]], source=str(z["source"]))


def skeleton_fig(kp_frame, edges, colors=("#d62728", "#1f77b4", "#2ca02c"),
                 title="SLEAP skeletons — drag the frame slider", height=520):
    """Plotly figure of one frame's rank/track-colored skeletons on a blank canvas.

    kp_frame: (mice, nodes, 2) in image pixels (y grows downward). Each mouse index keeps a fixed
    color, so identities stay put as you scrub. Missing nodes/animals (NaN) are simply not drawn."""
    import plotly.graph_objects as go
    traces = []
    for m in range(kp_frame.shape[0]):
        kp = kp_frame[m]
        ok = np.isfinite(kp).all(1)
        ex, ey = [], []
        for u, v in edges:
            if ok[u] and ok[v]:
                ex += [kp[u, 0], kp[v, 0], None]
                ey += [kp[u, 1], kp[v, 1], None]
        col = colors[m % len(colors)]
        traces.append(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=col, width=2),
                                 name=f"mouse {m}", hoverinfo="skip"))
        traces.append(go.Scatter(x=kp[ok, 0], y=kp[ok, 1], mode="markers",
                                 marker=dict(color=col, size=7), showlegend=False, hoverinfo="skip"))
    f = go.Figure(traces)
    f.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    f.update_layout(height=height, title=title, margin=dict(l=10, r=10, t=40, b=10),
                    template="plotly_white")
    return f


# ============================================================================ allocentric transform
def _centroids(mouse_kp):
    """mouse_kp (T,15,2) -> (T,2) body centroid (nan-mean over the finite body nodes per frame)."""
    import warnings
    body = mouse_kp[:, BODY_NODES, :]
    with warnings.catch_warnings():                # all-NaN frames -> NaN centroid (expected)
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(body, axis=1)


def _heading(mouse_kp):
    """(T,15,2) -> (T,2) unit heading vector TTI->head (the direction the mouse faces)."""
    v = mouse_kp[:, HEAD, :] - mouse_kp[:, TTI, :]
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n < 1e-6, np.nan, n)


def _anchor_transform(focal_kp):
    """Center + rotation that put the FOCAL mouse at the origin facing +Y, taken at the first
    frame where its head and TTI are both valid. Returns (center(2,), R(2,2), ok:bool).

    R rotates the focal heading (TTI->head) onto +Y so every event is viewed from the
    approacher's own body frame — this is what makes the features *allocentric* (identity- and
    arena-pose-invariant: only the geometry of the social configuration remains)."""
    head_ok = np.isfinite(focal_kp[:, HEAD, :]).all(1)
    tti_ok = np.isfinite(focal_kp[:, TTI, :]).all(1)
    valid = np.where(head_ok & tti_ok)[0]
    if len(valid) == 0:
        return np.zeros(2), np.eye(2), False
    a = valid[0]
    center = focal_kp[a, TTI, :]
    v = focal_kp[a, HEAD, :] - center
    phi = np.arctan2(v[1], v[0])
    alpha = np.pi / 2 - phi                      # rotate heading angle phi -> +Y (pi/2)
    c, s = np.cos(alpha), np.sin(alpha)
    R = np.array([[c, -s], [s, c]])
    return center, R, True


def allocentricize(kp_event):
    """kp_event (T,3,15,2) world coords, mouse 0 = focal (approacher). Returns the same array
    rotated+centered into the approacher's body frame (for features; NOT for rendering)."""
    center, R, ok = _anchor_transform(kp_event[:, 0])
    if not ok:
        return kp_event
    shifted = kp_event - center[None, None, None, :]
    return np.einsum("ij,tmnj->tmni", R, shifted)


# ============================================================================ features
FEATURE_NAMES = [
    "appr_speed_mean", "appr_speed_max", "appe_speed_mean", "appe_speed_max",
    "appr_body_len", "appe_body_len", "appr_angvel", "appe_angvel",
    "pair_dist_mean", "pair_dist_min",
    "appr_nose_to_appe_tti_min", "appe_nose_to_appr_tti_min",
    "appr_faces_appe", "appe_faces_appr", "closing_speed", "heading_alignment",
    "bystander_dist_mean", "bystander_dist_min", "triangle_area_mean",
]
N_FEATURES = len(FEATURE_NAMES)


def _nanmean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else 0.0


def _nanmax(x):
    x = np.asarray(x, float)
    return float(np.nanmax(x)) if np.isfinite(x).any() else 0.0


def _nanmin(x):
    x = np.asarray(x, float)
    return float(np.nanmin(x)) if np.isfinite(x).any() else 0.0


def features_one(kp_event):
    """kp_event (T,3,15,2) world coords, ordered [approacher, approachee, bystander].
    Returns a (N_FEATURES,) float32 vector of interpretable allocentric features."""
    kp = allocentricize(kp_event)
    ap, ae, by = kp[:, 0], kp[:, 1], kp[:, 2]
    cen = {m: _centroids(kp[:, m]) for m in range(3)}

    def speed(m):
        return np.linalg.norm(np.diff(cen[m], axis=0), axis=1)      # (T-1,) px/frame

    def angvel(mouse_kp):
        h = _heading(mouse_kp)
        ang = np.arctan2(h[:, 1], h[:, 0])
        return np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)

    def pdist(m1, m2):
        return np.linalg.norm(cen[m1] - cen[m2], axis=1)            # (T,)

    def node_dist(a_kp, a_node, b_kp, b_node):
        return np.linalg.norm(a_kp[:, a_node] - b_kp[:, b_node], axis=1)

    def faces(src_kp, src_c, tgt_c):
        h = _heading(src_kp)
        to = tgt_c - src_c
        to = to / np.linalg.norm(to, axis=1, keepdims=True)
        return np.nansum(h * to, axis=1)                            # cos angle in [-1,1]

    d_ap_ae = pdist(0, 1)
    closing = -np.diff(d_ap_ae)                                     # +ve = approaching
    h_ap, h_ae = _heading(ap), _heading(ae)
    align = np.nansum(h_ap * h_ae, axis=1)                          # +1 same dir, -1 opposed

    f = [
        _nanmean(speed(0)), _nanmax(speed(0)), _nanmean(speed(1)), _nanmax(speed(1)),
        _nanmean(node_dist(ap, NOSE, ap, TTI)), _nanmean(node_dist(ae, NOSE, ae, TTI)),
        _nanmean(angvel(ap)), _nanmean(angvel(ae)),
        _nanmean(d_ap_ae), _nanmin(d_ap_ae),
        _nanmin(node_dist(ap, NOSE, ae, TTI)), _nanmin(node_dist(ae, NOSE, ap, TTI)),
        _nanmean(faces(ap, cen[0], cen[1])), _nanmean(faces(ae, cen[1], cen[0])),
        _nanmean(closing), _nanmean(align),
        _nanmean(pdist(0, 2)), _nanmin(pdist(0, 2)), _nanmean(_triangle_area(cen[0], cen[1], cen[2])),
    ]
    return np.asarray(f, dtype=np.float32)


def features_batch(kp):
    """kp (N,T,3,15,2) -> (N, N_FEATURES) float32."""
    return np.stack([features_one(kp[i]) for i in range(len(kp))], axis=0)


def _triangle_area(a, b, c):
    """Per-frame area of the triangle formed by three (T,2) centroid tracks."""
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


# ============================================================================ clustering pipeline
def standardize(X):
    """z-score columns. Returns (Xz, mean, std)."""
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xz = np.nan_to_num((X - mu) / sd)
    return Xz, mu, sd


def pca_scores(Xz, n_components):
    """PCA via sklearn. Returns (scores (N,k), explained_variance_ratio (k,), PCA object)."""
    from sklearn.decomposition import PCA
    p = PCA(n_components=n_components, random_state=0).fit(Xz)
    return p.transform(Xz), p.explained_variance_ratio_, p


def residualize(scores, drop_pcs):
    """Zero out the listed principal components (0-indexed). Used to partial out dominant
    nuisance axes (e.g. overall proximity / locomotor magnitude) before UMAP so the embedding
    reflects finer behavioral structure rather than 'how close / how fast' alone."""
    out = scores.copy()
    for pc in drop_pcs:
        if 0 <= pc < out.shape[1]:
            out[:, pc] = 0.0
    return out


def run_umap(X, n_neighbors=15, min_dist=0.1, n_components=2, seed=42):
    """Live UMAP embedding — HARD-DISABLED in the student path. A cold molab kernel spends ~30s in
    UMAP's numba JIT and the websocket times out, hanging the app, so NB05 must SELECT a precomputed
    embedding from ``load_umap_sweep()['emb_grid'][i, j]`` instead of ever calling this.

    Reachable only for instructors who export ``COURSE_ALLOW_LIVE_UMAP=1`` (e.g. when regenerating
    the sweep via tools/build_umap_sweep.py); otherwise it raises RuntimeError. ``import umap`` is
    kept lazy (inside this guard) so ``import course_utils`` succeeds with no umap-learn installed."""
    if os.environ.get("COURSE_ALLOW_LIVE_UMAP") != "1":
        raise RuntimeError(
            "Live UMAP is disabled in the student path (it JIT-compiles ~30s on a cold molab "
            "kernel and hangs the app). Select a precomputed embedding from "
            "load_umap_sweep()['emb_grid'][i, j] instead. Instructors regenerating the sweep can "
            "set the environment variable COURSE_ALLOW_LIVE_UMAP=1 to re-enable this.")
    import umap
    reducer = umap.UMAP(n_neighbors=int(n_neighbors), min_dist=float(min_dist),
                        n_components=int(n_components), random_state=int(seed))
    return reducer.fit_transform(X)


def run_hdbscan(emb, min_cluster_size=30, min_samples=None):
    """Density clustering. Returns integer labels (-1 = noise / unassigned)."""
    import hdbscan
    kw = dict(min_cluster_size=int(min_cluster_size))
    if min_samples:
        kw["min_samples"] = int(min_samples)
    return hdbscan.HDBSCAN(**kw).fit_predict(emb)


# Canonical defaults (tuned on the bundled data to expose a Dom-enriched aggression cluster).
# Notebooks 03/04/05 all reference these so their clusterings agree. These are the EXACT knobs that
# built data/umap_sweep.npz: n_neighbors=15 == nn_values[1] and min_dist=0.0 == md_values[0] pick the
# pinned default cell emb_grid[default_ij] (default_ij == [1, 0]), and min_cluster_size=15 (no
# min_samples) reproduces the shipped sweep default_labels EXACTLY. So a student running
# run_hdbscan(emb_grid[tuple(default_ij)], min_cluster_size=15) sees the SAME clusters as the canon.
CLUSTER_DEFAULTS = dict(pca_k=10, drop_pcs=(0, 2), n_neighbors=15,
                        min_dist=0.0, min_cluster_size=15, seed=42)


def cluster_pipeline(X, pca_k=10, drop_pcs=(0, 2), n_neighbors=15, min_dist=0.0,
                     min_cluster_size=15, seed=42):
    """The full standardize -> PCA -> residualize -> UMAP -> HDBSCAN chain in one call, so the
    clustering is reproducible across notebooks. Returns a dict with emb, labels, scores, evr."""
    Xz, mu, sd = standardize(X)
    scores, evr, pca = pca_scores(Xz, pca_k)
    res = residualize(scores, list(drop_pcs))
    emb = run_umap(res, n_neighbors, min_dist, 2, seed)
    labels = run_hdbscan(emb, min_cluster_size)
    return dict(emb=emb, labels=labels, scores=scores, evr=evr, Xz=Xz)


def load_umap_sweep(root=None):
    """Load the precomputed UMAP parameter sweep for notebook 03 (see tools/build_umap_sweep.py).
    Lets the notebook show how n_neighbors/min_dist reshape the map *without* running UMAP live
    (a ~30s numba JIT on a cold kernel that otherwise hangs the web app). Returns the npz dict."""
    z = np.load(data_path("data/umap_sweep.npz", root), allow_pickle=True)
    return {k: z[k] for k in z.files}


def sweep_grid_fig(emb_grid, nn_values, md_values, color_key, palette, names, height=680):
    """Small-multiples plot of a UMAP sweep: one panel per (n_neighbors, min_dist) cell, each
    colored the same way, so the parameter effect is visible at a glance. color_key is a per-point
    integer group id; palette/names map group id -> color/label."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    n_nn, n_md = len(nn_values), len(md_values)
    # One short label per column (min_dist) and per row (n_neighbors) instead of a full title per
    # panel — per-panel titles collide / clip at medium width. Row labels are drawn rotated on the
    # right, so keep them short ("nn=..") to stop the longest ("nn=100") clipping into its neighbour.
    col_titles = [f"min_dist={md:g}" for md in md_values]
    row_titles = [f"nn={nn}" for nn in nn_values]
    fig = make_subplots(rows=n_nn, cols=n_md, column_titles=col_titles, row_titles=row_titles,
                        horizontal_spacing=0.02, vertical_spacing=0.04)
    groups = [g for g in names if (color_key == g).any()]
    for i in range(n_nn):
        for j in range(n_md):
            emb = emb_grid[i, j]
            for g in groups:
                m = color_key == g
                fig.add_trace(go.Scattergl(
                    x=emb[m, 0], y=emb[m, 1], mode="markers", name=names[g],
                    marker=dict(size=3, opacity=0.6, color=palette[g]),
                    legendgroup=names[g], showlegend=(i == 0 and j == 0)),
                    row=i + 1, col=j + 1)
    # Spatial small-multiples: no ticks and no gridlines (grid off on spatial views, house style).
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(template="plotly_white", height=height,
                      title="UMAP parameter sweep — rows = n_neighbors (nn), cols = min_dist",
                      margin=dict(l=10, r=48, t=80, b=10))
    return fig


# ============================================================================ rank / condition statistics
DYADS = [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]     # directed (approacher_rank, approachee_rank)
DYAD_LABELS = [f"{RANK_NAMES[a]}>{RANK_NAMES[b]}" for a, b in DYADS]


def _dyad_index(ra, re):
    try:
        return DYADS.index((int(ra), int(re)))
    except ValueError:
        return -1


def rank_dyad_enrichment(labels, ranks_appr, ranks_appe, min_n=15):
    """For every cluster, test whether its directed rank-dyad composition differs from the rest of
    the data (chi-square on the 6 directed dyads, cluster vs rest), with a Bonferroni correction
    over the clusters tested. Returns a list of per-cluster dicts sorted by p-value.

    'Enriched dyad' = the dyad with the largest positive standardized (Pearson) residual — the one
    most over-represented in the cluster relative to expectation."""
    from scipy.stats import chi2_contingency
    labels = np.asarray(labels)
    di = np.array([_dyad_index(a, e) for a, e in zip(ranks_appr, ranks_appe)])
    valid = di >= 0
    clusters = sorted(c for c in set(labels[valid]) if c >= 0)
    tested = [c for c in clusters if (valid & (labels == c)).sum() >= min_n]
    results = []
    for c in tested:
        inc = valid & (labels == c)
        out = valid & (labels != c)
        in_counts = np.bincount(di[inc], minlength=6)
        out_counts = np.bincount(di[out], minlength=6)
        table = np.vstack([in_counts, out_counts])
        keep = table.sum(0) > 0
        if keep.sum() < 2:
            continue
        chi2, p, dof, exp = chi2_contingency(table[:, keep])
        resid = (table[0, keep] - exp[0]) / np.sqrt(exp[0])
        enr = np.where(keep)[0][int(np.argmax(resid))]
        results.append(dict(cluster=int(c), n=int(in_counts.sum()), chi2=float(chi2), p=float(p),
                            dof=int(dof), enriched_dyad=DYAD_LABELS[enr],
                            dyad_fracs={DYAD_LABELS[i]: float(in_counts[i] / max(1, in_counts.sum()))
                                        for i in range(6)}))
    m = max(1, len(tested))
    for r in results:
        r["p_bonf"] = min(1.0, r["p"] * m)
        r["sig"] = r["p_bonf"] < 0.05
    return sorted(results, key=lambda r: r["p"])


def condition_enrichment(labels, condition, min_n=15):
    """Per-cluster chi-square over pre/dep/post composition (cluster vs rest). Tests whether a
    behavioral cluster is more common in a despotism phase."""
    from scipy.stats import chi2_contingency
    labels = np.asarray(labels)
    cond = np.asarray(condition)
    cidx = {c: i for i, c in enumerate(CONDITIONS)}
    ci = np.array([cidx.get(c, -1) for c in cond])
    valid = ci >= 0
    clusters = sorted(c for c in set(labels[valid]) if c >= 0)
    tested = [c for c in clusters if (valid & (labels == c)).sum() >= min_n]
    results = []
    for c in tested:
        inc = valid & (labels == c)
        out = valid & (labels != c)
        in_counts = np.bincount(ci[inc], minlength=3)
        out_counts = np.bincount(ci[out], minlength=3)
        chi2, p, dof, exp = chi2_contingency(np.vstack([in_counts, out_counts]))
        enr = CONDITIONS[int(np.argmax((in_counts - exp[0]) / np.sqrt(exp[0])))]
        results.append(dict(cluster=int(c), n=int(in_counts.sum()), chi2=float(chi2), p=float(p),
                            enriched=enr,
                            fracs={CONDITIONS[i]: float(in_counts[i] / max(1, in_counts.sum()))
                                   for i in range(3)}))
    m = max(1, len(tested))
    for r in results:
        r["p_bonf"] = min(1.0, r["p"] * m)
        r["sig"] = r["p_bonf"] < 0.05
    return sorted(results, key=lambda r: r["p"])


# ============================================================================ skeleton-GIF rendering
def _event_bbox(kp_event, pad=40):
    pts = kp_event.reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(1)]
    if len(pts) < 2:
        return 0.0, 0.0, 1.0, 1.0
    x0, y0 = pts.min(0) - pad
    x1, y1 = pts.max(0) + pad
    s = max(x1 - x0, y1 - y0)                       # square, so aspect is preserved
    return float(x0), float(y0), float(s), float(s)


def render_frames(kp_event, ranks, contact_rel=0, cell=200, arrow=True):
    """kp_event (T,3,15,2) world coords ordered [appr,appe,by]; ranks (3,). Draw the rank-colored
    skeletons on a blank canvas (no video needed) and return a list of (cell,cell,3) uint8 frames.
    A white arrow points approacher->approachee; a red dot marks frames at/after contact onset."""
    from PIL import Image, ImageDraw
    T = kp_event.shape[0]
    x0, y0, sx, sy = _event_bbox(kp_event)
    cols = [RANK_RGB.get(int(ranks[m]), RANK_RGB[0]) for m in range(3)]

    def to_px(p):
        return ((p[0] - x0) / sx * cell, (p[1] - y0) / sy * cell)

    frames = []
    for t in range(T):
        img = Image.new("RGB", (cell, cell), (245, 245, 247))
        dr = ImageDraw.Draw(img)
        for m in range(3):
            kp = kp_event[t, m]
            ok = np.isfinite(kp).all(1)
            for u, v in SKELETON_EDGES:
                if ok[u] and ok[v]:
                    dr.line([to_px(kp[u]), to_px(kp[v])], fill=cols[m], width=2)
            for n in range(N_NODES):
                if ok[n]:
                    px, py = to_px(kp[n])
                    r = 3 if n in (HEAD, TTI) else 2
                    dr.ellipse([px - r, py - r, px + r, py + r], fill=cols[m])
        if arrow:
            ca = _centroids(kp_event[t:t + 1, 0])[0]
            cb = _centroids(kp_event[t:t + 1, 1])[0]
            if np.isfinite(ca).all() and np.isfinite(cb).all():
                dr.line([to_px(ca), to_px(cb)], fill=(255, 255, 255), width=2)
        if t >= contact_rel:
            dr.ellipse([cell - 16, 6, cell - 6, 16], fill=(220, 40, 40))
        frames.append(np.asarray(img))
    return frames


def gif_bytes(frames, fps=20):
    """Encode a list of RGB frames to GIF bytes (embeddable in a notebook)."""
    import imageio.v2 as imageio
    buf = io.BytesIO()
    imageio.mimsave(buf, frames, format="GIF", duration=1.0 / fps, loop=0)
    return buf.getvalue()


def event_gif_bytes(kp_event, ranks, contact_rel=0, cell=200, fps=20):
    return gif_bytes(render_frames(kp_event, ranks, contact_rel, cell), fps=fps)


def gif_img_html(gif, width=200, border="#ddd"):
    """Wrap GIF bytes in an <img> data-URI so it animates when embedded in a marimo cell
    (mo.md / mo.Html). marimo's static image widget would freeze the first frame."""
    uri = "data:image/gif;base64," + base64.b64encode(gif).decode()
    return (f'<img src="{uri}" width="{width}" '
            f'style="border:1px solid {border};border-radius:6px;margin:3px">')


def load_asset_gif(name, root=None):
    """Return the raw bytes of a pre-rendered exemplar GIF from data/exemplar_gifs/.

    These are small VIDEO-backed clips (real homecage frames + the rank-colored skeleton overlaid),
    committed to the repo so notebooks show what the behaviors actually look like. Reads the local
    file if present, else downloads it from REPO_RAW (so a bare cloud kernel / molab works) via the
    same mechanism as data_path. `name` is the bare filename, e.g. "tail_bite_1.gif". Pair with
    gif_img_html: ``mo.Html(cu.gif_img_html(cu.load_asset_gif("tail_bite_1.gif")))``."""
    rel = "data/exemplar_gifs/" + os.path.basename(name)
    with open(data_path(rel, root), "rb") as f:
        return f.read()


def grid_gif_bytes(events, ncols=3, cell=170, fps=20, pad=6):
    """events: list of (kp_event, ranks, contact_rel). Tile them into an ncols grid GIF, looping
    over the shortest common length. Returns GIF bytes."""
    clips = [render_frames(kp, rk, cr, cell) for kp, rk, cr in events]
    if not clips:
        return gif_bytes([np.full((cell, cell, 3), 245, np.uint8)], fps)
    T = min(len(c) for c in clips)
    n = len(clips)
    nrows = (n + ncols - 1) // ncols
    H = nrows * cell + (nrows + 1) * pad
    W = ncols * cell + (ncols + 1) * pad
    out = []
    for t in range(T):
        canvas = np.full((H, W, 3), 255, np.uint8)
        for i, clip in enumerate(clips):
            r, c = divmod(i, ncols)
            y = pad + r * (cell + pad)
            x = pad + c * (cell + pad)
            canvas[y:y + cell, x:x + cell] = clip[t]
        out.append(canvas)
    return gif_bytes(out, fps=fps)


# ============================================================================ MLP behavior classifier
def make_mlp(hidden=(64, 32), alpha=1e-3, max_iter=400, seed=0):
    """Standardize -> small MLP. sklearn only (no GPU). Returns an unfitted Pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPClassifier
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=max_iter,
                              random_state=seed)),
    ])


def eval_binary(y_true, y_score, thr=0.5):
    """ROC-AUC, average precision, and the confusion matrix at a threshold."""
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= thr).astype(int)
    out = {"n": int(len(y_true)), "n_pos": int(y_true.sum()),
           "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()}
    if len(set(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["avg_precision"] = float(average_precision_score(y_true, y_score))
    return out


# ============================================================================ derived-bundle loaders
def load_derived(tag="train", root=None):
    """Precomputed metadata + features + PCA aligned row-for-row with {tag}_events.npz
    (tools/build_derived.py). Keys: cage, sex, tod_hour, X (N,19), pca_scores (N,10),
    feature_names; the train file also has evr, pca_components, pca_mean, pca_std."""
    z = np.load(data_path(f"data/{tag}_derived.npz", root), allow_pickle=True)
    return {k: z[k] for k in z.files}


def load_continuous_tracks(cam="15", root=None):
    """One cage's continuous 24h span at 2 fps (tools/build_continuous_tracks.py). Returns dict:
    centroids (T,3,2), speed (T,3) px/s, tod_hour (T,), state_seq (T,) int, state_names, fps, sex."""
    z = np.load(data_path("data/continuous_tracks.npz", root), allow_pickle=True)
    cam = str(cam)
    sex = dict(zip(z["cams"].astype(str), z["sex"].astype(str)))
    return dict(centroids=z[f"cam{cam}_cen"].astype(np.float32),
                speed=z[f"cam{cam}_speed"].astype(np.float32),
                tod_hour=z[f"cam{cam}_tod"].astype(np.float32),
                state_seq=z[f"cam{cam}_state"].astype(int),
                state_names=[str(s) for s in z["state_names"]],
                fps=float(z["fps"]), sex=sex.get(cam, "?"), cam=cam)


def load_neural_demo(root=None):
    """Synthetic population raster for NB08 (tools/build_neural_demo.py). Keys: X_neural
    (trials,neurons) counts, y (trials,) hidden state, is_tuned (neurons,), emb2d (trials,2)."""
    z = np.load(data_path("data/neural_demo.npz", root), allow_pickle=True)
    return {k: z[k] for k in z.files}


# ============================================================================ pose QC
def node_reliability(kp):
    """Fraction of frames each of the 15 nodes is finite (tracked). kp (...,15,2) -> (15,)."""
    kp = np.asarray(kp, float)
    ok = np.isfinite(kp).all(axis=-1)                       # (...,15)
    return ok.reshape(-1, kp.shape[-2]).mean(axis=0)


def centroid_jumps(kp_event):
    """Per-track body-centroid displacement between frames — big spikes flag identity swaps.
    kp_event (T,3,15,2) -> (3, T-1) px/frame."""
    cen = np.stack([_centroids(kp_event[:, m]) for m in range(kp_event.shape[1])], axis=0)  # (3,T,2)
    return np.linalg.norm(np.diff(cen, axis=1), axis=2)     # (3,T-1)


# ============================================================================ time-series tools (numpy-only)
def wavelet_power(sig, freqs, fps, w0=6.0):
    """Morlet continuous-wavelet POWER of a 1-D signal (pure numpy; no pywt). Convolve the signal
    with a complex Morlet wavelet tuned to each frequency and take |.|^2.
    Inputs:  sig (T,) real signal; freqs (F,) Hz; fps sampling rate.
    Returns: (F, T) power — a spectrogram showing which rhythms are present when."""
    sig = np.asarray(sig, float)
    sig = np.nan_to_num(sig - np.nanmean(sig))
    T = len(sig)
    out = np.empty((len(freqs), T))
    dt = 1.0 / fps
    for i, f in enumerate(freqs):
        s = w0 / (2 * np.pi * f) / dt                       # wavelet scale in samples
        n = int(np.ceil(s * 6))
        t = np.arange(-n, n + 1)
        psi = np.exp(1j * w0 * t / s) * np.exp(-0.5 * (t / s) ** 2)
        psi /= (np.sqrt(s) * np.pi ** 0.25)                 # energy normalization
        out[i] = np.abs(np.convolve(sig, psi, mode="same")) ** 2
    return out


def cross_corr_lag(x, y, max_lag):
    """Normalized cross-correlation of x,y over integer lags [-max_lag, max_lag].
    Returns (lags, corr, peak_lag). peak_lag > 0 means x LEADS y (y follows x)."""
    x = np.nan_to_num(np.asarray(x, float)); y = np.nan_to_num(np.asarray(y, float))
    x = (x - x.mean()) / (x.std() + 1e-12); y = (y - y.mean()) / (y.std() + 1e-12)
    n = len(x); lags = np.arange(-max_lag, max_lag + 1)
    corr = np.empty(len(lags))
    for i, k in enumerate(lags):
        a, b = (x[:n - k], y[k:]) if k >= 0 else (x[-k:], y[:n + k])
        corr[i] = np.mean(a * b) if len(a) > 1 else 0.0
    peak = int(lags[np.argmax(corr)])
    return lags, corr, peak


def granger_pair(x, y, lags=5):
    """Pairwise Granger causality via a numpy VAR F-test (no statsmodels). Does the PAST of one
    series improve prediction of the other beyond its own past?
    Returns {f_xy,p_xy,f_yx,p_yx}: 'xy' = x->y (x helps predict y). Small p => directed influence."""
    from scipy.stats import f as fdist
    x = np.nan_to_num(np.asarray(x, float)); y = np.nan_to_num(np.asarray(y, float))

    def _test(target, source):
        L = lags; Y = target[L:]; rows = len(Y)
        own = np.column_stack([target[L - k - 1:len(target) - k - 1] for k in range(L)])
        ext = np.column_stack([source[L - k - 1:len(source) - k - 1] for k in range(L)])
        Xr = np.column_stack([np.ones(rows), own])
        Xu = np.column_stack([np.ones(rows), own, ext])
        rss = lambda X: float(np.sum((Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]) ** 2))
        rss_r, rss_u = rss(Xr), rss(Xu)
        df1, df2 = L, rows - Xu.shape[1]
        F = ((rss_r - rss_u) / df1) / (rss_u / df2 + 1e-12)
        return float(F), float(fdist.sf(F, df1, df2))
    f_xy, p_xy = _test(y, x)
    f_yx, p_yx = _test(x, y)
    return dict(f_xy=f_xy, p_xy=p_xy, f_yx=f_yx, p_yx=p_yx)


# ============================================================================ effect size / enrichment
def cohens_d(X, y):
    """Per-feature Cohen's d (standardized mean difference) between the two groups in binary y.
    X (N,F), y (N,) -> (F,). |d|~0.2 small, 0.5 medium, 0.8 large."""
    X = np.asarray(X, float); y = np.asarray(y).astype(bool)
    a, b = X[y], X[~y]
    pooled = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1)) /
                     (len(a) + len(b) - 2) + 1e-12)
    return (a.mean(0) - b.mean(0)) / pooled


def covariate_enrichment(in_group, covariate):
    """Chi-square test: is a categorical `covariate` distributed differently INSIDE vs OUTSIDE the
    boolean mask `in_group` (e.g. one cluster vs the rest)? EVENT-LEVEL — the naive test NB06 shows
    can mislead under pseudoreplication. Returns {chi2,p,dof,levels,residuals}."""
    from scipy.stats import chi2_contingency
    g = np.asarray(in_group).astype(bool); cov = np.asarray(covariate)
    levels = sorted(set(cov.tolist()))
    inc = np.array([np.sum(cov[g] == L) for L in levels])
    out = np.array([np.sum(cov[~g] == L) for L in levels])
    table = np.vstack([inc, out]); keep = table.sum(0) > 0
    chi2, p, dof, exp = chi2_contingency(table[:, keep])
    resid = (table[0, keep] - exp[0]) / np.sqrt(exp[0])
    return dict(chi2=float(chi2), p=float(p), dof=int(dof),
                levels=[L for L, k in zip(levels, keep) if k], residuals=resid)


def permutation_test(in_group, covariate, unit, n=5000, seed=0):
    """Honest enrichment p-value that respects clustering by `unit` (e.g. cage). The covariate is
    assumed constant within a unit (sex, cage). We permute the covariate label AT THE UNIT LEVEL,
    preserving within-unit structure, and compare |mean(in)-mean(out)| to the null. Returns
    {stat, p_emp}. This is NB06's antidote to pseudoreplication."""
    rng = np.random.RandomState(seed)
    g = np.asarray(in_group).astype(bool); unit = np.asarray(unit)
    _, cov_num = np.unique(np.asarray(covariate), return_inverse=True)
    stat = lambda c: abs(c[g].mean() - c[~g].mean())
    obs = stat(cov_num.astype(float))
    units = np.unique(unit)
    vals = np.array([cov_num[unit == u][0] for u in units], float)
    count = 0
    for _ in range(n):
        mapping = dict(zip(units, rng.permutation(vals)))
        if stat(np.array([mapping[u] for u in unit], float)) >= obs - 1e-12:
            count += 1
    return dict(stat=float(obs), p_emp=(count + 1) / (n + 1))


# ============================================================================ Markov / behavioral grammar
def discretize_states(speed, centroids, s_move=None, d_close=None):
    """Label each timepoint with a cage-level behavioral state from kinematics.
    speed (T,3) px/s, centroids (T,3,2). Thresholds default to data percentiles (mean-speed 40th,
    min-pair-distance 25th). Returns (state_seq (T,) int, state_names). 0 rest, 1 locomote, 2 huddle."""
    speed = np.asarray(speed, float); cen = np.asarray(centroids, float)
    mean_speed = np.nanmean(speed, axis=1)
    dif = cen[:, :, None, :] - cen[:, None, :, :]
    iu = np.triu_indices(cen.shape[1], k=1)
    min_pair = np.nanmin(np.linalg.norm(dif, axis=3)[:, iu[0], iu[1]], axis=1)
    if s_move is None:
        s_move = np.nanpercentile(mean_speed, 40)
    if d_close is None:
        d_close = np.nanpercentile(min_pair, 25)
    state = np.ones(len(mean_speed), int)
    state[mean_speed < s_move] = 0
    state[min_pair < d_close] = 2
    return state, ["rest", "locomote", "huddle"]


def transition_matrix(state_seq, n_states=None):
    """Row-stochastic Markov transition matrix. Returns (K,K): T[i,j] = P(next=j | now=i)."""
    s = np.asarray(state_seq, int)
    K = n_states or int(s.max() + 1)
    M = np.zeros((K, K))
    np.add.at(M, (s[:-1], s[1:]), 1)
    row = M.sum(1, keepdims=True)
    return np.divide(M, row, out=np.zeros_like(M), where=row > 0)


def stationary_dist(T, method="simulate", steps=50000, seed=0):
    """Long-run fraction of time spent in each state. Default 'simulate' runs a random walk on T
    (the intuitive definition — no eigen-decomposition). 'eig' returns the leading left eigenvector."""
    T = np.asarray(T, float); K = T.shape[0]
    if method == "eig":
        w, v = np.linalg.eig(T.T)
        pi = np.abs(np.real(v[:, np.argmin(np.abs(w - 1))]))
        return pi / pi.sum()
    rng = np.random.RandomState(seed)
    s = 0; counts = np.zeros(K)
    for _ in range(steps):
        counts[s] += 1
        s = rng.choice(K, p=T[s]) if T[s].sum() > 0 else rng.randint(K)
    return counts / counts.sum()


def transition_entropy(T):
    """Average uncertainty of the next state (bits), weighted by the stationary distribution.
    0 = perfectly predictable grammar; log2(K) = memoryless/uniform."""
    T = np.asarray(T, float)
    pi = stationary_dist(T, method="eig")
    with np.errstate(divide="ignore", invalid="ignore"):
        row_H = -np.nansum(np.where(T > 0, T * np.log2(T), 0.0), axis=1)
    return float(np.sum(pi * row_H))


def shuffle_transition_null(state_seq, n=1000, seed=0, stat="entropy"):
    """Null distribution of a transition statistic when temporal ORDER is destroyed (shuffle the
    sequence n times). Compare the real value to this to prove the grammar beats chance. stat =
    'entropy' (transition_entropy) or 'self' (mean self-transition prob). Returns (n,)."""
    rng = np.random.RandomState(seed)
    s = np.asarray(state_seq, int); K = int(s.max() + 1)
    out = np.empty(n)
    for i in range(n):
        M = transition_matrix(rng.permutation(s), K)
        out[i] = transition_entropy(M) if stat == "entropy" else float(np.mean(np.diag(M)))
    return out


# ============================================================================ time-of-day / activity clock
def time_of_day(event_key):
    """Approximate time-of-day (hours in [0,24)) an event occurred, from its event_key. Reverse
    cycle: lights ON 21:00-09:00, so the DARK/active phase is ~09:00-21:00."""
    import re
    _, stem, _, cf = str(event_key).split("|")
    h = int(re.search(r"T(\d{2})", stem).group(1))
    return (h + int(cf) / FPS / 3600.0) % 24.0


def activity_by_tod(speed, tod_hour, bin_min=30, n_boot=200, seed=0):
    """Circadian activity curve: mean movement speed binned by time-of-day, with a bootstrap 95% CI.
    speed (T,3) or (T,); tod_hour (T,) in [0,24). Returns {centers, curve, ci_low, ci_high}."""
    rng = np.random.RandomState(seed)
    sp = np.asarray(speed, float)
    a = np.nanmean(sp, axis=1) if sp.ndim == 2 else sp
    tod = np.asarray(tod_hour, float)
    edges = np.arange(0, 24 + 1e-9, bin_min / 60.0)
    centers = (edges[:-1] + edges[1:]) / 2
    curve = np.full(len(centers), np.nan); lo = np.full_like(curve, np.nan); hi = np.full_like(curve, np.nan)
    which = np.clip(np.digitize(tod, edges) - 1, 0, len(centers) - 1)
    for b in range(len(centers)):
        vals = a[which == b]; vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            continue
        curve[b] = vals.mean()
        boot = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n_boot)]
        lo[b], hi[b] = np.percentile(boot, [2.5, 97.5])
    return dict(centers=centers, curve=curve, ci_low=lo, ci_high=hi)


# ============================================================================ agreement / calibration / decoding
def cohens_kappa(a, b):
    """Cohen's kappa: inter-rater agreement corrected for chance agreement. a,b (N,) -> float
    (1 perfect, 0 chance)."""
    a = np.asarray(a); b = np.asarray(b)
    labels = sorted(set(a.tolist()) | set(b.tolist()))
    idx = {L: i for i, L in enumerate(labels)}; K = len(labels)
    M = np.zeros((K, K))
    for x, y in zip(a, b):
        M[idx[x], idx[y]] += 1
    N = M.sum(); po = np.trace(M) / N
    pe = np.sum(M.sum(0) * M.sum(1)) / N ** 2
    return float((po - pe) / (1 - pe + 1e-12))


def calibration_curve(y, scores, n_bins=10):
    """Reliability curve: bin predicted probabilities, return (frac_positive, mean_pred) per
    non-empty bin. A calibrated classifier lies on the diagonal."""
    y = np.asarray(y).astype(float); scores = np.asarray(scores, float)
    edges = np.linspace(0, 1, n_bins + 1)
    which = np.clip(np.digitize(scores, edges) - 1, 0, n_bins - 1)
    frac, mean = [], []
    for b in range(n_bins):
        m = which == b
        if m.any():
            frac.append(float(y[m].mean())); mean.append(float(scores[m].mean()))
    return np.array(frac), np.array(mean)


def synthetic_population_raster(n_neurons=60, n_trials=800, n_tuned=18, seed=7):
    """Toy neural population: spike counts (trials,neurons) driven by a hidden binary state y.
    Mirrors tools/build_neural_demo.py so students can regenerate/vary it. Returns (X, y, is_tuned)."""
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, n_trials)
    base = rng.uniform(1.0, 4.0, n_neurons)
    tuned = np.zeros(n_neurons, int); tuned[rng.choice(n_neurons, n_tuned, replace=False)] = 1
    sign = rng.choice([-1, 1], n_neurons)
    gain = 1.0 + tuned * sign * rng.uniform(0.4, 1.2, n_neurons)
    rates = base[None, :] * np.where(y[:, None] == 1, gain[None, :], 1.0)
    return rng.poisson(np.clip(rates, 0.05, None)).astype(int), y, tuned


def pca_loadings_fig(components, feature_names, k=6):
    """Heatmap of the first k PC loadings across features — shows what each component 'means'.

    Components are 1-INDEXED (PC1..PCk) to match the notebook prose. The colorscale is a diverging
    RdBu_r centered at 0, so RED = a POSITIVE loading (the feature pushes an event's score UP on that
    PC) and BLUE = negative — the same red-is-up convention the profile heatmaps use, so the two never
    contradict. Features run along the x-axis, PC labels down the y-axis. `k` defaults to 6."""
    import plotly.graph_objects as go
    C = np.asarray(components)[:k]
    k = C.shape[0]
    fig = go.Figure(go.Heatmap(z=C, x=list(feature_names), y=[f"PC{i + 1}" for i in range(k)],
                               colorscale="RdBu_r", zmid=0,
                               colorbar=dict(title="loading")))
    fig.update_yaxes(autorange="reversed")     # PC1 at the top
    fig.update_layout(template="plotly_white", height=110 + 55 * k,
                      title="PCA loadings — how features combine into components (red = pushes score up)",
                      margin=dict(l=10, r=10, t=40, b=130))
    return fig


def labelled_skeleton_fig(pose_2d, node_names=None, edges=None, color="#d62728",
                          reverse_y=True, height=520,
                          title="SLEAP skeleton — 15 named keypoints"):
    """A single skeleton drawn on a blank canvas with EVERY node's NAME printed beside it.

    Purpose: the "what am I looking at?" reference figure for NB01 — no mouse photo exists in the
    bundle, so this labelled diagram is the fallback that teaches the skeleton layout.

    Inputs:
        pose_2d    (15, 2) float array — one mouse, one frame, (x, y) pixel coords. NaN nodes are
                   skipped (not drawn, not labelled).
        node_names list of 15 str (defaults to cu.NODE_NAMES).
        edges      list of (src, dst) int pairs (defaults to cu.SKELETON_EDGES).
        color      single hex color for the whole skeleton — use the rank color, e.g.
                   cu.RANK_HEX[1] (Dom=red), so mouse coloring stays consistent with every notebook.
        reverse_y  True for raw image coords (y grows downward); the y-axis is flipped so the drawing
                   is upright.
    Output:
        a plotly.graph_objects.Figure (edges + node markers + one text label per finite node)."""
    import plotly.graph_objects as go
    node_names = list(node_names) if node_names is not None else list(NODE_NAMES)
    edges = list(edges) if edges is not None else list(SKELETON_EDGES)
    p = np.asarray(pose_2d, float)
    ok = np.isfinite(p).all(axis=1)
    ex, ey = [], []
    for u, v in edges:
        if ok[u] and ok[v]:
            ex += [p[u, 0], p[v, 0], None]
            ey += [p[u, 1], p[v, 1], None]
    fig = go.Figure()
    fig.add_scatter(x=ex, y=ey, mode="lines", line=dict(color=color, width=2),
                    hoverinfo="skip", showlegend=False)
    idx = np.where(ok)[0]
    fig.add_scatter(x=p[idx, 0], y=p[idx, 1], mode="markers+text",
                    marker=dict(color=color, size=9),
                    text=[f"{i} {node_names[i]}" for i in idx],
                    textposition="middle right", textfont=dict(size=11),
                    hoverinfo="skip", showlegend=False)
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False,
                     scaleanchor="x", scaleratio=1,
                     autorange="reversed" if reverse_y else True)
    fig.update_layout(template="plotly_white", height=height, title=title,
                      margin=dict(l=10, r=90, t=40, b=10))
    return fig


def roc_pr_fig(y, scores):
    """Side-by-side ROC and precision-recall curves for a binary decoder."""
    from plotly.subplots import make_subplots
    from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                                 average_precision_score)
    y = np.asarray(y).astype(int)
    fpr, tpr, _ = roc_curve(y, scores); prec, rec, _ = precision_recall_curve(y, scores)
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        f"ROC (AUC={roc_auc_score(y, scores):.3f})", f"PR (AP={average_precision_score(y, scores):.3f})"))
    fig.add_scatter(x=fpr, y=tpr, mode="lines", row=1, col=1, line=dict(color="#4c78a8"), showlegend=False)
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", row=1, col=1, line=dict(dash="dot", color="#bbb"),
                    showlegend=False)
    fig.add_scatter(x=rec, y=prec, mode="lines", row=1, col=2, line=dict(color="#e45756"), showlegend=False)
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ============================================================================ seaborn-style interactive plotly
# Round-3 directive 5: stop defaulting to bar charts. Every distribution comparison shows the RAW
# individual points, interactively (hover), in the house style. These are seaborn-equivalent displays
# (strip / violin+points / box+points / 2-D KDE / ECDF) built in pure plotly+numpy+scipy, plus a
# UMAP-feature-overlay scatter so the map axes can be given meaning (directive 7).
_QUAL_PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
                 "#eeca3b", "#b279a2", "#ff9da6", "#9d755d", "#bab0ac"]


def _group_colors(order, colors=None):
    """Map each group label -> hex color. Precedence: explicit `colors` dict (keyed by the label or
    its str) > rank auto-detect (Dom/Mid/Int/Sub -> cu.RANK_HEX, so mouse-rank coloring stays the
    house scheme) > a cycled qualitative palette. `order` is the ordered list of unique group labels."""
    rank_by_name = {"Dom": RANK_HEX[1], "Mid": RANK_HEX[2], "Int": RANK_HEX[2],
                    "Sub": RANK_HEX[3], "?": RANK_HEX[0], "unknown": RANK_HEX[0]}
    out = {}
    for i, g in enumerate(order):
        gs = str(g)
        if colors and g in colors:
            out[g] = colors[g]
        elif colors and gs in colors:
            out[g] = colors[gs]
        elif gs in rank_by_name:
            out[g] = rank_by_name[gs]
        else:
            out[g] = _QUAL_PALETTE[i % len(_QUAL_PALETTE)]
    return out


def _group_order(groups, group_order=None):
    return list(group_order) if group_order is not None else list(dict.fromkeys(np.asarray(groups).tolist()))


def _darken(hexcolor, factor=0.7):
    """Return a hex color scaled toward black by `factor` (0=black, 1=unchanged). Used to give
    overlaid raw points a slightly darker fill than their group's fill silhouette so they read on
    top of it."""
    h = str(hexcolor).lstrip("#")
    if len(h) != 6:
        return hexcolor
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % (int(r * factor), int(g * factor), int(b * factor))


def _robust_range(v, lo=1.0, hi=99.0, pad=0.05):
    """Return [low, high] for a VISIBLE axis clipped to the [lo, hi] percentiles of v (default 1/99)
    with a little padding — so a handful of extreme outliers don't flatten the rest of the cloud.
    Points outside this window are still plotted, just off the default view. Returns None if v has
    too few finite values or a degenerate spread (caller then leaves the axis auto-ranged)."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if v.size < 3:
        return None
    a, b = (float(z) for z in np.nanpercentile(v, [lo, hi]))
    if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
        a, b = float(np.nanmin(v)), float(np.nanmax(v))
        if b <= a:
            return None
    span = b - a
    return [a - span * pad, b + span * pad]


def strip_points_fig(values, groups, group_order=None, colors=None, jitter=0.09,
                     point_size=6, opacity=0.7, show_mean=True, hover=None,
                     title="", xlabel="", ylabel="value", height=430, seed=0, robust=True):
    """Categorical strip plot: EVERY individual data point, jittered horizontally, colored by group,
    with a hover readout — the honest replacement for a bar-of-means. A short horizontal line marks
    each group mean. `values` (N,) numeric; `groups` (N,) categorical labels; optional `hover` (N,)
    per-point text (e.g. event index). `robust` clips the value axis to the [1, 99] percentile so
    outliers don't skew the view (points still plotted). Returns a plotly Figure.

    Pass `show_mean=False` for LOCO-style groups that hold a single value each (one point per cohort):
    a mean line over one point is a redundant tick that reads as a bar and is best suppressed."""
    import plotly.graph_objects as go
    values = np.asarray(values, float); groups = np.asarray(groups)
    order = _group_order(groups, group_order)
    cmap = _group_colors(order, colors)
    rng = np.random.RandomState(seed)
    hv = None if hover is None else np.asarray(hover)
    fig = go.Figure()
    for i, g in enumerate(order):
        m = groups == g
        yv = values[m]; keep = np.isfinite(yv); yv = yv[keep]
        x = i + (rng.rand(len(yv)) - 0.5) * 2 * jitter
        txt = [str(t) for t in hv[m][keep]] if hv is not None else None
        fig.add_scatter(x=x, y=yv, mode="markers", name=str(g),
                        marker=dict(size=point_size, color=cmap[g], opacity=opacity,
                                    line=dict(width=0.5, color="white")),
                        text=txt,
                        hovertemplate=(("%{text}<br>" if txt is not None else "") +
                                       f"{g}: %{{y:.3f}}<extra></extra>"))
        if show_mean and len(yv):
            mu = float(np.nanmean(yv))
            fig.add_scatter(x=[i - 0.28, i + 0.28], y=[mu, mu], mode="lines",
                            line=dict(color=cmap[g], width=3), showlegend=False,
                            hovertemplate=f"{g} mean: {mu:.3f}<extra></extra>")
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(order))),
                     ticktext=[str(g) for g in order], title=xlabel)
    fig.update_yaxes(title=ylabel)
    if robust:
        ry = _robust_range(values)
        if ry:
            fig.update_yaxes(range=ry)
    fig.update_layout(template="plotly_white", height=height, title=title,
                      margin=dict(l=10, r=10, t=50, b=10), showlegend=len(order) > 1)
    return fig


def violin_points_fig(values, groups, group_order=None, colors=None, points="all",
                      show_box=True, title="", xlabel="", ylabel="value", height=450,
                      robust=True, seed=0):
    """Violin (kernel-density silhouette) per group with the raw points overlaid ON TOP and a mean
    line — shows the full shape of each distribution AND every observation. `robust` clips the value
    axis to the [1, 99] percentile (outliers still plotted). Args as `strip_points_fig`.

    CAVEAT — pick the right chart. A violin is only honest when n is large (>~50 per group) AND the
    distributions differ in SHAPE. For small n, a small mean shift, or heavy tails the kernel density
    invents structure (spurious bimodality, mass past impossible values) and the effect hides inside
    the silhouette — prefer ``ecdf_fig`` (tail comparison), ``strip_points_fig`` (every point, small
    n) or ``dumbbell_fig`` (paired data) instead.

    The raw points are drawn as a SEPARATE jittered scatter beside each half-violin (not inside the
    translucent fill) with a white outline and a darkened marker, so they read clearly on top of the
    0.2-opacity silhouette rather than vanishing under it."""
    import plotly.graph_objects as go
    values = np.asarray(values, float); groups = np.asarray(groups)
    order = _group_order(groups, group_order)
    cmap = _group_colors(order, colors)
    rng = np.random.RandomState(seed)
    fig = go.Figure()
    for i, g in enumerate(order):
        yv = values[groups == g]; yv = yv[np.isfinite(yv)]
        # faint half-violin + box, no points inside it (points go on their own trace on top)
        fig.add_trace(go.Violin(y=yv, name=str(g), line_color=cmap[g], fillcolor=cmap[g],
                                opacity=0.2, points=False, side="positive", width=0.9,
                                box_visible=show_box, meanline_visible=True,
                                x=np.full(len(yv), i, float)))
        if points and len(yv):
            jx = -0.18 + (rng.rand(len(yv)) - 0.5) * 0.14      # jittered, beside the violin
            fig.add_scatter(x=np.full(len(yv), i, float) + jx, y=yv, mode="markers",
                            name=str(g), legendgroup=str(g), showlegend=False,
                            marker=dict(size=5, color=_darken(cmap[g]), opacity=0.85,
                                        line=dict(width=0.8, color="white")),
                            hovertemplate=f"{g}: %{{y:.3f}}<extra></extra>")
    fig.update_yaxes(title=ylabel)
    fig.update_xaxes(title=xlabel, tickmode="array", tickvals=list(range(len(order))),
                     ticktext=[str(g) for g in order], range=[-0.6, len(order) - 0.4])
    if robust:
        ry = _robust_range(values)
        if ry:
            fig.update_yaxes(range=ry)
    fig.update_layout(template="plotly_white", height=height, title=title,
                      margin=dict(l=10, r=10, t=50, b=10), showlegend=len(order) > 1)
    return fig


def box_points_fig(values, groups, group_order=None, colors=None, title="",
                   xlabel="", ylabel="value", height=450, robust=True):
    """Box-and-whisker per group with ALL points overlaid (jittered) — quartiles plus every raw
    observation. `robust` clips the value axis to the [1, 99] percentile (outliers still plotted).
    Args as `strip_points_fig`."""
    import plotly.graph_objects as go
    values = np.asarray(values, float); groups = np.asarray(groups)
    order = _group_order(groups, group_order)
    cmap = _group_colors(order, colors)
    fig = go.Figure()
    for g in order:
        yv = values[groups == g]; yv = yv[np.isfinite(yv)]
        fig.add_trace(go.Box(y=yv, name=str(g), boxpoints="all", jitter=0.4, pointpos=0,
                             marker=dict(size=4, color=cmap[g], opacity=0.6),
                             line=dict(color=cmap[g]), fillcolor="rgba(0,0,0,0)"))
    fig.update_yaxes(title=ylabel); fig.update_xaxes(title=xlabel)
    if robust:
        ry = _robust_range(values)
        if ry:
            fig.update_yaxes(range=ry)
    fig.update_layout(template="plotly_white", height=height, title=title,
                      margin=dict(l=10, r=10, t=50, b=10), showlegend=len(order) > 1)
    return fig


def kde2d_fig(x, y, gridsize=120, colorscale="Viridis", show_points=True, point_color="#333333",
              hover=None, title="", xlabel="x", ylabel="y", height=480, bw_method=None,
              robust=True):
    """2-D density via scipy.stats.gaussian_kde: a filled contour of where (x, y) pairs concentrate,
    with the raw points optionally overlaid. Use it for two-feature joint distributions (e.g. speed
    vs distance) instead of an opaque scatter. `x`, `y` (N,) numeric. `robust` clips BOTH visible
    axes to the [1, 99] percentile so outliers don't stretch the view (the KDE is still computed on
    the full data; points outside stay plotted). Returns a plotly Figure."""
    import plotly.graph_objects as go
    from scipy.stats import gaussian_kde
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bw_method)
    xi = np.linspace(x.min(), x.max(), gridsize); yi = np.linspace(y.min(), y.max(), gridsize)
    XX, YY = np.meshgrid(xi, yi)
    Z = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    fig = go.Figure(go.Contour(x=xi, y=yi, z=Z, colorscale=colorscale,
                               contours=dict(coloring="fill"), colorbar=dict(title="density")))
    if show_points:
        txt = None if hover is None else [str(t) for t in np.asarray(hover)[ok]]
        fig.add_scatter(x=x, y=y, mode="markers",
                        marker=dict(size=3, color=point_color, opacity=0.35),
                        text=txt, showlegend=False,
                        hovertemplate=(("%{text}<br>" if txt is not None else "") +
                                       "%{x:.2f}, %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title=xlabel); fig.update_yaxes(title=ylabel)
    if robust:
        rx = _robust_range(x); ry = _robust_range(y)
        if rx:
            fig.update_xaxes(range=rx)
        if ry:
            fig.update_yaxes(range=ry)
    fig.update_layout(template="plotly_white", height=height, title=title,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def scatter_points_fig(x, y, groups=None, group_order=None, colors=None, point_size=7,
                       opacity=0.75, hover=None, annotate_r=True, robust=True, trendline=True,
                       title="", xlabel="x", ylabel="y", height=460):
    """Scatter of individual (x, y) points — one dot per observation, with hover — for showing a
    CORRELATION honestly (NOT a density: use kde2d_fig for that). Optionally colored by `groups`.
    `annotate_r` writes the Pearson r (and p, n), computed on all finite pairs, in a corner box;
    `trendline` adds the least-squares fit line; `robust` clips both visible axes to the [1, 99]
    percentile so a few extremes don't flatten the cloud (outlier points are still plotted, just
    off the default view). Returns a plotly Figure."""
    import plotly.graph_objects as go
    from scipy.stats import pearsonr
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    fig = go.Figure()
    if groups is None:
        txt = None if hover is None else [str(t) for t in np.asarray(hover)]
        fig.add_scatter(x=x, y=y, mode="markers", showlegend=False, text=txt,
                        marker=dict(size=point_size, color="#4c78a8", opacity=opacity,
                                    line=dict(width=0.5, color="white")),
                        hovertemplate=(("%{text}<br>" if txt is not None else "") +
                                       "%{x:.3f}, %{y:.3f}<extra></extra>"))
    else:
        grp = np.asarray(groups)
        order = _group_order(grp, group_order); cmap = _group_colors(order, colors)
        hv = None if hover is None else np.asarray(hover)
        for g in order:
            m = grp == g
            txt = [str(t) for t in hv[m]] if hv is not None else None
            fig.add_scatter(x=x[m], y=y[m], mode="markers", name=str(g), text=txt,
                            marker=dict(size=point_size, color=cmap[g], opacity=opacity,
                                        line=dict(width=0.5, color="white")),
                            hovertemplate=(("%{text}<br>" if txt is not None else "") +
                                           f"{g}: %{{x:.3f}}, %{{y:.3f}}<extra></extra>"))
    if ok.sum() >= 3:
        r, p = pearsonr(x[ok], y[ok])
        if trendline:
            b1, b0 = np.polyfit(x[ok], y[ok], 1)
            xr = np.array([float(x[ok].min()), float(x[ok].max())])
            fig.add_scatter(x=xr, y=b0 + b1 * xr, mode="lines", showlegend=False,
                            line=dict(color="#555", width=2, dash="dash"), hoverinfo="skip")
        if annotate_r:
            fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.98, xanchor="left",
                               yanchor="top", showarrow=False,
                               text=f"r = {r:.3f}<br>p = {p:.2g}  (n = {int(ok.sum())})",
                               font=dict(size=13), bgcolor="rgba(255,255,255,0.72)",
                               bordercolor="#ccc", borderwidth=1, align="left")
    fig.update_xaxes(title=xlabel); fig.update_yaxes(title=ylabel)
    if robust:
        rx = _robust_range(x); ry = _robust_range(y)
        if rx:
            fig.update_xaxes(range=rx)
        if ry:
            fig.update_yaxes(range=ry)
    fig.update_layout(template="plotly_white", height=height, title=title,
                      margin=dict(l=10, r=10, t=50, b=10), showlegend=groups is not None)
    return fig


def ecdf_fig(values, groups=None, group_order=None, colors=None, title="",
             xlabel="value", ylabel="cumulative fraction", height=430, robust=True):
    """Empirical cumulative distribution function, one step curve per group: F(v) = fraction of the
    group at or below v. A crisp way to compare whole distributions (stochastic dominance) without
    binning. `groups=None` plots a single curve. `robust` (default True) clips the x-axis to the
    [1, 99] percentile of the POOLED finite values so a heavy tail doesn't waste most of the panel on
    empty whitespace (the full curves are still drawn; only the default view is trimmed). Returns a
    plotly Figure."""
    import plotly.graph_objects as go
    values = np.asarray(values, float)
    groups = np.zeros(len(values), int) if groups is None else np.asarray(groups)
    order = _group_order(groups, group_order)
    cmap = _group_colors(order, colors)
    fig = go.Figure()
    for g in order:
        yv = np.sort(values[groups == g][np.isfinite(values[groups == g])])
        if not len(yv):
            continue
        cy = np.arange(1, len(yv) + 1) / len(yv)
        fig.add_scatter(x=yv, y=cy, mode="lines", name=str(g),
                        line=dict(color=cmap[g], width=2, shape="hv"))
    fig.update_xaxes(title=xlabel); fig.update_yaxes(title=ylabel, range=[0, 1.02])
    if robust:
        rx = _robust_range(values)
        if rx:
            fig.update_xaxes(range=rx)
    fig.update_layout(template="plotly_white", height=height, title=title,
                      margin=dict(l=10, r=10, t=50, b=10), showlegend=len(order) > 1)
    return fig


def umap_colored_by_feature_fig(emb, feature_values, name="feature", colorscale="Viridis",
                                point_size=5, opacity=0.85, hover=None, title=None,
                                height=520, robust=True):
    """Scatter of a PRECOMPUTED 2-D embedding (emb (N,2)) colored by one continuous feature — the
    tool for giving UMAP axes meaning (directive 7). Overlay each of the 19 features in turn to see
    which vary across the map. `feature_values` (N,) numeric; `name` labels the colorbar. `robust`
    clips the color scale to the 2nd/98th percentile so a few outliers don't wash out the map.
    Never runs UMAP — it only paints the precomputed layout. Returns a plotly Figure."""
    import plotly.graph_objects as go
    emb = np.asarray(emb, float); v = np.asarray(feature_values, float)
    cmin = cmax = None
    if robust and np.isfinite(v).any():
        cmin, cmax = [float(z) for z in np.nanpercentile(v, [2, 98])]
    txt = None if hover is None else [str(t) for t in np.asarray(hover)]
    fig = go.Figure(go.Scattergl(
        x=emb[:, 0], y=emb[:, 1], mode="markers",
        marker=dict(size=point_size, color=v, colorscale=colorscale, cmin=cmin, cmax=cmax,
                    opacity=opacity, colorbar=dict(title=name), line=dict(width=0)),
        text=txt,
        hovertemplate=(("%{text}<br>" if txt is not None else "") +
                       f"{name}=%{{marker.color:.3f}}<extra></extra>")))
    fig.update_xaxes(title="UMAP-1", showticklabels=False)
    fig.update_yaxes(title="UMAP-2", showticklabels=False)
    fig.update_layout(template="plotly_white", height=height,
                      title=title or f"UMAP colored by {name}",
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ============================================================================ UMAP objective teaching toy
def umap_objective_toy(n_per_blob=30, n_blobs=3, dim=8, blob_sep=6.0, blob_std=1.0,
                       n_neighbors=10, n_epochs=250, lr=1.0, snapshot_every=50,
                       grad_clip=4.0, eps=1e-3, seed=0):
    """A tiny, fast, PURE-NUMPY demonstration of what UMAP actually optimizes — no umap-learn, so it
    never trips the no-live-UMAP rule (that rule is about the real 2499-point map; this is a ~90-point
    teaching toy that runs in well under a second).

    It builds `n_blobs` Gaussian blobs in `dim`-dimensional space, converts high-D distances into
    fuzzy neighbor memberships (the high-D graph UMAP tries to reproduce), then runs a short
    attractive/repulsive gradient descent on a 2-D layout so a notebook can WATCH the layout organize
    and can plot the two forces.

    Method (matches UMAP's structure, simplified for teaching):
      * high-D membership p_{j|i} = exp(-max(0, d_ij - rho_i) / sigma_i) over each point's k nearest
        neighbors, where rho_i = distance to the nearest neighbor and sigma_i is set by binary search
        so the row's memberships sum to log2(k); symmetrized to P = p + p^T - p*p^T (the fuzzy union).
      * low-D similarity q_ij = 1 / (1 + ||y_i - y_j||^2) (the Student-t / Cauchy kernel with a=b=1).
      * fuzzy cross-entropy loss CE = -sum P*log q + (1-P)*log(1-q); full-batch gradient
        grad_i = sum_j 2 (y_i - y_j) [ P_ij q_ij - (1-P_ij) q_ij^2/(1-q_ij) ]. The first term is the
        ATTRACTIVE force (neighbors pull together), the second is the REPULSIVE force (everything
        pushes apart); gradients are norm-clipped to `grad_clip` for stability.

    Returns a dict:
      X_high (N,dim), labels (N,) blob id, P (N,N) high-D fuzzy membership,
      snapshots list[(epoch, Y (N,2))] of the optimizing layout, Y_final (N,2),
      loss_history (n_epochs,),
      high_dist, high_membership : per-pair (upper triangle) high-D distance and P — the membership
        curve UMAP fits,
      low_dist, low_membership   : per-pair low-D distance and q at the final layout,
      attractive, repulsive      : per-pair force magnitude (P*q and (1-P)*q^2/(1-q)) aligned with
        low_dist, so a notebook can plot force-vs-distance (neighbors pull, non-neighbors push)."""
    rng = np.random.RandomState(seed)
    # 1. high-D blobs on a sphere of radius blob_sep so they are cleanly separated
    centers = rng.randn(n_blobs, dim)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True) * blob_sep
    X = np.vstack([centers[b] + rng.randn(n_per_blob, dim) * blob_std for b in range(n_blobs)])
    labels = np.repeat(np.arange(n_blobs), n_per_blob)
    N = len(X)
    # 2. high-D pairwise distances
    D = np.sqrt(np.maximum(0.0, ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)))
    # 3. smooth-kNN fuzzy memberships
    k = int(min(n_neighbors, N - 1))
    target = np.log2(k)
    P = np.zeros((N, N))
    for i in range(N):
        nbr = np.argsort(D[i])[1:k + 1]
        d = D[i, nbr]; rho = d[0]
        lo, hi = 1e-3, 1e3
        for _ in range(40):                              # binary search sigma_i
            sig = 0.5 * (lo + hi)
            s = np.exp(-np.maximum(0.0, d - rho) / sig).sum()
            if s > target:
                hi = sig
            else:
                lo = sig
        P[i, nbr] = np.exp(-np.maximum(0.0, d - rho) / sig)
    P = P + P.T - P * P.T                                 # fuzzy union (symmetrize)
    np.fill_diagonal(P, 0.0)
    # 4. optimize a 2-D layout by full-batch fuzzy cross-entropy gradient descent
    Y = rng.randn(N, 2) * 1e-2
    snapshots = [(0, Y.copy())]
    loss_history = []
    for ep in range(1, n_epochs + 1):
        diff = Y[:, None, :] - Y[None, :, :]             # (N,N,2)
        d2 = (diff ** 2).sum(-1)                          # (N,N)
        q = 1.0 / (1.0 + d2)
        np.fill_diagonal(q, 0.0)
        coeff = 2.0 * (P * q - (1.0 - P) * (q * q) / (1.0 - q + eps))
        np.fill_diagonal(coeff, 0.0)
        grad = (coeff[:, :, None] * diff).sum(axis=1)    # (N,2)
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        grad = np.where(gn > grad_clip, grad / gn * grad_clip, grad)
        Y = Y - lr * grad
        with np.errstate(divide="ignore", invalid="ignore"):
            ce = -(P * np.log(q + 1e-12) + (1 - P) * np.log(1 - q + 1e-12))
        loss_history.append(float(np.nansum(np.triu(ce, 1))))
        if ep % snapshot_every == 0 or ep == n_epochs:
            snapshots.append((ep, Y.copy()))
    # 5. membership / force curves (upper triangle, aligned)
    iu = np.triu_indices(N, 1)
    d2f = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    low_d = np.sqrt(d2f)[iu]; low_q = (1.0 / (1.0 + d2f))[iu]
    high_P = P[iu]
    return dict(X_high=X, labels=labels, P=P, snapshots=snapshots, Y_final=Y,
                loss_history=np.asarray(loss_history),
                high_dist=D[iu], high_membership=high_P,
                low_dist=low_d, low_membership=low_q,
                attractive=high_P * low_q,
                repulsive=(1.0 - high_P) * low_q ** 2 / (1.0 - low_q + eps))


def umap_objective_layout_fig(toy, snapshot=-1, title=None, height=460):
    """Scatter one snapshot of `umap_objective_toy(...)`'s optimizing 2-D layout, colored by blob id
    (the ground-truth cluster). `snapshot` indexes toy['snapshots'] (-1 = final). Lets a notebook
    step through epochs and watch the blobs separate."""
    import plotly.graph_objects as go
    ep, Y = toy["snapshots"][snapshot]
    lab = toy["labels"]
    fig = go.Figure()
    for b in sorted(set(lab.tolist())):
        m = lab == b
        fig.add_scatter(x=Y[m, 0], y=Y[m, 1], mode="markers", name=f"blob {b}",
                        marker=dict(size=8, color=_QUAL_PALETTE[b % len(_QUAL_PALETTE)],
                                    line=dict(width=0.5, color="white")))
    fig.update_layout(template="plotly_white", height=height,
                      title=title or f"UMAP toy layout — epoch {ep}",
                      margin=dict(l=10, r=10, t=50, b=10))
    fig.update_xaxes(showticklabels=False); fig.update_yaxes(showticklabels=False)
    return fig


# ============================================================================ house-style + robust helpers
def robust_range(x, lo=1.0, hi=99.0, pad=0.05):
    """PUBLIC robust axis range: [low, high] clipped to the [lo, hi] percentiles of x (default 1/99)
    with a little padding, so a few extreme outliers don't flatten the rest of the cloud. Returns
    None when x has too few finite values or a degenerate spread (caller then leaves the axis auto).

    Thin wrapper of the internal `_robust_range` the built-in fig helpers already use, exposed so a
    notebook author can clip an inline go.Figure the SAME way:
        r = cu.robust_range(vals)
        if r: fig.update_xaxes(range=r)"""
    return _robust_range(x, lo=lo, hi=hi, pad=pad)


def apply_house_style(fig, title=None, legend="inside", spatial=False, height=None):
    """The single house-style fixer to apply to any hand-built inline go.Figure so the whole course
    looks like one deck and the systemic legend/title/grid defects (brief 1d) are fixed in ONE place.

    Args:
        fig      a plotly Figure (returned, mutated in place).
        title    optional overall title string (set only if given).
        legend   'inside' (default) keeps plotly's default legend but, when a `title` is present,
                 raises the top margin to ~70 so a top-anchored legend can NEVER overlap the title;
                 'below' moves the legend under the plot (horizontal, y=-0.2) and raises the bottom
                 margin; None hides the legend entirely.
        spatial  True for pixel / skeleton / map / raster views: turns BOTH axes' gridlines off and
                 locks equal aspect (y scaleanchor=x, scaleratio=1) so shapes aren't distorted.
        height   optional pixel height.

    Resolves the legend-over-title collision and grid-on-spatial defects everywhere at once."""
    fig.update_layout(template="plotly_white")
    m = dict(l=10, r=10, t=50, b=10)
    if legend == "below":
        fig.update_layout(showlegend=True,
                          legend=dict(orientation="h", yanchor="top", y=-0.2,
                                      xanchor="center", x=0.5))
        m["b"] = 70
    elif legend is None:
        fig.update_layout(showlegend=False)
    else:                                    # "inside" (default)
        if title is not None:
            m["t"] = 70
    if title is not None:
        fig.update_layout(title=title)
    if spatial:
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=m)
    if height is not None:
        fig.update_layout(height=height)
    return fig


def fmt_p(p):
    """Format a p-value robustly in float64 so a tiny value never underflows to '0.0e+00' (the
    float32 bug). Returns the bare number string (no 'p = ' prefix):
        >= 1e-300  -> '1.6e-34' style for small, '0.032' style for ordinary
        underflow   -> '< 1e-300'
        non-finite  -> 'nan'."""
    p = float(np.float64(p))
    if not np.isfinite(p):
        return "nan"
    if p <= 0.0 or p < 1e-300:
        return "< 1e-300"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3g}"


# ============================================================================ paired / before-after displays
def dumbbell_fig(before, after, labels=None, before_name="before", after_name="after",
                 color_by_direction=True, colors=None, title="", xlabel="value",
                 height=None, sort_by="after"):
    """Paired before/after per item: a dot at each end joined by a line, one horizontal row per item.
    The line/markers are GREEN when the value increases (after > before) and RED when it decreases —
    the honest display for PAIRED data (cages pre->dep, raw->corrected motion index) that ``box_points``
    would throw the pairing away on.

    Args:
        before, after         (N,) arrays, aligned per item.
        labels                (N,) row labels (default '0'..'N-1').
        before_name/after_name legend labels for the two endpoints.
        color_by_direction    True -> green up / red down; False -> a single neutral gray line.
        colors                optional (N,) per-item line colors (e.g. rank hex); OVERRIDES direction.
        sort_by               'after' | 'before' | 'delta' | None — row ordering (bottom-to-top).
    Returns a plotly Figure."""
    import plotly.graph_objects as go
    before = np.asarray(before, float); after = np.asarray(after, float)
    n = len(before)
    labels = [str(l) for l in labels] if labels is not None else [str(i) for i in range(n)]
    idx = np.arange(n)
    if sort_by == "after":
        idx = np.argsort(after)
    elif sort_by == "before":
        idx = np.argsort(before)
    elif sort_by == "delta":
        idx = np.argsort(after - before)
    up, down = "#2ca02c", "#d62728"
    fig = go.Figure()
    for row, i in enumerate(idx):
        if colors is not None:
            lc = colors[i]
        elif color_by_direction:
            lc = up if after[i] >= before[i] else down
        else:
            lc = "#888888"
        fig.add_scatter(x=[before[i], after[i]], y=[row, row], mode="lines",
                        line=dict(color=lc, width=2.5), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=before[idx], y=list(range(n)), mode="markers", name=before_name,
                    marker=dict(size=9, color="#9aa0a6", line=dict(width=1, color="white")),
                    hovertemplate=f"{before_name}: %{{x:.3f}}<extra></extra>")
    fig.add_scatter(x=after[idx], y=list(range(n)), mode="markers", name=after_name,
                    marker=dict(size=9, color="#1f77b4", line=dict(width=1, color="white")),
                    hovertemplate=f"{after_name}: %{{x:.3f}}<extra></extra>")
    fig.update_yaxes(tickmode="array", tickvals=list(range(n)),
                     ticktext=[labels[i] for i in idx], showgrid=False)
    fig.update_xaxes(title=xlabel)
    fig.update_layout(template="plotly_white", title=title,
                      height=height or max(320, 26 * n + 90),
                      margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.12, x=0.5, xanchor="center"))
    return fig


def slopegraph_fig(values_by_stage, stage_names, labels=None, colors=None, title="",
                   ylabel="value", height=430, robust=True):
    """One connected line per item across S >= 2 ordered stages (e.g. raw | rigid | pw-rigid): the
    paired-across-stages display for the multi-stage motion-correction comparison.

    Args:
        values_by_stage  (N, S) array — row = item, column = stage.
        stage_names      length-S labels for the x positions.
        labels           (N,) per-item hover labels.
        colors           optional (N,) per-item line colors; default a single translucent color so
                         the COLLECTIVE drift across stages is what reads.
        robust           clip the y-axis to the pooled 1/99 percentile.
    Returns a plotly Figure."""
    import plotly.graph_objects as go
    V = np.asarray(values_by_stage, float)
    n, S = V.shape
    labels = [str(l) for l in labels] if labels is not None else [str(i) for i in range(n)]
    xs = list(range(S))
    fig = go.Figure()
    for i in range(n):
        lc = colors[i] if colors is not None else "rgba(76,120,168,0.35)"
        fig.add_scatter(x=xs, y=V[i], mode="lines+markers", name=labels[i], showlegend=False,
                        line=dict(color=lc, width=1.5), marker=dict(size=5, color=lc),
                        hovertemplate=f"{labels[i]}<br>%{{x}}: %{{y:.3f}}<extra></extra>")
    fig.update_xaxes(tickmode="array", tickvals=xs, ticktext=list(stage_names),
                     range=[-0.3, S - 0.7], showgrid=False)
    fig.update_yaxes(title=ylabel)
    if robust:
        ry = _robust_range(V.ravel())
        if ry:
            fig.update_yaxes(range=ry)
    fig.update_layout(template="plotly_white", height=height, title=title,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def paired_diff_fig(diffs, title="", test="wilcoxon", kind="hist", xlabel="difference",
                    height=430, nbins=40):
    """Distribution of per-pair differences with a 0 reference line and the paired-test statistic +
    p annotated — the honest one-number-per-pair view of a before/after change.

    Args:
        diffs   (N,) per-pair differences (e.g. after - before).
        test    'wilcoxon' (signed-rank on the diffs vs 0) or None to skip the annotation.
        kind    'hist' (default) or 'ecdf'.
    The p-value is computed in float64 and formatted with ``fmt_p`` so a tiny p never prints 0.
    Returns a plotly Figure."""
    import plotly.graph_objects as go
    d = np.asarray(diffs, float); d = d[np.isfinite(d)]
    if kind == "ecdf":
        fig = ecdf_fig(d, title="", xlabel=xlabel, ylabel="cumulative fraction", height=height)
    else:
        fig = go.Figure(go.Histogram(x=d, nbinsx=nbins, marker=dict(color="#4c78a8")))
        fig.update_xaxes(title=xlabel); fig.update_yaxes(title="count")
        fig.update_layout(template="plotly_white", height=height,
                          margin=dict(l=10, r=10, t=50, b=10))
    fig.add_vline(x=0.0, line=dict(color="#d62728", width=2, dash="dash"))
    ann = None
    if test == "wilcoxon" and len(d) >= 1 and np.any(d != 0):
        from scipy.stats import wilcoxon
        try:
            stat, p = wilcoxon(d)
            ann = f"Wilcoxon W = {float(stat):.1f}<br>p = {fmt_p(p)}  (n = {len(d)})"
        except ValueError:
            ann = None
    if ann:
        fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.98, xanchor="right",
                           yanchor="top", showarrow=False, text=ann, align="left",
                           font=dict(size=13), bgcolor="rgba(255,255,255,0.75)",
                           bordercolor="#ccc", borderwidth=1)
    fig.update_layout(title=title)
    return fig


# ============================================================================ proportions with CIs
def wilson_ci(k, n, z=1.96):
    """Wilson score confidence interval for a binomial proportion k/n. Returns (lo, hi). Handles
    n == 0 (returns (0.0, 1.0)) and is far better than the normal approximation for small n or rates
    near 0/1 (it never spills below 0 or above 1). Vectorized: k, n may be scalars or arrays."""
    k = np.asarray(k, float); n = np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        phat = np.where(n > 0, k / n, 0.5)
        denom = 1.0 + z * z / n
        center = (phat + z * z / (2 * n)) / denom
        half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
        lo = center - half; hi = center + half
    lo = np.where(n > 0, np.clip(lo, 0.0, 1.0), 0.0)
    hi = np.where(n > 0, np.clip(hi, 0.0, 1.0), 1.0)
    if np.ndim(k) == 0 and np.ndim(n) == 0:
        return float(lo), float(hi)
    return lo, hi


def proportion_ci_fig(counts, totals, labels, colors=None, group_order=None, z=1.96,
                      title="", ylabel="rate", xlabel="", height=430):
    """A point per group at its rate counts/totals with Wilson error bars — the honest replacement
    for the tautological 'split by X, then plot X' pattern: plot the OUTCOME rate (e.g. aggression
    rate) per group with a CI that shows how (un)certain each rate is given its n.

    Args:
        counts, totals  (G,) successes and trials per group.
        labels          (G,) group labels.
        colors          optional dict/list mapping group -> color (rank auto-detect otherwise).
    Returns a plotly Figure."""
    import plotly.graph_objects as go
    counts = np.asarray(counts, float); totals = np.asarray(totals, float)
    labels = [str(l) for l in labels]
    order = _group_order(labels, group_order)
    cmap = _group_colors(order, colors if isinstance(colors, dict) else None)
    if isinstance(colors, (list, tuple, np.ndarray)):
        cmap = {labels[i]: colors[i] for i in range(len(labels))}
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(totals > 0, counts / totals, np.nan)
    lo, hi = wilson_ci(counts, totals, z)
    pos = {g: i for i, g in enumerate(order)}
    fig = go.Figure()
    for i, g in enumerate(labels):
        xi = pos[g]
        fig.add_scatter(x=[xi], y=[rate[i]], mode="markers", name=g, showlegend=False,
                        marker=dict(size=12, color=cmap.get(g, "#4c78a8"),
                                    line=dict(width=1, color="white")),
                        error_y=dict(type="data", symmetric=False,
                                     array=[hi[i] - rate[i]], arrayminus=[rate[i] - lo[i]],
                                     color=cmap.get(g, "#4c78a8"), thickness=2, width=6),
                        hovertemplate=(f"{g}: %{{y:.3f}}"
                                       f"<br>[{lo[i]:.3f}, {hi[i]:.3f}]  "
                                       f"n={int(totals[i])}<extra></extra>"))
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(order))),
                     ticktext=order, title=xlabel, range=[-0.5, len(order) - 0.5])
    fig.update_yaxes(title=ylabel)
    fig.update_layout(template="plotly_white", height=height, title=title,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ============================================================================ leave-one-PC-out AUROC
def _logreg_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=1000))])


def loo_pc_auroc(scores, y, n_splits=5, seed=0):
    """Leave-one-PC-out cross-validated AUROC: how much each principal component contributes to
    predicting the binary label y. Returns a dict:
        {0: auroc_without_PC1, 1: auroc_without_PC2, ..., 'full': auroc_with_all_PCs}
    (keys are 0-indexed column indices; 'full' uses every column). A logistic decoder is scored with
    stratified k-fold cross-val (honest, not in-sample). A LOWER auroc_without_i => PC i mattered
    more. This is the real replacement for the inert 'drop a PC and re-scatter' cell — dropping a PC
    that carries signal visibly moves the number."""
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    S = np.asarray(scores, float); y = np.asarray(y).astype(int)
    k = S.shape[1]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def _auroc(X):
        proba = cross_val_predict(_logreg_pipeline(), X, y, cv=cv, method="predict_proba")[:, 1]
        return float(roc_auc_score(y, proba))

    out = {"full": _auroc(S)}
    for i in range(k):
        out[i] = _auroc(np.delete(S, i, axis=1))
    return out


def loo_pc_auroc_fig(scores, y, n_splits=5, seed=0, title="Leave-one-PC-out AUROC", height=430):
    """Bar chart of ``loo_pc_auroc``: one bar per PC = the cross-validated AUROC with that PC removed,
    with a dashed reference line at the full-model AUROC. Bars that dip BELOW the line are the PCs
    that carry aggression signal; bars at/above the line are redundant. Returns a plotly Figure."""
    import plotly.graph_objects as go
    res = loo_pc_auroc(scores, y, n_splits=n_splits, seed=seed)
    full = res["full"]
    ks = sorted(i for i in res if i != "full")
    vals = [res[i] for i in ks]
    drop = [full - v for v in vals]
    cols = ["#d62728" if d > 0 else "#4c78a8" for d in drop]     # red = removing it hurt
    fig = go.Figure(go.Bar(x=[f"PC{i + 1}" for i in ks], y=vals, marker=dict(color=cols),
                           hovertemplate="%{x} removed<br>AUROC=%{y:.3f}<extra></extra>"))
    fig.add_hline(y=full, line=dict(color="#333", width=2, dash="dash"),
                  annotation_text=f"all PCs = {full:.3f}", annotation_position="top left")
    fig.update_yaxes(title="AUROC (that PC removed)")
    fig.update_layout(template="plotly_white", height=height, title=title,
                      margin=dict(l=10, r=10, t=60, b=10))
    return fig


# ============================================================================ blocked / leaky cross-validation
def blocked_cv_auroc(X, y, order=None, scheme="blocked", n_splits=5, clf=None):
    """Per-fold decoder AUROC under one of three cross-validation schemes — the core of the
    CV-leakage lesson. Temporally autocorrelated samples (neighboring calcium frames) make a random
    split LEAK, inflating AUROC; a time-respecting split gives the honest number.

    scheme:
        'shuffle'     random StratifiedKFold(shuffle=True) — the LEAKY one (neighbors land in both
                      train and test); reproduces the optimistic ~0.95.
        'blocked'     n_splits CONTIGUOUS blocks in `order`; each block is the test fold once, the
                      rest train — keeps temporally-adjacent samples together, so ~0.70-0.82.
        'contiguous'  a single ordered split (first ~1-1/n_splits train, last block test); returns a
                      length-1 array — the strictest, forward-in-time test.
    `order` is the temporal ordering of the rows (default np.arange(N)); rows are ranked by it so any
    monotonic index works. `clf` defaults to a standardized logistic decoder. Returns an np.ndarray
    of per-fold AUROC."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.base import clone
    X = np.asarray(X, float); y = np.asarray(y).astype(int)
    n = len(y)
    order = np.arange(n) if order is None else np.asarray(order)
    time_rank = np.argsort(np.argsort(order))          # 0..n-1 position in time
    base = clf if clf is not None else _logreg_pipeline()

    def _fit_score(tr, te):
        if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
            return np.nan
        m = clone(base).fit(X[tr], y[tr])
        proba = m.predict_proba(X[te])[:, 1]
        return float(roc_auc_score(y[te], proba))

    if scheme == "shuffle":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        return np.array([_fit_score(tr, te) for tr, te in cv.split(X, y)])
    order_idx = np.argsort(time_rank)                  # row indices sorted by time
    if scheme == "contiguous":
        cut = int(n * (1.0 - 1.0 / n_splits))
        tr, te = order_idx[:cut], order_idx[cut:]
        return np.array([_fit_score(tr, te)])
    # blocked: contiguous test blocks along the time order
    blocks = np.array_split(order_idx, n_splits)
    out = []
    for b in blocks:
        te = b
        tr = np.setdiff1d(order_idx, te, assume_unique=False)
        out.append(_fit_score(tr, te))
    return np.array(out)


# ============================================================================ confusion matrix
def confusion_fig(cm, labels, normalize="row", title="Confusion matrix", height=430):
    """Heatmap of a confusion matrix. `normalize='row'` (default) divides each row by its sum so the
    COLOR encodes recall (correct-vs-error within each true class), NOT raw magnitude — essential
    under class imbalance, where a raw-count colorscale just paints the majority class. 'col'
    normalizes by predicted class (precision); None shows raw counts. Cells are annotated with the
    (normalized) value; the raw count rides along in the hover. Returns a plotly Figure."""
    import plotly.graph_objects as go
    cm = np.asarray(cm, float)
    if normalize == "row":
        denom = cm.sum(1, keepdims=True); Z = np.divide(cm, denom, out=np.zeros_like(cm), where=denom > 0)
        cbar = "recall (row-normalized)"
    elif normalize == "col":
        denom = cm.sum(0, keepdims=True); Z = np.divide(cm, denom, out=np.zeros_like(cm), where=denom > 0)
        cbar = "precision (col-normalized)"
    else:
        Z = cm; cbar = "count"
    labels = [str(l) for l in labels]
    text = [[(f"{Z[i, j]:.2f}" if normalize else f"{int(cm[i, j])}") for j in range(cm.shape[1])]
            for i in range(cm.shape[0])]
    fig = go.Figure(go.Heatmap(
        z=Z, x=labels, y=labels, colorscale="Blues",
        zmin=0, zmax=(1 if normalize else None), colorbar=dict(title=cbar),
        text=text, texttemplate="%{text}",
        customdata=cm, hovertemplate="true=%{y} pred=%{x}<br>count=%{customdata}<extra></extra>"))
    fig.update_xaxes(title="predicted", side="bottom")
    fig.update_yaxes(title="true", autorange="reversed")
    fig.update_layout(template="plotly_white", height=height, title=title,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ============================================================================ scatter + marginal histograms
def scatter_marginal_fig(x, y, groups=None, group_order=None, colors=None, robust=True,
                         point_size=6, opacity=0.8, hover=None, nbins=40,
                         title="", xlabel="x", ylabel="y", height=560):
    """A scatter with marginal histograms on the top and right — opaque points, no smoothing. The
    honest replacement for a 2-D KDE when n is small or the tails are heavy (a KDE over-smooths and
    can spill probability mass to impossible values). `robust` clips both scatter axes to the pooled
    1/99 percentile (points still plotted). Optionally colored by `groups`. Returns a plotly Figure."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    x = np.asarray(x, float); y = np.asarray(y, float)
    fig = make_subplots(rows=2, cols=2, column_widths=[0.82, 0.18], row_heights=[0.18, 0.82],
                        horizontal_spacing=0.02, vertical_spacing=0.02,
                        shared_xaxes=True, shared_yaxes=True)
    if groups is None:
        grp = np.zeros(len(x), int); order = [0]; cmap = {0: "#4c78a8"}; single = True
    else:
        grp = np.asarray(groups); order = _group_order(grp, group_order)
        cmap = _group_colors(order, colors); single = False
    hv = None if hover is None else np.asarray(hover)
    for g in order:
        m = grp == g
        txt = [str(t) for t in hv[m]] if hv is not None else None
        fig.add_trace(go.Scattergl(
            x=x[m], y=y[m], mode="markers", name=str(g), showlegend=not single, text=txt,
            marker=dict(size=point_size, color=cmap[g], opacity=opacity, line=dict(width=0.4, color="white")),
            hovertemplate=(("%{text}<br>" if txt is not None else "") + "%{x:.3f}, %{y:.3f}<extra></extra>")),
            row=2, col=1)
        fig.add_trace(go.Histogram(x=x[m], nbinsx=nbins, marker=dict(color=cmap[g]), opacity=0.6,
                                   showlegend=False), row=1, col=1)
        fig.add_trace(go.Histogram(y=y[m], nbinsy=nbins, marker=dict(color=cmap[g]), opacity=0.6,
                                   showlegend=False), row=2, col=2)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title=xlabel, row=2, col=1); fig.update_yaxes(title=ylabel, row=2, col=1)
    if robust:
        rx = _robust_range(x); ry = _robust_range(y)
        if rx:
            fig.update_xaxes(range=rx, row=2, col=1)
        if ry:
            fig.update_yaxes(range=ry, row=2, col=1)
    fig.update_layout(template="plotly_white", height=height, title=title,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ============================================================================ time-frequency: dominant frequency
def dominant_frequency(power, freqs, interior=None):
    """Per-frame dominant frequency from a (F, T) wavelet-power spectrogram, taking the argmax ONLY
    over positive frequencies AND over the non-edge-padded INTERIOR time columns — so it does not
    floor at the lowest frequency because of edge-padding artifacts (the bug that made the NB3
    detector report ~1 Hz for everything).

    Args:
        power    (F, T) power (e.g. from ``wavelet_power``).
        freqs    (F,) frequency axis (Hz).
        interior  which time columns are trustworthy (not edge-padded). Accepts: None (default ->
                  central 60%, trimming 20% off each edge), a boolean mask (T,), an index array, a
                  slice, or a (start, end) fraction/index tuple.
    Returns a (T,) float array: the dominant frequency at each interior frame and NaN outside the
    interior. Take ``np.nanmedian`` of it for a single robust scalar."""
    P = np.asarray(power, float); f = np.asarray(freqs, float)
    F, T = P.shape
    fmask = f > 0
    mask = np.zeros(T, bool)
    if interior is None:
        a, b = int(round(0.2 * T)), int(round(0.8 * T))
        mask[a:max(a + 1, b)] = True
    elif isinstance(interior, slice):
        mask[interior] = True
    elif isinstance(interior, tuple) and len(interior) == 2:
        a, b = interior
        if isinstance(a, float) or isinstance(b, float):
            a, b = int(round(a * T)), int(round(b * T))
        mask[int(a):int(b)] = True
    else:
        interior = np.asarray(interior)
        if interior.dtype == bool:
            mask = interior
        else:
            mask[interior] = True
    out = np.full(T, np.nan)
    fpos = np.where(fmask)[0]
    if len(fpos) == 0:
        return out
    cols = np.where(mask)[0]
    if len(cols):
        sub = P[np.ix_(fpos, cols)]                    # (Fpos, ncols)
        best = fpos[np.argmax(sub, axis=0)]
        out[cols] = f[best]
    return out


# ============================================================================ grid-cell gridness score
def gridness_score(autocorr2d):
    """Standard 60-degree gridness of a 2-D spatial autocorrelogram: how hexagonally symmetric the
    ring of surrounding peaks is. Correlate the autocorrelogram's central annulus with rotated copies
    of itself at 30/60/90/120/150 deg and return

        gridness = min(corr@60, corr@120) - max(corr@30, corr@90, corr@150)

    A true grid cell scores > 0 (its ring repeats every 60 deg); a place field or border/multi-field
    cell scores <= 0. This gives NB8 an HONEST grid call instead of eyeballing 'are there satellites'.

    `autocorr2d` is a square-ish 2-D autocorrelogram (e.g. of a rate map). Returns a float."""
    from scipy.ndimage import rotate
    A = np.asarray(autocorr2d, float)
    A = np.nan_to_num(A, nan=0.0)
    cy, cx = (np.array(A.shape) - 1) / 2.0
    yy, xx = np.mgrid[0:A.shape[0], 0:A.shape[1]]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_out = 0.5 * min(A.shape)                          # inscribed radius
    r_in = 0.15 * r_out                                 # exclude the central peak
    ring = (r >= r_in) & (r <= r_out)
    base = A[ring]

    def _corr(angle):
        Ar = rotate(A, angle, reshape=False, order=1, mode="constant", cval=0.0)
        v = Ar[ring]
        if base.std() < 1e-12 or v.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(base, v)[0, 1])

    c = {a: _corr(a) for a in (30, 60, 90, 120, 150)}
    return min(c[60], c[120]) - max(c[30], c[90], c[150])
