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
    "https://raw.githubusercontent.com/Elmaestrotango/sleap-social-behavior-lab/main",
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
    """UMAP embedding. n_neighbors = local vs global balance; min_dist = cluster tightness."""
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
# Notebooks 03/04/05 all reference these so their clusterings agree.
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
    titles = [f"n_neighbors={nn}, min_dist={md:g}" for nn in nn_values for md in md_values]
    fig = make_subplots(rows=n_nn, cols=n_md, subplot_titles=titles,
                        horizontal_spacing=0.04, vertical_spacing=0.08)
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
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.update_layout(template="plotly_white", height=height,
                      title="UMAP parameter sweep — same points, different knobs",
                      margin=dict(l=10, r=10, t=60, b=10))
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


def pca_loadings_fig(components, feature_names, k=3):
    """Heatmap of the first k PC loadings across features — shows what each component 'means'."""
    import plotly.graph_objects as go
    C = np.asarray(components)[:k]
    fig = go.Figure(go.Heatmap(z=C, x=list(feature_names), y=[f"PC{i}" for i in range(k)],
                               colorscale="RdBu", zmid=0))
    fig.update_layout(template="plotly_white", height=110 + 55 * k,
                      title="PCA loadings — how features combine into components",
                      margin=dict(l=10, r=10, t=40, b=130))
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
