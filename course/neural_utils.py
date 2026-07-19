"""
neural_utils.py — the shared engine for the course's Week-2 *neural* arm.

Companion to ``course_utils.py`` (the SLEAP/behavior arm). Everything the neural notebooks
(NB09 motion-correction, NB10 calcium extraction, NB11 demixed calcium, NB12 place/grid cells,
NB13/NB14 social-isolation) rely on lives here so notebook cells stay short and readable:

  * data fetching + caching (eLife video, Google-Drive h5/xlsx, Dropbox zip) — downloaded at
    runtime into a gitignored cache, never re-hosted;
  * loaders that parse the real files into plain numpy/pandas (CNMF h5, SI xlsx+h5, rat .mat);
  * a lazy, subsampling video reader (imageio) so a bare cloud kernel never loads a whole movie;
  * vectorized analysis helpers that reproduce what the 2025 ``2025/edge_*.py`` scripts did, but
    cleaner (z-score, interpolation resample, occupancy-normalized rate maps, Skaggs spatial
    information, the social-neuron test, sequence sorting, background subtraction, motion index);
  * house-style Plotly figure builders (``template="plotly_white"``, tight margins, colorbars).

Self-contained: numpy / scipy / scikit-learn / plotly / h5py / pandas / imageio / gdown only.
No lab code, no GPU. Mirrors the 2025 EDGE analyses in ``2025/edge_*.py``.

Data sources (imported from the SAME places the 2025 scripts used — download + cache at runtime):
  NB10 striatum miniscope video : eLife direct URL          -> striatum.mp4
  NB11 CNMF demixed calcium     : Google Drive 1PYLeqT8...   -> 221007_4-0_D2_neuron_refined.h5
  NB09 motion-correction movie  : Google Drive 1cx7vG3w...   -> 1-s2.0-S0165027017302753-mmc3.mp4
  NB12 rat place/grid data      : Dropbox zip                -> *.mat
  NB13/14 social isolation      : Google Drive x3            -> SI3 xlsx + social_bouts + calcium h5
"""
from __future__ import annotations
import os
import zipfile
import urllib.request
import numpy as np

# The behavior arm's engine. neural_utils runs with course/ on sys.path, so this plain import
# works in every notebook. We IMPORT the shared figure/stat helpers from here rather than
# re-implement them, so the neural notebooks inherit the round-3 fixes (violin points-on-top,
# ecdf robust-clip, RdBu_r loadings, house-style legend/grid, float64 p-values, ...). All of
# course_utils' heavy deps (plotly/scipy/sklearn) are imported lazily inside function bodies, so
# `import course_utils` stays as cheap as importing this module.
import course_utils as cu


# ============================================================================ source registry
# Canonical identifiers for every remote asset, taken verbatim from the 2025 scripts so the neural
# arm imports from exactly the same places (see module docstring).
STRIATUM_URL = ("https://static-movie-usa.glencoesoftware.com/mp4/10.7554/570/"
                "874741fc038d0057c935ce00334c834e1c4c6136/elife-28728-video1.mp4")
STRIATUM_NAME = "striatum.mp4"

CNMF_GDRIVE_ID = "1PYLeqT88IH_9JWNPwYaUC9WJVT2IqevL"
CNMF_NAME = "221007_4-0_D2_neuron_refined.h5"

MOCO_GDRIVE_ID = "1cx7vG3w0FjxsWpMKSrV2XYFKV8kwa7Y7"
MOCO_NAME = "1-s2.0-S0165027017302753-mmc3.mp4"

RAT_ZIP_URL = ("https://www.dropbox.com/scl/fi/w6afx6u18tuo7xc439iq2/HW1_RatData.zip"
               "?rlkey=fj6dorj1udb45nc4i9j3j7mz6&dl=1")
RAT_FILES = ["20160609T194655.mat", "20160607T172746.mat", "20160521T142946.mat"]

# NOTE: the two small SI Drive ids are cross-labeled in the source docs vs. their actual payloads.
# Verified by file magic + open: id 1POp... returns the Excel entrances table, and id 1Mh8...
# returns the social_bouts HDF5. (The .00.h5 calcium id is correct.) We map by *content* here so
# load_si() lands the right bytes in the right filename.
SI_ENTRANCES_ID = "1POpRqpA_QaWfZhxswQvLSs9uBnnHrmhZ"   # -> Excel: SI3_2022_Entrance_Frames.xlsx
SI_ENTRANCES_NAME = "SI3_2022_Entrance_Frames.xlsx"
SI_BOUTS_ID = "1Mh8oGKNyKpT5WS0Wu92SULFFanvqmSMf"       # -> HDF5: social_bouts.00.h5
SI_BOUTS_NAME = "social_bouts.00.h5"
SI_CALCIUM_ID = "1UthpsskvkHbKKDsbQjUxyVN4Xkd-ZJuN"     # -> HDF5: calcium.00.h5
SI_CALCIUM_NAME = "calcium.00.h5"

# The nine per-session behavior channels in social_bouts.00.h5 (order matches the 2025 script).
SI_SESSION_KEYS = ["is_ag_sniffed", "is_ag_sniffing", "is_of_sniffed", "is_of_sniffing",
                   "is_social", "is_social_receiver", "is_social_sender", "is_touched",
                   "is_touching"]
BEHAVIOR_FPS = 25   # social_bouts sampling rate
IMAGING_FPS = 30    # calcium sampling rate


# ============================================================================ cache + fetch
def find_root(start=None):
    """Walk up from ``start`` (or cwd) for the repo root (a folder holding both course/ and data/);
    fall back to the cwd if not found."""
    p = start or os.getcwd()
    for _ in range(8):
        if os.path.isdir(os.path.join(p, "course")) and os.path.isdir(os.path.join(p, "data")):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return start or os.getcwd()


def cache_dir(root=None):
    """Return a writable cache directory ``<root>/data/_neural_cache`` (created if missing).

    All remote downloads live here. It is gitignored — never committed — so a fresh clone stays
    small and each machine repopulates it on first run."""
    root = root or find_root()
    d = os.path.join(root, "data", "_neural_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _looks_like_html(path, sniff=512):
    """True if ``path`` starts with an HTML/redirect page rather than real binary payload — the
    classic failure mode of a Drive/Dropbox link that returned a warning page instead of the file."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(sniff).lstrip().lower()
        return head.startswith(b"<!doctype html") or head.startswith(b"<html")
    except OSError:
        return False


def fetch_url(url, name, root=None):
    """Download ``url`` to ``<cache>/name`` via urllib if not already cached; return the local path.

    Idempotent: a non-empty cached file is reused. Used for the eLife striatum video and any other
    plain HTTP(S) asset."""
    dst = os.path.join(cache_dir(root), name)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    os.replace(tmp, dst)
    return dst


def fetch_gdrive(file_id, name, root=None):
    """Download a Google-Drive file (by id) to ``<cache>/name`` via gdown if not cached; return the
    local path. Passing ``id=`` (rather than a ``uc?id=`` URL) is gdown's robust path and handles
    Drive's large-file confirmation page automatically — the API equivalent of ``gdown --fuzzy``."""
    import gdown
    dst = os.path.join(cache_dir(root), name)
    if os.path.exists(dst) and os.path.getsize(dst) > 0 and not _looks_like_html(dst):
        return dst
    gdown.download(id=file_id, output=dst, quiet=True)
    if not os.path.exists(dst) or os.path.getsize(dst) == 0:
        raise RuntimeError(f"gdrive download failed for {file_id} -> {name}")
    return dst


def fetch_zip_dropbox(url=RAT_ZIP_URL, root=None, extract=True, subdir="rat_data"):
    """Download a Dropbox zip (forcing ``dl=1``) into the cache and unzip it.

    Returns the directory containing the extracted files (``<cache>/<subdir>``) when ``extract`` is
    True, else the path to the downloaded ``.zip``. Used for the NEU 457 rat place/grid data."""
    url = url.replace("dl=0", "dl=1")
    if "dl=1" not in url:
        url += ("&" if "?" in url else "?") + "dl=1"
    zip_path = fetch_url(url, "HW1_RatData.zip", root)
    if not extract:
        return zip_path
    out_dir = os.path.join(cache_dir(root), subdir)
    os.makedirs(out_dir, exist_ok=True)
    # Only unzip if the expected .mat files aren't already there.
    have = {f for f in os.listdir(out_dir) if f.endswith(".mat")}
    if not have:
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                # Flatten any internal folder structure so callers find *.mat directly in out_dir.
                base = os.path.basename(member)
                if not base or member.endswith("/"):
                    continue
                with zf.open(member) as src, open(os.path.join(out_dir, base), "wb") as dst:
                    dst.write(src.read())
    return out_dir


# ============================================================================ video reader
def read_video(path, step=1, max_frames=None, gray=True):
    """Lazily read a video into a ``(F, H, W)`` (gray) or ``(F, H, W, 3)`` (color) float32 array.

    Subsamples with ``step`` (keep every ``step``-th frame) and stops after ``max_frames`` kept
    frames — so with ``max_frames`` set it NEVER decodes the whole movie past what it needs. Reads
    one sequential pass via imageio's ffmpeg plugin (robust; no fragile random seeks).

    Parameters
    ----------
    path : str            local video file.
    step : int            keep every ``step``-th frame (default 1 = all).
    max_frames : int|None  cap on number of kept frames.
    gray : bool           average RGB -> luminance-ish gray (mean over channels).

    Returns
    -------
    (F, H, W) float32 if gray else (F, H, W, 3) float32.
    """
    import imageio.v2 as imageio
    reader = imageio.get_reader(path, "ffmpeg")
    frames = []
    try:
        for i, frame in enumerate(reader):
            if i % step:
                continue
            arr = np.asarray(frame, dtype=np.float32)
            if gray and arr.ndim == 3:
                arr = arr.mean(axis=-1)
            frames.append(arr)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        reader.close()
    return np.stack(frames, axis=0)


def video_meta(path):
    """Return imageio metadata dict for ``path`` (fps, nframes estimate, size) without decoding."""
    import imageio.v2 as imageio
    reader = imageio.get_reader(path, "ffmpeg")
    try:
        return dict(reader.get_meta_data())
    finally:
        reader.close()


def split_thirds(mov, crop_y=True):
    """Split a side-by-side motion-correction movie into (raw, rigid, pw_rigid) panels.

    ``mov`` is ``(F, H, W)``; the frame is three equal-width panels laid left→right. Mirrors the
    2025 NB09 script: ``w = W // 3`` and (optionally) crop the top/bottom ``w//2`` rows so the
    kymograph rows line up. Returns three ``(F, H', w)`` arrays."""
    F, H, W = mov.shape[:3]
    w = W // 3
    if crop_y:
        y0, y1 = w // 2, H - w // 2
    else:
        y0, y1 = 0, H
    raw = mov[:, y0:y1, :w]
    rigid = mov[:, y0:y1, w:2 * w]
    pwr = mov[:, y0:y1, 2 * w:3 * w]
    return raw, rigid, pwr


# ============================================================================ loaders
def load_cnmf(path=None, root=None):
    """Load a CNMF-E demixed-calcium ``.h5`` (NB11). Downloads from Google Drive if ``path`` is None.

    Keys in the file: A (spatial footprints, n_neurons x n_pixels), C (calcium, n_frames x
    n_neurons), Cn (correlation image, H x W), S (deconvolved spikes, n_frames x n_neurons),
    Fs (fps scalar). Returns a dict::

        A (n_neurons, n_pixels) float, C (n_frames, n_neurons) float, Cn (H, W) float,
        S (n_frames, n_neurons) float, Fs float, img_shape (H, W), n_neurons int, n_frames int
    """
    import h5py
    if path is None:
        path = fetch_gdrive(CNMF_GDRIVE_ID, CNMF_NAME, root)
    with h5py.File(path, "r") as f:
        A = f["A"][:]
        C = f["C"][:]
        Cn = f["Cn"][:]
        S = f["S"][:]
        Fs = float(np.asarray(f["Fs"][:]).squeeze())
    return dict(A=A, C=C, Cn=Cn, S=S, Fs=Fs, img_shape=tuple(Cn.shape),
                n_neurons=int(A.shape[0]), n_frames=int(C.shape[0]))


def footprint(A, neuron_ind, img_shape):
    """Reshape one CNMF spatial footprint row ``A[neuron_ind]`` to the ``img_shape`` image (NB11)."""
    return np.asarray(A[neuron_ind]).reshape(img_shape)


def footprint_montage(A, img_shape):
    """Max-projection of all peak-normalized footprints -> ``img_shape`` image (the NB11 A_max view).

    Vectorized version of the 2025 script: ``(A / A.max(axis=1, keepdims=True)).max(axis=0)``."""
    A = np.asarray(A, dtype=np.float32)
    peak = A.max(axis=1, keepdims=True)
    peak[peak == 0] = 1.0
    return (A / peak).max(axis=0).reshape(img_shape)


def load_si(root=None, entrances_path=None, bouts_path=None, calcium_path=None):
    """Load the SI3_2022 social-isolation dataset (NB13 + NB14). Downloads the three Drive files if
    paths are None. Returns a dict::

        entrances      pandas.DataFrame  one row per session (cols incl. "Isolation Length",
                                         "Int_Entry").
        behavior       list[dict]        per session: {key: (T_beh,) bool array} for SI_SESSION_KEYS.
        imaging        list[np.ndarray]  per session: C (n_frames, n_neurons) calcium.
        session_keys   list[str]         the nine behavior channels.
        n_sessions     int
        behavior_fps   int = 25
        imaging_fps    int = 30

    ``is_social = is_social_receiver | is_social_sender`` (already materialized in the file).
    Behavior and imaging are on different clocks — resample with ``interp_resample`` before
    aligning (see NB14 / ``social_neuron_mask``).
    """
    import pandas as pd
    import h5py
    if entrances_path is None:
        entrances_path = fetch_gdrive(SI_ENTRANCES_ID, SI_ENTRANCES_NAME, root)
    if bouts_path is None:
        bouts_path = fetch_gdrive(SI_BOUTS_ID, SI_BOUTS_NAME, root)
    if calcium_path is None:
        calcium_path = fetch_gdrive(SI_CALCIUM_ID, SI_CALCIUM_NAME, root)

    entrances = pd.read_excel(entrances_path)
    n_sessions = len(entrances)

    behavior = []
    with h5py.File(bouts_path, "r") as f:
        for s in range(n_sessions):
            g = f[f"session_{s}"]
            behavior.append({k: g[k][:] for k in SI_SESSION_KEYS})

    imaging = []
    with h5py.File(calcium_path, "r") as f:
        for s in range(n_sessions):
            imaging.append(f[f"session_{s}"]["C"][:])

    return dict(entrances=entrances, behavior=behavior, imaging=imaging,
                session_keys=list(SI_SESSION_KEYS), n_sessions=int(n_sessions),
                behavior_fps=BEHAVIOR_FPS, imaging_fps=IMAGING_FPS)


def si_condition_label(condition):
    """Normalize an "Isolation Length" cell to a condition label, matching the 2025 NB14 script:
    group-housed controls (``"GH"`` in the string) -> ``"control"``, otherwise the raw label
    (e.g. ``"24hr"``, ``"7d"``)."""
    return "control" if "GH" in str(condition) else str(condition)


def load_rat_mat(path):
    """Load one rat place/grid ``.mat`` (NB12). Returns a dict::

        left     (T, 2) float   left-eye tracked position (x, y).
        right    (T, 2) float   right-eye tracked position.
        spikes   (T, n_neurons) spike counts per frame.
        centroid (T, 2) float   (left + right) / 2  — the animal's position estimate.
    """
    from scipy.io import loadmat
    m = loadmat(path)
    left = np.asarray(m["left"], dtype=float)
    right = np.asarray(m["right"], dtype=float)
    spikes = np.asarray(m["spikes"])
    centroid = (left + right) / 2.0
    return dict(left=left, right=right, spikes=spikes, centroid=centroid)


# ============================================================================ analysis helpers
def zscore(x, axis=0, eps=1e-12):
    """Z-score along ``axis`` (subtract mean, divide by std). NaN-safe divide (std==0 -> eps)."""
    x = np.asarray(x, dtype=float)
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return (x - mu) / np.where(sd < eps, eps, sd)


def interp_resample(y, n_out, axis=0):
    """Resample ``y`` to ``n_out`` samples along ``axis`` by linear interpolation on a normalized
    [0, 1] grid — exactly the ``scipy.interpolate.interp1d`` trick the NB14 script used to put
    calcium (30 fps) onto the behavior (25 fps) clock::

        interp1d(linspace(0,1,n_in), y, axis)(linspace(0,1,n_out))
    """
    import scipy.interpolate
    y = np.asarray(y, dtype=float)
    n_in = y.shape[axis]
    f = scipy.interpolate.interp1d(np.linspace(0, 1, num=n_in), y, axis=axis)
    return f(np.linspace(0, 1, num=n_out))


def rate_map(x, y, spikes, bins=20, smooth_sigma=None):
    """Occupancy-normalized spatial firing-rate map (NB12 place-cell analysis).

    Unlike the 2025 script's ``histogram2d(..., weights=spikes, density=True)`` (which is really a
    spike *density*), this divides the per-bin spike total by the per-bin occupancy so each bin is a
    true rate = spikes / frames-spent-there. NaN positions are dropped.

    Parameters
    ----------
    x, y : (T,) position coordinates (e.g. ``centroid[:, 0]``, ``centroid[:, 1]``).
    spikes : (T,) spike counts (or a boolean spiking mask).
    bins : int or [xedges, yedges]   passed to np.histogram2d.
    smooth_sigma : float|None        optional Gaussian smoothing (bins) of the rate map.

    Returns
    -------
    dict with::
        rate       (bx, by) float   spikes / occupancy, 0 where never visited.
        occupancy  (bx, by) float   frame counts per bin.
        spike_map  (bx, by) float   summed spikes per bin.
        xedges, yedges             bin edges.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    spikes = np.asarray(spikes, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(spikes)
    x, y, spikes = x[ok], y[ok], spikes[ok]
    spike_map, xe, ye = np.histogram2d(x, y, bins=bins, weights=spikes)
    occ, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(occ > 0, spike_map / occ, 0.0)
    if smooth_sigma:
        from scipy.ndimage import gaussian_filter
        rate = gaussian_filter(rate, smooth_sigma)
    return dict(rate=rate, occupancy=occ, spike_map=spike_map, xedges=xe, yedges=ye)


def spatial_information(rate, occupancy):
    """Skaggs spatial information in **bits/spike** for a rate map.

        SI = sum_i  p_i * (r_i / rbar) * log2(r_i / rbar)

    where ``p_i`` is the occupancy probability of bin ``i``, ``r_i`` its firing rate, and
    ``rbar = sum_i p_i r_i`` the occupancy-weighted mean rate. Bins with zero occupancy or zero
    rate contribute nothing (the standard convention). A spatially uninformative cell -> ~0 bits;
    a sharply tuned place/grid cell -> larger positive values."""
    rate = np.asarray(rate, dtype=float)
    occ = np.asarray(occupancy, dtype=float)
    p = occ / occ.sum() if occ.sum() > 0 else occ
    rbar = float((p * rate).sum())
    if rbar <= 0:
        return 0.0
    ratio = np.where(rate > 0, rate / rbar, 1.0)  # log(1)=0 for empty bins -> no contribution
    with np.errstate(divide="ignore", invalid="ignore"):
        bits = p * (rate / rbar) * np.log2(ratio)
    return float(np.nansum(np.where(rate > 0, bits, 0.0)))


def social_neuron_mask(neurons, is_social, method="ratio", thresh=None):
    """The SI social-neuron test (NB14): which neurons respond around social frames.

    ``neurons`` is ``(T, n_neurons)`` (z-score it first, as the 2025 script does), ``is_social`` a
    ``(T,)`` boolean. Three methods reproduce the three variants in the 2025 script:

      * ``"ratio"`` (default, thresh=1.5): mean |activity| in social vs non-social frames;
        ``ratio = |x[social]|.mean / |x[~social]|.mean > thresh``.
      * ``"delta"`` (thresh=0.5): signed mean difference; ``|mean(social) - mean(~social)| > thresh``.
      * ``"percentile"`` (thresh=0.01): a neuron is "active" when it exceeds its 95th / dips below
        its 5th percentile; mask = fraction of social frames that are active > thresh.

    Returns a ``(n_neurons,)`` boolean mask. ``n_social_neurons = mask.sum()``.
    """
    neurons = np.asarray(neurons, dtype=float)
    is_social = np.asarray(is_social, dtype=bool)
    if method == "ratio":
        thresh = 1.5 if thresh is None else thresh
        soc = np.abs(neurons[is_social]).mean(axis=0)
        non = np.abs(neurons[~is_social]).mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(non > 0, soc / non, 0.0)
        return ratio > thresh
    if method == "delta":
        thresh = 0.5 if thresh is None else thresh
        delta = neurons[is_social].mean(axis=0) - neurons[~is_social].mean(axis=0)
        return np.abs(delta) > thresh
    if method == "percentile":
        thresh = 0.01 if thresh is None else thresh
        hi = np.percentile(neurons, 95, axis=0, keepdims=True)
        lo = np.percentile(neurons, 5, axis=0, keepdims=True)
        is_active = (neurons > hi) | (neurons < lo)
        is_responsive = is_active & is_social.reshape(-1, 1)
        return is_responsive.mean(axis=0) > thresh
    raise ValueError(f"unknown method {method!r} (ratio|delta|percentile)")


def sequence_sort(raster, thresh=5.0, descending=True):
    """Order neurons by the time of their first supra-threshold crossing (NB11 sequence view).

    ``raster`` is ``(n_neurons, T)`` (z-scored). Returns an index array ``sort_inds`` such that
    ``raster[sort_inds]`` is ordered by first-crossing time. Matches the 2025 script::

        first = argmax(raster > thresh, axis=1); argsort(first)[::-1]

    (``descending=True`` reproduces the script's ``[::-1]`` so early-firing neurons sit at the
    bottom of an imshow.) Neurons that never cross get ``argmax == 0`` and sort to the front.
    """
    raster = np.asarray(raster, dtype=float)
    first = np.argmax(raster > thresh, axis=1)
    order = np.argsort(first)
    return order[::-1] if descending else order


def background_subtract(frames):
    """Median-background removal + per-pixel z-score of the foreground (NB10 calcium extraction).

    Reproduces the 2025 calcium demo pipeline::

        bg = median(frames, 0); fg = frames - bg; fg = (fg - fg.mean(0)) / fg.std(0)

    ``frames`` is ``(F, H, W)``. Returns a dict::
        fg (F, H, W) z-scored foreground, bg (H, W) median background,
        mu (H, W), std (H, W) foreground mean/std used for the z-score.
    Take ``roi = fg[:, y0:y1, x0:x1].mean(axis=(1, 2))`` for a per-frame ROI trace.
    """
    frames = np.asarray(frames, dtype=np.float32)
    bg = np.median(frames, axis=0).astype(np.float32)
    fg = frames - bg
    mu = fg.mean(axis=0)
    sd = fg.std(axis=0)
    sd_safe = np.where(sd == 0, 1.0, sd)
    fg = (fg - mu) / sd_safe
    return dict(fg=fg, bg=bg, mu=mu, std=sd)


def motion_index(frames):
    """Mean absolute frame-to-frame difference — a scalar jitter score (NB09 motion correction).

    ``frames`` is ``(F, H, W)``. A well-stabilized movie has smaller consecutive-frame differences,
    so the expected ordering is ``motion_index(raw) > motion_index(rigid) > motion_index(pw_rigid)``.
    Returns a float."""
    frames = np.asarray(frames, dtype=np.float32)
    return float(np.abs(np.diff(frames, axis=0)).mean())


def motion_index_trace(frames):
    """Per-frame version of :func:`motion_index`: the mean |Δ| between each frame and the previous,
    as a ``(F-1,)`` array (handy for plotting a jitter timeseries)."""
    frames = np.asarray(frames, dtype=np.float32)
    return np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))


# ============================================================================ plotly house style
_MARGIN = dict(l=10, r=10, t=40, b=10)


def _apply(fig, title, height):
    fig.update_layout(template="plotly_white", height=height, margin=dict(_MARGIN))
    if title:
        fig.update_layout(title=title)
    return fig


def _robust_clim(M, lo=1.0, hi=99.0):
    """Robust (zmin, zmax) COLOR limits for an image/heatmap: the [lo, hi] percentiles (default
    1/99) of the finite values of ``M`` — so a few saturated pixels (a burned-in timestamp, one
    hot bin) don't blow out the color scale and hide the structure. Returns (None, None) when
    there are too few finite values or the spread is degenerate (caller then leaves color auto)."""
    v = np.asarray(M, float).ravel()
    v = v[np.isfinite(v)]
    if v.size < 3:
        return None, None
    a, b = (float(z) for z in np.nanpercentile(v, [lo, hi]))
    if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
        return None, None
    return a, b


def heatmap_fig(M, x=None, y=None, title="", xlabel="", ylabel="", colorscale="Viridis",
                zmid=None, zmin=None, zmax=None, colorbar_title="", height=420, robust=True):
    """House-style heatmap of a 2D matrix ``M`` (rows plotted top→bottom as given).

    ``robust`` (default True) clips the COLOR scale to the 1/99 percentile of ``M`` when the caller
    left ``zmin``/``zmax``/``zmid`` all unset, so a lone extreme cell doesn't wash out the rest.
    Any explicit ``zmin``/``zmax``/``zmid`` fully disables the auto-clip (backward compatible)."""
    import plotly.graph_objects as go
    M = np.asarray(M)
    if robust and zmin is None and zmax is None and zmid is None:
        zmin, zmax = _robust_clim(M)
    hm = go.Heatmap(z=M, x=x, y=y, colorscale=colorscale, zmid=zmid,
                    zmin=zmin, zmax=zmax, colorbar=dict(title=colorbar_title))
    fig = go.Figure(hm)
    fig.update_xaxes(title=xlabel, showgrid=False)
    fig.update_yaxes(title=ylabel, showgrid=False)
    return _apply(fig, title, height)


def image_fig(img, title="", colorscale="gray", zmin=None, zmax=None, colorbar_title="",
              equal_aspect=True, height=460, robust=True):
    """Display a 2D image (grayscale frame, correlation image, footprint) with image y-orientation
    (row 0 at top) and, by default, an equal-aspect (square-pixel) axis.

    ``robust`` (default True) clips the COLOR scale to the 1/99 percentile when ``zmin``/``zmax``
    are unset — the fix for max-/mean-projections where a burned-in timestamp or letterbox band
    saturates the scale and the cells vanish. Grid is off (spatial view)."""
    import plotly.graph_objects as go
    img = np.asarray(img)
    if robust and zmin is None and zmax is None:
        zmin, zmax = _robust_clim(img)
    fig = go.Figure(go.Heatmap(z=img, colorscale=colorscale, zmin=zmin, zmax=zmax,
                               colorbar=dict(title=colorbar_title)))
    fig.update_yaxes(autorange="reversed")
    if equal_aspect:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(visible=False, showgrid=False)
    fig.update_yaxes(visible=False, showgrid=False)
    return _apply(fig, title, height)


def raster_fig(neurons, x=None, y=None, title="", xlabel="Time (frames)", ylabel="Neuron",
               colorscale="Viridis", zmid=None, zmin=None, zmax=None, colorbar_title="",
               height=440, robust=True, zmax_floor=3.0):
    """Population raster: a ``(n_neurons, T)`` matrix as a heatmap (one row per neuron).

    Pass ``x`` (length T) to set explicit column coordinates — e.g. real frame indices after the
    time axis has been down-sampled for display — so overlays drawn in those units line up.

    ``robust`` (default True) sets the color scale from the 1/99 percentile of the data BUT keeps
    ``zmax`` at or above ``zmax_floor`` (default 3, roughly a z-scored "event" level): without the
    floor a quiet session's p99 sits in the noise (~1) and the whole raster saturates; with a lone
    hot bin the p99 keeps that bin from blanking every other row. Explicit ``zmin``/``zmax`` win."""
    z = np.asarray(neurons)
    if robust:
        lo, hi = _robust_clim(z, 1.0, 99.0)
        if zmin is None and lo is not None:
            zmin = lo
        if zmax is None and hi is not None:
            zmax = max(hi, float(zmax_floor))
    return heatmap_fig(z, x=x, y=y, title=title, xlabel=xlabel, ylabel=ylabel,
                       colorscale=colorscale, zmid=zmid, zmin=zmin, zmax=zmax,
                       colorbar_title=colorbar_title, height=height, robust=False)


def trace_fig(t, y, title="", xlabel="Time", ylabel="", names=None, height=280, robust=True):
    """Line plot of one or several 1D traces. ``y`` may be ``(T,)`` or ``(T, k)``/list-of-traces;
    ``t`` is the shared x (or None -> sample index).

    ``robust`` (default True) pins the y-axis to the 1/99 percentile of all the plotted values so a
    single spike can't auto-rescale the view and flatten the informative band (e.g. hiding the 6×
    cell/background gap in the NB7 ROI traces). Full traces are still drawn — only the default view
    is clipped; pass ``robust=False`` to auto-range."""
    import plotly.graph_objects as go
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    if t is None:
        t = np.arange(y.shape[0])
    fig = go.Figure()
    for j in range(y.shape[1]):
        nm = None if names is None else names[j]
        fig.add_scatter(x=t, y=y[:, j], mode="lines", name=nm,
                        showlegend=names is not None)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title=ylabel)
    if robust:
        ry = cu.robust_range(y)
        if ry:
            fig.update_yaxes(range=ry)
    return _apply(fig, title, height)


def overlay_fig(traj, pts, title="", traj_name="path", pts_name="spikes", pts_color="#e45756",
                traj_color="#4c78a8", height=560, equal_aspect=True, reverse_y=False, robust=True):
    """Trajectory + event-location overlay (NB12 spike-position plots): draw the animal's path
    ``traj`` ``(T, 2)`` as a line and overlay ``pts`` ``(K, 2)`` (e.g. positions at spike frames)
    as markers. Set ``reverse_y=True`` for image-style coordinates.

    ``robust`` (default True) clips both axes to the 1/99 percentile of the pooled trajectory+event
    coordinates so a stray tracking jump doesn't shrink the arena into a corner. Grid off (spatial);
    legend below the plot so it never collides with the title."""
    import plotly.graph_objects as go
    traj = np.asarray(traj, dtype=float)
    fig = go.Figure()
    fig.add_scatter(x=traj[:, 0], y=traj[:, 1], mode="lines",
                    line=dict(color=traj_color, width=1), name=traj_name, opacity=0.6)
    xs = [traj[:, 0]]
    ys = [traj[:, 1]]
    if pts is not None and len(pts):
        pts = np.asarray(pts, dtype=float)
        fig.add_scatter(x=pts[:, 0], y=pts[:, 1], mode="markers",
                        marker=dict(color=pts_color, size=5), name=pts_name)
        xs.append(pts[:, 0]); ys.append(pts[:, 1])
    if robust:
        rx = cu.robust_range(np.concatenate(xs))
        ry = cu.robust_range(np.concatenate(ys))
        if rx:
            fig.update_xaxes(range=rx)
        if ry:
            fig.update_yaxes(range=ry)
    if equal_aspect:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    if reverse_y:
        fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    cu.apply_house_style(fig, title=title or None, legend="below", height=height)
    return fig


def kymograph_fig(kymo, title="", xlabel="Position (px)", ylabel="Time (frames)",
                  colorscale="gray", zmin=None, zmax=None, height=460, robust=True):
    """Kymograph: a ``(T, W)`` slice (one image row sampled over time) as a heatmap with time on the
    y-axis (NB09 motion-correction jitter view). Straight vertical streaks = stable; wiggles =
    residual motion.

    ``robust`` (default True) clips the color scale to the 1/99 percentile when ``zmin``/``zmax``
    are unset, so a bright edge or timestamp streak doesn't wash out the motion contrast."""
    import plotly.graph_objects as go
    kymo = np.asarray(kymo)
    if robust and zmin is None and zmax is None:
        zmin, zmax = _robust_clim(kymo)
    fig = go.Figure(go.Heatmap(z=kymo, colorscale=colorscale, zmin=zmin, zmax=zmax,
                               showscale=False))
    fig.update_yaxes(autorange="reversed", title=ylabel, showgrid=False)
    fig.update_xaxes(title=xlabel, showgrid=False)
    return _apply(fig, title, height)




# ============================================================================ image / frame hygiene
# Remove burned-in overlays ONCE, up front, so the max-projection color scale isn't saturated by a
# timestamp or letterbox band and the "dead-space" background ROIs aren't contaminated (the bug
# behind the 2.9x vs 6x cell/background ratio in NB7).
def _hw_axes(frames):
    """Return (array, height_axis, width_axis) for a frame stack. Accepts (H, W), (F, H, W), or
    (F, H, W, C); raises on anything else."""
    a = np.asarray(frames)
    if a.ndim == 2:            # (H, W)
        return a, 0, 1
    if a.ndim == 3:            # (F, H, W)
        return a, 1, 2
    if a.ndim == 4:            # (F, H, W, C)
        return a, 1, 2
    raise ValueError(f"frames must be 2-4 D (H,W | F,H,W | F,H,W,C), got shape {a.shape}")


def crop_border(frames, top=0, bottom=0, left=0, right=0):
    """Crop fixed borders off every frame — the one-liner that removes miniscope letterbox bands so
    they stop dominating the max-projection color scale and contaminating background ROIs.

    ``frames`` may be a single image ``(H, W)``, a stack ``(F, H, W)``, or a color stack
    ``(F, H, W, C)``; the frame/channel axes are left untouched. ``top``/``bottom``/``left``/``right``
    are pixel counts trimmed from each edge (0 = keep). Returns a view into the input array."""
    a, hax, wax = _hw_axes(frames)
    H, W = a.shape[hax], a.shape[wax]
    ys = slice(int(top), H - int(bottom) if bottom else H)
    xs = slice(int(left), W - int(right) if right else W)
    idx = [slice(None)] * a.ndim
    idx[hax], idx[wax] = ys, xs
    return a[tuple(idx)]


def mask_region(frames, boxes, fill=0.0):
    """Paint one or more rectangular regions to a constant ``fill`` on every frame — used to blank a
    burned-in timestamp/scale-bar. Because the region becomes STATIC, a subsequent median-background
    subtraction removes it cleanly (a live clock survives median subtraction and saturates the
    projection; a constant patch does not).

    ``frames`` is ``(H, W)`` / ``(F, H, W)`` / ``(F, H, W, C)``. ``boxes`` is a single
    ``(y0, y1, x0, x1)`` tuple or an iterable of them (pixel bounds, half-open). Returns a COPY with
    the boxes filled; the input is not mutated."""
    a, hax, wax = _hw_axes(frames)
    a = a.copy()
    boxes = np.asarray(boxes)
    if boxes.ndim == 1:
        boxes = boxes[None, :]
    for y0, y1, x0, x1 in boxes:
        idx = [slice(None)] * a.ndim
        idx[hax] = slice(int(y0), int(y1))
        idx[wax] = slice(int(x0), int(x1))
        a[tuple(idx)] = fill
    return a


# ============================================================================ per-neuron-normalized shuffle
def per_neuron_normalized_shuffle(stats_obs, stats_null):
    """Score each neuron's observed statistic against ITS OWN circular-shift null, so a two-neuron
    comparison lands on a common normalized axis instead of a shared ABSOLUTE one. On an absolute
    axis a high-variance artifact neuron always sits "farthest right" regardless of whether it beats
    its own null (the NB8 §6 defect, brief 1b); the per-neuron z / percentile fixes that.

    Args:
        stats_obs   (N,) observed statistic per neuron (e.g. spatial information, social ratio).
        stats_null  (N, n_shuffles) null statistics — each row is that neuron's own shuffle
                    distribution. A (n_shuffles, N) array is accepted and transposed to match N.
    Returns a dict::
        z           (N,) (obs - null_mean) / null_std per neuron (0 where the null has no spread).
        percentile  (N,) fraction of that neuron's null strictly below its observed value (0..1).
        p           (N,) one-sided permutation p = (1 + #{null >= obs}) / (1 + n_shuffles).
        null_mean   (N,) per-neuron null mean.
        null_std    (N,) per-neuron null std.
    """
    obs = np.asarray(stats_obs, float).ravel()
    null = np.asarray(stats_null, float)
    N = obs.shape[0]
    if null.ndim != 2:
        raise ValueError("stats_null must be 2-D (N, n_shuffles)")
    if null.shape[0] != N and null.shape[1] == N:
        null = null.T
    if null.shape[0] != N:
        raise ValueError(f"stats_null neuron axis {null.shape} doesn't match {N} neurons")
    mu = np.nanmean(null, axis=1)
    sd = np.nanstd(null, axis=1)
    z = np.where(sd > 0, (obs - mu) / np.where(sd > 0, sd, 1.0), 0.0)
    pct = np.nanmean((null < obs[:, None]).astype(float), axis=1)
    nsh = null.shape[1]
    p = (1.0 + np.nansum((null >= obs[:, None]).astype(float), axis=1)) / (1.0 + nsh)
    return dict(z=z, percentile=pct, p=p, null_mean=mu, null_std=sd)


# ============================================================================ deduped shared builders
# These distribution + scatter figure builders are identical in intent to the behavior arm's, so we
# IMPORT them straight from course_utils rather than maintain a second copy. The neural notebooks
# thus inherit every round-3 fix automatically: violin points-drawn-on-top (fill opacity down,
# points as a separate scatter beside a half-violin), ecdf/scatter robust axis-clipping, the
# qualitative palette + rank auto-coloring, float64 stats. Re-bound at module scope so existing
# `nu.strip_points_fig(...)` call sites keep working unchanged.
strip_points_fig = cu.strip_points_fig
violin_points_fig = cu.violin_points_fig
box_points_fig = cu.box_points_fig
kde2d_fig = cu.kde2d_fig
ecdf_fig = cu.ecdf_fig
scatter_points_fig = cu.scatter_points_fig
umap_colored_by_feature_fig = cu.umap_colored_by_feature_fig

# Shared stat/figure helpers re-exported so neural-notebook authors reach them via nu.* without a
# separate `import course_utils` cell. All are pure numpy/scipy/sklearn/plotly with lazy heavy
# imports, so binding them here costs nothing at import time.
apply_house_style = cu.apply_house_style
robust_range = cu.robust_range
fmt_p = cu.fmt_p
scatter_marginal_fig = cu.scatter_marginal_fig
confusion_fig = cu.confusion_fig
blocked_cv_auroc = cu.blocked_cv_auroc
dominant_frequency = cu.dominant_frequency
gridness_score = cu.gridness_score
slopegraph_fig = cu.slopegraph_fig
dumbbell_fig = cu.dumbbell_fig
paired_diff_fig = cu.paired_diff_fig
wilson_ci = cu.wilson_ci
proportion_ci_fig = cu.proportion_ci_fig
loo_pc_auroc = cu.loo_pc_auroc
loo_pc_auroc_fig = cu.loo_pc_auroc_fig
