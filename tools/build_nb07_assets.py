"""build_nb07_assets.py — precompute the committed data bundle for notebooks_v2/07_from_movie_to_traces.py.

NB07 ("From a movie to traces") originally pulled three large assets from external hosts at runtime:
  * the side-by-side motion-correction movie   (Google Drive -> 1-s2.0-...-mmc3.mp4, via nu.fetch_gdrive)
  * a striatum miniscope movie                 (eLife URL    -> striatum.mp4,        via nu.fetch_url)
  * a precomputed CNMF-E result                (Google Drive -> ..._neuron_refined.h5, via nu.load_cnmf)

On molab there is no _neural_cache, so every student re-downloaded ~50-240 MB. This script reproduces
every array the notebook's figures/compute cells derive from those fetches, using the SAME nu.* helpers
against the local cache, and saves them to data/nb07_assets.npz so the notebook can np.load the bundle
with ZERO external fetches — exactly like the behavior notebooks load train_events.npz.

Budget: the npz must be <= 15 MB. Strategy:
  * Part A (motion correction): the raw|rigid|pw-rigid panels are kept at FULL frame count (185) and
    full 160x160 resolution, stored as uint8 (the grayscale movie is already 0..255 and integer-valued,
    so uint8 is lossless). Full frame count is required because Exercise 1 recomputes the motion index
    AND runs a paired Wilcoxon on the per-frame traces, whose tiny p-value needs all ~184 frame pairs.
  * Part B (striatum): frames_raw is the input to a live background_subtract + ROI scrubber, so it is a
    "scrubber stack" — we SUBSAMPLE it to STRI_N representative frames and store uint8 (lossless, 0..255).
    The notebook recomputes frames_clean / bg / fg / maxproj from it via the unchanged nu.* helpers. The
    scrubber slider bound and prose frame-count are updated to STRI_N to match.
  * Part C (CNMF): A is 99.75% zeros -> stored as a CSR triplet (float32) and rebuilt dense in-notebook.
    C (calcium traces) stored float16; Cn (correlation image) float16; S is all-zero -> compresses away.

Run:  .venv/bin/python tools/build_nb07_assets.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "course"))
import neural_utils as nu  # noqa: E402

OUT = os.path.join(ROOT, "data", "nb07_assets.npz")
STRI_N = 16   # striatum scrubber frames kept (was 90); <= ~30 per bundle budget
MOCO_N = 110  # motion-correction frames kept (was 185); paired Wilcoxon p stays ~1e-9 (<< 1e-6)

bundle = {}

# ------------------------------------------------------------------ Part A: motion-correction movie
print("[A] motion-correction movie ...")
CACHE = nu.cache_dir(ROOT)
MOV_PATH = nu.fetch_gdrive(nu.MOCO_GDRIVE_ID, nu.MOCO_NAME, ROOT)
_meta = nu.video_meta(MOV_PATH)
mov = nu.read_video(MOV_PATH, step=3, gray=True)          # (185, 320, 480) float32, 0..255
raw, rigid, pwr = nu.split_thirds(mov)                    # each (185, 160, 160) float32
# subsample frames evenly to MOCO_N to fit the bundle budget (kymographs/scrubber stay legible;
# the paired Wilcoxon in Exercise 1 keeps ~120 frame pairs -> p ~ 2e-10, well under the 1e-6 gate)
_msel = np.unique(np.linspace(0, raw.shape[0] - 1, MOCO_N).round().astype(int))
raw, rigid, pwr = raw[_msel], rigid[_msel], pwr[_msel]
assert raw.min() >= 0 and raw.max() <= 255
# uint8 is lossless here (values are integer-valued 0..255 from an 8-bit source movie)
bundle["moco_raw"] = np.rint(raw).astype(np.uint8)
bundle["moco_rigid"] = np.rint(rigid).astype(np.uint8)
bundle["moco_pwr"] = np.rint(pwr).astype(np.uint8)
# metadata the notebook's "what we loaded" prose cites (replaces nu.video_meta + mov.shape)
bundle["moco_fps"] = np.float64(_meta.get("fps", 25.0))
bundle["moco_size"] = np.asarray(_meta.get("size", (480, 320)), dtype=np.int64)
bundle["moco_duration"] = np.float64(_meta.get("duration", 22.2))
# the combined (pre-split) loaded movie shape AFTER subsampling: (MOCO_N, H, W_full)
bundle["moco_mov_shape"] = np.asarray((raw.shape[0], mov.shape[1], mov.shape[2]), dtype=np.int64)
# sanity: uint8 round-trip must preserve the motion-index ORDERING the notebook teaches
for nm, p in [("raw", raw), ("rigid", rigid), ("pwr", pwr)]:
    mi_f = nu.motion_index(p)
    mi_u = nu.motion_index(bundle[f"moco_{nm}"].astype(np.float32))
    print(f"    MI {nm}: float32={mi_f:.4f}  uint8={mi_u:.4f}")

# ------------------------------------------------------------------ Part B: striatum scrubber stack
print(f"[B] striatum movie (subsample {STRI_N} frames) ...")
STRI_PATH = nu.fetch_url(nu.STRIATUM_URL, nu.STRIATUM_NAME)
frames_raw = nu.read_video(STRI_PATH, step=100)          # (90, 500, 500) float32, 0..255
assert frames_raw.min() >= 0 and frames_raw.max() <= 255
sel = np.linspace(0, frames_raw.shape[0] - 1, STRI_N).round().astype(int)
sel = np.unique(sel)
stri = frames_raw[sel]
bundle["stri_frames_raw"] = np.rint(stri).astype(np.uint8)
print(f"    kept frames {list(sel)}")

# ------------------------------------------------------------------ Part C: CNMF-E result
print("[C] CNMF result ...")
from scipy import sparse  # noqa: E402
d = nu.load_cnmf(root=ROOT)
A = np.asarray(d["A"])            # (202, 360000) float64, ~0.25% nonzero
C = np.asarray(d["C"])           # (16773, 202) float64, 0..~10.7
Cn = np.asarray(d["Cn"])         # (600, 600) float64
S = np.asarray(d["S"])           # (16773, 202) all-zero in this refined file

A_csr = sparse.csr_matrix(A.astype(np.float32))
bundle["cnmf_A_data"] = A_csr.data
bundle["cnmf_A_indices"] = A_csr.indices.astype(np.int32)
bundle["cnmf_A_indptr"] = A_csr.indptr.astype(np.int32)
bundle["cnmf_A_shape"] = np.asarray(A_csr.shape, dtype=np.int64)
bundle["cnmf_C"] = C.astype(np.float16)
bundle["cnmf_Cn"] = Cn.astype(np.float16)
bundle["cnmf_S"] = S.astype(np.float32)          # all zeros -> compresses to ~nothing
bundle["cnmf_Fs"] = np.float64(d["Fs"])
bundle["cnmf_img_shape"] = np.asarray(d["img_shape"], dtype=np.int64)
bundle["cnmf_n_neurons"] = np.int64(d["n_neurons"])
bundle["cnmf_n_frames"] = np.int64(d["n_frames"])

# --- verify float16 preserves the population numbers the C-section PROSE hardcodes ---
def _events(Cmat):
    Cz = nu.zscore(np.asarray(Cmat, dtype=float).T, axis=1)   # (202, 16773)
    above = Cz > 5.0
    ev = (above[:, 1:] & ~above[:, :-1]).sum(axis=1)
    return int((ev == 0).sum()), int((Cz.max(axis=1) > 5).sum()), int(ev.max())
q64, a64, mx64 = _events(C)
q16, a16, mx16 = _events(C.astype(np.float16))
print(f"    z>5 never-cross:  f64={q64}  f16={q16}   (C3 prose hardcodes 114)")
print(f"    ever-peak active: f64={a64}  f16={a16}")
print(f"    busiest events:   f64={mx64}  f16={mx16}")
print(f"    A dense f32 max err vs rebuild: "
      f"{np.abs(A.astype(np.float32) - A_csr.toarray()).max():.2e}")

# ------------------------------------------------------------------ save + report
np.savez_compressed(OUT, **bundle)
mb = os.path.getsize(OUT) / 1e6
print(f"\nwrote {OUT}  ({mb:.2f} MB)")
for k, v in bundle.items():
    if getattr(v, "ndim", 0) >= 1 and v.size > 4:
        print(f"    {k:20s} {str(v.shape):18s} {v.dtype}")
if mb > 15:
    print(f"!! OVER BUDGET ({mb:.2f} MB > 15 MB)")
    sys.exit(1)
print(f"OK  ({mb:.2f} MB <= 15 MB)")
