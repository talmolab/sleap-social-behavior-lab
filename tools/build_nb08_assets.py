#!/usr/bin/env python
"""build_nb08_assets.py — precompute the committed data bundle for notebook
``notebooks_v2/08_reading_the_population.py`` so the student path loads ONLY from
``data/nb08_assets.npz`` with ZERO runtime external fetches.

NB08 pulls three external datasets at runtime via the ``neural_utils`` (nu) helpers,
all cached under ``data/_neural_cache`` locally but absent on molab:

  * ``nu.load_cnmf()``          — striatal CNMF-E session (Google Drive h5, ~34 MB)
  * ``nu.fetch_zip_dropbox`` + ``nu.load_rat_mat`` — rat place/grid .mat (Dropbox zip)
  * ``nu.load_si()``            — social-isolation cohort (240 MB calcium h5 + bouts + xlsx)

Every figure / compute cell in NB08 derives from small products of those raw files:

  Part 1  (CNMF)   C_z = zscore(C.T)  (202, 16773) raster;  Cn correlation image (600,600)
  Part 2a (rat)    per-session centroid (T,2) + spikes (T,n) drive rate maps / SI / gridness
  Part 2b+4 (SI)   per-session z-scored+cropped neural block (4500, n_cells) + is_social_sender
                   label (4500,) drive the ECDF, ratio scatter, leakage decoders, ROC, and the
                   run_all-gated 18-session blocked-CV scatter.

NB08 has NO raw-frame video scrubber (unlike NB07): all sliders act on DERIVED arrays, so
nothing here is a raw movie — we never touch striatum.mp4 / the 240 MB calcium raster past the
per-session traces actually displayed.

COMPRESSION (hard budget: npz <= 12 MB).  The neural blocks are large (C_z alone is 202*16773;
the 18 SI sessions total 17.7 M samples), so we:
  * quantize each z-scored neural block to int8 over a fixed z-range (step ~0.12 z);
  * DELTA-encode along the time axis.  Calcium is smooth, and the max |consecutive quantized
    diff| is 41 (SI) / 56 (C_z) < 127, so the int8 delta is LOSSLESS w.r.t. the int8
    quantization — cumsum reconstructs the quantized block exactly.
  * store single images (Cn) as float16 and rat centroid as float16, spikes as uint8 (lossless).

Run:  .venv/bin/python tools/build_nb08_assets.py            # build + verify
"""
from __future__ import annotations
import os, sys, io
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "course"))
import neural_utils as nu  # noqa: E402

OUT = os.path.join(ROOT, "data", "nb08_assets.npz")

# quant ranges (z-units): asymmetric because z-scored calcium has a long positive (transient) tail
SI_LO, SI_HI = -8.0, 24.0
# C_z is only ever used with display floor z=0 and activation thresholds in [3, 6]; a tight [-2, 14]
# range gives a fine 0.063-z step in that critical band (reproduces the split-half CV / sequenceness
# pinned numbers exactly) while keeping the int8 time-delta overflow-safe (max |consecutive diff|=103).
CZ_LO, CZ_HI = -2.0, 14.0


def q8(a, lo, hi):
    """Quantize float z-scores to int8 over [lo, hi] (256 levels, -128..127)."""
    a = np.clip(np.asarray(a, np.float32), lo, hi)
    return np.round((a - lo) / (hi - lo) * 255.0 - 128.0).astype(np.int8)


def dq8(q, lo, hi):
    """Inverse of q8: int8 -> approximate float z-score."""
    return (q.astype(np.float32) + 128.0) / 255.0 * (hi - lo) + lo


def delta_encode(q, axis):
    """Delta-encode an int8 array along ``axis``. First slice kept verbatim; rest are diffs.
    Lossless as long as consecutive diffs fit int8 (verified for these data)."""
    q16 = q.astype(np.int16)
    d = np.diff(q16, axis=axis)
    first = np.take(q16, [0], axis=axis)
    out = np.concatenate([first, d], axis=axis)
    assert out.min() >= -128 and out.max() <= 127, "delta overflow int8"
    return out.astype(np.int8)


def delta_decode(d, axis):
    """Inverse of delta_encode: cumulative sum along ``axis`` -> int8 block (as int16, bounded)."""
    return np.cumsum(d.astype(np.int16), axis=axis)


# ------------------------------------------------------------------ build
def build():
    store = {}

    # ---- Part 1 · CNMF ----
    print("loading CNMF ...")
    d = nu.load_cnmf()
    C = d["C"]                                   # (n_frames, n_neurons)
    Cn = d["Cn"]                                 # (H, W)
    C_z = nu.zscore(C.T, axis=1).astype(np.float32)   # (n_neurons, n_frames) — matches notebook
    store["cz_delta"] = delta_encode(q8(C_z, CZ_LO, CZ_HI), axis=1)
    store["Cn"] = Cn.astype(np.float16)
    store["Fs"] = np.float32(d["Fs"])
    store["n_frames"] = np.int64(d["n_frames"])
    store["n_neurons"] = np.int64(d["n_neurons"])

    # ---- Part 2a · rat place/grid ----
    print("loading rat sessions ...")
    rat_dir = nu.fetch_zip_dropbox(root=ROOT)
    rat_names = list(nu.RAT_FILES)
    store["rat_names"] = np.array(rat_names)
    for i, name in enumerate(rat_names):
        rd = nu.load_rat_mat(os.path.join(rat_dir, name))
        store[f"rat_centroid_{i}"] = rd["centroid"].astype(np.float16)
        store[f"rat_spikes_{i}"] = rd["spikes"].astype(np.uint8)   # counts, max 10 -> lossless

    # ---- Part 2b + 4 · social isolation ----
    print("loading SI (240 MB calcium) and precomputing per-session blocks ...")
    si = nu.load_si()
    ent, beh, img = si["entrances"], si["behavior"], si["imaging"]
    ns, bfps = si["n_sessions"], si["behavior_fps"]
    cond_labels = [nu.si_condition_label(ent["Isolation Length"].iloc[s]) for s in range(ns)]
    store["si_cond_labels"] = np.array(cond_labels)
    store["si_n_sessions"] = np.int64(ns)
    for s in range(ns):
        iss = beh[s]["is_social_sender"].astype(bool)
        r = nu.zscore(nu.interp_resample(img[s], len(iss), axis=0), axis=0)   # exact notebook order
        e = int(ent["Int_Entry"].iloc[s]); t0, t1 = e, int(e + 3 * 60 * bfps)
        block = r[t0:t1].astype(np.float32)          # (T_crop, n_cells)
        soc = iss[t0:t1]
        store[f"si_delta_{s}"] = delta_encode(q8(block, SI_LO, SI_HI), axis=0)
        store[f"si_social_{s}"] = soc.astype(bool)

    store["si_lo"] = np.float32(SI_LO); store["si_hi"] = np.float32(SI_HI)
    store["cz_lo"] = np.float32(CZ_LO); store["cz_hi"] = np.float32(CZ_HI)

    print(f"saving -> {OUT}")
    np.savez_compressed(OUT, **store)
    mb = os.path.getsize(OUT) / 1e6
    print(f"bundle size: {mb:.2f} MB  (budget 12 MB)")
    return mb


# ------------------------------------------------------------------ verify fidelity
def verify():
    print("\n=== verifying reconstructed bundle vs. live originals ===")
    z = np.load(OUT, allow_pickle=False)

    # --- C_z reconstruct ---
    C_z_r = dq8(delta_decode(z["cz_delta"], axis=1), float(z["cz_lo"]), float(z["cz_hi"]))
    d = nu.load_cnmf()
    C_z0 = nu.zscore(d["C"].T, axis=1).astype(np.float32)
    print(f"C_z shape {C_z_r.shape} vs {C_z0.shape}; max|abs err| {np.abs(C_z_r - C_z0).max():.4f} "
          f"(quant step {(float(z['cz_hi'])-float(z['cz_lo']))/255:.4f})")

    # sequenceness pinned numbers (Part 3 / exercise 2): ENTRY, WIN_LEN
    from scipy.stats import spearmanr
    ENTRY = int(7488 * (30 / 25)); WIN_LEN = 3 * 60 * 30

    def seqness(raster, thr=5.0):
        first = np.argmax(raster > thr, axis=1)
        r, _ = spearmanr(np.arange(raster.shape[0]), first)
        return 0.0 if np.isnan(r) else abs(float(r))

    for tag, cz in [("orig", C_z0), ("recon", C_z_r)]:
        win = cz[:, ENTRY:ENTRY + WIN_LEN]
        order = nu.sequence_sort(win, thresh=5.0)
        print(f"  [{tag}] seq_unsorted={seqness(win):.3f}  seq_sorted={seqness(win[order]):.3f}")

    # split-half CV (Part 3)
    def splithalf(cz):
        win = cz[:, ENTRY:ENTRY + WIN_LEN]; half = WIN_LEN // 2
        A_h, B_h = win[:, :half], win[:, half:]
        def heldout(ro):
            b = B_h[ro]; active = b.max(axis=1) > 5.0
            first = np.argmax(b > 5.0, axis=1)[active]
            if active.sum() < 3: return 0.0
            r, _ = spearmanr(np.arange(int(active.sum())), first)
            return 0.0 if np.isnan(r) else abs(float(r))
        learned = nu.sequence_sort(A_h, thresh=5.0)
        cvl = heldout(learned)
        rng = np.random.RandomState(1)
        sh = np.array([heldout(rng.permutation(win.shape[0])) for _ in range(500)])
        p = float((1 + np.sum(sh >= cvl)) / (1 + len(sh)))
        return cvl, p
    print(f"  split-half orig  = {splithalf(C_z0)}")
    print(f"  split-half recon = {splithalf(C_z_r)}")

    # --- rat SI pinned numbers (neuron 5 real, neuron 10 artifact) ---
    rat_dir = nu.fetch_zip_dropbox(root=ROOT)
    name = "20160609T194655.mat"
    rd = nu.load_rat_mat(os.path.join(rat_dir, name))
    ctr0, spk0 = rd["centroid"], rd["spikes"]
    i = list(nu.RAT_FILES).index(name)
    ctr_r = z[f"rat_centroid_{i}"].astype(np.float32); spk_r = z[f"rat_spikes_{i}"]

    def si_of(ctr, spk_col):
        rm = nu.rate_map(ctr[:, 0], ctr[:, 1], spk_col, bins=20)
        return nu.spatial_information(rm["rate"], rm["occupancy"])
    for nrn in (5, 10):
        print(f"  neuron {nrn}: SI orig={si_of(ctr0, spk0[:, nrn]):.3f}  recon={si_of(ctr_r, spk_r[:, nrn]):.3f}")
    print(f"  rat spikes lossless: {np.array_equal(spk0.astype(np.uint8), spk_r)}")

    # --- SI decoder pinned numbers (session 6 leakage: shuffle vs blocked) ---
    si = nu.load_si()
    ent, beh, img, bfps = si["entrances"], si["behavior"], si["imaging"], si["behavior_fps"]
    s = 6
    iss = beh[s]["is_social_sender"].astype(bool)
    r = nu.zscore(nu.interp_resample(img[s], len(iss), axis=0), axis=0)
    e = int(ent["Int_Entry"].iloc[s]); t0, t1 = e, int(e + 3 * 60 * bfps)
    X0 = r[t0:t1].astype(np.float32); y0 = iss[t0:t1].astype(int)
    X_r = dq8(delta_decode(z["si_delta_%d" % s], axis=0), float(z["si_lo"]), float(z["si_hi"]))
    y_r = z["si_social_%d" % s].astype(int)
    print(f"  SI s6 block shape {X_r.shape} vs {X0.shape}; label match {np.array_equal(y0, y_r)}; "
          f"max|abs err| {np.abs(X_r - X0).max():.4f}")

    def mk():
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        return Pipeline([("scale", StandardScaler()),
                         ("lr", LogisticRegression(solver="liblinear", C=0.1,
                                                   class_weight="balanced", max_iter=1000))])
    for tag, X, y in [("orig", X0, y0), ("recon", X_r, y_r)]:
        sh = float(np.nanmean(nu.blocked_cv_auroc(X, y, scheme="shuffle", clf=mk())))
        bl = float(np.nanmean(nu.blocked_cv_auroc(X, y, scheme="blocked", clf=mk())))
        print(f"  [{tag}] AUROC shuffle={sh:.3f}  blocked={bl:.3f}  gap={sh-bl:.3f}")

    z.close()


if __name__ == "__main__":
    build()
    verify()
