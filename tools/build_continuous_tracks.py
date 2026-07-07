"""Build data/continuous_tracks.npz for NB07 (the activity clock + Markov grammar).

Extracts ONE continuous ~24h span (2 fps) for three cages from the lab's per-30min tracks
matrices, so students can analyze behavior OVER TIME (circadian activity, behavioral-state
sequences) without any live pose-IO. Pure numpy — reads only `{TM_BASE}/{cohort}/{stem}_tracks_matrix.npz`
(key 'tracks_matrix', shape (frames,15,2,3)); no despotism utils / sleap-io needed.

Cages: 15 (hero, M) + 10 (context, F) + 13 (context, M) — gives a sex contrast for the clock.
Downsampled 50fps->2fps, body-centroid per track, cage-level behavioral states from kinematics.

Instructors only:  uv run python tools/build_continuous_tracks.py
"""
import os
import re
import glob
from concurrent.futures import ProcessPoolExecutor

import numpy as np

TM_BASE = "/snlkt/isaac/id_switch/despotism/tracks_matrices"
COHORT = "20260222_dep"           # the food-restriction phase
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "continuous_tracks.npz")

CAMS = {"15": "M", "10": "F", "13": "M"}   # hero(M), context(F), context(M)
HERO = "15"
N_CHUNKS = 48                     # 48 * 30 min = 24 h continuous
FPS = 50
DS = 25                           # 50 -> 2 fps
BODY = list(range(9))             # nose..R_haunch, the compact body used for the centroid
DATE_RE = re.compile(r"-(\d{4}-\d{2}-\d{2})T(\d{2})")


def chunk_index(path):
    m = re.search(r"cam\.\d+\.(\d+)-", os.path.basename(path))
    return int(m.group(1)) if m else 10**9


def process_chunk(path):
    """One 30-min chunk -> (n_ds, 3, 2) body centroids at 2 fps (float32; nan where all-nan)."""
    tm = np.load(path)["tracks_matrix"]            # (frames,15,2,3)
    tm = np.transpose(tm, (0, 3, 1, 2))            # (frames,3,15,2)
    body = tm[:, :, BODY, :]                       # (frames,3,9,2)
    with np.errstate(invalid="ignore"):
        cen = np.nanmean(body, axis=2)             # (frames,3,2)
    return cen[::DS].astype(np.float32)            # (n_ds,3,2)


def start_hour(first_stem):
    m = DATE_RE.search(first_stem)
    return int(m.group(2)) if m else 15


def build_cam(cam):
    files = sorted(glob.glob(f"{TM_BASE}/{COHORT}/*cam.{cam}.*_tracks_matrix.npz"), key=chunk_index)
    files = files[:N_CHUNKS]
    if not files:
        print(f"  cam {cam}: NO FILES at {TM_BASE}/{COHORT}")
        return None
    with ProcessPoolExecutor(max_workers=8) as ex:
        cens = list(ex.map(process_chunk, files))
    cen = np.concatenate(cens, axis=0)             # (T,3,2) at 2 fps
    T = cen.shape[0]
    # speed (px/s): centroid displacement between 2-fps samples * 2
    d = np.linalg.norm(np.diff(cen, axis=0), axis=2) * 2.0   # (T-1,3)
    speed = np.vstack([d, d[-1:]])                            # (T,3) pad last
    # time-of-day: chunks are contiguous 30-min blocks from the first chunk's start hour
    h0 = start_hour(os.path.basename(files[0]))
    tod = (h0 + np.arange(T) / (FPS / DS) / 3600.0) % 24.0    # (T,)
    return dict(cen=cen.astype(np.float16), speed=speed.astype(np.float16),
                tod=tod.astype(np.float16), n_files=len(files))


def discretize(speed, cen):
    """Cage-level behavioral state per timepoint from kinematics (data-driven thresholds):
       0 rest (all mice slow), 2 huddle (two mice very close), 1 locomote (otherwise moving)."""
    mean_speed = np.nanmean(speed, axis=1)                    # (T,)
    # min pairwise centroid distance among the 3 tracks
    dif = cen[:, :, None, :] - cen[:, None, :, :]             # (T,3,3,2)
    pd = np.linalg.norm(dif, axis=3)                          # (T,3,3)
    iu = np.triu_indices(3, k=1)
    min_pair = np.nanmin(pd[:, iu[0], iu[1]], axis=1)         # (T,)
    s_move = np.nanpercentile(mean_speed, 40)                 # below = resting
    d_close = np.nanpercentile(min_pair, 25)                  # below = huddling
    state = np.ones(len(mean_speed), np.int8)                 # default locomote
    state[mean_speed < s_move] = 0                            # rest
    state[min_pair < d_close] = 2                             # huddle (overrides)
    return state, ["rest", "locomote", "huddle"], float(s_move), float(d_close)


def main():
    out = {"cams": np.array(list(CAMS)), "sex": np.array([CAMS[c] for c in CAMS]),
           "hero": HERO, "fps": FPS / DS, "cohort": COHORT}
    for cam in CAMS:
        print(f"cam {cam} ({CAMS[cam]}) ...", flush=True)
        r = build_cam(cam)
        if r is None:
            continue
        state, names, s_move, d_close = discretize(r["speed"].astype(np.float32),
                                                   r["cen"].astype(np.float32))
        out[f"cam{cam}_cen"] = r["cen"]
        out[f"cam{cam}_speed"] = r["speed"]
        out[f"cam{cam}_tod"] = r["tod"]
        out[f"cam{cam}_state"] = state
        print(f"  cam {cam}: T={r['cen'].shape[0]} ({r['n_files']} chunks), "
              f"states rest/loco/huddle = {np.bincount(state, minlength=3).tolist()}")
    out["state_names"] = np.array(["rest", "locomote", "huddle"])
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
