"""Decode the bundled example `.slp` clips into a tiny npz the course can load WITHOUT sleap-io.

Why: `sleap-io>=0.6` pulls a heavy dependency chain (pynwb, skia-python, shapely, ...) that is
slow/fragile to install on bare cloud kernels (molab) and is not needed just to *view* a clip. So
we pre-decode one short clip here — on an instructor machine that has sleap-io — into fixed-order
keypoint arrays, and notebook 01 loads that instead.

This also fixes a correctness bug: `LabeledFrame.numpy()` returns instances in per-frame
instance-local order, so the same array slot maps to different tracks across frames (the skeleton
colors flicker as you scrub). Here we place each instance into its own fixed **track slot**, so
slot m is always the same animal.

Run (instructors only, needs the `build` extra):
    uv sync --extra build
    uv run python tools/decode_example_slp.py
"""
import os

import numpy as np
import sleap_io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw_slp")

# Prefer the same clip notebook 01 used to pick first.
CANDIDATES = ["example_dep.slp", "example_pre.slp", "example_post.slp", "example_heldout.slp"]
OUT = os.path.join(RAW, "example_slp_decoded.npz")


def decode(slp_path):
    labels = sio.load_slp(slp_path)
    tracks = list(labels.tracks)
    track_idx = {t: i for i, t in enumerate(tracks)}
    n_tracks = len(tracks)
    n_nodes = len(labels.skeletons[0].nodes)
    frames = labels.labeled_frames

    # (frames, tracks, nodes, 2) with each instance in its FIXED track slot; NaN where absent.
    kp = np.full((len(frames), n_tracks, n_nodes, 2), np.nan, np.float32)
    for fi, lf in enumerate(frames):
        for inst in lf.instances:
            if inst.track is not None:
                kp[fi, track_idx[inst.track]] = inst.numpy()

    skel = labels.skeletons[0]
    node_names = [n.name for n in skel.nodes]
    edges = np.array(
        [(node_names.index(e.source.name), node_names.index(e.destination.name))
         for e in skel.edges],
        dtype=np.int64,
    )
    return kp, np.array(node_names), edges, [t.name for t in tracks]


def main():
    slp_path = next((os.path.join(RAW, f) for f in CANDIDATES
                     if os.path.exists(os.path.join(RAW, f))), None)
    if slp_path is None:
        raise SystemExit(f"No example .slp found in {RAW}")
    kp, node_names, edges, track_names = decode(slp_path)
    np.savez_compressed(
        OUT, kp=kp, node_names=node_names, edges=edges,
        source=os.path.basename(slp_path), track_names=np.array(track_names),
    )
    print(f"wrote {OUT}")
    print(f"  source     {os.path.basename(slp_path)}")
    print(f"  kp shape   {kp.shape}  (frames, tracks, nodes, xy)")
    print(f"  tracks     {track_names}")
    print(f"  size       {os.path.getsize(OUT) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
