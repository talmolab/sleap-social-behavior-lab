#!/usr/bin/env python3
"""Rebuild the fast_directed_*.gif exemplars for NB01 (data/exemplar_gifs/).

Video-backed clips: real homecage frames + the rank-colored SLEAP skeleton overlaid, zoomed on the
interacting pair (approacher slot 0, approachee slot 1), with a SMALL white direction arrow at the
approacher pointing toward the approachee. Rank colors match the lab convention used by the overlay
renderer (render_aggression_fleer): Dom=red, Mid=blue, Sub=green (drawn in BGR via cv2).

Overlay geometry (skeleton, crop, title band) is reused from the lab renderer. The source events were
recovered by content-matching the committed GIFs against train_events.npz (skeleton-overlay IoU); the
recovered event_keys are hard-coded below so this rebuild is deterministic.

Run with a python that has cv2 (e.g. /home/itang/miniconda3/bin/python). Reads raw video from
/snlkt/isaac/homecage/{cohort}/{stem}*.mp4 and keypoints from data/train_events.npz.
"""
import os, sys, glob, tempfile, subprocess
import numpy as np
import cv2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS = os.path.join(REPO, "data", "train_events.npz")
OUTDIR = os.path.join(REPO, "data", "exemplar_gifs")
VIDEO_DIRS = ["/snlkt/isaac/homecage", "/snlkt/data/Isaac/homecage"]

# NB01 skeleton (15 nodes) — matches course_utils.SKELETON_EDGES
SKELETON_EDGES = [(0, 1), (1, 2), (1, 5), (1, 3), (1, 6), (1, 4), (1, 11),
                  (11, 7), (11, 8), (11, 9), (11, 10), (11, 12), (12, 13), (12, 14)]
N_NODES = 15
# rank -> BGR (Dom red, Mid blue, Sub green)  [render_aggression_fleer convention]
RANK_BGR = {1: (0, 0, 255), 2: (255, 0, 0), 3: (0, 200, 0), 0: (170, 170, 170)}

WIDTH = 200          # content width in px (matches committed GIFs)
HDR = 22             # dark title band height
PAD_FRAC = 0.10      # crop padding as fraction of pair bbox side
FPS = 10
CONTACT_REL = 40     # contact frame index within the 130-frame event window
WIN = list(range(4, 4 + 23 * 4, 4))   # 23 event-frame indices spanning the approach + contact
ARROW_MAX = 26       # max white-arrow length in CONTENT px (shrunk from full centroid-to-centroid)

# recovered source events (see module docstring)
EXEMPLARS = {
    "fast_directed_1": "20260222_post|cam.09.00097-2026-03-02T12|m1-m2|33884",
    "fast_directed_2": "20260222_dep|cam.10.00282-2026-02-22T15|m1-m2|24312",
    "fast_directed_3": "20260222_post|cam.14.00043-2026-03-02T12|m0-m1|44638",
    "fast_directed_4": "12192025_post|cam.12.00094-2025-12-23T18|m1-m2|83764",
    "fast_directed_5": "20260222_dep|cam.10.00095-2026-02-22T15|m1-m2|83769",
}
TITLE = "fast/close/directed"


def find_video(coh, stem):
    for vd in VIDEO_DIRS:
        h = sorted(glob.glob(os.path.join(vd, coh, f"{stem}*.mp4")))
        if h:
            return h[0]
    return None


def centroid(kp_slot):
    v = np.isfinite(kp_slot).all(1)
    return kp_slot[v].mean(0) if v.sum() >= 3 else np.array([np.nan, np.nan])


def build_one(name, key, kp, ranks):
    coh, stem, _, cf = key.split("|")
    cf = int(cf)
    vid = find_video(coh, stem)
    assert vid, f"no video for {key}"

    # crop = bbox of the interacting PAIR (slots 0,1) over the shown window, padded
    pair = kp[WIN][:, :2, :, :]                     # (T,2,15,2)
    pts = pair[np.isfinite(pair).all(-1)]
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    w = x1 - x0; h = y1 - y0
    pad = PAD_FRAC * max(w, h)
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
    side_w = x1 - x0; side_h = y1 - y0
    # keep the crop near-square (aspect in [0.8, 1.25]) so the mice stay reasonably sized,
    # matching the committed exemplars — expand the shorter side about the bbox center.
    cxm = (x0 + x1) / 2; cym = (y0 + y1) / 2
    if side_h < 0.8 * side_w:
        side_h = 0.8 * side_w; y0 = cym - side_h / 2; y1 = cym + side_h / 2
    elif side_w < 0.8 * side_h:
        side_w = 0.8 * side_h; x0 = cxm - side_w / 2; x1 = cxm + side_w / 2

    cap = cv2.VideoCapture(vid)
    assert cap.isOpened()
    VW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); VH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # clamp crop into the frame (keep window size where possible by shifting, else clip)
    if x1 - x0 <= VW:
        if x0 < 0: x1 -= x0; x0 = 0
        if x1 > VW: x0 -= (x1 - VW); x1 = VW
    x0 = max(0.0, x0); x1 = min(float(VW), x1)
    if y1 - y0 <= VH:
        if y0 < 0: y1 -= y0; y0 = 0
        if y1 > VH: y0 -= (y1 - VH); y1 = VH
    y0 = max(0.0, y0); y1 = min(float(VH), y1)
    x0i, y0i, x1i, y1i = int(x0), int(y0), int(x1), int(y1)
    side_w = x1i - x0i; side_h = y1i - y0i
    sc = WIDTH / side_w
    content_h = int(round(side_h * sc))
    H_out = HDR + content_h

    def to_px(p):
        return (int((p[0] - x0i) * sc), HDR + int((p[1] - y0i) * sc))
    frames = []
    for gi, t in enumerate(WIN):
        cap.set(cv2.CAP_PROP_POS_FRAMES, cf - CONTACT_REL + t)
        ret, fr = cap.read()
        if not ret:
            break
        crop = fr[y0i:y1i, x0i:x1i]
        crop = cv2.resize(crop, (WIDTH, content_h), interpolation=cv2.INTER_LINEAR)
        # title band
        canvas = np.empty((H_out, WIDTH, 3), np.uint8)
        canvas[:HDR] = (20, 20, 20)
        canvas[HDR:] = crop
        # skeletons (pair thick, bystander thin+dim)
        for m in range(3):
            col = RANK_BGR.get(int(ranks[m]), RANK_BGR[0])
            pair_m = m in (0, 1)
            c = col if pair_m else tuple(int(x * 0.55) for x in col)
            thick = 2 if pair_m else 1
            rad = 3 if pair_m else 1
            k = kp[t, m]
            v = np.isfinite(k).all(1)
            for a, b in SKELETON_EDGES:
                if v[a] and v[b]:
                    cv2.line(canvas, to_px(k[a]), to_px(k[b]), c, thick, cv2.LINE_AA)
            for n in range(N_NODES):
                if v[n]:
                    cv2.circle(canvas, to_px(k[n]), rad, c, -1, cv2.LINE_AA)
        # small white direction arrow: approacher centroid -> toward approachee (length capped)
        ca = centroid(kp[t, 0]); cb = centroid(kp[t, 1])
        if np.isfinite(ca).all() and np.isfinite(cb).all():
            pa = np.array(to_px(ca), float); pb = np.array(to_px(cb), float)
            d = pb - pa; n = np.hypot(*d)
            if n > 1:
                tip = pa + d / n * min(n, ARROW_MAX)
                cv2.arrowedLine(canvas, (int(pa[0]), int(pa[1])), (int(tip[0]), int(tip[1])),
                                (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)
        # red contact dot (top-right of content) once contact has begun
        if t >= CONTACT_REL:
            cv2.circle(canvas, (WIDTH - 12, HDR + 10), 5, (0, 0, 255), -1, cv2.LINE_AA)
        # title text
        cv2.putText(canvas, TITLE, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (235, 235, 235), 1, cv2.LINE_AA)
        frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    cap.release()

    out = os.path.join(OUTDIR, name + ".gif")
    with tempfile.TemporaryDirectory() as td:
        import imageio.v2 as imageio
        for j, fr in enumerate(frames):
            imageio.imwrite(f"{td}/{j:03d}.png", fr)
        pal = f"{td}/pal.png"
        subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i", f"{td}/%03d.png",
                        "-vf", "palettegen=stats_mode=diff", pal], capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i", f"{td}/%03d.png", "-i", pal,
                        "-lavfi", "paletteuse=dither=bayer:bayer_scale=3", out], capture_output=True)
    return out, len(frames), os.path.getsize(out)


def main():
    d = np.load(EVENTS, allow_pickle=True)
    keys = d["event_key"].astype(str)
    kp_all = d["kp"].astype(np.float32)
    ranks_all = d["ranks"]
    for name, key in EXEMPLARS.items():
        idx = int(np.where(keys == key)[0][0])
        out, nf, sz = build_one(name, key, kp_all[idx], ranks_all[idx])
        print(f"{os.path.basename(out)}: {nf} frames, {sz/1e6:.3f} MB")


if __name__ == "__main__":
    main()
