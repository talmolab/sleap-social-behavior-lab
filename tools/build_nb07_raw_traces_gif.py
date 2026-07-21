"""Build data/assets/nb07_raw_traces.gif — a short animation of raw calcium traces
drawing in over time behind a moving time cursor. Source: data/nb07_assets.npz ONLY.

Matplotlib GIF styled to echo the plotly_white house style (white bg, tight, muted
qualitative colorway). Kept small: 6 traces, modest size, ~64 frames, palette-quantized.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NPZ = ROOT / "data" / "nb07_assets.npz"
OUT = ROOT / "data" / "assets" / "nb07_raw_traces.gif"
OUT.parent.mkdir(parents=True, exist_ok=True)

d = np.load(NPZ, allow_pickle=True)
C = np.asarray(d["cnmf_C"], np.float32)          # (T, N) temporal components
Fs = float(d["cnmf_Fs"])                          # 30 Hz

# --- pick a lively window and a handful of active ROIs ------------------------
N_TRACES = 6
WIN_SEC = 45.0
win = int(WIN_SEC * Fs)
# choose the window (start) whose summed activity across neurons is largest
step = win // 2
starts = range(0, C.shape[0] - win, step)
best = max(starts, key=lambda s: float(C[s:s + win].sum()))
seg = C[best:best + win]                           # (win, N)
# most active neurons within that window
order = np.argsort(seg.max(0))[::-1]
rois = np.sort(order[:N_TRACES])                   # keep vertical order stable

# down-sample time for a light GIF (~5 Hz display)
DS = 6
tv = np.arange(0, seg.shape[0], DS)
Y = seg[tv][:, rois]                               # (Tds, N_TRACES)
tsec = tv / Fs
Td = Y.shape[0]

# normalize each trace to [0,1] then stack with offsets
Yn = (Y - Y.min(0)) / (np.ptp(Y, 0) + 1e-6)
OFF = 1.25
offsets = np.arange(N_TRACES) * OFF

# plotly_white default qualitative colorway
COLORWAY = ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3"]

# --- animate: traces draw in left->right behind a time cursor -----------------
N_FRAMES = 64
FIG_W, FIG_H, DPI = 5.0, 3.1, 72
cursor_idx = np.linspace(1, Td, N_FRAMES).astype(int)

frames = []
for k in cursor_idx:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    for j in range(N_TRACES):
        base = offsets[j]
        ax.plot(tsec[:k], Yn[:k, j] + base, color=COLORWAY[j], lw=1.3)
    xc = tsec[k - 1]
    ax.axvline(xc, color="#888888", lw=1.0, alpha=0.7)
    ax.set_xlim(tsec[0], tsec[-1])
    ax.set_ylim(-0.4, offsets[-1] + 1.4)
    ax.set_yticks(offsets)
    ax.set_yticklabels([f"ROI {int(r)}" for r in rois], fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_title("Raw calcium traces", fontsize=10)
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(False)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.tight_layout(pad=0.6)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    frames.append(buf)
    plt.close(fig)

# palette-quantize to shrink the GIF hard
imageio.mimsave(OUT, frames, format="GIF", duration=1000.0 / 16, loop=0,
                subrectangles=True, palettesize=64)

# also drop one frame for inspection
imageio.imwrite("/tmp/nb07_traces_frame.png", frames[len(frames) * 3 // 4])

sz = OUT.stat().st_size
print(f"window start frame={best} ({best/Fs:.1f}s) rois={rois.tolist()}")
print(f"frames={len(frames)} size={Y.shape} gif={sz/1024:.1f} KiB -> {OUT}")
