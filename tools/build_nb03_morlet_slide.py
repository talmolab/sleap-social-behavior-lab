"""Build data/assets/nb03_morlet_slide.gif — a Morlet wavelet sliding across a
signal, with the running wavelet response drawn beneath. Pure numpy+matplotlib,
GIF via imageio. Kept small: modest size, ~50 frames, palette-quantized.

Also verifies that a single static Morlet-wavelet figure is trivially buildable
(the notebook renders that live).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "assets", "nb03_morlet_slide.gif")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

plt.rcParams["font.family"] = "Liberation Sans"

# ---------------------------------------------------------------------------
# Morlet wavelet (real part) — a Gaussian-windowed cosine.
def morlet(t, w0=6.0):
    """Real Morlet: cosine carrier under a Gaussian envelope."""
    return np.cos(w0 * t) * np.exp(-(t ** 2) / 2.0)

# Wavelet kernel sampled on a compact support.
kt = np.linspace(-4, 4, 161)
w0 = 6.0
kernel = morlet(kt, w0)
env = np.exp(-(kt ** 2) / 2.0)

# ---------------------------------------------------------------------------
# Signal: a sine wave whose local frequency briefly matches the wavelet,
# so the response visibly peaks there (legible cause -> effect).
x = np.linspace(0, 20, 1000)
# base sine plus a burst near the middle at the wavelet's carrier frequency
signal = 0.6 * np.sin(1.1 * x)
burst = np.exp(-((x - 10.0) ** 2) / 6.0) * np.sin(3.0 * x)
signal = signal + burst
signal = signal / np.max(np.abs(signal))

# Precompute full response via sliding dot product (normalized correlation).
# We map the fixed-length kernel onto the signal's x-grid width.
dx = x[1] - x[0]
kernel_width_x = (kt[-1] - kt[0])            # 8.0 in signal x-units
n_k = int(round(kernel_width_x / dx))
# resample kernel to signal spacing
k_on_x = np.interp(np.linspace(kt[0], kt[-1], n_k), kt, kernel)
k_on_x = k_on_x - k_on_x.mean()
k_norm = k_on_x / np.sqrt(np.sum(k_on_x ** 2))

half = n_k // 2
centers = np.arange(half, len(x) - half)
response = np.full(len(x), np.nan)
for c in centers:
    seg = signal[c - half:c - half + n_k]
    response[c] = np.dot(seg, k_norm)
# normalize response to [-1,1] for display
rmax = np.nanmax(np.abs(response))
response_disp = response / rmax

# ---------------------------------------------------------------------------
# Animation frames: wavelet center slides left -> right across valid range.
N_FRAMES = 50
frame_centers = np.linspace(half, len(x) - half - 1, N_FRAMES).astype(int)

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"

frames = []
W_IN, H_IN, DPI = 4.6, 3.2, 90   # ~414x288 px

for c in frame_centers:
    cx = x[c]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(W_IN, H_IN), dpi=DPI, sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.0], "hspace": 0.15},
    )

    # --- Top: signal + wavelet positioned at current center ---
    ax1.plot(x, signal, color="0.55", lw=1.3, zorder=1)
    # wavelet mapped onto signal x-grid, scaled for visibility
    wx = np.linspace(cx - kernel_width_x / 2, cx + kernel_width_x / 2, n_k)
    wy = 0.85 * k_on_x / np.max(np.abs(k_on_x))
    wenv = 0.85 * np.interp(np.linspace(kt[0], kt[-1], n_k), kt, env)
    ax1.fill_between(wx, wenv, -wenv, color=ORANGE, alpha=0.12, zorder=2)
    ax1.plot(wx, wy, color=ORANGE, lw=1.8, zorder=3)
    ax1.axvline(cx, color=ORANGE, lw=0.8, ls=":", alpha=0.7, zorder=2)
    ax1.set_ylim(-1.25, 1.25)
    ax1.set_yticks([])
    ax1.set_title("Morlet wavelet sliding across a signal",
                  fontsize=10, pad=4)
    for s in ("top", "right", "left"):
        ax1.spines[s].set_visible(False)
    ax1.tick_params(labelbottom=False)

    # --- Bottom: running response, revealed up to current center ---
    mask = np.arange(len(x)) <= c
    r_show = np.where(mask, response_disp, np.nan)
    ax2.axhline(0, color="0.8", lw=0.8, zorder=1)
    ax2.plot(x, r_show, color=BLUE, lw=1.8, zorder=3)
    # marker at leading edge
    if not np.isnan(response_disp[c]):
        ax2.plot([cx], [response_disp[c]], "o", color=BLUE, ms=5, zorder=4)
    ax2.axvline(cx, color=ORANGE, lw=0.8, ls=":", alpha=0.7, zorder=2)
    ax2.set_ylim(-1.15, 1.15)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_ylabel("response", fontsize=9)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.set_xlabel("time", fontsize=9)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    frames.append(buf.copy())
    plt.close(fig)

# ---------------------------------------------------------------------------
# Palette-quantize each frame to shrink the GIF (few distinct colors here).
def quantize(frame, n_colors=32):
    from PIL import Image
    im = Image.fromarray(frame)
    return np.asarray(
        im.convert("P", palette=Image.ADAPTIVE, colors=n_colors)
        .convert("RGB")
    )

q_frames = [quantize(f) for f in frames]

imageio.mimsave(OUT, q_frames, format="GIF", duration=0.08, loop=0)

size_kb = os.path.getsize(OUT) / 1024
print(f"wrote {OUT}  ({size_kb:.1f} KB, {len(q_frames)} frames, "
      f"{frames[0].shape[1]}x{frames[0].shape[0]} px)")

# Save one representative frame for legibility inspection.
insp = os.path.join(HERE, "_nb03_frame_inspect.png")
imageio.imwrite(insp, frames[len(frames) // 2])
print(f"inspect frame: {insp}")

# ---------------------------------------------------------------------------
# Confirm the static Morlet figure is trivially buildable (notebook renders live).
figs, axs = plt.subplots(figsize=(4, 2.2))
axs.plot(kt, kernel, color=ORANGE, lw=2)
axs.plot(kt, env, color="0.6", lw=1, ls="--")
axs.plot(kt, -env, color="0.6", lw=1, ls="--")
axs.set_title("Morlet wavelet (real)")
static_png = os.path.join(HERE, "_nb03_static_inspect.png")
figs.tight_layout()
figs.savefig(static_png, dpi=90)
plt.close(figs)
print(f"static ok: {static_png}")
