"""Original schematic: Ca2+ entry via voltage-gated calcium channel -> indicator fluorescence."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch, Polygon

plt.rcParams["font.family"] = "Liberation Sans"
plt.rcParams["svg.fonttype"] = "none"

CA = r"$\mathrm{Ca^{2+}}$"

# palette (neutral / pedagogical)
C_OUT = "#eef4fb"; C_IN = "#fef6ec"
C_MEM = "#f3c9a0"; C_MEMTAIL = "#d9a877"
C_CA = "#2c7fb8"; C_CA_EDGE = "#1a4f73"
C_CHAN = "#7fb069"; C_CHAN_EDGE = "#3f6b2e"
C_GCAMP_DARK = "#9e9e9e"
C_GLOW = "#f6e04b"; C_GLOW_EDGE = "#c9a800"
C_TXT = "#222222"; C_STEP = "#c0392b"

fig, ax = plt.subplots(figsize=(11.5, 7.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 64); ax.axis("off")

# --- regions ---------------------------------------------------------------
mem_lo, mem_hi = 36, 44
ax.add_patch(plt.Rectangle((0, mem_hi), 100, 64 - mem_hi, facecolor=C_OUT, edgecolor="none", zorder=0))
ax.add_patch(plt.Rectangle((0, 0), 100, mem_lo, facecolor=C_IN, edgecolor="none", zorder=0))

# lipid bilayer with a pore gap
pore_x0, pore_x1 = 43, 57
head_r = 1.15
for x in np.arange(2, 100, 3.0):
    if pore_x0 - 1 < x < pore_x1 + 1:
        continue
    ax.add_patch(Circle((x, mem_hi - head_r - 0.2), head_r, facecolor=C_MEM, edgecolor=C_MEMTAIL, lw=0.5, zorder=2))
    ax.plot([x, x], [mem_hi - 2*head_r - 0.2, (mem_lo+mem_hi)/2], color=C_MEMTAIL, lw=0.8, zorder=1)
    ax.add_patch(Circle((x, mem_lo + head_r + 0.2), head_r, facecolor=C_MEM, edgecolor=C_MEMTAIL, lw=0.5, zorder=2))
    ax.plot([x, x], [mem_lo + 2*head_r + 0.2, (mem_lo+mem_hi)/2], color=C_MEMTAIL, lw=0.8, zorder=1)

ax.text(3, 61.5, "Outside the cell (extracellular)", fontsize=14.5, fontweight="bold", color=C_TXT, va="top")
ax.text(3, 33, "Inside the cell (cytoplasm)", fontsize=14.5, fontweight="bold", color=C_TXT, va="top")
ax.text(97, mem_hi + 3.0, "cell membrane (lipid bilayer)", fontsize=10.5, style="italic", color="#8a5a2b", ha="right", va="bottom")

# --- voltage-gated calcium channel (funnel through the pore) ---------------
cx = (pore_x0 + pore_x1) / 2
neck = 3.0
verts_L = [(cx - 7, mem_hi + 2.5), (cx - neck, mem_hi - 1.5),
           (cx - neck, mem_lo + 1.5), (cx - 4.5, mem_lo - 2.5)]
verts_R = [(cx + 4.5, mem_lo - 2.5), (cx + neck, mem_lo + 1.5),
           (cx + neck, mem_hi - 1.5), (cx + 7, mem_hi + 2.5)]
for vs in (verts_L, verts_R):
    ax.add_patch(Polygon(vs, closed=True, facecolor=C_CHAN, edgecolor=C_CHAN_EDGE, lw=1.6, zorder=3))
ax.text(cx, mem_lo - 10.5, "voltage-gated\ncalcium channel", fontsize=12, fontweight="bold",
        color=C_CHAN_EDGE, ha="center", va="top")
# depolarization charge cue
for sx in (cx - 9.5, cx + 9.5):
    ax.text(sx, (mem_lo+mem_hi)/2, "+\n+\n+", fontsize=10.5, color=C_STEP,
            ha="center", va="center", fontweight="bold", linespacing=0.9)

def ca_ion(x, y, r=1.7, z=5, sup=True):
    ax.add_patch(Circle((x, y), r, facecolor=C_CA, edgecolor=C_CA_EDGE, lw=1.0, zorder=z))
    ax.text(x, y, "Ca", fontsize=8.2, color="white", ha="center", va="center", fontweight="bold", zorder=z+1)
    if sup:
        ax.text(x + r*0.9, y + r*0.75, "2+", fontsize=6.2, color=C_CA_EDGE, ha="left", va="bottom", zorder=z+1)

# extracellular Ca2+ pool (kept clear of the step-1 label lane on the right)
out_pts = [(11,58),(19,52),(29,58),(37,50),(13,47),(27,47),(66,58),(78,58),(90,58),
           (84,50),(72,50),(95,50)]
for (x, y) in out_pts:
    ca_ion(x, y)
ax.text(50, 61.0, CA + " is plentiful outside the cell", fontsize=11, style="italic",
        color=C_CA_EDGE, ha="center")

# Ca2+ streaming down through the open channel
for (x, y) in [(cx, mem_hi + 1.0), (cx, (mem_lo+mem_hi)/2), (cx, mem_lo - 1.5)]:
    ca_ion(x, y, r=1.5, z=6)
# influx arrow (channel -> down into cytoplasm)
ax.add_patch(FancyArrowPatch((cx, mem_hi + 5.5), (cx, mem_lo - 5.5),
             arrowstyle="-|>", mutation_scale=24, lw=3.0, color=C_STEP, zorder=4))
ax.text(cx - 4.5, mem_lo - 4.0, CA + " influx", fontsize=12, fontweight="bold", color=C_STEP, ha="right", va="center")

# --- indicator (GCaMP): unbound (dim) -> bound (glowing), LEFT half --------
gx0, gy0 = 14, 15
ax.add_patch(Circle((gx0, gy0), 3.4, facecolor=C_GCAMP_DARK, edgecolor="#6f6f6f", lw=1.4, zorder=5))
ax.text(gx0, gy0, "GCaMP", fontsize=8.0, color="white", ha="center", va="center", fontweight="bold", zorder=6)
ax.text(gx0, gy0 - 5.6, "indicator, dim\n(no calcium bound)", fontsize=10, color="#5a5a5a", ha="center", va="top")

ax.add_patch(FancyArrowPatch((gx0 + 5, gy0 + 0.5), (35 - 5, gy0 + 0.5),
             arrowstyle="-|>", mutation_scale=22, lw=2.4, color=C_TXT, zorder=4))
ax.text((gx0 + 35)/2, gy0 + 3.0, CA + " binds", fontsize=10.5, color=C_TXT, ha="center")

gx1, gy1 = 38, 15
for rr, aa in [(9.0, 0.12), (7.0, 0.20), (5.3, 0.35)]:
    ax.add_patch(Circle((gx1, gy1), rr, facecolor=C_GLOW, edgecolor="none", alpha=aa, zorder=4))
for ang in np.linspace(0, 2*np.pi, 12, endpoint=False):
    ax.plot([gx1 + 4.2*np.cos(ang), gx1 + 8.0*np.cos(ang)],
            [gy1 + 4.2*np.sin(ang), gy1 + 8.0*np.sin(ang)],
            color=C_GLOW_EDGE, lw=1.5, zorder=4, alpha=0.8)
ax.add_patch(Circle((gx1, gy1), 3.6, facecolor=C_GLOW, edgecolor=C_GLOW_EDGE, lw=1.6, zorder=6))
ax.text(gx1, gy1, "GCaMP", fontsize=8.0, color="#5a4b00", ha="center", va="center", fontweight="bold", zorder=7)
ca_ion(gx1 + 3.0, gy1 + 3.0, r=1.1, z=7, sup=False)
ax.text(gx1, gy1 - 6.6, "bound to " + CA + " → fluoresces", fontsize=10,
        color=C_GLOW_EDGE, ha="center", va="top", fontweight="bold")

# --- numbered step legend (RIGHT half, in cytoplasm) -----------------------
lx, lw_box = 60, 39
ly_top, ly_bot = 31.5, 2.0
ax.add_patch(FancyBboxPatch((lx, ly_bot), lw_box, ly_top - ly_bot,
             boxstyle="round,pad=0.6,rounding_size=1.2",
             facecolor="white", edgecolor="#cfcfcf", lw=1.2, zorder=7))
ax.text(lx + 1.8, ly_top - 2.2, "What happens, step by step", fontsize=12,
        fontweight="bold", color=C_TXT, va="top", zorder=8)
steps = [
    "Depolarization opens the channel.",
    CA + " flows in, down its gradient.",
    CA + " binds the indicator (GCaMP).",
    "Indicator fluoresces; brightness\ntracks the cell's activity.",
]
sy = ly_top - 6.2
for i, txt in enumerate(steps, 1):
    ax.add_patch(Circle((lx + 3.0, sy), 1.7, facecolor=C_STEP, edgecolor="white", lw=1.2, zorder=8))
    ax.text(lx + 3.0, sy, str(i), fontsize=11, color="white", ha="center", va="center", fontweight="bold", zorder=9)
    ax.text(lx + 6.5, sy, txt, fontsize=10.8, color=C_TXT, ha="left", va="center", zorder=8, linespacing=1.05)
    sy -= 6.3

ax.set_title("How calcium enters a neuron and drives the indicator's fluorescence",
             fontsize=16, fontweight="bold", color=C_TXT, pad=10)

fig.tight_layout()
out = "/nadata/snlkt/home/itang/teaching/sleap-social-behavior-lab/data/assets/nb07_calcium_schematic.png"
fig.savefig(out, dpi=130, facecolor="white", bbox_inches="tight")
print("saved", out)
