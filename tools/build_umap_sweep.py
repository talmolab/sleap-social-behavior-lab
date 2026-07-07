"""Precompute a UMAP parameter sweep for NB05, so students SEE how n_neighbors / min_dist reshape
the behavioral map INSTANTLY — never calling UMAP live (it JIT-compiles ~30s on a cold molab kernel
and hangs the app). NB05 only SELECTS from this grid.

Output: data/umap_sweep.npz
    emb_grid   (5,5,N,2) f32   embeddings across n_neighbors x min_dist
    nn_values  (5,) int  |  md_values (5,) f32
    default_ij (2,) int          grid index of the course default
    default_labels (N,) int      HDBSCAN labels on the default embedding
    agg_lift   float             aggression enrichment of the best cluster (the HONEST NB05 target)
    overall_agg_rate float
    agg_label, ranks, sex, condition   (N,)  for coloring

Run in the course env:  uv run python tools/build_umap_sweep.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "course"))
import course_utils as cu  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "umap_sweep.npz")

NN_VALUES = [5, 15, 30, 50, 100]      # local detail -> global shape
MD_VALUES = [0.0, 0.1, 0.3, 0.5, 0.8] # tight blobs  -> spread out
D = cu.CLUSTER_DEFAULTS


def best_agg_lift(labels, agg, min_n=15):
    """Enrichment of the most aggression-dense cluster vs the overall rate (honest, not maximized)."""
    overall = agg.mean()
    best = 0.0
    for c in sorted(set(labels)):
        if c < 0:
            continue
        m = labels == c
        if m.sum() < min_n:
            continue
        lift = (agg[m].mean() / overall) if overall > 0 else 0.0
        best = max(best, lift)
    return float(best), float(overall)


def main():
    events = cu.load_events(os.path.join(ROOT, "data", "train_events.npz"))
    d = np.load(os.path.join(ROOT, "data", "train_derived.npz"), allow_pickle=True)
    scores = d["pca_scores"]                                  # precomputed PCA (matches NB04)
    res = cu.residualize(scores, list(D["drop_pcs"]))
    N = len(scores)
    agg = events["agg_label"].astype(int)

    emb_grid = np.zeros((len(NN_VALUES), len(MD_VALUES), N, 2), np.float32)
    for i, nn in enumerate(NN_VALUES):
        for j, md in enumerate(MD_VALUES):
            emb_grid[i, j] = cu.run_umap(res, n_neighbors=nn, min_dist=md, seed=D["seed"])
            print(f"  UMAP n_neighbors={nn:>3} min_dist={md:.1f}  done", flush=True)

    di, dj = NN_VALUES.index(D["n_neighbors"]), MD_VALUES.index(D["min_dist"])
    default_labels = cu.run_hdbscan(emb_grid[di, dj], min_cluster_size=D["min_cluster_size"])
    lift, overall = best_agg_lift(default_labels, agg)
    nclust = len([c for c in set(default_labels) if c >= 0])
    print(f"  default cell nn={D['n_neighbors']} md={D['min_dist']} mcs={D['min_cluster_size']}: "
          f"{nclust} clusters, best aggression lift = {lift:.2f}x (overall rate {overall:.2f})")

    np.savez_compressed(
        OUT, emb_grid=emb_grid, nn_values=np.array(NN_VALUES),
        md_values=np.array(MD_VALUES, np.float32), default_ij=np.array([di, dj]),
        default_labels=default_labels.astype(np.int32),
        agg_lift=np.float32(lift), overall_agg_rate=np.float32(overall),
        agg_label=agg.astype(np.int32), ranks=events["ranks"][:, 0].astype(np.int32),
        sex=d["sex"], condition=events["condition"].astype(str))
    print(f"wrote {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)  grid {emb_grid.shape}")


if __name__ == "__main__":
    main()
