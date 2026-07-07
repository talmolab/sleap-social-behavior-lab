"""Precompute a UMAP parameter sweep for notebook 03, so students SEE how n_neighbors / min_dist
reshape the map instantly — without waiting ~30 s for numba to JIT-compile UMAP on a cloud kernel,
and without a slider drag re-running UMAP over and over (which hangs the web app).

Output: data/umap_sweep.npz
    emb_grid   (n_nn, n_md, N, 2) float32   UMAP embeddings across the parameter grid
    nn_values  (n_nn,) int                  n_neighbors values swept
    md_values  (n_md,) float                min_dist values swept
    default_ij (2,) int                      grid index of the course default (15, 0.0)
    default_labels (N,) int                  HDBSCAN labels on the default embedding
    agg_label  (N,) int   ranks (N,) int   condition (N,) str    (for coloring)

Run in the course env (umap is a default dep):
    uv run python tools/build_umap_sweep.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "course"))
import course_utils as cu  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "umap_sweep.npz")

NN_VALUES = [5, 15, 50]          # local detail  ->  global shape
MD_VALUES = [0.0, 0.3, 0.8]      # tight blobs   ->  spread out
D = cu.CLUSTER_DEFAULTS


def main():
    events = cu.load_events(os.path.join(ROOT, "data", "train_events.npz"))
    X = cu.features_batch(events["kp"].astype("float32"))
    Xz, _, _ = cu.standardize(X)
    scores, _, _ = cu.pca_scores(Xz, D["pca_k"])
    res = cu.residualize(scores, list(D["drop_pcs"]))
    N = len(X)

    emb_grid = np.zeros((len(NN_VALUES), len(MD_VALUES), N, 2), np.float32)
    for i, nn in enumerate(NN_VALUES):
        for j, md in enumerate(MD_VALUES):
            emb_grid[i, j] = cu.run_umap(res, n_neighbors=nn, min_dist=md, seed=D["seed"])
            print(f"  UMAP n_neighbors={nn:>2} min_dist={md:.1f}  done")

    di = NN_VALUES.index(D["n_neighbors"])
    dj = MD_VALUES.index(D["min_dist"])
    default_labels = cu.run_hdbscan(emb_grid[di, dj], min_cluster_size=D["min_cluster_size"])

    np.savez_compressed(
        OUT,
        emb_grid=emb_grid,
        nn_values=np.array(NN_VALUES),
        md_values=np.array(MD_VALUES, np.float32),
        default_ij=np.array([di, dj]),
        default_labels=default_labels.astype(np.int32),
        agg_label=events["agg_label"].astype(np.int32),
        ranks=events["ranks"][:, 0].astype(np.int32),
        condition=events["condition"].astype(str),
    )
    print(f"wrote {OUT}  ({os.path.getsize(OUT) / 1024:.0f} KB)  grid {emb_grid.shape}")


if __name__ == "__main__":
    main()
