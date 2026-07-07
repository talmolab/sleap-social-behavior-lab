"""Build data/readout_board.csv — the cumulative 'Readout Board' benchmarks every notebook shows
beside the student's own freshly-computed number (beat-the-benchmark, not a canned prop).

Gauge A 'size of the representation' falls through Phase 1; Gauge B 'held-out readiness' rises
through Phase 2. Values are computed from the REAL bundle so self-checks are never graded vs noise.

Instructors:  uv run python tools/build_readout_board.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "course"))
import course_utils as cu  # noqa: E402


def main():
    tr = cu.load_events(os.path.join(ROOT, "data", "train_events.npz"))
    he = cu.load_events(os.path.join(ROOT, "data", "heldout_events.npz"))
    dtr = cu.load_derived("train")
    dhe = cu.load_derived("heldout")
    sweep = np.load(os.path.join(ROOT, "data", "umap_sweep.npz"), allow_pickle=True)

    raw_dim = int(np.prod(tr["kp"].shape[1:]))                   # 130*3*15*2
    n_pc6 = 6
    cum6 = float(dtr["evr"][:n_pc6].sum())

    # honest decoders on the SHIPPED features (train CV + real held-out cage)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    Xtr, ytr = dtr["X"], tr["agg_label"].astype(int)
    Xhe, yhe = dhe["X"], he["agg_label"].astype(int)
    Xz, mu, sd = cu.standardize(Xtr)
    log_cv = cross_val_score(LogisticRegression(max_iter=1000), Xz, ytr, cv=5, scoring="roc_auc").mean()
    mlp = cu.make_mlp(); mlp.fit(Xtr, ytr)
    from sklearn.metrics import roc_auc_score
    mlp_heldout = roc_auc_score(yhe, mlp.predict_proba(Xhe)[:, 1])

    rows = [
        # gauge, notebook, stage, value, unit, note
        ("A", "NB01", "raw pose per event", raw_dim, "numbers", "130 frames x 3 mice x 15 nodes x 2"),
        ("A", "NB02", "allocentric features", 19, "numbers", "interpretable body-frame features"),
        ("A", "NB04", "principal components", n_pc6, "dims", f"{cum6:.0%} of variance kept"),
        ("A", "NB05", "behavioral map", 2, "dims", "UMAP coordinates -> syllables"),
        ("A", "NB05", "one syllable", 1, "label", "a single behavioral category"),
        ("B", "NB05", "best aggression cluster lift", round(float(sweep["agg_lift"]), 2), "x",
         "honest enrichment of the top cluster vs base rate"),
        ("B", "NB06", "features -> aggression (train CV)", round(float(log_cv), 3), "AUROC",
         "logistic, 5-fold"),
        ("B", "NB08", "held-out cage decode", round(float(mlp_heldout), 3), "AUROC",
         "MLP trained on 7 cages, tested on Cage 16 (unseen)"),
    ]
    df = pd.DataFrame(rows, columns=["gauge", "notebook", "stage", "value", "unit", "note"])
    out = os.path.join(ROOT, "data", "readout_board.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
