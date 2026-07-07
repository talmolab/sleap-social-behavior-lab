"""Build data/neural_demo.npz — a small SYNTHETIC population raster for NB08's neural payoff.

The course's finale ("cash the neural check") runs the STUDENT'S OWN decoding pipeline on a neural
population, showing the behavior decoder and a neural decoder are literally the same math. We ship a
synthetic raster (no real ephys, no heavy deps): N trials x K neurons of spike counts, driven by a
hidden binary state (analogous to a behavioral state / stimulus). Some neurons are state-selective,
most are noise — exactly the regime where population decoding beats any single neuron.

  X_neural   (n_trials, n_neurons) int16   trial spike counts
  y          (n_trials,) int8              hidden state (0/1) = the thing to decode
  is_tuned   (n_neurons,) int8             which neurons are state-selective (for teaching)
  emb2d      (n_trials, 2) f32             PCA of the raster (CEBRA-style epilogue coordinates)

Deterministic (fixed seed). Instructors:  uv run python tools/build_neural_demo.py
"""
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "neural_demo.npz")
N_TRIALS, N_NEURONS, N_TUNED = 800, 60, 18
SEED = 7


def main():
    rng = np.random.RandomState(SEED)
    y = rng.randint(0, 2, N_TRIALS)                              # hidden state per trial
    base = rng.uniform(1.0, 4.0, N_NEURONS)                      # baseline rate per neuron
    tuned = np.zeros(N_NEURONS, np.int8)
    tuned[rng.choice(N_NEURONS, N_TUNED, replace=False)] = 1
    # state-selective gain: tuned neurons up (or down) in state 1; untuned unaffected
    sign = rng.choice([-1, 1], N_NEURONS)
    gain = 1.0 + tuned * sign * rng.uniform(0.4, 1.2, N_NEURONS)  # multiplicative when state==1
    rates = base[None, :] * np.where(y[:, None] == 1, gain[None, :], 1.0)
    X = rng.poisson(np.clip(rates, 0.05, None)).astype(np.int16)  # (trials, neurons)

    # CEBRA-style 2-D view = PCA of z-scored raster (precomputed so NB08 has no live fit)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    from sklearn.decomposition import PCA
    emb2d = PCA(n_components=2, random_state=0).fit_transform(Xz).astype(np.float32)

    np.savez_compressed(OUT, X_neural=X, y=y.astype(np.int8), is_tuned=tuned, emb2d=emb2d)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    auc = cross_val_score(LogisticRegression(max_iter=1000), Xz, y, cv=5, scoring="roc_auc").mean()
    print(f"wrote {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)")
    print(f"  {N_TRIALS} trials x {N_NEURONS} neurons ({N_TUNED} tuned); "
          f"population decode AUC = {auc:.3f} (single best neuron is much weaker)")


if __name__ == "__main__":
    main()
