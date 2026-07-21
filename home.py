import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # 🐭 SLEAP Social-Behavior Lab

        An interactive course in analyzing **animal social behavior from pose-tracking data** —
        from raw [SLEAP](https://sleap.ai) output all the way to a trained behavior classifier.

        **Work through the lessons in order.** Each one opens as a live notebook: drag a slider or
        edit an input and the analysis re-runs instantly. Your session is private to you.

        **Week 1 — Behavior (lessons 01–05).**

        | # | Lesson | What you build |
        |---|--------|----------------|
        | 1 | [Pose & Identity](/01) | Read the pose tensor `(frames, mice, nodes, xy)`; why one identity error corrupts everything downstream |
        | 2 | [The Body-Centered View](/02) | Center + rotate into a body frame; the 19 social features and why behavior is rotationally invariant |
        | 3 | [Behavior in Time](/03) | Distributions, wavelet rhythm, and who-leads-whom coordination |
        | 4 | [Collapsing to a Manifold](/04) | PCA, the UMAP objective, and a clustered map of behavioral syllables |
        | 5 | [How Analyses Mislead](/05) | The statistics that keep you honest: pseudoreplication, multiple comparisons, circular analysis, and CV leakage |

        **Week 2 — Dynamics, decoding, and the neural basis (lessons 06–08).** Week 2 opens on the
        transition grammar of behavior in time and a decoder tested across cohorts (lesson 06), then
        turns to the brain: the same computational moves — matrix factorization, manifolds, decoding —
        now on miniscope and two-photon neural recordings.

        | # | Lesson | What you build |
        |---|--------|----------------|
        | 6 | [Dynamics & Decoding](/06) | The transition grammar in time, then a decoder tested across cohorts |
        | 7 | [From Movie to Traces](/07) | Motion-correct a miniscope movie, then demix it into per-neuron footprints and calcium traces (CNMF) |
        | 8 | [Reading the Population](/08) | Spatial tuning, a firing sequence, and a population decoder that reads social state off the neurons |

        ---
        Mice are colored by dominance **rank**: 🔴 Dom &nbsp;&nbsp; 🔵 Mid &nbsp;&nbsp; 🟢 Sub.
        """
    )
    return


if __name__ == "__main__":
    app.run()
