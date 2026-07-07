# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "marimo>=0.9",
#     "numpy>=1.24,<2.1",
#     "scipy>=1.11",
#     "pandas>=2.0",
#     "scikit-learn>=1.3",
#     "numba>=0.59",
#     "umap-learn>=0.5.6",
#     "hdbscan>=0.8.36",
#     "plotly>=5.20",
#     "imageio>=2.34",
#     "pillow>=10.0",
# ]
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 04 · Which clusters carry *rank*?

        A behavioral cluster is interesting if its composition depends on biology. We test two
        questions per cluster:

        1. **Rank** — are some directed dominance dyads (e.g. Dom→Sub) over-represented? If
           "fight-initiation" is a real cluster, dominants should do the approaching.
        2. **Condition** — is the cluster more common in `pre`, `dep`(rivation), or `post`?

        Both are **contingency-table tests**. For a cluster vs. the rest,

        $$\chi^2=\sum_{c}\frac{(O_c-E_c)^2}{E_c},\qquad E_c=\frac{(\text{row}) (\text{col})}{N},
        \qquad r_c=\frac{O_c-E_c}{\sqrt{E_c}}$$

        The standardized residual $r_c$ says *which* category drives a hit. Because we test many
        clusters, we **Bonferroni**-correct: $p_{\text{adj}}=\min(1,\,m\,p)$ for $m$ clusters.
        """
    )
    return


@app.cell
def _():
    import os
    import sys
    import numpy as np
    import plotly.graph_objects as go

    import urllib.request

    _RAW = os.environ.get(
        "COURSE_REPO_RAW",
        "https://raw.githubusercontent.com/Elmaestrotango/sleap-social-behavior-lab/main",
    )

    def _find_root():
        p = os.getcwd()
        for _ in range(6):
            if os.path.isdir(os.path.join(p, "course")) and os.path.isdir(os.path.join(p, "data")):
                return p
            p = os.path.dirname(p)
        return None

    # On a bare cloud notebook (e.g. molab) there is no repo checkout: fetch course_utils.py, then
    # let cu.bootstrap() download the bundled data on first use.
    ROOT = _find_root() or os.getcwd()
    _cu = os.path.join(ROOT, "course", "course_utils.py")
    if not os.path.exists(_cu):
        os.makedirs(os.path.dirname(_cu), exist_ok=True)
        urllib.request.urlretrieve(_RAW + "/course/course_utils.py", _cu)
    sys.path.insert(0, os.path.join(ROOT, "course"))
    import course_utils as cu

    ROOT, DATA, SCRATCH = cu.bootstrap()

    events = cu.load_events(os.path.join(ROOT, "data", "train_events.npz"))
    X = cu.features_batch(events["kp"].astype("float32"))
    cl = cu.cluster_pipeline(X, **cu.CLUSTER_DEFAULTS)     # canonical defaults (same as nb03)
    labels = cl["labels"]
    return cu, events, go, labels, np


@app.cell
def _(cu, events, labels):
    rank_stats = cu.rank_dyad_enrichment(labels, events["ranks"][:, 0], events["ranks"][:, 1])
    cond_stats = cu.condition_enrichment(labels, events["condition"])
    best_cluster = rank_stats[0]["cluster"] if rank_stats else 0
    return best_cluster, cond_stats, rank_stats


@app.cell
def _(events, labels, mo, rank_stats):
    def _agg_frac(c):
        m = labels == c
        return events["agg_label"][m].mean() if m.any() else 0.0
    _rows = "\n".join(
        f"| {'**C'+str(r['cluster'])+'**' if r['sig'] else 'C'+str(r['cluster'])} | {r['n']} | "
        f"{_agg_frac(r['cluster']):.0%} | {r['enriched_dyad']} | {r['chi2']:.1f} | {r['p']:.1e} | "
        f"{r['p_bonf']:.1e} | {'✅' if r['sig'] else ''} |"
        for r in rank_stats)
    mo.md(
        "### Rank-dyad enrichment (per cluster, vs. rest)\n\n"
        "| cluster | n | % aggression | most-enriched dyad | χ² | p | p (Bonferroni) | sig |\n"
        "|---|---|---|---|---|---|---|---|\n" + _rows +
        "\n\n*Bold + ✅ = survives Bonferroni. The significant cluster is typically the "
        "high-aggression one, and its enriched dyad is Dom-initiated.*"
    )
    return


@app.cell
def _(best_cluster, labels, mo):
    cluster_sel = mo.ui.dropdown(options=[str(c) for c in sorted(c for c in set(labels) if c >= 0)],
                                 value=str(best_cluster), label="inspect cluster")
    cluster_sel
    return (cluster_sel,)


@app.cell
def _(cluster_sel, cu, events, go, labels, np, rank_stats):
    _c = int(cluster_sel.value)
    _di = np.array([cu._dyad_index(a, b) for a, b in zip(events["ranks"][:, 0], events["ranks"][:, 1])])
    _valid = _di >= 0
    _marg = np.bincount(_di[_valid], minlength=6) / max(1, _valid.sum())
    _in = _valid & (labels == _c)
    _clu = np.bincount(_di[_in], minlength=6) / max(1, _in.sum())
    _fig = go.Figure()
    _fig.add_bar(x=cu.DYAD_LABELS, y=_marg, name="all events (marginal)", marker_color="#bab0ac")
    _fig.add_bar(x=cu.DYAD_LABELS, y=_clu, name=f"cluster C{_c}", marker_color="#d62728")
    _rec = next((r for r in rank_stats if r["cluster"] == _c), None)
    _t = f"C{_c} rank-dyad composition" + (f" — enriched {_rec['enriched_dyad']}, "
                                           f"p_bonf={_rec['p_bonf']:.1e}" if _rec else "")
    _fig.update_layout(barmode="group", template="plotly_white", height=360, title=_t,
                       yaxis_title="fraction of directed events", margin=dict(l=10, r=10, t=40, b=10))
    _fig
    return


@app.cell
def _(cluster_sel, cu, events, labels, mo, np):
    _c = int(cluster_sel.value)
    _idx = np.where(labels == _c)[0]
    _rng = np.random.RandomState(0)
    _pick = _rng.choice(_idx, size=min(9, len(_idx)), replace=False)
    _evs = [(events["kp"][i].astype("float32"), events["ranks"][i], int(events["contact_rel"][i]))
            for i in _pick]
    _gif = cu.grid_gif_bytes(_evs, ncols=3, cell=150, fps=16)
    mo.md(
        f"### Exemplars of C{_c}\n\n{cu.gif_img_html(_gif, width=470)}\n\n"
        f"*A 3×3 sample (rank-colored: 🔴 Dom 🔵 Mid 🟢 Sub). Does the behavior look coherent?*"
    )
    return


@app.cell
def _(cond_stats, mo):
    _rows = "\n".join(
        f"| {'**C'+str(r['cluster'])+'**' if r['sig'] else 'C'+str(r['cluster'])} | {r['n']} | "
        f"{r['enriched']} | {r['fracs']['pre']:.2f} / {r['fracs']['dep']:.2f} / {r['fracs']['post']:.2f} | "
        f"{r['p']:.1e} | {r['p_bonf']:.1e} | {'✅' if r['sig'] else ''} |"
        for r in cond_stats)
    mo.md(
        "### Condition enrichment (pre / dep / post)\n\n"
        "| cluster | n | enriched | pre/dep/post fracs | p | p (Bonferroni) | sig |\n"
        "|---|---|---|---|---|---|---|\n" + _rows +
        "\n\n> ⚠️ **Large-n caution:** the giant catch-all cluster can be 'significant' for condition "
        "with fractions like 0.36/0.31/0.33 — a tiny effect made significant only by thousands of "
        "points. Always read the **effect size** (the fractions), not just the p-value. The "
        "aggression cluster's tilt toward `dep` is the biologically meaningful one."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Takeaway
        Unsupervised clustering + a simple contingency test already recovers a real result: a
        coherent **aggressive-approach cluster that dominants initiate** (Dom-enriched dyad), and it
        leans toward the deprivation phase. But clusters are fuzzy and aggression is smeared across a
        few of them — so to build a *reliable detector* we switch to **supervised learning**: label
        examples (next) and train a classifier (after).

        **Next → `05_label_exemplars.py`.**
        """
    )
    return


if __name__ == "__main__":
    app.run()
