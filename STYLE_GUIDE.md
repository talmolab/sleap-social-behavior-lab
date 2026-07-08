# STYLE_GUIDE.md — Tone / Pedagogy Revision (all 14 notebooks)

This is the durable style layer that sits alongside `COURSE_DESIGN.md`. The 14 notebooks are
technically correct and the narrative structure and graphics are liked. The problem is **diction**:
the prose "reads a bit like a murder mystery novel." This course is a scientific **learning
exercise** for students who may be new to **both** behavioral neuroscience **and** Python.

You are **revising an existing, verified notebook**. Keep every working data-loading / computation /
self-check / no-live-UMAP cell **exactly as-is**. Rewrite the **prose, exercises, and figures** only.
Each notebook must still export cleanly headless.

---

## 1. Tone & diction (all 14)

- **Formal and plain.** Remove theatrical / mystery diction and dramatic beat-names. Banned framing
  includes: "the crisis", "the reversal", "the gut-punch", "the bet", "forbidden fruit", "the hunt",
  "graduation", "cash the check", "the threat", "the payoff", exclamatory hype, and ominous
  foreshadowing ("You are about to meet that problem in its rawest form."). State things directly.
- Keep a **light, professional framing**: the student studies social behavior and its neural basis.
  State that plainly, not cinematically.
- Drop email-style "FROM: Circuit Team → TO: Behavior Team" cold-open theatrics; open with the
  scientific purpose instead.

### The exact transformation wanted

> **TOO FLAMBOYANT:**
> "A neural experiment is coming: a laser that flips a hypothalamic switch, a probe in mPFC. But a
> manipulation is worthless without an objective readout of what each mouse actually does —
> hand-scoring won't survive review."

> **RIGHT REGISTER:**
> "Your job will be to study the role of the mPFC in social behavior. However, to understand behavior,
> we need a clearer understanding of what behavior IS. Today we will be using SLEAP, a software that
> ..."

---

## 2. Structure every section as: WHY → DEFINITIONS → METHOD

- **WHY** — open each notebook and each major section by explaining, in plain language, why it
  matters (the scientific purpose).
- **DEFINITIONS** — define the terms a newcomer needs *before* they can follow. Define jargon on
  first use, plainly, with a concrete example or analogy. Examples of terms to define: keypoint,
  behavior segmentation, what it means to "decode", principal component, clustering, Markov chain,
  allocentric, AUROC.
- **METHOD** — present the method. For **every function used**, state in plain words its **PURPOSE**,
  its **INPUTS**, and its **OUTPUTS**.
- Assume **no fluency** in neuroscience or coding.

---

## 3. Brain regions: ease off in Week 1 (NB01–07)

- In Week 1 dial **way** back on brain-region name-dropping and the "neural twin" comparisons. With
  no recordings yet, it reads as pretentious.
- **Remove** the neural-twin citation accordions and the running brain-twin device from Week-1 prose.
- A **single plain motivating sentence is the maximum**, e.g. "this is also how neuroscientists
  quantify behavior."
- **Keep genuine neural comparisons only where real neural data justifies them:** NB08 (the
  neural-demo decode check) and Week 2 (NB09–14, which use real imaging data). There, keep them but
  state them plainly.

---

## 4. Naming: remove "Hero" (all 14)

- **Delete the "Hero Event" branding everywhere.** Refer to the running example plainly as **"our
  example interaction"** or **"the example approach event."**
- Label the two interacting mice as **approacher** and **approachee** (or **subject** / **partner**).
- Select the running example by its **stable `event_key`** via `cu.event_index_by_key(ev, KEY)` —
  NEVER a raw integer index. The bundle is periodically rebuilt (adding a cohort changed N 1500→2499
  and re-ordered rows), so an integer index silently drifts to a different event; a key is stable.
  Whatever the current example is, describe it from the data, and never surface the word "Hero".

---

## 5. Color scheme (HARD constraint, all 14)

Mice are colored **only by rank**, identically across all 14 notebooks. Use `cu.RANK_HEX` /
`cu.RANK_RGB`. **Never** introduce any other color mapping for mice.

| Rank | Name | Hex | RGB |
|------|------|-----|-----|
| 1 | **Dom** | `#d62728` (red) | (214, 39, 40) |
| 2 | **Int / Mid** | `#1f77b4` (blue) | (31, 119, 180) |
| 3 | **Sub** | `#2ca02c` (green) | (44, 160, 44) |
| 0 | unknown | `#969696` (gray) | (150, 150, 150) |

Neural notebooks with no rank may use neutral palettes, but any mouse-rank coloring uses this scheme.

---

## 6. Lean heavily on GIFs (all 14)

The subject is behavior; students learn by **seeing** it. Use the rendered skeleton GIF helpers
generously — prefer a GIF over a paragraph when illustrating a behavior:

- `cu.event_gif_bytes(kp_event, ranks, contact_rel, cell, fps)` → GIF bytes for one event.
- `cu.grid_gif_bytes(events, ncols, cell, fps)` → tiled grid GIF (max 5×5) of several events.
- `cu.gif_img_html(gif, width)` → wraps GIF bytes in an animating `<img>` data-URI for `mo.md` /
  `mo.Html` (a plain marimo image widget freezes the first frame).

Use them to show the example event, and to show what a method's output **means**: exemplars of a
cluster, of a high-frequency-wavelet event, of a correlated pair, of a predicted-aggression event.
Neural notebooks: lean on the real movies / rasters they already load.

---

## 7. Coding exercises: gentle, fill-in-the-blank (all 14)

- **Never** hand the student a daunting from-scratch code block. Give a **scaffold** with a few
  **blanked** lines/variables to fill:

  ```python
  # TODO: compute each mouse's speed.
  # Replace ____ with np.diff(cen, axis=0) — the per-frame change in centroid position.
  # The line below already has the centroids `cen` (shape (T, 2)) and takes the norm for you.
  speed = np.linalg.norm(____, axis=1)   # (T-1,) px/frame
  ```

- Be explicit about **exactly which line(s)** to edit, what each surrounding line already does, and
  what the resulting **plot** should look like: "you should see two curves; the red Dom mouse's speed
  should spike near contact."
- The output of every exercise is a **plot** the student compares against a described / expected
  picture.
- Keep the **revealable solution** (accordion) and the **tolerance-band self-check**, but keep the
  student-edited surface **small and clearly marked**.

---

## 8. Readout Board Gauge A bug — fix in every notebook that has a board

**Symptom.** Gauge A used a plotly Indicator with `mode="number+delta"` and
`delta={"reference": 11700}`. Because Gauge A's own value is smaller than 11,700 (e.g. 19 features in
NB02), the delta rendered as a confusing **negative** — `▼ -11,681` (the "-11k" that was reported) —
and the extra delta line crowded / overlapped the two-line title.

**Fix.**
- Use **`mode="number"`** (drop the delta). Show **only** the correct **positive** representation
  size for **this** notebook.
- Keep the **"was 11,700 raw"** context in the **title text**, not as a delta.
- Give the figure **enough height and top margin** that the two-line titles do not overlap
  (`height=230`, `margin.t≈95`).
- **No negative deltas anywhere.**

**Gauge A value per notebook:** NB01 = 11,700 · NB02 = 19 · NB03 = 19 · NB04 ≈ 6 · NB05 = 2 (map) /
1 (syllable) · later notebooks show their own representation size (NB08 = 1 decision).

The exact corrected `readout_fig` to paste is in `REVISION_CONTRACT` (Task 2) and in the contract
returned by the PREP agent.

Notebooks whose grep hit `number+delta`:

| Notebook | Line | Gauge | Verdict |
|----------|------|-------|---------|
| `02_body_eye_view.py` | 118 | A (size) | **BUG** — value 19, ref 11700 → **−11,681**. Fix to `mode="number"`. |
| `01_raw_signal.py` | 89 | A (size) | Pointless zero-delta (ref == value == 11700). Fix to `mode="number"` (baseline; no "was X raw"). |
| `06_reading_the_map.py` | 164 | B (rising AUROC) | delta ref == value → ≈0.000. Drop the delta; use `gauge+number`. |
| `08_decoder_graduates.py` | 156, 884 | B (rising AUC) | delta ref == value (0.86) → ≈0.000. Gauge A already `mode="number"`. Drop the delta on B. |

---

## 9. Preserve (do not break)

This is a prose / exercise / figure revision. Do **not** touch:

- All working code, data loading, and computed self-check **pin values**.
- The **no-live-UMAP** rule (UMAP is precomputed; live UMAP crashes the kernel).
- Valid **marimo** structure: one-assignment-per-name, `_`-prefixed locals, last-expression render,
  `hide_code` on prose cells, sliders placed adjacent to their output via `mo.vstack`.

**Re-verify each notebook exports headless after revising.**

---
---

# Round 3 (consolidation + depth)

Everything in Sections 1–9 above still applies (plain formal diction; WHY → DEFINITIONS → METHOD;
no "Hero" branding; mice colored by rank via `cu.RANK_HEX`; fill-in-the-blank exercises; define all
jargon; `hide_code` prose; sliders adjacent via `mo.vstack`; **no live UMAP on the real data**;
molab-safety). Layer these directives on top.

The course is being **consolidated 14 → 10 notebooks** (5 behavior + 5 neural) and substantially
**deepened**: each notebook targets **~4 hours** of genuine work, not ~10 minutes. Do not pad — add
real EDA, worked steps, deeper "what and why" prose, and the new methods below.

## R3.1 Remove the readout board / counter entirely

Delete the two-gauge "size of representation / readiness" panels (`readout_fig`, the Gauge-A/B
Indicators, `data/readout_board.csv` usage) from **every** notebook. Replace with a
**question-driven throughline**: each notebook **opens** by briefly recalling the question the
previous notebook answered and stating the question **this** notebook asks, and **closes** by stating
the answer reached and the next question it raises. A logical chain of scientific questions, not a
numeric tracker. (Section 8's Gauge-A bug note is now moot — the board is gone.)

## R3.2 Behavior and its neural basis, no hierarchy

We study social behavior and its neural basis; neither is treated as primary or as coming "first."
Do not self-label the student as a "behavioral neuroscientist" as an identity, and do not claim we
study behavior first and neural data second. Remove any wording that treats the behavioral work as
aspiring-to or validated-by "real" neuroscience ("with the same rigor as neuroscientists", "this is
how neuroscientists do it" used as external validation, etc.). Write confidently in the **first person
plural** ("we quantify…", "we ask…").

## R3.3 Much longer, deeper notebooks (~4 h each)

Expand every notebook: more EDA, more worked steps, deeper prose, and the extra methods below.

## R3.4 Lean heavily on GIFs in every behavior notebook

Students learn by **seeing** behavior. Use `cu.event_gif_bytes` / `cu.grid_gif_bytes` /
`cu.gif_img_html` generously and specifically: exemplars of each cluster, high- vs low-frequency
events, a correlated (interacting) pair vs an uncorrelated pair, each behavioral state, and decoder
correct-vs-mistaken predictions. Candidate event indices are pinned in the PREP contract. Neural
notebooks: lean on the real microscope movies / rasters.

## R3.5 Richer, interactive, seaborn-style plots

Stop defaulting to bar charts. Show **individual data points**, interactive (hover). Use the new
seaborn-style plotly helpers (added to `course_utils.py`, mirrored generic ones in `neural_utils.py`):
`strip_points_fig`, `violin_points_fig`, `box_points_fig`, `kde2d_fig`, `ecdf_fig`,
`umap_colored_by_feature_fig`. Every distribution comparison shows the raw points, not just a summary
bar. House style: `template="plotly_white"`, tight margins, clear titles, colorbars; mice colored
Dom=red/Int=blue/Sub=green (the helpers auto-map rank names to `cu.RANK_HEX`).

## R3.6 UMAP: show the objective function (notebook 4)

Explain and **demonstrate** what UMAP optimizes: high-D fuzzy neighbor memberships vs low-D
memberships, the cross-entropy objective, attractive (neighbors pull) and repulsive (non-neighbors
push) forces. Use `cu.umap_objective_toy(...)` — a small, fast, **pure-numpy** interactive toy
(~90 points) that optimizes a 2-D layout live and returns snapshots + the membership/force curves.
This does **not** violate no-live-UMAP: the real 2499-point map still comes from the precomputed
sweep (`cu.load_umap_sweep()`); only the tiny teaching toy runs live (`< 1 s`, no umap-learn).

## R3.7 Cluster in feature space so the map axes MEAN something

Do not present UMAP as a black box. Color the map by each of the 19 features
(`cu.umap_colored_by_feature_fig`) so students see which features vary across the map; show a
per-cluster **feature profile** (which features are high/low per cluster) with seaborn-style displays;
render exemplar GIFs per cluster; make the cluster figures resemble a real analysis (colored by
cluster, clear density, feature overlays).

## R3.8 Gradual Python-skill ramp across the exercises

Build coding skill progressively across the 10 notebooks **and** within each: array indexing/slicing
(`kp[frame, mouse, node]`) → array arithmetic + boolean masks → writing a small function → a loop then
its vectorized form → calling a library (numpy stats, sklearn). **State which Python skill each
exercise practices.**

## R3.9 Very verbose code annotation in exercises

For every line the student edits: a comment stating exactly what to change and **why it matters
scientifically**. Guide them to the edit, describe the expected plot, explain the reasoning.
Over-comment rather than under-comment.

## R3.10 New 2-cohort data + the real sex finding

The dataset now has **two food-deprivation cohorts** (do **NOT** name the project). Cage identity is
**cohort-unique** and `cu.load_derived('train')` has a `'cohort'` field and a cohort-unique `'cage'`
field. Notebook 4's statistics section teaches a **real, replicated** finding
(`heading_alignment` sex difference survives cage-level testing) with a **positive control** and a
**negative control** (`appr_body_len`, body size / dimorphism, fails → the pseudoreplication lesson).
Use the pinned values exactly.

---

## R3 Pinned values (re-verified against committed bundle `4d79758`, cohort names deliberately not the project)

- **train** N = **2499**, aggression base rate **0.320**; **heldout** N = **780** (`cam16`, single
  cohort, **all female**), base rate **0.385**.
- **Two cohorts** (identified only by their date tags `12192025` and `20260222`):
  counts **1282 + 1217**; conditions **pre=824 / dep=896 / post=779**.
- **Cage identity is cohort-unique** = `cohort_index*100 + cam`. **Train cages = 9–15 and 109–115**
  (7 + 7 = **14 cages**, balanced **7 M / 7 F**). ⚠ **Correction:** the draft directive said
  "9–**16**"; the actual train cages are **9–15**. Cage **16** is the **held-out** `cam16` (cohort
  `20260222`, all female). Events: **1181 M / 1318 F**. `load_derived('train')` keys include
  `cohort`, `cage`, `sex`.
- **PCA:** 6-PC cumulative EVR = **0.714** (first-6 EVR = `[.178, .168, .129, .092, .082, .064]`).
  The shipped 10-component PCA caps at **0.889**, so reaching 90% variance needs **11 PCs**
  (refit a full-rank `PCA(n_components=19)` on the standardized X to show this; cum@10 = 0.889,
  cum@11 = 0.915).
- **UMAP sweep:** `agg_lift` = **1.19×** (base 0.320); **4 clusters** at the default cell
  (`emb_grid[tuple(default_ij)]`, `default_ij = [1,0]`). Canonical clusters = the sweep
  `default_labels`. Cluster sizes / aggression fractions: c0 = 53/0.377, c1 = 149/0.322,
  c2 = 1423/0.297, **c3 = 730/0.381 (purest / largest aggression-enriched)**.
- **Aggression decoder (logistic):** train 5-fold CV AUROC = **0.851 ± 0.006**; `cam16` held-out
  AUROC = **0.873**; **leave-one-cohort-out** AUROC = **0.825** (test cohort `20260222`) /
  **0.859** (test cohort `12192025`) — use LOCO as the honest cross-dataset generalization test
  (notebook 5).
- **THE SEX FINDING (real, survives cage-level → notebook 4):** feature `heading_alignment`
  (Cohen d **0.23**, **M < F**: median M ≈ −0.14, F ≈ +0.05). Event-level Mann–Whitney
  **p = 6.5e-9**; **cage-level permutation** (14 cohort-unique cages, shuffle sex, statistic =
  |mean of per-cage feature means, M − F|) **p_emp = 0.0094 → SURVIVES**; replicates in **both**
  cohorts (event MWU p = 3.7e-9 and p = 0.015).
  **NEGATIVE CONTROL:** `appr_body_len` (body size / dimorphism) event p = **5.4e-22** BUT
  cage-level permutation p = **0.078 → does NOT survive** (the pseudoreplication lesson, now with a
  positive **and** a negative control side by side). Aggression-rate-by-sex is **null**
  (event χ² p = **0.60**, cage-level null by every test).
  ⚠ **Do not use `cu.permutation_test` for this feature-by-sex test** — that helper tests a
  categorical composition (e.g. is a cluster sex-enriched) and returns a different p (0.0006). The
  cage-level number authors should reproduce (0.0094) comes from permuting sex across the 14 cages
  after collapsing each cage to its mean `heading_alignment` (see the PREP contract for the exact
  snippet).
- **FOOD-DEP effect (survives → notebook 4):** feature `bystander_dist_mean`; event Mann–Whitney
  **p = 6.8e-6** (median **359 → 469 px** under deprivation); **cage-level paired Wilcoxon**
  (14 cages, pre vs dep means) **p = 0.0052** (mean **+48 px** under dep), replicates both cohorts.
  Aggression rate pre vs dep is **null** (event p = 0.29).

## R3 New helpers added to `course_utils.py` (and generic ones mirrored in `neural_utils.py`)

Seaborn-style interactive plotly (plotly + numpy + scipy only, `plotly_white`, hover, rank auto-color
where mice are grouped): `strip_points_fig`, `violin_points_fig`, `box_points_fig`, `kde2d_fig`,
`ecdf_fig`, `umap_colored_by_feature_fig`. UMAP-objective teaching toy (pure numpy, `< 1 s`, no
umap-learn): `umap_objective_toy` (+ `umap_objective_layout_fig`). Exact signatures and one-line
usage examples are in the PREP contract.
