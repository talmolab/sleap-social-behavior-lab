# COURSE BIBLE & NOTEBOOK BLUEPRINTS
## Reading the Social Brain: Build the Decoder for the Rig

*Definitive design document — single source of truth for notebook authors and the data-prep step. Version 1.0. Built by grafting the winning "manifold" spine onto the "circuit/decoder" mission frame, with every judge fix applied.*

---

# 1. The Narrative

### Theme
**Reading the Social Brain — Build the Decoder for the Rig**

### Logline
A systems-neuroscience lab has the causal tools to probe the circuits of social dominance — dmPFC synaptic strength sets who wins the tube test, VMHvl gates aggression — but the recording rig isn't ready. Your one-week job is to turn raw SLEAP pose from three mice living together into the objective behavioral **decoder** that experiment cannot be interpreted without. Along the way you discover the twist the week was quietly built on: **every method you use to read behavior is, step for step, a method neuroscientists already use to read the brain.**

### The throughline (one story spine)

You are the new **behavior team** in the Social Circuits Lab. A neural experiment is coming — but a laser that flips a hypothalamic switch, or a probe in mPFC, is scientifically worthless without an automated, honest readout of what each mouse actually *does*. Hand-scoring won't survive review. So the week is a single mission with a hard constraint and a deferred payoff: **build the readout, and prove it works on a cage the lab has never let you see** — Camera 16, "the animal on the rig," sealed until the final notebook. Every notebook ships one piece the circuit team needs before they trust a single trial, and the whole build is disciplined by one falsifiable bet planted on day one: *will a decoder trained on seven cages survive a brand-new one — the only test a circuit experiment would believe?*

The mission has **two phases the student can feel**. **Phase 1 (Discover)** collapses an impossibly high-dimensional signal onto its low-dimensional shape: SLEAP pose (11,700 numbers per event) becomes 19 body-frame features, then ~6 principal components, then a 2-D behavioral map carved into syllables — exactly the collapse neuroscientists find when a population of neurons turns out to live on a low-dimensional manifold. **Phase 2 (Operationalize)** puts that representation to work: test what it encodes, model how behavior moves through it in time, define ground truth by hand, and train and honestly validate the decoder. The geometric collapse is the *how*; the decoder for the rig is the *why*; and the recurring **Neural Twin** reveal is the payoff that makes the two into one — the mathematics of reading behavior *is* the mathematics of reading brains. The week closes by cashing that check literally: the student runs their exact decoding pipeline on a small population raster and hands the validated readout back to the waiting rig.

### Recurring narrative devices (drop into EVERY notebook)

1. **The Lab-Meeting Briefing** *(opening cell, `hide_code=True`)* — a short "**FROM: Circuit Team → TO: Behavior Team**" memo stating (a) the one deliverable this notebook ships, (b) the exact circuit question it unblocks, and (c) **the day's lab-meeting question** (the real hypothesis on the table). Single controlling metaphor: the **lab**. No mountaineering/expedition vocabulary anywhere.

2. **The Neuroscience Connection — load-bearing, not skippable.** One or two *integrated sentences in the main rendered prose* at the notebook's open and close tie the day's method to the brain (this satisfies HARD requirement #1 — the neuro throughline is never quarantined). A styled **"Deeper: the paper & where the analogy stops"** `mo.accordion` then carries the full citation, the one-line *shared-mathematics* statement, a **species/preparation tag**, and a mandatory **"where the analogy stops"** clause. The accordion is enrichment; the plot beat lives in the prose.

3. **The Hero Event — Event #742, Cage 15 (male).** One real aggression approach, followed end to end and re-rendered by whatever method the notebook introduces: skeleton GIF → body-frame geometry → wavelet spectrogram + lead-lag trace → a point on the PCA/UMAP map → its syllable → the student's own label → a confident decoder hit. **Discipline against category errors:** the single event is used only for single-event beats; for population/temporal beats (grammar, activity clock) we zoom to a *Cage-15 session*, and NB07 stages an explicit "baton hand-off" that locates Event #742 on the day's activity clock before widening out.

4. **The Readout Board** *(cumulative tracker, top + bottom of every notebook)* — an honest two-gauge status panel. **Gauge A, "size of the representation,"** falls through Phase 1 (11,700 → 19 → ~6 → 2-D map → 1 syllable) with a one-line honesty note that these are *different kinds* of reduction, not one magic number. **Gauge B, "held-out readiness,"** rises through Phase 2. Implemented as a committed **`readout_board.csv`** of *benchmark* values each notebook displays **beside the student's own freshly-computed number** (beat-the-benchmark, not a canned prop); degrades gracefully if scratch state is missing.

5. **The Sealed Cage 16** — a visible-but-**redacted** panel (event count shown, skeletons greyed, labels blacked out) with a **"notebooks until unlock"** counter. It plants the day-one bet and pays it off when it opens in NB08. Forbidden fruit must be on the tree, not merely named.

6. **The Neuromatch exercise scaffold** *(identical every notebook)* — a **Toolbox** cell naming every helper with its exact inputs/outputs; a **Hypothesis banner** (one pre-registered-style line); a **TODO stub**; a revealable **`mo.accordion` solution**; a colored **pass/fail self-check** with a **tolerance band**. Heavy methods use a **two-tier** design: a Tier-1 hand-computation the student genuinely writes, with the equation-heavy internals supplied as a named black-box helper.

7. **"What we threw away / how it breaks"** *(closing box)* — names the information this stage discards, gives 2–3 concrete failure modes *on this dataset*, and poses one open-ended **"how would you analyze this?"** prompt (HARD requirement #5).

8. **"What we ship next"** *(closer)* — a plain-language summary plus a one-line hook to the next deliverable, advancing the mission.

### Behavior ↔ Brain method-twin table

References are checked and corrected (see fixes noted). "Where the analogy stops" is mandatory in-notebook.

| Notebook method | Neural twin | Real, correctly-attributed reference |
|---|---|---|
| SLEAP pose **estimation** + multi-animal **identity tracking**; swaps when mice touch | Spike **detection** (estimation) + **spike sorting / data association** (identity over time); touch-swaps ≈ overlapping-spike "collisions" | Pereira et al. 2022 *Nat. Methods* (SLEAP — this course's own lab lineage); Mathis et al. 2018 *Nat. Neurosci.* (DeepLabCut); Lewicki 1998 *Network*; Pachitariu et al. 2016 (Kilosort). *Stops at:* estimation≈detection, tracking≈sorting — don't collapse them. |
| **Body-frame transform** (center on TTI, rotate heading→+y) | **Egocentric** (self-centered) coding, and the egocentric↔allocentric **transformation** machinery; place/grid/HD cells are the *allocentric endpoint* the brain converts to | O'Keefe & Dostrovsky 1971 *Brain Res.* (place); Hafting et al. 2005 *Nature* (grid); Taube, Muller & Ranck 1990 *J. Neurosci.* (HD); **Alexander et al. 2020 *Science Advances* 6:eaaz2322** (RSC egocentric boundary vectors — *corrected journal*); social: Danjo et al. 2018 *Science* (**rats**), Omer et al. 2018 *Science* (**bats**). *Stops at:* the transform is egocentric; the code name "allocentric" is a field misnomer we teach openly. |
| 19 hand-designed features (facing **cosine**, closing speed) | **Tuning curves** / feature selectivity; a facing-cosine feature ≈ **cosine directional tuning** | Hubel & Wiesel 1962 (V1 orientation); Georgopoulos et al. 1986 *Science* (motor-cortex cosine tuning). *Stops at:* our detectors are *designed*; tuning curves are *learned/measured*. |
| **Morlet wavelet** time-frequency of speed/tail motion | Time-frequency of **LFP/EEG oscillations** (same math, different signal) | Torrence & Compo 1998; Cohen 2014 *Analyzing Neural Time Series Data*; Buzsáki & Draguhn 2004 *Science*; Berman et al. 2014 *J. R. Soc. Interface* (wavelet postural spectrogram, MotionMapper). *Stops at:* a wagging tail is not an LFP — only the transform transfers; matching Hz ≠ matching mechanism. |
| **Cross-correlation lead-lag** (primary) / pairwise **Granger** (stretch) between two mice's motion | **Directed functional connectivity** (who leads whom) | Granger 1969 *Econometrica*; Seth, Barrett & Barnett 2015 *J. Neurosci.*; **Bressler & Seth 2011 *NeuroImage* 58:323–329** (*corrected journal*); social motivator: Kingsbury et al. 2019 *Cell* (dmPFC inter-brain coupling predicts dominance — **GLM/correlation, not Granger**; flagged as looser analogy). *Stops at:* prediction ≠ causation; a shared third mouse can fake it. |
| **PCA** of 19 features (+ **residualize**) | **Neural population manifold** / low-D population geometry; demixing a nuisance axis | Stephens et al. 2008 *PLoS Comput. Biol.* (eigenworms); Cunningham & Yu 2014 *Nat. Neurosci.*; Gallego et al. 2017 *Neuron*; Kobak et al. 2016 *eLife* (dPCA). *Stops at:* behavior manifold ≈ neural manifold is shared math, not biological identity. |
| **UMAP (precomputed) + HDBSCAN** → behavioral syllables | Unsupervised behavioral-state discovery via nonlinear embedding + density clustering | McInnes et al. 2018 (UMAP); **Berman et al. 2014** (MotionMapper, t-SNE — the direct *method* ancestor); **Hsu & Yttri 2021 *Nat. Commun.*** (B-SOiD, UMAP+cluster); Campello et al. 2013 (HDBSCAN); Wiltschko et al. 2015 *Neuron* (the *concept* "syllable" only — **not** the method here). *Stops at:* UMAP distances/cluster sizes are not metric. |
| First-order **observed** Markov transition matrix + stationary dist + entropy vs shuffle | **Hidden**-state sequence models (HMM / AR-HMM) — taught via the honest *observed-vs-hidden* distinction | Wiltschko et al. 2015 *Neuron* (AR-HMM, MoSeq — belongs *here*); Markowitz et al. 2018 *Cell* (striatum reads syllable transitions); Jones et al. 2007 *PNAS*; Mazzucato et al. 2015 *J. Neurosci.* *Stops at:* we label our states; the brain's are latent and must be inferred — that inference is the HMM. |
| **MLP decoder**, leave-one-cage-out on Cage 16; **synthetic population-raster decode** | Population **decoding**; cross-session/cross-subject **BMI** generalization; joint neural-behavior embedding | Georgopoulos et al. 1986 *Science*; Glaser et al. 2020 *eNeuro*; Gilja et al. 2012 *Nat. Neurosci.* (stable BMI across sessions); Schneider, Lee & Mathis 2023 *Nature* (CEBRA). *Stops at:* CEBRA uses behavior as a **contrastive label** to shape a neural embedding — not a symmetric "merge." |
| **Mission:** decode dominance/aggression to gate a circuit experiment | Neural circuits of social dominance & aggression the readout is time-aligned to | Wang et al. 2011 *Science* (mPFC efficacy ↔ **tube-test** rank, Hu lab); Zhou et al. 2017 *Science* (winner effect, MD-thalamus→dmPFC); **Lin** et al. 2011 *Nature* (VMHvl locus, **Dayu Lin**/Anderson lab); Lee et al. 2014 *Nature* & Hashikawa et al. 2017 *Nat. Neurosci.* (Esr1⁺ VMHvl, male/female); **Padilla-Coreano et al. 2022 *Nature*** (mPFC encodes/decodes competitive rank via tracking + HMM/GLM — the *direct* proof that students are building the behavioral half of a published neural pipeline); Chen & Hong 2018 *Neuron* (review); Anderson & Perona 2014 *Neuron* (computational ethology). *Stops at:* tube-test rank ≠ homecage aggression/despotism — correlated but dissociable axes (see rank caveat). |

---

# 2. Course Architecture

**Eight notebooks.** Granger and wavelet convolution are **folded into the EDA notebook (NB03)** as coordination-and-rhythm tools, never standalone. **Markov dynamics is its own notebook (NB07)**, paired with the activity-clock (both "behavior in time"). **Labeling is merged into the decoder notebook (NB08)** so the label-noise ceiling functions as the final dramatic threat right before the payoff, rather than a low-energy penultimate chore.

| # | Title | Role — how the mission escalates | Phase |
|---|---|---|---|
| **NB01** | The Raw Signal | Meet SLEAP pose as a high-D state vector; sort identities (spike sorting); plant the held-out bet; seal Cage 16 | Discover |
| **NB02** | The Body's-Eye View | Egocentric transform + the 19 features; the honest allocentric-naming beat | Discover |
| **NB03** | Feeling the Signal in Time | EDA: distributions, correlations, **wavelet rhythm** + **lead-lag coordination** (Granger folded, optional) | Discover |
| **NB04** | The Collapse I — PCA | Compress 19 → ~6; the behavioral manifold; residualization as a *choice* | Discover |
| **NB05** | The Collapse II — the Map | Precomputed UMAP + HDBSCAN → syllables; honest unsupervised recovery | Discover |
| **NB06** | Reading the Map (the reversal) | Enrichment stats done right; the exciting result **collapses** under cage-level rigor | Operationalize |
| **NB07** | Behavior in Time | Observed Markov grammar + the activity clock (continuous one-cage dataset) | Operationalize |
| **NB08** | The Decoder Graduates | Label → train → **unlock Cage 16** → honest held-out test → **cash the neural check** | Operationalize |

**Difficulty / skill progression.** NB01–02 are literacy and geometry (array shapes, one 2-D rotation taught by hand). NB03 introduces signals-over-time gently (looking, not modeling). NB04–05 are the conceptual peak of Phase 1 (dimensionality, manifolds) taught intuition-first with no linear algebra assumed. NB06 is the rigor peak (multiple comparisons, pseudoreplication) and the emotional low point (the reversal). NB07 adds temporal modeling. NB08 assembles everything into the payoff. **Two crises give the week an arc:** the **midpoint reversal** at NB06 (a beloved rank/sex result dies under proper statistics) and the **final threat** at NB08 (label noise may cap the decoder before it even meets Cage 16).

**Day allocation (~20h / 5 days × 4h):**
- **Day 1:** NB01 + NB02 (data literacy + reference frames)
- **Day 2:** NB03 + begin NB04
- **Day 3:** finish NB04 + NB05 (the collapse; the map)
- **Day 4:** NB06 + NB07 (rigor + dynamics)
- **Day 5:** NB08 (the graduation)

Each notebook targets **~2.5h core** with clearly-marked **stretch accordions** (e.g., Granger F-test in NB03, jPCA aside in NB04, CEBRA epilogue in NB08) so a cohort that runs long still completes the core spine.

---

# 3. Per-Notebook Blueprints

Conventions for all notebooks: plotly for interactive plots; Liberation Sans, no grid, 1.5× text, SVG+PNG for any exported static figure (house style); prose/markdown cells `hide_code=True`; every expensive step precomputed or gated behind `mo.ui.form`; sliders `debounce=True`.

---

## NB01 — The Raw Signal: SLEAP pose, identity, and the sorting problem

**Narrative beat / lab-meeting question.** *Briefing:* "Welcome to the Behavior Team. Before we point a laser at Cage 16, prove the pose we'll read behavior from is trustworthy." *Lab-meeting question:* **"The arrays label mouse 0 the 'approacher.' Is that real — and what happens to everything downstream if an identity is wrong?"** The Hero Event #742 appears in raw skeleton form; the **Sealed Cage 16** panel is introduced with its unlock counter; the **held-out bet** is written on the board: *a decoder is only trustworthy if it survives a cage it never saw.*

**Learning objectives.** Read a pose tensor axis by axis; distinguish *track slot* from *identity*; audit missingness; understand why one identity swap is catastrophic; frame identity tracking as spike sorting.

**Data used.** `load_slp_demo()` (a real decoded `.slp` clip → `kp (frames, mice=3, nodes=15, xy=2)`, `node_names`, `edges`); `train_events.npz` via `load_events()` → event tensor `(N=1500, T=130, 3, 15, 2)` plus `agg_label`, `category`, `condition`, `cage`, `sex`, `contact_rel`, `event_key`. Plus **`neural_demo.npz`** for a small "your data's neural cousin" panel (a population raster shown beside the pose tensor to make "two sciences" concrete from hour one).

**Content outline (cell by cell, high level).**
- Briefing + mission + Sealed Cage 16 + the bet.
- Verbose axis anatomy of `kp`: print shapes; name every axis; the 15-node star skeleton and edge list; `NaN` = untracked.
- Interactive **frame scrubber** (`skeleton_fig` + `mo.ui.slider`) on Event #742.
- **EDA:** per-node tracked-fraction bar chart; per-mouse frame-presence; minimum inter-mouse centroid distance per frame; events/frames per condition and per cage (so students know their sample).
- **Identity vs slots:** show a constructed track-swap; overlay the swap on the min-distance trace to reveal swaps cluster where mice are closest.
- **Neural cousin panel:** a spike raster from `neural_demo` beside the pose tensor — "both are high-dimensional signals over time; both have a detection-and-identity problem." (Returned to in NB08.)

**Prebuilt functions (inputs → outputs).**
- `load_slp_demo() -> dict{kp:(F,3,15,2) float, node_names:list[15], edges:list[(i,j)]}`
- `load_events(name='train') -> dict{kp:(N,130,3,15,2), agg_label:(N,), category:(N,), condition:(N,), cage:(N,), sex:(N,), contact_rel:(N,), event_key:(N,)}`
- `skeleton_fig(pose_2d:(3,15,2), edges) -> plotly.Figure`
- **NEW** `node_reliability(kp) -> (15,) fraction finite per node`
- **NEW** `centroid_jumps(kp) -> (3, F) per-track per-frame centroid displacement`

**Coding exercise (hypothesis-driven).**
- **Hypothesis banner:** *"Mouse 0 (the 'approacher') truly moves more than mouse 1 in the second before contact."*
- **Tools made explicit:** `centroid_jumps`; `np.nanmean`; `scipy.stats` sign test (or a supplied `sign_test(a,b)` helper).
- **TODO stub:** for each of 1500 events, compute each mouse's mean tail-base speed over the 50 frames before `contact_rel`; return the fraction of events where mouse 0 > mouse 1; run a sign test.
- **Solution approach:** vectorized `np.diff` on the TTI (node 11) track; paired comparison; report fraction + p.
- **Self-check (tolerance band):** fraction is high and significantly > 0.5 (assert within a band around the precomputed value); and identify the least-reliable node (`node_reliability`) — predict **tail_tip**.

**Conceptual questions.** Why does a single identity swap corrupt a "who-approached-whom" label more catastrophically than a handful of dropped nodes? Which of these is the behavioral analog of a spike-sorting *merge* error vs a *split* error?

**Equations / failure modes / open-ended.** No equations. *Failure modes on this data:* tail-chain nodes drop out (this is exactly why the tail-mark rank labels carry ~16% error — foreshadow the standing caveat); swaps peak at contact. *Open-ended:* "If you could add one cheap sensor to disambiguate identity at contact, what would it be, and what pose feature would it fix?"

**Neuroscience Connection.** *Prose:* "Keeping each mouse's identity straight across frames is the same problem a spike sorter solves — binding ambiguous detections to a stable source over time." *Accorion:* Lewicki 1998 / Kilosort; SLEAP (this course's lab lineage). *Where it stops:* pose **estimation** ≈ spike **detection**; **identity tracking** ≈ **sorting** — two different steps.

**Ending + hook.** Board: representation = **11,700 raw numbers per event**; readiness = 0. "Next we ship the feature layer — but first we have to choose a point of view, the same choice the brain makes with reference frames."

---

## NB02 — The Body's-Eye View: egocentric reference frames and the 19 features

**Narrative beat / lab-meeting question.** *Briefing:* "Arena coordinates are useless to us — a fight is a fight in any corner. Re-express every event the way the brain does: relative to the animal itself." *Lab-meeting question:* **"After we strip out where-in-the-cage and which-way-facing, what social geometry is left — and does aggression arrive from a different direction?"**

**Learning objectives.** Build translation+rotation from intuition with no matrix prerequisite; understand invariance; read the 19 features; **confront the allocentric-naming truth honestly.**

**Data used.** `train_events.npz` (`kp`, `agg_label`, `contact_rel`, `sex`, `cage`). Precomputed feature matrix **`X (1500,19)`** shipped in the bundle (so no live 1500-event loop).

**Content outline.**
- Briefing.
- **Rotation toy** (mirrors the PCA long-axis toy): a slider rotates a single 3-point skeleton; students *discover by hand* the heading angle that snaps the approacher to face +y; "spin the paper until the mouse faces up." Equations only in an accordion.
- Apply `allocentricize` to Event #742; **side-by-side raw vs body-frame** interactive view.
- **The honest terminology beat** (main prose, not buried): the code calls these "allocentric" features, but the transform is **egocentric** (self-centered). We keep the field's name and explain why — and use it to teach the real distinction between egocentric coding and the allocentric world-map the brain also maintains.
- The **19 features** with plain-English meaning each (closing_speed, approacher/approachee facing cosines, pair distances, nose→tail-base distances, triangle area, bystander distances, body length, angular velocity).
- **Invariance demo:** translate + rotate a whole event by a random arena pose; watch raw coordinates swing while the 19 features stay fixed to ~1e-4.
- Feature distributions split by `agg_label` (plotly violins, p-values annotated).

**Prebuilt functions (inputs → outputs).**
- `allocentricize(event_kp:(130,3,15,2)) -> (130,3,15,2)` centered on approacher TTI, heading rotated to +y. *Failure case:* if head or TTI missing, falls back to identity — state this.
- `_heading(pose) -> angle`; `_anchor_transform(pose, center, angle) -> pose` (shown for the toy).
- `features_one(event_kp) -> (19,)`; `features_batch(kp) -> (N,19)` (**now vectorized** — see Engine Plan); `FEATURE_NAMES: list[19]`.

**Coding exercise (hypothesis-driven).**
- **Hypothesis banner:** *"Aggressive contacts arrive from a different angle than non-aggressive ones."*
- **Tools:** `allocentricize`; the approachee TTI index; `np.histogram2d`; `agg_label`.
- **TODO stub:** (a) implement recenter+rotate for one event from `(center, heading)` and assert the approacher lands at the origin facing +y; (b) pool all events, take the approachee centroid in the approacher's body frame at `contact_rel`, build a front/behind/beside occupancy density, split by `agg_label`, and test the front-vs-rear fraction difference.
- **Solution approach:** vectorized transform over the pooled contact frames; 2-D density; two-proportion test.
- **Self-check:** part (a) origin+heading assertion (exact); part (b) front-vs-rear fraction difference has the expected sign and magnitude within a band.

**Conceptual questions.** Which brain systems compute in which frame — place/grid/HD (allocentric) vs retrosplenial/parietal (the transform)? Why is removing the approacher's own arena pose the thing that *isolates* social geometry? Why does the field call an egocentric transform "allocentric," and does the name matter for the science?

**Equations / failure modes / open-ended.** 2-D rotation matrix (accordion only). *Failure modes:* missing head/TTI → identity fallback silently produces arena-frame features (audit for it); a mis-estimated heading rotates the whole event wrongly. *Open-ended:* "Head-direction cells *encode* heading; your transform *factors it out*. Sketch how a brain could use an HD signal to move between the two frames."

**Neuroscience Connection.** *Prose:* "You just did, by hand, what retrosplenial and parietal cortex do continuously — convert a self-centered view into a stable frame — while place, grid, and head-direction cells hold the world-anchored map at the other end." *Accordion:* O'Keefe 1971; Hafting 2005; Taube 1990; **Alexander et al. 2020 Science Advances**; social coding Danjo 2018 (rats) / Omer 2018 (bats). *Where it stops:* the transform is egocentric; place cells are the allocentric endpoint — opposite ends of the same computation; conspecific-coding papers are rat/bat, not mouse.

**Ending + hook.** Board: representation = **19 features**. "Before we compress these 19 numbers, let's *look* at them — in value, in time, and in frequency — the way a physiologist reads a raw trace."

---

## NB03 — Feeling the Signal in Time: rhythm and coordination (EDA)

**Narrative beat / lab-meeting question.** *Briefing:* "Two mice in an encounter are two coupled systems with a tempo and a direction. Give us a rhythm readout and a who-leads-whom readout — the behavioral twins of the oscillation and connectivity analyses we run on the brain." *Lab-meeting question:* **"In the run-up to contact, who moves first — and is the leader the aggressor?"** This is the "last look before the collapse": we survey what coordination and rhythm PCA/UMAP are about to average away.

**Learning objectives.** Read distributions and a correlation heatmap (foreshadowing that features are *not* independent → PCA); read a signal in time and in frequency; measure lead-lag without regression; understand the common-cause confound.

**Data used.** `train_events.npz` (`kp`, `X`, `agg_label`, `sex`, `condition`, `contact_rel`) plus **new fields `initiator_idx`, `fleer_idx`** (nullable; built upstream from the project's initiator/fleer logic). For any live coordination loop, a **balanced ≤200-event subsample** is used so it runs <2s; the full-1500 result is shown from a precomputed array beside it.

**Content outline.**
- Briefing.
- Feature histograms; **19×19 correlation heatmap** (plotly) — "these knobs move together; that's why 19 will collapse to ~6."
- **Rhythm sub-section:** per-frame closing-distance and speed traces for Event #742; a **Morlet wavelet spectrogram** of the approacher's speed → dominant-frequency readout. Taught as "a sinusoid slid along the signal"; the time-frequency (Heisenberg) trade-off and edge effects named as failure modes over a 2.6 s window.
- **Coordination sub-section (primary tool = cross-correlation lag):** shift one mouse's speed against the other, find the peak-correlation lag = "who moves first." Runs on **pre-contact frames only**, where approacher and *initiator* can dissociate. A tiny intuition toy: "predict B's next step from A's last step."
- **Granger (stretch accordion):** the same question via a pure-numpy VAR restricted-vs-unrestricted F-test; hard common-cause caveat (the bystander/arousal as a shared driver); note bivariate ≠ conditional.
- Every view split by `sex` and `condition`.

**Prebuilt functions (inputs → outputs).**
- **NEW** `wavelet_power(sig:(T,), freqs, fps) -> (n_freqs, T)` (pure numpy Morlet convolution; no `pywt`).
- **NEW** `cross_corr_lag(x:(T,), y:(T,), max_lag) -> (lags, corr), peak_lag` (regression-free).
- **NEW (stretch)** `granger_pair(x, y, lags) -> {f_xy, f_yx, p_xy, p_yx}` (numpy VAR F-test; no `statsmodels`).
- Reuse `features_batch`, `FEATURE_NAMES`.

**Coding exercise (hypothesis-driven).**
- **Hypothesis banner:** *"In the pre-contact window, the mouse that moves first is the aggression initiator more often than chance."*
- **Tools:** `cross_corr_lag`; `initiator_idx`; pre-contact speed traces; a supplied shuffle helper.
- **TODO stub:** for aggression events with a defined `initiator_idx`, compute the pre-contact lead-lag between the two mice, label the "leader," and test whether leader == initiator above chance; compare with a within-event shuffle null.
- **Solution approach:** vectorized lag search on the subsample; binomial test vs chance; report the shuffle-null distribution.
- **Self-check (robustness, not a canned p):** the leader↔initiator match rate exceeds the shuffle null's 95th percentile *and* the effect survives shuffling (assert the *gap*, within tolerance). If the signal is weak, the graded answer is "does not robustly exceed chance" — pre-verified at build time.

**Conceptual questions.** Name the confound: both mice may be driven by a common cause (the third mouse; shared arousal), so lead-lag is coordination, not proof of driving. When does a wavelet beat an FFT? Why is testing on pre-contact frames essential to avoid recovering the approacher "by construction"?

**Equations / failure modes / open-ended.** Cross-correlation (intuitive); Granger F-test (accordion). *Failure modes:* 130-frame windows are short and nonstationary → noisy estimates; approacher≠initiator matters. *Open-ended:* "How would you condition out the bystander mouse to move from bivariate to conditional coordination?"

**Neuroscience Connection.** *Prose:* "The wavelet you just ran on a mouse's speed is the exact transform neuroscientists run on LFP to find theta and gamma; the lead-lag you measured is the behavioral face of directed functional connectivity — Kingsbury and colleagues found dmPFC coupling *between two brains* tracks their dominance relationship." *Accordion:* Cohen 2014; Buzsáki 2004; Seth 2015; **Bressler & Seth 2011 NeuroImage**; Kingsbury 2019 (GLM/correlation, *not* Granger). *Where it stops:* matching frequency ≠ matching mechanism; Granger measures prediction, not cause.

**Ending + hook.** Board unchanged (still 19; we *looked*, didn't compress). "Nineteen correlated knobs are really a handful of independent ones — next we find them."

---

## NB04 — The Collapse I: PCA and the dimensionality of behavior

**Narrative beat / lab-meeting question.** *Briefing:* "Before you show us clusters, tell us: is one boring axis — how-close, how-fast — dominating everything? The opto readout can't be confounded by overall activity." *Lab-meeting question:* **"How many dimensions does mouse social behavior actually have, and does hunger move it along one of them?"**

**Learning objectives.** Understand variance-maximizing axes with zero linear algebra; read a scree plot and PC loadings; treat residualization as a *modeling choice*, not a fact.

**Data used.** `train_events.npz` `X`, `condition`, `cage`, `agg_label`; shipped **precomputed `pca_scores`, `explained_variance_ratio`, `pca_components`** (so nothing >2s runs live).

**Content outline.**
- Briefing.
- **Intuition-first 2-D toy:** two correlated features; a slider rotates a candidate axis; students find the max-spread axis *by hand*, then PCA is revealed to do it automatically. Eigen-equations in an accordion only.
- `standardize` then `pca_scores`; **scree / cumulative-variance** plot; the 90%-variance dimension → Gauge A falls to ~6.
- **PC loadings as "eigen-behaviors"** (`pca_loadings_fig`): which features load together; name PC1 (expected: a proximity/locomotor-magnitude axis).
- **Residualization beat (the drop_pcs teaching moment):** the axis we may drop as "nuisance" for later clustering is the *same* speed/closing axis that carries the hunger effect. Presented deliberately as the open question "nuisance is a choice," with an interactive toggle showing dropping PC1 does **not** erase the aggression signal.
- Project all events onto PC1–PC3; color by `condition`.

**Prebuilt functions (inputs → outputs).**
- `standardize(X) -> (Xz, mean, std)`
- `pca_scores(Xz) -> {scores:(N,k), explained_variance_ratio:(k,), components:(k,19)}`
- `residualize(Xz, drop=(...)) -> X_resid` (zeroes chosen PCs, reconstructs)
- **NEW** `pca_loadings_fig(components, FEATURE_NAMES) -> plotly.Figure`

**Coding exercise (hypothesis-driven).**
- **Hypothesis banner:** *"Food-deprived (dep) events sit higher on the PC that loads on speed/closing than pre/post events."*
- **Tools:** `pca_scores`; `condition`; `cage`; Mann-Whitney U.
- **TODO stub:** compute the scree curve and 90% dimension; project events onto the PC with top speed/closing loadings; test dep vs pre+post with Mann-Whitney; then repeat aggregating one value **per cage** to preview the pseudoreplication lesson.
- **Self-check:** 90% dimension within a band around the precomputed value; dep-shift direction correct at the event level; note whether it survives cage aggregation (sets up NB06).

**Conceptual questions.** Why can a **low-variance** PC be the *decodable* one (rare-but-meaningful aggression)? Which PC would *you* call "nuisance," and why is that a choice, not a fact?

**Equations / failure modes / open-ended.** Covariance eigen-decomposition (accordion). *Failure modes:* PCA is linear and variance-greedy — a rare behavior hides in a tiny PC; standardization changes which axis "wins." *Open-ended:* "If aggression lives in a low-variance direction, what would you do instead of PCA?" (motivates UMAP).

**Neuroscience Connection.** *Prose:* "You just found the behavioral manifold — the same discovery Stephens made for the worm and Cunningham & Yu for cortex: a signal that looks high-dimensional actually traces a low-dimensional shape." *Accordion:* Stephens 2008; Cunningham & Yu 2014; Gallego 2017; Kobak 2016 (dPCA ≈ residualization); jPCA aside (Churchland 2012 — *rotational-dynamics variant*, not plain PCA). *Where it stops:* shared geometry, not biological identity.

**Ending + hook.** Board: representation = **~6 PCs**. "Linear axes can't unfold a curved manifold — next we lay it flat and carve it into behaviors."

---

## NB05 — The Collapse II: the behavioral map (UMAP + HDBSCAN)

**Narrative beat / lab-meeting question.** *Briefing:* "Give us the catalog of what these mice DO — discovered from the data, not from our assumptions." *Lab-meeting question:* **"Can we find aggression without ever being told which events are aggressive?"** This is Phase 1's climax: social life becomes a landscape you can point at.

**Learning objectives.** Read a UMAP map without over-reading it; run density clustering; validate a cluster by eye and by enrichment; internalize the **hard molab rule** (UMAP is never live).

**Data used.** **`umap_sweep.npz`** (expanded **5×5** grid `emb_grid (5,5,1500,2)` over `n_neighbors × min_dist`, plus canonical `default_labels` so NB05–07 agree); `train_events.npz` `category`, `agg_label`; precomputed exemplar GIFs.

**Content outline.**
- Briefing + the engineering lesson: **why live UMAP crashes molab** (28 s numba JIT + websocket timeout) → we *select* from a precomputed sweep.
- `sweep_grid_fig` small-multiples over the 5×5 grid; the two knobs (n_neighbors = local↔global; min_dist = tightness) explored by selection, not recompute.
- `run_hdbscan` (the one fast live compute) with a debounced `min_cluster_size` slider; label −1 = noise.
- Overlay `category` tags and **Hero Event #742** on the map; render a cluster's exemplars with `grid_gif_bytes` (≤5×5 grid).
- **How UMAP lies:** show two seeds/min_dist settings that redraw the picture; state what you may *not* conclude (inter-cluster distance, cluster size).

**Prebuilt functions (inputs → outputs).**
- `load_umap_sweep() -> {emb_grid:(5,5,1500,2), n_neighbors:(5,), min_dist:(5,), default_labels:(1500,)}`
- `sweep_grid_fig(emb_grid, ...) -> plotly.Figure`
- `run_hdbscan(emb:(1500,2), min_cluster_size, min_samples) -> labels:(1500,)`
- `grid_gif_bytes(events, idxs, edges, rows, cols) -> bytes`; `gif_img_html(bytes) -> html`

**Coding exercise (hypothesis-driven).**
- **Hypothesis banner:** *"At least one data-driven cluster is enriched for aggression above the ~0.30 base rate."*
- **Tools:** `run_hdbscan`; `emb_grid`; `agg_label`; `category`.
- **TODO stub:** cluster a chosen embedding, compute each cluster's aggression fraction, find the purest cluster, quantify **lift** vs base rate, and render its exemplar GIFs to eyeball coherence.
- **Solution approach:** group `agg_label` by cluster; sort by fraction; compare to chance.
- **Self-check (honest tolerance band):** the purest cluster's aggression fraction exceeds base rate by at least the **pre-verified, build-time lift** (authors set this to the *actual* achievable value after retuning `CLUSTER_DEFAULTS` and the 5×5 sweep; if it is modest, the exercise **reports the modest lift honestly** and the conceptual question does the teaching). **Extension** replaces the dead mounting/grooming task (mounting n=3, tail_bite n=7 — too rare to cluster) with **aggression vs quiet-contact / ledge** separability, which the corpus supports.

**Conceptual questions.** Is a "cluster" a real behavior or an artifact of the knobs — how would you decide? Why do subtle, rare categories (mounting, grooming) resist density clustering while gross ones (aggression vs rest) separate?

**Equations / failure modes / open-ended.** No equations (UMAP is a graph layout, described intuitively). *Failure modes:* a large undifferentiated blob can swallow a majority of events; seed sensitivity. *Open-ended:* "The map merged two behaviors you can tell apart by eye — what feature would you add to split them?"

**Neuroscience Connection.** *Prose:* "Embedding a cloud of behavior and carving it into recurring modules is the computational-ethology move pioneered by MotionMapper and B-SOiD — the same idea behind 'behavioral syllables,' which we'll model properly in two notebooks." *Accordion:* McInnes 2018 (UMAP); **Berman 2014** (t-SNE, the method ancestor); **Hsu & Yttri 2021** (B-SOiD, UMAP+cluster); Campello 2013 (HDBSCAN); Wiltschko 2015 for the *concept* only. *Where it stops:* MoSeq is an AR-HMM, **not** embed-then-cluster — that twin belongs to NB07.

**Ending + hook.** Board: representation = **a 2-D map + discrete syllables**; Phase 1 complete. "A map is only science if you can test claims on it — and that's where our beautiful result is about to get complicated."

---

## NB06 — Reading the Map: statistics done honestly (THE REVERSAL)

**Narrative beat / lab-meeting question.** *Briefing:* "This is the exact test we'll run after opto — does a variable shift the cluster distribution? Prove the logic on variables we already have, and don't fool yourselves." *Lab-meeting question:* **"Did hunger rewrite the ethogram — and is the sex difference you're excited about real, or is it just cage identity?"** This is the **midpoint reversal**: the event-level sex/rank result that looked exciting **collapses** the moment cage becomes the unit and the 16% ID error is honored.

**Learning objectives.** Contingency tables, standardized residuals, Bonferroni; the pseudoreplication trap (cage is the unit); permutation nulls at the correct level; effect size vs p; why rank must be handled with a loud caveat.

**Data used.** `train_events.npz` `category`/cluster labels, `condition`, `sex`, `cage`, `ranks`; canonical cluster labels from NB05.

**Content outline.**
- Briefing.
- `condition_enrichment` per cluster (χ², standardized Pearson residuals) → **Bonferroni** across clusters as the headline multiple-comparisons lesson. Condition is **within-cage** (pre/dep/post present inside every cage) → the clean, honest headline.
- **The reversal, staged:** the event-level `sex_enrichment` looks significant; then a **cage-level permutation null** (shuffle labels at the cage unit, n=4 M / 3 F) shows it evaporates — sex is 100% confounded with cage. Presented as a gut-punch, not a footnote: "the number you loved was pseudoreplication."
- `rank_dyad_enrichment` with an explicit **rank-reliability caveat cell** (~16% tail-mark ID error; tube-test rank ≠ homecage aggression); reason about which way the error biases the test.
- Effect sizes reported beside every p; per-cage aggregation shown throughout.

**Prebuilt functions (inputs → outputs).**
- `condition_enrichment(labels, condition) -> {chi2, p, residuals:(K,3), p_bonf}`
- `rank_dyad_enrichment(labels, ranks) -> {table, chi2, p, residuals}`
- **NEW** `covariate_enrichment(labels, covariate, unit=None) -> {chi2, p, residuals}` (generalizes to sex; `unit='cage'` switches to cage-level)
- **NEW** `permutation_test(labels, covariate, unit, n=5000) -> p_emp` (shuffles at `unit`)

**Coding exercise (hypothesis-driven).**
- **Hypothesis banner:** *"Food deprivation changes cluster composition (condition effect) — and it survives when cage, not event, is the unit."*
- **Tools:** `condition_enrichment`; `permutation_test(unit='cage')`; `covariate_enrichment`.
- **TODO stub:** build the cluster×condition table, run χ² + Bonferroni, confirm the top dep-enriched cluster with a **cage-level** 5000-shuffle null; then run the same machinery on **sex** and report whether it survives cage-level shuffling.
- **Self-check:** condition effect present and Bonferroni-surviving (within band); **the graded correct answer for sex is "cannot conclude — pseudoreplicated / underpowered at n=4 vs 3."** No self-check is graded against noise (build-time verified).

**Conceptual questions.** Why does testing 15 clusters inflate false positives? Why do we trust condition (within-cage) over sex (between-cage) here? How does ~16% rank mislabeling bias a rank-dyad χ² — toward or away from significance? Where does χ² break with small cell counts?

**Equations / failure modes / open-ended.** χ² and standardized residuals (intuitive; formula in accordion). *Failure modes:* pseudoreplication; small-n cells; multiple comparisons. *Open-ended:* "With only 7 cages, what design or analysis would give you real power to test a sex effect?"

**Neuroscience Connection.** *Prose:* "This is exactly how a circuit lab reads out a manipulation — does stimulation shift the state distribution? — and it's why the same cluster-permutation discipline governs EEG/MEG and inter-brain decoding of dominance." *Accordion:* Maris & Oostenveld 2007 (cluster-permutation); Kingsbury 2019 (dmPFC decoding of dominance). *Where it stops:* our 'states' are clusters of behavior, not neurons.

**Ending + hook.** Board: readiness rises (validated *what* the map encodes: condition yes, sex not from this design). "So far every event was a frozen snapshot. Behavior *moves* — next we model the grammar of how."

---

## NB07 — Behavior in Time: the grammar and the clock

**Narrative beat / lab-meeting question.** *Briefing:* "The rig runs for hours. We need to know whether behavior has *memory* — does the state now predict the state next — and when these mice are active, before we can say opto changed the dynamics." *Lab-meeting question:* **"Does hunger reorganize the behavioral grammar, and does it flatten or sharpen the daily activity rhythm?"** The **baton hand-off:** Event #742 is located on the activity clock ("we followed one event; now watch the day it lived inside") before we widen to the session.

**Learning objectives.** Build a first-order **observed** Markov transition matrix; read a stationary distribution via *simulation* (not eigendecomposition); measure transition entropy against a shuffle null; handle a single-cage activity clock honestly; understand observed-vs-hidden (Markov vs HMM).

**Data used.** **NEW `continuous_tracks.npz`** — Cage 15 (hero) plus **2 context cages**, dark/active phase, downsampled to **2 fps**, `float16` centroids+speed per mouse, an **int8 coarse movement-state sequence** (rest / locomote / social-proximity) derived by `discretize_states`, and a time-of-day index. (≤15 MB — see Engine Plan.) Event `event_key` timestamps for the baton hand-off.

**Content outline.**
- Briefing + baton hand-off (hero event on the clock).
- `discretize_states` on the continuous tracks → a **contiguous** state sequence (the valid Markov substrate — sparse 130-frame events are *not* a chain; state this explicitly).
- `transition_matrix` → heatmap; **stationary distribution by random-walk simulation** (`stationary_dist(method='simulate')`): release a walker on the graph, watch where it spends time — no eigen-math.
- `transition_entropy` per condition; **shuffle null mandatory**; self-transition ("stickiness") bars.
- **Activity clock:** `activity_by_tod` across the dark phase, overlaid pre/dep/post, with a **within-cage bootstrap** over 30-min bins for an uncertainty band. Framed as an **n≈3 case study**, explicitly *not* a population circadian claim (reverse cycle: lights ON 21:00–09:00).

**Prebuilt functions (inputs → outputs).**
- **NEW** `load_continuous_tracks(cam) -> {centroids:(3,T,2), speed:(3,T), state_seq:(T,), tod_hour:(T,), fps}`
- **NEW** `discretize_states(tracks, thresholds) -> state_seq:(T,) int, state_names`
- **NEW** `transition_matrix(state_seq, n_states) -> (K,K) row-normalized`
- **NEW** `stationary_dist(T, method='simulate', steps=100000) -> (K,)`
- **NEW** `transition_entropy(T) -> float`; `shuffle_transition_null(state_seq, n) -> null_entropies`
- **NEW** `activity_by_tod(tracks, bin_min=30) -> {curve, ci_low, ci_high}` (bootstrap CI)
- **NEW** `time_of_day(event_key) -> float hour`

**Coding exercise (hypothesis-driven).**
- **Hypothesis banner:** *"Deprivation raises self-transitions / lowers grammar entropy (behavior gets 'stickier') in this cage."*
- **Tools:** `discretize_states`, `transition_matrix`, `transition_entropy`, `shuffle_transition_null`.
- **TODO (two-tier):** Tier-1 — build the transition matrix *by counting* state-pair transitions with `np.add.at` and row-normalizing (student writes this); Tier-2 — call `transition_entropy` and the shuffle null. Compare entropy pre/dep/post with a bootstrap CI; plot the activity clock.
- **Self-check (robustness):** the entropy/self-transition shift has the expected direction **and** exceeds the shuffle null (assert the gap, tolerance band). The clock is graded descriptively (does the curve + CI render), not as an inference.

**Conceptual questions.** What does a first-order Markov assumption throw away (memory beyond one step)? Why is a shuffle null mandatory? Crucially: **we labeled our states — a true HMM/AR-HMM infers *hidden* states from emissions; what would that buy us?** Why can't the sparse approach-event clusters form a valid chain?

**Equations / failure modes / open-ended.** Transition matrix + stationary distribution (via simulation; eigen-form in accordion). *Failure modes:* n≈3 cages can't support circadian inference; long self-transitions can dominate the matrix; short recordings bias entropy. *Open-ended:* "How would you detect a *hidden* state the observed labels miss?"

**Neuroscience Connection.** *Prose:* "A transition matrix over behavioral states is the observed cousin of the hidden-Markov models neuroscientists fit to latent brain states — and MoSeq's AR-HMM is exactly this idea applied to mouse behavior, with its syllable transitions read out by the striatum." *Accordion:* Wiltschko 2015 (AR-HMM); Markowitz 2018 (striatum); Jones 2007; Mazzucato 2015. *Where it stops:* ours is a fully-observed chain; the brain's states are latent and must be inferred — that inference is the HMM.

**Ending + hook.** Board: readiness rises (dynamics validated). "We have a representation, we know what it encodes and how it moves. The decoder needs a teacher — and then it meets Cage 16."

---

## NB08 — The Decoder Graduates: labels, the held-out cage, and the neural check (PAYOFF)

**Narrative beat / lab-meeting question.** *Briefing:* "This is it. But a decoder is only as honest as its labels — hand-score ground truth, train the readout, then meet Cage 16, the animal on the rig, for the first time. If it reads a cage it never saw, the laser turns on." *Lab-meeting question:* **"Does the decoder generalize to a brand-new cage — and is it trustworthy enough to time-align to a neural recording?"** Structured as a **three-beat resolution**: (Act 1) define ground truth and confront the **label-noise ceiling — the final threat**; (Act 2) train; (Act 3) **unlock Cage 16**, validate honestly, and cash the neural check.

**Learning objectives.** Build an ethogram by clicking; measure agreement against a reference and feel the accuracy ceiling; train and honestly evaluate a decoder leave-one-cage-out; read what it relies on; experience — not merely read — the behavior↔brain equivalence.

**Data used.** `train_events.npz` (features `X`, `pca_scores`, canonical cluster labels, coordination features), `answer_key.csv`, **`heldout_events.npz` (Camera 16 — unlocked here; ~470 events, ~180 aggression, ground-truth labels; if cam16 is a female cage by the even-camera convention, the test doubles as cross-sex)**, and **`neural_demo.npz`** (synthetic population raster + hidden-state labels; optional precomputed CEBRA-style 2-D joint-embedding coordinates).

**Content outline.**
- Briefing + Sealed Cage 16 still locked.
- **Act 1 — Ground truth & the ceiling (the crisis):** a click-to-label grid over exemplar GIFs (`mo.ui.batch` of GIF cells gated by an `mo.ui.form`; labeling guidelines in an accordion). Assemble aggression / not-aggression labels. Compute agreement vs `answer_key` with `cohens_kappa` — **named honestly as agreement-with-a-reference (accuracy), not inter-rater reliability** (a genuine second-labeler pass is offered as a stretch path to make "inter-rater" honest). Measure how false-positive rate on the deliberately-ambiguous `mlp_fp` exemplars moves as the criterion tightens; **predict how this label noise caps the decoder's ceiling.**
- **Act 2 — Train:** `make_mlp` on the 19 features; a logistic-regression linear baseline. **Feature-set comparison so earlier stages demonstrably feed the readout:** 19 features vs PCA scores (NB04) vs +cluster-membership one-hot (NB05) vs +coordination features (NB03). Within-cage split first.
- **Act 3 — Cage 16 unlocks:** train on cams 9–15, evaluate on cam16 with `eval_binary`; `roc_pr_fig`, confusion matrix, `calibration_curve`; a debounced **decision-threshold slider** with live precision/recall for the opto readout; quantify the **within-cage vs LOCO gap**; `permutation_importance` = the decoder's "receptive field"; render predicted-aggression clips on Cage 16.
- **Simulate the opto readout:** inject a synthetic "VMHvl stim doubled attack" shift into Cage 16 and ask whether the decoder catches it at the chosen threshold and sample size (turns the mission's hypothetical into a lived, interactive beat).
- **Cash the neural check:** run the **identical** `make_mlp` / `eval_binary` on `neural_demo`'s population raster to decode its hidden state — "you just decoded a brain with the exact code you built for behavior." **CEBRA epilogue:** interact with the precomputed joint neural-behavior embedding, with the honest limit that CEBRA uses behavior as a contrastive label, not a symmetric merge.
- Close the Readout Board; return memo from the circuit team: "Readout validated — opto trial GREEN-LIT for Cage 16."

**Prebuilt functions (inputs → outputs).**
- `make_mlp() -> sklearn Pipeline (impute → scale → MLPClassifier)`; `fit(X, y)`
- `eval_binary(model, X, y) -> {roc_auc, average_precision, confusion, y_score}`
- **NEW** `cohens_kappa(a, b) -> float` (agreement)
- **NEW** `roc_pr_fig(y, scores) -> plotly.Figure`; `calibration_curve(y, scores) -> (frac_pos, mean_pred)`
- **NEW** `load_neural_demo() -> {X_neural:(n,neurons), y:(n,)}` and `synthetic_population_raster(...)` used to build it
- Reuse `grid_gif_bytes`, `event_gif_bytes`, `sklearn.inspection.permutation_importance`.

**Coding exercise (hypothesis-driven).**
- **Hypothesis banner:** *"A decoder trained on cages 9–15 detects aggression in never-seen Cage 16 (expected held-out ROC-AUC ≈ 0.86), and generalization costs accuracy vs a within-cage split."*
- **Tools:** `make_mlp`, `eval_binary`, `permutation_importance`; the four feature sets; `heldout_events.npz`.
- **TODO stub:** train on cams 9–15, evaluate on cam16; report ROC/PR and confusion; quantify the within-cage vs LOCO gap; run permutation importance; **then** run the same `make_mlp`/`eval_binary` on `neural_demo` and log both to the board.
- **Self-check:** held-out AUC within a tolerance band of ~0.86; LOCO AUC < within-cage AUC (the honest gap); the neural-demo decode beats chance.

**Conceptual questions.** Why is held-out **cage** (not held-out event) the only honest test — cage as the true unit, the pseudoreplication guard, the neural analog being a held-out session? How does the ~16% identity-label error cap the achievable ceiling? For a causal experiment, argue precision vs recall (a false "attack" fakes an effect; a missed one hides it). **What circuit experiment would you now run, with this decoder gating it?**

**Equations / failure modes / open-ended.** ROC/AUC and calibration (intuitive). *Failure modes:* a decoder that secretly learned the cage (e.g., cam16 tail-mark dropout) rather than the behavior; label noise ceiling; threshold miscalibration under class imbalance. *Open-ended:* "Your decoder is ready — describe the opto experiment it unblocks and how you'd time-align it."

**Neuroscience Connection.** *Prose:* "Leave-one-cage-out is the behavioral face of a cross-session BMI decoder that must survive a new recording — and Padilla-Coreano and colleagues did the mirror image, decoding competitive rank from mPFC ensembles with tracking + an HMM/GLM. You built the behavioral half of a published neural-decoding pipeline; now you've run it on neurons too." *Accordion:* Georgopoulos 1986; Glaser 2020; Gilja 2012; **Padilla-Coreano 2022**; Schneider/Lee/Mathis 2023 (CEBRA). *Where it stops:* CEBRA contrastively shapes a neural embedding using behavior — the poetic "one space" is the aspiration, contrastive learning is the mechanism.

**Ending + summary.** Board complete: **11,700 raw numbers → one trustworthy decision on a cage you never trained on**, plus a decoder that reads a population raster with the same code. The mission resolves; the readout ships to the rig. "Next week, every stage you built gets a neural twin — and you already know how to run it."

---

# 4. Data & Engine Plan

### Data artifacts to build

1. **`train_events.npz` (augment existing).** Keep `kp (1500,130,3,15,2)`, `agg_label`, `category`, `condition`, `ranks`, `contact_rel`, `event_key`. **ADD:** `cage`, `sex` (joined from `cohort_meta.csv`), `tod_hour` (from `event_key`, reverse-cycle aware), and **`initiator_idx`, `fleer_idx`** (nullable, derived upstream from the project's existing initiator/fleer logic — unblocks the genuine who-initiates exercise in NB03). **Precompute & ship** `X (1500,19)`, `pca_scores`, `explained_variance_ratio`, `pca_components`, and canonical cluster labels (so NB02/04/05/06/07 do no >2 s live compute).

2. **`heldout_events.npz` (augment).** Same schema + added fields for Camera 16. Confirm cam16 sex; if female, document the cross-sex bonus.

3. **`umap_sweep.npz` (expand).** From 3×3 to **5×5** `emb_grid (5,5,1500,2)` over `n_neighbors × min_dist`, with the axis values and **canonical `default_labels`**. Retune `CLUSTER_DEFAULTS` to maximize a real aggression-enriched cluster; **record the achieved lift** as the pre-verified NB05 self-check target (if modest, the notebook teaches it honestly). ~0.3–1 MB.

4. **`continuous_tracks.npz` (NEW).** Cage 15 (hero) + 2 context cages, dark/active phase, **2 fps**, `float16` centroids `(3,T,2)` + speed `(3,T)`, **int8 `state_seq (T,)`** from `discretize_states`, `tod_hour (T,)`, `fps`. Budget **≤15 MB total** (≈5 MB/cage). Store only centroid+speed+state (not all 15 nodes). Supports NB07's grammar (n≈3, honest) and the single-cage clock (with bootstrap CI).

5. **`neural_demo.npz` (NEW).** Synthetic population raster `(n_trials, n_neurons)` with a hidden binary state modulating firing + labels; optional precomputed **CEBRA-style 2-D joint-embedding coordinates** for the epilogue. Small (<1 MB). Lets NB08 *cash* the neural twin with the identical pipeline (no heavy deps in the student path).

6. **`answer_key.csv` (existing).** Reference labels for NB08's agreement check.

7. **Rendered exemplar GIFs.** Skeleton-on-blank-canvas, short, small, **≤5×5 grids**, precomputed per canonical cluster and for Hero Event #742. Follow H.264/`yuv420p` for any MP4; GIFs kept tiny.

8. **`readout_board.csv` (NEW).** Committed benchmark values per stage for the tracker; each notebook displays its student-computed number beside the benchmark.

**Bundle budget.** Current ~15 MB (train 11 MB + heldout 3.5 MB) + continuous ~15 MB + sweep ~1 MB + neural_demo <1 MB + GIFs a few MB ≈ **~32–40 MB — comfortably inside 50–75 MB.** Add a `tools/` build check that **fails the build if the bundle exceeds 60 MB.** Per-asset caps: continuous ≤15 MB, all precomputed demo arrays combined ≤10 MB.

### NEW `course_utils.py` helpers (name — signature → returns)

- `node_reliability(kp) -> (15,)` — per-node finite fraction.
- `centroid_jumps(kp) -> (3, F)` — per-track per-frame centroid displacement (swap flags).
- `wavelet_power(sig, freqs, fps) -> (n_freqs, T)` — pure-numpy Morlet convolution (no `pywt`).
- `cross_corr_lag(x, y, max_lag) -> (lags, corr), peak_lag` — regression-free coordination.
- `granger_pair(x, y, lags) -> {f_xy, f_yx, p_xy, p_yx}` — numpy VAR F-test (stretch; no `statsmodels`).
- `cohens_d(X, y) -> (19,)` — effect-size feature ranking.
- `covariate_enrichment(labels, covariate, unit=None) -> {chi2, p, residuals}` — generalizes condition/sex; `unit='cage'` for cage-level.
- `permutation_test(labels, covariate, unit, n=5000) -> p_emp` — shuffle at `unit` (cage).
- `discretize_states(tracks, thresholds) -> state_seq, state_names`.
- `transition_matrix(state_seq, n_states) -> (K,K)`.
- `stationary_dist(T, method='simulate', steps=100000) -> (K,)` — default random-walk simulation, not eig.
- `transition_entropy(T) -> float`; `shuffle_transition_null(state_seq, n) -> null`.
- `load_continuous_tracks(cam) -> dict` (centroids, speed, state_seq, tod_hour, fps).
- `time_of_day(event_key) -> float`; `activity_by_tod(tracks, bin_min=30) -> {curve, ci_low, ci_high}`.
- `cohens_kappa(a, b) -> float` (labeled as *agreement*).
- `pca_loadings_fig(components, names) -> Figure`; `roc_pr_fig(y, scores) -> Figure`; `calibration_curve(y, scores) -> (frac_pos, mean_pred)`.
- `load_neural_demo() -> {X_neural, y}`; `synthetic_population_raster(n_neurons, n_trials, state_seq, rng) -> (X, y)`.
- **Vectorize `features_batch`** (currently a per-event Python loop — violates the repo vectorization standard and the >2 s molab rule) and **precompute `X` into the bundle**.

### Molab-safety / engineering notes (mandatory)

- **No live UMAP anywhere in the student path.** Remove or hard-guard `run_umap` from the student-importable surface; NB05 only *selects* from `emb_grid`. HDBSCAN is the single fast live clustering call.
- **Precompute or gate everything >2 s:** ship `X`, `pca_scores`, cluster labels; gate `features_batch` (also vectorized), permutation nulls, and MLP `fit` behind `mo.ui.form`; use a **balanced ≤200-event subsample** for any live coordination/Granger loop and show the full precomputed result beside it.
- **Debounce all sliders** (`debounce=True`); use `mo.ui.form`/`mo.ui.batch` to gate expensive recompute.
- **Light deps only:** `numpy`, `scipy`, `scikit-learn`, `plotly`, `hdbscan`, `marimo`. **No `statsmodels`, no `pywt`, no `sleap-io`** in the student bootstrap (a pre-decoded npz replaces `.slp` loading; Granger/Morlet are hand-rolled).
- **Self-checks use tolerance bands** and are **verified at build time** against the real bundled arrays so no student is ever graded against noise. For any exercise where the honest signal is weak (NB03 coordination, NB05 recovery, NB06 sex), the graded correct answer is the honest one ("does not exceed chance" / "cannot conclude").
- **The Readout Board and any cross-notebook state degrade gracefully** — molab forks per session, so each notebook recomputes its own stage value and displays it beside the committed benchmark; a missing scratch/JSON never errors.
- **Pre-authoring gate:** the notebooks cannot be built until (1) `features_batch` is vectorized and `X`/PCA are shipped, (2) `run_umap` is guarded and the 5×5 sweep + retuned cluster labels are built, (3) the continuous-tracks npz and its helpers exist and pass unit tests, (4) `initiator_idx`/`fleer_idx` are populated, and (5) every self-check gold value is pinned from the real bundle.