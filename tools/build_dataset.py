#!/usr/bin/env python
"""tools/build_dataset.py — build the small bundled teaching dataset from the lab pipeline.

This is the ONLY script that touches the real lab data on /snlkt. Students never run it — the
outputs it writes to ../data/ are committed to the repo. It exists for provenance/reproducibility.

Run once with a python that has numpy/pandas/cv2 + the despotism approach_behavior utils on path
(conda base works):

    /home/itang/miniconda3/bin/python tools/build_dataset.py

Source: two food-deprivation cohorts (20260222 and 12192025), each {pre,dep,post}, so `condition`
(pre / dep(rivation) / post) is a real independent variable AND `sex` gets genuine cross-cohort
replication (cage numbers collide across cohorts, so cohort is tracked explicitly — see
cohort_meta.csv and build_derived.py). Held-out cage = camera 16 (present ONLY in 20260222;
leave-one-cage-out for the inference notebook). Per event we keep only a short keypoint window
(world coords, mice ordered [approacher, approachee, bystander]) + its per-mouse ranks + condition +
registry label. No video.

It also writes data/_scratch/slp_todo.json listing a few (cohort, stem, frame) to trim into tiny
.slp files for the 'load SLEAP' notebook — run tools/trim_slp.py (sleap-io env) afterward.
"""
import os, sys, json
import numpy as np
import pandas as pd

ID_DIR = "/home/itang/notebooks-kay/despotism/hcm"
sys.path.insert(0, ID_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "course"))
from utils import approach_behavior as ab   # noqa: E402
import course_utils as cu                    # noqa: E402  (for the feature-signal sanity check)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
SCRATCH = os.path.join(DATA, "_scratch")
COHORTS = ["20260222_pre", "20260222_dep", "20260222_post",
           "12192025_pre", "12192025_dep", "12192025_post"]
# One cohort_meta CSV per cohort DATE. Cage numbers (9,10,...) COLLIDE across cohorts and even carry
# different sexes, so we never key by bare cage — the combined data/cohort_meta.csv adds a `cohort`
# column and build_derived.py looks sex up by (cohort, cage).
META_CSVS = {
    "20260222": "/snlkt/isaac/id_switch/despotism/cohort_meta/hcm_mouse_cohort_meta_20260222.csv",
    "12192025": "/snlkt/isaac/id_switch/despotism/cohort_meta/hcm_mouse_cohort_meta_12192025.csv",
}
HELDOUT_CAM = "16"                   # only 20260222 has cam16 -> held-out set is single-cohort
PAD_BEFORE, WINDOW = 40, 90          # T = 130 frames (~2.6 s @ 50 fps)
T = PAD_BEFORE + WINDOW
SEED = 7
rng = np.random.RandomState(SEED)

# per-set caps — raised for the 2-cohort bundle so both cohorts and both sexes are well represented
# (stratified() spreads by cohort). Kept so the committed bundle stays within the 75 MB budget.
TRAIN_CAPS = dict(aggression=800, mlp_fp=500, other=400, background=800)
HELD_CAPS = dict(aggression=300, mlp_fp=130, other=100, background=250)
OTHER_CATS = ["bystander_ledge", "anogenital", "side_kissing", "grooming", "mounting",
              "tail_bite", "double_ledge"]


def cam_of(stem):
    return stem.split(".")[1] if "." in stem else "?"


def condition_of(cohort):
    return cohort.split("_")[-1]


def load_registry():
    df = pd.read_csv(ab.REGISTRY_CSV, dtype=str)
    df = df[df["cohort"].isin(COHORTS)].copy()
    df["cam"] = df["stem"].map(cam_of)
    return df


def load_background():
    df = pd.read_csv(ab.EVENTS_CSV)
    df = df[df["cohort"].isin(COHORTS)].copy()
    df["cam"] = df["stem"].map(cam_of)
    df["event_key"] = (df["cohort"].astype(str) + "|" + df["stem"].astype(str) + "|"
                       + df["pair"].astype(str) + "|" + df["contact_start"].astype(str))
    if "contact_s" in df.columns:
        df = df[df["contact_s"] >= 1.0]
    return df


def write_cohort_meta(path):
    """Read every cohort's meta CSV, tag it with a `cohort` (date) column, align columns (the extra
    Rank_postbaseline that only 20260222 carries is NaN-filled for the others), concat, and write the
    combined table. `cohort` + `Cage` together identify a physical cage; sex must be looked up by that
    pair, never by bare Cage (cage numbers collide and even flip sex across cohorts)."""
    frames = []
    for date, csv in META_CSVS.items():
        m = pd.read_csv(csv)
        m.insert(0, "cohort", date)
        frames.append(m)
    combined = pd.concat(frames, ignore_index=True, sort=False)   # union of cols, NaN-fill missing
    combined.to_csv(path, index=False)
    return combined


def stratified(df, n, by="cohort"):
    """Sample n rows spread across the `by` groups as evenly as available."""
    if len(df) <= n:
        return df
    groups = list(df.groupby(by))
    per = max(1, n // len(groups))
    picks = [g.sample(min(per, len(g)), random_state=rng) for _, g in groups]
    out = pd.concat(picks)
    if len(out) < n:                                   # top up randomly from the remainder
        rest = df.drop(out.index)
        out = pd.concat([out, rest.sample(min(n - len(out), len(rest)), random_state=rng)])
    return out.head(n)


def select_split(reg, bg, caps, cam_keep):
    """Return a DataFrame of selected events (cols: cohort, stem, pair, contact_start, event_key,
    category) for the given camera-keep predicate. Registry rows carry a category; background rows
    are category '' (non-aggression by construction — they are un-docketed approaches)."""
    reg = reg[reg["cam"].map(cam_keep)]
    bg = bg[bg["cam"].map(cam_keep)]
    reg_keys = set(reg["event_key"]) if "event_key" in reg else set(
        reg["cohort"] + "|" + reg["stem"] + "|" + reg["pair"] + "|" + reg["contact_start"])
    parts = []
    agg = reg[reg["category"] == "aggression"]
    parts.append(stratified(agg, caps["aggression"]))
    fp = reg[reg["category"] == "mlp_fp"]
    parts.append(stratified(fp, caps["mlp_fp"]))
    oth = reg[reg["category"].isin(OTHER_CATS)]
    parts.append(stratified(oth, caps["other"], by="category"))
    bg_keys = set(bg["event_key"])
    bg_free = bg[~bg["event_key"].isin(reg_keys)]
    parts.append(stratified(bg_free, caps["background"]))
    sel = pd.concat(parts).drop_duplicates("event_key")
    cols = ["cohort", "stem", "pair", "contact_start", "event_key"]
    sel = sel.reindex(columns=cols + ["category"])
    sel["category"] = sel["category"].fillna("")
    return sel.reset_index(drop=True)


def extract(sel):
    """Group by (cohort, stem) so each 30-min tracks_matrix is loaded once. For each event compute
    the corrected [appr, appe, by] order, per-mouse ranks, and the keypoint window."""
    recs = []
    sel = sel.copy()
    sel["contact_start"] = sel["contact_start"].astype(int)
    for (coh, stem), grp in sel.groupby(["cohort", "stem"]):
        tmpath = f"{ab.TM_BASE}/{coh}/{stem}_tracks_matrix.npz"
        if not os.path.exists(tmpath):
            continue
        try:
            tm = np.load(tmpath)["tracks_matrix"]          # (frames, 15, 2, 3)
        except Exception:
            continue
        nframes = tm.shape[0]
        for _, r in grp.iterrows():
            cs = int(r["contact_start"])
            if cs - PAD_BEFORE < 0 or cs + WINDOW > nframes:
                continue
            try:
                appr, appe, by = ab.assign_event(tm, cs, r["pair"])
            except Exception:
                continue
            slot_ranks = ab.event_track_ranks(coh, stem, cs)   # [r_slot0, r_slot1, r_slot2] or None
            if slot_ranks is None:
                ranks = [0, 0, 0]
            else:
                ranks = [int(slot_ranks[appr]), int(slot_ranks[appe]), int(slot_ranks[by])]
            win = tm[cs - PAD_BEFORE:cs + WINDOW]           # (T,15,2,3)
            kp = np.transpose(win, (0, 3, 1, 2))[:, [appr, appe, by]]   # (T,3,15,2)
            cat = r["category"] if isinstance(r["category"], str) else ""
            recs.append(dict(kp=kp.astype(np.float16), ranks=ranks, condition=condition_of(coh),
                             contact_rel=PAD_BEFORE, event_key=r["event_key"], category=cat,
                             agg_label=int(cat == "aggression")))
    return recs


def to_npz(recs, path):
    np.savez_compressed(
        path,
        kp=np.stack([r["kp"] for r in recs]),
        ranks=np.array([r["ranks"] for r in recs], dtype=np.int16),
        condition=np.array([r["condition"] for r in recs]),
        contact_rel=np.array([r["contact_rel"] for r in recs], dtype=np.int32),
        event_key=np.array([r["event_key"] for r in recs]),
        category=np.array([r["category"] for r in recs]),
        agg_label=np.array([r["agg_label"] for r in recs], dtype=np.int8),
    )
    mb = os.path.getsize(path) / 1e6
    print(f"  wrote {path}  ({len(recs)} events, {mb:.1f} MB)")


def sanity(recs, tag):
    """Quick check that the compact features carry aggression + rank signal (so the notebooks land)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    kp = np.stack([r["kp"].astype(np.float32) for r in recs])
    X = cu.features_batch(kp)
    y = np.array([r["agg_label"] for r in recs])
    Xz, _, _ = cu.standardize(X)
    print(f"  [{tag}] {len(recs)} events, {y.sum()} aggression; feature dim {X.shape[1]}")
    if y.sum() >= 10 and (y == 0).sum() >= 10:
        auc = cross_val_score(LogisticRegression(max_iter=1000), Xz, y, cv=4, scoring="roc_auc")
        print(f"  [{tag}] aggression-vs-rest logistic AUC = {auc.mean():.3f} +/- {auc.std():.3f}")


def main():
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(SCRATCH, exist_ok=True)
    reg, bg = load_registry(), load_background()
    print(f"registry rows {len(reg)}, background events {len(bg)}")

    train_sel = select_split(reg, bg, TRAIN_CAPS, lambda c: c != HELDOUT_CAM)
    held_sel = select_split(reg, bg, HELD_CAPS, lambda c: c == HELDOUT_CAM)
    print(f"selected train {len(train_sel)}, heldout(cam{HELDOUT_CAM}) {len(held_sel)}")

    train = extract(train_sel)
    held = extract(held_sel)
    print(f"extracted train {len(train)}, heldout {len(held)}")

    to_npz(train, os.path.join(DATA, "train_events.npz"))
    to_npz(held, os.path.join(DATA, "heldout_events.npz"))
    sanity(train, "train")
    sanity(held, "heldout")

    # answer key for the training set (used by the labeling notebook as a fallback / grading aid)
    pd.DataFrame([{"event_key": r["event_key"], "category": r["category"],
                   "agg_label": r["agg_label"]} for r in train]).to_csv(
        os.path.join(DATA, "answer_key.csv"), index=False)
    meta = write_cohort_meta(os.path.join(DATA, "cohort_meta.csv"))
    print(f"  wrote answer_key.csv, cohort_meta.csv ({len(meta)} rows, "
          f"cohorts={sorted(meta['cohort'].unique().tolist())})")

    # pick a few events (one aggression per condition + one heldout) to trim into demo .slp files
    todo = []
    for coh in COHORTS:
        cand = [r for r in train if r["category"] == "aggression" and
                r["event_key"].split("|")[0] == coh]
        if cand:
            ek = cand[0]["event_key"]
            _, stem, _, cs = ek.split("|")
            todo.append(dict(cohort=coh, stem=stem, contact_start=int(cs), tag=condition_of(coh)))
    hc = [r for r in held if r["category"] == "aggression"]
    if hc:
        _, stem, _, cs = hc[0]["event_key"].split("|")
        todo.append(dict(cohort=hc[0]["event_key"].split("|")[0], stem=stem,
                         contact_start=int(cs), tag="heldout"))
    json.dump(todo, open(os.path.join(SCRATCH, "slp_todo.json"), "w"), indent=2)
    print(f"  wrote _scratch/slp_todo.json ({len(todo)} clips) -> run tools/trim_slp.py next")


if __name__ == "__main__":
    main()
