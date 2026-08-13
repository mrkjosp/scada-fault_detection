"""
CWRU bearing dataset -> motor health classifier (real-data path).

Trains on the Case Western Reserve University (CWRU) Bearing Data Set --
real accelerometer recordings of bearings with seeded inner-race /
outer-race / ball faults. Data is not shipped (licensing + size); download
the .mat files yourself into data/cwru/ (see docs/CWRU.md).

Pipeline:
  .mat drive-end signal -> windows -> time-domain features (RMS, kurtosis,
  crest factor, ...) -> RandomForest.

Run:
    python -m training.cwru_motor --data-dir data/cwru --out-dir models --plot
"""
import argparse
import glob
import os
import re

import numpy as np


# Bearing geometry for the CWRU drive-end 6205-2RS JEM SKF bearing, used to
# compute characteristic defect frequencies for envelope features.
CWRU_DE_BEARING = {
    "n_balls": 9,
    "d_ball_in": 0.3126,
    "d_pitch_in": 1.537,
    "contact_angle_deg": 0.0,
}
CWRU_SAMPLE_RATE = 12000     # Hz (12k drive-end set; 48k set also exists)
WINDOW_SAMPLES = 2048
HOP_SAMPLES = 1024           # 50% overlap -- see load_cwru() for the split implications


def defect_frequencies(rpm: float, bearing=CWRU_DE_BEARING) -> dict:
    """BPFO/BPFI/BSF in Hz for a given shaft speed (docs/PHYSICS.md §1.3)."""
    fr = rpm / 60.0
    n = bearing["n_balls"]
    ratio = bearing["d_ball_in"] / bearing["d_pitch_in"]
    cos_phi = np.cos(np.deg2rad(bearing["contact_angle_deg"]))
    bpfo = (n / 2) * fr * (1 - ratio * cos_phi)
    bpfi = (n / 2) * fr * (1 + ratio * cos_phi)
    bsf = (bearing["d_pitch_in"] / (2 * bearing["d_ball_in"])) * fr * \
          (1 - (ratio * cos_phi) ** 2)
    return {"f_shaft": fr, "BPFO": bpfo, "BPFI": bpfi, "BSF": bsf}


def _window_features(sig: np.ndarray, fs: int, win: int = WINDOW_SAMPLES,
                      hop: int = HOP_SAMPLES):
    """Time-domain features per window.

    Uses range(0, len(sig)-win+1, hop) rather than range(0, len(sig)-win, hop)
    -- the latter drops the final legitimate window whenever len(sig)-win is
    an exact multiple of hop (including len(sig)==win, where it would return
    empty instead of one row).
    """
    feats = []
    for start in range(0, len(sig) - win + 1, hop):
        w = sig[start:start + win]
        rms = np.sqrt(np.mean(w ** 2))
        mean = w.mean()
        std = w.std()
        kurt = (np.mean((w - mean) ** 4) / (std ** 4)) if std > 1e-12 else 3.0
        crest = (np.max(np.abs(w)) / rms) if rms > 1e-12 else 0.0
        peak = np.max(np.abs(w))
        feats.append([rms, kurt, crest, peak, std])
    return np.array(feats)


FEATURE_NAMES = ["rms", "kurtosis", "crest_factor", "peak", "std"]


def load_cwru(data_dir: str, test_frac: float = 0.2, gap_windows: int = 2):
    """Load CWRU .mat files and split into train/test PER FILE, temporally.

    A random split over pooled overlapping windows (hop=1024 < win=2048)
    can put window i in train and its 50%-overlapping neighbor i+1 in test
    -- real leakage, not theoretical.

    Fix: per recording, the first (1-test_frac) of the timeline is train and
    the last test_frac is test. train_sig = sig[:A] and test_sig = sig[B:]
    with B >= A are disjoint slices, so no window from one can share a
    sample with a window from the other regardless of gap size. The
    `gap_windows` gap is for a different reason: it adds temporal distance
    so the boundary windows aren't near-duplicates in *content* (vibration
    is autocorrelated over short spans).

    Every file contributes to both train and test, but no signal does. For
    a stronger test (generalizing to an unseen operating condition), hold an
    entire load (CWRU has 0/1/2/3 HP) out of training instead -- see
    run_crossload().

    Label is inferred from the filename prefix: normal_*.mat / ir_*.mat /
    or_*.mat / ball_*.mat. Returns (X_tr, y_tr, X_te, y_te). Exits rather
    than fabricating results if no data is present.
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        raise SystemExit("Install scipy first:  pip install scipy")

    paths = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
    if not paths:
        raise SystemExit(
            f"No .mat files in {data_dir}/.\n"
            "Download the CWRU Bearing Data Set and name files by fault class:\n"
            "  normal_*.mat  ir_*.mat  or_*.mat  ball_*.mat\n"
            "See docs/CWRU.md. This script will not fabricate data."
        )

    label_map = {"normal": "normal", "ir": "inner_race",
                 "or": "outer_race", "ball": "ball"}
    X_tr, y_tr, X_te, y_te = [], [], [], []
    skipped = []
    for p in paths:
        prefix = os.path.basename(p).split("_")[0].lower()
        label = label_map.get(prefix)
        if label is None:
            skipped.append(os.path.basename(p))
            print(f"  skip (unknown class prefix {prefix!r}): {os.path.basename(p)}")
            continue
        mat = loadmat(p)
        de_keys = [k for k in mat if k.endswith("_DE_time")]
        if not de_keys:
            skipped.append(os.path.basename(p))
            print(f"  skip (no _DE_time key): {os.path.basename(p)}")
            continue
        sig = np.asarray(mat[de_keys[0]]).ravel().astype(float)

        n_samples = len(sig)
        split_sample = int(n_samples * (1 - test_frac))
        gap_samples = gap_windows * HOP_SAMPLES

        train_sig = sig[:max(0, split_sample - gap_samples)]
        test_sig = sig[split_sample + gap_samples:]

        wf_tr = _window_features(train_sig, CWRU_SAMPLE_RATE)
        wf_te = _window_features(test_sig, CWRU_SAMPLE_RATE)
        if len(wf_tr) == 0 or len(wf_te) == 0:
            skipped.append(os.path.basename(p))
            print(f"  skip (recording too short to split): {os.path.basename(p)}")
            continue

        X_tr.append(wf_tr); y_tr += [label] * len(wf_tr)
        X_te.append(wf_te); y_te += [label] * len(wf_te)
        print(f"  {os.path.basename(p)}: {len(wf_tr)} train / {len(wf_te)} test "
              f"windows -> {label}")

    if not X_tr:
        raise SystemExit("Found .mat files but none usable; check naming/keys.")

    # A misnamed file (e.g. "IR_007_0hp.mat") falls into the "unknown class
    # prefix" skip above and looks like it worked -- surface it as a summary
    # rather than leaving it buried in per-file logs.
    found_labels = set(y_tr)
    missing = set(label_map.values()) - found_labels
    if missing:
        print(f"  [!] WARNING: no usable files found for class(es) {sorted(missing)} "
              f"-- check filenames against docs/CWRU.md's naming convention. "
              f"Training will proceed with only {sorted(found_labels)}.")
    if skipped:
        print(f"  [!] {len(skipped)} file(s) skipped entirely: {skipped}")

    return (np.vstack(X_tr), np.array(y_tr), np.vstack(X_te), np.array(y_te))


def main(data_dir: str, out_dir: str, make_plot: bool):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report

    os.makedirs(out_dir, exist_ok=True)
    print("Loading CWRU data (per-file temporal train/test split)...")
    X_tr, y_tr, X_te, y_te = load_cwru(data_dir)
    print(f"  train windows: {len(X_tr)}  test windows: {len(X_te)}  "
          f"classes: {sorted(set(y_tr))}")

    clf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                 class_weight="balanced", random_state=42,
                                 n_jobs=-1)
    clf.fit(X_tr, y_tr)
    print(classification_report(y_te, clf.predict(X_te), zero_division=0))

    import joblib
    out = os.path.join(out_dir, "motor_health_cwru_model.joblib")
    joblib.dump(clf, out)
    print(f"saved -> {out}")
    print("Reminder: report the REAL accuracy you measured (~90s%), not 99%.")

    if make_plot:
        _plot_signals(data_dir, out_dir)


def _plot_signals(data_dir: str, out_dir: str):
    """Plot one window per fault class -- read the impulsiveness off the plot."""
    try:
        from scipy.io import loadmat
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("plotting needs scipy + matplotlib; skipping")
        return
    classes = {"normal": "normal", "ir": "inner_race",
               "or": "outer_race", "ball": "ball"}
    fig, axes = plt.subplots(len(classes), 1, figsize=(9, 8), sharex=True)
    for ax, (prefix, label) in zip(axes, classes.items()):
        hits = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_*.mat")))
        if not hits:
            ax.set_title(f"{label}: no file"); continue
        mat = loadmat(hits[0])
        de = [k for k in mat if k.endswith("_DE_time")]
        if not de:
            continue
        sig = np.asarray(mat[de[0]]).ravel()[:2048]
        ax.plot(sig, linewidth=0.5)
        ax.set_title(f"{label} (drive-end vibration)")
    fig.tight_layout()
    path = os.path.join(out_dir, "cwru_signals.png")
    fig.savefig(path, dpi=110)
    print(f"saved signal plot -> {path}")


CANONICAL_RPM_BY_HP = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

# Verified against CWRU's own published tables (Normal Baseline +
# 12k Drive End Bearing Fault Data). This is the "Normal + 12k Drive-End"
# subset most published CWRU work means by default -- 64 files: 4 normal +
# 4 loads x (IR + Ball at 007/014/021/028" + Outer race at 007" x3
# positions + Outer race at 014"/021" x1 position each).
#
# Deliberately excludes the 48k Drive-End / 12k Fan-End tables even though
# some mirrors bundle everything into one flat folder by bare file number --
# those use a different sample rate/sensor, and CWRU_SAMPLE_RATE is
# hardcoded to 12000 here. crossload_discover() reports those file numbers
# as "out of scope" rather than silently corrupting features for them.
CWRU_12K_DE_FILE_LABEL = {}
CWRU_12K_DE_FILE_HP = {}


def _register_block(numbers, label):
    for n, hp in zip(numbers, (0, 1, 2, 3)):
        CWRU_12K_DE_FILE_LABEL[n] = label
        CWRU_12K_DE_FILE_HP[n] = hp


_register_block([97, 98, 99, 100], "normal")
for _nums in ([105, 106, 107, 108], [169, 170, 171, 172],
              [209, 210, 211, 212], [3001, 3002, 3003, 3004]):
    _register_block(_nums, "inner_race")
for _nums in ([118, 119, 120, 121], [185, 186, 187, 188],
              [222, 223, 224, 225], [3005, 3006, 3007, 3008]):
    _register_block(_nums, "ball")
for _nums in ([130, 131, 132, 133], [144, 145, 146, 147], [156, 158, 159, 160],
              [197, 198, 199, 200], [234, 235, 236, 237],
              [246, 247, 248, 249], [258, 259, 260, 261]):
    _register_block(_nums, "outer_race")

assert len(CWRU_12K_DE_FILE_LABEL) == 64, \
    f"expected 64 registered CWRU 12k-DE + Normal files, got {len(CWRU_12K_DE_FILE_LABEL)}"

# File-number ranges known to belong to the 48k Drive-End or 12k Fan-End
# tables -- reported as out-of-scope, not UNRESOLVED. Not exhaustive, just
# the ranges that show up in this kind of "full dataset" mirror.
_KNOWN_OUT_OF_SCOPE = set(range(109, 165)) | set(range(174, 266)) | set(range(270, 319))
_KNOWN_OUT_OF_SCOPE -= set(CWRU_12K_DE_FILE_LABEL)

FAULT_KEYWORDS = {
    "normal": ["normal", "baseline"],
    "inner_race": ["innerrace", "inner_race", "ir007", "ir014", "ir021", "ir028"],
    "outer_race": ["outerrace", "outer_race", "or007", "or014", "or021", "or028"],
    "ball": ["ball", "b007", "b014", "b021", "b028"],
}

_HP_PATH_RE = re.compile(r"(?:^|[_/\\ ])load[_ ]?([0-3])(?:[_/\\ ]|$)|([0-3])\s*hp")
_BARE_ID_RE = re.compile(r"(?:^|[/\\])0*(\d+)\.mat$")


def _label_from_path(low: str):
    for lbl, kws in FAULT_KEYWORDS.items():
        if any(kw in low for kw in kws):
            return lbl
    return None


def _hp_from_path(low: str):
    m = _HP_PATH_RE.search(low)
    if not m:
        return None
    g = m.group(1) or m.group(2)
    return int(g) if g is not None else None


def _hp_from_rpm(mat: dict):
    rpm_keys = [k for k in mat if k.upper().endswith("RPM")]
    if not rpm_keys:
        return None, None
    try:
        rpm = float(np.asarray(mat[rpm_keys[0]]).ravel()[0])
    except Exception:
        return None, None
    hp = min(CANONICAL_RPM_BY_HP, key=lambda k: abs(CANONICAL_RPM_BY_HP[k] - rpm))
    return hp, rpm


def crossload_discover(data_dir: str):
    """Recursively find every .mat file under data_dir, resolve a (fault
    label, HP load) for each, and print one line per file so the mapping is
    auditable before it's trusted.

    Resolution order:
      1. Bare CWRU file ID (e.g. "105.mat") looked up in
         CWRU_12K_DE_FILE_LABEL/_HP -- verified against case.edu directly.
         Primary path for mirrors that keep original numeric filenames.
      2. If the ID is in _KNOWN_OUT_OF_SCOPE (48k DE / 12k FE), report as
         out-of-scope and exclude -- a deliberate scope decision, not a
         resolution failure.
      3. Otherwise fall back to path-keyword + RPM heuristics, for mirrors
         that rename files descriptively. RPM is bucketed to the nearest of
         the 4 canonical rig speeds; if keyword-label and RPM-hp disagree,
         flagged UNRESOLVED rather than silently picking one.

    Returns a list of {path, label, hp, sig} dicts, only for files that
    resolved cleanly.
    """
    from scipy.io import loadmat

    paths = sorted(glob.glob(os.path.join(data_dir, "**", "*.mat"), recursive=True))
    if not paths:
        raise SystemExit(
            f"No .mat files found under {data_dir}/ (recursive search). "
            f"Check the path -- if you used kagglehub.dataset_download(), "
            f"pass its return value straight in as --crossload-dir."
        )

    print(f"Found {len(paths)} .mat file(s) under {data_dir}/\n")
    records = []
    n_out_of_scope = 0
    for p in paths:
        rel = os.path.relpath(p, data_dir)
        low = rel.lower().replace(" ", "").replace("-", "_")

        mat = loadmat(p)
        de_keys = [k for k in mat if k.endswith("_DE_time")]
        if not de_keys:
            print(f"  SKIP        {rel:60s} no _DE_time key in file")
            continue
        sig = np.asarray(mat[de_keys[0]]).ravel().astype(float)

        id_match = _BARE_ID_RE.search(rel)
        file_id = int(id_match.group(1)) if id_match else None

        if file_id is not None and file_id in CWRU_12K_DE_FILE_LABEL:
            label = CWRU_12K_DE_FILE_LABEL[file_id]
            hp = CWRU_12K_DE_FILE_HP[file_id]
            _, rpm = _hp_from_rpm(mat)
            rpm_str = f"{rpm:.1f}rpm" if rpm is not None else "no-RPM-key"
            print(f"  ok          {rel:60s} id={file_id} label={label} hp={hp} "
                  f"(12k-DE table, {rpm_str}) n_samples={len(sig)}")
            records.append({"path": p, "label": label, "hp": hp, "sig": sig})
            continue

        if file_id is not None and file_id in _KNOWN_OUT_OF_SCOPE:
            print(f"  OUT-OF-SCOPE {rel:59s} id={file_id} "
                  f"(48k Drive-End or 12k Fan-End -- excluded, see module docstring)")
            n_out_of_scope += 1
            continue

        label = _label_from_path(low)
        hp_rpm, rpm = _hp_from_rpm(mat)
        hp_path = _hp_from_path(low)

        if hp_rpm is not None and hp_path is not None and hp_rpm != hp_path:
            print(f"  UNRESOLVED  {rel:60s} label={label} "
                  f"RPM says hp={hp_rpm} ({rpm:.1f} rpm) but path says hp={hp_path} "
                  f"-- disagreement, not guessing, fix FAULT_KEYWORDS/regex or rename")
            continue
        hp = hp_rpm if hp_rpm is not None else hp_path

        ok = label is not None and hp is not None
        tag = "ok" if ok else "UNRESOLVED"
        rpm_str = f"{rpm:.1f}rpm" if rpm is not None else "no-RPM-key"
        print(f"  {tag:11s} {rel:60s} label={label} hp={hp} ({rpm_str}) "
              f"n_samples={len(sig)}")
        if not ok:
            continue

        records.append({"path": p, "label": label, "hp": hp, "sig": sig})

    if n_out_of_scope:
        print(f"\n({n_out_of_scope} file(s) excluded as out-of-scope: 48k Drive-End "
              f"/ 12k Fan-End -- see module docstring)")

    print(f"\nResolved {len(records)}/{len(paths)} files.")
    found_labels = {r["label"] for r in records}
    missing = set(FAULT_KEYWORDS) - found_labels
    if missing:
        print(f"[!] WARNING: no resolved files for class(es) {sorted(missing)}. "
              f"Check the UNRESOLVED lines above.")
    found_hps = {r["hp"] for r in records}
    missing_hp = set(CANONICAL_RPM_BY_HP) - found_hps
    if missing_hp:
        print(f"[!] WARNING: no resolved files for HP load(s) {sorted(missing_hp)}.")
    return records


def crossload_sanity_check(train_records, test_records, test_hp):
    """Hard-fail on the two ways this split could quietly leak: the same
    file on both sides, or a train file actually belonging to the held-out
    HP bucket."""
    train_paths = {r["path"] for r in train_records}
    test_paths = {r["path"] for r in test_records}
    overlap = train_paths & test_paths
    if overlap:
        raise SystemExit(f"LEAK: {len(overlap)} file(s) in both train and test: "
                          f"{sorted(overlap)[:5]}")
    if any(r["hp"] == test_hp for r in train_records):
        raise SystemExit(f"LEAK: a train record is tagged hp=={test_hp} "
                          f"(the held-out load) -- split logic is wrong.")
    if any(r["hp"] != test_hp for r in test_records):
        raise SystemExit(f"LEAK: a test record is NOT tagged hp=={test_hp}.")
    train_hp = sorted({r["hp"] for r in train_records})
    print(f"Sanity check passed: {len(train_paths)} train file(s) (HP {train_hp}), "
          f"{len(test_paths)} test file(s) (HP {test_hp}), no file or HP overlap.")


def crossload_build_xy(records):
    X, y = [], []
    for r in records:
        feats = _window_features(r["sig"], CWRU_SAMPLE_RATE)
        if len(feats) == 0:
            print(f"  [!] {os.path.basename(r['path'])}: too short to window, skipping")
            continue
        X.append(feats)
        y += [r["label"]] * len(feats)
    if not X:
        raise SystemExit("No windows built -- every file was too short?")
    return np.vstack(X), np.array(y)


def run_crossload(data_dir: str, test_hp: int, out_dir: str):
    """CROSS-LOAD generalization test: train on HP loads other than
    test_hp, test on test_hp entirely -- a genuinely unseen operating
    condition, not just a later time-slice of a recording it partly trained
    on (that's what main()/load_cwru() do).

    The split here is by whole file, so the hop-overlap leak load_cwru()
    guards against can't cross the train/test boundary -- that boundary is
    between separate files. A clean result here isn't automatically
    suspicious the way a naive pooled-random split's would be, but a
    labeling bug that puts even one test_hp file in train would still
    quietly inflate the number, which is what crossload_sanity_check
    catches.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    import joblib

    os.makedirs(out_dir, exist_ok=True)
    records = crossload_discover(data_dir)

    train_records = [r for r in records if r["hp"] != test_hp]
    test_records = [r for r in records if r["hp"] == test_hp]
    if not train_records or not test_records:
        raise SystemExit(
            f"Need resolved files on both sides of the split: got "
            f"{len(train_records)} train / {len(test_records)} test for "
            f"test_hp={test_hp}. Check the crossload_discover() printout "
            f"above -- probably some files are UNRESOLVED that shouldn't be."
        )

    crossload_sanity_check(train_records, test_records, test_hp)

    print("\nBuilding features...")
    X_tr, y_tr = crossload_build_xy(train_records)
    X_te, y_te = crossload_build_xy(test_records)
    print(f"train windows: {len(X_tr)} (HP != {test_hp}, classes {sorted(set(y_tr))})")
    print(f"test windows:  {len(X_te)} (HP == {test_hp}, classes {sorted(set(y_te))})")

    clf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                  class_weight="balanced", random_state=42,
                                  n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    print(f"\nClassification report on UNSEEN {test_hp} HP load "
          f"(cross-load, whole-file split -- see run_crossload() docstring):")
    print(classification_report(y_te, y_pred, zero_division=0))

    out = os.path.join(out_dir, f"motor_health_cwru_crossload_{test_hp}hp_model.joblib")
    joblib.dump(clf, out)
    print(f"saved -> {out}")
    return clf


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/cwru",
                    help="within-file test: expects normal_*.mat / ir_*.mat / "
                         "or_*.mat / ball_*.mat")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--skip-within-file", action="store_true",
                    help="skip the within-file (temporal-split) test")
    p.add_argument("--crossload-dir", default=None,
                    help="root dir to recursively search for .mat files for "
                         "the CROSS-LOAD test (auto-discovers layout/labels "
                         "-- see crossload_discover()). Omit to skip this test.")
    p.add_argument("--test-hp", type=int, default=3, choices=[0, 1, 2, 3],
                    help="HP load to hold out entirely for the cross-load test")
    args = p.parse_args()

    if not args.skip_within_file:
        main(args.data_dir, args.out_dir, args.plot)

    if args.crossload_dir:
        run_crossload(args.crossload_dir, args.test_hp, args.out_dir)
