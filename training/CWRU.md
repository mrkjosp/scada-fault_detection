# Using the CWRU Bearing Data Set (motor real-data path)

The motor is validated on **real** accelerometer data from the Case Western
Reserve University (CWRU) Bearing Data Center — the field-standard benchmark for
rolling-element bearing faults, recorded on a real test rig with seeded
inner-race, outer-race, and ball defects.

## 1. Download
Get the drive-end (DE) `.mat` files from the CWRU Bearing Data Center website
(public, free for research). Grab at least:
- one **Normal** baseline recording,
- one **Inner Race** fault,
- one **Outer Race** fault,
- one **Ball** fault,
(ideally at a single load/speed first, e.g. the 0 HP / 1797 rpm set).

> Verify the current download location and license yourself; the dataset has
> moved hosts over the years. Do not commit the `.mat` files.

## 2. Name and place the files
Drop them in `data/cwru/` named by class prefix so the loader can label them:

```
data/cwru/normal_0hp.mat
data/cwru/ir_007_0hp.mat     # inner race, 0.007" defect
data/cwru/or_007_0hp.mat     # outer race
data/cwru/ball_007_0hp.mat   # ball
```

`data/` is gitignored.

## 3. Train + plot
```bash
pip install scipy
python -m training.cwru_motor --data-dir data/cwru --out-dir models --plot
```
This writes `models/motor_health_cwru_model.joblib` and
`models/cwru_signals.png` (one window per class).

## 4. What to look at (the WHY of kurtosis)
Open `cwru_signals.png`. The **normal** trace is near-Gaussian noise; the
**inner/outer-race** traces show periodic **impulsive spikes** (the rolling
element striking the spall). Those impulses are exactly what pushes **kurtosis
above 3** before overall RMS moves much — read that off your own plot, in your
own words. Then relate the spike spacing to the BPFO/BPFI frequencies computed
in `training/cwru_motor.py:defect_frequencies()` (and `docs/PHYSICS.md §1.3`).

## 5. How the split actually works 
`training/cwru_motor.py` does **not** pool all windows and randomly shuffle
them into train/test — a 50%-overlap sliding window (`hop=1024` on a
`win=2048` window) means adjacent windows share half their samples, so a
random split puts near-duplicate windows on both sides of the boundary. That
was a real bug in an earlier version of this script, not a hypothetical risk.

The fix: for each recording, the first 80% of the timeline (by sample index,
not window index) is train and the last 20% is test. Because these are
disjoint slices (`sig[:A]` and `sig[B:]` with `B >= A`), no window built from
one can ever share a sample with a window from the other — that's already
guaranteed regardless of gap size. The extra gap of a few windows dropped at
the boundary exists for a different reason: it adds temporal distance so the
last train window and first test window aren't near-duplicates in *content*
(a rotating shaft's vibration is autocorrelated over short spans, so a
zero-gap boundary pair would still look unusually similar even without
sharing literal samples). Every file contributes to both train and test, but
no signal does. If you want a stronger test — generalizing to an unseen
*operating condition*, not just an unseen time segment — download more than
one load per class (CWRU provides 0/1/2/3 HP) and hold an entire load out of
training instead.

## 6. Honest reporting
Expect roughly **90-something %** accuracy, not 99%. If you still hit 99%
after this fix, the remaining suspects are: too few distinct recordings (only
one file per class means train and test are still the same physical bearing
and rig, just different time segments — real generalization needs multiple
recordings), or a `gap_windows` too small for the fault to look meaningfully
different across the boundary. Report the split method next to the number, not
just the number.
