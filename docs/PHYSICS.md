# Physics of the Machine Simulators

This documents the governing equations behind `simulator/generate_data.py`.
Depth is matched to scope: the **motor** is
the deep machine (real CWRU bearing data + edge deployment); **pump** and
**compressor** are physics-simulated supporting machines and are documented
concisely.

> The synthetic data is generated from the simplified
> models below. They are good enough to separate fault classes and to exercise
> the full edge pipeline, but they are *not* a substitute for field data. The
> motor is additionally validated on the real CWRU bearing dataset
> (see `docs/CWRU.md` / `training/cwru_motor.py`).

> **Note on sources.** The equations below follow standard, widely-taught
> treatments (induction-machine circuit theory, rolling-element bearing
> diagnostics, centrifugal-pump affinity laws, isentropic compression) rather
> than one specific paper. The named references are where these are commonly
> found.


## 1. Induction Motor (deep machine)

### 1.1 Stator current vs. load
Three-phase real power relates current and load through the power-factor and
efficiency:

    P_out = √3 · V · I · cos φ · η
⇒  I = P_out / (√3 · V · cos φ · η)

The simulator uses an approximately linear no-load→rated interpolation (a
simplified Heyland-circle behaviour), which is accurate across the normal
operating band and steepens under overload (rising slip and magnetic
saturation):

    I(load) = I_no_load + (I_rated − I_no_load) · (load_pct / 100)

Sanity check at rated load (nameplate 7.5 kW, 400 V, η≈0.891, cos φ≈0.84):
I = 7500 / (√3 · 400 · 0.84 · 0.891) ≈ **14.5 A** — within ~5% of the
I_RATED = 15.2 A the simulator actually uses, not an exact match (the
idealized formula omits stray-load losses and saturation margin a real
nameplate current already includes). `compute_current()` is anchored to
I_RATED directly, not to this formula.

### 1.2 Bearing/winding temperature (thermal lag)
Heat comes from copper loss I²R; the frame stores and sheds it per Newton's law
of cooling with a single thermal time constant τ:

    T(t) = T_ambient + (I² · R_stator · θ_thermal) · (1 − e^(−t/τ))

with θ_thermal ≈ 0.8 °C/W and τ ≈ 1200 s. The key qualitative consequence used
by the model: **temperature cannot step**; its maximum rate is
≈ (T_steady − T_now)/τ per second. This is *why* `temp_rate_of_change` is a
useful feature — a developing fault shows as a slow, bounded climb, not a jump.

### 1.3 Vibration and the bearing-fault signature
The vibration signal is a sum of a broadband noise floor, a 1× shaft-speed
imbalance term, and (under a bearing defect) an impulsive term at a bearing
characteristic frequency, amplitude-modulated at shaft speed:

    v(t) = A_base·noise
         + A_imbalance·sin(2π f_shaft t)
         + A_bearing·sin(2π f_BPFO t)·modulation

- **Imbalance** is sinusoidal → kurtosis stays ≈ 3 (near-Gaussian).
- **Bearing defect** is impulsive (sharp periodic strikes as a rolling element
  hits a spall) → kurtosis rises **above 3 before RMS moves much**. This is the
  physical justification for using kurtosis as an early, complementary feature
  to RMS.

**Bearing characteristic frequencies** (for the CWRU work). For a bearing with
n rolling elements, ball diameter d, pitch diameter D, contact angle φ, and
shaft frequency f_r:

    BPFO = (n/2)·f_r·(1 − (d/D)cos φ)      (outer-race defect)
    BPFI = (n/2)·f_r·(1 + (d/D)cos φ)      (inner-race defect)
    BSF  = (D/2d)·f_r·(1 − ((d/D)cos φ)²)  (ball/rolling-element defect)

These let you predict *where* the spectral energy of a seeded fault should
appear in the CWRU signals — read it off your own FFT/envelope plot rather than
trusting the formula blindly.

**References:**
- Induction-machine current/power: standard electric-machinery texts, e.g.
  Fitzgerald, *Electric Machinery*.
- Bearing fault frequencies & envelope analysis: Randall & Antoni, "Rolling
  Element Bearing Diagnostics — A Tutorial," *Mechanical Systems and Signal
  Processing* (2011).
- ISO 10816-3 vibration severity zones: the A/B/C/D thresholds in
  `config.settings.MACHINES` should be checked against the standard for your
  specific machine class/size before being quoted as compliant.

---

## 2. Centrifugal Pump (supporting, concise)

### 2.1 Affinity laws
For a centrifugal pump at speed N (the reason load maps cleanly to current):

    Q ∝ N        (flow)
    H ∝ N²       (head)
    P ∝ N³       (shaft power)

The simulator computes current from hydraulic power demand:

    P_hydraulic = ρ·g·Q·H,   P_shaft = P_hydraulic / η_pump
    I = I_no_load + (P_shaft / P_rated)·(I_rated − I_no_load)

### 2.2 Fault signatures (qualitative)
- **Cavitation:** broadband, high-frequency vibration as vapour bubbles collapse
  → large rise in the noise floor (RMS up, broadband).
- **Impeller damage:** stronger vane-pass-frequency component.
- **Dry running:** loss of liquid cooling → rapid temperature rise (the model
  adds tens of °C by fault stage).

**Reference:** centrifugal-pump affinity laws and cavitation behaviour — standard
turbomachinery/pump-handbook material, e.g. Karassik et al., *Pump Handbook*.

---

## 3. Reciprocating / Screw Compressor (supporting, concise)

### 3.1 Isentropic compression power
Shaft power scales with the isentropic work factor for pressure ratio P₂/P₁ and
ratio of specific heats γ:

    P = (P₁·Q·γ/(γ−1))·((P₂/P₁)^((γ−1)/γ) − 1) / η_isentropic

Loaded current ≈ 35–42 A; unloaded (venting) ≈ 10–12 A. Valve leakage requires
extra power for the same delivered air (+10–20%).

### 3.2 Temperature
Discharge temperature is governed by the compression ratio and after-cooler
effectiveness:

    T_discharge = T_ambient + (T_theoretical − T_ambient)·(1 − cooler_efficiency)

A healthy after-cooler (90% efficient) holds discharge near ~49 °C at 25 °C
ambient (25 + (267−25)×0.10 ≈ 49.2 °C); a fouled/blocked cooler lets it climb
toward the theoretical adiabatic discharge temperature (~267 °C).

### 3.3 Fault signatures (qualitative)
- **Valve leakage:** new harmonic content at 3× shaft frequency (a simplified
  single added harmonic, not a full harmonic series) and higher discharge temp.
- **Bearing wear:** impulsive vibration (same kurtosis logic as the motor).
- **Overheating:** rising discharge temperature with thermal lag.

**Reference:** isentropic compression / adiabatic temperature rise — standard
thermodynamics texts, e.g. Cengel & Boles, *Thermodynamics: An Engineering
Approach*. Compressor-specific fault diagnostics are qualitative here and not
tied to a single source.

---

## How the physics maps to the 12 features
| feature | physical reason |
|---------|-----------------|
| current_A, current_normalized | load proxy; deviation from rated flags overload |
| bearing_temp_C, temp_margin | thermal state; margin to limit |
| vibration_rms_mm_s | overall mechanical severity (ISO 10816-3) |
| vibration_kurtosis | **impulsiveness** → early bearing-defect discriminator |
| rolling_mean_current_30s | smoothed load trend |
| rolling_std_vibration_30s | vibration variability / instability |
| temp_rate_of_change | developing thermal fault (bounded by τ) |
| current_rate_of_change | sudden electrical change |
| load_pct, ambient_temp | operating-point context (separates fault from duty) |