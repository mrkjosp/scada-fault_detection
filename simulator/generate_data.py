"""
generate_data.py — physics-based training-data generator for all 3 machines.

Simulates an induction motor, a centrifugal pump, and a reciprocating
compressor from real datasheet constants, injects machine + sensor faults with
controlled distributions, and writes one CSV per machine to
training_data/{motor,pump,compressor}_training_data.csv (the schema
training/train_models.py expects).

Run:
    python -m simulator.generate_data --rows 50000
"""
import argparse
import os
from typing import Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd


class CircularBuffer:
    """
    Fixed-size ring buffer for vibration sampling (mirrors the C ring buffer in
    edge_firmware/src/edge_node.c). Used here to window raw synthetic vibration
    samples into the RMS/kurtosis features that become training labels, the
    same way the ESP32 firmware windows real ADC samples on-device.
    """

    def __init__(self, size: int = 64):
        self.size = size
        self.buffer = [0.0] * size
        self.write_ptr = 0
        self.count = 0

    def push(self, value: float):
        self.buffer[self.write_ptr] = value
        self.write_ptr = (self.write_ptr + 1) % self.size
        if self.count < self.size:
            self.count += 1

    def get_window(self) -> np.ndarray:
        if self.count < self.size:
            return np.array(self.buffer[:self.count])
        idx = [(self.write_ptr + i) % self.size for i in range(self.size)]
        return np.array([self.buffer[i] for i in idx])

    def rms(self) -> float:
        """v_rms = sqrt(mean(v**2)) — ISO 10816 vibration severity metric."""
        if self.count == 0:
            return 0.0
        window = self.get_window()
        return float(np.sqrt(np.mean(window ** 2)))

    def kurtosis(self) -> float:
        """
        4th standardized moment. ~3.0 for Gaussian noise (normal bearing);
        rises above 3 for impulsive signals (bearing defect strikes) before
        RMS moves much, making it an early, complementary discriminator.
        """
        if self.count < 4:
            return 3.0
        window = self.get_window()
        mean = np.mean(window)
        std = np.std(window)
        if std < 1e-10:
            return 3.0
        return float(np.mean(((window - mean) / std) ** 4))

    def is_full(self) -> bool:
        return self.count >= self.size

    def clear(self):
        self.buffer = [0.0] * self.size
        self.write_ptr = 0
        self.count = 0


# Fault-injection distribution used by generate_training_data() for every machine.
SIM_CONFIG = {
    "normal_fraction": 0.55,
    "early_fault_fraction": 0.25,
    "critical_fault_fraction": 0.15,
    "sensor_fault_fraction": 0.05,
    "noise_pct": 0.03,   # +/-3% measurement noise on physics-computed values
}


class InductionMotorSimulator:
    """
    Physics-based simulator for ABB M2AA 132M-4 induction motor.
    
    All equations from:
    - IEC 60034-1: Rotating electrical machines
    - SKF Bearing Reliability Manual
    - ISO 10816-3: Vibration severity for rotating machines (Group 2)
    - Heyland circle diagram for induction motor characteristics
    """
    
    # ── Nameplate / Datasheet Constants ──
    P_RATED = 7500.0        # W, rated output power
    V_RATED = 400.0         # V, line-to-line (3-phase)
    I_RATED = 15.2          # A, rated current
    N_SYNC = 1500.0         # RPM, synchronous speed (50Hz, 4-pole)
    N_RATED = 1460.0        # RPM, rated speed
    ETA = 0.891             # Efficiency at rated load
    COS_PHI = 0.84          # Power factor at rated load
    R_STATOR = 0.56         # Ω, stator resistance per phase (star)
    T_CLASS_F = 155.0       # °C, winding temperature limit (Class F)
    TAU_THERMAL = 1200.0    # s, thermal time constant
    THETA_THERMAL = 0.8     # °C/W, thermal resistance (winding to ambient)
    
    # ── Derived Constants ──
    I_NO_LOAD = 0.35 * I_RATED   # = 5.32 A (magnetizing current)
    SLIP_RATED = (N_SYNC - N_RATED) / N_SYNC  # = 0.0267
    F_SHAFT = N_RATED / 60.0     # = 24.33 Hz
    
    # ── Bearing Frequencies (SKF 6208-2Z deep-groove ball bearing) ──
    # BPFO = (n_balls/2) × (1 - d/D × cos(α)) × f_shaft
    # Verified against real 6208 datasheets (NTN/SNR): BPFO=3.607×, BPFI=5.393×,
    # FTF=0.401× at 1× shaft speed. Since BPFO+BPFI = n_balls exactly (property
    # of the formula), n=9 balls 
    F_BPFO = 3.607 * F_SHAFT     # = 87.7 Hz (outer race defect)
    F_BPFI = 5.393 * F_SHAFT     # = 131.2 Hz (inner race defect)
    F_BSF = 2.42 * F_SHAFT       # = 58.9 Hz (ball spin frequency, standard formula;
                                  #   note some bearing catalogs list "BSF" as 2x
                                  #   this -- the ball contacts each race once per
                                  #   revolution, so a defect on the ball produces
                                  #   an impact twice per ball rotation)
    F_FTF = 0.401 * F_SHAFT      # = 9.8 Hz (cage/train frequency)
    
    # ISO 10816-3 vibration severity zones live in config.settings.MACHINES
    # (single source of truth for the dashboard/API).
    
    def __init__(self):
        self.machine_id = "motor_1"
        self.machine_name = "ABB M2AA 132M-4 Induction Motor"
        
        # State variables (evolve over time)
        self.t = 0.0                    # Simulation time (seconds)
        self.T_winding = 25.0           # °C, initial winding temperature
        self.T_bearing = 25.0           # °C, initial bearing temperature
        self.bearing_severity = 0.0     # 0.0 = new, 1.0 = failed
        self.imbalance_severity = 0.0   # 0.0 = balanced, 1.0 = severe
        
        # Fault state
        self.fault_type = "none"
        self.fault_stage = 0            # 0=none, 1=early, 2=mid, 3=critical
        self.machine_state = "normal"
        
        # Sensor fault injection
        self.sensor_fault_type = "none"  # none, stuck_value, drift, spike, out_of_range, noise_flood
        self.sensor_fault_target = "none"  # current, temperature, vibration
        self.sensor_fault_start_time = 0
        self.stuck_value = 0.0
        self.drift_rate = 0.0
        
        # Vibration circular buffer (simulates 500Hz sampling, 64-sample window)
        self.vib_buffer = CircularBuffer(size=64)
        
    
    def set_fault(self, fault_type: str, stage: int):
        """
        Set machine fault type and severity stage.
        
        fault_type: 'none', 'bearing_degradation', 'rotor_imbalance', 
                    'overload', 'overheating'
        stage: 0 (none), 1 (early), 2 (mid), 3 (critical)
        """
        self.fault_type = fault_type
        self.fault_stage = stage
        
        if fault_type == "bearing_degradation":
            self.bearing_severity = [0.0, 0.2, 0.6, 1.0][stage]
        elif fault_type == "rotor_imbalance":
            self.imbalance_severity = [0.0, 0.3, 0.6, 1.0][stage]
        else:
            self.bearing_severity = 0.0
            self.imbalance_severity = 0.0
        
        if stage == 0:
            self.machine_state = "normal"
        elif stage == 1:
            self.machine_state = "early_fault"
        else:
            self.machine_state = "critical_fault"
    
    def set_sensor_fault(self, fault_type: str, target: str):
        """
        Inject a sensor hardware fault.
        
        fault_type: 'none', 'stuck_value', 'drift', 'spike', 
                    'out_of_range', 'noise_flood'
        target: 'current', 'temperature', 'vibration'
        """
        self.sensor_fault_type = fault_type
        self.sensor_fault_target = target
        self.sensor_fault_start_time = self.t
        
        if fault_type == "stuck_value":
            # Sensor freezes at its current value
            self.stuck_value = {
                'current': 13.0,
                'temperature': 55.0,
                'vibration': 1.5
            }.get(target, 0.0)
        elif fault_type == "drift":
            # Drift saturates at ±30% of true value, reached linearly over
            # ~120s (sign picked once per fault episode .
            self.drift_rate = (0.30 / 120.0) * np.random.choice([-1.0, 1.0])
    
    def compute_current(self, load_pct: float) -> float:
        """
        Compute stator current from load percentage.
        
        Motor current model (simplified Heyland diagram):
        I(load) = I_no_load + (I_rated - I_no_load) × (load_pct / 100)
        
        This is approximately linear for induction motors from
        no-load to rated load. At overload, current increases more
        steeply due to increased slip and saturation effects.
        
        Physics basis (order-of-magnitude sanity check, not the formula this
        method actually uses):
        P_out = V × I × √3 × cos(φ) × η
        I = P_out / (V × √3 × cos(φ) × η)
        At load_pct = 100%: I = 7500 / (400 × 1.732 × 0.84 × 0.891) ≈ 14.5 A 
        within ~5% of I_RATED = 15.2 A, not an exact match. The gap is
        expected: this idealized formula omits stray-load losses and core
        saturation margin , so "same order of magnitude" is the right bar, not exact equality.
        compute_current() below is anchored directly to I_RATED (the
        datasheet value), not to this formula.
        """
        load_frac = load_pct / 100.0
        
        # Linear approximation (valid from 0% to ~120% load)
        I = self.I_NO_LOAD + (self.I_RATED - self.I_NO_LOAD) * load_frac
        
        # Fault effects on current
        if self.fault_type == "bearing_degradation":
            # Bearing drag increases current: +3/8/15% by stage.
            I *= [1.0, 1.03, 1.08, 1.15][self.fault_stage]
        elif self.fault_type == "overload":
            # Overload: current above rated
            I *= (1 + 0.15 * self.fault_stage / 3.0)
        elif self.fault_type == "rotor_imbalance":
            # Imbalance: current relatively unchanged
            pass
        elif self.fault_type == "overheating":
            # Fan failure: current relatively normal
            pass
        
        return I
    
    def compute_temperature(self, current_A: float, ambient_temp: float, dt: float) -> Tuple[float, float]:
        """
        Compute winding and bearing temperatures using Newton's law of cooling
        with thermal mass.
        
        T_winding(t) = T_ambient + (I² × R_stator × θ_thermal) × (1 - e^(-t/τ))
        
        Where:
        - I²R_stator = copper losses (heat source), per phase
        - θ_thermal = 0.8 °C/W (thermal resistance, winding to ambient)
        - τ = 1200s (thermal time constant of motor frame+winding mass)
        
        The exponential term means temperature CANNOT change instantly.
        Maximum rate ≈ (T_steady - T_current) / τ per second.
        
        Bearing temperature:
        T_bearing = T_ambient + 0.3 × (T_winding - T_ambient) + friction_heat
        SKF specifies that bearing temperature should be 10-20°C above ambient
        in normal operation, never exceeding 80°C for standard grease.
        """
        # Steady-state winding temperature for current current level
        # Using all 3 phases: total copper loss = 3 × I² × R
        P_copper = 3.0 * current_A**2 * self.R_STATOR
        T_winding_steady = ambient_temp + P_copper * self.THETA_THERMAL
        
        # Fault effects on steady-state temperature
        if self.fault_type == "bearing_degradation":
            # Friction heat from damaged bearing
            T_winding_steady += [0, 5, 15, 25][self.fault_stage]
        elif self.fault_type == "overheating":
            # Fan failure: thermal resistance increases ~2x
            T_winding_steady = ambient_temp + P_copper * self.THETA_THERMAL * 2.0
            T_winding_steady += [0, 10, 25, 40][self.fault_stage]
        elif self.fault_type == "overload":
            # Already accounted for by higher current
            pass
        
        # Apply thermal time constant (first-order lag)
        # dT/dt = (T_steady - T) / τ
        dT = (T_winding_steady - self.T_winding) / self.TAU_THERMAL * dt
        
        # Clamp rate of change to 0.1°C/s (physical limit for this motor size)
        dT = max(-0.1 * dt, min(0.1 * dt, dT))
        self.T_winding += dT
        
        # Bearing temperature: tracks winding with additional coupling
        T_bearing_steady = ambient_temp + 0.3 * (self.T_winding - ambient_temp)
        
        # Bearing fault adds friction heat
        if self.fault_type == "bearing_degradation":
            T_bearing_steady += [0, 5, 15, 30][self.fault_stage]
        
        # Bearing thermal time constant is shorter (smaller mass)
        tau_bearing = 300.0  # seconds
        dT_bearing = (T_bearing_steady - self.T_bearing) / tau_bearing * dt
        dT_bearing = max(-0.1 * dt, min(0.1 * dt, dT_bearing))
        self.T_bearing += dT_bearing
        
        return self.T_winding, self.T_bearing
    
    def compute_vibration(self, load_pct: float, dt: float) -> Tuple[float, float]:
        """
        Compute vibration RMS and kurtosis per ISO 10816-3.
        
        Vibration model:
        v(t) = A_base × noise 
             + A_imbalance × sin(2π × f_shaft × t)           [1× shaft]
             + A_bearing × sin(2π × f_BPFO × t) × modulation [bearing defect]
        
        Where:
        - A_base depends on motor quality and load (0.3-0.8 mm/s for new motor)
        - A_imbalance increases with rotor_imbalance fault
        - A_bearing increases with bearing_degradation fault
        - Modulation: bearing defect signals are amplitude-modulated at f_shaft
        
        Normal motor vibration (ISO Zone A): v_rms < 1.8 mm/s
        At crest factor ~1.4, that's a peak of ~2.5 mm/s (peak-to-peak would
        be roughly double the peak, ~5 mm/s, for a not-strongly-impulsive
        signal).

        For kurtosis:
        - Normal: ~3.0 (Gaussian noise-like)
        - Early bearing: 4-6 (periodic impacts embedded in noise)
        - Advanced/critical bearing: 8-15+ (strong periodic impacts) and
          climbing with fault_stage -- this generator does NOT model the
          real-world late-stage flattening back toward Gaussian kurtosis
          that can occur once a bearing is so spalled the damage acts like
          continuous broadband roughness rather than a discrete impact
          site.
        """
        # Simulate at 500Hz for the circular buffer
        n_samples = int(500 * dt)  # 500 samples per second
        t_samples = self.t + np.arange(n_samples) / 500.0
        
        # Base vibration amplitude (function of load)
        A_base = 0.3 + 0.5 * (load_pct / 100.0)  # 0.3 to 0.8 mm/s base
        
        # Shaft frequency component (always present: residual imbalance)
        A_1x = 0.2 + 0.15 * (load_pct / 100.0)  # Small 1× component
        
        # Imbalance fault: 1× increases dramatically
        if self.fault_type == "rotor_imbalance":
            A_1x *= (1 + [0, 1.5, 4.0, 8.0][self.fault_stage])
        
        # Bearing defect component
        A_bearing = 0.0
        if self.fault_type == "bearing_degradation":
            A_bearing = [0.0, 0.3, 1.5, 5.0][self.fault_stage]
        
        # Generate time-domain signal
        signal = np.zeros(n_samples)
        for i, t in enumerate(t_samples):
            # Gaussian base noise
            v = A_base * np.random.randn() * 0.3
            
            # 1× shaft frequency (always present)
            v += A_1x * np.sin(2 * np.pi * self.F_SHAFT * t)
            
            # 2× shaft frequency (misalignment, always small)
            v += 0.1 * A_1x * np.sin(2 * np.pi * 2 * self.F_SHAFT * t)
            
            # Bearing defect: BPFO with amplitude modulation at shaft freq
            if A_bearing > 0:
                modulation = 1.0 + 0.5 * np.sin(2 * np.pi * self.F_SHAFT * t)
                v += A_bearing * modulation * np.sin(2 * np.pi * self.F_BPFO * t)
                # Add impulsive content (increases kurtosis)
                if np.random.random() < 0.05 * self.bearing_severity:
                    v += A_bearing * 3.0 * np.random.randn()
            
            signal[i] = v
            self.vib_buffer.push(v)
        
        # Compute RMS and kurtosis from buffer
        vib_rms = self.vib_buffer.rms()
        vib_kurtosis = self.vib_buffer.kurtosis()
        
        return vib_rms, vib_kurtosis
    
    def apply_sensor_fault(self, current_A: float, bearing_temp_C: float,
                           vibration_rms: float) -> Tuple[float, float, float, str]:
        """
        Apply sensor fault injection to the computed physics values.
        
        KEY PRINCIPLE: Only ONE sensor is affected. The other two remain
        physics-accurate. This is how the ML model learns to distinguish
        sensor faults from machine faults.
        """
        sensor_health = "all_ok"
        elapsed = self.t - self.sensor_fault_start_time
        
        if self.sensor_fault_type == "none":
            return current_A, bearing_temp_C, vibration_rms, sensor_health
        
        target = self.sensor_fault_target
        
        if self.sensor_fault_type == "stuck_value":
            # Sensor reads the same value regardless of actual physics
            if target == "current":
                current_A = self.stuck_value
                sensor_health = "current_fault"
            elif target == "temperature":
                bearing_temp_C = self.stuck_value
                sensor_health = "temp_fault"
            elif target == "vibration":
                vibration_rms = self.stuck_value
                sensor_health = "vib_fault"
        
        elif self.sensor_fault_type == "drift":
            # Value shifts by drift_rate per second, saturating at ±30%
            drift_amount = np.clip(self.drift_rate * elapsed, -0.30, 0.30)
            if target == "current":
                current_A *= (1 + drift_amount)
                sensor_health = "current_fault"
            elif target == "temperature":
                bearing_temp_C *= (1 + drift_amount)
                sensor_health = "temp_fault"
            elif target == "vibration":
                vibration_rms *= (1 + drift_amount)
                sensor_health = "vib_fault"
        
        elif self.sensor_fault_type == "spike":
            # Single-sample jump (loose connection)
            if np.random.random() < 0.1:  # 10% chance per sample
                if target == "current":
                    current_A = current_A * (5 + 3 * np.random.randn())
                    sensor_health = "current_fault"
                elif target == "temperature":
                    bearing_temp_C = 250 + 50 * np.random.randn()
                    sensor_health = "temp_fault"
                elif target == "vibration":
                    vibration_rms = 40 + 10 * np.random.randn()
                    sensor_health = "vib_fault"
        
        elif self.sensor_fault_type == "out_of_range":
            # Physically impossible value. Direction (too high vs too low)
            # is deliberately varied across the 3 machines below so the
            # label doesn't correlate with "always pegs high" -- motor and
            # compressor peg temperature high, pump pegs it low; current and
            # vibration go low/negative on all three.
            if target == "current":
                current_A = -10.0
                sensor_health = "current_fault"
            elif target == "temperature":
                bearing_temp_C = 300.0
                sensor_health = "temp_fault"
            elif target == "vibration":
                vibration_rms = -5.0
                sensor_health = "vib_fault"
        
        elif self.sensor_fault_type == "noise_flood":
            # Variance spikes 10× while mean roughly correct
            if target == "current":
                current_A += np.random.randn() * current_A * 0.5
                sensor_health = "current_fault"
            elif target == "temperature":
                bearing_temp_C += np.random.randn() * 20
                sensor_health = "temp_fault"
            elif target == "vibration":
                vibration_rms += abs(np.random.randn() * vibration_rms * 2)
                sensor_health = "vib_fault"
        
        return current_A, bearing_temp_C, vibration_rms, sensor_health
    
    def step(self, load_pct: float, ambient_temp: float, dt: float = 1.0) -> dict:
        """
        Advance simulation by dt seconds. Returns complete sensor reading.
        
        This is the main physics engine loop. Order of computation matters:
        1. Current (instantaneous, from load)
        2. Vibration (fast dynamics, from mechanical state)
        3. Temperature (slow dynamics, lags current and vibration)
        
        This ordering reflects physical reality: current changes first
        (electrical), vibration responds next (mechanical), temperature
        responds last (thermal inertia).
        """
        self.t += dt
        
        # 1. Compute current from load
        current_A = self.compute_current(load_pct)
        
        # 2. Compute vibration (fast dynamics)
        vib_rms, vib_kurtosis = self.compute_vibration(load_pct, dt)
        
        # 3. Compute temperature (slow dynamics, lags current)
        T_winding, T_bearing = self.compute_temperature(current_A, ambient_temp, dt)
        
        # Add controlled noise (±3% on physics-computed values)
        noise = SIM_CONFIG['noise_pct']
        current_A *= (1 + np.random.uniform(-noise, noise))
        T_bearing_noisy = T_bearing * (1 + np.random.uniform(-noise, noise))
        vib_rms *= (1 + np.random.uniform(-noise, noise))
        
        # Physical bounds (hard clip)
        current_A = max(0, min(current_A, self.I_RATED * 3))
        T_bearing_noisy = max(-10, min(T_bearing_noisy, 200))
        vib_rms = max(0, min(vib_rms, 50))
        vib_kurtosis = max(1.5, min(vib_kurtosis, 20))
        
        # Apply sensor fault injection (if active)
        current_A, T_bearing_noisy, vib_rms, sensor_health = \
            self.apply_sensor_fault(current_A, T_bearing_noisy, vib_rms)
        
        
        # Determine machine state label
        if self.fault_stage == 0:
            machine_state = "normal"
        elif self.fault_stage == 1:
            machine_state = "early_fault"
        else:
            machine_state = "critical_fault"
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "machine_id": self.machine_id,
            "machine_name": self.machine_name,
            "load_pct": round(load_pct, 1),
            "ambient_temp": round(ambient_temp, 1),
            "current_A": round(current_A, 3),
            "bearing_temp_C": round(T_bearing_noisy, 2),
            "vibration_rms_mm_s": round(vib_rms, 4),
            "vibration_kurtosis": round(vib_kurtosis, 3),
            "machine_state": machine_state,
            "fault_type": self.fault_type,
            "sensor_health": sensor_health,
            "winding_temp_C": round(T_winding, 2),
            "T_bearing_true": round(T_bearing, 2),  # True value before sensor fault
        }


class CentrifugalPumpSimulator:
    """
    Physics-based simulator for Grundfos CR series centrifugal pump.
    
    Key physics:
    - Hydraulic power: P_h = ρ × g × Q × H
    - Affinity laws: Q ∝ N, H ∝ N², P ∝ N³
    - Cavitation: occurs when local pressure drops below vapor pressure
      → NPSHa < NPSHr → bubbles form and collapse → broadband vibration
    - Impeller damage: vane pass frequency harmonics increase
    - Dry running: no liquid → no load → current drops, temp rises (no cooling)
    """
    
    # ── Nameplate Constants ──
    P_RATED = 15000.0       # W
    Q_RATED = 50.0          # m³/h
    H_RATED = 32.0          # m (head at rated flow / BEP)
    H_SHUTOFF = 1.2 * H_RATED  # m, head at Q=0. Real centrifugal pump curves
                               # peak at shutoff and fall monotonically with
                               # flow (shutoff head is typically 110-130% of
                               # BEP head; 120% used here as a representative
                               # mid-range value, not a specific datasheet).
    N_RATED = 2900.0        # RPM
    ETA_PUMP = 0.78         # Pump efficiency at BEP
    ETA_MOTOR = 0.89        # Motor efficiency
    I_RATED = 28.0          # A
    N_VANES = 6             # Impeller vanes
    RHO = 1000.0            # kg/m³ (water)
    G = 9.81                # m/s²
    
    # ── Derived Constants ──
    I_NO_LOAD = 0.35 * I_RATED  # = 9.8 A
    F_SHAFT = N_RATED / 60.0    # = 48.33 Hz
    F_VANE = N_VANES * F_SHAFT  # = 290 Hz (vane pass frequency)
    R_STATOR = 0.32             # Ω (estimated for 15kW motor)
    TAU_THERMAL = 900.0         # s (shorter than motor — pump is wet)
    
    def __init__(self):
        self.machine_id = "pump_2"
        self.machine_name = "Grundfos CR Centrifugal Pump"
        self.t = 0.0
        self.T_bearing = 25.0
        self.T_winding = 25.0
        
        self.fault_type = "none"
        self.fault_stage = 0
        self.machine_state = "normal"
        
        self.sensor_fault_type = "none"
        self.sensor_fault_target = "none"
        self.sensor_fault_start_time = 0
        self.stuck_value = 0.0
        self.drift_rate = 0.0
        
        self.vib_buffer = CircularBuffer(size=64)
    
    def set_fault(self, fault_type: str, stage: int):
        self.fault_type = fault_type
        self.fault_stage = stage
        if stage == 0:
            self.machine_state = "normal"
        elif stage == 1:
            self.machine_state = "early_fault"
        else:
            self.machine_state = "critical_fault"
    
    def set_sensor_fault(self, fault_type: str, target: str):
        self.sensor_fault_type = fault_type
        self.sensor_fault_target = target
        self.sensor_fault_start_time = self.t
        if fault_type == "stuck_value":
            self.stuck_value = {'current': 22.0, 'temperature': 50.0, 'vibration': 2.0}.get(target, 0.0)
        elif fault_type == "drift":
            # Drift saturates at ±30% of true value, see InductionMotorSimulator
            self.drift_rate = (0.30 / 120.0) * np.random.choice([-1.0, 1.0])
    
    def compute_current(self, load_pct: float) -> float:
        """
        Current model based on hydraulic power demand.

        P_hydraulic = ρ × g × Q × H = 1000 × 9.81 × (Q/3600) × H
        P_shaft = P_hydraulic / η_pump
        I = I_no_load + (P_shaft / P_rated) × (I_rated - I_no_load)

        H(Q) is a standard pump-curve parabola fit through two known points
        -- (Q=0, H=H_SHUTOFF) and (Q=Q_RATED, H=H_RATED) -- so head falls
        monotonically as flow rises, the way a real pump curve does. (An
        earlier version used a curve centered on rated flow that INCREASED
        toward BEP from both directions, i.e. shutoff head was the lowest
        point on the curve -- backwards from every real pump datasheet.)

        At rated: P_h = 1000 × 9.81 × (50/3600) × 32 = 4360 W
        P_shaft = 4360 / 0.78 = 5590 W → well below 15kW (pump motor oversized)
        """
        load_frac = load_pct / 100.0
        Q = self.Q_RATED * load_frac  # m³/h
        k = (self.H_SHUTOFF - self.H_RATED) / self.Q_RATED**2
        H = self.H_SHUTOFF - k * Q**2
        H = max(0, H)
        
        P_hydraulic = self.RHO * self.G * (Q / 3600.0) * H
        P_shaft = P_hydraulic / max(self.ETA_PUMP, 0.1)
        
        I = self.I_NO_LOAD + (P_shaft / self.P_RATED) * (self.I_RATED - self.I_NO_LOAD)
        
        # Fault effects
        if self.fault_type == "cavitation":
            # Cavitation: current fluctuates wildly (±15%)
            I *= (1 + 0.15 * np.random.randn() * self.fault_stage / 3)
        elif self.fault_type == "impeller_damage":
            I *= (1 + 0.05 * self.fault_stage)
        elif self.fault_type == "dry_running":
            I *= (1 - 0.6 * self.fault_stage / 3)  # Current drops to 40%
        
        return max(0, I)
    
    def compute_temperature(self, current_A: float, ambient_temp: float, dt: float):
        P_copper = 3.0 * current_A**2 * self.R_STATOR
        theta = 0.5  # °C/W (lower thermal resistance — wet pump)
        T_steady = ambient_temp + P_copper * theta
        
        if self.fault_type == "dry_running":
            # No liquid cooling → temperature rises rapidly
            T_steady += [0, 20, 40, 60][self.fault_stage]
        elif self.fault_type == "cavitation":
            T_steady += [0, 5, 10, 15][self.fault_stage]
        
        dT = (T_steady - self.T_winding) / self.TAU_THERMAL * dt
        dT = max(-0.1 * dt, min(0.1 * dt, dT))
        self.T_winding += dT
        
        T_bearing_steady = ambient_temp + 0.25 * (self.T_winding - ambient_temp)
        if self.fault_type == "dry_running":
            T_bearing_steady += [0, 15, 30, 45][self.fault_stage]
        
        dT_b = (T_bearing_steady - self.T_bearing) / 300.0 * dt
        dT_b = max(-0.1 * dt, min(0.1 * dt, dT_b))
        self.T_bearing += dT_b
        
        return self.T_winding, self.T_bearing
    
    def compute_vibration(self, load_pct: float, dt: float):
        n_samples = int(500 * dt)
        t_samples = self.t + np.arange(n_samples) / 500.0
        
        A_base = 0.4 + 0.6 * (load_pct / 100.0)
        A_1x = 0.3
        A_vane = 0.15  # Normal vane pass amplitude
        
        if self.fault_type == "cavitation":
            # Broadband noise increases dramatically
            A_base *= (1 + [0, 1.5, 3.0, 5.0][self.fault_stage])
        elif self.fault_type == "impeller_damage":
            A_vane *= (1 + [0, 3.0, 6.0, 8.0][self.fault_stage])
        elif self.fault_type == "dry_running":
            A_base *= 0.3  # Low vibration (no fluid forces)
        
        for i, t in enumerate(t_samples):
            v = A_base * np.random.randn() * 0.3
            v += A_1x * np.sin(2 * np.pi * self.F_SHAFT * t)
            v += A_vane * np.sin(2 * np.pi * self.F_VANE * t)
            
            # Cavitation: random high-energy impacts
            if self.fault_type == "cavitation" and np.random.random() < 0.08 * self.fault_stage:
                v += A_base * 4 * abs(np.random.randn())
            
            self.vib_buffer.push(v)
        
        return self.vib_buffer.rms(), self.vib_buffer.kurtosis()
    
    def apply_sensor_fault(self, current_A, bearing_temp_C, vibration_rms):
        sensor_health = "all_ok"
        elapsed = self.t - self.sensor_fault_start_time
        
        if self.sensor_fault_type == "none":
            return current_A, bearing_temp_C, vibration_rms, sensor_health
        
        target = self.sensor_fault_target
        
        if self.sensor_fault_type == "stuck_value":
            if target == "current":
                current_A = self.stuck_value; sensor_health = "current_fault"
            elif target == "temperature":
                bearing_temp_C = self.stuck_value; sensor_health = "temp_fault"
            elif target == "vibration":
                vibration_rms = self.stuck_value; sensor_health = "vib_fault"
        elif self.sensor_fault_type == "drift":
            drift_amount = np.clip(self.drift_rate * elapsed, -0.30, 0.30)
            if target == "current":
                current_A *= (1 + drift_amount); sensor_health = "current_fault"
            elif target == "temperature":
                bearing_temp_C *= (1 + drift_amount); sensor_health = "temp_fault"
            elif target == "vibration":
                vibration_rms *= (1 + drift_amount); sensor_health = "vib_fault"
        elif self.sensor_fault_type == "spike":
            if np.random.random() < 0.1:
                if target == "current":
                    current_A *= (5 + 3*np.random.randn()); sensor_health = "current_fault"
                elif target == "temperature":
                    bearing_temp_C = 250 + 50*np.random.randn(); sensor_health = "temp_fault"
                elif target == "vibration":
                    vibration_rms = 40 + 10*np.random.randn(); sensor_health = "vib_fault"
        elif self.sensor_fault_type == "out_of_range":
            # Physically impossible value; this machine pegs temperature
            # LOW (motor/compressor peg it high) see InductionMotorSimulator
            # for why that's deliberate, not an inconsistency.
            if target == "current":
                current_A = -10.0; sensor_health = "current_fault"
            elif target == "temperature":
                bearing_temp_C = -50.0; sensor_health = "temp_fault"
            elif target == "vibration":
                vibration_rms = -5.0; sensor_health = "vib_fault"
        elif self.sensor_fault_type == "noise_flood":
            if target == "current":
                current_A += np.random.randn() * current_A * 0.5; sensor_health = "current_fault"
            elif target == "temperature":
                bearing_temp_C += np.random.randn() * 20; sensor_health = "temp_fault"
            elif target == "vibration":
                vibration_rms += abs(np.random.randn() * vibration_rms * 2); sensor_health = "vib_fault"
        
        return current_A, bearing_temp_C, vibration_rms, sensor_health
    
    def step(self, load_pct: float, ambient_temp: float, dt: float = 1.0) -> dict:
        self.t += dt
        
        current_A = self.compute_current(load_pct)
        vib_rms, vib_kurtosis = self.compute_vibration(load_pct, dt)
        T_winding, T_bearing = self.compute_temperature(current_A, ambient_temp, dt)
        
        noise = SIM_CONFIG['noise_pct']
        current_A *= (1 + np.random.uniform(-noise, noise))
        T_bearing_noisy = T_bearing * (1 + np.random.uniform(-noise, noise))
        vib_rms *= (1 + np.random.uniform(-noise, noise))
        
        current_A = max(0, min(current_A, self.I_RATED * 3))
        T_bearing_noisy = max(-10, min(T_bearing_noisy, 200))
        vib_rms = max(0, min(vib_rms, 50))
        vib_kurtosis = max(1.5, min(vib_kurtosis, 20))
        
        current_A, T_bearing_noisy, vib_rms, sensor_health = \
            self.apply_sensor_fault(current_A, T_bearing_noisy, vib_rms)
        
        
        if self.fault_stage == 0:
            machine_state = "normal"
        elif self.fault_stage == 1:
            machine_state = "early_fault"
        else:
            machine_state = "critical_fault"
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "machine_id": self.machine_id,
            "machine_name": self.machine_name,
            "load_pct": round(load_pct, 1),
            "ambient_temp": round(ambient_temp, 1),
            "current_A": round(current_A, 3),
            "bearing_temp_C": round(T_bearing_noisy, 2),
            "vibration_rms_mm_s": round(vib_rms, 4),
            "vibration_kurtosis": round(vib_kurtosis, 3),
            "machine_state": machine_state,
            "fault_type": self.fault_type,
            "sensor_health": sensor_health,
            "winding_temp_C": round(T_winding, 2),
            "T_bearing_true": round(T_bearing, 2),
        }


class ReciprocatingCompressorSimulator:
    """
    Physics-based simulator for Atlas Copco GA22 reciprocating compressor.
    
    Key physics:
    - Isentropic compression: T2 = T1 × (P2/P1)^((γ-1)/γ)
    - For air (γ=1.4): T2 = 298 × (8/1)^(0.286) = 298 × 1.81 = 540K = 267°C
    - After-cooler brings this down to ~49°C in normal operation
    - Blocked cooler: temperature rises toward theoretical T2
    - Valve leakage: re-compresses leaked air → more current for same output
    """
    
    # ── Nameplate Constants ──
    P_RATED = 22000.0       # W
    P1 = 1.0                # bar (inlet)
    P2 = 8.0                # bar (discharge)
    FAD = 3.5               # m³/min (Free Air Delivery)
    N_RATED = 1460.0        # RPM
    I_RATED = 42.0          # A
    GAMMA = 1.4             # Heat capacity ratio for air
    
    # ── Derived Constants ──
    I_NO_LOAD = 0.25 * I_RATED  # = 10.5 A (unloaded)
    F_SHAFT = N_RATED / 60.0    # = 24.33 Hz
    R_STATOR = 0.21             # Ω (estimated for 22kW motor)
    TAU_THERMAL = 600.0         # s (compact, air-cooled)
    # Theoretical discharge temp depends on ambient, which varies per row --
    # see compute_temperature(), which recomputes it from the real ambient_temp
    # every call. (25°C reference: T2 ≈ 540K ≈ 267°C, see class docstring.)
    
    def __init__(self):
        self.machine_id = "compressor_3"
        self.machine_name = "Atlas Copco GA22 Compressor"
        self.t = 0.0
        self.T_discharge = 25.0  # °C (after cooler)
        self.T_bearing = 25.0
        self.T_winding = 25.0
        
        self.fault_type = "none"
        self.fault_stage = 0
        self.machine_state = "normal"
        
        self.sensor_fault_type = "none"
        self.sensor_fault_target = "none"
        self.sensor_fault_start_time = 0
        self.stuck_value = 0.0
        self.drift_rate = 0.0
        
        self.vib_buffer = CircularBuffer(size=64)
    
    def set_fault(self, fault_type: str, stage: int):
        self.fault_type = fault_type
        self.fault_stage = stage
        if stage == 0:
            self.machine_state = "normal"
        elif stage == 1:
            self.machine_state = "early_fault"
        else:
            self.machine_state = "critical_fault"
    
    def set_sensor_fault(self, fault_type: str, target: str):
        self.sensor_fault_type = fault_type
        self.sensor_fault_target = target
        self.sensor_fault_start_time = self.t
        if fault_type == "stuck_value":
            self.stuck_value = {'current': 35.0, 'temperature': 45.0, 'vibration': 3.0}.get(target, 0.0)
        elif fault_type == "drift":
            # Drift saturates at ±30% of true value, see InductionMotorSimulator
            self.drift_rate = (0.30 / 120.0) * np.random.choice([-1.0, 1.0])
    
    def compute_current(self, load_pct: float) -> float:
        """
        Compressor power model:
        P = (P1 × Q × γ/(γ-1)) × ((P2/P1)^((γ-1)/γ) - 1) / η_isentropic
        
        Loaded (compressing): I ≈ 35-42A
        Unloaded (venting): I ≈ 10-12A (just spinning)
        
        Valve leakage: needs more power for same output (+10-20%)
        """
        load_frac = load_pct / 100.0
        
        # Isentropic work per unit volume
        pressure_ratio = self.P2 / self.P1
        work_factor = (self.GAMMA / (self.GAMMA - 1)) * \
                      (pressure_ratio**((self.GAMMA - 1) / self.GAMMA) - 1)
        
        # Shaft power (P1 in Pa, Q in m³/s)
        P1_pa = self.P1 * 1e5  # bar to Pa
        Q_m3s = (self.FAD / 60.0) * load_frac  # m³/s
        eta_isentropic = 0.75  # Typical for reciprocating
        
        P_shaft = P1_pa * Q_m3s * work_factor / eta_isentropic
        
        # Current
        I = self.I_NO_LOAD + (P_shaft / self.P_RATED) * (self.I_RATED - self.I_NO_LOAD)
        
        # Fault effects
        if self.fault_type == "valve_leakage":
            I *= (1 + 0.05 * self.fault_stage)  # +5/10/15%
        elif self.fault_type == "bearing_wear":
            I *= (1 + 0.017 * self.fault_stage)  # +1.7/3.4/5%
        
        return max(0, I)
    
    def compute_temperature(self, current_A: float, ambient_temp: float, dt: float):
        """
        After-cooler model:
        - Normal: T_discharge ≈ 49°C (cooler working at 90% efficiency)
        - Blocked cooler: T rises toward theoretical 267°C (uncooled adiabatic
          discharge temp at 8:1 compression, 25°C ambient)

        T_after_cooler = T_ambient + (T_theoretical - T_ambient) × (1 - cooler_efficiency)
        At cooler_eff=0.90, ambient=25°C: T_ac = 25 + (267-25) × 0.10 ≈ 49.2°C
        """
        T_in_kelvin = (ambient_temp + 273.15)
        T_discharge_theoretical = T_in_kelvin * (self.P2/self.P1)**((self.GAMMA-1)/self.GAMMA) - 273.15
        
        # Normal cooler efficiency
        cooler_eff = 0.90  # 90% efficient
        
        if self.fault_type == "overheating":
            # Blocked cooler: efficiency drops
            cooler_eff *= (1 - 0.3 * self.fault_stage)  # 70/40/10% efficiency
        
        T_discharge_steady = ambient_temp + (T_discharge_theoretical - ambient_temp) * (1 - cooler_eff)
        
        if self.fault_type == "valve_leakage":
            T_discharge_steady += [0, 10, 20, 30][self.fault_stage]
        
        dT = (T_discharge_steady - self.T_discharge) / self.TAU_THERMAL * dt
        dT = max(-0.1 * dt, min(0.1 * dt, dT))
        self.T_discharge += dT
        
        # Bearing temperature
        P_copper = 3.0 * current_A**2 * self.R_STATOR
        T_winding_steady = ambient_temp + P_copper * 0.4
        if self.fault_type == "bearing_wear":
            T_winding_steady += [0, 5, 15, 25][self.fault_stage]
        
        dT_w = (T_winding_steady - self.T_winding) / self.TAU_THERMAL * dt
        dT_w = max(-0.1 * dt, min(0.1 * dt, dT_w))
        self.T_winding += dT_w
        
        T_bearing_steady = ambient_temp + 0.3 * (self.T_winding - ambient_temp)
        if self.fault_type == "bearing_wear":
            T_bearing_steady += [0, 5, 10, 20][self.fault_stage]
        
        dT_b = (T_bearing_steady - self.T_bearing) / 300.0 * dt
        dT_b = max(-0.1 * dt, min(0.1 * dt, dT_b))
        self.T_bearing += dT_b
        
        return self.T_winding, self.T_bearing
    
    def compute_vibration(self, load_pct: float, dt: float):
        n_samples = int(500 * dt)
        t_samples = self.t + np.arange(n_samples) / 500.0
        
        A_base = 0.5 + 0.8 * (load_pct / 100.0)  # Compressors are noisier
        A_1x = 0.4
        
        # Reciprocating: strong 1× and 2× components (piston motion)
        A_2x = 0.3
        
        if self.fault_type == "valve_leakage":
            # New harmonic component at 3x shaft frequency (leaking valve
            # re-compression disturbs the normal 1x/2x piston-motion
            # spectrum). Simplified to one new harmonic, not a full
            # harmonic series.
            A_3x = 0.2 * self.fault_stage
        else:
            A_3x = 0.0
        
        bearing_severity = 0.0
        if self.fault_type == "bearing_wear":
            bearing_severity = [0.0, 0.2, 0.6, 1.0][self.fault_stage]
        
        A_bearing = bearing_severity * 3.0
        
        for i, t in enumerate(t_samples):
            v = A_base * np.random.randn() * 0.25
            v += A_1x * np.sin(2 * np.pi * self.F_SHAFT * t)
            v += A_2x * np.sin(2 * np.pi * 2 * self.F_SHAFT * t)
            v += A_3x * np.sin(2 * np.pi * 3 * self.F_SHAFT * t)
            
            if A_bearing > 0:
                f_bpfo = 3.607 * self.F_SHAFT
                modulation = 1.0 + 0.5 * np.sin(2 * np.pi * self.F_SHAFT * t)
                v += A_bearing * modulation * np.sin(2 * np.pi * f_bpfo * t)
                if np.random.random() < 0.05 * bearing_severity:
                    v += A_bearing * 3.0 * np.random.randn()
            
            self.vib_buffer.push(v)
        
        return self.vib_buffer.rms(), self.vib_buffer.kurtosis()
    
    def apply_sensor_fault(self, current_A, bearing_temp_C, vibration_rms):
        sensor_health = "all_ok"
        elapsed = self.t - self.sensor_fault_start_time
        
        if self.sensor_fault_type == "none":
            return current_A, bearing_temp_C, vibration_rms, sensor_health
        
        target = self.sensor_fault_target
        
        if self.sensor_fault_type == "stuck_value":
            if target == "current": current_A = self.stuck_value; sensor_health = "current_fault"
            elif target == "temperature": bearing_temp_C = self.stuck_value; sensor_health = "temp_fault"
            elif target == "vibration": vibration_rms = self.stuck_value; sensor_health = "vib_fault"
        elif self.sensor_fault_type == "drift":
            drift = np.clip(self.drift_rate * elapsed, -0.30, 0.30)
            if target == "current": current_A *= (1+drift); sensor_health = "current_fault"
            elif target == "temperature": bearing_temp_C *= (1+drift); sensor_health = "temp_fault"
            elif target == "vibration": vibration_rms *= (1+drift); sensor_health = "vib_fault"
        elif self.sensor_fault_type == "spike":
            if np.random.random() < 0.1:
                if target == "current": current_A *= (5+3*np.random.randn()); sensor_health = "current_fault"
                elif target == "temperature": bearing_temp_C = 250+50*np.random.randn(); sensor_health = "temp_fault"
                elif target == "vibration": vibration_rms = 40+10*np.random.randn(); sensor_health = "vib_fault"
        elif self.sensor_fault_type == "out_of_range":
            # Physically impossible value; pegs high, like the motor (pump
            # pegs low) -- deliberate direction diversity, see
            # InductionMotorSimulator.apply_sensor_fault().
            if target == "current": current_A = -10.0; sensor_health = "current_fault"
            elif target == "temperature": bearing_temp_C = 300.0; sensor_health = "temp_fault"
            elif target == "vibration": vibration_rms = -5.0; sensor_health = "vib_fault"
        elif self.sensor_fault_type == "noise_flood":
            if target == "current": current_A += np.random.randn()*current_A*0.5; sensor_health = "current_fault"
            elif target == "temperature": bearing_temp_C += np.random.randn()*20; sensor_health = "temp_fault"
            elif target == "vibration": vibration_rms += abs(np.random.randn()*vibration_rms*2); sensor_health = "vib_fault"
        
        return current_A, bearing_temp_C, vibration_rms, sensor_health
    
    def step(self, load_pct: float, ambient_temp: float, dt: float = 1.0) -> dict:
        self.t += dt
        
        current_A = self.compute_current(load_pct)
        vib_rms, vib_kurtosis = self.compute_vibration(load_pct, dt)
        T_winding, T_bearing = self.compute_temperature(current_A, ambient_temp, dt)
        
        noise = SIM_CONFIG['noise_pct']
        current_A *= (1 + np.random.uniform(-noise, noise))
        T_bearing_noisy = T_bearing * (1 + np.random.uniform(-noise, noise))
        vib_rms *= (1 + np.random.uniform(-noise, noise))
        
        current_A = max(0, min(current_A, self.I_RATED * 3))
        T_bearing_noisy = max(-10, min(T_bearing_noisy, 200))
        vib_rms = max(0, min(vib_rms, 50))
        vib_kurtosis = max(1.5, min(vib_kurtosis, 20))
        
        current_A, T_bearing_noisy, vib_rms, sensor_health = \
            self.apply_sensor_fault(current_A, T_bearing_noisy, vib_rms)
        
        
        if self.fault_stage == 0:
            machine_state = "normal"
        elif self.fault_stage == 1:
            machine_state = "early_fault"
        else:
            machine_state = "critical_fault"
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "machine_id": self.machine_id,
            "machine_name": self.machine_name,
            "load_pct": round(load_pct, 1),
            "ambient_temp": round(ambient_temp, 1),
            "current_A": round(current_A, 3),
            "bearing_temp_C": round(T_bearing_noisy, 2),
            "vibration_rms_mm_s": round(vib_rms, 4),
            "vibration_kurtosis": round(vib_kurtosis, 3),
            "machine_state": machine_state,
            "fault_type": self.fault_type,
            "sensor_health": sensor_health,
            "winding_temp_C": round(T_winding, 2),
            "T_bearing_true": round(T_bearing, 2),
            "discharge_temp_C": round(self.T_discharge, 2),
        }


def generate_training_data(simulator, machine_faults: list, total_rows: int = 50000) -> pd.DataFrame:
    """
    Generate physics-based training data with controlled fault distribution.

    Each fault scenario is one continuous simulated run (temperature/vibration
    evolve smoothly sample-to-sample — TAU_THERMAL is on the order of minutes),
    so consecutive rows within a run are highly correlated, near-duplicate
    feature vectors. Every row is tagged with a `chunk_id`: a new id every 100
    consecutive samples, and always a new id at the start of a fresh
    simulation run. train_models.py splits train/test by `chunk_id` (not by
    row) so a sample and its near-duplicate neighbor can't end up on opposite
    sides of the split — a plain random row split leaks exactly that and
    inflates accuracy on data like this.
    """
    all_rows = []
    config = SIM_CONFIG
    chunk_size = 100
    segment_id = [0]  # mutable closure cell

    def run_segment(step_fn, n_rows):
        """Run n_rows of simulator.step() calls, tagging each row with a
        chunk_id unique to this simulation run (segment_id * large offset +
        local chunk index), and append to all_rows."""
        sid = segment_id[0]
        segment_id[0] += 1
        for i in range(n_rows):
            data = step_fn(i)
            data["chunk_id"] = sid * 1_000_000 + (i // chunk_size)
            all_rows.append(data)

    rows_normal = int(total_rows * config['normal_fraction'])
    rows_early = int(total_rows * config['early_fault_fraction'])
    rows_critical = int(total_rows * config['critical_fault_fraction'])
    rows_sensor = int(total_rows * config['sensor_fault_fraction'])

    print(f"\n{'='*60}")
    print(f"Generating data for: {simulator.machine_name}")
    print(f"{'='*60}")
    print(f"  Normal rows:    {rows_normal}")
    print(f"  Early fault:    {rows_early}")
    print(f"  Critical fault: {rows_critical}")
    print(f"  Sensor fault:   {rows_sensor}")

    # -- 1. Normal operation data --
    print("\n  [1/4] Generating normal operation data...")
    simulator.__init__()
    simulator.set_fault("none", 0)
    simulator.set_sensor_fault("none", "none")

    def normal_step(i):
        load = 60 + 30 * np.sin(2 * np.pi * i / 3600) + np.random.uniform(-5, 5)
        load = max(30, min(110, load))
        ambient = 25 + 5 * np.sin(2 * np.pi * i / 86400) + np.random.uniform(-1, 1)
        return simulator.step(load_pct=load, ambient_temp=ambient)

    run_segment(normal_step, rows_normal)

    # -- 2. Early fault data --
    print("  [2/4] Generating early fault data...")
    rows_per_fault = rows_early // len(machine_faults)

    for fault_name in machine_faults:
        simulator.__init__()
        simulator.set_fault(fault_name, 1)  # Stage 1 (early)
        simulator.set_sensor_fault("none", "none")

        for _ in range(200):  # let temperature stabilize; discarded
            simulator.step(load_pct=75.0, ambient_temp=25.0)

        def early_step(i):
            load = 60 + 30 * np.sin(2 * np.pi * i / 1800) + np.random.uniform(-5, 5)
            load = max(30, min(110, load))
            ambient = 25 + np.random.uniform(-2, 2)
            return simulator.step(load_pct=load, ambient_temp=ambient)

        run_segment(early_step, rows_per_fault)
        print(f"    Early {fault_name}: {rows_per_fault} rows")

    # -- 3. Critical fault data (stage 2 then stage 3, one continuous run) --
    print("  [3/4] Generating critical fault data...")
    rows_per_fault_crit = rows_critical // len(machine_faults)
    half = rows_per_fault_crit // 2

    for fault_name in machine_faults:
        simulator.__init__()
        simulator.set_fault(fault_name, 2)  # Stage 2 (mid)
        simulator.set_sensor_fault("none", "none")
        for _ in range(200):
            simulator.step(load_pct=80.0, ambient_temp=25.0)

        sid = segment_id[0]
        segment_id[0] += 1

        def crit_stage_step(i, load_center, load_amp, period):
            load = load_center + load_amp * np.sin(2 * np.pi * i / period) + np.random.uniform(-3, 3)
            load = max(30, min(110, load))
            ambient = 25 + np.random.uniform(-2, 2)
            return simulator.step(load_pct=load, ambient_temp=ambient)

        for i in range(half):
            data = crit_stage_step(i, 70, 20, 1200)
            data["chunk_id"] = sid * 1_000_000 + (i // chunk_size)
            all_rows.append(data)

        simulator.set_fault(fault_name, 3)  # Stage 3 (critical) — same run
        for _ in range(200):
            simulator.step(load_pct=85.0, ambient_temp=25.0)

        for i in range(half):
            data = crit_stage_step(i, 75, 15, 900)
            data["chunk_id"] = sid * 1_000_000 + ((half + i) // chunk_size)
            all_rows.append(data)

        print(f"    Critical {fault_name}: {rows_per_fault_crit} rows")

    # -- 4. Sensor fault data --
    print("  [4/4] Generating sensor fault data...")
    sensor_targets = ["current", "temperature", "vibration"]
    sensor_fault_types = ["stuck_value", "drift", "spike", "out_of_range", "noise_flood"]

    rows_per_combo = rows_sensor // (len(sensor_targets) * len(sensor_fault_types))
    # Floor only prevents a *zero-row* class (unlearnable), not a small one.

    if rows_per_combo < 10:
        print(f"    [!] --rows too small for the configured sensor_fraction "
              f"({rows_per_combo}/combo requested); flooring to 10/combo, "
             f"which will overshoot SIM_CONFIG['sensor_fault_fraction'].")
        rows_per_combo = 10

    for target in sensor_targets:
        for sfault in sensor_fault_types:
            simulator.__init__()
            simulator.set_fault("none", 0)  # machine itself is healthy
            simulator.set_sensor_fault(sfault, target)

            for _ in range(100):
                simulator.step(load_pct=75.0, ambient_temp=25.0)

            def sensor_step(i):
                load = 70 + 20 * np.sin(2 * np.pi * i / 600) + np.random.uniform(-3, 3)
                load = max(30, min(110, load))
                ambient = 25 + np.random.uniform(-1, 1)
                return simulator.step(load_pct=load, ambient_temp=ambient)

            run_segment(sensor_step, rows_per_combo)
            simulator.set_sensor_fault("none", "none")

    print(f"    Sensor faults: {len(sensor_targets) * len(sensor_fault_types) * rows_per_combo} rows")

    df = pd.DataFrame(all_rows)
    print(f"\n  Total rows generated: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    return df


MACHINE_SPECS = {
    "motor": (InductionMotorSimulator,
              ["bearing_degradation", "rotor_imbalance", "overload", "overheating"]),
    "pump": (CentrifugalPumpSimulator,
             ["cavitation", "impeller_damage", "dry_running"]),
    "compressor": (ReciprocatingCompressorSimulator,
                   ["valve_leakage", "bearing_wear", "overheating"]),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=50000, help="rows per machine")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="training_data", help="output directory for CSVs")
    args = p.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print("=" * 60)
    print("TRAINING DATA GENERATION")
    print("=" * 60)

    summary = {}
    for machine, (sim_cls, faults) in MACHINE_SPECS.items():
        sim = sim_cls()
        df = generate_training_data(sim, faults, total_rows=args.rows)
        out_path = os.path.join(args.out, f"{machine}_training_data.csv")
        df.to_csv(out_path, index=False)
        print(f"\n  Saved to: {out_path}")
        summary[machine] = df

    print("\n" + "=" * 60)
    print("DATA GENERATION SUMMARY")
    print("=" * 60)
    for machine, df in summary.items():
        print(f"\n{machine}:")
        print(f"  Total rows: {len(df)}")
        print(f"  Machine states: {dict(df['machine_state'].value_counts())}")
        print(f"  Fault types:    {dict(df['fault_type'].value_counts())}")
        print(f"  Sensor health:  {dict(df['sensor_health'].value_counts())}")


if __name__ == "__main__":
    main()
