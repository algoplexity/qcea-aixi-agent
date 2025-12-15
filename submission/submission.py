"""
QCEA-AIXI AGENT SUBMISSION
Horizon 2: The Reflective Physicist  (Neural BDM & Homeostatic Regulation)
Maintainer: Algoplexity
Model: The Aletheia-Phronesis Architecture (with "Aletheia Box" Telemetry)

THEORETICAL BASIS:
1. SENSOR: Tiny Recursive Model (TRM) trained on Wolfram Prime 9 Rules.
2. RISK ENGINE: "Algorithmic Governor" based on Damage Spreading Rates (Lyapunov Exponents).
3. CONTROL: Homeostatic Feedback Loop (Gamma) optimizing Log-Likelihood survival.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import pathlib
import math
from birdgame.trackers.trackerbase import TrackerBase

# ==============================================================================
# 1. PHYSICS CONSTANTS (The Lyapunov Spectrum)
# ==============================================================================
# These weights are not arbitrary heuristics. They are derived from the 
# Intrinsic Algorithmic Volatility (Damage Spreading Rate) of the generative rules.
# Citations: Wolfram (1984), Bagnoli et al. (1992), Sherrington (1991).

# Mapping TRM Output Heads (0-8) to Chaos Weights:
# Heads 0,1 (Class 1 - Stasis):   Wt 1.0  (Baseline measurement error)
# Heads 2,3 (Class 2 - Linear):   Wt 1.2  (Linear drift risk)
# Heads 4-6 (Class 3 - Chaotic):  Wt 10.0 (Exponential divergence / Fractal)
# Heads 7,8 (Class 4 - Complex):  Wt 3.0  (Non-linear interaction / Soliton)

CHAOS_WEIGHTS_TENSOR = torch.tensor([1.0, 1.0, 1.2, 1.2, 10.0, 10.0, 10.0, 3.0, 3.0])

# ==============================================================================
# 2. NEURAL ARCHITECTURE (The Sensor)
# ==============================================================================
class TinyRecursiveModel(nn.Module):
    """
    The AIT Physicist.
    Acts as the Observation Operator (Neural BDM) to collapse market superposition 
    into a topological state probability distribution.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 64)
        self.rnn = nn.GRU(64, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 9))
        
    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        _, h_n = self.rnn(encoded)
        return self.head(h_n.squeeze(0))


"""
==============================================================================
QCEA-AIXI AGENT SUBMISSION (INSTRUMENTED FOR FORENSIC LOGGING - CSV-FREE)
==============================================================================
Horizon 2: The Reflective Physicist
Maintainer: Algoplexity
Model: The Aletheia-Phronesis Architecture (with "Aletheia Box" Telemetry)

This version includes a dependency-free telemetry logger to act as a "flight
data recorder," capturing the agent's internal state during live competition
for post-mortem forensic analysis.
==============================================================================
"""



# ==============================================================================
# 3. THE AGENT CLASS (The Cybernetic Governor with Telemetry)
# ==============================================================================
class QCEAAgent(TrackerBase):
    def __init__(self, h=1):
        super().__init__(h)

        # --- A. CORE INITIALIZATION (Unchanged) ---
        self.history = []
        self.window = 30
        self.device = torch.device("cpu")
        self.physicist = TinyRecursiveModel().to(self.device)
        self.model_loaded = False

        # --- B. HOMEOSTATIC STATE (Unchanged) ---
        self.gamma = 10.0
        self.last_pred = None
        self.target_ll = -1.0
        self.learning_rate = 0.20

        # --- C. RESOURCE LOADING (Unchanged) ---
        possible_paths = [
            pathlib.Path('/workspace/resources'),
            pathlib.Path(__file__).parent / 'resources',
            pathlib.Path('.')
        ]
        for p in possible_paths:
            model_path = p /'reflective_gate.pth'
            if model_path.exists():
                try:
                    self.physicist.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.physicist.eval()
                    self.model_loaded = True
                    break
                except Exception as e:
                    print(f"⚠️ Model load failed: {e}")
        if not self.model_loaded:
            print("⚠️ CRITICAL: Running in Fallback Mode.")

        # --- D. NEW: THE ALETHEIA BOX (FLIGHT DATA RECORDER - CSV-FREE) ---
        self.telemetry_log_path = pathlib.Path('./telemetry.csv')
        self.telemetry_log_file = None
        self.log_header = [
            'timestamp', 'dove_location', 's2_rule_probs', 's2_entropy',
            's2_algo_multiplier', 's1_gamma', 's1_target_ll', 's1_realized_ll',
            'final_mu', 'final_sigma', 'sigma_component_theoretical',
            'sigma_component_kinetic_floor'
        ]
        try:
            self.telemetry_log_file = open(self.telemetry_log_path, 'w')
            self.telemetry_log_file.write(','.join(self.log_header) + '\n')
        except Exception as e:
            print(f"CRITICAL: Failed to initialize telemetry logger: {e}")
            self.telemetry_log_file = None

    def tick(self, p, m=None):
        """
        The Cybernetic Feedback Loop. This version is instrumented to log state
        before and after the reflective step.
        """
        # --- NEW: Initialize telemetry record for this tick ---
        telemetry_data = {h: None for h in self.log_header}
        telemetry_data['timestamp'] = p.get('time')
        val = p.get('dove_location')
        telemetry_data['dove_location'] = val

        # --- CORE TICK LOGIC (Unchanged) ---
        if val is None: return
        if isinstance(val, float) and np.isnan(val): return

        if self.last_pred is not None:
            mu, sigma = self.last_pred
            variance = max(sigma ** 2, 1e-9) # Defensive programming
            diff = val - mu
            try:
                ll = -0.5 * math.log(2 * math.pi * variance) - (0.5 * (diff**2) / variance)
            except ValueError:
                ll = -10.0
            
            # --- NEW: Log the realized "pain" signal ---
            telemetry_data['s1_realized_ll'] = ll

            # --- HOMEOSTATIC REGULATION (Unchanged) ---
            error = self.target_ll - ll
            if error > 0:
                self.gamma = self.gamma * (1.0 + self.learning_rate)
            else:
                self.gamma = self.gamma * 0.98
            self.gamma = max(min(self.gamma, 50.0), 1.0)

        # --- NEW: Store the partially filled telemetry data for predict() ---
        self.last_telemetry_data = telemetry_data

        # --- MEMORY UPDATE (Unchanged) ---
        self.history.append(float(val))
        self.add_to_quarantine(p['time'], val)
        self.pop_from_quarantine(p['time'])

    def _default_pred(self):
        loc = self.history[-1] if self.history else 0
        return {"type": "builtin", "name": "norm", "params": {"loc": loc, "scale": 20}}

    def predict(self):
        """
        The Policy Step. This version logs all internal state variables
        at the point of decision.
        """
        # --- WARMUP (Unchanged) ---
        if len(self.history) < self.window + 10:
            return self._default_pred()

        # --- PHASE 1 & 2: SENSING & INFERENCE (Unchanged) ---
        algo_multiplier = 2.0
        entropy = 2.0
        try:
            recent = pd.Series(self.history[-(self.window + 10):])
            vel = recent.diff()
            acc = vel.diff().dropna()
            bins = pd.qcut(acc.values, 4, labels=False, duplicates='drop')
            grid = np.eye(4)[bins.astype(int)][-self.window:]
            if len(grid) == self.window and self.model_loaded:
                t_grid = torch.FloatTensor(grid).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.physicist(t_grid)
                    probs = torch.softmax(logits, dim=1)
                p_np = probs.cpu().numpy()[0]
                entropy = -np.sum(p_np * np.log(p_np + 1e-9))
                algo_multiplier = float(torch.sum(probs * CHAOS_WEIGHTS_TENSOR.to(self.device)).item())
                if entropy > 1.5:
                    algo_multiplier *= 1.2
        except Exception:
            algo_multiplier = 5.0
            vel = pd.Series(self.history).diff()

        if self.model_loaded:
            k = 5.0
            threshold = 0.8
            w = 1.0 - (1.0 / (1.0 + np.exp(-k * (entropy - threshold))))
        else:
            w = 0.5

        # --- PHASE 3: ACT (Unchanged) ---
        v_curr = vel.iloc[-1]
        if pd.isna(v_curr): v_curr = 0.0
        current_vol = vel.std()
        if pd.isna(current_vol) or current_vol == 0: current_vol = 1.0
        mu_newton = self.history[-1] + v_curr
        mu_boltz = np.mean(self.history[-10:])
        final_mu = w * mu_newton + (1 - w) * mu_boltz

        # --- FINAL SIGMA CALCULATION (Unchanged) ---
        theoretical_sigma = current_vol * algo_multiplier * self.gamma
        current_price = self.history[-1]
        price_floor = abs(current_price * 0.005)
        kinetic_floor = abs(v_curr) * 2.0
        final_sigma = max(theoretical_sigma, price_floor, kinetic_floor, 2.0)

        # --- NEW: POPULATE AND WRITE THE FULL TELEMETRY RECORD (CSV-FREE) ---
        if hasattr(self, 'last_telemetry_data') and self.telemetry_log_file:
            telemetry = self.last_telemetry_data
            # Populate the rest of the dictionary
            telemetry['s2_rule_probs'] = probs.cpu().numpy()[0] if 'probs' in locals() else None
            telemetry['s2_entropy'] = entropy
            telemetry['s2_algo_multiplier'] = algo_multiplier
            telemetry['s1_gamma'] = self.gamma
            telemetry['s1_target_ll'] = self.target_ll
            telemetry['final_mu'] = final_mu
            telemetry['final_sigma'] = final_sigma
            telemetry['sigma_component_theoretical'] = theoretical_sigma
            telemetry['sigma_component_kinetic_floor'] = kinetic_floor
            
            # Manually construct and write the CSV string
            try:
                row_values = []
                for key in self.log_header:
                    value = telemetry.get(key)
                    if value is None:
                        row_values.append('')
                    elif isinstance(value, (np.ndarray, list, torch.Tensor)):
                        # Format list/array with '|' delimiter to avoid breaking CSV
                        formatted_list = '|'.join(f"{v:.6f}" for v in value)
                        row_values.append(f'"{formatted_list}"')
                    else:
                        row_values.append(str(value))
                self.telemetry_log_file.write(','.join(row_values) + '\n')
            except Exception as e:
                print(f"ERROR: Telemetry write failed: {e}")

        # --- FINAL STEPS (Unchanged) ---
        self.last_pred = (final_mu, final_sigma)
        return {
            "type": "builtin",
            "name": "norm",
            "params": {"loc": float(final_mu), "scale": float(final_sigma)}
        }

    def __del__(self):
        """Ensure the log file is properly closed when the agent is destroyed."""
        if hasattr(self, 'telemetry_log_file') and self.telemetry_log_file:
            try:
                self.telemetry_log_file.close()
            except:
                pass
