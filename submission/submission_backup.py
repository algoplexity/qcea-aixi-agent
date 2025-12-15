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

# FILE: submission.py (Final Version with Telemetry)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import pathlib
import math
from collections import deque

from birdgame.trackers.trackerbase import TrackerBase

class TinyRecursiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 64)
        self.rnn = nn.GRU(64, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 9))
    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        _, h_n = self.rnn(encoded)
        return self.head(h_n.squeeze(0))

class ReflectiveHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.mu = nn.Linear(32, 1)
        self.sigma = nn.Linear(32, 1)
    def forward(self, rules, vel):
        combined = torch.cat([rules, vel], dim=1)
        h = self.net(combined)
        mu = self.mu(h)
        log_sigma = self.sigma(h)
        log_sigma = torch.clamp(log_sigma, min=-3.0, max=5.0)
        return mu, log_sigma

# ==============================================================================
# THE AGENT CLASS (with Telemetry Recorder)
# ==============================================================================
class QCEAAgent(TrackerBase):
    def __init__(self, h=1):
        super().__init__(h)

        self.history = []
        self.window = 30
        self.device = torch.device("cpu")
        
        self.physicist = TinyRecursiveModel().to(self.device)
        self.head = ReflectiveHead().to(self.device)
        self.physicist_loaded = False
        self.head_loaded = False
        
        # --- LOAD RESOURCES ---
        resources_path = self._find_resources_path()
        if resources_path:
            # Load Physicist
            try:
                physicist_path = resources_path / 'trm_expert.pth'
                self.physicist.load_state_dict(torch.load(physicist_path, map_location=self.device))
                self.physicist.eval()
                self.physicist_loaded = True
                print("INFO: AIT Physicist (trm_expert.pth) loaded.")
            except Exception as e:
                print(f"⚠️ Physicist load failed: {e}")
            # Load Head
            try:
                head_path = resources_path / 'reflective_gate.pth'
                self.head.load_state_dict(torch.load(head_path, map_location=self.device))
                self.head.eval()
                self.head_loaded = True
                print("INFO: Reflective Head (reflective_gate.pth) loaded.")
            except Exception as e:
                print(f"⚠️ Head load failed: {e}")
        
        self.fallback_mode = not (self.physicist_loaded and self.head_loaded)
        if self.fallback_mode:
            print("⚠️ CRITICAL: One or more models failed. Running in Fallback Mode.")

        # --- HOMEOSTATIC STATE ---
        self.gamma = 10.0
        self.last_pred = None
        self.target_ll = -1.0
        self.learning_rate = 0.20

        # --- NEW: ADDED BACK THE ALETHEIA BOX TELEMETRY RECORDER ---
        self.telemetry_log_path = pathlib.Path('./telemetry.csv')
        self.telemetry_log_file = None
        self.log_header = [
            'timestamp', 'dove_location', 's2_rule_probs', 's2_entropy',
            's2_algo_multiplier', 's1_gamma', 's1_target_ll', 's1_realized_ll',
            'pred_pos', 'neural_sigma', 'final_sigma'
        ]
        try:
            # Use 'w' mode to create/overwrite the file at the start of each run
            self.telemetry_log_file = open(self.telemetry_log_path, 'w')
            self.telemetry_log_file.write(','.join(self.log_header) + '\n')
            print("INFO: Telemetry logger initialized.")
        except Exception as e:
            print(f"CRITICAL: Failed to initialize telemetry logger: {e}")
            self.telemetry_log_file = None

    def _find_resources_path(self):
        # ... (implementation as before) ...
        for p in [pathlib.Path('/workspace/resources'), pathlib.Path('.')]:
            if p.exists(): return p
        return None

    def tick(self, p, m=None):
        telemetry_data = {h: None for h in self.log_header}
        telemetry_data['timestamp'] = p.get('time')
        val = p.get('dove_location')
        telemetry_data['dove_location'] = val

        if val is None or (isinstance(val, float) and np.isnan(val)): return
        
        if self.last_pred is not None:
            mu, sigma = self.last_pred
            variance = max(sigma ** 2, 1e-9)
            diff = val - mu
            try:
                ll = -0.5 * math.log(2 * math.pi * variance) - (0.5 * (diff**2) / variance)
            except ValueError:
                ll = -10.0
            
            telemetry_data['s1_realized_ll'] = ll

            error = self.target_ll - ll
            if error > 0:
                self.gamma = self.gamma * (1.0 + self.learning_rate)
            else:
                self.gamma = self.gamma * 0.98
            self.gamma = max(min(self.gamma, 50.0), 1.0)
        
        self.last_telemetry_data = telemetry_data
        self.history.append(float(val))
        self.add_to_quarantine(p['time'], val)
        self.pop_from_quarantine(p['time'])

    def _default_pred(self):
        loc = self.history[-1] if self.history else 0
        return {"type": "builtin", "name": "norm", "params": {"loc": loc, "scale": 20}}

    def predict(self):
        if len(self.history) < self.window + 10 or self.fallback_mode:
            return self._default_pred()

        pred_pos, neural_sigma, entropy, algo_multiplier = 0.0, 1.0, 2.0, 2.0
        probs_for_log = None
        try:
            # Phase 1: Sensing
            recent = pd.Series(self.history[-(self.window + 10):])
            vel = recent.diff()
            acc = vel.diff().dropna()
            bins = pd.qcut(acc.values, 4, labels=False, duplicates='drop')
            grid = np.eye(4)[bins.astype(int)][-self.window:]
            
            t_grid = torch.FloatTensor(grid).unsqueeze(0).to(self.device)
            t_vel = torch.FloatTensor([vel.iloc[-1]]).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.physicist(t_grid)
                probs = torch.softmax(logits, dim=1)
                probs_for_log = probs.cpu().numpy()[0]
                
                # Phase 2: Laws
                pred_vel, pred_logvar = self.head(probs, t_vel)
                neural_sigma = np.exp(0.5 * pred_logvar.item())
            
            p_np = probs.cpu().numpy()[0]
            entropy = -np.sum(p_np * np.log(p_np + 1e-9))
            
            # Phase 3: Action
            pred_pos = self.history[-1] + pred_vel.item()
            algo_multiplier = float(torch.sum(probs * CHAOS_WEIGHTS_TENSOR.to(self.device)).item())
            if entropy > 1.5: algo_multiplier *= 1.2
            
        except Exception as e:
            print(f"ERROR during prediction logic: {e}")
            return self._default_pred()
        
        current_vol = vel.std() if not pd.isna(vel.std()) else 1.0
        theoretical_sigma = current_vol * algo_multiplier * self.gamma
        price_floor = abs(self.history[-1] * 0.005)
        kinetic_floor = abs(vel.iloc[-1]) * 2.0
        final_sigma = max(theoretical_sigma, price_floor, kinetic_floor, 2.0, neural_sigma)

        # --- TELEMETRY WRITE ---
        if hasattr(self, 'last_telemetry_data') and self.telemetry_log_file:
            telemetry = self.last_telemetry_data
            telemetry['s2_rule_probs'] = probs_for_log
            telemetry['s2_entropy'] = entropy
            telemetry['s2_algo_multiplier'] = algo_multiplier
            telemetry['s1_gamma'] = self.gamma
            telemetry['s1_target_ll'] = self.target_ll
            telemetry['pred_pos'] = pred_pos
            telemetry['neural_sigma'] = neural_sigma
            telemetry['final_sigma'] = final_sigma
            
            try:
                row_values = []
                for key in self.log_header:
                    value = telemetry.get(key)
                    if value is None: row_values.append('')
                    elif isinstance(value, (np.ndarray, list)):
                        formatted_list = '|'.join(f"{v:.6f}" for v in value)
                        row_values.append(f'"{formatted_list}"')
                    else: row_values.append(str(value))
                self.telemetry_log_file.write(','.join(row_values) + '\n')
            except Exception as e:
                print(f"ERROR: Telemetry write failed: {e}")

        self.last_pred = (pred_pos, final_sigma)
        return {
            "type": "builtin",
            "name": "norm",
            "params": {"loc": float(pred_pos), "scale": float(final_sigma)}
        }

    def __del__(self):
        if hasattr(self, 'telemetry_log_file') and self.telemetry_log_file:
            try:
                self.telemetry_log_file.close()
                print("INFO: Telemetry file closed.")
            except: pass

