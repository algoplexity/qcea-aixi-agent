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

# FILE: submission.py

# ==============================================================================
# 0. IMPORTS (Moved to the top of the file)
# ==============================================================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import pathlib
import math
from birdgame.trackers.trackerbase import TrackerBase

# ==============================================================================
# 1. GLOBAL CONSTANTS & HYPERPARAMETERS
# ==============================================================================
# Now that 'torch' is imported, this line will execute correctly.
CHAOS_WEIGHTS_TENSOR = torch.tensor([1.0, 1.0, 1.2, 1.2, 10.0, 10.0, 10.0, 3.0, 3.0])
PRIME_NINE_RULES = [30, 54, 60, 90, 102, 110, 126, 150, 182]
RULE_ID_MAP = {rule: i for i, rule in enumerate(PRIME_NINE_RULES)}
INVERSE_RULE_ID_MAP = {i: rule for i, rule in enumerate(PRIME_NINE_RULES)}

# ==============================================================================
# 2. NEURAL ARCHITECTURES
# ==============================================================================
class TinyRecursiveModel(nn.Module):
    """ The AIT Physicist (Sensor) """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 64)
        self.rnn = nn.GRU(64, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 9))
    
    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        _, h_n = self.rnn(encoded)
        return self.head(h_n.squeeze(0))

class ReflectiveGate(nn.Module):
    """ The Ensemble Gating Network (Mind) """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# ==============================================================================
# 3. THE AGENT CLASS
# ==============================================================================
class QCEAAgent(TrackerBase):

    def __init__(self, h=1):
        super().__init__(h)
        self.history = []
        
        # --- Aletheia Box (Telemetry Recorder) ---
        self.telemetry = []
        self.step_counter = 0

        # --- LOAD MODELS (Corrected Pathing Logic) ---
        
        # This gets the directory of the current script: /workspace/submission/code/
        code_root = pathlib.Path(__file__).parent.resolve()
        
        # By calling .parent again, we go one level up to the submission root.
        # This should be: /workspace/submission/
        submission_root = code_root.parent

        self.physicist = TinyRecursiveModel()
        # Now, build the path from the correct submission root.
        # This will correctly resolve to: /workspace/submission/resources/trm_expert.pth
        physicist_path = os.path.join(submission_root, 'resources', 'trm_expert.pth')
        self.physicist.load_state_dict(torch.load(physicist_path))
        self.physicist.eval()

        self.gate = ReflectiveGate()
        # Do the same for the gate model.
        gate_path = os.path.join(submission_root, 'resources', 'reflective_gate.pth')
        self.gate.load_state_dict(torch.load(gate_path))
        self.gate.eval()

        # --- HOMEOSTATIC STATE (remains the same) ---
        self.gamma = 2.0
        self.last_pred = None
        self.target_ll = -1.0
        self.learning_rate = 0.05

    def tick(self, p, m=None):
        val = p.get('dove_location')
        if val is None or (isinstance(val, float) and np.isnan(val)): 
            # If the value is invalid, save telemetry state and exit
            self.save_telemetry_state(p['time'], None, None, None, None, None, None, None)
            return
        
        current_price = float(val)

        # --- LEARNING & REGULATION (The Feedback Loop) ---
        log_likelihood = None
        if self.last_pred is not None:
            mu, sigma = self.last_pred
            variance = sigma ** 2
            diff = current_price - mu
            ll = -0.5 * math.log(2 * math.pi * variance) - (0.5 * (diff**2) / variance)
            log_likelihood = ll
            
            error = self.target_ll - ll 
            if error > 0: 
                self.gamma = self.gamma * (1.0 + self.learning_rate)
            else: 
                self.gamma = self.gamma * (1.0 - (self.learning_rate / 2.0))
            self.gamma = max(min(self.gamma, 20.0), 1.0)
        
        self.history.append(current_price)
        self.add_to_quarantine(p['time'], current_price)
        self.pop_from_quarantine(p['time'])
        
        # We need the prediction for telemetry, so we call it here.
        # This is a bit inefficient but necessary for complete logging.
        pred_dict = self.predict()
        
        # Save full telemetry for this step
        if pred_dict is not None:
            self.save_telemetry_state(
                timestamp=p['time'],
                price=current_price,
                pred_mu=pred_dict['params']['loc'],
                pred_sigma=pred_dict['params']['scale'],
                log_likelihood=log_likelihood,
                gamma=self.gamma,
                ensemble_weight=pred_dict['internal_state']['ensemble_weight'],
                diagnosed_rule=pred_dict['internal_state']['diagnosed_rule']
            )
        else:
             self.save_telemetry_state(p['time'], current_price, None, None, log_likelihood, self.gamma, None, None)


    def predict(self):
        if len(self.history) < 21:
            return None 

        # --- SENSE (Feature Engineering) ---
        series = pd.Series(self.history)
        ret = series.pct_change().dropna()
        vel = ret.diff().dropna()
        acc = vel.diff().dropna()
        jerk = acc.diff().dropna()

        if len(jerk) < 20: return None
        
        # --- PERCEIVE (The AIT Physicist) ---
        features = pd.concat([ret, vel, acc, jerk], axis=1).iloc[-20:].values
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.physicist(features_tensor)
            rule_id = logits.argmax().item()
            diagnosed_rule = INVERSE_RULE_ID_MAP[rule_id]
            
            # --- REFLECT (The Gating Network) ---
            # Use softmax for probability distribution
            state_vector = torch.softmax(logits, dim=1)
            w_newton = self.gate(state_vector).item()

        # --- ACT (Adaptive Regulation) ---
        v_curr = vel.iloc[-1]
        current_vol = vel.std()
        
        mu_newton = self.history[-1] + v_curr
        mu_boltz = np.mean(self.history[-10:])
        
        final_mu = w_newton * mu_newton + (1 - w_newton) * mu_boltz
        
        base_sigma = max(current_vol, 1.0)
        final_sigma = base_sigma * self.gamma
        
        self.last_pred = (final_mu, final_sigma)
        
        return {
            "type": "builtin", 
            "name": "norm", 
            "params": {"loc": float(final_mu), "scale": float(final_sigma)},
            "internal_state": { # For telemetry
                "ensemble_weight": w_newton,
                "diagnosed_rule": diagnosed_rule
            }
        }
    
    def save_telemetry_state(self, timestamp, price, pred_mu, pred_sigma, log_likelihood, gamma, ensemble_weight, diagnosed_rule):
        self.telemetry.append({
            "step": self.step_counter,
            "timestamp": timestamp,
            "price": price,
            "pred_mu": pred_mu,
            "pred_sigma": pred_sigma,
            "log_likelihood": log_likelihood,
            "gamma": gamma,
            "ensemble_weight": ensemble_weight,
            "diagnosed_rule": diagnosed_rule,
        })
        self.step_counter += 1

    def close(self):
        telemetry_df = pd.DataFrame(self.telemetry)
        # The platform should provide a writable directory.
        # Saving to '/workspace/submission/code/' which is usually the CWD.
        output_path = '/workspace/submission/code/telemetry.csv'
        telemetry_df.to_csv(output_path, index=False)
        print(f"Telemetry saved to {output_path}")
