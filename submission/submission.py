"""
QCEA-AIXI AGENT SUBMISSION
Horizon 2: The Reflective Physicist
Maintainer: Algoplexity
Horizon 2 QCEA-AIXI Agent (Ensemble Gate + Prime 9 Sensor)
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
# 1. NEURAL ARCHITECTURE (The Sensor)
# ==============================================================================
class TinyRecursiveModel(nn.Module):
    """
    The AIT Physicist.
    Acts as the Observation Operator to collapse market superposition 
    into a topological state (Rule ID).
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

# ==============================================================================
# 2. THE AGENT CLASS
# ==============================================================================
class QCEAAgent(TrackerBase):
    def __init__(self, h=1):
        super().__init__(h)
        # ... (Load Model and Init History as before) ...
        
        # --- HOMEOSTATIC STATE ---
        self.gamma = 2.0        # The "Panic Factor" (Starts conservative)
        self.last_pred = None   # To store (mu, sigma) of the previous step
        self.target_ll = -1.0   # The "Comfort Zone" (Acceptable Likelihood)
        self.learning_rate = 0.05

    def tick(self, p, m=None):
        val = p.get('dove_location')
        if val is None or (isinstance(val, float) and np.isnan(val)): return
        
        # 1. EVALUATE PREVIOUS PREDICTION (The Feedback Loop)
        if self.last_pred is not None:
            mu, sigma = self.last_pred
            # Calculate Log-Likelihood of the actual value given our Gaussian
            # LL = -0.5 * log(2*pi) - log(sigma) - 0.5 * ((x - mu)/sigma)^2
            variance = sigma ** 2
            diff = val - mu
            ll = -0.5 * math.log(2 * math.pi * variance) - (0.5 * (diff**2) / variance)
            
            # 2. ADJUST GAMMA (The Homeostatic Regulator)
            # If LL is worse than target (e.g. -5.0 vs -1.0), error is negative.
            # We want Gamma to INCREASE when performance is BAD.
            error = self.target_ll - ll 
            
            if error > 0: # We are hurting (Performance is worse than target)
                # Panic fast: Increase Gamma
                self.gamma = self.gamma * (1.0 + self.learning_rate)
            else: # We are safe (Performance is better than target)
                # Relax slow: Decrease Gamma
                self.gamma = self.gamma * (1.0 - (self.learning_rate / 2.0))
                
            # Clamp Gamma to sane limits (e.g., never below 1.0, never above 20.0)
            self.gamma = max(min(self.gamma, 20.0), 1.0)

        # 3. Update History
        self.history.append(float(val))
        self.add_to_quarantine(p['time'], val)
        self.pop_from_quarantine(p['time'])

    def predict(self):
        # ... (Warmup and Sensing Phase remain the same) ...
        
        # --- PHASE 3: ACT (Adaptive Regulation) ---
        v_curr = vel.iloc[-1]
        current_vol = vel.std()
        
        # Base Experts (Notice: No magic multipliers here, just raw physics)
        mu_newton = self.history[-1] + v_curr
        mu_boltz = np.mean(self.history[-10:])
        
        # Weighted Synthesis
        final_mu = w * mu_newton + (1 - w) * mu_boltz
        
        # --- THE ADAPTIVE VARIANCE ---
        # Instead of fixed multipliers, we use the learnt Gamma.
        # Gamma represents "How many standard deviations of safety do I need right now?"
        
        # Base Sigma is the realized volatility
        base_sigma = max(current_vol, 1.0)
        
        # Apply the Panic Factor
        final_sigma = base_sigma * self.gamma
        
        # Store prediction for next tick's evaluation
        self.last_pred = (final_mu, final_sigma)
        
        return {
            "type": "builtin", 
            "name": "norm", 
            "params": {"loc": float(final_mu), "scale": float(final_sigma)}
        }
