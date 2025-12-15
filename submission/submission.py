"""
QCEA-AIXI AGENT SUBMISSION
Horizon 2: The Reflective Physicist  (Neural BDM & Homeostatic Regulation)
Maintainer: Algoplexity
Model: The Aletheia-Phronesis Architecture

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

# ==============================================================================
# 3. THE AGENT CLASS (The Cybernetic Governor)
# ==============================================================================
class QCEAAgent(TrackerBase):
    def __init__(self, h=1):
        super().__init__(h)
        
        # --- A. INITIALIZATION ---
        self.history = []       
        self.window = 30        
        self.device = torch.device("cpu")
        self.physicist = TinyRecursiveModel().to(self.device)
        self.model_loaded = False
        
        # --- B. HOMEOSTATIC STATE (Adaptive, Not Magic) ---
        # 1. Paranoid Initialization: Start High (Gamma=10).
        # We assume the environment is hostile until proven safe.
        self.gamma = 10.0        
        
        # 2. Feedback Variables
        self.last_pred = None   
        self.target_ll = -1.0   # Target Log-Likelihood (Survival Threshold)
        
        # 3. Fast Adaptation
        # High learning rate to react instantly to regime shifts.
        self.learning_rate = 0.20 
        
        # --- C. RESOURCE LOADING ---
        possible_paths = [
            pathlib.Path('/workspace/resources'),
            pathlib.Path(__file__).parent / 'resources',
            pathlib.Path('.')
        ]
        
        for p in possible_paths:
            model_path = p / 'trm_expert.pth'
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

    def tick(self, p, m=None):
        """
        The Cybernetic Feedback Loop (QCEA Law 8 & 16).
        Adjusts internal anxiety (Gamma) based on external pain (Likelihood).
        """
        val = p.get('dove_location')
        if val is None: return
        if isinstance(val, float) and np.isnan(val): return
        
        # --- REFLECTIVE STEP ---
        if self.last_pred is not None:
            mu, sigma = self.last_pred
            
            # Calculate realized Log-Likelihood
            variance = sigma ** 2
            diff = val - mu
            try:
                ll = -0.5 * math.log(2 * math.pi * variance) - (0.5 * (diff**2) / variance)
            except ValueError:
                ll = -10.0 
            
            # --- HOMEOSTATIC REGULATION ---
            # Error > 0: We are performing WORSE than target -> Increase Gamma
            # Error < 0: We are performing BETTER than target -> Decrease Gamma
            error = self.target_ll - ll 
            
            if error > 0: 
                # Pain: Exponential expansion (Survival Reflex)
                self.gamma = self.gamma * (1.0 + self.learning_rate)
            else: 
                # Comfort: Linear relaxation (Greed)
                self.gamma = self.gamma * 0.98
                
            # Bounds: 
            # Lower bound 1.0 (Physical Limit). 
            # Upper bound 50.0 (Cap for extreme crashes).
            self.gamma = max(min(self.gamma, 50.0), 1.0)

        # Update Memory
        self.history.append(float(val))
        self.add_to_quarantine(p['time'], val)
        self.pop_from_quarantine(p['time'])

    def predict(self):
        """
        The Policy Step: Combining Neural BDM with Adaptive Regulation.
        """
        # Warmup
        if len(self.history) < self.window + 10: 
            return self._default_pred()
            
        # --- PHASE 1: SENSING (Neural BDM) ---
        algo_multiplier = 2.0 # Fallback
        entropy = 2.0         # Fallback
        
        try:
            recent = pd.Series(self.history[-(self.window + 10):])
            vel = recent.diff()
            acc = vel.diff().dropna()
            
            # MILS Encoding
            bins = pd.qcut(acc.values, 4, labels=False, duplicates='drop')
            grid = np.eye(4)[bins.astype(int)][-self.window:]
            
            if len(grid) == self.window and self.model_loaded:
                # TRM Inference
                t_grid = torch.FloatTensor(grid).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.physicist(t_grid)
                    probs = torch.softmax(logits, dim=1) # Shape [1, 9]
                
                # A. Calculate Entropy (Confusion Signal)
                p_np = probs.numpy()[0]
                entropy = -np.sum(p_np * np.log(p_np + 1e-9))
                
                # B. Calculate Algorithmic Volatility (The Physics Constant)
                # Weighted sum of probabilities * Lyapunov weights
                # This replaces the magic numbers with physics.
                algo_multiplier = float(torch.sum(probs * CHAOS_WEIGHTS_TENSOR.to(self.device)).item())
                
                # Penalty for Confusion (High Entropy -> Widen further)
                if entropy > 1.5:
                    algo_multiplier *= 1.2
                
        except Exception: 
            # If sensing fails, assume chaos
            algo_multiplier = 5.0
            vel = pd.Series(self.history).diff()
            
        # --- PHASE 2: INFERENCE (Ensemble Weighting) ---
        # Still use Entropy to blend Newton (Trend) vs Boltzmann (Mean Rev)
        k = 5.0
        threshold = 0.8
        w = 1.0 - (1.0 / (1.0 + np.exp(-k * (entropy - threshold))))
            
        # --- PHASE 3: ACT (The Algorithmic Governor) ---
        v_curr = vel.iloc[-1]
        current_vol = vel.std()
        if pd.isna(current_vol) or current_vol == 0: current_vol = 1.0
        
        # Experts
        mu_newton = self.history[-1] + v_curr
        mu_boltz = np.mean(self.history[-10:])
        
        # Synthesis
        final_mu = w * mu_newton + (1 - w) * mu_boltz
        
        # --- FINAL SIGMA CALCULATION ---
        # Formula: Sigma = Volatility * Physics_Constant * Panic_Factor
        # 1. Base Volatility (Empirical)
        # 2. Algo Multiplier (Theoretical Risk from Rule Class)
        # 3. Gamma (Adaptive Risk from Recent Performance)
        
        raw_sigma = current_vol * algo_multiplier * self.gamma
        
        # Absolute Floor (0.5% of Price) - The ultimate backstop
        current_price = self.history[-1]
        abs_floor = abs(current_price * 0.005)
        
        final_sigma = max(raw_sigma, abs_floor, 2.0)
        
        # Store for feedback
        self.last_pred = (final_mu, final_sigma)
        
        return {
            "type": "builtin", 
            "name": "norm", 
            "params": {"loc": float(final_mu), "scale": float(final_sigma)}
        }
    
    def _default_pred(self):
        loc = self.history[-1] if self.history else 0
        return {"type": "builtin", "name": "norm", "params": {"loc": loc, "scale": 20}}
