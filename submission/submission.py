"""
QCEA-AIXI AGENT SUBMISSION
Horizon 2: The Reflective Physicist
Maintainer: Algoplexity
Horizon 2 QCEA-AIXI Agent (Ensemble Gate + Prime 9 Sensor)
Model: The Aletheia-Phronesis Architecture
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
# 2. THE AGENT CLASS (The Adaptive Governor)
# ==============================================================================
class QCEAAgent(TrackerBase):
    def __init__(self, h=1):
        super().__init__(h)
        
        # --- A. INITIALIZATION & MEMORY (Fixes AttributeError) ---
        self.history = []       # The raw time series
        self.window = 30        # The "Perception" Window
        self.device = torch.device("cpu")
        self.physicist = TinyRecursiveModel().to(self.device)
        self.model_loaded = False
        
        # --- B. HOMEOSTATIC STATE (Cybernetic Regulation) ---
        # Instead of magic numbers, we learn the "Panic Factor" (Gamma)
        self.gamma = 2.0        # Starts conservative (2x Volatility)
        self.last_pred = None   # Stores (mu, sigma) of the previous step for feedback
        self.target_ll = -1.0   # The "Comfort Zone" (Target Log-Likelihood)
        self.learning_rate = 0.05 # How fast we adapt to pain
        
        # --- C. ROBUST RESOURCE LOADING ---
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
            print("⚠️ CRITICAL: Running in Fallback Mode (Statistical Only).")

    def tick(self, p, m=None):
        """
        Information Ingestion & Cybernetic Feedback Loop
        """
        val = p.get('dove_location')
        
        # 1. Validation
        if val is None: return
        if isinstance(val, float) and np.isnan(val): return
        
        # 2. THE REFLECTIVE LOOP (QCEA Law 8: Subjectivity)
        # Evaluate how much "pain" the last prediction caused.
        if self.last_pred is not None:
            mu, sigma = self.last_pred
            
            # Calculate Log-Likelihood of the ACTUAL value given our PREDICTION
            # LL = -0.5 * log(2*pi*sigma^2) - (error^2 / 2*sigma^2)
            variance = sigma ** 2
            diff = val - mu
            try:
                ll = -0.5 * math.log(2 * math.pi * variance) - (0.5 * (diff**2) / variance)
            except ValueError:
                ll = -10.0 # Punishment for impossible variance
            
            # 3. HOMEOSTATIC ADJUSTMENT
            # Error > 0 means we are performing WORSE than target (Pain).
            # Error < 0 means we are performing BETTER than target (Comfort).
            error = self.target_ll - ll 
            
            if error > 0: 
                # We are hurting: Panic Fast (Increase Gamma)
                self.gamma = self.gamma * (1.0 + self.learning_rate)
            else: 
                # We are safe: Relax Slow (Decrease Gamma)
                self.gamma = self.gamma * (1.0 - (self.learning_rate / 2.0))
                
            # Clamp Gamma to sane limits (Never below 1.0, never above 20.0)
            self.gamma = max(min(self.gamma, 20.0), 1.0)

        # 4. Update Memory
        self.history.append(float(val))
        
        # Engine housekeeping
        self.add_to_quarantine(p['time'], val)
        self.pop_from_quarantine(p['time'])

    def predict(self):
        """
        The Policy Step: Aletheia (Un-concealment) + Phronesis (Prudence)
        """
        # Warmup / Fallback
        if len(self.history) < self.window + 10: 
            return self._default_pred()
            
        # --- PHASE 1: SENSING (AIT Physicist) ---
        try:
            recent = pd.Series(self.history[-(self.window + 10):])
            vel = recent.diff()
            acc = vel.diff().dropna()
            
            # MILS Quantile Binning
            bins = pd.qcut(acc.values, 4, labels=False, duplicates='drop')
            grid = np.eye(4)[bins.astype(int)][-self.window:]
            
            if len(grid) != self.window: return self._default_pred()
            
            # Inference
            t_grid = torch.FloatTensor(grid).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.physicist(t_grid)
                probs = torch.softmax(logits, dim=1)
                
            # Metric: Entropy (The "Confusion" Signal)
            p_np = probs.numpy()[0]
            entropy = -np.sum(p_np * np.log(p_np + 1e-9))
                
        except: 
            # If sensing fails, default to max entropy
            entropy = 2.0
            vel = pd.Series(self.history).diff()
            
        # --- PHASE 2: INFERENCE (Weighting) ---
        if self.model_loaded:
            # Low Entropy -> Trust Trend (w=1)
            # High Entropy -> Trust Mean Reversion (w=0)
            k = 5.0
            threshold = 0.8
            w = 1.0 - (1.0 / (1.0 + np.exp(-k * (entropy - threshold))))
        else:
            w = 0.5 
            
        # --- PHASE 3: ACT (Adaptive Regulation) ---
        v_curr = vel.iloc[-1]
        current_vol = vel.std()
        if pd.isna(current_vol) or current_vol == 0: current_vol = 1.0
        
        # Expert A: Newton (Trend)
        mu_newton = self.history[-1] + v_curr
        
        # Expert B: Boltzmann (Mean Reversion)
        mu_boltz = np.mean(self.history[-10:])
        
        # Synthesis
        final_mu = w * mu_newton + (1 - w) * mu_boltz
        
        # --- THE ADAPTIVE VARIANCE (No Magic Numbers) ---
        # We rely on the "Gamma" learned in the tick() loop.
        # This scales the empirical volatility based on recent pain.
        base_sigma = max(current_vol, 1.0)
        final_sigma = base_sigma * self.gamma
        
        # Store for next tick's feedback loop
        self.last_pred = (final_mu, final_sigma)
        
        return {
            "type": "builtin", 
            "name": "norm", 
            "params": {"loc": float(final_mu), "scale": float(final_sigma)}
        }
    
    def _default_pred(self):
        # Safe fallback
        loc = self.history[-1] if self.history else 0
        return {"type": "builtin", "name": "norm", "params": {"loc": loc, "scale": 10}}
