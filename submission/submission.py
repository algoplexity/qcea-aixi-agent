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
        
        # --- INITIALIZATION ---
        self.device = torch.device("cpu")
        self.physicist = TinyRecursiveModel().to(self.device)
        self.model_loaded = False
        
        # --- ROBUST RESOURCE LOADING ---
        # Scans valid paths for the Docker container environment
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
                    # print(f"✅ AIT Physicist loaded from {model_path}")
                    break
                except Exception as e:
                    print(f"⚠️ Model load failed: {e}")
        
        if not self.model_loaded:
            print("⚠️ CRITICAL: Running in Fallback Mode (Statistical Only).")

        # State Memory
        self.history = []
        self.window = 30

    def tick(self, p, m=None):
        """
        Information Ingestion (Law 1: Info-Action Cycle)
        """
        val = p.get('dove_location')
        
        # Handle Engine Initialization Artifacts (NaNs)
        if val is None: return
        if isinstance(val, float) and np.isnan(val): return
        
        self.history.append(float(val))
        
        # Engine housekeeping
        self.add_to_quarantine(p['time'], val)
        self.pop_from_quarantine(p['time'])

    def predict(self):
        """
        The Policy Step.
        Maps Physics (Entropy) -> Strategy (Sigma).
        """
        # Warmup / Fallback
        if len(self.history) < self.window + 10: 
            return self._default_pred()
            
        # --- PHASE 1: SENSING (The Physicist) ---
        try:
            # Second-Order MILS Encoding (Velocity)
            # We use Velocity for CrunchDAO as established in Notebook 2
            recent = pd.Series(self.history[-(self.window + 10):])
            vel = recent.diff()
            acc = vel.diff().dropna()
            
            bins = pd.qcut(acc.values, 4, labels=False, duplicates='drop')
            grid = np.eye(4)[bins.astype(int)][-self.window:]
            
            if len(grid) != self.window: return self._default_pred()
            
            # Inference
            t_grid = torch.FloatTensor(grid).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.physicist(t_grid)
                probs = torch.softmax(logits, dim=1)
                
            # METRIC: Shannon Entropy (Confusion)
            p_np = probs.numpy()[0]
            entropy = -np.sum(p_np * np.log(p_np + 1e-9))
                
        except: return self._default_pred()
            
        # --- PHASE 2: INFERENCE (The Reflex) ---
        # QCEA Law 9: Adaptive Need.
        # If Entropy is High, we distrust Trend (Newton).
        # If Entropy is Low, we trust Trend.
        
        if self.model_loaded:
            # Sigmoid Transfer Function
            # Threshold = 0.8 bits (Derived from Notebook 5)
            k = 5.0
            threshold = 0.8
            w = 1.0 - (1.0 / (1.0 + np.exp(-k * (entropy - threshold))))
        else:
            w = 0.5 # Fallback
            
        # --- PHASE 3: ACT (Ensemble Synthesis) ---
        v_curr = vel.iloc[-1]
        
        # Expert A: Newtonian (Inertia/Trend)
        # Rule 170 Logic: Project current velocity forward
        mu_newton = self.history[-1] + v_curr
        sigma_newton = max(vel.std(), 0.1) # Tight wings
        
        # Expert B: Boltzmann (Entropy/Chaos)
        # Rule 60 Logic: Mean Reversion to local average
        mu_boltz = np.mean(self.history[-10:])
        sigma_boltz = max(vel.std() * 4, 5.0) # Wide wings (Safety)
        
        # Synthesis
        final_mu = w * mu_newton + (1 - w) * mu_boltz
        final_sigma = w * sigma_newton + (1 - w) * sigma_boltz
        
        # QCEA Law 16: Survival Floor
        final_sigma = max(final_sigma, 0.1)
        
        return {
            "type": "builtin", 
            "name": "norm", 
            "params": {"loc": float(final_mu), "scale": float(final_sigma)}
        }

    def _default_pred(self):
        # Safe fallback
        loc = self.history[-1] if self.history else 0
        return {"type": "builtin", "name": "norm", "params": {"loc": loc, "scale": 10}}
