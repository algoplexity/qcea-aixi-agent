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
from birdgame.trackers.trackerbase import TrackerBase

# ==============================================================================
# 1. NEURAL ARCHITECTURE
# ==============================================================================

class TinyRecursiveModel(nn.Module):
    """The AIT Physicist (Sensor)"""
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
    """The Reflective Mind (Ensemble Gating)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 32), nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, rules):
        return self.net(rules)

# ==============================================================================
# 2. THE AGENT CLASS (Required by Platform)
# ==============================================================================

class QCEAAgent(TrackerBase):
    def __init__(self, h=1):
        super().__init__(h)
        
        # --- INITIALIZATION ---
        self.device = torch.device("cpu")
        self.physicist = TinyRecursiveModel().to(self.device)
        self.gate = ReflectiveGate().to(self.device)
        self.model_loaded = False
        
        # --- PATH FIX ---
        # The platform mounts resources at /workspace/resources
        # We check absolute path first, then relative as fallback
        possible_paths = [
            pathlib.Path('/workspace/resources'),
            pathlib.Path(__file__).parent / 'resources',
            pathlib.Path('.')
        ]
        
        resource_dir = None
        for p in possible_paths:
            if (p / 'trm_expert.pth').exists():
                resource_dir = p
                break
        
        if resource_dir:
            try:
                self.physicist.load_state_dict(torch.load(resource_dir / 'trm_expert.pth', map_location=self.device))
                self.gate.load_state_dict(torch.load(resource_dir / 'reflective_gate.pth', map_location=self.device))
                self.physicist.eval()
                
                # Online Learning Optimizer
                self.optimizer = torch.optim.Adam(self.gate.parameters(), lr=0.005)
                self.model_loaded = True
                # print("✅ QCEA Models Loaded.")
            except Exception as e:
                print(f"⚠️ Model Load Error: {e}")
        else:
            print("⚠️ Critical: Resource directory not found.")

        # State
        self.history = []
        self.window = 30
        self.last_state = None # (probs, mu, sigma)

    def tick(self, p, m=None):
        """
        Ingest new data and learn.
        """
        val = p.get('dove_location')
        
        # Handle Initialization Artifacts
        if val is None: return
        if isinstance(val, float) and np.isnan(val): return
        val = float(val)

        # --- A. ONLINE LEARNING ---
        if self.model_loaded and self.last_state is not None:
            try:
                probs, pred_mu, pred_sigma = self.last_state
                target_vel = val - self.history[-1]
                t_target = torch.FloatTensor([[target_vel]])
                
                # Re-run forward for gradient graph
                w = self.gate(probs)
                
                # Ensemble Logic Reconstruction
                last_v = self.history[-1] - self.history[-2]
                mu_n, sig_n = last_v, 1.0
                mu_b, sig_b = 0.0, 10.0
                
                mu_mix = w * mu_n + (1 - w) * mu_b
                sig_mix = w * sig_n + (1 - w) * sig_b
                
                var = sig_mix ** 2
                loss = 0.5 * (torch.log(var) + (t_target - mu_mix)**2 / var)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            except: pass

        # Update History
        self.history.append(val)
        
        # Required Engine Logic
        self.add_to_quarantine(p['time'], val)
        self.pop_from_quarantine(p['time'])

    def predict(self):
        """
        Output the distribution.
        """
        # Warmup / Fallback
        if len(self.history) < self.window + 10: 
            return self._default_pred()
            
        # --- B. SENSING ---
        try:
            recent = pd.Series(self.history[-(self.window + 10):])
            vel = recent.diff()
            acc = vel.diff().dropna()
            
            bins = pd.qcut(acc.values, 4, labels=False, duplicates='drop')
            grid = np.eye(4)[bins.astype(int)][-self.window:]
            
            if len(grid) != self.window: return self._default_pred()
            
            t_grid = torch.FloatTensor(grid).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.physicist(t_grid)
                probs = torch.softmax(logits, dim=1)
                
        except: return self._default_pred()
            
        # --- C. ACT ---
        if self.model_loaded:
            with torch.no_grad(): w = self.gate(probs).item()
        else: w = 0.5
            
        v_curr = vel.iloc[-1]
        
        # Newton
        mu_newton = self.history[-1] + v_curr
        sigma_newton = max(vel.std(), 0.1)
        
        # Boltzmann
        mu_boltz = np.mean(self.history[-10:])
        sigma_boltz = max(vel.std() * 4, 5.0)
        
        # Synthesis
        final_mu = w * mu_newton + (1 - w) * mu_boltz
        final_sigma = w * sigma_newton + (1 - w) * sigma_boltz
        
        # Store state
        if self.model_loaded:
            self.last_state = (probs.detach(), final_mu, final_sigma)
        
        final_sigma = max(final_sigma, 0.1)
        
        return {
            "type": "builtin", 
            "name": "norm", 
            "params": {"loc": float(final_mu), "scale": float(final_sigma)}
        }

    def _default_pred(self):
        loc = self.history[-1] if self.history else 0
        return {"type": "builtin", "name": "norm", "params": {"loc": loc, "scale": 10}}
