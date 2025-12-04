"""
QCEA-AIXI AGENT SUBMISSION
Horizon 2: The Reflective Physicist
Horizon 2 QCEA-AIXI Agent (Ensemble Gate + Prime 9 Sensor)
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import pathlib

# ==============================================================================
# 1. NEURAL ARCHITECTURE
# ==============================================================================

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

class ReflectiveGate(nn.Module):
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
# 2. INITIALIZATION (Updated for 'resources' folder)
# ==============================================================================

device = torch.device("cpu")
physicist = TinyRecursiveModel().to(device)
gate = ReflectiveGate().to(device)

# --- PATH FIX ---
# We look for weights inside the 'resources' subdirectory
BASE_DIR = pathlib.Path(__file__).parent.absolute()
RESOURCES_DIR = BASE_DIR / 'resources'

try:
    physicist.load_state_dict(torch.load(RESOURCES_DIR / 'trm_expert.pth', map_location=device))
    gate.load_state_dict(torch.load(RESOURCES_DIR / 'reflective_gate.pth', map_location=device))
    physicist.eval()
    optimizer = torch.optim.Adam(gate.parameters(), lr=0.005)
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    print(f"⚠️ Critical Warning: Models not found in {RESOURCES_DIR}. Error: {e}")

# ==============================================================================
# 3. AGENT STATE
# ==============================================================================
history = []
window = 30
last_state = None 

# ==============================================================================
# 4. PREDICTION LOOP
# ==============================================================================

def get_prediction(observation):
    global history, last_state, MODEL_LOADED
    
    val = observation.get('dove_location')
    if val is None or (isinstance(val, float) and np.isnan(val)): return default_pred()
    val = float(val)
    
    # A. ONLINE LEARNING
    if MODEL_LOADED and last_state is not None:
        try:
            probs, pred_mu, pred_sigma = last_state
            target_vel = val - history[-1]
            t_target = torch.FloatTensor([[target_vel]])
            
            w = gate(probs)
            last_v = history[-1] - history[-2]
            mu_n, sig_n = last_v, 1.0
            mu_b, sig_b = 0.0, 10.0
            mu_mix = w * mu_n + (1 - w) * mu_b
            sig_mix = w * sig_n + (1 - w) * sig_b
            
            var = sig_mix ** 2
            loss = 0.5 * (torch.log(var) + (t_target - mu_mix)**2 / var)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except: pass

    history.append(val)
    if len(history) < window + 10: return default_pred()
    
    # B. SENSING
    try:
        recent = pd.Series(history[-(window + 10):])
        vel = recent.diff()
        acc = vel.diff().dropna()
        bins = pd.qcut(acc.values, 4, labels=False, duplicates='drop')
        grid = np.eye(4)[bins.astype(int)][-window:]
        if len(grid) != window: return default_pred()
        
        t_grid = torch.FloatTensor(grid).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = physicist(t_grid)
            probs = torch.softmax(logits, dim=1)
            
    except: return default_pred()
        
    # C. ACT
    if MODEL_LOADED:
        with torch.no_grad(): w = gate(probs).item()
    else: w = 0.5
        
    v_curr = vel.iloc[-1]
    mu_newton = history[-1] + v_curr
    sigma_newton = max(vel.std(), 0.1)
    
    mu_boltz = np.mean(history[-10:])
    sigma_boltz = max(vel.std() * 4, 5.0)
    
    final_mu = w * mu_newton + (1 - w) * mu_boltz
    final_sigma = w * sigma_newton + (1 - w) * sigma_boltz
    
    if MODEL_LOADED:
        last_state = (probs.detach(), final_mu, final_sigma)
    
    final_sigma = max(final_sigma, 0.1)
    
    return {"type": "builtin", "name": "norm", "params": {"loc": float(final_mu), "scale": float(final_sigma)}}

def default_pred():
    loc = history[-1] if history else 0
    return {"type": "builtin", "name": "norm", "params": {"loc": loc, "scale": 10}}
