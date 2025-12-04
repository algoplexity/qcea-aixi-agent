"""
QCEA-AIXI AGENT SUBMISSION
Horizon 2: The Reflective Physicist
Maintainer: Algoplexity
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import pathlib

# ==============================================================================
# 1. NEURAL ARCHITECTURE (Must match Training Exactly)
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
        # Input: 9 Rule Probabilities
        # Output: 1 Mixing Weight (0=Entropy, 1=Inertia)
        self.net = nn.Sequential(
            nn.Linear(9, 32), nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, rules):
        return self.net(rules)

# ==============================================================================
# 2. INITIALIZATION (Cold Start)
# ==============================================================================

# Force CPU for stability and latency guarantees (<50ms)
device = torch.device("cpu")

physicist = TinyRecursiveModel().to(device)
gate = ReflectiveGate().to(device)

# Load Weights relative to this script
BASE_DIR = pathlib.Path(__file__).parent.absolute()

try:
    physicist.load_state_dict(torch.load(BASE_DIR / 'trm_expert.pth', map_location=device))
    gate.load_state_dict(torch.load(BASE_DIR / 'reflective_gate.pth', map_location=device))
    physicist.eval()
    
    # QCEA Law 2: Strategic Systems must adapt.
    # We keep the Gate trainable for Online Learning.
    # We use a conservative LR to prevent "Seizures" in production.
    optimizer = torch.optim.Adam(gate.parameters(), lr=0.005)
    
    MODEL_LOADED = True
except Exception as e:
    # Fail Gracefully (QCEA Law 16: Survival First)
    MODEL_LOADED = False
    print(f"⚠️ Critical Warning: Models not loaded ({e}). Running in Fallback Mode.")

# ==============================================================================
# 3. THE AGENT STATE
# ==============================================================================
history = []
window = 30
last_state = None # Stores (probs, mu_mix, sigma_mix) for learning

# ==============================================================================
# 4. THE MAIN LOOP (Information-Action Cycle)
# ==============================================================================

def get_prediction(observation):
    """
    CrunchDAO Entry Point.
    Input: Dict with 'dove_location', 'time', etc.
    Output: Dict with distribution params.
    """
    global history, last_state, MODEL_LOADED
    
    val = observation.get('dove_location')
    
    # Handle Engine Initialization Artifacts (NaNs)
    if val is None: return default_pred()
    if isinstance(val, float) and np.isnan(val): return default_pred()
    
    val = float(val)
    
    # --- A. LEARNING (Self-Reinforcement) ---
    # Update the Mind based on the error of the previous prediction
    if MODEL_LOADED and last_state is not None:
        try:
            probs, pred_mu, pred_sigma = last_state
            
            # Target: The deviation from the previous step
            target_vel = val - history[-1]
            t_target = torch.FloatTensor([[target_vel]])
            
            # Re-Calculate Ensemble Logic for Gradient Graph
            # (We re-run the forward pass to connect the graph)
            w = gate(probs)
            
            last_v = history[-1] - history[-2]
            
            # Expert A: Newtonian (Inertia)
            mu_n, sig_n = last_v, 1.0
            
            # Expert B: Boltzmann (Entropy)
            mu_b, sig_b = 0.0, 10.0
            
            # The Mix
            mu_mix = w * mu_n + (1 - w) * mu_b
            sig_mix = w * sig_n + (1 - w) * sig_b
            
            # NLL Loss (Minimizing Surprise)
            var = sig_mix ** 2
            loss = 0.5 * (torch.log(var) + (t_target - mu_mix)**2 / var)
            
            # Update Weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        except Exception:
            pass # Never crash in production

    # Update History
    history.append(val)
    
    # Warmup check
    if len(history) < window + 10: return default_pred()
    
    # --- B. SENSING (The AIT Physicist) ---
    # Second-Order MILS Encoding (Acceleration)
    try:
        recent = pd.Series(history[-(window + 10):])
        vel = recent.diff()
        acc = vel.diff().dropna()
        
        # MILS Quantile Binning
        bins = pd.qcut(acc.values, 4, labels=False, duplicates='drop')
        grid = np.eye(4)[bins.astype(int)][-window:]
        
        if len(grid) != window: return default_pred()
        
        # Inference
        t_grid = torch.FloatTensor(grid).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = physicist(t_grid)
            probs = torch.softmax(logits, dim=1)
            
    except:
        return default_pred()
        
    # --- C. INFER & ACT (The Ensemble) ---
    # 1. Gating
    if MODEL_LOADED:
        with torch.no_grad():
            w = gate(probs).item()
    else:
        w = 0.5 # Fallback mix
        
    # 2. Experts
    v_curr = vel.iloc[-1]
    
    # Newton
    mu_newton = history[-1] + v_curr
    sigma_newton = max(vel.std(), 0.1)
    
    # Boltzmann
    mu_boltz = np.mean(history[-10:])
    sigma_boltz = max(vel.std() * 4, 5.0)
    
    # 3. Synthesis
    final_mu = w * mu_newton + (1 - w) * mu_boltz
    final_sigma = w * sigma_newton + (1 - w) * sigma_boltz
    
    # Store state for next learning step (Detach to stop graph growth)
    if MODEL_LOADED:
        last_state = (probs.detach(), final_mu, final_sigma)
    
    # QCEA Law 16: Survival Floor
    final_sigma = max(final_sigma, 0.1)
    
    return {
        "type": "builtin", 
        "name": "norm", 
        "params": {"loc": float(final_mu), "scale": float(final_sigma)}
    }

def default_pred():
    # Fallback: Mean Reversion with High Uncertainty
    loc = history[-1] if history else 0
    return {"type": "builtin", "name": "norm", "params": {"loc": loc, "scale": 10}}
