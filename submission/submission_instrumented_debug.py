# FILE: submission_instrumented_debug.py

import pandas as pd
import numpy as np
import math
from collections import deque

# Assuming TrackerBase is provided by the environment
from birdgame.trackers.trackerbase import TrackerBase

class QCEADebugAgent(TrackerBase):
    """
    An instrumented version of the System 1 agent to debug the death spiral.
    It will log every critical internal state variable.
    """
    def __init__(self, h=1):
        super().__init__(h)
        print("INFO: Instrumented Debug Agent Initializing.")
        
        # --- HOMEOSTATIC STATE ---
        self.gamma = 2.0
        self.last_pred = None
        self.target_ll = -1.0
        self.learning_rate = 0.05
        self.tick_count = 0
        
        print("INFO: Instrumented Debug Agent Initialized Successfully.")

    def tick(self, p, m=None):
        self.tick_count += 1
        val = p.get('dove_location')
        
        print(f"--- TICK {self.tick_count} ---")
        print(f"INPUT: Received value: {val}")

        if val is None or (isinstance(val, float) and np.isnan(val)):
            print("WARN: Invalid value received. Skipping tick logic.")
            return

        if self.last_pred is not None:
            mu, sigma = self.last_pred
            variance = max(sigma ** 2, 1e-9)
            diff = val - mu
            ll = -0.5 * math.log(2 * math.pi * variance) - (0.5 * (diff**2) / variance)
            
            error = self.target_ll - ll
            
            # --- LOGGING THE FEEDBACK LOOP ---
            print(f"LEARN: Last Pred (mu, sigma): ({mu:.4f}, {sigma:.4f})")
            print(f"LEARN: Log-Likelihood (Pain): {ll:.4f}")
            print(f"REGULATE: Target LL: {self.target_ll}, Error: {error:.4f}")
            print(f"REGULATE: Gamma (Before): {self.gamma:.4f}")

            if error > 0:
                self.gamma *= (1.0 + self.learning_rate)
            else:
                self.gamma *= (1.0 - (self.learning_rate / 2.0))
                
            self.gamma = max(min(self.gamma, 20.0), 1.0)
            
            print(f"REGULATE: Gamma (After): {self.gamma:.4f}")
        
        self.history.append(float(val))

    def predict(self):
        print(f"--- PREDICT (Tick {self.tick_count}) ---")
        if len(self.history) < 20:
            print("INFO: Warmup period. Not making a prediction.")
            return

        prices = pd.Series(list(self.history))
        vel = prices.diff().dropna()

        if vel.empty:
            print("WARN: Velocity is empty. Not making a prediction.")
            return

        v_curr = vel.iloc[-1]
        current_vol = vel.rolling(window=10).std().iloc[-1]
        
        if np.isnan(current_vol): current_vol = 1.0 # Handle NaN volatility

        mu_newton = self.history[-1] + v_curr
        mu_boltz = np.mean(self.history[-10:])
        w = 0.7
        final_mu = w * mu_newton + (1 - w) * mu_boltz
        
        base_sigma = max(current_vol, 0.1)
        final_sigma = base_sigma * self.gamma
        
        self.last_pred = (final_mu, final_sigma)

        # --- LOGGING THE ACTION ---
        print(f"ACT: Base Sigma: {base_sigma:.4f}, Final Sigma: {final_sigma:.4f}")
        print(f"ACT: Final Prediction (mu, sigma): ({final_mu:.4f}, {final_sigma:.4f})")
        
        return {
            "type": "builtin", 
            "name": "norm", 
            "params": {"loc": float(final_mu), "scale": float(final_sigma)}
        }
