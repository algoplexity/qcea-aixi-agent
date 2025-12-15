# FILE: submission_instrumented_debug.py


import math
from collections import deque

# Assuming TrackerBase is provided by the environment
from birdgame.trackers.trackerbase import TrackerBase

# FILE: submission_instrumented_debug_pure_python.py

import math
from collections import deque


# ==============================================================================
#  DEPENDENCY-FREE, INSTRUMENTED DEBUG AGENT
# ==============================================================================
class QCEADebugAgent(TrackerBase):
    """
    An instrumented version of the System 1 agent to debug the death spiral.
    This version uses ONLY pure Python and standard libraries to pass the 
    competition's dependency checks.
    """
    def __init__(self, h=1):
        super().__init__(h)
        print("INFO: Instrumented Debug Agent (Pure Python) Initializing.")
        
        # --- HOMEOSTATIC STATE ---
        self.gamma = 2.0
        self.last_pred = None
        self.target_ll = -1.0
        self.learning_rate = 0.05
        self.tick_count = 0
        
        print("INFO: Instrumented Debug Agent Initialized Successfully.")

    def _calculate_std(self, data):
        """Helper function to calculate standard deviation in pure Python."""
        n = len(data)
        if n < 2:
            return 1.0 # Cannot calculate stddev, return a default volatility
        
        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / n
        return math.sqrt(variance)

    def tick(self, p, m=None):
        self.tick_count += 1
        val = p.get('dove_location')
        
        print(f"--- TICK {self.tick_count} ---")
        print(f"INPUT: Received value: {val}")

        if val is None or (isinstance(val, float) and math.isnan(val)):
            print("WARN: Invalid value received. Skipping tick logic.")
            return

        # --- THE FEEDBACK LOOP (LEARN & REGULATE) ---
        if self.last_pred is not None:
            mu, sigma = self.last_pred
            variance = max(sigma ** 2, 1e-9)
            diff = val - mu
            # This is the "Pain Signal"
            ll = -0.5 * math.log(2 * math.pi * variance) - (0.5 * (diff**2) / variance)
            
            error = self.target_ll - ll
            
            # --- LOGGING THE INTERNAL STATE ---
            print(f"LEARN: Last Pred (mu, sigma): ({mu:.4f}, {sigma:.4f})")
            print(f"LEARN: Log-Likelihood (Pain): {ll:.4f}")
            print(f"REGULATE: Target LL: {self.target_ll}, Error: {error:.4f}")
            print(f"REGULATE: Gamma (Before): {self.gamma:.4f}")

            if error > 0: # Performance is worse than target (we are hurting)
                self.gamma *= (1.0 + self.learning_rate)
            else: # Performance is better than target (we are safe)
                self.gamma *= (1.0 - (self.learning_rate / 2.0))
                
            self.gamma = max(min(self.gamma, 20.0), 1.0)
            
            print(f"REGULATE: Gamma (After): {self.gamma:.4f}")
        
        self.history.append(float(val))

    def predict(self):
        print(f"--- PREDICT (Tick {self.tick_count}) ---")
        if len(self.history) < 20:
            print("INFO: Warmup period. Not making a prediction.")
            return

        # --- DATA PREPARATION (PURE PYTHON) ---
        # Convert deque to list for indexing
        price_history = list(self.history)
        
        # Re-implement pandas.diff()
        velocity = [price_history[i] - price_history[i-1] for i in range(1, len(price_history))]
        
        if not velocity:
            print("WARN: Velocity is empty. Not making a prediction.")
            return

        # --- THE "ACT" PHASE ---
        v_curr = velocity[-1]
        
        # Re-implement pandas.rolling().std()
        vel_window = velocity[-10:]
        current_vol = self._calculate_std(vel_window)
        
        # Re-implement numpy.mean()
        hist_window = price_history[-10:]
        mean_boltz = sum(hist_window) / len(hist_window)

        mu_newton = self.history[-1] + v_curr
        mu_boltz = mean_boltz
        
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
