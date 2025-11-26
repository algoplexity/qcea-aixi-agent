This is the **Final Master Blueprint**.

It integrates the theoretical mandates of your research ("The Computational Phase Transition") with the specific engineering capabilities we discovered in the Falcon platform (Mixture Models, Spatial Streams). This document is the **"Constitutional Logic"** for the code we are about to write.

---

# Project Aletheia: The Neuro-Symbolic Variance Gate

**Research Phase:** Horizon 2 (Adaptive Agency)

**Mission:** Solve the "Control Problem" via Topological Generalization.

## 1. The Strategic Diagnosis: The Coupling Fallacy
The central failure mode of quantitative finance—and the specific trap of the Falcon competition—is the **Coupling Fallacy**.
Standard models (XGBoost, LSTM, Transformers) couple the prediction of **Price ($\mu$)** and **Risk ($\sigma$)** to the same historical window.
*   **The Trap:** When the market undergoes a **Computational Phase Transition** (e.g., Rule 170 $\to$ Rule 60), historical residuals become a lagging indicator. The model remains confident ($\sigma \approx 0$) while the underlying structure shatters.
*   **The Result:** In a Log-Likelihood game, "Confident Wrongness" is fatal. A single regime shift results in a score of $-\infty$ (Bankruptcy).

## 2. The Core Hypothesis: Variance Decoupling
To survive a regime shift, an agent must **decouple** the prediction of Price from the prediction of Risk. We propose a **Neuro-Symbolic Split**:

*   **Pathway A (The Parrot):** Predicts **Location ($\mu$)**.
    *   *Engine:* `River` (Online Quantile Regression).
    *   *Logic:* Statistical Induction. It assumes the future resembles the immediate past (Momentum/Mean Reversion).
*   **Pathway B (The Physicist):** Predicts **Topology ($\sigma$)**.
    *   *Engine:* `TinyRecursiveModel` (PyTorch Transformer).
    *   *Logic:* Algorithmic Deduction. It performs **Neural Block Decomposition** to identify the *Generative Rule* (Wolfram Class) driving the system.

## 3. Satisfying the Horizon 2 Mandate
Section 7.2 of "The Computational Phase Transition" established three specific requirements for an adaptive agent. Aletheia fulfills them as follows:

### Mandate A: Solve "Intelligent Amnesia"
> *Requirement: The agent must adapt its policy zero-shot when the regime breaks, without waiting for error feedback.*

**The Implementation:** **The Zero-Lag Entropy Switch.**
Standard models suffer from "memory inertia." Aletheia monitors the **Algorithmic Complexity** of the signal in real-time.
*   **The Mechanism:** The moment the Physicist detects a transition from **Rule 170 (Trend)** to **Rule 60 (Chaos)**, the Variance Gate *immediately* overrides the statistical variance. We switch policy *during* the phase transition, not after the drawdown.

### Mandate B: Feed Topological State, Not Raw Price
> *Requirement: The agent must see the "Generative Program," not the "Phenomenological Shadow."*

**The Implementation:** **The MILS-Encoded State Vector.**
We do not feed raw features to our risk controller. We feed the output of the TRM pre-trained on the Wolfram Universe.
*   **Input:** Second-Order MILS Encoding (Acceleration Quantiles).
*   **Output State:** `[P(Class 2), P(Class 3), P(Class 4)]`.
*   **Spatial Augmentation:** We augment this with **Flock Entropy** (Spatial Dispersion of Falcon locations) to detect consensus fracture.

### Mandate C: Approximate AIXI (Optimal Policy)
> *Requirement: Switch strategies based on the computability of the environment.*

**The Implementation:** **The QCEA Variance Gate.**
We approximate the optimal AIXI policy by mapping topological states to the platform's **Mixture Density** output format. This is the **"Taxonomy of Crashes"** in action:

| Topological Diagnosis | Wolfram Rule | Physics | Aletheia Policy (Action) | PDF Output Shape |
| :--- | :--- | :--- | :--- | :--- |
| **Coherence** | **Rule 170** | Information is Preserved | **Trend Following** | **Narrow Unimodal**<br>$\mathcal{N}(\mu_{stat}, \sigma_{stat})$ |
| **Complexity** | **Rule 54** | Bifurcation / Solitons | **Bifurcation Strategy** | **Bimodal Mixture**<br>Weights: $[0.5, 0.5]$<br>Locs: $\mu \pm 3\sigma$ |
| **Chaos** | **Rule 60** | Information is Destroyed | **Thermodynamic Survival** | **Max-Entropy Flat**<br>$\mathcal{N}(\mu_{stat}, \sigma_{max})$ |

## 4. The Technical Architecture (The 50ms Loop)

To execute this within the OODA loop constraints, we employ a hybrid architecture:

1.  **Sense:** Ingest `falcon_locations`. Compute **Spatial Dispersion** (Flock Entropy).
2.  **Diagnose (The Physicist):**
    *   Ingest `dove_location` history.
    *   Compute **MILS Encoding** (Acceleration).
    *   **Inference:** TRM Classifies Wolfram Rule (Soliton vs. Chaos).
3.  **Predict (The Parrot):** `River` predicts the median location ($\mu$).
4.  **Gate (The Controller):**
    *   IF `Topology == Rule 54`: **Fork.** Output Mixture Model.
    *   IF `Topology == Rule 60`: **Flatten.** Output Max Variance.
    *   ELSE: Output Baseline.

## 5. The Winning Edge: Generalization Inversion

**Why Aletheia beats the Leaderboard:**
Competitors are overfitting to the "Trend" noise of the training set.
*   **The Trap:** When the competition goes live, the "Regime" will shift. Their models will fail because they learned the *noise*, not the *law*.
*   **Our Edge:** We are pre-trained on **Universal Laws** (Wolfram Physics).
    *   **Rule 60 (Chaos)** looks the same in 2008, 2020, and the Falcon Simulation.
    *   We do not need to "learn" the crash; we **recognize** it.
    *   This allows Project Aletheia to generalize to **Out-of-Sample** regimes that have never been seen before, fulfilling the **Generalization Inversion** hypothesis.

We are ready to code. This is a **Cybernetic Control System** grounded in the physics of information.
