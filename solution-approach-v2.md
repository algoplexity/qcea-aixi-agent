This is the final, definitive Solution Plan for **Project Aletheia**.

It integrates the theoretical foundations of **QCEA-T**, the "Generalization Inversion" discovery from your research, and the hermeneutic insight regarding **MILS as a Selection Mechanism**.

---

# Project Aletheia: The Neuro-Symbolic Variance Gate

**Research Phase:** Horizon 2 (Adaptive Agency)

**Mission:** Solve the "Control Problem" via Topological Generalization.

---

## 1. The Core Epistemological Pivot

The central failure mode of quantitative finance—and the specific trap of the Falcon competition—is the **Coupling Fallacy**.
Standard models (XGBoost, LSTM, Transformers) couple the prediction of **Price ($\mu$)** and **Risk ($\sigma$)** to the same historical feature set.
*   **The Trap:** When the market undergoes a **Computational Phase Transition** (e.g., Rule 170 $\to$ Rule 60), historical residuals become a lagging indicator. The model remains confident ($\sigma \approx 0$) while the underlying structure shatters.
*   **The Consequence:** In a Log-Likelihood game, "Confident Wrongness" is fatal. A single regime shift results in a score of $-\infty$ (Bankruptcy).

**The Solution:** We implement a **Neuro-Symbolic Split**:
1.  **Pathway A (The Parrot):** Predicts **Location ($\mu$)** using statistical induction on a MILS-sparsified signal.
2.  **Pathway B (The Physicist):** Predicts **Topology ($\sigma$)** using algorithmic deduction (The AIT Physicist) on the generative rules.

---

## 2. The Technical Architecture (The Cybernetic Loop)

We employ a 4-stage pipeline designed to execute within the 50ms OODA loop of the Falcon platform.

### Module 0: The MILS Filter (Sparsification)
*   **Objective:** Filter the input stream. Most `falcon_locations` are redundant or noise.
*   **Mechanism (Online MILS):** We measure the **Algorithmic Mutual Information** between each Falcon and the Dove.
    *   We track the residual stream $r_t = (Dove_t - Falcon_{i,t})$.
    *   We calculate the **Permutation Entropy** of $r_t$.
    *   *Selection Logic:* If $H(r_t) \approx 1$ (Max Entropy), the Falcon is adding pure noise. We prune it. We only feed the "Prime Falcons" (Low Entropy Residuals) to the predictor.

### Module 1: The Parrot (Location Prediction)
*   **Objective:** Predict the median price ($\mu$).
*   **Mechanism:** `River` (Online Quantile Regression).
*   **Logic:** Statistical Induction. It assumes the future resembles the immediate past (Momentum/Mean Reversion) of the *selected* Prime Falcons.
*   **Output:** A baseline statistical confidence $\mathcal{N}(\mu_{stat}, \sigma_{stat})$.

### Module 2: The Physicist (Topological Diagnosis)
*   **Objective:** Diagnose the **Generative Rule** (State).
*   **Mechanism:** `TinyRecursiveModel` (TRM) pre-trained on the Wolfram Universe.
*   **Logic:**
    1.  **MILS Encoding:** Convert the Dove's history into **Acceleration Quantiles** (removing price scale).
    2.  **Inference:** The TRM classifies the sequence into a Wolfram Class.
    3.  **Spatial Augmentation:** Calculate **Flock Entropy** (Standard Deviation of the active Falcons) to detect consensus fracture.

### Module 3: The QCEA Gate (Control)
*   **Objective:** Form the final Probability Density Function (PDF).
*   **Logic:** We map the Topological State to an Optimal Policy (approximating AIXI).

| AIT Diagnosis | Physical State | Action (Policy) | Output Distribution |
| :--- | :--- | :--- | :--- |
| **Rule 170 / 15** | **Coherence.** Information preserved. | **Exploit.** Trust the Parrot. | **Unimodal Narrow**<br>$\mathcal{N}(\mu_{stat}, \sigma_{stat})$ |
| **Rule 54 / 110** | **Complexity.** Bifurcation / Solitons. | **Hedge.** Expect a Breakout. | **Bimodal Mixture**<br>$\frac{1}{2}\mathcal{N}(\mu-k, \sigma) + \frac{1}{2}\mathcal{N}(\mu+k, \sigma)$ |
| **Rule 30 / 60** | **Chaos.** Information destroyed. | **Survive.** Maximize Entropy. | **Flat / Wide**<br>$\mathcal{N}(\mu_{stat}, \sigma_{safety})$ |

---

## 3. Meeting the Research Mandates

This architecture rigorously satisfies the requirements set out in the Conclusion of **"The Computational Phase Transition"**:

1.  **Solving Intelligent Amnesia (Zero-Lag Switching):**
    By monitoring the *structure* of the signal (Module 2) rather than the *error* of the prediction, Aletheia detects a regime shift (Rule 60) instantly. It switches from "Exploit" to "Survive" at $t=0$, fulfilling the mandate to adapt without memory inertia.

2.  **Feeding State, Not Price:**
    Module 3 (The Gate) makes decisions based on the **Topological State Vector** (e.g., "The market is running a Soliton program"), not on the raw price level. This fulfills the requirement to operate on the "Generative Program."

3.  **Approximate AIXI (Optimal Policy):**
    We respect the bounds of Computability. When the Physicist detects High Kolmogorov Complexity (Rule 60), we acknowledge that the series is incompressible/unpredictable. The optimal AIXI action is to output a Maximum Entropy distribution. Aletheia enforces this theoretically optimal behavior via the Variance Gate.

---

## 4. The Winning Edge

**Why Aletheia wins the Falcon Competition:**

1.  **Generalization Inversion:** Competitors are overfitting to the "Trend" noise of the training set. Aletheia is pre-trained on **Universal Physics**. We recognize the "Crash Topology" because we have seen it in the Wolfram Universe, even if it has never appeared in the Falcon training data.
2.  **Log-Likelihood Survival:** In a game scored by $\log(P_{true})$, one "Black Swan" event destroys the score. Aletheia's "Survival Mode" (Rule 60 Override) ensures we never take the catastrophic penalty that bankrupts statistical models.
3.  **Signal Purity:** By applying **MILS as a Filter** (Module 0), we remove the "Noise Falcons" that confuse standard models, ensuring our "Parrot" is trained on the highest-fidelity signal available.

**Status:** The theoretical architecture is complete. The component code blocks are verified. We are ready to assemble the final `aletheia_tracker.py`.
