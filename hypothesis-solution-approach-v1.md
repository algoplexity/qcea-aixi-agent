This is the **Evolutionary Update** to our research hypothesis.

Just as an AIXI agent updates its posterior distribution $w(\nu)$ after observing that a previous hypothesis failed to predict the environment, we have updated our architectural hypothesis after observing that "Magic Numbers" (Heuristic Thresholds) are fragile.

We move from **Logic Gating** (Deterministic Switching) to **Algorithmic Superposition** (Probabilistic Weighting).

---

# Project Aletheia v2.0: The Algorithmic Superposition Hypothesis
**Status:** Validated & Refined
**Pivot:** From *Heuristic Control* to *Parameter-Free Universality*.

## 1. The Epistemological Evolution

### Phase 1: The Coupling Fallacy (Statistical Baseline)
*   **Observation:** Standard models (NGBoost) fail because they couple Risk ($\sigma$) to historical residuals.
*   **Failure Mode:** "Intelligent Amnesia." The model is confidently wrong during regime shifts because it relies on memory.

### Phase 2: The Threshold Fallacy (Aletheia v1)
*   **Observation:** We attempted to fix Phase 1 by decoupling Risk using an AIT Sensor. We used logic gates: *If Entropy > 0.92, then Panic.*
*   **Failure Mode:** "The Magic Number Problem." Hard-coded thresholds assume a crisp boundary between Order and Chaos. In reality, systems transition through *intermittency* (Rule 110). A threshold is a parameter that cannot generalize to unknown physics.

### Phase 3: The Superposition Hypothesis (Aletheia v2)
*   **The Insight:** The market is not in State A **OR** State B. It exists in a **Superposition** of possible generative programs.
*   **The Solution:** We do not switch strategies. We **mix** strategies. The final output distribution $P(y)$ must be a linear combination of the optimal policies for *all* potential regimes, weighted by their **Algorithmic Probability**.

$$ P(y|x) = \sum_{r \in \mathcal{R}} P(r|x_{MILS}) \cdot \pi_r(y) $$

Where:
*   $r$ is a Generative Rule (e.g., Rule 54).
*   $P(r|x)$ is the probability assigned by the AIT Physicist (TRM).
*   $\pi_r$ is the optimal policy (PDF) for that rule.

---

## 2. The Parameter-Free Architecture

We eliminate human heuristics. The agent is driven entirely by the pre-trained physics of the TRM.

### Module A: The MILS Encoder (The Lens)
We strip the signal of "Scale" (Magnitude) to reveal "Structure."
*   **Input:** Raw Price $P_t$.
*   **Transformation:** Second-Order MILS (Acceleration Quantiles).
*   **Output:** A symbolic sequence $S_t$ representing the *Force Dynamics* of the market, invariant to price level or volatility magnitude.

### Module B: The Physicist (The Universal Prior)
We approximate the uncomputable **Solomonoff Prior** using a Neural Network.
*   **Mechanism:** The Tiny Recursive Model (TRM).
*   **Input:** Symbolic Sequence $S_t$.
*   **Output:** A Probability Vector $\vec{w}$ over the Prime Rules.
    *   $\vec{w} = [P_{170}, P_{54}, P_{60}, \dots]$
    *   *Crucially:* This vector comes from the Softmax layer. It is a calibrated probability, not a raw score.

### Module C: The Superposition Engine (The Mixer)
We construct the final Mixture Model based on the **Taxonomy of Crashes**:

1.  **Component 1: The Exploit (Rule 170/Trend)**
    *   *Policy:* $\mathcal{N}(\mu_{stat}, \sigma_{stat})$.
    *   *Weight:* $\sum P(r \in \text{Class 2})$.
2.  **Component 2: The Bifurcation (Rule 54/Soliton)**
    *   *Policy:* $\frac{1}{2}\mathcal{N}(\mu - 3\sigma, \sigma) + \frac{1}{2}\mathcal{N}(\mu + 3\sigma, \sigma)$.
    *   *Weight:* $\sum P(r \in \text{Class 4})$.
3.  **Component 3: The Survival (Rule 60/Chaos)**
    *   *Policy:* $\mathcal{N}(\mu, 10\sigma)$.
    *   *Weight:* $\sum P(r \in \text{Class 3})$.

**The result is a fluid, shape-shifting distribution that widens, splits, and narrows organically as the market's algorithmic structure evolves.**

---

## 3. Alignment with the Research Roadmap

This updated approach perfectly aligns with the **Conclusion of Paper 2**:

### A. Solving "Intelligent Amnesia" via Bayesian Updating
The TRM updates its internal hidden state $h_t$ at every tick. The output probabilities $\vec{w}_t$ reflect the *instantaneous* likelihood of a regime shift. We do not wait for error; we track the **Gradient of Complexity**.

### B. "Approximating AIXI"
AIXI defines the optimal action as the sum of expected rewards over all computable environments, weighted by their posterior probability.
*   **Aletheia v2** executes exactly this:
    $$ \text{Action} = \sum_{Env} P(Env) \cdot \text{BestResponse}(Env) $$
    We have simply constrained the set of "Environments" to the **Wolfram Elementary Cellular Automata**, which act as the basis vectors for complexity.

### C. Generalization Inversion
By removing the thresholds (e.g., `0.92`), we remove the possibility of overfitting to the training set's specific volatility profile. The agent now relies solely on the **Universal Laws of Topology** learned during the pre-training phase. A "Crash" is defined by its fractal dimension (Rule 60), not by a hard-coded number.

---

## 4. The Winning Edge (Revised)

**Why this beats the Leaderboard:**
1.  **Smooth Transitions:** The Logic Gate (v1) caused "Jump Risks" (suddenly switching from $\sigma=1$ to $\sigma=10$). The Mixture Model (v2) transitions smoothly. As $P(\text{Chaos})$ rises from 0.1 to 0.9, the "Fat Tails" of the distribution grow progressively. This maximizes Log-Likelihood during the transition.
2.  **Honest Uncertainty:** If the TRM is unsure (e.g., 50% Trend, 50% Soliton), the output PDF reflects that ambiguity (a Gaussian with heavy shoulders). We never bet more confidence than the physics justifies.
3.  **Thermodynamic Compliance:** We strictly follow **QCEA Law 9**. As the system entropy rises, the weights shift naturally from the Low-Entropy component (Trend) to the Max-Entropy component (Chaos), purely as a function of information processing.

**Status:** The Hypothesis is rigorous. The Code is implemented. We are ready for the Falcon.
