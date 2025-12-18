# The QCEA-AIXI Agent: A Dual-Process Architecture for Financial Cognition

[![Status](https://img.shields.io/badge/Horizon-2%3A%20Agency-success.svg)](https://github.com/algoplexity/algoplexity) [![Framework](https://img.shields.io/badge/Framework-UAI%20%7C%20QCEA%20%7C%20NL%20%7C%20Coherence-purple.svg)](https://github.com/algoplexity/algoplexity) [![Field](https://img.shields.io/badge/Field-Algorithmic%20Cognitive%20Science-blue.svg)](https://github.com/algoplexity/algoplexity)

**The Reflective Physicist: Operationalizing Universal Artificial Intelligence in Entropic Financial Environments.**

This repository contains the official implementation of **Horizon 2** of the [Algoplexity Research Program](https://github.com/algoplexity/algoplexity). Its mission is to solve the **Control Problem**: engineering an autonomous agent capable of **Zero-Shot Adaptation** within the "Dancing Landscape" of complex financial markets.

---

## 🧠 The Architecture: A "Nested Learning" Synthesis

The central challenge of real-world agency is the **Frequency Gap**—standard models possess static weights (Zero Frequency) and fast attention (High Frequency), but lack a mechanism to adapt to market regimes that change at a *Medium Frequency*.

To solve this, we have engineered a **Nested Optimization Architecture**, operationalizing **Nested Learning Theory** [Behrouz et al., 2025] and **Coherence Theory** [Williams, 2025]. The agent is not a single policy, but a hierarchy of control loops:

### **System 0 (Static): The Coherence Veto (The "Innate")**
*   **NL Role:** Managing the **Cold Start** and **Overload Regimes** ($\Lambda > \eta_K$).
*   **Mechanism:** A hard-coded **Lyapunov Prior**. At $t=0$ (before history exists) or when the environment drifts faster than the agent can update, System 0 enforces a "Maximal Chaos" assumption ($\lambda > 0$).
*   **Action:** It clamps the `uncertainty_factor` to **0.5**, effectively "betting small" to prevent Ruin Events until coherence is established.

### **System 1 (Fast): The Homeostatic Agent (The "Deep Optimizer")**
*   **Cognitive Scale:** The Spinal Cord / Reflex.
*   **NL Role:** Managing the **Fast Context Flow** ($1/t$).
*   **Mechanism:** This is a tactical, sub-cognitive agent designed to win the 50-millisecond dogfight. It solves the problem of **Bounded Rationality** through a **Reflective Cybernetic Loop**:
    1.  **SENSE (Entropic Valuation):** The agent monitors **Epistemic Fragility** ($dH/d\tau$). A spike in $dH/d\tau$ signals that the current context compression is failing.
    2.  **ACT (The Iron Dome):** Implementing **QCEA Law 16 (Survival First)**, the agent enforces a dynamic variance floor (`max(volatility, 1.0)`). This prevents "Ruin Events" by ensuring the agent never bets more precision than the environment allows.
    3.  **REGULATE (Homeostasis):** It acts as a heuristic **Deep Optimizer**, continuously adjusting its **`gamma` ("Panic Factor")** in real-time based on predictive error.

### **System 2 (Slow): The AIT Physicist (The "Context Compressor")**
*   **Cognitive Scale:** The Prefrontal Cortex / Mind.
*   **NL Role:** Managing the **Slow Context Flow** ($1/W$).
*   **Mechanism:** This is a strategic, meta-cognitive observer designed to understand the changing "weather of the war."
    1.  **DIAGNOSE (The Cognitive EEG):** A **Tiny Recursive Model (TRM)**, pre-trained on the Wolfram Computational Universe, acts as the Observation Operator. It collapses the market state into **Cognitive Saturation** (Rule 54 Solitons) or **Cognitive Overload** (Rule 60 Fractals).
    2.  **INFER (The Reflective Gate):** Based on this diagnosis, the agent dynamically weights an ensemble of internal experts:
        *   **Expert A (Newton):** Optimizes for **Inertia** (Rule 170). Best for stable trends (Tracking Regime).
        *   **Expert B (Boltzmann):** Optimizes for **Entropy** (Maximum Uncertainty). Best for chaotic transitions (Critical Lag Regime).

### **The Synthesis: Continuum Memory**
By using the "Slow" System 2 to modulate the learning rate of the "Fast" System 1, we achieve **Continuum Memory**. The agent adapts its optimization trajectory in real-time, effectively "remembering" how to behave in a specific physical regime without needing to retrain its weights.

---

## 🏆 Performance: The Generalization Inversion

Our agent was deployed in the **CrunchDAO Falcon Competition**, a hostile, real-world environment characterized by non-stationary structural breaks.

| Metric | Baseline (NGBoost) | QCEA-AIXI Agent | Improvement |
| :--- | :--- | :--- | :--- |
| **Score (Likelihood)** | 1.16 | **1.38** | **+19.0%** |
| **Survival Rate** | - | **100%** | **Robust** |

**Key Finding:** The implementation of **System 0 (The Coherence Veto)** was critical. Early prototypes without this prior suffered "Ruin Events" at $t<50$. The full Nested Architecture successfully navigated the "Cold Start" ($t=0$ to $t=80$) without capital destruction, handing off to System 2 for profit generation.

---

## 📂 Repository Structure

*   **`submission.ipynb`**: The definitive **Nested Learning Implementation**. Contains the full `QCEAAgent` class, the "Fast" Homeostat, the "Slow" Physicist, and the Pre-Submission Test Harness.
*   **`models/`**: Contains the frozen `trm_expert.pth` (System 2 Sensor) and `reflective_gate.pth`.
*   **`notebooks/`**:
    *   `01_H2_The_Data_Foundry.ipynb`: Generating the "Nightmare Mode" theoretical benchmark.
    *   `02_H2_The_Spatial_Encoder.ipynb`: Training the Universal Prior for the AIT Physicist.
    *   `03_H2_The_Falcon_Gauntlet.ipynb`: Benchmarking the standalone System 1 (Homeostat).
    *   `04_H2_The_Cybernetic_Loop.ipynb`: The grand synthesis simulation.
    *   `05_H2_Shadow_Diagnostic.ipynb`: Visualizing the "Newton vs. Boltzmann" switching logic in real-time.

---

## 📊 Shared Data Artifacts (Hugging Face)

To ensure unassailable reproducibility, this Horizon operates on an immutable scientific benchmark:

*   **[QCEA Adaptive Agent Benchmark](https://huggingface.co/datasets/algoplexity/qcea-adaptive-agent-benchmark):** The 2D "Dancing Landscape" corpus for Horizon 2.

---

## 📚 References & Foundations

Our work synthesizes foundational theories into a coherent engineering framework:

1.  **Nested Learning Theory:** The mechanics of hierarchical optimization.
    *   *Behrouz, A., et al. (2025). Nested Learning: The Illusion of Deep Learning Architectures. NeurIPS 2025.*
2.  **Coherence Theory:** The thermodynamics of regime transitions.
    *   *Williams, C. F. (2025). Eco-evolutionary regime transitions as coherence loss in hereditary updating.*
3.  **Universal Artificial Intelligence (UAI):** The mathematics of optimal general intelligence.
    *   *Hutter, M. (2024). An Introduction to Universal Artificial Intelligence.*
4.  **Quantum-Complex-Entropic-Adaptive (QCEA):** The thermodynamics of strategic viability.
    *   *Williams, C. F. (2025). Strategy as Ontology.*
5.  **Algorithmic Information Dynamics (AID):** The physics of causal structure.
    *   *Zenil, H. (2022). Algorithmic Information Dynamics of Cellular Automata.*

---

## 🔗 Citation

If you utilize this architecture or the benchmark dataset in your research, please cite the Horizon 2 initiative:

```bibtex
@misc{qcea_aixi_2025,
  author = {Mak, Yeu Wen},
  title = {The Reflective Physicist: Operationalizing Universal Artificial Intelligence in Entropic Financial Environments},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/algoplexity/qcea-aixi-agent}}
}
```
