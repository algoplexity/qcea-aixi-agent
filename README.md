
# The QCEA-AIXI Agent: A Dual-Process Architecture for Financial Cognition

[![Status](https://img.shields.io/badge/Horizon-2%3A%20Agency-success.svg)](https://github.com/algoplexity/algoplexity) [![Framework](https://img.shields.io/badge/Framework-UAI%20%7C%20QCEA%20%7C%20AID-purple.svg)](https://github.com/algoplexity/algoplexity) [![Field](https://img.shields.io/badge/Field-Algorithmic%20Cognitive%20Science-blue.svg)](https://github.com/algoplexity/algoplexity)

**The Reflective Physicist: Operationalizing Universal Artificial Intelligence in Entropic Financial Environments.**

This repository contains the official implementation of **Horizon 2** of the [Algoplexity Research Program](https://github.com/algoplexity/algoplexity). Its mission is to solve the **Control Problem**: engineering an autonomous agent capable of **Zero-Shot Adaptation** within the "Dancing Landscape" of complex financial markets.

---

## 🧠 The Architecture: A "Thinking, Fast and Slow" Synthesis

The central challenge of real-world agency is the **tyranny of the clock**—the need for millisecond-level reaction times that makes slow, deliberative reasoning (the ideal of AIXI) computationally infeasible. To solve this, we have engineered a novel **Dual-Process Cognitive Architecture**:

### **System 1 (Fast): The Homeostatic Agent (The "Reflex")**
This is our tactical, sub-cognitive agent designed to win the 50-millisecond dogfight. It solves the problem of **Bounded Rationality** through a **Reflective Cybernetic Loop**:

1.  **SENSE (The Hard-Wired Reflex):** The agent monitors **Entropic Valuation** ($dH/d\tau$). When the "Epistemic Fragility" of the market spikes, the agent actively modulates its **"Wingspan"** (Sigma/Uncertainty).
2.  **ACT (The Iron Dome):** Implementing **QCEA Law 16 (Survival First)**, the agent enforces a dynamic variance floor (`max(volatility, 1.0)`). This prevents "Ruin Events" by ensuring the agent never bets more precision than the environment allows.
3.  **REGULATE (Homeostasis):** It continuously adjusts its **`gamma` ("Panic Factor")** based on predictive error (The Somatic Marker), ensuring **Survival is a Precondition for Optimization**.

### **System 2 (Slow): The AIT Physicist (The "Mind")**
This is our strategic, meta-cognitive observer. It solves the problem of **Non-Stationarity** by diagnosing the market's underlying computational state.

1.  **DIAGNOSE (The Cognitive EEG):** A **Tiny Recursive Model (TRM)**, pre-trained on the Wolfram Computational Universe, acts as the Observation Operator. It classifies the market regime into **Cognitive Saturation** (Rule 54 Solitons) or **Cognitive Overload** (Rule 60 Fractals).
2.  **INFER (The Reflective Gate):** Based on this diagnosis, the agent dynamically weights an ensemble of internal experts:
    *   **Expert A (Newton):** Optimizes for **Inertia** (Rule 170). Best for stable trends.
    *   **Expert B (Boltzmann):** Optimizes for **Entropy** (Maximum Uncertainty). Best for chaotic transitions.

---

## 🏆 Performance: The Generalization Inversion

Our agent was deployed in the **CrunchDAO Falcon Competition**, a hostile, real-world environment characterized by non-stationary structural breaks.

| Metric | Baseline (NGBoost) | QCEA-AIXI Agent | Improvement |
| :--- | :--- | :--- | :--- |
| **Score (Likelihood)** | 1.16 | **1.38** | **+19.0%** |
| **Survival Rate** | - | 100% (Post-Patch) | **Stable** |

**Key Finding:** The agent demonstrated **Entropic Plasticity**. By conditioning its policy on the *structure* of the market (Diagnosis) rather than just the *price* (Correlation), it successfully navigated "Black Swan" events that bankrupted purely statistical baselines.

---

## 📂 Repository Structure

*   **`submission.py`**: The standalone **System 1 (Homeostatic Agent)** currently live on CrunchDAO. Includes the "Iron Dome" logic.
*   **`models/`**: Contains the frozen `trm_expert.pth` (System 2 Sensor).
*   **`notebooks/`**:
    *   `01_H2_The_Data_Foundry.ipynb`: Generating the "Nightmare Mode" theoretical benchmark.
    *   `02_H2_The_Spatial_Encoder.ipynb`: Training the Universal Prior for the AIT Physicist.
    *   `03_H2_The_Falcon_Gauntlet.ipynb`: Benchmarking the standalone System 1.
    *   `04_H2_The_Cybernetic_Loop.ipynb`: The grand synthesis simulation.
    *   `05_H2_Shadow_Diagnostic.ipynb`: Visualizing the "Newton vs. Boltzmann" switching logic in real-time.

---

## 📊 Shared Data Artifacts (Hugging Face)

To ensure unassailable reproducibility, this Horizon operates on an immutable scientific benchmark:

*   **[QCEA Adaptive Agent Benchmark](https://huggingface.co/datasets/algoplexity/qcea-adaptive-agent-benchmark):** The 2D "Dancing Landscape" corpus.

---

## 📚 References & Foundations

Our work synthesizes foundational theories into a coherent engineering framework:

1.  **Universal Artificial Intelligence (UAI):** The mathematics of optimal general intelligence.
    *   *Hutter, M. (2024). An Introduction to Universal Artificial Intelligence.*
2.  **Quantum-Complex-Entropic-Adaptive (QCEA):** The thermodynamics of strategic viability.
    *   *Williams, C. F. (2025a). Strategy as Ontology.*
3.  **Entropic Valuation:** The economic theory of epistemic fragility.
    *   *Williams, C. F. (2025b). From Temporal Decay to Epistemic Fragility.*
4.  **Algorithmic Information Dynamics (AID):** The physics of causal structure.
    *   *Zenil, H. (2022). Algorithmic Information Dynamics of Cellular Automata.*

---

## 🔗 Citation

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
