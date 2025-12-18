# The QCEA-AIXI Agent: A Nested Learning Architecture for Financial Cognition

[![Status](https://img.shields.io/badge/Horizon-2%3A%20Agency-success.svg)](https://github.com/algoplexity/algoplexity) [![Framework](https://img.shields.io/badge/Framework-UAI%20%7C%20QCEA%20%7C%20NL%20%7C%20Coherence-purple.svg)](https://github.com/algoplexity/algoplexity)

**The Reflective Physicist: Operationalizing Universal Artificial Intelligence via System 0-1-2 Hierarchies.**

This repository contains the official implementation of **Horizon 2** of the [Algoplexity Research Program](https://github.com/algoplexity/algoplexity). Its mission is to solve the **Control Problem**: engineering an autonomous agent capable of **Zero-Shot Adaptation** within the "Dancing Landscape" of complex financial markets.

---

## 🧠 The Architecture: A "Nested Learning" Synthesis

The central challenge of real-world agency is the **Frequency Gap**—standard models possess static weights (Zero Frequency) and fast attention (High Frequency), but lack a mechanism to adapt to market regimes that change at a *Medium Frequency*.

To solve this, we have engineered a **Nested Optimization Architecture**, operationalizing **Nested Learning Theory** [Behrouz et al., 2025] and **Coherence Theory** [Williams, 2025]. The agent is not a single policy, but a hierarchy of control loops:

### **System 0 (Innate): The Coherence Veto**
*   **Protocol Kernel:** `CIv2` (Autopoiesis).
*   **Function:** **Survival Initialization.**
*   **Mechanism:** Before the agent enters the market, it runs a **Monte Carlo Simulation** (`simulate_eco_evo_transition`) against synthetic "Ruin Scenarios" (Class 3 Chaos). It evolves a "Genetic Gamma" ($\gamma_0$)—a baseline paranoia level required to survive the first 100 ticks.
*   **Why it matters:** This solves the **Cold Start Problem**. Unlike a standard RL agent that must die to learn, System 0 provides the **Evolutionary Prior** necessary for immediate viability.

### **System 1 (Fast): The Homeostatic Reflex**
*   **Protocol Kernel:** `CIv1` (Cybernetic Feedback).
*   **Function:** **Pain Avoidance.**
*   **Mechanism:** A high-frequency **PID Controller** that monitors the agent's own predictive error in real-time.
    *   *If Error > Threshold:* The "Sympathetic Nervous System" activates. $\gamma$ (Uncertainty) spikes exponentially. The agent widens its wingspan to survive the shock.
    *   *If Error < Threshold:* The "Parasympathetic Nervous System" activates. $\gamma$ relaxes linearly towards the System 2 target.
*   **Why it matters:** This provides **Kinetic Requisite Variety**. It reacts to volatility faster than the neural network can think.

### **System 2 (Slow): The AIT Physicist**
*   **Protocol Kernel:** `CIv5` (Structural Breaks) & `CIv7` (Joint Failure).
*   **Function:** **Topological Reasoning.**
*   **Mechanism:** A **Tiny Recursive Model (TRM)** pre-trained on the Wolfram Computational Universe acts as the **"Cognitive EEG."**
    1.  **Sense:** It maps the noisy price history to a probability distribution over Generative Rules (e.g., Rule 54 Solitons vs Rule 60 Fractals).
    2.  **Infer:** A **Reflective Gate** maps this topological state to a `target_gamma`.
*   **Why it matters:** It provides **Strategic Direction**. It tells System 1 *where* the baseline risk should be, allowing the agent to distinguish between "Safe Trends" (Rule 170) and "Dangerous Complexity" (Rule 54).

---

## 📂 Repository Structure

*   **`main.py`**: The definitive **Nested Learning Implementation**. Contains the full `QCEAAgent` class, the "Fast" Homeostat, the "Slow" Physicist, and the System 0 Simulator.
*   **`trm_expert.pth`**: The frozen weights of the System 2 Sensor (Horizon 1 Artifact).
*   **`reflective_gate.pth`**: The learned policy mapping Topology to Uncertainty.
*   **`notebooks/`**:
    *   `01_H2_The_Data_Foundry.ipynb`: Generating the "Nightmare Mode" theoretical benchmark.
    *   `02_H2_The_Spatial_Encoder.ipynb`: Training the Universal Prior for the AIT Physicist.
    *   `03_H2_The_Falcon_Gauntlet.ipynb`: Benchmarking the standalone System 1 (Homeostat).
    *   `04_H2_The_Cybernetic_Loop.ipynb`: The grand synthesis simulation.

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
4.  **Algorithmic Information Dynamics (AID):** The physics of causal structure.
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
