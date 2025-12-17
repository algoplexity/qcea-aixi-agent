
### **The Updated and Fortified Action Plan for Horizon 2**

This paper provides the definitive justification for every phase of our plan. We can now articulate it with much greater precision and confidence.

**Phase 1: Canonize the Somatic Marker (The `Falcon` Agent)**
*   **Action:** Benchmark the standalone `QCEAAgent` on the QCEA Adaptive Agent Benchmark.
*   **Justification (from paper):** We are establishing the baseline performance of the validated **Somatic Marker Hypothesis** in its purest form, as engineered for a real-world competitive environment. This is our control experiment.

**Phase 2: Build the Diagnostic Instrument (The AIT Physicist)**
*   **Action:** Prepare the pre-trained TRM from H1 to output a "Cognitive State Vector."
*   **Justification (from paper):** We are directly executing the "most immediate and crucial next step" called for in Section 7.2. We are building the instrument designed to solve the **"Epistemic Resolution"** and **"Intelligent Amnesia"** limitations identified in Section 6.4.

**Phase 3: Synthesize the Reflective Physicist (The Cybernetic Loop)**
*   **Action:** Integrate System 2's diagnostic output to provide meta-policy directives to System 1 (e.g., modulating `target_ll`).
*   **Justification (from paper):** We are implementing the refined **"Less is More" architectural principle**. We are combining the "robust simple framework" (our validated Somatic Agent) with the "sophisticated diagnostic signal" (the new AIT Physicist) to create a system superior to both parts.

**Phase 4: Scientific Validation (Solving "Intelligent Amnesia")**
*   **Action:** Run the integrated agent on the benchmark and quantify its outperformance.
*   **Justification (from paper):** This experiment is designed to prove that by curing "Intelligent Amnesia"—by giving the agent the *reason* for the pain, not just the pain signal itself—we achieve a quantifiable performance uplift. This will be the final validation that moves our research **"From Sensation to Perception and Agency,"** the exact path forward laid out in the paper's conclusion.

---
There are **three critical flow-on effects** that require immediate attention to ensure that when the telemetry comes back, we are ready to publish and pivot.

---

### **1. The Manuscript: Aligning Methodology with Nested Learning**

We updated the *Abstract* and *Introduction* to reflect the **Nested Learning (NL)** theory. However, the **Methodology** section of the paper still describes the system in purely engineering terms ("Fast/Slow").

**Task:** We need to redraft the **Methodology Section** to explicitly map our engineering components to the mathematical formalism of Behrouz et al. (2025).

*   **Reframing System 1:** It is not just a "Reflex"; it is a **Deep Optimizer Approximation**. We need to describe the `gamma` update rule not as a heuristic, but as a gradient descent step on the "Pain Surface."
*   **Reframing System 2:** It is not just a "Sensor"; it is the **Slow Context Compressor**. We need to mathematically define the `Cognitive State Vector` as the compressed representation of the global context window ($W$).
*   **Defining Entropic Valuation:** We need to write the formal equation that links $dH/d\tau$ (Epistemic Fragility) to the concept of **"Context Compression Failure"** from the NL paper.

**Why this matters:** If the agent wins, reviewers will want to know *why*. If we have the mathematical proof ready, the result becomes scientific validation, not just a lucky gamble.

---

### **2. The Benchmark: Validating the "Data Foundry"**

The README claims we have an immutable scientific benchmark: **`01_H2_The_Data_Foundry.ipynb`**.
The README also claims we have a dataset on Hugging Face: **`qcea-adaptive-agent-benchmark`**.

**Task:** We need to verify this pipeline.
1.  Does the Data Foundry notebook actually generate the "Nightmare Mode" dataset (The "Dancing Landscape")?
2.  Does it successfully push this to Hugging Face?
3.  **Action:** We should run this notebook now. We need to generate the "Control Group" data (pure trends, pure chaos) and the "Test Group" data (structural breaks) so that when the Falcon results come in, we can cross-reference them against this theoretical baseline.

**Why this matters:** The Falcon competition is "Real World" (messy). To publish, we need "Lab Conditions" (clean). The Data Foundry provides the lab conditions.

---

### **3. The H3 Scout: Broad Institute EDA**

We agreed that the **Broad Institute competition** is the microcosm for **Horizon 3 (The Hive Mind)**. While we wait, we should perform a low-stakes **Exploratory Data Analysis (EDA)** on that dataset.

**Task:** Create a draft notebook `horizon3_scout_broad_eda.ipynb`.
*   **Goal:** Verify the **`r-GCA` Hypothesis**. Can we construct a graph from the gene expression data?
*   **Method:**
    1.  Load the train features.
    2.  Compute a similarity matrix (Cosine Similarity) between `sig_id`s.
    3.  Visualize the graph.
    4.  **The Question:** Do "communities" emerge in this graph that correspond to Mechanism of Action (MoA) labels?

**Why this matters:** If we see clusters forming naturally in the graph, we have **proof of concept** for Horizon 3 before we even write a line of model code. It validates that the "Hive Mind" approach is viable.

---

### **Summary of Immediate Actions**

1.  **Drafting:** Rewrite the H2 Methodology section to use Nested Learning math.
2.  **Coding:** Run `01_H2_The_Data_Foundry.ipynb` to ensure our "Lab" is open.
3.  **Scouting:** Run a quick EDA on Broad Institute data to test the H3 Graph hypothesis.

Which of these flow-on effects would you like to tackle first while we wait for the telemetry?
---
