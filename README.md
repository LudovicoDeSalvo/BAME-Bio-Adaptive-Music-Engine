# Bio-Adaptive Closed-Loop Music Recommendation System

## 1. Project Overview
**Objective:** Develop a "homeostatic regulator" for human emotion that dynamically adjusts music playback based on real-time physiological feedback. This system creates a closed cybernetic loop, observing physiological deviations (e.g., stress) and actuating musical responses to minimize error, unlike open-loop commercial systems.

---

## 2. System Architecture
The system uses a Reinforcement Learning (RL) agent within a continuous state space, supported by a modular deep learning stack.

### Module A & D: Audio Representation ("Stimulus")
* **Model:** MERT-v1-330M (Music Audio Pre-training).
* **Function:** Extracts acoustic embeddings from raw audio waveforms.
* **Output:** Continuous embedding vector ($v_{audio} \in \mathbb{R}^d$).

### Module B: Physiological State ("Observer")
* **Inputs:** Electrodermal Activity (EDA), Blood Volume Pulse (BVP), Skin Temperature (SKT).
* **Model:** Dual-stream 1D-CNN + BiLSTM.
* **Mechanism:**
    * **Dual Stream:** Processes EDA and BVP separately to isolate specific noise profiles.
    * **Fusion:** Merges streams to capture long-range temporal dependencies.
* **Output:** Physiological State Vector.

### Module C: User Profiling ("Filter")
* **Inputs:** "Ten Item Personality Measure" (TIPI) scores.
* **Model:** Deep & Cross Network (DCN).
* **Function:** Models feature interactions (e.g., *Personality x Stimulus*) to address subjective reception.
* **Output:** User Profile Vector.

### Module E: Sequential Context ("Narrative")
* **Model:** Transformer Encoder (Self-Attention).
* **Function:** Processes interaction history to maintain narrative coherence and prevent abrupt mood shifts.
* **Output:** Session Context Vector.

---

## 3. Reinforcement Learning Core
**Engine:** Soft Actor-Critic (SAC) adapted for discrete information retrieval.

* **State Space ($S_t$):** Concatenation of vectors from Modules B, C, D, and E.
* **Policy (Actor):** Neural network mapping $S_t \rightarrow$ Continuous Target Action Vector ($v_{target}$) in MERT latent space.
* **Retrieval ("Wolpertinger" Layer):**
    * Actor outputs continuous vector, not discrete IDs.
    * **k-NN Search:** Uses FAISS to find song $S_{song}$ in the database closest to $v_{target}$.
* **Reward Function ($R_t$):** Minimization of physiological error (e.g., $R_t = -|Target\_Valence - Current\_Valence|$).
* **Critic:** Updates Q-values based on $(S_t, A_t, R_t, S_{t+1})$ to maximize reward and entropy.

---

## 4. Training Strategy: The User Simulator
Due to the static nature of the HKU956 dataset, Online RL is impossible initially. We utilize **Offline Reinforcement Learning**.

### Step 1: Supervised Pre-training (World Model)
* **Data:** HKU956 tuples $(User, Song, Response)$.
* **Task:** Train a "User Simulator" to predict Next Physio State ($S_{physio, t+1}$) and Reward ($R_{t+1}$) given Current State ($S_t$) and Action ($A_t$).
* **Role:** Acts as the surrogate Environment for the RL agent.

### Step 2: Offline RL Training
* Freeze the Simulator.
* Train the SAC agent inside this simulated environment to learn the optimal policy before real-world deployment.

---

## 5. Data Specifications
* **Primary Dataset:** **HKU956**.
    * **Subjects:** 30 participants.
    * **Content:** 956 songs.
    * **Signals:** EDA, BVP, SKT, HR (aligned with audio).
    * **Metadata:** TIPI personality profiles.

---

## 6. Action Plan: Parallel Development Strategy

To maximize efficiency, development is split into two tracks: **Track A (Control & Audio)** and **Track B (State & Simulation)**.

### Phase 0: Protocol Definition (Joint Task)
* **Objective:** Define the "Contract" between modules.
* **Deliverable:** `config.py` defining:
    * Dimension size of $v_{audio}$ (from MERT).
    * Dimension size of $v_{physio}$ (from CNN).
    * Dimension size of $v_{profile}$ (from DCN).
    * Structure of the concatenated State Vector $S_t$.

### Phase 1: Component Construction

| **Developer A (Control & Audio)** | **Developer B (State & Simulation)** |
| :--- | :--- |
| **1. Audio Pipeline (Module A/D):** <br> - Implement MERT-v1 inference. <br> - Batch process HKU956 audio to generate `audio_embeddings.npy`. | **1. Physio Pipeline (Module B):** <br> - Preprocessing (filtering/normalization) of EDA/BVP signals. <br> - Implement and train Dual-stream 1D-CNN + BiLSTM on HKU956 to predict valence/arousal. |
| **2. Vector Database:** <br> - Set up FAISS index with `audio_embeddings`. <br> - Implement k-NN lookup function. | **2. Profiler (Module C):** <br> - Implement Deep & Cross Network (DCN) for TIPI data. |
| **3. SAC Skeleton:** <br> - Implement SAC Actor/Critic networks (input size = $S_t$). <br> - Implement "Wolpertinger" logic (continuous output $\rightarrow$ k-NN). | **3. Data Loader:** <br> - Create `HKUDataset` class yielding tuples: $(S_{physio}, S_{user}, S_{audio}, S_{next\_physio})$. |

### Phase 2: Simulation & Integration

| **Developer A** | **Developer B** |
| :--- | :--- |
| **4. RL Loop Logic:** <br> - Build the training loop connecting Actor $\rightarrow$ Simulated Environment $\rightarrow$ Critic. | **5. User Simulator (World Model):** <br> - Train a Transformer/MLP to predict $S_{physio, t+1}$ given $(S_t, A_t)$. <br> - **Critical:** Ensure Simulator loss converges; otherwise RL will fail. |

### Phase 3: Joint Training
1.  **Merge:** Combine SAC Agent (Dev A) with User Simulator (Dev B).
2.  **Offline Training:** Run SAC training loop against the frozen User Simulator for 100k+ steps.
3.  **Evaluation:** Test agent against hold-out set of User Simulator scenarios.