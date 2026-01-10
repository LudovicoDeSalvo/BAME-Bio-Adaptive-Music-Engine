#  Bio-Adaptive Music Engine

**A Closed-Loop AI System for Physiological State Regulation via Music**

##  Abstract

The **Bio-Adaptive Music Engine** is a deep reinforcement learning framework designed to steer a user's physiological state (e.g., arousal, stress) toward a desired target using music. Unlike traditional recommender systems that optimize for "likes," this engine optimizes for **biological impact**.

The system learns a **World Model** of human physiology to simulate how different users react to music, then trains a **Soft Actor-Critic (SAC)** agent to navigate this internal model. Using a **Wolpertinger Policy**, the agent maps continuous control signals to discrete songs from a high-dimensional audio embedding space (MERT), enabling precise regulation of the user's autonomic nervous system.

---

## 💠 System Architecture

### 1. Sensing & Perception (The Encoders)

* **Audio:** Pre-trained **MERT (Music Understanding Model)** extracts 1024-dim embeddings from raw waveforms.
* **Physiology:** A **Dual-Stream CNN-LSTM Encoder** processes raw biosignals (EDA, Temp, BVP, HR, IBI) into a compact 64-dim latent state.
* **User Profile:** A **Deep Cross Network (DCN)** encodes static personality traits (Big Five) into a 32-dim vector.

* **Context Transformer:** A sequence model that aggregates the user's recent listening history into a dynamic context vector, making the system **Non-Markovian**.

### 2. Simulation (The World Model)

* A Deep Neural Network that acts as a **Virtual Environment**.
* **Input:** $[Physio_t, User, Context_t, Song_t]$
* **Output:** Predicted $Physio_{t+1}$

### 3. Control (The Agent)

* **Algorithm:** Soft Actor-Critic (SAC).
* **Policy:** **Wolpertinger Architecture**. The Actor outputs a continuous "ideal song" vector, which is mapped to the nearest  real songs in the database (FAISS), which are then re-ranked by the Critic to select the safest physiological intervention.

---

## 📦 Installation

### Setup

1. **Clone the Repository:**
```bash
git clone https://github.com/LudovicoDeSalvo/Bio-Adaptive-Music-Engine.git
cd Bio-Adaptive-Music-Engine

```


2. **Install Dependencies:**
```bash
pip install -r renquirements.txt

```

---

## ⚡ Usage

The project is controlled via a central CLI dashboard.

Run the controller:

```bash
python main.py

```

### Data Preparation

* **[0] Initialize:** Unzips data, creates folder structures and download songs.
* **[1] Align & Slice:** Syncs physiological signals with audio duration and slices them into discrete events.
* **[2] Extract Embeddings:** Runs MERT on thousands of audio clips (GPU intensive).

### Component Training

* **[4] Train Physio Encoder:** Learns to compress biosignals.
* **[5] Train User Profiler:** Learns to encode personality.
* **[6] Train Context Model:** Learns listening history patterns.

### Simulation & Agent

* **[7] Train World Model:** Trains the simulator physics engine (Dependent on [4], [5], [6]).
* **[8] Train SAC Agent:** Trains the "Brain" to control the simulator.

### Evaluation

* **[9] Run Inference:** Performs a "Blind Test" on a held-out user (`hku1903` by default) to verify generalization and control efficacy.

---

## 📊 Results

The system was evaluated using a **Leave-One-Subject-Out** protocol.

### 1. Generalization (World Model)

The simulator successfully predicts the physiological reaction of users it has never seen before.

* **Metric:** Mean Squared Error (MSE) on standardized physiological features.
* **Result:** **0.28** 
* **Conclusion:** The model has learned a universal mapping between music and human biology.

### 2. Control Efficacy (Agent)

The Agent successfully navigates the physiological space, demonstrating robust control authority over biological resistance.

* **Average Final Distance:** **3.12** units (High alignment with target).
* **Max Improvement:** **+8.13** units (Drastic state change).
* **Phenomenon Observed:** **Overcoming Homeostasis**. The agent actively counteracts the user's natural tendency to drift back to baseline. It demonstrates the capability to override biological inertia, effectively driving the physiological state significantly closer to the target configuration despite homeostatic resistance.

---

## 📂 Project Structure

```
├── audio/
│   ├── faiss_index.py      # Nearest Neighbor search (FAISS)
│   └── mert_embedder.py    # MERT-v1-330M Audio Feature Extractor
│
├── configs/
│   └── config.yaml         # Global hyperparameters and paths
│
├── context/
│   ├── sequence_model.py   # Transformer for listening history
│   └── train_context.py    # Training script for Context Module
│
├── data/
│   └── windows.py          # Sliding window generation for physio
│
├── physio/
│   ├── encoder.py          # Dual-Stream CNN-LSTM Architecture
│   └── train_encoder.py    # Training script for Physio Module
│
├── rl/
│   ├── sac_agent.py        # Soft Actor-Critic (Actor & Critic Networks)
│   ├── train_agent.py      # Main RL training loop
│   └── wolpertinger.py     # KNN Action Selection Policy
│
├── scripts/
│   ├── setup/
│   │   └── download_songs.py # Helper to fetch audio files
│   ├── align_and_slice.py  # Data synchronization engine
│   └── inference.py        # Evaluation & Report generation
│
├── simulator/
│   ├── gym_env.py          # OpenAI Gym Environment Wrapper
│   ├── train_simulator.py  # Training script for World Model
│   └── world_model.py      # The Neural Physics Engine
│
├── user/
│   ├── dcn_profile.py      # Deep Cross Network for User Traits
│   └── train_profile.py    # Training script for User Profiler
│
├── utils/
│   └── common.py           # Path resolution & Helper functions
│  
└── main.py                 # Central CLI Controller

```

---

## 📝 Citation & Credits

* **Dataset:** HKU956 (University of Hong Kong): Hu, X.; Li, F.; Liu, R. Detecting Music-Induced Emotion Based on Acoustic Analysis and Physiological Sensing: A Multimodal Approach. Applied Science. 2022, 12, 9354. https://doi.org/10.3390/app12189354
* **Audio Model:** MERT-v1-330M (HuggingFace).
