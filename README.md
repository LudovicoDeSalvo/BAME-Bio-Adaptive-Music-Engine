# Bio-Adaptive Music Engine (BAME)

**A Closed-Loop AI System for Physiological State Regulation via Music**

## Abstract

The **Bio-Adaptive Music Engine** is a deep reinforcement learning framework that
steers a user's **physiological latent state** toward a desired target using
music. Unlike recommender systems that optimize for "likes," this engine
optimizes for **biological impact**.

The system learns a compact **physiological latent** (an affect-relevant
embedding trained to predict valence/arousal), builds a **World Model** that
simulates how that latent evolves when a user hears a song, and trains a
**goal-conditioned Soft Actor-Critic (SAC)** agent to drive the latent toward a
target. A **Wolpertinger policy** maps the agent's continuous control signal to
real songs from a high-dimensional audio embedding space (MERT) via FAISS, then
re-ranks the candidates with the critic.

> **Scope note.** The "physiological state" regulated here is the 64-dim
> **encoder latent**, not raw biosignal units. The agent is trained and evaluated
> **inside the learned World Model** (an in-silico simulator). Real-world
> experimentation is required to confirm any in-silico result.

---

## System Architecture

### 1. Sensing & Perception (The Encoders)

* **Audio:** Pre-trained **MERT-v1-330M** extracts 1024-dim embeddings from raw
  waveforms. Every MERT/action vector is **L2-normalized** onto the unit sphere
  so FAISS retrieval, the World Model, and the replay buffer share one manifold.
* **Physiology:** A **Dual-Stream CNN-LSTM encoder** processes per-window
  biosignal statistics (EDA, TEMP → dermal stream; BVP, HR, IBI → cardio stream)
  into a **64-dim latent**. Inputs are per-channel **z-scored** with persisted
  stats (single source of truth across training and inference).
* **User Profile:** A **Deep & Cross Network (DCN)** encodes Big-Five personality
  scores into a **32-dim** vector. The min-max scaler is persisted with the
  embedding pool so inference reuses the exact training normalizer.
* **Context Transformer:** A sequence model aggregating the user's recent
  per-chunk listening history (last 5 MERT vectors) into a **128-dim** context
  vector, making the state **non-Markovian**.

### 2. Simulation (The World Model)

* A residual MLP acting as a **virtual environment**.
* **Input:** `[physio_latent_t (64), user (32), context_t (128)]` + `song_t (1024)`
* **Output:** predicted `physio_latent_{t+1} (64)` (residual: `physio_t + Δ`)

### 3. Control (The Agent)

* **Algorithm:** Soft Actor-Critic (SAC) with tanh-squashed actions, twin
  critics, and automatic entropy tuning.
* **Goal-conditioned observation:** the agent sees
  `[physio_latent, user, context, target_physio_latent]` (**288-dim**). The
  target is part of the observation — the reward is goal-relative, so the policy
  must be able to condition on the goal it is asked to reach.
* **Reward:** scale-aware negative RMS distance to the target latent (per-dim
  normalized by the physio-pool std), multiplied by `training.reward_scale` so
  the control signal is not swamped by the 1024-dim entropy term.
* **Policy:** **Wolpertinger**. The actor emits a continuous "ideal song" vector;
  FAISS returns the *k* nearest real songs; the critic re-ranks them and selects
  the highest-value intervention. Only the **applied** (retrieved) song vector is
  stored in the replay buffer, so the critic's `Q(s,a)` matches the dynamics that
  produced the reward.

---

## Data Model & Trial Keys

The HKU956 dataset contains **replays**: a participant may hear the same
`song_id` more than once under different `song_no` values. The unique
**playback/trial key is `(participant_id, song_id, song_no)`** — never
`song_id` alone.

* Clip IDs are `"{song_id}_{participant}_{song_no}_{chunk}"`. Including
  `song_no` prevents two replays from colliding (which would otherwise overwrite
  one play's audio clip and share a single MERT embedding across two distinct
  trials).
* The World Model, Context Transformer, and inference all key transitions by the
  full trial key so distinct replays are never merged into one trajectory.

---

## Installation

```bash
git clone https://github.com/LudovicoDeSalvo/Bio-Adaptive-Music-Engine.git
cd Bio-Adaptive-Music-Engine
pip install -r requirements.txt
```

`faiss-gpu` is listed in `requirements.txt`; if the CUDA wheel will not resolve,
install `faiss-cpu` instead.

---

## Usage

The project is driven by a central CLI dashboard:

```bash
python main.py
```

### Data Preparation

* **[0] Preparation:** create folders, unzip `HKU956.zip`, optionally download songs.
* **[1] Align & Slice:** sync physiological signals with audio, slice into 30s
  chunks (15s stride), z-score the per-window features, and cache everything.
* **[2] Extract Embeddings (MERT):** run MERT on each audio clip (GPU intensive),
  L2-normalize, and save the song embeddings + ID map.
* **[3] Verify Dataset:** print dataset statistics from the physio cache.

### Component Training

* **[4] Train Physio Encoder:** learns the 64-dim physiological latent.
* **[5] Train User Profiler:** learns the 32-dim personality embedding.
* **[6] Train Context Model:** learns per-chunk listening-history patterns.

### Simulation & Agent

* **[7] Train World Model:** trains the simulator. Depends on [4], [5], [6].
* **[8] Train SAC Agent:** trains the goal-conditioned agent against the World
  Model. The default step budget exceeds the warmup so updates actually run.

### Evaluation

* **[9] Run Inference:** single held-out user "blind test"
  (`hku1903` by default), reporting World-Model **latent MSE** and agent control
  vs. a **random-action baseline**.
* **True LOSO:** `python -m scripts.inference loso` holds out **every**
  participant in turn and aggregates latent MSE and control improvement as
  **mean ± 95% CI**, against the random-action baseline. Each fold **retrains the
  full pipeline** (user, physio, context, World Model, agent) with that subject
  excluded, so no fold's models or fitted normalizers ever saw the subject they
  are scored on. This is expensive (one full pipeline train per participant); the
  sweep restores the default-holdout checkpoints when it finishes.
* **Quick LOSO (leaky):** `python -m scripts.inference loso-quick` reuses the one
  checkpoint set trained with only the config holdout excluded. Fast, but every
  other subject leaked into training — it is a smoke test, **not** a valid
  generalization estimate, and is labelled `quick_leaky` in its summary.

For the single-holdout protocol the holdout user is excluded from **all**
component training (physio, user, context, World Model) so the subject never
leaks into the models or their fitted normalizers. The window normalizer is
fitted by the physio encoder on its holdout-excluded split and persisted next to
the embeddings, so inference normalizes the holdout with stats that never saw it.

---

## Results

Two evaluation protocols are provided:

* **Single-holdout blind test** (`run_inference_protocol`) — one held-out user.
* **Leave-One-Subject-Out** (`run_loso_protocol`) — every participant held out in
  turn, aggregated as mean ± 95% CI with a random-action baseline.

### Metrics

* **Generalization (World Model):** MSE over the full **64-dim physiological
  latent** on unseen users. This is the encoder-output space (affect-relevant,
  trained to predict valence/arousal), **not raw biosignal units** — the scale is
  set by encoder training, so compare runs trained under the same configuration.
* **Control Efficacy (Agent):** mean improvement (initial − final distance to the
  target latent) over an episode, using the **same scale-aware RMS distance the
  reward optimizes** so the agent is scored on what it learned. Reported
  alongside a **true random-action baseline** — a uniformly random real song
  applied each step, with **no critic re-ranking** — so control authority is
  measured against genuine chance (the baseline does not borrow the trained
  critic's selection). Because the agent is rolled out inside the World Model,
  this measures the agent's ability to drive the *simulator* toward a real
  held-out target state.

> ⚠️ **Numbers pending re-measurement.** Earlier headline figures predate the
> current fixes (goal-conditioned observation, trial-keyed clip IDs, window-level
> input normalization, reward scaling, SAC tanh squashing, manifold
> L2-normalization, persisted normalizers, true LOSO). They are no longer valid
> and must be regenerated by re-running the pipeline end-to-end with
> `training.seed` set for reproducibility. Real-world experimentation is still
> required to confirm any in-silico result.

---

## Configuration

Key fields in `configs/config.yaml`:

| Section    | Field                   | Meaning                                              |
|------------|-------------------------|------------------------------------------------------|
| `model`    | `action_dim`            | 1024 — must match MERT                                |
| `model`    | `physio_embedding_dim`  | 64 — PhysioEncoder output (and World Model output)    |
| `model`    | `profile_embedding_dim` | 32 — DCNProfile output                                |
| `model`    | `context_embedding_dim` | 128 — ContextTransformer output                       |
| `training` | `max_steps`             | episode horizon (env, training, eval all read this)   |
| `training` | `reward_scale`          | lifts the O(1) reward above the entropy term          |
| `training` | `target_entropy_scale`  | SAC entropy target = −action_dim × scale (1.0 canonical) |
| `training` | `holdout_user`          | subject excluded from training, used for inference    |
| `training` | `seed`                  | global RNG seed for reproducible runs                 |

The agent observation dim is derived: `physio + user + context + target` = 288.
The World-Model input dim is `physio + user + context` = 224 (it never sees the
target).

---

## Project Structure

```
├── audio/
│   ├── faiss_index.py      # Nearest-neighbor song retrieval (FAISS, cosine)
│   └── mert_embedder.py    # MERT-v1-330M audio feature extractor
├── configs/
│   └── config.yaml         # Global hyperparameters and paths
├── context/
│   ├── sequence_model.py   # Transformer over listening history
│   └── train_context.py    # Context module training (trial-keyed)
├── data/
│   └── windows.py          # Window features + persisted z-score normalizer
├── physio/
│   ├── encoder.py          # Dual-stream CNN-LSTM encoder
│   └── train_encoder.py    # Physio encoder training
├── rl/
│   ├── sac_agent.py        # Soft Actor-Critic (twin critics, auto-entropy)
│   ├── train_agent.py      # RL training loop + replay buffer
│   └── wolpertinger.py     # KNN action selection + critic re-ranking
├── scripts/
│   ├── download_songs.py   # Audio fetch helper
│   ├── align_and_slice.py  # Data synchronization + caching
│   └── inference.py        # Evaluation, LOSO, and report generation
├── simulator/
│   ├── gym_env.py          # Gymnasium env (goal-conditioned observation)
│   ├── train_simulator.py  # World Model training (trial-keyed transitions)
│   └── world_model.py      # Residual neural physio simulator
├── user/
│   ├── dcn_profile.py      # Deep & Cross Network for user traits
│   └── train_profile.py    # User profiler training
├── tests/                  # Unit tests (env contract, dims, normalizer, …)
├── utils/
│   └── common.py           # Paths, seeding, L2-normalize, config, helpers
└── main.py                 # Central CLI controller
```

---

## Citation & Credits

* **Dataset:** HKU956 (University of Hong Kong): Hu, X.; Li, F.; Liu, R.
  *Detecting Music-Induced Emotion Based on Acoustic Analysis and Physiological
  Sensing: A Multimodal Approach.* Applied Sciences. 2022, 12, 9354.
  https://doi.org/10.3390/app12189354
* **Audio Model:** MERT-v1-330M (HuggingFace).
</content>
</invoke>
