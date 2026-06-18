import os
import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from user.dcn_profile import DCNProfile
from simulator.world_model import WorldModel
from context.sequence_model import ContextTransformer
from physio.encoder import DualStreamEncoder, split_streams
from rl.sac_agent import SACAgent
from rl.wolpertinger import WolpertingerPolicy
from data.windows import apply_window_norm
from utils.common import load_config, resolve_path, get_device, set_seed, load_npz_checked, l2_normalize, parse_clip_id

# --- Configuration ---
device = get_device()
config = load_config("configs/config.yaml")

REPORT_DIR = resolve_path("reports/evaluation")
os.makedirs(REPORT_DIR, exist_ok=True)
TARGET_USER = config.get('training', {}).get('holdout_user', 'hku1903')
SEED = int(config.get('training', {}).get('seed', 42))

# dims from config (single source of truth — no hardcoded 224/1024/64)
_mc = config.get("model", {})
PHYSIO_DIM = int(_mc.get("physio_embedding_dim", 64))
USER_DIM = int(_mc.get("profile_embedding_dim", 32))
CONTEXT_DIM = int(_mc.get("context_embedding_dim", 128))
ACTION_DIM = int(_mc.get("action_dim", 1024))
# World-model input state: [physio, user, context].
STATE_DIM = PHYSIO_DIM + USER_DIM + CONTEXT_DIM
# Agent observation: world-model state + target_physio (goal-conditioned). Must
# match MusicEnv.observation_space so the loaded SAC actor/critic dims line up.
OBS_DIM = STATE_DIM + PHYSIO_DIM

def load_user_vector(user_id):
    """ generates the 32-dim user embeddings for the holdout user"""

    print(f">> [Inference] Generating profile vector for {user_id}...")

    profile_dim = config['model']['profile_embedding_dim']
    csv_path = resolve_path(config['paths']['personality_csv'])
    model_path = resolve_path("user/checkpoints/profile_model.pth")

    df = pd.read_csv(csv_path)
    user_row = df[df['participant_id'].astype(str) == str(user_id)]

    if len(user_row) == 0:
        raise ValueError(f"User {user_id} not found in personality CSV")

    # Use the SAME min-max scaler that was fitted at training time (persisted in
    # user_embeddings.npz). The old `/10.0` heuristic did not match the trained
    # normalizer and invalidated the embedding.
    pool = load_npz_checked(resolve_path(config['paths']['user_embeddings']),
                            ['norm_min', 'norm_max', 'feat_cols'])
    feat_cols = [str(c) for c in pool['feat_cols']]
    mn = pool['norm_min'].astype(np.float32)
    mx = pool['norm_max'].astype(np.float32)

    raw_vals = user_row[feat_cols].values.astype(np.float32)
    norm_vals = (raw_vals - mn) / (mx - mn)

    model = DCNProfile(input_dim=len(feat_cols), embedding_dim=profile_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        input_tensor = torch.tensor(norm_vals).to(device)
        user_emb, _ = model(input_tensor)
        
    return user_emb.cpu().numpy()[0]

def get_real_physio_embeddings(user_id):
    """ generates ground truth physio embeddings """

    print(f">> [Inference] encoding ground yruth physio for {user_id}...")

    cache_path = resolve_path(config['paths']['physio_cache'])
    data = np.load(cache_path, allow_pickle=True)
    all_pids = data['participant_ids']
    
    mask = [str(p) == str(user_id) for p in all_pids]
    mask = np.array(mask)
    
    if not np.any(mask):
        print(f" Warning: No raw physio data found for {user_id}")
        return {}, {}, []

    user_windows = data['window_features'][mask]
    # Apply the SAME window normalizer the encoder trained with. The authoritative
    # source is the encoder-side normalizer persisted in physio_embeddings.npz: it
    # was fitted on the holdout-EXCLUDED split (true LOSO refits it per fold), so
    # the holdout subject never leaks into its own normalization. Fall back to the
    # (legacy) cache stats only if the embedding file lacks them.
    win_mean = win_std = None
    try:
        emb_meta = np.load(resolve_path(config['paths']['physio_embeddings']), allow_pickle=True)
        if 'win_norm_mean' in emb_meta and 'win_norm_std' in emb_meta:
            win_mean, win_std = emb_meta['win_norm_mean'], emb_meta['win_norm_std']
    except (FileNotFoundError, OSError):
        pass
    if win_mean is None and 'win_norm_mean' in data and 'win_norm_std' in data:
        win_mean, win_std = data['win_norm_mean'], data['win_norm_std']
    if win_mean is not None:
        user_windows = apply_window_norm(user_windows, win_mean, win_std)
    user_clip_ids = data['clip_ids'][mask]
    user_song_ids = data['song_ids'][mask]
    # song_no = per-playback trial id; keeps repeated plays of the same song
    # from being merged into one trajectory. Legacy caches lack it -> key by sid.
    has_song_nos = 'song_nos' in data
    user_song_nos = data['song_nos'][mask] if has_song_nos else None
    
    enc_path = resolve_path("physio/checkpoints/physio_encoder.pth")
    encoder = DualStreamEncoder(embedding_dim=PHYSIO_DIM).to(device)
    encoder.load_state_dict(torch.load(enc_path, map_location=device))
    encoder.eval()
    
    embeddings = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(user_windows), batch_size):

            batch = torch.tensor(user_windows[i:i+batch_size], dtype=torch.float32).to(device)
            dermal, cardio = split_streams(batch)
            emb = encoder(dermal, cardio)
            embeddings.append(emb.cpu().numpy())
            
    final_embs = np.vstack(embeddings)
    
    clip_map = {str(k): v for k, v in zip(user_clip_ids, final_embs)}
    
    song_map = {}

    for i, sid in enumerate(user_song_ids):
        sid = str(sid)
        cid = str(user_clip_ids[i])
        try:
            _sid, _pid, _sno, chunk_idx = parse_clip_id(cid)
            # tuple key (sid, sno) for trial-aware caches; bare sid for legacy
            skey = (sid, str(user_song_nos[i])) if has_song_nos else sid
            if skey not in song_map: song_map[skey] = []
            song_map[skey].append((chunk_idx, final_embs[i], cid))
        except (ValueError, IndexError):
            continue

    return clip_map, song_map, final_embs

def eval_world_model(user_vec, physio_lookup, user_id=None, make_plot=True):
    """ test 1: prediction + graph """

    user_id = user_id or TARGET_USER

    if not physio_lookup:
        print(" !!! Skipping world model eval (No data).")
        return float('nan'), float('nan')

    print(f">> [Inference] Testing world model...")
    wm_path = resolve_path(config['paths']['world_model_path'])
    ctx_path = resolve_path("context/checkpoints/context_model.pth")

    world_model = WorldModel.load(wm_path, device=device, state_dim=STATE_DIM,
                                  action_dim=ACTION_DIM, physio_dim=PHYSIO_DIM)

    ctx_model = ContextTransformer(ACTION_DIM, CONTEXT_DIM).to(device)
    ctx_model.load_state_dict(torch.load(ctx_path, map_location=device))
    ctx_model.eval()

    ratings = pd.read_csv(resolve_path(config['paths']['ratings_csv']))
    user_ratings = ratings[ratings['participant_id'].astype(str) == str(user_id)].sort_values('song_no')
    
    audio_embs = np.load(resolve_path("data/processed/song_embeddings.npy"), allow_pickle=True)
    audio_ids = np.load(resolve_path("data/processed/song_id_map.npy"), allow_pickle=True)
    audio_map = {str(k): v for k, v in zip(audio_ids, audio_embs)}
    
    errors = []
    # Persistence baseline: predict next == current. Because chunks overlap 50%
    # (30s @ 15s stride) the two physio latents are autocorrelated, so a large
    # part of any low WM MSE is trivial copying. Reporting this baseline beside
    # the WM MSE exposes how much error the model actually removes over "do nothing".
    persistence_errors = []
    history_buffer = []

    predictions = []
    ground_truths = []

    for _, row in user_ratings.iterrows():
        sid = str(row['song_id'])

        # match the per-playback trial (sid, song_no); fall back to the bare
        # sid key for legacy caches that were built without song_nos.
        clip_list = physio_lookup.get((sid, str(row['song_no'])))
        if clip_list is None:
            clip_list = physio_lookup.get(sid)

        if not clip_list:
            continue

        clips = sorted(clip_list, key=lambda x: x[0])

        # Mirror train_simulator EXACTLY: one transition per CONTIGUOUS chunk pair,
        # each using its OWN chunk's audio embedding as the action, with context
        # recomputed per transition from the running (chunk-level) history. The
        # old code reused the first chunk's embedding for every transition, which
        # measured the world model off its training distribution.
        for i in range(len(clips) - 1):
            curr_idx, curr_physio, curr_cid = clips[i]
            next_idx, next_physio_real, _ = clips[i + 1]

            if next_idx != curr_idx + 1:
                continue
            if curr_cid not in audio_map:
                continue

            action_vec = l2_normalize(audio_map[curr_cid])

            with torch.no_grad():
                if not history_buffer:
                    ctx = np.zeros(CONTEXT_DIM, dtype=np.float32)
                else:
                    seq = np.array(history_buffer[-5:])
                    inp = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
                    ctx = ctx_model(inp).cpu().numpy()[0]

            state = np.concatenate([curr_physio, user_vec, ctx])
            s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            a_t = torch.tensor(action_vec, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_physio = world_model(s_t, a_t).cpu().numpy()[0]

            # MSE in the 64-dim physio LATENT space (the encoder output the world
            # model predicts), NOT raw biosignal units. The latent is trained to
            # be valence/arousal-predictive, so it is the affect-relevant state.
            mse = np.mean((pred_physio - next_physio_real) ** 2)
            errors.append(mse)
            # next == current (no model), same transitions.
            persistence_errors.append(np.mean((curr_physio - next_physio_real) ** 2))

            # Summarize the FULL latent vector (mean over dims), not dim[0].
            predictions.append(float(np.mean(pred_physio)))
            ground_truths.append(float(np.mean(next_physio_real)))

            history_buffer.append(action_vec)

    avg_mse = float(np.mean(errors)) if errors else float('nan')
    persistence_mse = float(np.mean(persistence_errors)) if persistence_errors else float('nan')
    print(f"   >> WM MSE: {avg_mse:.5f} | persistence (next=current) MSE: {persistence_mse:.5f}")
    
    # graph
    if make_plot and len(ground_truths) > 0:
        plt.figure(figsize=(10, 5))
        limit = min(100, len(ground_truths))
        plt.plot(ground_truths[:limit], label="Real physio latent (mean over dims)", alpha=0.7)
        plt.plot(predictions[:limit], label="Predicted physio latent (mean over dims)", linestyle="--", alpha=0.7)
        plt.title(f"World Model Generalization: {user_id} (latent MSE: {avg_mse:.4f})")
        plt.xlabel("Step (Song Chunk)")
        plt.ylabel("Physio latent value (encoder output)")
        plt.legend()
        plt.savefig(os.path.join(REPORT_DIR, "physio_prediction.png"))
        plt.close()
        print(f"   >> Graph saved to {REPORT_DIR}/physio_prediction.png")

    return avg_mse, persistence_mse

def eval_agent_control(user_vec, valid_targets, user_id=None, policy="agent",
                       make_plot=True, seed=SEED):
    """ test 2: agent control vs. random-action baseline.

    policy="agent"  -> SAC actor proto-action + Wolpertinger (FAISS retrieve +
                       critic re-rank of the k neighbors).
    policy="random" -> a uniformly random REAL song applied each step, with NO
                       critic re-ranking. This is a true chance baseline: the old
                       code still routed the random proto through Wolpertinger, so
                       the trained critic re-ranked the candidates and the
                       "baseline" inherited the agent's selection authority —
                       inflating it and understating the agent's true control gain.
    Returns a results DataFrame so the LOSO loop can aggregate.
    """

    user_id = user_id or TARGET_USER
    NUM_SESSIONS = 15
    K_NEIGHBORS = 10
    # Episode horizon from config (single source of truth, matches the env).
    EVAL_HORIZON = int(config.get('training', {}).get('max_steps', 50))
    rng = np.random.default_rng(seed)

    print(f">> [Inference] Testing agent ({policy}, Wolpertinger k={K_NEIGHBORS})...")

    agent_path = resolve_path("rl/checkpoints/sac_final_actor.pth")
    critic_path = resolve_path("rl/checkpoints/sac_final_critic.pth")

    # OBS_DIM (state + target), NOT STATE_DIM: the actor/critic were trained on
    # the goal-conditioned observation.
    agent = SACAgent(OBS_DIM, ACTION_DIM)
    agent.actor.load_state_dict(torch.load(agent_path, map_location=device))
    agent.actor.eval()

    agent.critic.load_state_dict(torch.load(critic_path, map_location=device))
    agent.critic.eval()

    wolpertinger = WolpertingerPolicy(k_neighbors=K_NEIGHBORS, device=device)

    from simulator.gym_env import MusicEnv
    env = MusicEnv()
    env.action_space.seed(seed)

    results = []

    for session in tqdm(range(NUM_SESSIONS)):
        state, _ = env.reset(seed=seed + session)

        env.current_user = user_vec

        if len(valid_targets) > 0:
            t_idx = int(rng.integers(0, len(valid_targets)))
            env.target_physio = valid_targets[t_idx]

            # Start from THIS user's own physio (a real state), not the random/
            # other-user embedding left over from env.reset(). Keeps the state
            # vector internally consistent (holdout physio + holdout profile).
            # Picked distinct from the target when more than one state exists; with
            # a single state, start == target (degenerate but still consistent),
            # never a foreign pool sample.
            s_idx = t_idx
            if len(valid_targets) > 1:
                while s_idx == t_idx:
                    s_idx = int(rng.integers(0, len(valid_targets)))
            env.current_physio = valid_targets[s_idx]

        # Rebuild the goal-conditioned observation after overriding physio/user/
        # target (target is the LAST physio_dim block, matching MusicEnv).
        env.state = np.concatenate([env.current_physio, env.current_user,
                                    env.current_context, env.target_physio])

        target = env.target_physio
        rewards = []
        # Scale-aware RMS distance — the SAME geometry the reward optimizes (see
        # MusicEnv.distance), so `improvement` scores the agent on what it learned,
        # not a different (raw-Euclidean) metric. Pre-action baseline distance so
        # `improvement` covers the full trajectory.
        catalog = getattr(wolpertinger.retriever, "embeddings", None)
        distances = [env.distance(env.current_physio, target)]

        for step in range(EVAL_HORIZON):
            state_tensor = torch.FloatTensor(env.state).unsqueeze(0).to(device)
            if policy == "random":
                # True chance baseline: apply a uniformly random REAL song,
                # bypassing the critic entirely. Fall back to a sampled proto only
                # if the catalog is unavailable.
                if catalog is not None and len(catalog) > 0:
                    real_action = catalog[int(rng.integers(0, len(catalog)))]
                else:
                    proto = env.action_space.sample()
                    real_action, _ = wolpertinger.select_action(proto, agent.critic, state_tensor)
            else:
                proto = agent.select_action(env.state, evaluate=True)
                real_action, _ = wolpertinger.select_action(proto, agent.critic, state_tensor)

            next_state, reward, _, _, _ = env.step(real_action)
            dist = env.distance(env.current_physio, target)
            rewards.append(reward)
            distances.append(dist)

        results.append({
            "session": session,
            "avg_reward": np.mean(rewards),
            "final_dist": distances[-1],
            "initial_dist": distances[0],
            "improvement": distances[0] - distances[-1]
        })

        # graph
        if make_plot and policy == "agent" and session == 0:
            plt.figure(figsize=(8, 5))
            plt.plot(distances, label="Distance to Target State")
            plt.xlabel("Step (Song)")
            plt.ylabel("Scale-aware RMS distance")
            plt.title(f"Agent Control Trajectory ({user_id})")
            plt.legend()
            plt.savefig(os.path.join(REPORT_DIR, "agent_control_trajectory.png"))
            plt.close()
            print(f"   >> Graph saved to {REPORT_DIR}/agent_control_trajectory.png")

    df_res = pd.DataFrame(results)
    if make_plot:
        print("\n>> Agent performance:")
        print(df_res.describe())
        df_res.to_csv(os.path.join(REPORT_DIR, "agent_metrics.csv"), index=False)

    return df_res

def run_inference_protocol():
    print(f"--- EVALUATION REPORT FOR {TARGET_USER} ---")
    set_seed(SEED)

    try:
        u_vec = load_user_vector(TARGET_USER)

        _, physio_lookup, valid_targets = get_real_physio_embeddings(TARGET_USER)

        physics_score, persistence_score = eval_world_model(u_vec, physio_lookup, user_id=TARGET_USER)
        agent_df = eval_agent_control(u_vec, valid_targets, user_id=TARGET_USER, policy="agent")
        base_df = eval_agent_control(u_vec, valid_targets, user_id=TARGET_USER,
                                     policy="random", make_plot=False)

        summary = {
            "user_id": TARGET_USER,
            "world_model_mse": float(physics_score),
            "persistence_mse": float(persistence_score),
            "agent_avg_reward": float(agent_df["avg_reward"].mean()),
            "agent_improvement": float(agent_df["improvement"].mean()),
            "random_baseline_improvement": float(base_df["improvement"].mean()),
            "strategy": "Wolpertinger (k=10)"
        }

        with open(os.path.join(REPORT_DIR, "summary.json"), "w") as f:
            json.dump(summary, f, indent=4)

        print(f"\n Report generated in {REPORT_DIR}")

    except Exception as e:
        print(f"\n !!! Inference Failed: {e}")
        import traceback
        traceback.print_exc()


def _mean_ci(values):
    """mean and 95% confidence interval half-width. NaNs (e.g. a fold with no
    contiguous transitions) are dropped before aggregating."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = float(np.mean(values))
    if n < 2:
        return mean, 0.0
    ci = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(n)
    return mean, ci


# Per-fold retraining budgets for true LOSO. Match the interactive single-holdout
# pipeline so fold models are comparable to the default run.
LOSO_PHYSIO_EPOCHS = 150
LOSO_WORLD_EPOCHS = 70
LOSO_AGENT_STEPS = 20000


def _retrain_for_holdout(uid):
    """Retrain EVERY component that produces an eval-time embedding/model with
    `uid` held out, overwriting the shared checkpoints/pools in place. This is
    what makes LOSO *real*: without it, only the config holdout is ever excluded,
    so every other 'held-out' subject was actually in the training set of the
    encoder, profiler, context model, world model AND their fitted normalizers.

    Order respects dependencies: user + physio (independent) -> context (song
    embeddings only) -> world model (needs physio/user/context) -> agent (needs
    world + context + pools). Imports are local to keep module import light and
    avoid pulling the training stack at eval time.
    """
    from user.train_profile import train_user_model
    from physio.train_encoder import train_physio_model
    from context.train_context import train_context_model
    from simulator.train_simulator import train_world_model
    from rl.train_agent import train_sac_agent

    print(f"\n=== [LOSO] Retraining full pipeline with holdout={uid} ===")
    train_user_model(holdout_user=uid)
    train_physio_model(epochs=LOSO_PHYSIO_EPOCHS, holdout_user=uid)
    train_context_model(holdout_user=uid)
    train_world_model(epochs=LOSO_WORLD_EPOCHS, holdout_user=uid)
    train_sac_agent(steps=LOSO_AGENT_STEPS)


def run_loso_protocol(participants=None, retrain=True, restore=True):
    """Leave-One-Subject-Out evaluation.

    retrain=True  (default) -> TRUE LOSO: every fold retrains the full pipeline
                   with that subject held out, so no fold's models or normalizers
                   ever saw the subject they are evaluated on. Expensive (a full
                   pipeline train per participant).
    retrain=False -> QUICK / LEAKY: reuse the single set of checkpoints trained
                   with only the config holdout excluded. Fast, but every
                   participant except the config holdout leaked into training —
                   the numbers are optimistic and NOT a valid generalization
                   estimate. Printed loudly as such.
    restore=True  -> after a true-LOSO sweep, retrain once more on the config
                   holdout so the on-disk checkpoints are left in the default
                   state (the sweep overwrites them per fold)."""
    set_seed(SEED)

    if participants is None:
        ratings = pd.read_csv(resolve_path(config['paths']['ratings_csv']))
        rated = set(ratings['participant_id'].astype(str))
        if retrain:
            # Don't depend on a pre-existing (config-holdout) user pool: enumerate
            # straight from the personality CSV intersected with rated subjects.
            pdf = pd.read_csv(resolve_path(config['paths']['personality_csv']))
            participants = [str(p) for p in pdf['participant_id'] if str(p) in rated]
        else:
            pool = load_npz_checked(resolve_path(config['paths']['user_embeddings']),
                                    ['participant_ids'])
            participants = [str(p) for p in pool['participant_ids'] if str(p) in rated]

    mode = "TRUE (retrain per fold)" if retrain else "QUICK/LEAKY (shared checkpoints)"
    print(f"--- LOSO EVALUATION ({mode}) over {len(participants)} participants ---")
    if not retrain:
        print(" !!! WARNING: quick mode reuses one checkpoint set. Only the config")
        print("     holdout is truly excluded; all other subjects leaked into")
        print("     training. Do NOT report these as generalization numbers.")

    rows = []
    for uid in participants:
        try:
            if retrain:
                _retrain_for_holdout(uid)
            u_vec = load_user_vector(uid)
            _, physio_lookup, valid_targets = get_real_physio_embeddings(uid)
            if not physio_lookup:
                continue
            mse, persistence = eval_world_model(u_vec, physio_lookup, user_id=uid, make_plot=False)
            agent_df = eval_agent_control(u_vec, valid_targets, user_id=uid,
                                          policy="agent", make_plot=False)
            base_df = eval_agent_control(u_vec, valid_targets, user_id=uid,
                                         policy="random", make_plot=False)
            rows.append({
                "user_id": uid,
                "world_model_mse": float(mse),
                "persistence_mse": float(persistence),
                "agent_improvement": float(agent_df["improvement"].mean()),
                "random_improvement": float(base_df["improvement"].mean()),
            })
        except Exception as e:
            print(f"   !! Skipping {uid}: {e}")
            continue

    if retrain and restore:
        cfg_holdout = config.get('training', {}).get('holdout_user')
        if cfg_holdout:
            print(f"\n=== [LOSO] Restoring default checkpoints (holdout={cfg_holdout}) ===")
            try:
                _retrain_for_holdout(cfg_holdout)
            except Exception as e:
                print(f"   !! Restore failed ({e}); checkpoints reflect the last fold.")

    if not rows:
        print(" !!! LOSO produced no results.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(REPORT_DIR, "loso_per_user.csv"), index=False)

    mse_m, mse_ci = _mean_ci(df["world_model_mse"])
    pers_m, pers_ci = _mean_ci(df["persistence_mse"])
    ag_m, ag_ci = _mean_ci(df["agent_improvement"])
    rb_m, rb_ci = _mean_ci(df["random_improvement"])

    summary = {
        "n_participants": len(df),
        "mode": "true_loso" if retrain else "quick_leaky",
        "world_model_mse_mean": mse_m, "world_model_mse_ci95": mse_ci,
        "persistence_mse_mean": pers_m, "persistence_mse_ci95": pers_ci,
        "agent_improvement_mean": ag_m, "agent_improvement_ci95": ag_ci,
        "random_baseline_improvement_mean": rb_m, "random_baseline_improvement_ci95": rb_ci,
    }
    with open(os.path.join(REPORT_DIR, "loso_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("\n>> LOSO summary (mean +/- 95% CI):")
    print(f"   World-model MSE      : {mse_m:.5f} +/- {mse_ci:.5f}")
    print(f"   Persistence MSE      : {pers_m:.5f} +/- {pers_ci:.5f}")
    print(f"   Agent improvement    : {ag_m:.5f} +/- {ag_ci:.5f}")
    print(f"   Random baseline impr.: {rb_m:.5f} +/- {rb_ci:.5f}")
    print(f"\n LOSO report written to {REPORT_DIR}")


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "loso":
        run_loso_protocol(retrain=True)          # true LOSO (retrains per fold)
    elif arg == "loso-quick":
        run_loso_protocol(retrain=False)         # fast, leaky, shared checkpoints
    else:
        run_inference_protocol()