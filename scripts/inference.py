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
from physio.encoder import DualStreamEncoder
from rl.sac_agent import SACAgent
from rl.wolpertinger import WolpertingerPolicy
from utils.common import load_config, resolve_path

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
config = load_config("configs/config.yaml")

REPORT_DIR = "reports/evaluation"
os.makedirs(REPORT_DIR, exist_ok=True)
TARGET_USER = config.get('training', {}).get('holdout_user', 'hku1903')

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
        
    feat_cols = [c for c in df.columns if c.endswith('_score')]
    raw_vals = user_row[feat_cols].values.astype(np.float32)
    norm_vals = raw_vals / 10.0 
    
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
    user_clip_ids = data['clip_ids'][mask]
    user_song_ids = data['song_ids'][mask]
    
    enc_path = resolve_path("physio/checkpoints/physio_encoder.pth")
    encoder = DualStreamEncoder(embedding_dim=64).to(device)
    encoder.load_state_dict(torch.load(enc_path, map_location=device))
    encoder.eval()
    
    embeddings = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(user_windows), batch_size):

            batch = torch.tensor(user_windows[i:i+batch_size], dtype=torch.float32).to(device)
            eda = batch[:, :, 0:6]
            temp = batch[:, :, 12:18]
            dermal = torch.cat([eda, temp], dim=-1)
            bvp = batch[:, :, 6:12]
            hr = batch[:, :, 18:24]
            ibi = batch[:, :, 24:30]
            cardio = torch.cat([bvp, hr, ibi], dim=-1)
            emb = encoder(dermal, cardio)
            embeddings.append(emb.cpu().numpy())
            
    final_embs = np.vstack(embeddings)
    
    clip_map = {str(k): v for k, v in zip(user_clip_ids, final_embs)}
    
    song_map = {}

    for i, sid in enumerate(user_song_ids):
        sid = str(sid)
        cid = str(user_clip_ids[i])
        try:
            chunk_idx = int(cid.split('_')[-1])
            if sid not in song_map: song_map[sid] = []
            song_map[sid].append((chunk_idx, final_embs[i], cid))
        except: continue
        
    return clip_map, song_map, final_embs

def eval_world_model(user_vec, physio_lookup):
    """ test 1: prediction + graph """

    if not physio_lookup:
        print(" !!! Skipping world model eval (No data).")
        return 0.0

    print(f">> [Inference] Testing world model...")
    wm_path = resolve_path(config['paths']['world_model_path'])
    ctx_path = resolve_path("context/checkpoints/context_model.pth")
    
    world_model = WorldModel(state_dim=224, action_dim=1024, physio_dim=64).to(device)
    world_model.load_state_dict(torch.load(wm_path, map_location=device))
    world_model.eval()
    
    ctx_model = ContextTransformer(1024, 128).to(device)
    ctx_model.load_state_dict(torch.load(ctx_path, map_location=device))
    ctx_model.eval()
    
    ratings = pd.read_csv(resolve_path(config['paths']['ratings_csv']))
    user_ratings = ratings[ratings['participant_id'].astype(str) == str(TARGET_USER)].sort_values('song_no')
    
    audio_embs = np.load(resolve_path("data/processed/song_embeddings.npy"), allow_pickle=True)
    audio_ids = np.load(resolve_path("data/processed/song_id_map.npy"), allow_pickle=True)
    audio_map = {str(k): v for k, v in zip(audio_ids, audio_embs)}
    
    errors = []
    history_buffer = []
    
    predictions = []
    ground_truths = []
    
    for _, row in user_ratings.iterrows():
        sid = str(row['song_id'])
        
        with torch.no_grad():
            if not history_buffer:
                ctx = np.zeros(128, dtype=np.float32)
            else:
                seq = np.array(history_buffer[-5:])
                inp = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
                ctx = ctx_model(inp).cpu().numpy()[0]
        
        if sid in physio_lookup:
            clips = sorted(physio_lookup[sid], key=lambda x: x[0])
            
            action_vec = None
            for c_idx, _, c_id in clips:
                if c_id in audio_map:
                    action_vec = audio_map[c_id]
                    break
            
            if action_vec is not None:
                for i in range(len(clips) - 1):
                    curr_physio = clips[i][1]
                    next_physio_real = clips[i+1][1]
                    
                    state = np.concatenate([curr_physio, user_vec, ctx])
                    
                    s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    a_t = torch.tensor(action_vec, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        pred_physio = world_model(s_t, a_t).cpu().numpy()[0]
                    
                    mse = np.mean((pred_physio - next_physio_real)**2)
                    errors.append(mse)
                    
                    predictions.append(pred_physio[0]) 
                    ground_truths.append(next_physio_real[0])
                    
                history_buffer.append(action_vec)
    
    avg_mse = np.mean(errors) if errors else 0.0
    print(f"   >> MSE: {avg_mse:.5f}")
    
    # graph
    if len(ground_truths) > 0:
        plt.figure(figsize=(10, 5))
        limit = min(100, len(ground_truths))
        plt.plot(ground_truths[:limit], label="Real Physio", alpha=0.7)
        plt.plot(predictions[:limit], label="Predicted Physio", linestyle="--", alpha=0.7)
        plt.title(f"World Model Generalization: {TARGET_USER} (MSE: {avg_mse:.4f})")
        plt.xlabel("Step (Song Chunk)")
        plt.ylabel("Physio Feature Value (Standardized)")
        plt.legend()
        plt.savefig(os.path.join(REPORT_DIR, "physio_prediction.png"))
        plt.close()
        print(f"   >> Graph saved to {REPORT_DIR}/physio_prediction.png")
    
    return avg_mse

def eval_agent_control(user_vec, valid_targets):
    """ test 2: agent + graph """

    NUM_SESSIONS = 15
    K_NEIGHBORS = 10
    
    print(f">> [Inference] Testing agent ( Wolpertinger k={K_NEIGHBORS})...")
    
    state_dim = 224
    action_dim = 1024
    agent_path = resolve_path("rl/checkpoints/sac_final_actor.pth")
    critic_path = resolve_path("rl/checkpoints/sac_final_critic.pth")
    
    agent = SACAgent(state_dim, action_dim)
    agent.actor.load_state_dict(torch.load(agent_path, map_location=device))
    agent.actor.eval()
    
    agent.critic.load_state_dict(torch.load(critic_path, map_location=device))
    agent.critic.eval()
    
    wolpertinger = WolpertingerPolicy(k_neighbors=K_NEIGHBORS, device=device)
    
    from simulator.gym_env import MusicEnv
    env = MusicEnv()
    
    results = []
    
    for session in tqdm(range(NUM_SESSIONS)):
        state, _ = env.reset()
        
        env.current_user = user_vec 
        
        if len(valid_targets) > 0:
            rand_idx = np.random.randint(0, len(valid_targets))
            env.target_physio = valid_targets[rand_idx]
        
        env.state = np.concatenate([env.current_physio, env.current_user, env.current_context])
        
        target = env.target_physio
        rewards = []
        distances = []
        
        for step in range(20):
            proto = agent.select_action(env.state, evaluate=True)
            
            state_tensor = torch.FloatTensor(env.state).unsqueeze(0).to(device)
            real_action, _ = wolpertinger.select_action(proto, agent.critic, state_tensor)
            
            next_state, reward, _, _, _ = env.step(real_action)
            dist = np.linalg.norm(env.current_physio - target)
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
        if session == 0:
            plt.figure(figsize=(8, 5))
            plt.plot(distances, label="Distance to Target State")
            plt.xlabel("Step (Song)")
            plt.ylabel("Euclidean Distance")
            plt.title(f"Agent Control Trajectory ({TARGET_USER})")
            plt.legend()
            plt.savefig(os.path.join(REPORT_DIR, "agent_control_trajectory.png"))
            plt.close()
            print(f"   >> Graph saved to {REPORT_DIR}/agent_control_trajectory.png")

    df_res = pd.DataFrame(results)
    print("\n>> Agent performance:")
    print(df_res.describe())
    df_res.to_csv(os.path.join(REPORT_DIR, "agent_metrics.csv"), index=False)
    
    return df_res["avg_reward"].mean()

def run_inference_protocol():
    print(f"--- EVALUATION REPORT FOR {TARGET_USER} ---")
    
    try:
        u_vec = load_user_vector(TARGET_USER)
        
        _, physio_lookup, valid_targets = get_real_physio_embeddings(TARGET_USER)
        
        physics_score = eval_world_model(u_vec, physio_lookup)
        control_score = eval_agent_control(u_vec, valid_targets)
        
        summary = {
            "user_id": TARGET_USER,
            "world_model_mse": float(physics_score),
            "agent_avg_reward": float(control_score),
            "strategy": "Wolpertinger (k=10)"
        }
        
        with open(os.path.join(REPORT_DIR, "summary.json"), "w") as f:
            json.dump(summary, f, indent=4)
            
        print(f"\n Report generated in {REPORT_DIR}")
        
    except Exception as e:
        print(f"\n !!! Inference Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_inference_protocol()