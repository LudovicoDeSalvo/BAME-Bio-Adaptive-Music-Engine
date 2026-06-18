import torch
import numpy as np
import os
import random
from collections import deque
from tqdm import tqdm

from simulator.gym_env import MusicEnv
from rl.sac_agent import SACAgent
from rl.wolpertinger import WolpertingerPolicy
from utils.common import ensure_dir, get_device, load_config, resolve_path, set_seed

# --- Configuration ---
BATCH_SIZE = 64
REPLAY_SIZE = 100000
START_STEPS = 1000  # warmup steps of random proto-actions before learning
TRAIN_STEPS = 20000  # default budget; must exceed START_STEPS or 0 updates happen
SAVE_EVERY = 1000
CHECKPOINT_DIR = "rl/checkpoints"
LOG_EVERY = 1000  # steps between diagnostic prints (loss / alpha / reward MA)
REWARD_MA_WINDOW = 20  # episodes in the reward moving average
# Critic loss above this (or non-finite) flags Q divergence. The healthy regime
# is O(reward^2) ~ 1e5 here; runaway runs reach 1e9+ within a few thousand steps.
CRITIC_DIVERGE_THRESHOLD = 1e9
COLLAPSE_WINDOW = 50  # episodes inspected for the song-collapse metric

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=REPLAY_SIZE):

        self.ptr = 0
        self.size = 0
        self.capacity = capacity
        self.state = np.zeros((capacity, state_dim))
        self.action = np.zeros((capacity, action_dim))
        self.reward = np.zeros((capacity, 1))
        self.next_state = np.zeros((capacity, state_dim))
        # Stores the continue-mask (1 = not done, 0 = terminal/truncated), NOT a
        # done flag; SAC bootstraps the next-state value by reward + mask*gamma*...
        self.mask = np.zeros((capacity, 1))

    def add(self, state, action, reward, next_state, mask):

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.mask[self.ptr] = mask
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):

        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state[ind],
            self.action[ind],
            self.reward[ind],
            self.next_state[ind],
            self.mask[ind]
        )

def train_sac_agent(steps=TRAIN_STEPS, config_path="configs/config.yaml",
                    overrides=None, quiet=False, return_metrics=False, save=True):
    """Train the SAC agent.

    overrides : optional dict of training-section hyperparameter overrides
                (e.g. {"target_entropy_scale": 0.05, "lr": 1e-4, "reward_scale": 1.0}).
                Used by the hyperparameter sweep; None reproduces config.yaml.
    quiet     : suppress per-episode / progress output (sweep trials).
    return_metrics : return a metrics dict describing the run (for the sweep).
    save      : write checkpoints. Disabled during sweeps so trials don't clobber
                the real `sac_final` checkpoint.
    """

    ensure_dir(CHECKPOINT_DIR)

    # reproducibility: seed every RNG before anything stochastic happens
    config = load_config(config_path)

    # Effective training config = on-disk config.yaml + per-trial overrides.
    tcfg = dict(config.get("training", {}))
    tcfg.update(overrides or {})

    seed = int(tcfg.get("seed", 42))
    set_seed(seed)

    # Fail loud BEFORE building the env: a missing world model would otherwise
    # silently fall back to an untrained random simulator, and the agent would
    # train against noise while looking like a real run.
    wm_path = resolve_path(config['paths']['world_model_path'])
    ctx_path = resolve_path("context/checkpoints/context_model.pth")
    missing = [p for p in (wm_path, ctx_path) if not os.path.exists(p)]
    if missing:
        print(f" !!! Cannot train agent: missing {missing}. "
              f"Train the World Model [7] and Context Model [6] first.")
        return None if return_metrics else None

    # env and agent initialization. Pass the merged training block to the env so
    # reward_scale / max_steps honor the overrides too.
    env = MusicEnv(overrides={"training": tcfg})
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if not quiet:
        print(f">> [Agent] state dim: {state_dim}, action Dim: {action_dim}")

    # Resolve every swept hyperparameter from the merged config (single source).
    target_entropy_scale = float(tcfg.get("target_entropy_scale", 1.0))
    lr = float(tcfg.get("lr", 3e-4))
    gamma = float(tcfg.get("gamma", 0.99))
    tau = float(tcfg.get("tau", 0.005))
    hidden_dim = int(tcfg.get("hidden_dim", 256))
    batch_size = int(tcfg.get("batch_size", BATCH_SIZE))
    k_neighbors = int(tcfg.get("k_neighbors", 10))
    # alpha=None -> auto-tune entropy temperature; a float fixes it (disables the
    # auto-tuner). max_grad_norm=None -> no clipping; a float clips actor+critic.
    alpha = tcfg.get("alpha", None)
    if alpha is not None:
        alpha = float(alpha)
    max_grad_norm = tcfg.get("max_grad_norm", None)
    if max_grad_norm is not None:
        max_grad_norm = float(max_grad_norm)
    q_target_min = float(tcfg.get("q_target_min", -1000.0))
    # Exploration: Gaussian noise added to the proto-action during training,
    # decayed linearly to 0. Diversifies the songs Wolpertinger retrieves so the
    # policy doesn't collapse to one action — without raising alpha (which would
    # re-inflate the 1024-dim entropy bonus). Eval is unaffected (deterministic).
    expl_noise = float(tcfg.get("expl_noise", 0.0))

    agent = SACAgent(state_dim, action_dim, target_entropy_scale=target_entropy_scale,
                     lr=lr, gamma=gamma, tau=tau, hidden_dim=hidden_dim,
                     alpha=alpha, max_grad_norm=max_grad_norm, q_target_min=q_target_min)
    memory = ReplayBuffer(state_dim, action_dim)

    # wolpertinger
    device = get_device()

    try:
        wolpertinger = WolpertingerPolicy(k_neighbors=k_neighbors, device=device)
        if not quiet:
            print(f">> [Agent] wolpertinger active (k={k_neighbors}).")
    except Exception as e:
        print(f" !!! FAISS error: {e}")
        return None if return_metrics else None

    # train
    env.action_space.seed(seed)
    state, _ = env.reset(seed=seed)
    episode_reward = 0

    # Warmup must be strictly less than the budget, otherwise the update gate
    # (step >= warmup) never fires and the agent does ZERO gradient steps.
    warmup = min(START_STEPS, max(1, steps // 2))

    if not quiet:
        print(f">> [Agent] Training for {steps} steps (warmup {warmup})...")

    # Diagnostics: without these the run prints only raw episode reward and you
    # cannot tell critic divergence from actor collapse from a bad alpha. Track a
    # reward moving average plus the per-interval mean critic/actor loss.
    recent_rewards = deque(maxlen=REWARD_MA_WINDOW)
    recent_songs = deque(maxlen=COLLAPSE_WINDOW)
    loss_sum = {"critic": 0.0, "actor": 0.0, "n": 0}

    # Metrics accumulated for the sweep leaderboard.
    last_critic_loss = float("nan")
    last_actor_loss = float("nan")
    max_alpha = 0.0
    critic_diverged = False
    reward_ma_history = []

    for step in tqdm(range(steps), disable=quiet):

        # action selection
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        catalog = getattr(wolpertinger.retriever, "embeddings", None)

        if step < warmup:
            # True-random warmup: apply a random REAL catalog song directly,
            # bypassing the (untrained) critic re-rank — otherwise the random
            # proto is still routed through Wolpertinger and the critic filters
            # the candidates, so warmup is not actually uniform exploration.
            if catalog is not None and len(catalog) > 0:
                j = int(np.random.randint(len(catalog)))
                real_action = catalog[j]
                song_id = wolpertinger.retriever.song_ids[j]
            else:
                proto_action = env.action_space.sample()
                real_action, song_id = wolpertinger.select_action(proto_action, agent.critic, state_tensor)
        else:
            proto_action = agent.select_action(state, evaluate=False)
            if expl_noise > 0:
                decay = max(0.0, 1.0 - step / max(1, steps))
                proto_action = proto_action + np.random.randn(action_dim).astype(np.float32) * expl_noise * decay
            real_action, song_id = wolpertinger.select_action(proto_action, agent.critic, state_tensor)
        
        # env step
        next_state, reward, terminated, truncated, _ = env.step(real_action)
        done = terminated or truncated

        # Bootstrap mask: only a TRUE terminal state (terminated) should zero the
        # next-state value. `truncated` is a time-limit cutoff, not an absorbing
        # state, so we must still bootstrap through it — zeroing the mask on
        # truncation would teach the agent that the horizon end has no future
        # value and bias every Q target at the episode boundary.
        mask = 0 if terminated else 1

        # Store the APPLIED action (Wolpertinger's real_action), not proto_action:
        # the critic's Q(s,a) must match the dynamics/reward that produced (r, s').
        memory.add(state, real_action, reward, next_state, mask)
        
        state = next_state
        episode_reward += reward
        
        # weights
        if step >= warmup:
            critic_loss, actor_loss = agent.update_parameters(memory, batch_size)
            loss_sum["critic"] += critic_loss
            loss_sum["actor"] += actor_loss
            loss_sum["n"] += 1
            last_critic_loss, last_actor_loss = critic_loss, actor_loss
            max_alpha = max(max_alpha, agent.log_alpha.exp().item())
            if not np.isfinite(critic_loss) or critic_loss > CRITIC_DIVERGE_THRESHOLD:
                critic_diverged = True

        # Diagnostic line: mean loss over the interval, current alpha, reward MA,
        # and buffer fill. A healthy run shows critic_loss settling, actor_loss
        # trending down, alpha stabilizing, and reward_ma climbing toward 0.
        if step > 0 and step % LOG_EVERY == 0 and loss_sum["n"] > 0:
            alpha = agent.log_alpha.exp().item()
            avg_c = loss_sum["critic"] / loss_sum["n"]
            avg_a = loss_sum["actor"] / loss_sum["n"]
            reward_ma = (sum(recent_rewards) / len(recent_rewards)) if recent_rewards else float("nan")
            reward_ma_history.append(reward_ma)
            if not quiet:
                tqdm.write(
                    f">> [step {step}] critic_loss={avg_c:.3f} actor_loss={avg_a:.3f} "
                    f"alpha={alpha:.4f} reward_ma({len(recent_rewards)}ep)={reward_ma:.1f} "
                    f"buffer={memory.size}"
                )
            loss_sum = {"critic": 0.0, "actor": 0.0, "n": 0}

        # Periodic checkpoint at the loop level (NOT nested under `if done`:
        # done-steps land at 49,99,... and never coincide with the save period).
        if save and step > 0 and step % SAVE_EVERY == 0:
            agent.save(os.path.join(CHECKPOINT_DIR, "sac_checkpoint"))

        if done:
            recent_rewards.append(episode_reward)
            recent_songs.append(song_id)
            if not quiet:
                tqdm.write(f"Episode finished. Reward: {episode_reward:.2f} | Last song: {song_id}")
            state, _ = env.reset()
            episode_reward = 0

    # save
    if save:
        agent.save(os.path.join(CHECKPOINT_DIR, "sac_final"))
    if not quiet:
        print(">> [Agent] Training complete!")

    # Final flush: record the end-of-training reward MA so short runs (and the
    # tail past the last LOG_EVERY boundary) always contribute a sample. Without
    # this, any run shorter than LOG_EVERY has an empty history and scores as nan.
    if recent_rewards:
        reward_ma_history.append(sum(recent_rewards) / len(recent_rewards))

    if not return_metrics:
        return None

    # Collapse fraction: share of the most-frequent song over the last
    # COLLAPSE_WINDOW episodes. ~1.0 = degenerate (one song always picked);
    # near 1/n_songs = healthy diversity.
    if recent_songs:
        counts = {}
        for s in recent_songs:
            counts[s] = counts.get(s, 0) + 1
        collapse_fraction = max(counts.values()) / len(recent_songs)
    else:
        collapse_fraction = float("nan")

    final_reward_ma = reward_ma_history[-1] if reward_ma_history else float("nan")
    finite_ma = [r for r in reward_ma_history if np.isfinite(r)]
    best_reward_ma = max(finite_ma) if finite_ma else float("nan")

    # Scale-invariant objective: raw reward_ma is in units of
    # reward_scale * max_steps (episode-summed -RMS_distance * reward_scale), so
    # it is NOT comparable across trials that vary reward_scale or max_steps.
    # Divide it back out to recover the mean per-step normalized distance to the
    # target (std units) — the same geometry the env/eval optimize. Lower=better.
    reward_scale = float(tcfg.get("reward_scale", 10.0))
    max_steps = int(tcfg.get("max_steps", 50))
    denom = max_steps * reward_scale
    mean_norm_distance = (-best_reward_ma / denom) if (denom and np.isfinite(best_reward_ma)) else float("nan")

    return {
        "hyperparams": {
            "target_entropy_scale": target_entropy_scale, "lr": lr, "gamma": gamma,
            "tau": tau, "hidden_dim": hidden_dim, "batch_size": batch_size,
            "k_neighbors": k_neighbors,
            "reward_scale": float(tcfg.get("reward_scale", 10.0)),
            "max_steps": int(tcfg.get("max_steps", 50)),
            "alpha": alpha, "max_grad_norm": max_grad_norm,
            "q_target_min": q_target_min, "expl_noise": expl_noise,
        },
        "steps": steps,
        "n_episodes": len(recent_rewards),  # last-window count; full count not retained
        "final_reward_ma": final_reward_ma,
        "best_reward_ma": best_reward_ma,
        "mean_norm_distance": mean_norm_distance,  # scale-invariant, lower=better
        "final_critic_loss": last_critic_loss,
        "final_actor_loss": last_actor_loss,
        "final_alpha": agent.log_alpha.exp().item(),
        "max_alpha": max_alpha,
        "critic_diverged": critic_diverged,
        "collapse_fraction": collapse_fraction,
        "reward_ma_history": reward_ma_history,
    }

if __name__ == "__main__":
    train_sac_agent()