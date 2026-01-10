import torch
import numpy as np
import os
import random
from tqdm import tqdm

from simulator.gym_env import MusicEnv
from rl.sac_agent import SACAgent
from rl.wolpertinger import WolpertingerPolicy
from utils.common import ensure_dir

# --- Configuration ---
BATCH_SIZE = 64
REPLAY_SIZE = 100000
START_STEPS = 1000  
TRAIN_STEPS = 2000
CHECKPOINT_DIR = "rl/checkpoints"

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=REPLAY_SIZE):

        self.ptr = 0
        self.size = 0
        self.capacity = capacity
        self.state = np.zeros((capacity, state_dim))
        self.action = np.zeros((capacity, action_dim))
        self.reward = np.zeros((capacity, 1))
        self.next_state = np.zeros((capacity, state_dim))
        self.done = np.zeros((capacity, 1))

    def add(self, state, action, reward, next_state, done):

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):

        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state[ind],
            self.action[ind],
            self.reward[ind],
            self.next_state[ind],
            self.done[ind]
        )

def train_sac_agent(steps=TRAIN_STEPS):

    ensure_dir(CHECKPOINT_DIR)
    
    # env and agent initialization
    env = MusicEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f">> [Agent] state dim: {state_dim}, action Dim: {action_dim}")
    
    agent = SACAgent(state_dim, action_dim)
    memory = ReplayBuffer(state_dim, action_dim)
    
    # wolpertinger
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        wolpertinger = WolpertingerPolicy(k_neighbors=10, device=device)
        print(">> [Agent] wolpertinger active (k=10).")
    except Exception as e:
        print(f" !!! FAISS error: {e}")
        return

    # train
    state, _ = env.reset()
    episode_reward = 0
    
    print(f">> [Agent] Training for {steps} steps...")
    
    for step in tqdm(range(steps)):

        # action selection
        if step < START_STEPS:
            proto_action = env.action_space.sample()
        else:
            proto_action = agent.select_action(state, evaluate=False)

        # wolpertinger (continuous -> discrete real song)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        real_action, song_id = wolpertinger.select_action(proto_action, agent.critic, state_tensor)
        
        # env step
        next_state, reward, terminated, truncated, _ = env.step(real_action)
        done = terminated or truncated
        
        mask = 0 if done else 1
        
        # action storing
        memory.add(state, proto_action, reward, next_state, mask)
        
        state = next_state
        episode_reward += reward
        
        # weights
        if step >= START_STEPS:
            agent.update_parameters(memory, BATCH_SIZE)

        if done:
            tqdm.write(f"Episode finished. Reward: {episode_reward:.2f} | Last song: {song_id}")
            state, _ = env.reset()
            episode_reward = 0

            if step % 1000 == 0:
                agent.save(os.path.join(CHECKPOINT_DIR, "sac_checkpoint"))

    # save
    agent.save(os.path.join(CHECKPOINT_DIR, "sac_final"))
    print(">> [Agent] Training complete!")

if __name__ == "__main__":
    train_sac_agent()