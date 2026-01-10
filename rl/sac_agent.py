import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- Configuration ---
HIDDEN_DIM = 256
GAMMA = 0.99
TAU = 0.005
LR = 3e-4 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QCritic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, HIDDEN_DIM)
        self.l2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.l3 = nn.Linear(HIDDEN_DIM, 1)

        self.l4 = nn.Linear(state_dim + action_dim, HIDDEN_DIM)
        self.l5 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.l6 = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, state, action):

        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, HIDDEN_DIM)
        self.l2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        
        self.mean = nn.Linear(HIDDEN_DIM, action_dim)
        self.log_std = nn.Linear(HIDDEN_DIM, action_dim)

    def forward(self, state):

        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):

        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  
        action = x_t
        log_prob = normal.log_prob(x_t).sum(axis=-1)

        return action, log_prob, mean

class SACAgent:
    def __init__(self, state_dim, action_dim):

        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = QCritic(state_dim, action_dim).to(device)
        self.critic_target = QCritic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR)
        
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR)

    def select_action(self, state, evaluate=False):

        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        if evaluate:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)

        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size):

        # batch
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        
        reward_batch = torch.FloatTensor(reward_batch).to(device)
        mask_batch = torch.FloatTensor(mask_batch).to(device)

        # train critic
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state_batch)
            q1_next, q2_next = self.critic_target(next_state_batch, next_action)
            min_q_next = torch.min(q1_next, q2_next)
            
            alpha = self.log_alpha.exp()
            
            q_target = reward_batch + mask_batch * GAMMA * (min_q_next - alpha * next_log_prob.unsqueeze(1))

        q1, q2 = self.critic(state_batch, action_batch)
        
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # train actor
        action_new, log_prob_new, _ = self.actor.sample(state_batch)
        q1_new, q2_new = self.critic(state_batch, action_new)
        min_q_new = torch.min(q1_new, q2_new)
        
        actor_loss = ((alpha * log_prob_new.unsqueeze(1)) - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # alpha
        alpha_loss = -(self.log_alpha * (log_prob_new + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            
        return critic_loss.item(), actor_loss.item()
    
    def save(self, filename):

        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")

    def load(self, filename):
        
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))