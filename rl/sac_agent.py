import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.common import get_device

# --- Configuration ---
HIDDEN_DIM = 256
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
# Numerical floor for the tanh-Jacobian log-prob correction (prevents log(0)).
LOG_STD_EPS = 1e-6

device = get_device()


def _project(a):
    """Project an action batch onto the unit sphere (L2-normalize per row).

    The critic only ever trains on APPLIED actions (Wolpertinger's retrieved
    song vectors), which live on the unit sphere. The actor's tanh sample lives
    in the cube (-1,1)^d with norm ~sqrt(d). Feeding the raw tanh sample to the
    critic during the actor/TD updates queries Q far off its training manifold,
    so we project to the sphere first. The tanh log-prob is then an approximation
    of the on-sphere action density — the accepted, lowest-risk consistency fix.
    """
    return F.normalize(a, p=2, dim=-1, eps=1e-8)

class QCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super(QCritic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

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
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):

        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        """
        Returns (action, log_prob, deterministic_action).

        action               : tanh-squashed stochastic sample in (-1, 1)^action_dim
        log_prob             : [batch] log-likelihood of `action` WITH the tanh
                               change-of-variables Jacobian correction applied
        deterministic_action : tanh(mean), the squashed greedy action used for
                               evaluation. Returned in place of the raw Gaussian
                               mean so the eval path also respects Box(-1, 1).
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        # Reparameterised pre-squash sample.
        x_t = normal.rsample()

        # tanh squashing -> bounded action in (-1, 1).
        action = torch.tanh(x_t)

        # log-prob with tanh change-of-variables correction:
        #   log p(a) = log p(x) - sum_i log(1 - tanh(x_i)^2 + eps)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1.0 - action.pow(2) + LOG_STD_EPS)
        log_prob = log_prob.sum(axis=-1)

        # Deterministic / evaluation action is the squashed mean.
        deterministic_action = torch.tanh(mean)

        return action, log_prob, deterministic_action

class SACAgent:
    def __init__(self, state_dim, action_dim, target_entropy_scale=1.0,
                 lr=LR, gamma=GAMMA, tau=TAU, hidden_dim=HIDDEN_DIM,
                 alpha=None, max_grad_norm=None, q_target_min=-1000.0):
        """
        alpha         : None  -> auto-tune the entropy temperature (canonical SAC).
                        float -> FIXED temperature, autotuning disabled. Use a
                        small value (e.g. 1e-3) or 0.0 to suppress the entropy
                        bonus, whose `-alpha*log_prob` term — with log_prob summed
                        over a 1024-dim action — otherwise inflates the Q target
                        and diverges the critic regardless of the entropy target.
        max_grad_norm : None  -> no gradient clipping.
                        float -> clip actor & critic grad norm to this value. The
                        standard stability fix against critic-loss blow-up.
        """

        # Stored on the instance (not read from module globals) so the
        # hyperparameter sweep can construct agents with differing gamma/tau/lr
        # without mutating shared state.
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        # Every reward is -distance <= 0, so the true return is <= 0. The critic
        # otherwise drifts POSITIVE (overestimation under the collapsed replay
        # buffer + residual entropy-bonus inflation); clamping the bootstrap
        # target to (q_target_min, 0] enforces Q <= 0 and stops the divergence.
        self.q_target_min = q_target_min

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = QCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = QCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Canonical SAC heuristic is target_entropy = -action_dim. In a 1024-dim
        # action space that target is extreme and the alpha auto-tuner can become
        # unstable (the tanh-squashed log-prob is bounded, so the policy may never
        # reach it, driving alpha up). `target_entropy_scale` (config:
        # training.target_entropy_scale) exposes this lever; 1.0 reproduces the
        # canonical value, <1.0 relaxes the entropy demand.
        self.target_entropy = -float(action_dim) * float(target_entropy_scale)

        # auto_alpha: learnable log_alpha + its own optimizer (canonical).
        # fixed alpha: log_alpha is a constant (no grad, no optimizer); alpha=0 is
        # clamped to a tiny floor so log() is finite and exp() ~ 0.
        self.auto_alpha = alpha is None
        if self.auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            log_a = math.log(max(float(alpha), 1e-12))
            self.log_alpha = torch.full((1,), log_a, device=device)
            self.alpha_optimizer = None

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
            next_action = _project(next_action)  # onto critic's unit-sphere manifold
            q1_next, q2_next = self.critic_target(next_state_batch, next_action)
            min_q_next = torch.min(q1_next, q2_next)
            
            alpha = self.log_alpha.exp()
            
            q_target = reward_batch + mask_batch * self.gamma * (min_q_next - alpha * next_log_prob.unsqueeze(1))
            # Returns are <= 0 by construction; clamp the target to keep Q in a
            # valid range and prevent positive-overestimation divergence.
            q_target = torch.clamp(q_target, min=self.q_target_min, max=0.0)

        q1, q2 = self.critic(state_batch, action_batch)
        
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # train actor
        action_new, log_prob_new, _ = self.actor.sample(state_batch)
        q1_new, q2_new = self.critic(state_batch, _project(action_new))
        min_q_new = torch.min(q1_new, q2_new)

        actor_loss = ((alpha * log_prob_new.unsqueeze(1)) - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # alpha: only auto-tuned when no fixed alpha was supplied.
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob_new + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return critic_loss.item(), actor_loss.item()
    
    def save(self, filename):

        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")

    def load(self, filename):
        
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth"))