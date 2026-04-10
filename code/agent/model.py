"""
The Agent Brain — Pure Mathematical Reasoning Network

Architecture: Actor-Critic with residual connections
- No language, no text, no human priors
- Input: normalized numerical state vector
- Output: a continuous number (the answer)

The network evolves its internal representations purely through
gradient descent on mathematical reward signals.

Key design choices:
- Residual connections: allow gradients to flow cleanly through depth
- LayerNorm: keeps activations stable during long training
- SiLU activation: smooth, non-zero gradient everywhere (better than ReLU for math)
- Separate Actor (what answer?) and Critic (how confident?) heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """A residual block: allows the network to learn corrections, not just mappings."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.net(x))


class MathReasoningNetwork(nn.Module):
    """
    The core neural network.

    Takes a 5-dim state vector, outputs (action_mean, action_std, value).
    - action_mean: the agent's best guess at the answer
    - action_std:  the agent's uncertainty (learned, not fixed)
    - value:       how good this state is expected to be (for PPO critic)
    """

    def __init__(self, state_dim=5, hidden_dim=256, num_residual_blocks=4):
        super().__init__()

        # Input projection: lift 5-dim input to hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Deep residual trunk — where mathematical structure is learned
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)
        ])

        # Actor head: outputs normalized answer in [-1, 1] via tanh
        # Environment then scales this to the actual answer range per level.
        # This fixes the core issue: agent starts with bounded guesses instead
        # of random values in [-1e6, 1e6] which gave near-zero reward everywhere.
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),  # bound output to [-1, 1]
        )

        # Learned log std — agent learns its own uncertainty
        self.log_std = nn.Parameter(torch.zeros(1))

        # Critic head: estimates expected reward (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Small initial weights — the agent starts nearly random, learns everything."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, state):
        x = self.input_proj(state)
        for block in self.residual_blocks:
            x = block(x)

        mean = self.actor_mean(x)
        std = torch.exp(self.log_std.clamp(-4, 2))
        value = self.critic(x)

        return mean, std, value

    def get_action_and_value(self, state, action=None):
        """
        Sample an action from the learned distribution, compute log prob and value.
        Used during PPO training.
        """
        mean, std, value = self.forward(state)
        dist = torch.distributions.Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return action, log_prob, entropy, value.squeeze(-1)

    def get_value(self, state):
        _, _, value = self.forward(state)
        return value.squeeze(-1)


class PPOAgent:
    """
    Proximal Policy Optimization agent.

    PPO is a policy gradient method that:
    1. Collects experience by interacting with the environment
    2. Estimates advantages (how much better than expected)
    3. Updates the policy — but clips updates to prevent catastrophic forgetting
    4. Repeats

    No human demonstrations. No pre-training. Pure trial-and-reward evolution.
    """

    def __init__(
        self,
        state_dim=5,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.network = MathReasoningNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        print(f"Agent initialized on: {self.device}")
        total_params = sum(p.numel() for p in self.network.parameters())
        print(f"Network parameters: {total_params:,}")

    def select_action(self, state: np.ndarray, deterministic=False):
        """Select an action given a state. Called during environment interaction."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, std, _ = self.network(state_t)
            if deterministic:
                action = mean
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
        return action.squeeze().cpu().numpy()

    def update(self, rollout):
        """
        Run PPO update on a collected rollout.
        rollout is a dict with: states, actions, rewards, log_probs, values, dones
        """
        states = torch.FloatTensor(rollout["states"]).to(self.device)
        actions = torch.FloatTensor(rollout["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        rewards = torch.FloatTensor(rollout["rewards"]).to(self.device)
        values = torch.FloatTensor(rollout["values"]).to(self.device)

        # Compute returns and advantages
        returns = self._compute_returns(rewards)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update — multiple epochs over the same data
        total_loss = 0.0
        for _ in range(4):  # PPO epochs
            _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                states, actions.unsqueeze(-1)
            )

            # Ratio of new to old policy
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value function loss
            value_loss = F.mse_loss(new_values, returns)

            # Total loss
            loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / 4

    def _compute_returns(self, rewards, normalize=True):
        """Discounted cumulative returns."""
        returns = torch.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running
            returns[t] = running
        if normalize:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def save(self, path):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
        print(f"Agent saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Agent loaded from {path}")
