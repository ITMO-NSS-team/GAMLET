import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def append(self, state, action, reward, done):
        self.states.append(state.numpy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def get_arrays(self):
        return self.states, self.actions, self.rewards, self.dones

class PPO(nn.Module):
    metadata = {'name': 'PPO'}

    def __init__(self, state_dim, action_dim, hidden_dim: int = 128, gamma: float = 1.0, batch_size: int = 128, epsilon: float = 0.3,
                 pi_lr: float = 5e-4, v_lr: float = 5e-4, epoch_n: int = 30, device: str = 'cpu'):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.pi_model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        self.v_model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon

        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_lr)
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters(), lr=v_lr)

        self.epoch_n = epoch_n
        self.device = device

        self.buffer = Buffer()

    def act(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        pi_out = self.pi_model(state_tensor.unsqueeze(0))

        dist = Categorical(probs=pi_out)
        action = dist.sample()

        return action.squeeze(0).cpu().numpy().item()

    def update(self):
        states, actions, rewards, dones = self.buffer.get_arrays()
        rewards, dones = map(np.array, [rewards, dones])

        rewards, dones = rewards.reshape(-1, 1), dones.reshape(-1, 1)
        returns = np.zeros_like(rewards)
        returns[-1] = rewards[-1]

        for t in range(rewards.shape[0] - 2, -1, -1):
            returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns[t+1]

        states, actions, returns = map(torch.FloatTensor, map(np.array, [states, actions, returns]))
        states = states.to(self.device)
        actions = actions.to(self.device)
        returns = returns.to(self.device)

        pi_out = self.pi_model(states)
        dist = Categorical(probs=pi_out)
        old_log_probs = dist.log_prob(actions).detach()

        for epoch in range(self.epoch_n):
            idxs = np.random.permutation(returns.shape[0])

            for i in range(0, returns.shape[0], self.batch_size):
                b_idxs = idxs[i:i + self.batch_size]
                b_states = states[b_idxs]
                b_actions = actions[b_idxs]
                b_returns = returns[b_idxs]
                b_old_log_probs = old_log_probs[b_idxs]

                b_advantage = b_returns - self.v_model(b_states)

                b_pi_out = self.pi_model(b_states)
                b_dist = Categorical(probs=b_pi_out)
                b_new_log_probs = b_dist.log_prob(b_actions)

                b_ratio = torch.exp(b_new_log_probs - b_old_log_probs)
                pi_loss_1 = b_ratio * b_advantage.detach()
                pi_loss_2 = torch.clamp(b_ratio, 1 - self.epsilon, 1 + self.epsilon) * b_advantage.detach()

                pi_loss = -torch.mean(torch.min(pi_loss_1, pi_loss_2))
                pi_loss.backward()
                self.pi_optimizer.step()
                self.pi_optimizer.zero_grad()

                v_loss = torch.mean(b_advantage ** 2)
                v_loss.backward()
                self.v_optimizer.step()
                self.v_optimizer.zero_grad()

        return pi_loss, v_loss

    def append_to_buffer(self, s, a, r, next_s, terminated):
        self.buffer.append(s, a, r, terminated)

    def clear_buffer(self):
        self.buffer.clear()

    def save(self, path):
        torch.save(self.pi_model.state_dict(), path + '_pi')
        torch.save(self.v_model.state_dict(), path + '_v')

    def load(self, path):
        if self.pi_model is None and self.v_model is None:
            raise 'Can\'t load weight before model assignment'

        self.pi_model.load_state_dict(torch.load(path + '_pi'))
        self.v_model.load_state_dict(torch.load(path + '_v'))
