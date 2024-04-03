import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torchinfo import summary

from rl_core.agent.ppo import CategoricalMasked

class Buffer:
    def __init__(self):
        super().__init__()
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.masks = []

        # Policy LSTM (hidden, cell)
        self.pi_hidden_states = []
        self.pi_cells_states = []

        # Value LSTM (hidden, cell)
        self.v_hidden_states = []
        self.v_cells_states = []

    def append(self, state, action, reward, done, mask):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.masks.append(mask)

    def append_pi_lstm(self, h):
        self.pi_hidden_states.append(h[0].squeeze(0).squeeze(0).cpu())
        self.pi_cells_states.append(h[1].squeeze(0).squeeze(0).cpu())

    def append_v_lstm(self, h):
        self.v_hidden_states.append(h[0].squeeze(0).squeeze(0).cpu())
        self.v_cells_states.append(h[1].squeeze(0).squeeze(0).cpu())

    def get_arrays(self):
        return self.states, self.actions, self.rewards, self.dones, self.masks, \
            (self.pi_hidden_states, self.pi_cells_states), (self.v_hidden_states, self.v_cells_states)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.masks = []
        self.pi_hidden_states = []
        self.pi_cells_states = []
        self.v_hidden_states = []
        self.v_cells_states = []


class PolicyModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers: int = 1, device: str = 'cuda'):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers=num_layers).to(device)
        self.layer_hidden = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.layer_policy_logits = nn.Linear(hidden_dim, action_dim).to(device)

        self.device = device
        self.h_cell = None

    def get_init_state(self, batch_size):
        self.h_cell = (
            torch.zeros([self.num_layers, batch_size, self.hidden_dim], dtype=torch.float).to(self.device),
            torch.zeros([self.num_layers, batch_size, self.hidden_dim], dtype=torch.float).to(self.device)
        )

    def forward(self, state):
        batch_size = state.shape[0]

        if self.h_cell is None or batch_size != self.h_cell[0].shape[1]:
            self.get_init_state(batch_size)

        _, self.h_cell = self.lstm(state.unsqueeze(0), self.h_cell)
        h_out = F.elu(self.layer_hidden(self.h_cell[0][-1]))

        pi_logits_out = self.layer_policy_logits(h_out)

        return pi_logits_out


class ValueModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_layers, device):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers=num_layers).to(device)
        self.layer_hidden = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.layer_value = nn.Linear(hidden_dim, 1).to(device)

        self.device = device
        self.h_cell = None

    def get_init_state(self, batch_size):
        self.h_cell = (
            torch.zeros([self.num_layers, batch_size, self.hidden_dim], dtype=torch.float).to(self.device),
            torch.zeros([self.num_layers, batch_size, self.hidden_dim], dtype=torch.float).to(self.device)
        )

    def forward(self, state):
        batch_size = state.shape[0]

        if self.h_cell is None or batch_size != self.h_cell[0].shape[1]:
            self.get_init_state(batch_size)

        _, self.h_cell = self.lstm(state.unsqueeze(0), self.h_cell)
        h_out = F.elu(self.layer_hidden(self.h_cell[0][-1]))
        v_out = self.layer_value(h_out)

        return v_out


class RPPO:
    metadata = {'name': 'PPO'}

    def __init__(self,
                 state_dim: int, action_dim: int, hidden_dim: int = 256,
                 gamma: float = 0.99, epsilon: float = 0.2, tau: float = 0.25,
                 batch_size: int = 10, epoch_n: int = 3,
                 pi_lr: float = 3e-5, v_lr: float = 1e-2, device: str = 'cpu'
                 ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.pi_model = PolicyModel(state_dim, action_dim, hidden_dim, num_layers=1, device=device)
        self.v_model = ValueModel(state_dim, hidden_dim, num_layers=1, device=device)

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.tau = tau
        self.pi_lr = pi_lr
        self.v_lr = v_lr

        self.pi_optimizer = torch.optim.AdamW(self.pi_model.parameters(), lr=pi_lr)
        self.v_optimizer = torch.optim.AdamW(self.v_model.parameters(), lr=v_lr)

        self.epoch_n = epoch_n
        self.device = device

        self.buffer = Buffer()
        self.probs = None

    def append_to_buffer(self, s, a, r, terminated, mask):
        self.buffer.append(s, a, r, terminated, mask)

    def clear_buffer(self):
        self.buffer.clear()

    def act(self, state, mask):
        # Policy
        state_tensor = torch.FloatTensor(state).to(self.device)
        mask = torch.from_numpy(mask).to(torch.bool).to(self.device)

        pi_logits = self.pi_model(state_tensor.unsqueeze(0))
        pi_probs = F.softmax(pi_logits, dim=1)

        dist = CategoricalMasked(logits=pi_probs, mask=mask)
        action = dist.sample()

        self.probs = torch.where(mask, dist.probs, 0)
        self.buffer.append_pi_lstm(self.pi_model.h_cell)

        # Value
        _ = self.v_model(state_tensor.unsqueeze(0))
        self.buffer.append_v_lstm(self.v_model.h_cell)

        return action.squeeze(0).cpu().numpy().item()

    def update(self):
        # TODO: Save history of hiddens in memory and use to train
        states, actions, rewards, dones, masks, (h_pi_in, h_pi_out), (h_v_in, h_v_out) = self.buffer.get_arrays()
        rewards, dones = map(np.array, [rewards, dones])

        rewards, dones = rewards.reshape(-1, 1), dones.reshape(-1, 1)
        returns = np.zeros_like(rewards)
        returns[-1] = rewards[-1]

        for t in range(rewards.shape[0] - 2, -1, -1):
            returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns[t + 1]

        states, actions, returns, masks = map(torch.FloatTensor, map(np.array, [states, actions, returns, masks]))
        states = states.to(self.device)
        actions = actions.to(self.device)
        returns = returns.to(self.device)
        masks = masks.to(self.device)

        h_pi_in = torch.stack(h_pi_in).to(self.device)
        h_pi_out = torch.stack(h_pi_out).to(self.device)
        h_v_in = torch.stack(h_v_in).to(self.device)
        h_v_out = torch.stack(h_v_out).to(self.device)

        pi_out = self.pi_model(states)

        full_dist = Categorical(logits=pi_out)
        old_log_probs = full_dist.log_prob(actions).detach()

        masked_dist = CategoricalMasked(logits=pi_out, mask=masks.to(torch.bool))
        m_old_log_probs = masked_dist.log_prob(actions).detach()

        for epoch in range(self.epoch_n):
            idxs = np.random.permutation(returns.shape[0])

            for i in range(0, returns.shape[0], self.batch_size):
                b_idxs = idxs[i:i + self.batch_size]
                b_states = states[b_idxs]
                b_actions = actions[b_idxs]
                b_returns = returns[b_idxs]
                b_old_log_probs = old_log_probs[b_idxs]
                b_masks = masks[b_idxs]
                b_m_old_log_probs = m_old_log_probs[b_idxs]
                b_h_pi_in, b_h_pi_out = h_pi_in[b_idxs], h_pi_out[b_idxs]
                b_h_v_in, b_h_v_out = h_v_in[b_idxs], h_v_out[b_idxs]

                self.v_model.h_cell = (b_h_v_in.unsqueeze(0), b_h_v_out.unsqueeze(0))
                b_advantage = b_returns - self.v_model(b_states)

                self.pi_model.h_cell = (b_h_pi_in.unsqueeze(0), b_h_pi_out.unsqueeze(0))
                b_pi_out = self.pi_model(b_states)

                b_full_dist = Categorical(logits=b_pi_out)
                b_new_log_probs = b_full_dist.log_prob(b_actions)

                b_masked_dist = CategoricalMasked(logits=b_pi_out, mask=b_masks.to(torch.bool))
                b_m_new_log_probs = b_masked_dist.log_prob(b_actions)

                entropy = b_masked_dist.entropy().mean()
                entropy_penalty = -self.tau * entropy

                # KL-Divergence
                kld = torch.nn.functional.kl_div(b_m_new_log_probs, b_m_old_log_probs,
                                                 log_target=True).detach().cpu().item()

                b_ratio = torch.exp(b_new_log_probs - b_old_log_probs)
                pi_loss_1 = b_ratio * b_advantage.detach()
                pi_loss_2 = torch.clamp(b_ratio, 1 - self.epsilon, 1 + self.epsilon) * b_advantage.detach()

                pi_loss = -torch.mean(torch.min(pi_loss_1, pi_loss_2)) + entropy_penalty
                pi_loss.backward()
                self.pi_optimizer.step()
                self.pi_optimizer.zero_grad()

                v_loss = torch.mean(b_advantage ** 2)
                v_loss.backward()
                self.v_optimizer.step()
                self.v_optimizer.zero_grad()

        return pi_loss, v_loss, kld

    def save(self, path):
        torch.save(self.pi_model.state_dict(), path + '_pi')
        torch.save(self.v_model.state_dict(), path + '_v')

    def load(self, path):
        if self.pi_model is None and self.v_model is None:
            raise 'Can\'t load weight before model assignment'

        self.pi_model.load_state_dict(torch.load(path + '_pi'))
        self.v_model.load_state_dict(torch.load(path + '_v'))

    def create_log_report(self, log_dir):
        with io.open(f'{log_dir}/params.log', 'w', encoding='utf-8') as file:
            file.write('-- PARAMS --\n')
            file.write(f'state_dim {self.state_dim}\n')
            file.write(f'action_dim {self.action_dim}\n')
            file.write(f'hidden_dim {self.hidden_dim}\n')
            file.write('--\n')
            file.write(f'gamma {self.gamma}\n')
            file.write(f'epsilon {self.epsilon}\n')
            file.write(f'tau {self.tau}\n')
            file.write(f'epoch_n {self.epoch_n}\n')
            file.write(f'batch_size {self.batch_size}\n')
            file.write('--\n')
            file.write(f'pi_lr {self.pi_lr}\n')
            file.write(f'v_lr {self.v_lr}\n')
            file.write(f'device {self.device}\n')
            # file.write('\n-- ARCHITECTURE --\n')
            # file.write('- PI MODEL -\n')
            # pi_model = str(summary(self.pi_model, (1, self.state_dim), verbose=0))
            # file.write(f'{pi_model}')
            # file.write('--\n')
            # file.write('- V MODEL -\n')
            # v_model = str(summary(self.v_model, (1, self.state_dim), verbose=0))
            # file.write(f'{v_model}')

