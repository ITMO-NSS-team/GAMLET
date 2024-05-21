import io
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch import einsum
from einops import reduce
from torchinfo import summary


class Buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.masks = []

    def append(self, state, action, reward, done, mask):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.masks.append(mask)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.masks = []

    def get_arrays(self):
        return self.states, self.actions, self.rewards, self.dones, self.masks

    def get_size(self):
        states_size = self.convert_size(self.states.__sizeof__())
        actions_size = self.convert_size(self.actions.__sizeof__())
        rewards_size = self.convert_size(self.rewards.__sizeof__())
        dones_size = self.convert_size(self.dones.__sizeof__())
        masks_size = self.convert_size(self.masks.__sizeof__())

        return f'states: {states_size}, actions: {actions_size}, rewards: {rewards_size}, dones: {dones_size}, masks: {masks_size}'

    @staticmethod
    def convert_size(size_bytes):
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)

        return "%s %s" % (s, size_name[i])


class CategoricalMasked(Categorical):
    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self.mask = mask
        self.batch, self.nb_action = logits.size()

        if mask is None:
            super(CategoricalMasked, self).__init__(logits=logits)
        else:
            self.mask_value = torch.tensor(
                torch.finfo(logits.dtype).min, dtype=logits.dtype
            )

            logits = torch.where(self.mask, logits, self.mask_value)

            super(CategoricalMasked, self).__init__(logits=logits)

    def entropy(self):
        if self.mask is None:
            return super().entropy()

        p_log_p = einsum("ij,ij->ij", self.logits, self.probs)

        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device)
        )

        return -reduce(p_log_p, "b a -> b", "sum", b=self.batch, a=self.nb_action)


class PPO(nn.Module):
    """ https://arxiv.org/abs/1707.06347 """
    metadata = {'name': 'PPO'}

    def __init__(self,
                 state_dim: int, action_dim: int, hidden_dim: int = 512,
                 gamma: float = 0.99, epsilon: float = 0.2, tau: float = 1,
                 batch_size: int = 32,  epoch_n: int = 10,
                 pi_lr: float = 1e-4, v_lr: float = 1e-4,
                 device: str = 'cpu'
                 ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.pi_model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim), nn.ReLU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim), nn.ReLU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim), nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim / 2)), nn.ReLU(),
            nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4)), nn.ReLU(),
            nn.Linear(int(hidden_dim / 4), action_dim),
            nn.Softmax(dim=-1)
        ).to(device)

        self.v_model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim), nn.ReLU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim), nn.ReLU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim), nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim / 2)), nn.ReLU(),
            nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4)), nn.ReLU(),
            nn.Linear(int(hidden_dim / 4), 1),
        ).to(device)

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

    def act(self, state, mask):
        state_tensor = torch.FloatTensor(state).to(self.device)
        pi_out = self.pi_model(state_tensor.unsqueeze(0))

        mask = torch.from_numpy(mask).to(torch.bool).to(self.device)
        dist = CategoricalMasked(logits=pi_out, mask=mask)
        action = dist.sample()

        self.probs = torch.where(mask, dist.probs, 0)

        return action.squeeze(0).cpu().numpy().item()

    def update(self):
        states, actions, rewards, dones, masks = self.buffer.get_arrays()
        rewards, dones = map(np.array, [rewards, dones])

        rewards, dones = rewards.reshape(-1, 1), dones.reshape(-1, 1)
        returns = np.zeros_like(rewards)
        returns[-1] = rewards[-1]

        for t in range(rewards.shape[0] - 2, -1, -1):
            returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns[t+1]

        states, actions, returns, masks = map(torch.FloatTensor, map(np.array, [states, actions, returns, masks]))
        states = states.to(self.device)
        actions = actions.to(self.device)
        returns = returns.to(self.device)
        masks = masks.to(self.device)

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

                b_advantage = b_returns - self.v_model(b_states)

                b_pi_out = self.pi_model(b_states)

                b_full_dist = Categorical(logits=b_pi_out)
                b_new_log_probs = b_full_dist.log_prob(b_actions)

                b_masked_dist = CategoricalMasked(logits=b_pi_out, mask=b_masks.to(torch.bool))
                b_m_new_log_probs = b_masked_dist.log_prob(b_actions)

                entropy = b_masked_dist.entropy().mean()
                entropy_penalty = -self.tau * entropy

                # KL-Divergence
                kld = torch.nn.functional.kl_div(b_m_new_log_probs, b_m_old_log_probs, log_target=True).detach().cpu().item()

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

    def append_to_buffer(self, s, a, r, terminated, mask):
        self.buffer.append(s, a, r, terminated, mask)

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
            file.write('\n-- ARCHITECTURE --\n')
            file.write('- PI MODEL -\n')
            pi_model = str(summary(self.pi_model, (1, self.state_dim), verbose=0))
            file.write(f'{pi_model}')
            file.write('--\n')
            file.write('- V MODEL -\n')
            v_model = str(summary(self.v_model, (1, self.state_dim), verbose=0))
            file.write(f'{v_model}')
