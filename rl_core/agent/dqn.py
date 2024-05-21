import io
import random

import numpy as np
import torch.optim
from torch import nn
from torchinfo import summary


class Qfunction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        return self.network(_input)

    def save(self, path: str):
        torch.save(self.network.state_dict(), path)

    def load(self, path: str):
        self.network.load_state_dict(torch.load(path))


class DQN:
    """ https://arxiv.org/abs/1312.5602 """
    metadata = {'name': 'DQN'}

    def __init__(self, state_dim, action_dim, hidden_dim=512, gamma=0.01, lr=1e-4, batch_size=64, eps_decrease=1e-6,
                 eps_min=1e-3, device='cuda'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.q_function = Qfunction(state_dim, action_dim, hidden_dim, device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.eps = 1
        self.eps_decrease = eps_decrease
        self.eps_min = eps_min
        self.device = device
        self.memory = []
        self.optimizer = torch.optim.Adam(self.q_function.parameters(), lr=lr)

        self.probs = None

    def get_action(self, state, mask):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        mask_tensor = torch.tensor(mask, dtype=torch.bool).to(self.device)

        q_values = self.q_function(state)
        m_q_values = torch.where(mask_tensor, q_values, -torch.inf)
        argmax_action = torch.argmax(m_q_values)
        probs = self.eps * np.ones(self.action_dim) / sum(mask)
        m_probs = np.where(mask, probs, 0)
        m_probs[argmax_action] += 1 - self.eps
        action = np.random.choice(np.arange(self.action_dim), p=m_probs)

        return action

    def fit(self, state, action, reward, done, masks, next_state):
        loss_ = np.nan
        self.memory.append([state, action, reward, int(done), masks, next_state])

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, masks, next_states = map(torch.tensor, list(zip(*batch)))

            states = states.to(torch.float32).to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)
            masks = masks.to(torch.bool).to(self.device)
            next_states = next_states.to(torch.float32).to(self.device)

            targets = rewards + self.gamma * (1 - dones) * torch.max(self.q_function(next_states), dim=1).values
            q_values = self.q_function(states)[torch.arange(self.batch_size), actions]

            loss = torch.mean((q_values - targets.detach()) ** 2)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.eps > self.eps_min:
                self.eps -= self.eps_decrease

                if self.eps < 0:
                    self.eps = self.eps_min

            loss_ = loss.detach().cpu().item()

        return loss_

    def save(self, path: str):
        self.q_function.save(path)

    def load(self, path: str):
        self.q_function.load(path)

    def create_log_report(self, log_dir):
        with io.open(f'{log_dir}/params.log', 'w', encoding='utf-8') as file:
            file.write('-- PARAMS --\n')
            file.write(f'state_dim {self.state_dim}\n')
            file.write(f'action_dim {self.action_dim}\n')
            file.write(f'hidden_dim {self.hidden_dim}\n')
            file.write('--\n')
            file.write(f'gamma {self.gamma}\n')
            file.write(f'epsilon {self.eps}\n')
            file.write(f'eps_min {self.eps_min}\n')
            file.write(f'eps_decrease {self.eps_decrease}\n')
            file.write(f'batch_size {self.batch_size}\n')
            file.write(f'device {self.device}\n')
            file.write('--\n')
            file.write(f'pi_lr {self.lr}\n')
            file.write('\n-- ARCHITECTURE --\n')
            file.write('- PI MODEL -\n')
            q_function = str(summary(self.q_function, (1, self.state_dim), verbose=0))
            file.write(f'{q_function}')
