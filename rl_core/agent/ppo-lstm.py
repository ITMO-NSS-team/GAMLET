import io
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.next_states = []
        self.probs_action = []
        self.hiddens_in = []
        self.hiddens_out = []


    def append(self, state, action, reward, done, mask, next_state, prob_a, h_in, h_out):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.masks.append(mask)
        self.next_states.append(next_state)
        self.probs_action.append([prob_a])
        self.hiddens_in.append([h_in])
        self.hiddens_out.append([h_out])

    def get_arrays(self):
        return self.states, self.actions, self.rewards, self.dones, self.masks,\
            self.next_state, self.prob_a, self.h_in, self.h_out

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.masks = []
        self.next_states = []
        self.probs_action = []
        self.hiddens_in = []
        self.hiddens_out = []


class PPO_LSTM(nn.Module):
    metadata = {'name': 'PPO'}

    def __init__(self,
                 state_dim: int, action_dim: int, hidden_dim: int = 256,
                 gamma: float = 0.99, epsilon: float = 0.2, tau: float = 0.25,
                 batch_size: int = 10, epoch_n: int = 3,
                 lr: float = 3e-5, device: str = 'cpu'
                 ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, 2 * hidden_dim),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(2 * hidden_dim, hidden_dim)
        self.fc_pi = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.fc_v = nn.Linear(hidden_dim, 1)

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.tau = tau
        self.lr = lr

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.epoch_n = epoch_n
        self.device = device

        self.buffer = Buffer()
        self.probs = None

    def pi(self, x, hidden):
        x = self.fc1(x)
        x = x.view(-1, 1, self.hidden_dim)
        x, lstm_hidden = self.lstm(x, hidden)

        prob = F.softmax(x, dim=2)

        return prob, lstm_hidden

    def v(self, x, hidden):
        x = self.fc1(x)
        x = x.view(-1, 1, self.hidden_dim)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)

        return v

    def init_hidden(self):
        return (torch.zeros([1, 1, self.hidden_dim], dtype=torch.float), torch.zeros([1, 1, self.hidden_dim], dtype=torch.float))

    def to_buffer(self, transition):
        self.buffer.append(transition)

    def create_batch(self):
        states, actions, rewards, states_prime, probs_action, hiddens_in, hiddens_out, dones = [], [], [], [], [], [], [], []

        for transition in self.buffer:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition

            states.append(s)
            actions.append([a])
            rewards.append([r])
            states_prime.append(s_prime)
            probs_action.append([prob_a])
            hiddens_in.append([h_in])
            hiddens_out.append([h_out])
            dones.append(done)

        s = torch.tensor(states, dtype=torch.float32)
        a = torch.tensor(actions, dtype=torch.int)
        r = torch.tensor(rewards, dtype=torch.float32)
        s_prime = torch.tensor(states_prime, dtype=torch.float32)
        done = torch.tensor(dones, dtype=torch.bool)
        prob_a = torch.tensor(probs_action, dtype=torch.float32)

        self.buffer = []

        return s, a, r, s_prime, dones, prob_a

    def act(self, state, mask):
        state_tensor = torch.FloatTensor(state).to(self.device)
        pi_out, h_out = self.fc_pi()

    def update(self):
        s, a, r, s_prime, done, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.create_batch()

        first_hidden = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(self.epoch_n):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + self.gamma * v_prime * done
            v_s = self.v(s, first_hidden).squeeze(1)

            delta = td_target - v_s
            delta = delta.detach().numpy()

            advantage = []


