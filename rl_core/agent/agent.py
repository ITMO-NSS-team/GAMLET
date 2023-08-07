from collections import deque

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def to_tensor(x, dtype=np.float64):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=dtype)
    x = torch.from_numpy(x).float()
    return x


def sample_batch(replay_buffer, n_samples):
    n_samples = min(len(replay_buffer), n_samples)

    indices = np.random.choice(len(replay_buffer), n_samples, replace=False)
    states, actions, rewards, next_actions, terminateds = [], [], [], [], []

    for i in indices:
        s, a, r, n_s, done = replay_buffer[i]

        states.append(s.numpy())
        actions.append(a)
        rewards.append(r)
        next_actions.append(n_s.numpy())
        terminateds.append(done)

    return np.array(states), np.array(actions), np.array(rewards), np.array(next_actions), np.array(terminateds)


class ActorBatch:
    def __init__(self):
        self.logprobs = []
        self.q_values = []

    def append(self, log_prob, q_value):
        self.logprobs.append(log_prob)
        self.q_values.append(q_value)

    def clear(self):
        self.logprobs.clear()
        self.q_values.clear()


class ActorCriticModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        from_dim = input_dim
        layers = []

        for to_dim in hidden_dims:
            layers.extend([
                nn.Linear(from_dim, to_dim),
                nn.ReLU()
            ])
            from_dim = to_dim

        self.net = nn.Sequential(*layers).to(self.device)

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)

        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
        ).to(self.device)

    def forward(self, state):
        state = self.net(state.to(self.device))

        action_probs = self.actor_head(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        q_values = self.critic_head(state)

        log_prob = dist.log_prob(action)
        action = action.item()

        # adv = q_values[action] - torch.sum(q_values * action_probs, -1)

        q_value = q_values[action]

        return action, log_prob, q_value

    def evaluate(self, state):
        state = self.net(state.to(self.device))

        q_values = self.critic_head(state)

        return q_values


class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, hidden_dims, lr, gamma, critic_rb_size):
        self.lr = lr
        self.gamma = gamma

        self.ac_model = ActorCriticModel(state_dim, hidden_dims, action_dim)
        self.actor_opt = self.critic_opt = self.opt = torch.optim.Adam(self.ac_model.parameters(), lr=lr)

        self.actor_batch = ActorBatch()
        self.critic_rb = deque(maxlen=critic_rb_size)

    def act(self, state):
        action, logprob, q_value = self.ac_model(to_tensor(state))
        self.actor_batch.append(logprob.cpu(), q_value.detach().cpu().numpy())
        return action

    def append_to_replay_buffer(self, s, a, r, next_s, terminated):
        self.critic_rb.append((s, a, r, next_s, terminated))

    def evaluate(self, state):
        return self.ac_model.evaluate(state)

    def update(self, rollout_size, critic_batch_size, critic_updates_per_actor):
        if len(self.actor_batch.q_values) < rollout_size:
            return None, None

        actor_loss = self.update_actor()
        critic_loss = self.update_critic(critic_batch_size, critic_updates_per_actor)

        return actor_loss, critic_loss

    def update_actor(self):
        Q_s_a = to_tensor(self.actor_batch.q_values)
        logprobs = torch.stack(self.actor_batch.logprobs)

        Q_s_a = (Q_s_a - Q_s_a.mean()) / (Q_s_a.std() + 1e-7)

        loss = torch.mean(-logprobs * Q_s_a)
        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        self.actor_batch.clear()

        return loss

    def update_critic(self, batch_size, n_updates=1):
        if len(self.critic_rb) < batch_size:
            return

        for _ in range(n_updates):
            train_batch = sample_batch(self.critic_rb, batch_size)
            states, actions, rewards, next_states, terminateds = train_batch

            self.critic_opt.zero_grad()
            loss = self.compute_td_loss(states, actions, rewards, next_states, terminateds)
            loss.backward()
            self.critic_opt.step()

        return loss

    def compute_td_loss(self, states, actions, rewards, next_states, terminated, regularizer=.1):
        s = to_tensor(states)
        a = to_tensor(actions, int).long()
        r = to_tensor(rewards)
        s_next = to_tensor(next_states)
        term = to_tensor(terminated, bool)

        Q_s_a = torch.gather(
            self.evaluate(s).cpu(), dim=1, index=torch.unsqueeze(a, 1)
        ).squeeze(1)

        with torch.no_grad():
            Q_sn = self.evaluate(s_next).cpu()
            V_sn, _ = torch.max(Q_sn, axis=-1)

        target = r + self.gamma * V_sn * torch.logical_not(term)
        td_error = Q_s_a - target

        loss = torch.mean(td_error ** 2)
        loss += regularizer * Q_s_a.mean()

        return loss

    def save(self, path):
        torch.save(self.ac_model.state_dict(), path)

    def load(self, path):
        if self.ac_model is None:
            raise 'Can\'t load weight before model assignment'

        self.ac_model.load_state_dict(torch.load(path))
