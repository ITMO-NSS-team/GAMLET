import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from meta_automl.utils import project_root
from rl_core.agent.decision_transformer import DecisionTransformer
from rl_core.environments.time_series import TimeSeriesPipelineEnvironment
from rl_core.utils import OFFLINE_TRAJECTORIES, define_data_for_experiment

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def encode_one_hot(target_dim, num_dims):
    output = np.zeros(num_dims)
    output[target_dim] = 1

    return output

class EnvDataset(torch.utils.data.Dataset):
    def __init__(self, env, max_length, num_trajectories, goal):
        self.data = []
        self.max_length = max_length
        self.env = env

        while len(self.data) < num_trajectories:
            state, _ = env.reset()
            trajectory = []
            total_reward = 0

            chosen_trajectory = random.choice(OFFLINE_TRAJECTORIES)   # TODO: Sample Expert Trajectory

            # for t in range(max_length):
            for action in chosen_trajectory:
                mask_action = env.valid_action_mask()
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                trajectory.append((state, action, reward, mask_action))
                total_reward += reward
                state = next_state

                if done:
                    break

            for t in range(max_length - len(trajectory)):
                mask_action = env.valid_action_mask()
                trajectory.append((state, action, 0, mask_action))

            # if total_reward >= goal:
            self.data.append(trajectory)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trajectory = self.data[idx]
        states, actions, rewards, a_masks = zip(*trajectory)
        actions = [encode_one_hot(action, self.env.action_space.n) for action in actions]

        returns_to_go = np.cumsum(rewards[::-1])[::-1]

        return (returns_to_go, states, actions, a_masks)


def process_sample(R, s, a, max_length):
    pad_len_R = max_length - len(R)
    pad_len_s = max_length - len(s)
    pad_len_a = max_length - len(a)

    states = np.vstack((np.zeros((pad_len_s, len(s[0]))), s))
    actions = np.array([np.zeros(env.action_space.n)] * pad_len_a + list(a))
    return_to_go = np.pad(np.array(R), (pad_len_R, 0), 'constant')
    mask = np.zeros(max_length)

    if pad_len_s > 0:
        mask[-pad_len_s:] = 1

    return (return_to_go, states, actions, mask)


def collate_batch(batch):
    result = []

    max_length = max(len(sample[0]) for sample in batch)

    for R, s, a, a_mask in batch:
        return_to_go, states, actions, mask = process_sample(R, s, a, max_length)

        result.append((
            torch.FloatTensor(return_to_go),
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.LongTensor(mask),
            a_mask
        ))

    return zip(*result)

def evaluate_model(model, env, max_length, target_return, info_return=False):
    model.eval()

    state, _ = env.reset()
    done = False
    total_reward = 0
    R = [target_return]
    s, a, = [state], [encode_one_hot(0, env.action_space.n)]
    timesteps = np.arange(max_length)

    with torch.no_grad():
        while not done and len(s) < max_length:
            returns_to_go, states, actions, mask = process_sample(R, s, a, len(s))
            returns_to_go_tensor = torch.tensor(returns_to_go, dtype=torch.float).unsqueeze(0).unsqueeze(-1).to(DEVICE)

            states_tensor = torch.tensor(states, dtype=torch.float).unsqueeze(0).to(DEVICE)
            actions_tensor = torch.tensor(actions, dtype=torch.float).unsqueeze(0).to(DEVICE)
            timesteps_tensor = torch.tensor(timesteps, dtype=torch.long)[:len(s)].unsqueeze(0).to(DEVICE)
            mask_tensor = torch.tensor(mask, dtype=torch.long).unsqueeze(0).to(DEVICE)
            mask_action = env.valid_action_mask()

            action = model(
                returns_to_go_tensor / target_return,
                states_tensor,
                actions_tensor,
                timesteps_tensor,
                mask_tensor,
                mask_action
            )

            action = action.squeeze(0)[-1]
            action = torch.argmax(torch.softmax(action, dim=-1))
            action = action.cpu().detach().item()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            R.append(R[-1] - reward)
            s.append(state)

            a[-1] = encode_one_hot(action, env.action_space.n)
            a.append(encode_one_hot(0, env.action_space.n))

    if info_return:
        return total_reward, info

    else:
        return total_reward


def validate_model(model, dataloader, max_length, target_return):
    model.eval()  # Set the model to evaluation mode
    criterion = torch.nn.CrossEntropyLoss()
    timesteps = torch.LongTensor(np.arange(max_length)).unsqueeze(0).to(DEVICE)
    total_loss = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader), desc="Validation")

        for returns_to_go, states, actions, masks, a_masks in pbar:
            tensor_actions = torch.stack(actions).to(DEVICE)
            tensor_masks = torch.stack(masks).to(DEVICE)
            tensor_returns_to_go = torch.stack(returns_to_go).unsqueeze(-1).to(DEVICE)
            tensor_states = torch.stack(states).to(DEVICE)
            batch_timesteps = timesteps.repeat(len(states), 1)

            predicted_actions = model(
                tensor_returns_to_go / target_return,
                tensor_states,
                tensor_actions,
                batch_timesteps,
                tensor_masks,
                a_masks
            )

            tensor_actions = tensor_actions.reshape(-1, actions[0].shape[1])
            predicted_actions = predicted_actions.reshape(-1, actions[0].shape[1])

            tensor_actions = tensor_actions[tensor_masks.reshape(-1) == 0]
            predicted_actions = predicted_actions[tensor_masks.reshape(-1) == 0]
            loss = criterion(predicted_actions, tensor_actions.detach())

            total_loss += loss.item()
            pbar.set_postfix({"loss":loss.item()})

    average_loss = total_loss / len(dataloader)
    print(f"Average validation loss: {average_loss}")
    return average_loss


def train_model(model, train_dataloader, test_dataloader, optimizer, epochs, grad_clip_norm, target_return):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    timesteps = torch.LongTensor(np.arange(max_length)).unsqueeze(0).to(DEVICE)

    for epoch in tqdm(range(epochs), desc="Epoch"):
        total_loss = 0
        pbar = tqdm(train_dataloader, desc="Batch", total=len(train_dataloader))

        for returns_to_go, states, actions, masks, a_masks in pbar:
            optimizer.zero_grad()
            tensor_actions = torch.stack(actions).to(DEVICE)
            tensor_masks = torch.stack(masks).to(DEVICE)
            tensor_returns_to_go = torch.stack(returns_to_go).unsqueeze(-1).to(DEVICE)
            tensor_states = torch.stack(states).to(DEVICE)
            batch_timesteps = timesteps.repeat(len(states), 1)

            predicted_actions = model(
                tensor_returns_to_go / target_return,
                tensor_states,
                tensor_actions,
                batch_timesteps,
                tensor_masks,
                a_masks
            )

            tensor_actions = tensor_actions.reshape(-1, actions[0].shape[1])
            predicted_actions = predicted_actions.reshape(-1, actions[0].shape[1])

            tensor_actions = tensor_actions[tensor_masks.reshape(-1) == 0]
            predicted_actions = predicted_actions[tensor_masks.reshape(-1) == 0]
            loss = criterion(predicted_actions, tensor_actions.detach())

            loss.backward()

            if grad_clip_norm >= 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        print(f"Epoch {epoch}, Loss: {total_loss / len(train_dataloader)}")
        average_reward = np.mean([
            evaluate_model(model, env, max_length=max_length, target_return=max_return) for _ in tqdm(
                range(100), desc="Eval epsiode")
        ])
        print(f"Average Total Reward at epoch {epoch}: {average_reward}")

        validate_model(model, test_dataloader, max_length, target_return=max_return)


def inference_model(model, env_dataloader_test, test_list, max_length):
    train_data, test_data, meta_data = env_dataloader_test.get_data()
    env.load_data(train_data, test_data, meta_data)

    for _ in test_list:
        total_reward, info = evaluate_model(model, env, max_length, target_return=True, info_return=True)
        print(f'total_reward {total_reward} metric {info["metric"]}')
        print(f'\ninfo: {info}')
        info['pipeline'].show()



if __name__ == '__main__':
    number_of_nodes_in_pipeline = 5

    env = TimeSeriesPipelineEnvironment(
        max_number_of_nodes=number_of_nodes_in_pipeline,
        using_number_of_nodes=number_of_nodes_in_pipeline,
        render_mode='none',
        metadata_dim=126,
        is_use_dataloader=True
    )

    env_dataloader_train, env_dataloader_test, train_list, test_list = define_data_for_experiment()
    env.load_dataloader(env_dataloader_train)

    max_length = 100
    max_return = 100
    dataset_num = 5000

    dataset = EnvDataset(env, max_length=max_length, num_trajectories=dataset_num, goal=None)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=collate_batch)

    state_dim = env.state_dim
    action_dim = env.action_space.n

    embed_dim = 1024
    num_heads = 8
    num_layers = 8
    grad_clip_norm = 0.25

    model = DecisionTransformer(
        state_dim,
        action_dim,
        max_length,
        embed_dim,
        num_heads,
        num_layers
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    average_reward = np.mean([
        evaluate_model(model, env, max_length=max_length, target_return=max_return) for _ in tqdm(
            range(10), desc="Untrained eval")
    ])
    n_epochs = 40
    train_model(
        model, train_dataloader, test_dataloader, optimizer, epochs=n_epochs, grad_clip_norm=grad_clip_norm,
        target_return=max_return
    )

    exp_name = f'DecisionTransformer'
    log_dir = f'{project_root()}/MetaFEDOT/rl_core/agent/tensorboard_logs/dt/{number_of_nodes_in_pipeline}/{exp_name}'
    name = f'{model.metadata["name"]}_{n_epochs}'
    # os.mkdir(f'{log_dir}/weights')
    path = f'{log_dir}/weights/{name}'
    model.save(path)

    validate_model(model, test_dataloader, max_length, target_return=max_return)

    average_reward = np.mean([evaluate_model(model, env, max_length=max_length, target_return=max_return) for _ in
                              tqdm(range(100), desc="Trained eval")])
    print(f"Trained Average Reward: {average_reward}")






