import datetime
import io
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from gamlet.utils import project_root
from rl_core.agent.dqn import DQN
from rl_core.environments.time_series import TimeSeriesPipelineEnvironment
from rl_core.utils import define_data_for_experiment


def print_params(experiment_name, number_of_nodes_in_pipeline=8):
    log_dir = f'{project_root()}/MetaFEDOT/rl_core/agent/tensorboard_logs/dqn/' \
              f'{number_of_nodes_in_pipeline}/{experiment_name}'

    with io.open(f'{log_dir}/params.log', 'r', encoding='utf-8') as file:
        lines = file.readlines()

        hidden_dim = lines[3].split(' ')[1].replace("\n", "")
        gamma = lines[5].split(' ')[1].replace("\n", "")
        epsilon = lines[6].split(' ')[1].replace("\n", "")
        eps_min = lines[7].split(' ')[1].replace("\n", "")
        eps_decrease = lines[8].split(' ')[1].replace("\n", "")

    print(hidden_dim, '\t', gamma, '\t', epsilon, '\t', eps_min, '\t', eps_decrease)


def run_experiment(n_episodes, number_of_nodes_in_pipeline, hidden_dim, gamma, eps, eps_min, eps_decrease):
    dataloader_train, dataloader_test, train_list, test_list = define_data_for_experiment()

    env = TimeSeriesPipelineEnvironment(
        max_number_of_nodes=number_of_nodes_in_pipeline,
        render_mode='none',
        metadata_dim=126
    )

    state_dim, action_dim = env.state_dim, env.action_dim

    agent = DQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        gamma=gamma,
        eps_min=eps_min,
        eps_decrease=eps_decrease
    )

    time = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    log_dir = f'{project_root()}/MetaFEDOT/rl_core/agent/tensorboard_logs/dqn/{number_of_nodes_in_pipeline}_val/{time}'
    tb_writer = SummaryWriter(log_dir=log_dir)
    agent.create_log_report(log_dir)

    # -- Starting experiment --
    total_rewards = []
    total_metrics = []

    period = 20

    for episode in range(1, n_episodes + 1):
        print(f'-- Starting {episode} episode --')

        train_data, test_data, meta_data = dataloader_train.get_data()
        env.load_data(train_data, test_data, meta_data)
        print(f'{dataloader_train.dataset_name}')
        state = env.reset()
        mask = env.get_available_actions_mask()

        done = False
        episode_reward = 0
        episode_loss = []

        print('[', end='')
        while not done:
            action = agent.get_action(state, mask)
            print(f'{action}', end=', ')

            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

            loss = agent.fit(state, action, reward, done, mask, next_state)
            state = next_state
            mask = env.get_available_actions_mask()

            episode_loss.append(loss)

        print(']', end='')

        print(f'\n{info["pipeline"]}, metric {info["metric"]}, reward {episode_reward}')
        print(f'{info["validation_rules"]}')

        total_rewards.append(episode_reward)
        metric_value = info['metric'] if info['metric'] else 100000
        total_metrics.append(metric_value)

        print('-- Update tensorboard --')
        tb_writer.add_scalar('loss', np.mean(episode_loss), episode)

        tb_writer.add_scalar('reward', episode_reward, episode)
        tb_writer.add_scalar('metric', metric_value, episode)

        if episode % period == 0:
            message = f'Mean reward per {period} episode'
            tb_writer.add_scalar(message, np.mean(total_rewards), episode)

            message = f'Mean reward per last {period} episode'
            tb_writer.add_scalar(message, np.mean(total_rewards[-period:]), episode)

            message = f'Mean metric per {period} episode'
            tb_writer.add_scalar(message, np.mean(total_metrics), episode)

            message = f'Mean metric per last {period} episode'
            tb_writer.add_scalar(message, np.mean(total_metrics[-period:]), episode)

        print(f'-- Finishing {episode} episode --\n')

    # -- Saving Agent ---
    name = f'{env.metadata["name"]}_{number_of_nodes_in_pipeline}_{state_dim}_{agent.metadata["name"]}' \
           f'_{agent.hidden_dim}_{n_episodes}'
    path = f'{log_dir}/weight'

    if not os.path.exists(path):
        os.makedirs(path)

    agent.save(f'{path}/{name}')

    print('-- Validation Starts --\n')
    for test_dataset in test_list:
        train_data, test_data, meta_data = dataloader_test.get_data(test_dataset)

        env.load_data(train_data, test_data, meta_data)
        print(f'{dataloader_train.dataset_name}')
        state = env.reset()
        mask = env.get_available_actions_mask()

        done = False
        episode_reward = 0
        episode_loss = []

        print('[', end='')
        while not done:
            with torch.no_grad():
                action = agent.get_action(state, mask)

            print(f'{action}', end=', ')

            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

            state = next_state
            mask = env.get_available_actions_mask()

            episode_loss.append(loss)

        print(']', end='')

        print(f'\n{info["pipeline"]}, metric {info["metric"]}, reward {episode_reward}')
        print(f'{info["validation_rules"]}')

        total_rewards.append(episode_reward)
        metric_value = info['metric'] if info['metric'] else 100000
        total_metrics.append(metric_value)


if __name__ == '__main__':
    params = [
        [512, 0.75, 1, 1e-05, 0.01],
        # [1024, 0.25, 1, 0.01, 0.001],
        # [512, 0.5, 1, 0.001, 0.001],
        # [1024, 0.25, 1, 0.001, 0.01],
        # [1024, 0.95, 1, 0.001, 0.001],
        # [512, 0.25, 1, 0.0001, 0.001],
        # [512, 0.75, 1, 0.0001, 0.01],
        # [512, 0.95, 1, 0.0001, 0.001],
        # [1024, 0.1, 1, 0.01, 0.01],
        # [1024, 0.75, 1, 1e-05, 0.0001],
        # [1024, 0.9, 1, 1e-05, 0.0001],
        # [512, 0.25, 1, 0.01, 0.0001],
        # [1024, 0.9, 1, 0.001, 0.0001],
        # [512, 0.95, 1, 1e-05, 0.0001],
        # [512, 0.25, 1, 0.001, 0.0001],
        # [512, 0.5, 1, 0.01, 0.001],
        # [512, 0.25, 1, 0.0001, 0.01],
        # [1024, 0.05, 1, 0.01, 0.001],
        # [1024, 0.75, 1, 0.01, 0.001],
        # [1024, 0.9, 1, 0.0001, 0.0001],
    ]

    for h, g, e, e_m, e_d in params:
        run_experiment(2000, 8, h, g, e, e_m, e_d)
