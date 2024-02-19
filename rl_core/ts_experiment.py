import os

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from meta_automl.utils import project_root
from rl_core.agent.ppo import PPO
from rl_core.dataloader import DataLoader_TS
from rl_core.environments.time_series import TimeSeriesPipelineEnvironment


def define_data_for_experiment():
    data_folder_path = os.path.join(str(project_root()), 'MetaFEDOT\\data\\knowledge_base_time_series_0\\datasets\\')
    dataset_names = [name for name in os.listdir(data_folder_path)]

    train, test = train_test_split(dataset_names, test_size=3)

    train_datasets = {}
    for dataset in train:
        train_datasets[dataset] = os.path.join(data_folder_path, f'{dataset}/data.csv')

    test_datasets = {}
    for dataset in test:
        test_datasets[dataset] = os.path.join(data_folder_path, f'{dataset}/data.csv')

    path_to_meta_data = os.path.join(str(project_root()),
                                     'MetaFEDOT\\data\\knowledge_base_time_series_0\\meta_features_ts.csv')

    dataloader = DataLoader_TS(train_datasets, path_to_meta_data=path_to_meta_data)

    return dataloader


if __name__ == '__main__':
    number_of_nodes_in_pipeline = 10
    n_episodes = 5000

    dataloader = define_data_for_experiment()
    env = TimeSeriesPipelineEnvironment(render_mode='none', metadata_dim=0)
    state_dim, action_dim = env.state_dim, env.action_dim                               # TODO: Fixed shape for agent
    hidden_dim = 2048
    agent = PPO(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)

    log_dir = f'{project_root()}/MetaFEDOT/rl_core/agent/tensorboard_logs'
    tb_writer = SummaryWriter(log_dir=log_dir)

    # -- Starting experiment --

    total_rewards = []
    total_metrics = []

    for episode in range(1, n_episodes + 1):
        print(f'-- Starting {episode} episode --')
        train_data, test_data, meta_data = dataloader.get_data()
        env.load_data(train_data, test_data, meta_data)
        print(f'{dataloader.dataset_name}')
        state = env.reset()
        mask = env.get_available_actions()

        done = False
        episode_reward = 0
        episode_metric = 0

        print('[', end='')
        while not done:
            action = agent.act(state, mask)
            print(f'{action}', end=', ')

            next_state, reward, terminated, truncated, info = env.step(action)
            mask = env.get_available_actions()
            episode_reward += reward
            done = terminated or truncated

            agent.append_to_buffer(state, action, reward, done, mask)
            state = next_state

        print(']', end='')

        # if info['pipeline'].nodes != []:
        #     info['pipeline'].show()

        print(f'\n{info["pipeline"]}, metric {info["metric"]}, reward {episode_reward}')
        print(f'{info["validation_rules"]}')
        print(f'-- Finishing {episode} episode --')
        print('')
        total_rewards.append(episode_reward)
        metric_value = info['metric'] if info['metric'] else 100000
        total_metrics.append(metric_value)

        loss_1, loss_2 = agent.update()

        tb_writer.add_scalar('loss_1', loss_1, episode)
        tb_writer.add_scalar('loss_2', loss_2, episode)

        tb_writer.add_scalar('reward', episode_reward, episode)
        tb_writer.add_scalar('metric', metric_value, episode)

        if episode % 50 == 0:
            agent.clear_buffer()

        if episode % 5 == 0:
            message = f'Average return reward per 5 episode'
            tb_writer.add_scalar(message, np.mean(total_rewards), episode)

            message = f'Average pipelines metric per 5 episode'
            tb_writer.add_scalar(message, np.mean(total_metrics), episode)

    # -- Saving Agent ---
    name = f'{env.metadata["name"]}_{state_dim}_{agent.metadata["name"]}_{hidden_dim}_{n_episodes}'
    path = f'{project_root()}/MetaFEDOT/rl_core/agent/pretrained/{name}'
    agent.save(path)
