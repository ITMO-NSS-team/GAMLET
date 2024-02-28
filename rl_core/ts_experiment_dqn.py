import datetime

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from meta_automl.utils import project_root
from rl_core.agent.dqn import DQN
from rl_core.agent.ppo import PPO
from rl_core.environments.time_series import TimeSeriesPipelineEnvironment
from rl_core.utils import define_data_for_experiment


if __name__ == '__main__':
    number_of_nodes_in_pipeline = 3
    n_episodes = 2000

    dataloader, train_list, test_list = define_data_for_experiment()
    env = TimeSeriesPipelineEnvironment(max_number_of_nodes=number_of_nodes_in_pipeline, render_mode='none', metadata_dim=126)
    state_dim, action_dim = env.state_dim, env.action_dim
    agent = DQN(state_dim=state_dim, action_dim=action_dim)

    time = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    log_dir = f'{project_root()}/MetaFEDOT/rl_core/agent/tensorboard_logs/dqn/{number_of_nodes_in_pipeline}/{time}'
    tb_writer = SummaryWriter(log_dir=log_dir)
    agent.create_log_report(log_dir)

    # -- Starting experiment --

    total_rewards = []
    total_metrics = []

    for episode in range(1, n_episodes + 1):
        print(f'-- Starting {episode} episode --')

        train_data, test_data, meta_data = dataloader.get_data()
        env.load_data(train_data, test_data, meta_data)
        print(f'{dataloader.dataset_name}')
        state = env.reset()
        mask = env.get_available_actions_mask()

        done = False
        episode_reward = 0
        episode_metric = 0
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
        tb_writer.add_scalar('reward', episode_reward, episode)
        tb_writer.add_scalar('metric', metric_value, episode)

        period = 20
        if episode % period == 0:
            message = f'Mean reward'
            tb_writer.add_scalar(message, np.mean(total_rewards), episode)

            message = f'Mean reward per last {period} episode'
            tb_writer.add_scalar(message, np.mean(total_rewards[-period:]), episode)

            message = f'Mean metric'
            tb_writer.add_scalar(message, np.mean(total_metrics), episode)

            message = f'Mean metric per last {period} episode'
            tb_writer.add_scalar(message, np.mean(total_metrics[-period:]), episode)

        print(f'-- Finishing {episode} episode --\n')

    # -- Saving Agent ---
    name = f'{env.metadata["name"]}_{state_dim}_{agent.metadata["name"]}_{agent.hidden_dim}_{n_episodes}'
    path = f'{project_root()}/MetaFEDOT/rl_core/agent/pretrained/{name}'
    agent.save(path)
