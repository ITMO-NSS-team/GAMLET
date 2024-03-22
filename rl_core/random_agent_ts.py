import datetime

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from meta_automl.utils import project_root
from rl_core.agent.ppo import PPO
from rl_core.environments.time_series import TimeSeriesPipelineEnvironment
from rl_core.utils import define_data_for_experiment


if __name__ == '__main__':
    for number_of_nodes_in_pipeline in range(2, 11):
        n_episodes = 2000

        dataloader, _, _ = define_data_for_experiment()
        env = TimeSeriesPipelineEnvironment(render_mode='none', metadata_dim=126)
        state_dim, action_dim = env.state_dim, env.action_dim

        # time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f'{project_root()}/MetaFEDOT/rl_core/agent/tensorboard_logs/random_action/{number_of_nodes_in_pipeline}/random_agent'
        tb_writer = SummaryWriter(log_dir=log_dir)

        # -- Starting experiment --

        total_rewards = []
        total_metrics = []

        period = 20
        period_of_cleaning = 15
        period_of_heatmap = 100

        for episode in range(1, n_episodes + 1):
            print(f'-- Starting {episode} episode --')

            train_data, test_data, meta_data = dataloader.get_data()
            env.load_data(train_data, test_data, meta_data)
            print(f'{dataloader.dataset_name}')
            env.reset()

            done = False
            episode_reward = 0
            episode_metric = 0

            print('[', end='')
            while not done:
                available_actions = env.get_available_actions()
                action = np.random.choice(available_actions)

                print(f'{action}', end=', ')

                _, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            print(']', end='')

            print(f'\n{info["pipeline"]}, metric {info["metric"]}, reward {episode_reward}')
            print(f'{info["validation_rules"]}')

            total_rewards.append(episode_reward)
            metric_value = info['metric']
            total_metrics.append(metric_value)

            print('-- Update tensorboard --')
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

        print(f'Minimum reward {min(total_rewards)} -- Maximum reward {max(total_rewards)}')