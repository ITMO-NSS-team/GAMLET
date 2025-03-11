import os.path

import numpy as np
import wandb
from torch.utils.tensorboard import SummaryWriter

from gamlet.utils import project_root
from rl_core.agent.dqn import DQN
from rl_core.environments.time_series import TimeSeriesPipelineEnvironment
from rl_core.utils import define_data_for_experiment, OFFLINE_TRAJECTORIES


def run_experiment(n_episodes, number_of_nodes_in_pipeline, hidden_dim, gamma, eps_min, eps_decrease):
    wandb.init(
        project='DQN_TS_ENV',
        config={
            'n_episodes': n_episodes,
            'architecture': 'DQN',
            'environment': 'TimeSeries',
            'max_number_of_nodes_in_pipeline': number_of_nodes_in_pipeline,
            'metadata_dim': 126,
            'hidden_dim': hidden_dim,
            'gamma': gamma,
            'eps_min': eps_min,
            'eps_decrease': eps_decrease
        }
    )

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

    # time = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

    exp_name = f'h[{hidden_dim}]_g[{gamma}]_emin[{eps_min}]_edec[{eps_decrease}]'
    log_dir = f'{project_root()}/MetaFEDOT/rl_core/agent/tensorboard_logs/dqn/{number_of_nodes_in_pipeline}/{exp_name}'
    tb_writer = SummaryWriter(log_dir=log_dir)
    agent.create_log_report(log_dir)
    os.mkdir(f'{log_dir}/weights')

    # -- INIT MEMORY --
    for repeat in range(1):
        for trajectory in OFFLINE_TRAJECTORIES:
            train_data, test_data, meta_data = dataloader_train.get_data()
            env.load_data(train_data, test_data, meta_data)
            state = env.reset()
            mask = env.get_available_actions_mask()

            for action in trajectory:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.memory.append([state, action, reward, int(done), mask, next_state])
                state = next_state
                mask = env.get_available_actions_mask()

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

        wandb.log({'loss': np.mean(episode_loss), 'reward': episode_reward, 'metric': metric_value})

        if episode % period == 0:
            message = f'Mean reward per {period} episode'
            tb_writer.add_scalar(message, np.mean(total_rewards), episode)

            message = f'Mean reward per last {period} episode'
            tb_writer.add_scalar(message, np.mean(total_rewards[-period:]), episode)

            message = f'Mean metric per {period} episode'
            tb_writer.add_scalar(message, np.mean(total_metrics), episode)

            message = f'Mean metric per last {period} episode'
            tb_writer.add_scalar(message, np.mean(total_metrics[-period:]), episode)

            wandb.log({
                'm_reward': np.mean(total_rewards),
                f'{period}_reward': np.mean(total_rewards[-period:]),
                'm_metric': np.mean(total_metrics),
                f'{period}_metric': np.mean(total_metrics[-period:])
            })

        print(f'-- Finishing {episode} episode --\n')

    # -- Saving Agent ---
    name = f'{agent.metadata["name"]}_{agent.hidden_dim}_{n_episodes}'
    path = f'{log_dir}/weights/{name}'
    agent.save(path)

    wandb.finish()


if __name__ == '__main__':
    params = {
        'max_number_of_nodes': [8],
        'hidden_dim': [512, 1024],
        'gamma': [0.75, 0.9, 0.95, 0.99],
        'eps_min': [1e-2, 1e-3, 1e-4, 1e-5],
        'eps_decrease': [1e-2, 1e-3, 1e-4, 1e-5]
    }

    run_experiment(2000, 10, 512, 0.75, 1e-5, 1e-4)
