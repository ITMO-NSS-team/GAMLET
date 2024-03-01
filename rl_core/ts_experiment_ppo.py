import datetime
import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from meta_automl.utils import project_root
from rl_core.agent.ppo import PPO
from rl_core.environments.time_series import TimeSeriesPipelineEnvironment
from rl_core.utils import define_data_for_experiment

import seaborn as sns
import matplotlib.pyplot as plt

plt.ioff()
sns.set(font_scale=0.5)


if __name__ == '__main__':
    number_of_nodes_in_pipeline = 3
    n_episodes = 1000

    dataloader, train_list, test_list = define_data_for_experiment()
    env = TimeSeriesPipelineEnvironment(max_number_of_nodes=number_of_nodes_in_pipeline, render_mode='none', metadata_dim=126)
    state_dim, action_dim = env.state_dim, env.action_dim
    agent = PPO(state_dim=state_dim, action_dim=action_dim, device='cuda')

    time = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    log_dir = f'{project_root()}/MetaFEDOT/rl_core/agent/tensorboard_logs/ppo/{number_of_nodes_in_pipeline}/{time}'
    tb_writer = SummaryWriter(log_dir=log_dir)
    agent.create_log_report(log_dir)
    os.mkdir(f'{log_dir}/probs_heatmap')

    # -- Starting experiment --

    total_rewards = []
    total_metrics = []

    period = 20
    period_of_cleaning = 15
    period_of_heatmap = 100

    for episode in range(1, n_episodes + 1):
        print(f'-- Starting {episode} episode --')

        train_data, test_data, meta_data = dataloader.get_data('M4_Q20451')
        env.load_data(train_data, test_data, meta_data)
        print(f'{dataloader.dataset_name}')
        state = env.reset()
        mask = env.get_available_actions_mask()

        done = False
        episode_reward = 0
        episode_metric = 0
        probs_matrix = np.zeros((env.action_dim, env.max_number_of_actions))
        m_idx = 0

        print('[', end='')
        while not done:
            action, probs = agent.act(state, mask, mode='probs')
            # action = agent.act(state, mask)
            print(f'{action}', end=', ')

            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

            agent.append_to_buffer(state, action, reward, done, mask)
            state = next_state
            mask = env.get_available_actions_mask()
            probs_matrix[:, m_idx] = probs.detach().cpu().numpy()
            m_idx += 1

        print(']', end='')

        print(f'\n{info["pipeline"]}, metric {info["metric"]}, reward {episode_reward}')
        print(f'{info["validation_rules"]}')
        print(agent.buffer.get_size())

        total_rewards.append(episode_reward)
        metric_value = info['metric'] if info['metric'] else 100000
        total_metrics.append(metric_value)

        print('-- Starting agent update --')
        loss_1, loss_2, kld = agent.update()
        print(f'Q loss {loss_1} - V loss {loss_2} - KL Div {kld}')

        if episode % period_of_cleaning == 0:
            agent.clear_buffer()

        print('-- Update tensorboard --')
        tb_writer.add_scalar('loss_1', loss_1, episode)
        tb_writer.add_scalar('loss_2', loss_2, episode)
        tb_writer.add_scalar('kld', kld, episode)

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

        labels = (
            np.asarray(
                [np.around(value, 3) for value in probs_matrix.flatten()]
            )
        ).reshape(probs_matrix.shape)

        actions_labels = [str(env.get_action_code(action)) for action in range(len(probs_matrix[:, 0]))]

        fig = sns.heatmap(
            probs_matrix,
            annot=labels,
            yticklabels=actions_labels,
            linewidths=.05,
            vmin=0.,
            vmax=1.,
            linecolor='gray',
            cbar=False,
            annot_kws={"fontsize": 8}
        )
        plt.xlabel('Step')
        plt.ylabel('Action')
        plt.savefig(f'{log_dir}/probs_heatmap/{episode}.png', format='png')

        if episode % period_of_heatmap == 0:
            plt.show()

        plt.close()

        print(f'-- Finishing {episode} episode --\n')

    # -- Saving Agent ---
    name = f'{env.metadata["name"]}_{state_dim}_{agent.metadata["name"]}_{agent.hidden_dim}_{n_episodes}'
    path = f'{project_root()}/MetaFEDOT/rl_core/agent/pretrained/{name}'
    agent.save(path)
