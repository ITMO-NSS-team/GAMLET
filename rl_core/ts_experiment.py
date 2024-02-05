import os

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from meta_automl.utils import project_root
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


def define_agent():


if __name__ == '__main__':
    number_of_nodes_in_pipeline = 10
    n_episodes = 1000

    dataloader = define_data_for_experiment()
    env = TimeSeriesPipelineEnvironment(render_mode='pipeline_plot')
    state_dim, action_dim = env.state_dim, env.action_dim               # TODO: Fixed shape for agent
    hidden_dim = 1024
    agent = define_agent()

    tb_writer =

    # -- Starting experiment --

    total_rewards = []
    total_metrics = []

    for episode in tqdm(range(1, n_episodes + 1)):
        train_data, test_data, predict_input, meta_data = dataloader.get_data()
        env.load_data(train_data, test_data, predict_input)
        state = env.reset()

        done = False
        episode_reward = 0
        episode_metric = 0

        while not done:
            action = agent.act(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

            state = next_state

        loss_1, loss_2 = agent.update()

        tb_writer.add_scalar('loss 1', loss_1, episode)
        tb_writer.add_scalar('loss_2', loss_2, episode)

        if episode % 5 == 0:
            message = f'Average return reward per 5 episode'
            tb_writer.add_scalar(message, avg_return, episode)

            message = f'Average pipelines metric per 5}'
            tb_writer.add_scalar(message, avg_metric, episode)

    # -- Saving Agent ---
    name = f'{env.metadata["name"]}_{state_dim}_{agent.metadata["name"]}_{hidden_dim}_{n_episodes}'
    path = f'{project_root()}/MetaFEDOT/rl_core/agent/pretrained/{name}'
    agent.save(path)