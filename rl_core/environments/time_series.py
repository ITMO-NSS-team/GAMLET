import os
from itertools import product, combinations
from typing import Optional

import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import pygame
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from gymnasium import spaces

from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from meta_automl.utils import project_root
from rl_core.dataloader import DataLoader, DataLoader_TS

PLOT_PRED = False


class TimeSeriesPipelineEnvironment(gym.Env):
    """

    """
    metadata = {'name': 'time_series_env', 'render_modes': ['none', 'pipeline_plot']}

    def __init__(self, primitives: list[str] = None, max_number_of_nodes: int = 10, max_timestamp: int = 20,
                 metadata_dim=None, render_mode: str = None):
        self.max_number_of_nodes = max_number_of_nodes
        self.primitives = primitives if primitives else self._get_default_primitives()
        self._models = OperationTypesRepository().suitable_operation(task_type=TaskTypesEnum.ts_forecasting)
        self.number_of_primitives = len(self.primitives)

        ## -- OBSERVATION --
        # TODO: Observation space. Can be removed
        # self.observation_space = spaces.Dict(
        #     {
        #         "pipeline_structure": spaces.Graph(
        #             node_space=spaces.Box(low=0, high=self.max_number_of_nodes, shape=(1,)),
        #             edge_space=spaces.Box(
        #                 low=0, high=self.max_number_of_nodes,
        #                 shape=(self.max_number_of_nodes, self.max_number_of_nodes)
        #             ),
        #         )
        #     }
        # )

        ## -- ACTIONS --
        actions_dim = 0

        # Special actions
        self._special_action = {0: 'End of Pipeline'}

        actions_dim += len(self._special_action)

        # Actions for choosing primitive
        self._action_to_add_node = {}

        for i in range(actions_dim, self.number_of_primitives + 1):
            self._action_to_add_node[i] = self.primitives[i - actions_dim]

        actions_dim += len(self._action_to_add_node)

        # Actions for connecting nodes
        self._action_to_connecting = {}
        pairs = [item for item in product(range(0, self.max_number_of_nodes), repeat=2) if item[0] != item[1]]

        for i in range(actions_dim, actions_dim + len(pairs)):
            self._action_to_connecting[i] = pairs[i - actions_dim]

        actions_dim += len(self._action_to_connecting)
        self.action_dim = actions_dim
        self.action_space = spaces.Discrete(actions_dim)

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self._pipeline = Pipeline()
        self._nodes = []
        self._nodes_structure = np.zeros((self.max_number_of_nodes,), dtype=int)
        self._edges_structure = np.zeros((self.max_number_of_nodes, self.max_number_of_nodes), dtype=int)
        self._current_position = 0

        self.state_dim = 546  # TODO: Make it automatically

        self._metric = None
        self._is_valid = None

        self._train_data = None
        self._test_data = None
        self._meta_data = None

        self.max_timestamp = max_timestamp  # TODO: Requires to automatize
        self.timestamp = 0

    def _get_obs(self):
        node_structure = np.ravel(self._apply_one_hot_encoding(self._nodes_structure, self.number_of_primitives + 1))
        edge_structure = np.ravel(self._edges_structure)

        graph_structure = np.concatenate((node_structure, edge_structure))
        obs = np.concatenate((graph_structure, self._meta_data))

        return obs

    def _apply_one_hot_encoding(self, vector, num_classes=3):
        return np.squeeze(np.eye(num_classes)[vector.reshape(-1)])

    def _get_info(self):
        return {
            'pipeline': self._pipeline,
            'meta_data': self._meta_data,
            'nodes_structure': self._nodes_structure,
            'edge_structure': self._edges_structure,
            'metric': self._metric,
            'number_of_nodes': len(self._nodes),
            'is_valid': self._is_valid,
        }

    def get_available_actions(self):
        return print(self._special_action, self._action_to_add_node, self._action_to_connecting)

    def get_action_code(self, action):
        if action in self._special_action.keys():
            return self._special_action[action]

        elif action in self._action_to_add_node.keys():
            return self._action_to_add_node[action]

        elif action in self._action_to_connecting.keys():
            return self._action_to_connecting[action]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._pipeline = Pipeline()
        self._nodes = []
        self._nodes_structure = np.zeros((self.max_number_of_nodes,), dtype=int)
        self._edges_structure = np.zeros((self.max_number_of_nodes, self.max_number_of_nodes), dtype=int)
        self._current_position = 0
        self._metric = None
        self.timestamp = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation

    def step(self, action: int, mode: str = 'train'):
        assert action in self.action_space

        if action in self._special_action.keys():
            if len(self._nodes) == 0:
                terminated = True
                truncated = False
                observation = self._get_obs()
                info = self._get_info()
                reward = -100

            else:
                self._metric = self._run_validating_fitting_and_evaluating()

                terminated = True
                truncated = False
                observation = self._get_obs()
                info = self._get_info()
                reward = self._metric

        else:
            if action in self._action_to_add_node.keys():
                reward = self._apply_action_to_add_node(action)

            elif action in self._action_to_connecting.keys():
                reward = self._apply_action_to_connecting(action)

            terminated = False

            if self.timestamp >= self.max_timestamp:
                truncated = True
                reward += -100

                if not self._is_there_are_connections_in_pipeline():
                    reward += -50

            else:
                truncated = False

            self.timestamp += 1

            observation = self._get_obs()
            info = self._get_info()

            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'pipeline_plot' and len(self._nodes) > 0:
            self._pipeline.show()

    @staticmethod
    def _get_default_primitives():
        return OperationTypesRepository('all').suitable_operation(task_type=TaskTypesEnum.ts_forecasting)

    def _is_possible_to_add_new_node(self):
        return len(self._nodes) < self.max_number_of_nodes

    def _is_nodes_exist(self, node_from, node_to):
        node_from_exist = self._nodes_structure[node_from] != 0
        node_to_exist = self._nodes_structure[node_to] != 0

        return node_from_exist and node_to_exist

    def _run_validating_fitting_and_evaluating(self):
        try:
            self._pipeline.fit(self._train_data)

            y_pred = self._pipeline.predict(self._test_data).predict
            y_true = self._test_data.target

            if PLOT_PRED:
                plt.plot(range(0, len(self._train_data.target)), self._train_data.target)
                plt.plot(range(len(self._train_data.target), len(self._train_data.target) + len(y_pred)), y_pred)
                plt.plot(range(len(self._train_data.target), len(self._train_data.target) + len(y_true)), y_true)
                plt.show()

            metric = mean_absolute_error(y_true, y_pred)

            self._is_valid = True

        except:
            metric = 10000
            self._is_valid = False

        reward = self.get_reward_by_metric(metric)

        return reward

    def _apply_action_to_add_node(self, action):
        if self._is_possible_to_add_new_node():
            primitive = self._action_to_add_node[action]
            self._nodes.append(PipelineNode(primitive))
            self._nodes_structure[self._current_position] = action
            self._current_position += 1
            self._pipeline.add_node(self._nodes[-1])

            if primitive in self._models:
                reward = -0.5

            else:
                reward = -1.0

        else:
            # Mistake
            reward = -10.0

        return reward

    def _apply_action_to_connecting(self, action):
        node_from, node_to = self._action_to_connecting[action]

        if self._is_nodes_exist(node_from, node_to):
            self._pipeline.connect_nodes(node_parent=self._nodes[node_from], node_child=self._nodes[node_to])
            self._edges_structure[node_from][node_to] = 1
            reward = -1.5
        else:
            # Mistake
            reward = -10.0

        return reward

    def load_data(
            self,
            train_data: Optional[InputData],
            test_data: Optional[InputData],
            meta_data: Optional[np.ndarray]
    ):
        self._train_data = train_data
        self._test_data = test_data
        self._meta_data = meta_data

        return self

    @staticmethod
    def get_reward_by_metric(m, m_min=-10000, m_max=0):
        return ((-1 * m - m_min) / (m_max - m_min)) * 100

    def _is_there_are_connections_in_pipeline(self):
        matrix_with_no_connection = np.zeros((self.max_number_of_nodes, self.max_number_of_nodes), dtype=int)

        return np.array_equal(self._edges_structure, matrix_with_no_connection)


if __name__ == '__main__':
    data_folder_path = os.path.join(str(project_root()), 'MetaFEDOT\\data\\knowledge_base_time_series_0\\datasets\\')
    dataset_names = [name for name in os.listdir(data_folder_path)]

    train, test = train_test_split(dataset_names, test_size=3, random_state=42)

    train_datasets = {}
    for dataset in train:
        train_datasets[dataset] = os.path.join(data_folder_path, f'{dataset}/data.csv')

    path_to_meta_data = os.path.join(str(project_root()),
                                     'MetaFEDOT\\data\\knowledge_base_time_series_0\\meta_features_ts.csv')
    dataloader = DataLoader_TS(train_datasets, path_to_meta_data=path_to_meta_data)
    train_data, test_data, meta_data = dataloader.get_data(dataset_name='M4_W343')

    env = TimeSeriesPipelineEnvironment(render_mode='pipeline_plot', metadata_dim=125)
    env.load_data(train_data, test_data, meta_data)
    terminated = False

    total_reward = 0

    state = env.reset()

    while not terminated:
        print(env.get_available_actions())
        action = int(input())
        new_state, reward, terminated, truncated, info = env.step(action)
        print(f'{reward}, {info}')
        total_reward += reward
