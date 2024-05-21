import os
from itertools import product
from typing import Optional

import gymnasium as gym
import numpy as np
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.verification import verify_pipeline
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum
from gymnasium import spaces
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from gamlet.utils import project_root
from rl_core.dataloader import TimeSeriesDataLoader

PLOT_PRED = False


class TimeSeriesPipelineEnvironment(gym.Env):
    """ Environment for constructing time series pipelines

        Args:
             primitives - list of primitives which can be used for pipeline construction;
             max_number_of_nodes - maximum number of nodes in pipeline;
             using_number_of_nodes - number of nodes could be used. This params was used for schedule experiments for
                saving dimension of state vector;
             metadata_dim - dimension of metadata vector;
             is_use_dataloader - param for using dataloader instead loading data by method.
             output_mode - pending output type: dict or np.ndarray. Default np.ndarray;
             render_mode - ...
    """
    metadata = {'name': 'time_series_env', 'render_modes': ['none', 'pipeline_plot', 'pipeline_and_predict_plot']}

    def __init__(self, primitives: list[str] = None, max_number_of_nodes: int = 10, using_number_of_nodes: int = 10,
                 metadata_dim=None, is_use_dataloader=False, output_mode: type = np.ndarray, render_mode: str = None):
        self.max_number_of_nodes = max_number_of_nodes
        self.using_number_of_nodes = using_number_of_nodes
        self.primitives = primitives if primitives else self._get_default_primitives()
        self._models = OperationTypesRepository().suitable_operation(task_type=TaskTypesEnum.ts_forecasting)
        self.number_of_primitives = len(self.primitives)

        # -- ACTIONS --
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

        self.max_number_of_actions = self._get_maximum_number_of_actions_in_environment(self.max_number_of_nodes)

        # -- STATE --
        self._pipeline = Pipeline()
        self._nodes = []
        self._nodes_structure = np.zeros((self.max_number_of_nodes,), dtype=int)
        self._edges_structure = np.zeros((self.max_number_of_nodes, self.max_number_of_nodes), dtype=int)
        self._current_position = 0
        self._available_actions = None
        self._rules = {}

        self.metadata_dim = metadata_dim
        self.state_dim = self._get_state_dim()

        self._metric = None
        self.y_pred = None
        self.y_true = None

        self.is_use_dataloader = is_use_dataloader
        self.output_mode = output_mode
        self._dataloader = None
        self._train_data = None
        self._test_data = None
        self._meta_data = None

        if self.metadata_dim:
            self.observation_space = spaces.Dict(
                {
                    'meta': spaces.Box(low=0, high=1, shape=(self.metadata_dim,)),
                    'nodes': spaces.Box(
                        low=0, high=1,
                        shape=(self._nodes_structure.shape[0], len(self.primitives) + 1), dtype=np.int8),
                    'edges': spaces.Box(low=0, high=1, shape=self._edges_structure.shape, dtype=np.int8),
                }
            )

        else:
            self.observation_space = spaces.Dict(
                {
                    'nodes': spaces.Box(
                        low=0, high=1,
                        shape=(self._nodes_structure.shape[0], len(self.primitives) + 1), dtype=np.int8),
                    'edges': spaces.Box(low=0, high=1, shape=self._edges_structure.shape, dtype=np.int8),
                }
            )

        # -- REWARD --
        self.env_step = 0
        self._total_reward = 0

        self.max_reward = 1
        self.min_reward = -1

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

    def _get_obs(self) -> Optional[np.ndarray, dict]:
        """ Returns current environment's observation """
        if self.output_mode == np.ndarray:
            graph_structure = self._get_graph_structure()

            if self._meta_data is not None:
                obs = np.concatenate((graph_structure, self._meta_data))
            else:
                obs = graph_structure

        elif self.output_mode == dict:
            node_structure = self._apply_one_hot_encoding(self._nodes_structure, self.number_of_primitives + 1)
            edge_structure = self._edges_structure

            # For sb3 models
            obs = {'nodes': node_structure, 'edges': edge_structure}

            if self._meta_data is not None:
                obs['meta'] = self._meta_data
        else:
            raise TypeError(f'Invalid type for state required [nd.ndarray, dict], but state is {self.output_mode}')

        return obs

    def _get_info(self):
        """ Return additional information from the environment """
        return {
            'pipeline': self._pipeline,
            'meta_data': self._meta_data,
            'nodes_structure': self._nodes_structure,
            'edge_structure': self._edges_structure,
            'reward': self._total_reward,
            'metric': self._metric,
            'number_of_nodes': len(self._nodes),
            'validation_rules': self._rules,
            'y_pred': self.y_pred,
            'y_true': self.y_true,
        }

    def valid_action_mask(self):
        """ Used for masked agent from sb3-contrib """
        return self.get_available_actions_mask()

    def get_available_actions_mask(self) -> np.ndarray:
        """ Returns available actions in the current environment's state in binary mask """
        self._available_actions_mask = np.zeros((self.action_dim,), dtype=bool)

        # Special actions available always except the first action
        if self.env_step != 0:
            for action_idx in self._special_action.keys():
                self._available_actions_mask[action_idx] = True

        if self.env_step < self.using_number_of_nodes:
            # Adding nodes is available before the free space runs out
            for action_idx in self._action_to_add_node.keys():
                if self._is_possible_to_add_new_node():
                    self._available_actions_mask[action_idx] = True

        if self.env_step != self._get_maximum_number_of_actions_in_environment(self.using_number_of_nodes):
            # Adding connections between nodes is available:
            # If these nodes exist and are no connected
            for action_idx, nodes in self._action_to_connecting.items():
                is_nodes_exists = self._is_nodes_exist(node_from=nodes[0], node_to=nodes[1])
                is_nodes_connected = self._is_nodes_connected(node_a=nodes[0], node_b=nodes[1])

                self._available_actions_mask[action_idx] = is_nodes_exists and not is_nodes_connected

        return self._available_actions_mask

    def _get_available_actions(self) -> dict:
        all_actions = {}

        for d in (self._special_action, self._action_to_add_node, self._action_to_connecting):
            all_actions.update(d)

        mask = self.get_available_actions_mask()

        all_actions = {}

        for d in (self._special_action, self._action_to_add_node, self._action_to_connecting):
            all_actions.update(d)

        out = {}
        for (k, v), m in zip(all_actions.items(), mask):
            if m:
                out[k] = v

        return out

    def get_available_actions(self) -> np.ndarray:
        return np.fromiter(self._get_available_actions().keys(), dtype=int)

    def print_available_actions(self):
        """ Show all possible actions """
        return print(self._get_available_actions())

    def get_action_code(self, action: int) -> str:
        """ Returns primitive name by action """
        if action in self._special_action.keys():
            return self._special_action[action]

        elif action in self._action_to_add_node.keys():
            return self._action_to_add_node[action]

        elif action in self._action_to_connecting.keys():
            return self._action_to_connecting[action]

    def reset(self, seed: int = None, options: list = None, **kwargs) -> (np.ndarray, dict):
        """ Reset environment to initial state """
        super().reset(seed=seed)

        self._pipeline = Pipeline()
        self._nodes = []
        self._nodes_structure = np.zeros((self.max_number_of_nodes,), dtype=int)
        self._edges_structure = np.zeros((self.max_number_of_nodes, self.max_number_of_nodes), dtype=int)
        self._current_position = 0
        self._available_actions = np.zeros((self.action_dim,), dtype=bool)
        self._rules = {}
        self._metric = 10000
        self.env_step = 0
        self._total_reward = 0

        self.y_pred = None
        self.y_true = None

        if self.is_use_dataloader and self._dataloader:
            self._train_data, self._test_data, self._meta_data = self._dataloader.get_data()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: int, mode: str = 'train') -> (np.ndarray, int, bool, bool, dict):
        """ Apply action to environment

            Returns:
                - observation
                - reward
                - terminated
                - truncated
                - info
        """

        assert action in self.action_space

        # Checks if action is not valid
        if action not in self._get_available_actions().keys():
            terminated = False
            truncated = False
            reward = -0.01

            self.env_step += 1
            observation = self._get_obs()
            info = self._get_info()

        # Checks if action is from special actions (e.g. eof - End of Pipeline)
        elif action in self._special_action.keys():
            terminated, truncated, reward = self._apply_eop_action()

            observation = self._get_obs()
            info = self._get_info()

        else:
            terminated = False
            truncated = False
            reward = 0

            # Checks if action is for adding node
            if action in self._action_to_add_node.keys():
                self._apply_action_to_add_node(action)
                reward += 0.005

            # Checks if action is for connecting nodes
            elif action in self._action_to_connecting.keys():
                self._apply_action_to_connecting(action)
                reward += 0.01

            if self._pipeline.depth == -1:
                reward -= 0.8
                truncated = True

            self.env_step += 1
            observation = self._get_obs()
            info = self._get_info()

        self._total_reward += reward

        self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'pipeline_plot' and len(self._nodes) > 0:
            self._pipeline.show()

    def _is_possible_to_add_new_node(self):
        return len(self._nodes) < len(self._nodes_structure)

    def _is_nodes_exist(self, node_from, node_to):
        node_from_exist = self._nodes_structure[node_from] != 0
        node_to_exist = self._nodes_structure[node_to] != 0

        return node_from_exist and node_to_exist

    def _is_nodes_connected(self, node_a, node_b):
        connection_ab = self._edges_structure[node_a][node_b]
        connection_ba = self._edges_structure[node_b][node_a]

        return connection_ab or connection_ba

    def _return_detached_nodes(self) -> list:
        """ Detached node is a node without any connection in pipeline.
            The function checks the pipeline for finding it,
            returns a logical answer and the number of such nodes.
        """
        detached_nodes = []

        for i, _ in enumerate(self._nodes):
            if not (self._edges_structure[:, i].any() or self._edges_structure[i, :].any()):
                detached_nodes.append(i)

        return detached_nodes

    def _is_pipeline_without_connections(self) -> bool:
        """ Returns `True` if the Pipeline does not contain edges between nodes and `False` in the other case """
        empty_connection_pipeline = np.zeros((self.max_number_of_nodes, self.max_number_of_nodes), dtype=int)

        return np.array_equal(self._edges_structure, empty_connection_pipeline)

    def _is_pipeline_single_model(self) -> bool:
        """ Returns `True` if the Pipeline is a single node structure """
        return len(self._nodes) == 1

    def _is_node_operation(self, node):
        return node.name not in self._models

    def _return_inputs_node(self) -> list:
        """ Input node is a node from which the Pipeline starts. There may be several input nodes in the pipeline.
            The function returns input nodes.
        """
        input_nodes = []

        for i, _ in enumerate(self._nodes):
            if not self._edges_structure[:, i].any() and self._edges_structure[i, :].any():
                input_nodes.append(i)

        return input_nodes

    def _return_output_node(self) -> list:
        """ Output node is a node where the Pipeline ends. There should always be one output node.
            The function returns list of output nodes.
        """
        output_nodes = []

        for i, _ in enumerate(self._nodes):
            if self._edges_structure[:, i].any() and not self._edges_structure[i, :].any():
                output_nodes.append(i)

        return output_nodes

    def _apply_eop_action(self):
        """ Applying End of Pipeline actions and prepare the Pipeline for fitting and validating """
        terminated = True
        truncated = False
        reward = 0
        self._rules['valid'] = True

        # Single node pipeline
        self._rules['single_node_pipeline'] = self._is_pipeline_single_model()
        if self._rules['single_node_pipeline'] and self._is_node_operation(self._nodes[0]):
            self._rules['valid'] = False
            self._rules['invalid_single_node_pipeline'] = True

            reward -= 0.01

        else:
            self._rules['valid'] = True
            self._rules['invalid_single_node_pipeline'] = False

        # Pipeline with detached nodes
        self._rules['detached_nodes'] = self._return_detached_nodes()
        self._rules['number_of_detached_nodes'] = len(self._rules['detached_nodes'])
        if not self._rules['single_node_pipeline'] and self._rules['number_of_detached_nodes'] > 0:
            self._rules['valid'] = False

            reward -= 0.01 * self._rules['number_of_detached_nodes']

        else:
            self._rules['valid'] = True
            reward += 0.01

        # Inputs nodes in pipeline
        self._rules['input_nodes'] = self._return_inputs_node()
        self._rules['number_of_input_nodes'] = len(self._rules['input_nodes'])
        if not self._rules['single_node_pipeline'] and self._rules['number_of_input_nodes'] < 1:
            self._rules['valid'] = False
            reward -= 0.01

        # Outputs nodes in pipeline
        self._rules['output_nodes'] = self._return_output_node()
        self._rules['number_of_output_nodes'] = len(self._rules['output_nodes'])
        if not self._rules['single_node_pipeline'] and (self._rules['number_of_output_nodes'] != 1):
            self._rules['valid'] = False
            reward -= 0.01

        # Output nodes is not data operation primitive
        elif not self._rules['single_node_pipeline'] and (self._rules['number_of_output_nodes'] == 1):
            node = self._nodes[self._rules['output_nodes'][0]]

            if self._is_node_operation(node):
                self._rules['valid'] = False
                reward -= 0.01

            else:
                self._rules['valid'] = True
                reward += 0.01

        # Checks GOLEM rules (troubles with the craziest pipelines)
        if self._pipeline.depth != -1:
            self._rules['golem_rules'] = verify_pipeline(self._pipeline, task_type=self._train_data.task.task_type,
                                                         raise_on_failure=False)

            if self._rules['golem_rules']:
                self._rules['valid'] = True
                reward += 0.25

            else:
                self._rules['valid'] = False
                reward -= 0.5

        # else:
        #     self._rules['golem_rules'] = False
        #     self._rules['valid'] = False
        #     reward -= 25

        # Otherwise, if it is all fine, environment will try to fit and predict pipeline
        if self._rules['valid']:
            self._run_validating_fitting_and_evaluating()

        if self._rules['valid']:
            if self._metric > 10000:
                self._metric = 10000

            reward += self.get_reward_by_metric(self._metric)
            self._total_reward += reward

        return terminated, truncated, reward

    def _apply_action_to_add_node(self, action: int):
        """ Applying action from _action_to_add_node """
        primitive = self._action_to_add_node[action]
        self._nodes.append(PipelineNode(primitive))
        self._nodes_structure[self._current_position] = action
        self._current_position += 1
        self._pipeline.add_node(self._nodes[-1])

    def _apply_action_to_connecting(self, action: int):
        """ Applying connection between `node_from` with `node_to` """
        node_from, node_to = self._action_to_connecting[action]
        self._pipeline.connect_nodes(node_parent=self._nodes[node_from], node_child=self._nodes[node_to])
        self._edges_structure[node_from][node_to] = 1

    def _run_validating_fitting_and_evaluating(self):
        """ Trying to fit the pipeline """
        # TODO: Update function to avoid - try ... except ... -

        try:
            self._pipeline.fit(self._train_data)

        except:
            self._rules['valid'] = False
            self._rules['pipeline_fitting'] = False

        try:
            self.y_pred = self._pipeline.predict(self._test_data).predict
            self.y_true = self._test_data.target

            self._metric = mean_absolute_error(self.y_pred, self.y_true)
        except:
            self._rules['valid'] = False
            self._rules['pipeline_predicting'] = False

    def load_dataloader(self, dataloader: TimeSeriesDataLoader):
        self._dataloader = dataloader

    def load_data(self, train: Optional[InputData], test: Optional[InputData], meta: Optional[np.ndarray]):
        self._train_data = train
        self._test_data = test
        self._meta_data = meta

        return self

    @staticmethod
    def _get_default_primitives():
        return OperationTypesRepository('all').suitable_operation(task_type=TaskTypesEnum.ts_forecasting)

    @staticmethod
    def _apply_one_hot_encoding(vector: np.ndarray, num_classes: int = 3) -> np.ndarray:
        """ Applying One Hot Encoding to vector with current number of classes """
        return np.squeeze(np.eye(num_classes)[vector.reshape(-1)])

    @staticmethod
    def get_reward_by_metric(m, m_min=-10000, m_max=0):
        return (-1 * m - m_min) / (m_max - m_min)

    def _get_state_dim(self) -> int:
        graph_structure = self._get_graph_structure()
        graph_shape = graph_structure.shape[0]

        if self.metadata_dim:
            return graph_shape + self.metadata_dim

        return graph_shape

    def _get_graph_structure(self) -> np.ndarray:
        node_structure = np.ravel(self._apply_one_hot_encoding(self._nodes_structure, self.number_of_primitives + 1))
        edge_structure = np.ravel(self._edges_structure)

        return np.concatenate((node_structure, edge_structure))

    @staticmethod
    def _get_maximum_number_of_actions_in_environment(number_of_nodes) -> int:
        return int((number_of_nodes * (number_of_nodes + 1)) / 2) + 1


if __name__ == '__main__':
    data_folder_path = os.path.join(str(project_root()), 'MetaFEDOT\\data\\knowledge_base_time_series_0\\datasets\\')
    dataset_names = [name for name in os.listdir(data_folder_path)]

    train, test = train_test_split(dataset_names, test_size=3, random_state=42)

    train_datasets = {}
    for dataset in train:
        train_datasets[dataset] = os.path.join(data_folder_path, f'{dataset}/data.csv')

    path_to_meta_data = os.path.join(
        str(project_root()),
        'MetaFEDOT\\data\\knowledge_base_time_series_0\\meta_features_ts.csv'
    )

    dataloader = TimeSeriesDataLoader(train_datasets, path_to_meta_data=path_to_meta_data)
    train_data, test_data, meta_data = dataloader.get_data(dataset_name='M4_Q5278')

    env = TimeSeriesPipelineEnvironment(
        max_number_of_nodes=10,
        using_number_of_nodes=10,
        render_mode='pipeline_plot',
        metadata_dim=125
    )
    env.load_data(train_data, test_data, meta_data)
    terminated = False

    total_reward = 0
    done = False

    state = env.reset()

    while not done:
        env.print_available_actions()
        action = int(input())

        new_state, reward, terminated, truncated, info = env.step(action)
        print(f'reward {reward} \ninfo: {info}')

        print(env._pipeline.depth)

        total_reward += reward
        done = terminated or truncated

    info['pipeline'].show()
    print(f'\n{info["pipeline"]}, metric {info["metric"]}, reward {total_reward}')
    print(f'{info["validation_rules"]}')
