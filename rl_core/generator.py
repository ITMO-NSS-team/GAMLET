from collections import deque

import numpy as np
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum, Task
from tqdm import tqdm

from rl_core.agent.agent import ActorCriticAgent
from rl_core.dataloader import DataLoader
from rl_core.environments.embedding import EmbeddingPipelineGenerationEnvironment
from rl_core.environments.linear import LinearPipelineGenerationEnvironment

SUCCESS_RET = 0.75


class Generator:
    __env_dict = {
        'linear': LinearPipelineGenerationEnvironment,
        'embedding': EmbeddingPipelineGenerationEnvironment,
    }

    def __init__(self, task_type: str, state_dim: int, n_episodes: int, batch_size: int = 32):
        self.critic_updates_per_actor = None
        self.eval_schedule = None
        self.train_schedule = None
        self.n_episodes = n_episodes
        self.task_type = self._transform_task_type(task_type)
        self.task = Task(self.task_type)
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.env = self._get_default_environment(self.task_type, state_dim)

        self.smooth_ret_window = 10
        self.smooth_metric_window = 25
        self.success_ret = SUCCESS_RET
        self.episode_return_history = deque(maxlen=self.smooth_ret_window)
        self.episode_metric_history = deque(maxlen=self.smooth_metric_window)

        self.dataloader = None

        self.agent = None
        self.optimizer = None
        self.experience = None
        self.visualizer = None
        self.buffer = None

        self.total_train_reward = []
        self.batch = []

    @staticmethod
    def _get_default_environment(task_type, state_dim):
        # TODO: Adding tags "reinforce" into Fedot
        primitives = OperationTypesRepository('all').suitable_operation(task_type=task_type)
        # primitives = ['rf', 'dt', 'scaling', 'knn']

        return LinearPipelineGenerationEnvironment(state_dim=state_dim, primitives=primitives)

    @staticmethod
    def _transform_task_type(task: str):
        task_dict = {
            'regression': TaskTypesEnum.regression,
            'classification': TaskTypesEnum.classification,
            'ts_forecasting': TaskTypesEnum.ts_forecasting
        }

        return task_dict[task]

    def set_environment(self, env_name: str, primitives: list[str] = None):
        env = self.__env_dict[env_name]

        if not primitives:
            primitives = OperationTypesRepository('all').suitable_operation(task_type=self.task_type)

        self.env = env(state_dim=self.state_dim, primitives=primitives)
        return self

    def set_dataloader(self, datasets: dict = None, split_ratio: float = 0.7):
        self.dataloader = DataLoader(datasets, split_ratio)
        return self

    def set_agent(self, hidden_dims=(128, 128), lr=1e-3, gamma=0.99, replay_buffer_size=500, train_schedule=32,
                  eval_schedule=250, critic_updates_per_actor=4):
        self.train_schedule = train_schedule
        self.eval_schedule = eval_schedule
        self.critic_updates_per_actor = critic_updates_per_actor
        
        self.agent = ActorCriticAgent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            hidden_dims=hidden_dims,
            lr=lr,
            gamma=gamma,
            critic_rb_size=replay_buffer_size
        )

        return self

    def _fit(self):
        train_data, val_data = self.dataloader.get_data()
        self.env.load_data(train_data, val_data)

        s = self.env.reset()

        done, episode_return = False, 0
        eval = False

        for global_step in tqdm(range(1, self.n_episodes+1)):
            a = self.agent.act(s)
            s_next, r, terminated, truncated, info = self.env.step(a)
            episode_return += r
            done = terminated or truncated

            # train step
            self.agent.append_to_replay_buffer(s, a, r, s_next, terminated)
            self.agent.update(self.train_schedule, self.batch_size, self.critic_updates_per_actor)

            if global_step % self.eval_schedule == 0:
                eval = True

            s = s_next
            if done:
                episode_metric = info['metric_value']
                self.episode_metric_history.append(episode_metric)

                if eval:
                    self.episode_return_history.append(episode_return)
                    avg_return = np.mean(self.episode_return_history)
                    valid_pipeline = len(self.episode_metric_history)
                    avg_metric = np.mean(self.episode_metric_history)

                    print('\n')
                    print(f'{global_step=} | {avg_return=:.3f}')
                    print(f'{valid_pipeline=} | {avg_metric=:.3f}')

                    if avg_metric >= self.success_ret:
                        print('! Average return ')

                s = self.env.reset()
                done, episode_return = False, 0
                eval = False

    def fit(self):
        self._fit()

        return np.mean(self.episode_return_history), np.mean(self.episode_metric_history)

    def generate(self, path_to_dataset):
        train_data, val_data = self.dataloader.load_data(path_to_dataset)
        self.env.load_data(train_data, val_data)

        info = {}
        valid = False

        s = self.env.reset()

        while valid:
            a = self.agent.act(s)
            s_next, r, terminated, truncated, info = self.env.step(a)
            done = terminated or truncated

            s = s_next

            valid = info['is_valid']

        return info['pipeline'], info['metric_value']


