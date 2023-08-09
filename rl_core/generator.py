from collections import deque

import numpy as np
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum, Task
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from meta_automl.utils import project_root
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

        self.tensorboard_writer = None

        self.dataloader = None

        self.agent = None
        self.optimizer = None
        self.experience = None
        self.visualizer = None
        self.buffer = None

        self.hidden_dim = None
        self.critic_updates_per_actor = None
        self.eval_schedule = None
        self.train_schedule = None

        self.total_train_reward = []
        self.batch = []

    @staticmethod
    def _get_default_environment(task_type, state_dim):
        # TODO: Adding tags "reinforce" into Fedot
        primitives = OperationTypesRepository('all').suitable_operation(task_type=task_type)

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

            for d_primitves in ['lgbm', 'knn']:
                primitives.remove(d_primitves)

        self.env = env(state_dim=self.state_dim, primitives=primitives)
        return self

    def set_dataloader(self, datasets: dict = None, split_ratio: float = 0.7):
        self.dataloader = DataLoader(datasets, split_ratio)
        return self

    def set_agent(self, hidden_dims=(128, 128), lr=1e-3, gamma=0.99, replay_buffer_size=500, train_schedule=32,
                  eval_schedule=100, critic_updates_per_actor=4, path_to_weights=None):
        self.hidden_dim = hidden_dims[-1]
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

        if path_to_weights:
            self.load_agent(path_to_weights)

        return self

    def set_writer(self):
        log_dir = f'{project_root()}/MetaFEDOT/rl_core/agent/tensorboard_logs'
        self.tensorboard_writer = SummaryWriter(log_dir=log_dir)

        return self

    def _fit(self):
        for episode in tqdm(range(1, self.n_episodes+1)):
            train_data, val_data = self.dataloader.get_data()
            self.env.load_data(train_data, val_data)

            s = self.env.reset()

            done, episode_return = False, 0
            eval = False

            while not done:
                a = self.agent.act(s)
                s_next, r, terminated, truncated, info = self.env.step(a)
                episode_return += r
                done = terminated or truncated

                self.agent.append_to_replay_buffer(s, a, r, s_next, terminated)
                actor_loss, critic_loss = self.agent.update(self.train_schedule, self.batch_size,
                                                            self.critic_updates_per_actor)

                if self.tensorboard_writer and (actor_loss and critic_loss):
                    self.tensorboard_writer.add_scalar('Loss Actor', actor_loss, episode)
                    self.tensorboard_writer.add_scalar('Loss Critic', critic_loss, episode)

                if episode % self.eval_schedule == 0:
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
                        print(f'{episode=} | {avg_return=:.3f}')
                        print(f'{valid_pipeline=} | {avg_metric=:.3f}')

                        if avg_metric >= self.success_ret:
                            print(f'\n --- Average metric {avg_metric} >= Threshold {self.success_ret} ---')

                        if self.tensorboard_writer:
                            message = f'Average return reward per {self.smooth_ret_window}'
                            self.tensorboard_writer.add_scalar(message, avg_return, episode)

                            message = f'Count valid pipeline from max {self.smooth_metric_window}'
                            self.tensorboard_writer.add_scalar(message, valid_pipeline, episode)

                            message = f'Average pipelines metric per {self.smooth_metric_window}'
                            self.tensorboard_writer.add_scalar(message, avg_metric, episode)

    def fit(self):
        self._fit()

        return np.mean(self.episode_return_history), np.mean(self.episode_metric_history)

    def generate(self, path_to_dataset):
        train_data, val_data = self.dataloader.load_data(path_to_dataset)
        self.env.load_data(train_data, val_data)

        info = {}
        done = False

        s = self.env.reset()

        while not done:
            a = self.agent.act(s)
            s_next, r, terminated, truncated, info = self.env.step(a)
            done = terminated or truncated

            s = s_next

        if info['is_valid']:
            return info['pipeline'], info['metric_value']

        else:
            return None, 0

    def save_agent(self):
        name = f'{self.env._meta_info["name"]}_{self.state_dim}_a2c_{self.hidden_dim}_{self.n_episodes}'
        path = f'{project_root()}/MetaFEDOT/rl_core/agent/pretrained/{name}'
        self.agent.save(path)

        print(f'Model was saved in: {path}')

    def load_agent(self, path):
        self.agent.load(path)

        print(f'Model was loaded from: {path}')
