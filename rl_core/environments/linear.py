from abc import ABC

import numpy as np
import torch
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from gym import spaces

from rl_core.environments.base import PipelineGenerationEnvironment


class LinearPipelineGenerationEnvironment(PipelineGenerationEnvironment, ABC):
    """ Linear Pipeline Generation Environment """
    def __init__(self, state_dim: int, primitives: list):
        super().__init__(state_dim)
        self.primitives = ['pop'] + primitives + ['eop']
        self.action_dim = len(self.primitives)
        self.action_space = spaces.Discrete(self.action_dim)

        self.reset()
        self.position = 0

    def init_state(self):
        self.state = torch.tensor(np.zeros(self.state_dim))
        return self

    def update_state(self, action):
        self.state[self.position] = action
        self.position += 1

        return self

    def reset(self, **kwargs):
        self.pipeline = PipelineBuilder()
        self.time_step = 0
        self.metric_value = 0

        self.position = 0

        self.init_state()

        return self.state

    def _train_step(self, action):
        self.last_action = action

        if self.primitives[action] == 'eop' or self.position == self.state_dim:
            self.time_step += 1
            done = True

            if self._pipeline_constuction_validate(self.pipeline):
                reward = self.pipeline_fitting_and_evaluating()
            else:
                reward = -0.999

        elif self.primitives[action] == 'pop':
            self.time_step += 1
            reward = -0.005
            done = False

        else:
            self.time_step += 1
            reward = -0.001
            done = False

            self.update_state(action)

            primitive = self.primitives[action]
            self.pipeline.add_node(primitive)

        reward, done, info = self._environment_response(reward, done)

        return self.state, reward, done, info

    def _inference_step(self, action):
        raise NotImplementedError()

    def _environment_response(self, reward: float, done: bool) -> (int, bool, dict):
        info = {
            'pipeline': self.pipeline.build(),
            'time_step': self.time_step,
            'metric_value': self.metric_value,
        }

        return reward, done, info