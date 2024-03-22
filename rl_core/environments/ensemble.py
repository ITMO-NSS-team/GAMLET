from abc import ABC
from copy import deepcopy

import numpy as np
import torch
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from gym import spaces

from rl_core.agent.agent import to_tensor
from rl_core.environments.base import PipelineGenerationEnvironment


class EnsemblePipelineGenerationEnvironment(PipelineGenerationEnvironment, ABC):
    """ Linear Pipeline Generation Environment """
    _meta_info = {
        'name': 'ensemble',
    }

    def __init__(self, state_dim: int, primitives: list):
        super().__init__(state_dim)
        self.primitives = ['pop'] + primitives + ['eop']
        self.action_dim = len(self.primitives)
        self.action_space = spaces.Discrete(self.action_dim)

        self.reset()
        self.position = 0
        self.meta_model = None
        self.branch_idx = 0

        self.encoded_pipeline = []

    def init_state(self):
        self.state = to_tensor(np.zeros(self.state_dim))
        return self

    def update_state(self, action):
        self.state[self.position] = action
        self.position += 1

        return self

    def reset(self, **kwargs):
        self.pipeline = PipelineBuilder()
        self.encoded_pipeline = []
        self.is_valid = False
        self.time_step = 0
        self.metric_value = 0

        self.position = 0
        self.meta_model = None
        self.branch_idx = 0

        self.init_state()

        return self.state

    def _train_step(self, action, return_pipeline=False):
        terminated, truncated = False, False
        self.last_action = action

        if self.primitives[action] == 'eop' or self.position == self.state_dim:
            self.time_step += 1
            terminated = True

            self.pipeline.join_branches(self.meta_model)

            if self._pipeline_construction_validate(self.pipeline):
                reward = self.pipeline_fitting_and_evaluating()
            else:
                reward = -0.999

        elif self.primitives[action] == 'pop':
            self.time_step += 1
            reward = -0.005

        else:
            self.time_step += 1
            reward = -0.001

            primitive = self.primitives[action]

            self.encoded_pipeline.append(primitive)

            if self.position == 0:
                self.meta_model = primitive
            else:
                self.pipeline.add_branch(primitive, branch_idx=self.branch_idx)
                self.branch_idx += 1

            self.update_state(action)

        reward, info = self._environment_response(reward, return_pipeline)

        return deepcopy(self.state), reward, terminated, truncated, info

    def _inference_step(self, action):
        return self._train_step(action, return_pipeline=True)

    def _environment_response(self, reward: float, return_pipeline: bool) -> (int, bool, dict):
        info = {
            'pipeline': self.pipeline if return_pipeline else None,
            'time_step': self.time_step,
            'metric_value': self.metric_value,
            'is_valid': self.is_valid
        }

        return reward, info
