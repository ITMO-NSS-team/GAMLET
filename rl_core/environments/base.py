from typing import Optional

import gym
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from sklearn.metrics import roc_auc_score


class PipelineGenerationEnvironment(gym.Env):
    """ Base Pipeline Generation Environment """
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.state = None
        self.pipeline = PipelineBuilder()

        self.time_step = 0
        self.metric = roc_auc_score
        self.metric_value = None

        self.train_data: Optional[InputData] = None
        self.val_data: Optional[InputData] = None
        self.test_data: Optional[InputData] = None

    def step(self, action: int, mode: str = 'train'):
        assert self.action_space.contains(action)
        state, reward, done, info = None, 0, False, None

        if mode == 'train':
            state, reward, done, info = self._train_step(action)
        elif mode == 'inference':
            state, reward, done, info = self._inference_step(action)

        return state, reward, done, info

    def pipeline_fitting_and_evaluating(self) -> float:
        pipeline = self.pipeline.build()

        pipeline.fit(self.train_data)

        pred = pipeline.predict(self.val_data)

        y_pred = pred.predict
        y_true = self.val_data.target

        try:
            self.metric_value = self.metric(y_score=y_pred, y_true=y_true)
        except:
            self.metric_value = -0.999

        return self.metric_value

    def load_data(self, train_data: Optional[InputData] = None, val_data: Optional[InputData] = None,
                  test_data: Optional[InputData] = None):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        return self

    def environment_response(self, reward: float, done: bool) -> (int, bool, dict):
        raise NotImplementedError()

    def init_state(self):
        raise NotImplementedError()

    def get_state(self):
        return self.state

    def update_state(self, action):
        raise NotImplementedError()

    def reset(self, **kwargs):
        raise NotImplementedError()

    def render(self):
        raise NotImplementedError()

    def _train_step(self, action):
        raise NotImplementedError()

    def _inference_step(self, action):
        raise NotImplementedError()

    @staticmethod
    def _pipeline_constuction_validate(pipeline):
        try:
            if pipeline.build():
                return True
        except:
            return False