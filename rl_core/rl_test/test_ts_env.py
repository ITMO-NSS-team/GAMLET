from itertools import permutations

import numpy as np
import pytest

from rl_core.environments.time_series import TimeSeriesPipelineEnvironment
from rl_core.rl_test.utils import get_time_series


# {
#       1: 'adareg',
#       2: 'ar',
#       3: 'dtreg',
#       4: 'lasso',
#       5: 'lgbmreg',
#       6: 'linear',
#       7: 'rfr',
#       8: 'ridge',
#       9: 'polyfit',
#       10: 'sgdr',
#       11: 'glm',
#       12: 'ets',
#       13: 'locf',
#       14: 'ts_naive_average',
#       15: 'svr',
#       16: 'topological_features',
#       17: 'scaling',
#       18: 'normalization',
#       19: 'pca',
#       20: 'fast_ica',
#       21: 'poly_features',
#       22: 'ransac_lin_reg',
#       23: 'ransac_non_lin_reg',
#       24: 'isolation_forest_reg',
#       25: 'lagged',
#       26: 'sparse_lagged',
#       27: 'smoothing',
#       28: 'gaussian_filter',
#       29: 'diff_filter',
#       30: 'exog_ts'
#  }


@pytest.mark.parametrize('trajectory',
                         [
                             [2, 0], [9, 0], [11, 0], [12, 0], [13, 0], [14, 0],  # Single node pipelines
                             [25, 1, 31, 0], [25, 3, 31, 0], [25, 4, 31, 0], [25, 5, 31, 0], [25, 6, 31, 0],
                             # (1) lagged -> ..model..
                             [25, 7, 31, 0], [25, 8, 31, 0], [25, 10, 31, 0], [25, 15, 31, 0],
                             # (2) lagged -> ..model..
                         ]
                         )
def test_correct_pipelines(trajectory):
    train_data, test_data = get_time_series()

    env = TimeSeriesPipelineEnvironment(metadata_dim=None)
    env.load_data(train_data, test_data, meta=None)

    total_reward = 0

    env.reset()

    for action in trajectory:
        _, reward, _, _, info = env.step(action)
        total_reward += reward

    assert total_reward > 0
    assert info['validation_rules']['valid'] is True


@pytest.mark.parametrize('trajectory',
                         [
                             [1, 2, 3, 31, 41, 49, 0],  # Call pipeline.depth == -1
                         ]
                         )
def test_uncorrect_pipelines(trajectory):
    train_data, test_data = get_time_series()

    env = TimeSeriesPipelineEnvironment(metadata_dim=None)
    env.load_data(train_data, test_data, meta=None)

    total_reward = 0

    env.reset()

    for action in trajectory:
        _, reward, _, _, info = env.step(action)
        total_reward += reward

    assert total_reward < 0
    assert info['validation_rules']['valid'] is False


@pytest.mark.parametrize('max_number_of_nodes', [i for i in range(1, 25)])
def test_max_number_of_actions_in_pipelines(max_number_of_nodes):
    train_data, test_data = get_time_series()

    env = TimeSeriesPipelineEnvironment(
        max_number_of_nodes=max_number_of_nodes,
        metadata_dim=0,
        using_number_of_nodes=max_number_of_nodes
    )
    env.load_data(train_data, test_data, meta=None)
    env.reset()

    done = False
    action_number = 0

    while not done:
        available_actions = env.get_available_actions()

        if len(available_actions) == 1:
            action = np.random.choice(available_actions)
        else:
            action = np.random.choice(available_actions)

            while action == 0:
                action = np.random.choice(available_actions)

        _, _, terminated, truncated, info = env.step(action)
        action_number += 1

        done = terminated or truncated

    assert action_number == env.max_number_of_actions


@pytest.mark.parametrize('trajectory', [list(pair) for pair in permutations(range(1, 30), 2)])
def test_correct_pairs_pipelines(trajectory: list):
    train_data, test_data = get_time_series()

    env = TimeSeriesPipelineEnvironment(metadata_dim=None, max_number_of_nodes=2)
    env.load_data(train_data, test_data, meta=None)

    trajectory = trajectory + [31, 0]

    total_reward = 0
    env.reset()

    for action in trajectory:
        _, reward, _, _, info = env.step(action)
        total_reward += reward

    assert total_reward > 0
    assert info['validation_rules']['valid'] is True

    if info['validation_rules']['valid'] is True:
        print(info["pipeline"].structure)
