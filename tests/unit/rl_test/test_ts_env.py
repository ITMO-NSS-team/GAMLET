import pytest

from rl_core.environments.time_series import TimeSeriesPipelineEnvironment
from tests.unit.rl_test.utils import get_time_series

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
        [2, 0], [9, 0], [11, 0], [12, 0], [13, 0], [14, 0],                                 # Single node pipelines
        [25, 1, 31, 0], [25, 3, 31, 0], [25, 4, 31, 0], [25, 5, 31, 0], [25, 6, 31, 0],     # (1) lagged -> ..model..
        [25, 7, 31, 0], [25, 8, 31, 0], [25, 10, 31, 0], [25, 15, 31, 0],                   # (2) lagged -> ..model..
    ]
)
def test_correct_pipelines(trajectory):
    train_data, test_data = get_time_series()

    env = TimeSeriesPipelineEnvironment(metadata_dim=125)
    env.load_data(train_data, test_data, meta=None)

    total_reward = 0

    env.reset()

    for action in trajectory:
        _, reward, _, _, info = env.step(action)
        total_reward += reward

    assert total_reward > 0
    assert info['validation_rules']['valid'] is True
