import random

import torch
from wandb.integration.sb3 import WandbCallback

import wandb
import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

from rl_core.environments.time_series import TimeSeriesPipelineEnvironment
from rl_core.utils import define_data_for_experiment, OFFLINE_TRAJECTORIES


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()


if __name__ == '__main__':
    config = {
        'agent': 'MaskablePPO',
        'policy_type': 'MaskableMultiInputActorCriticPolicy',
        'total_timesteps': 250000,
        'env_name': 'TimeSeriesPipelineEnvironment',
        'environment': 'TimeSeries',
        'max_number_of_nodes_in_pipeline': 10,
        'using_number_of_nodes': 10,
        'metadata_dim': 126,
        'gamma': 0.99,
        'ent_coef': 0.15,
    }

    run = wandb.init(project='sb3_ts_ref', config=config, sync_tensorboard=True, monitor_gym=False, save_code=True)

    env = TimeSeriesPipelineEnvironment(
        max_number_of_nodes=config['max_number_of_nodes_in_pipeline'],
        using_number_of_nodes=config['using_number_of_nodes'],
        render_mode='none',
        metadata_dim=config['metadata_dim'],
        is_use_dataloader=True
    )

    # Define data for experiment
    dataloader_train, dataloader_test, train_list, test_list = define_data_for_experiment()
    # Load Dataloader
    env.load_dataloader(dataloader_train)

    # env = ActionMasker(env, mask_fn)

    model = PPO(
        'MultiInputPolicy',
        env,
        n_steps=4096,
        gamma=config['gamma'],
        ent_coef=config['ent_coef'],
        verbose=1,
        tensorboard_log=f'agent/tensorboard_logs/sb3_ppo/{run.id}',
        device='cuda'
    )

    # model.load('agent/pretrained/sb3_mppo/bzcdva6o/model.zip')

    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f'agent/pretrained/sb3_ppo/{run.id}',
            verbose=1
        )
    )

    # Validation
    train_data, test_data, meta_data = dataloader_train.get_data()
    env.load_data(train_data, test_data, meta_data)

    done = False
    state, _ = env.reset()

    while not done:
        # action, _state = model.predict(state, action_masks=env.valid_action_mask())
        action, _state = model.predict(state)
        print(f'{action}', end=', ')

        next_state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        state = next_state
        mask = env.get_available_actions_mask()

        print(f'reward {reward} \ninfo: {info}')

    info['pipeline'].show()

