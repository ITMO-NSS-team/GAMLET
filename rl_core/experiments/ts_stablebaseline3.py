import gymnasium as gym
import numpy as np
import wandb
from sb3_contrib import RecurrentPPO
from wandb.integration.sb3 import WandbCallback

from rl_core.environments.time_series import TimeSeriesPipelineEnvironment
from rl_core.utils import define_data_for_experiment


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()


if __name__ == '__main__':
    config = {
        'agent': 'RecurrentPPO_retrain',
        'policy_type': 'MaskableMultiInputActorCriticPolicy',
        'total_timesteps': 500000,
        'env_name': 'TimeSeriesPipelineEnvironment',
        'environment': 'TimeSeries',
        'max_number_of_nodes_in_pipeline': 10,
        'using_number_of_nodes': 10,
        'metadata_dim': 126,
        'gamma': 0.99,
        'ent_coef': 0.1,
        'clip_range': 0.2,
    }

    run = wandb.init(
        project='sb3_ts_recurrent',
        config=config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True
    )

    env_params = dict(
        max_number_of_nodes=config['max_number_of_nodes_in_pipeline'],
        using_number_of_nodes=config['using_number_of_nodes'],
        render_mode='none',
        metadata_dim=config['metadata_dim'],
        is_use_dataloader=True
    )

    env = TimeSeriesPipelineEnvironment(**env_params)

    # Define data for experiment
    dataloader_train, dataloader_test, train_list, test_list = define_data_for_experiment()
    # Load Dataloader
    env.load_dataloader(dataloader_train)

    # env = ActionMasker(env, mask_fn)

    model = RecurrentPPO(
        'MultiInputLstmPolicy',
        env,
        n_steps=4096,
        batch_size=1024,
        gamma=config['gamma'],
        ent_coef=config['ent_coef'],
        clip_range=config['clip_range'],
        verbose=1,
        tensorboard_log=f'agent/tensorboard_logs/sb3_rppo/{run.id}',
        device='cuda'
    )

    model.load('agent/pretrained/sb3_rppo/aj34y8g5/model.zip')

    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f'agent/pretrained/sb3_rppo/{run.id}',
            verbose=1
        )
    )

    # Validation
    train_data, test_data, meta_data = dataloader_train.get_data()
    env.load_data(train_data, test_data, meta_data)

    done = False
    state, _ = env.reset()

    while not done:
        action, _state = model.predict(state)
        # For MaskedPPO use next:
        # action, _state = model.predict(state, action_masks=env.valid_action_mask())
        print(f'{action}', end=', ')

        next_state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        state = next_state
        mask = env.get_available_actions_mask()

        print(f'reward {reward} \ninfo: {info}')

    info['pipeline'].show()
