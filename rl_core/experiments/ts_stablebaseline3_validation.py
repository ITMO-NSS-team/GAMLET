from sb3_contrib import RecurrentPPO
from sb3_contrib.common.wrappers import ActionMasker

from rl_core.environments.time_series import TimeSeriesPipelineEnvironment
from rl_core.experiments.ts_stablebaseline3 import mask_fn
from rl_core.utils import define_data_for_experiment

if __name__ == '__main__':
    env = TimeSeriesPipelineEnvironment(
        max_number_of_nodes=10,
        using_number_of_nodes=10,
        render_mode='none',
        metadata_dim=126,
        is_use_dataloader=False
    )

    dataloader_train, dataloader_test, train_list, test_list = define_data_for_experiment()
    train_data, test_data, meta_data = dataloader_train.get_data()
    env.load_dataloader(dataloader_train)
    env.load_data(train_data, test_data, meta_data)

    env = ActionMasker(env, mask_fn)

    model = RecurrentPPO.load("../agent/pretrained/sb3_rppo/zao760zl/model.zip")

    for dataset in test_list:
        print(f'{dataset}')
        train_data, test_data, meta_data = dataloader_test.get_data(dataset)
        env.load_data(train_data, test_data, meta_data)

        done = False
        state, _ = env.reset()

        while not done:
            # action, _state = model.predict(state, action_masks=env.valid_action_mask())
            action, _state = model.predict(state)
            # print(f'{action}', end=', ')

            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            state = next_state
            mask = env.get_available_actions_mask()

        print(f'pipeline {info["pipeline"].nodes} reward {reward} metric {info["metric"]}')
        if info['pipeline'].depth > 1:
            info['pipeline'].show()
        print('----\n')
