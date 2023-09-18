import os

from meta_automl.utils import project_root
from rl_core.generator import Generator

if __name__ == '__main__':
    task_type = 'classification'
    pipeline_len = 5
    n_episodes = 1000

    path_to_agent = os.path.join(str(project_root()), 'MetaFEDOT/rl_core/agent/pretrained/ensemble_5_a2c_128_1000')

    data_folder_path = os.path.join(str(project_root()), 'MetaFEDOT/rl_core/data/')

    train_datasets = {
        'amazon': os.path.join(data_folder_path, 'amazon_train.csv'),
        'australian': os.path.join(data_folder_path, 'australian_train.csv'),
        'bank': os.path.join(data_folder_path, 'bank_train.csv'),
        'blood': os.path.join(data_folder_path, 'blood_train.csv'),
        'jasmine': os.path.join(data_folder_path, 'jasmine_train.csv'),
        'kc1': os.path.join(data_folder_path, 'kc1_train.csv'),
    }

    primitives = ['scaling', 'simple_imputation', 'normalization', 'dt', 'logit', 'rf', 'knn']

    gen = Generator(task_type, state_dim=pipeline_len, n_episodes=n_episodes) \
        .set_environment(env_name='ensemble', primitives=primitives) \
        .set_dataloader(train_datasets) \
        .set_agent(
            eval_schedule=25,
            critic_updates_per_actor=25,
        ) \
        .set_writer()

    if path_to_agent:
        gen.load_agent(path=path_to_agent)
    else:
        gen.fit()
        gen.save_agent()

    test_datasets = {
        'amazon': os.path.join(data_folder_path, 'amazon_test.csv'),
        'australian': os.path.join(data_folder_path, 'australian_test.csv'),
        'bank': os.path.join(data_folder_path, 'bank_test.csv'),
        'blood': os.path.join(data_folder_path, 'blood_test.csv'),
        'jasmine': os.path.join(data_folder_path, 'jasmine_test.csv'),
        'kc1': os.path.join(data_folder_path, 'kc1_test.csv'),
    }

    for name, dataset in test_datasets.items():
        valid, not_valid = 0, 0

        for _ in range(25):
            pipeline, metric_value = gen.generate(path_to_dataset=dataset)

            if pipeline:
                pipeline.build().show()
                print('Test metric:', metric_value)
                valid += 1
            else:
                print('Pipeline generation failed!')
                not_valid += 1

        print(f'{valid=}, {not_valid=}')

