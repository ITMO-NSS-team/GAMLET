import os

from meta_automl.utils import project_root
from rl_core.generator import Generator

os.environ['OPENBLAS_NUM_THREADS'] = '1'

if __name__ == '__main__':
    task_type = 'classification'
    pipeline_len = 5
    n_episodes = 500

    scoring_train_path = os.path.join(str(project_root()), 'MetaFEDOT/rl_core/data/scoring_train.csv')
    airlines_train_path = os.path.join(str(project_root()), 'MetaFEDOT/rl_core/data/airlines.csv')

    train_datasets = {
        'scoring': scoring_train_path,
        'airlines': airlines_train_path
    }

    primitives = ['scaling', 'simple_imputation', 'normalization', 'dt', 'logit', 'rf', 'knn']

    gen = Generator(task_type, state_dim=pipeline_len, n_episodes=n_episodes) \
        .set_environment(env_name='linear', primitives=primitives) \
        .set_dataloader(train_datasets) \
        .set_agent(
            eval_schedule=10,
            critic_updates_per_actor=10,
        ) \
        .set_writer()

    gen.fit()

    gen.save_agent()

    scoring_test_path = os.path.join(str(project_root()), 'MetaFEDOT/rl_core/data/scoring_test.csv')

    valid, not_valid = 0, 0
    for _ in range(25):
        pipeline, metric_value = gen.generate(path_to_dataset=scoring_test_path)

        if pipeline:
            pipeline.show()
            print('Test metric:', metric_value)
            valid += 1
        else:
            print('Pipeline generation failed!')
            not_valid += 1

    print(f'{valid=}, {not_valid=}')

