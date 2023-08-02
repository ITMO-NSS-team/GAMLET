import os

from meta_automl.utils import project_root
from rl_core.generator import Generator

if __name__ == '__main__':
    task_type = 'classification'
    pipeline_len = 5
    n_episodes = 300

    scoring_train_path = os.path.join(str(project_root()), 'MetaFEDOT/rl_core/data/scoring_train.csv')
    airlines_train_path = os.path.join(str(project_root()), 'MetaFEDOT/rl_core/data/airlines.csv')

    train_datasets = {
        'scoring': scoring_train_path,
        'airlines': airlines_train_path
    }

    gen = Generator(task_type, state_dim=pipeline_len, n_episodes=n_episodes) \
        .set_environment(env_name='linear') \
        .set_dataloader(train_datasets) \
        .set_agent(eval_schedule=50)

    gen.fit()

    scoring_test_path = os.path.join(str(project_root()), 'MetaFEDOT/rl_core/data/scoring_test.csv')

    # TODO: Adding Saving / Loading pre-trained agent

    for _ in range(25):
        pipeline, metric_value = gen.generate(path_to_dataset=scoring_test_path)
        print(metric_value)
        pipeline.show()

