# import os.path
#
# import numpy as np
# import torch
# from fedot.core.pipelines.pipeline import Pipeline
# from fedot.core.repository.operation_types_repository import OperationTypesRepository
# from fedot.core.repository.tasks import TaskTypesEnum
#
# from gamlet.utils import project_root
# from rl_core.dataloader import DataLoader
# from rl_core.environments.ensemble import EnsemblePipelineGenerationEnvironment
# from rl_core.environments.linear import LinearPipelineGenerationEnvironment
#
#
# def test_linear_pipeline_environment():
#     primitives = OperationTypesRepository('all').suitable_operation(task_type=TaskTypesEnum.classification)
#
#     state_dim = 4
#     env = LinearPipelineGenerationEnvironment(state_dim=state_dim, primitives=primitives)
#
#     initial_state = env.reset()
#
#     assert initial_state.numpy().shape == (state_dim,)
#
#     path = os.path.join(str(project_root()), 'MetaFEDOT/rl_core/data/scoring_train.csv')
#
#     datasets = {
#         'scoring': path,
#     }
#
#     dataloader = DataLoader(datasets)
#
#     train_data, val_data = dataloader.get_data()
#     env.load_data(train_data, val_data)
#
#     assert train_data == env.train_data
#     assert val_data == env.val_data
#
#     action = env.action_space.sample()
#
#     new_state, r, done, info = env.step(action)
#
#     assert initial_state.shape == new_state.shape
#     assert np.alltrue(initial_state.numpy() == new_state.numpy())
#     assert isinstance(r, float)
#     assert isinstance(done, bool)
#     assert isinstance(info, dict)
#
#
# def test_linear_pipeline_environment_pipeline_build():
#     primitives = OperationTypesRepository('all').suitable_operation(
#         task_type=TaskTypesEnum.classification)
#
#     state_dim = 2
#     env = LinearPipelineGenerationEnvironment(state_dim=state_dim, primitives=primitives)
#
#     path = os.path.join(str(project_root()), 'MetaFEDOT/rl_core/data/scoring_train.csv')
#
#     datasets = {
#         'scoring': path,
#     }
#
#     dataloader = DataLoader(datasets)
#
#     train_data, val_data = dataloader.get_data()
#     env.load_data(train_data, val_data)
#
#     actions = [env.primitives.index(p) for p in ['rf', 'eop']]
#
#     for action in actions:
#         new_state, r, done, info = env.step(action)
#
#     assert isinstance(info['pipeline'], Pipeline)
#     assert isinstance(info['time_step'], int)
#     assert isinstance(info['metric_value'], float)
#     assert info['metric_value'] > 0.5
#
#
# def test_ensemble_pipeline_environment():
#     primitives = OperationTypesRepository('all').suitable_operation(task_type=TaskTypesEnum.classification)
#
#     state_dim = 4
#     env = EnsemblePipelineGenerationEnvironment(state_dim=state_dim, primitives=primitives)
#
#     initial_state = env.reset()
#
#     assert initial_state.numpy().shape == (state_dim,)
#
#     path = os.path.join(str(project_root()), 'MetaFEDOT/rl_core/data/scoring_train.csv')
#
#     datasets = {
#         'scoring': path,
#     }
#
#     dataloader = DataLoader(datasets)
#
#     train_data, val_data = dataloader.get_data()
#     env.load_data(train_data, val_data)
#
#     assert train_data == env.train_data
#     assert val_data == env.val_data
#
#     action = env.action_space.sample()
#
#     new_state, r, done, info = env.step(action)
#
#     assert initial_state.shape == new_state.shape
#     assert np.alltrue(initial_state.numpy() == new_state.numpy())
#     assert isinstance(r, float)
#     assert isinstance(done, bool)
#     assert isinstance(info, dict)
#
#
# def test_ensemble_pipeline_environment_pipeline_build():
#     primitives = OperationTypesRepository('all').suitable_operation(
#         task_type=TaskTypesEnum.classification)
#
#     state_dim = 2
#     env = EnsemblePipelineGenerationEnvironment(state_dim=state_dim, primitives=primitives)
#
#     path = os.path.join(str(project_root()), 'MetaFEDOT/rl_core/data/scoring_train.csv')
#
#     datasets = {
#         'scoring': path,
#     }
#
#     dataloader = DataLoader(datasets)
#
#     train_data, val_data = dataloader.get_data()
#     env.load_data(train_data, val_data)
#
#     actions = [env.primitives.index(p) for p in ['rf', 'scaling', 'knn', 'eop']]
#
#     for action in actions:
#         new_state, r, done, info = env.step(action)
#
#     assert isinstance(info['pipeline'], Pipeline)
#     assert isinstance(info['time_step'], int)
#     assert isinstance(info['metric_value'], float)
#     assert info['metric_value'] > 0.5
#
#
# def test_linear_pipeline_environment_pipeline_reset():
#     primitives = OperationTypesRepository('all').suitable_operation(
#         task_type=TaskTypesEnum.classification)
#
#     state_dim = 2
#     env = LinearPipelineGenerationEnvironment(state_dim=state_dim, primitives=primitives)
#
#     path = os.path.join(str(project_root()), 'MetaFEDOT/rl_core/data/scoring_train.csv')
#
#     datasets = {
#         'scoring': path,
#     }
#
#     dataloader = DataLoader(datasets)
#
#     train_data, val_data = dataloader.get_data()
#     env.load_data(train_data, val_data)
#
#     actions = [env.primitives.index(p) for p in ['rf', 'dt', 'eop']]
#
#     for action in actions:
#         new_state, r, done, info = env.step(action)
#
#     env.reset()
#
#     assert torch.all(env.state == torch.tensor(np.array([0, 0]), dtype=torch.float64))
#
#     for action in actions:
#         new_state, r, done, info = env.step(action)
#
#     assert torch.all(new_state == torch.tensor(np.array([8, 2]), dtype=torch.float64))
#
#
# def test_linear_pipeline_environment_state():
#     primitives = OperationTypesRepository('all').suitable_operation(
#         task_type=TaskTypesEnum.classification)
#
#     state_dim = 4
#     env = LinearPipelineGenerationEnvironment(state_dim=state_dim, primitives=primitives)
#
#     path = os.path.join(str(project_root()), 'MetaFEDOT/rl_core/data/scoring_train.csv')
#
#     datasets = {
#         'scoring': path,
#     }
#
#     dataloader = DataLoader(datasets)
#
#     train_data, val_data = dataloader.get_data()
#     env.load_data(train_data, val_data)
#
#     actions = [env.primitives.index(p) for p in ['knn', 'dt', 'scaling', 'rf']]
#     states = []
#
#     for action in actions:
#         new_state, r, done, info = env.step(action)
#         states.append(new_state.numpy())
#
#     assert np.unique(states, axis=1).shape[1] == len(states)
