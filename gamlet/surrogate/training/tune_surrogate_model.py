"""The module contains custom method to tune `surrogate.models.SurrogateModel`."""
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List

import numpy as np
import optuna
import torch
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna.trial import Trial
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader

from gamlet.surrogate import surrogate_model
from gamlet.surrogate.training import setup_loaders,do_training

def _generate_config(config, trial):
    # Model parameters
    # To avoid mismatch of arguments for torch attention (`embed_dim must be divisible by num_heads`).
    divisble_d_model_num_heads = find_divisible_pairs(
        list(range(*config["d_model"])), list(range(*config["num_heads"]))
    )
    # Hack to enable persistent storage in a database
    divisble_d_model_num_heads = list(map(str, divisble_d_model_num_heads))
    d_model, num_heads = eval(trial.suggest_categorical("d_model_num_heads", divisble_d_model_num_heads))

    config["d_model"] = d_model
    config["num_heads"] = num_heads
    config["dropout"] = trial.suggest_float(
        "dropout",
        *config["dropout"],
    )
    config["num_layers"] = trial.suggest_int(
        "num_layers",
        *config["num_layers"],
    )

    config["d_model_dset"] = trial.suggest_int("d_model_dset", *config["d_model_dset"])
    # config["gnn_type"] = trial.suggest_categorical("gnn_type", [x for x in config["gnn_type"]])
    config["k_hop"] = trial.suggest_int(
        "k_hop",
        *config["k_hop"],
    )
    config["global_pool"] = trial.suggest_categorical(
        "global_pool",
        config["global_pool"],
    )
    # Optimizer parameters
    config["lr"] = trial.suggest_loguniform(
        "lr",
        *config["lr"],
    )
    config["weight_decay"] = trial.suggest_loguniform(
        "weight_decay",
        *config["weight_decay"],
    )
    # temperature
    config["temperature"] = trial.suggest_int(
        "temperature",
        *config["temperature"],
    )
    return config


def objective(
    trial: Trial,
    config_base: Dict[str, Any],
    meta_data: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> Any:
    config = deepcopy(config_base)
    # generating config
    config["model"]["model_parameters"] = _generate_config(config["model"]["model_parameters"], trial)

    if config["tensorboard_logger"] is not None:
        config["tensorboard_logger"]["name"] += f"__trial_id_{trial._trial_id}"

    test_metric = []
    for i_it in range(3):
        test_result = do_training(train_loader, val_loader, test_loader, config, meta_data)
        test_metric.append(test_result[0]["test_ndcg"])
    return np.mean(test_metric)


def find_divisible_pairs(set1, set2):
    """"""
    divisible_pairs = []

    for num1 in set1:
        for num2 in set2:
            if num1 % num2 == 0:
                divisible_pairs.append((num1, num2))

    return divisible_pairs


def tune_surrogate_model(config: dict, n_trials: int):
    study = optuna.create_study(
        storage=config["path_to_optuna_db"],
        sampler=TPESampler(
            consider_prior=False,
            consider_magic_clip=True,
            n_startup_trials=10,
            n_ei_candidates=24,
            seed=0,
            multivariate=False,
        ),
        pruner=HyperbandPruner(),
        study_name="surrogate_model_hyperparameter_tuning",
        direction=config["direction"],
        load_if_exists=True,
    )

    train_loader, val_loader, test_loader, config, meta_data = setup_loaders(config)
    objective_function = partial(
        objective,
        config_base=config,
        meta_data=meta_data,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    study.optimize(
        objective_function,
        n_trials=n_trials,
        show_progress_bar=True,
    )
