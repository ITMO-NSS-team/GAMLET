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

from meta_automl.surrogate import surrogate_model

from .train_surrogate_model import get_datasets


def train_surrogate_model(
        config: Dict[str, Any],
        meta_data: Dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
) -> List[Dict[str, float]]:
    """Optimized version of
    `surrogate.training.surrogate_model.train_surrogate_model.train_surrogate_model.
    Data loading is moved outside the function to avoid the data reloading.
    """
    model_class = getattr(surrogate_model, config["model"]["name"])

    config["model"]["model_parameters"]["in_size"] = meta_data["in_size"]
    config["model"]["model_parameters"]["dim_dataset"] = meta_data["dim_dataset"]
    dim_feedforward = 2 * config["model"]["model_parameters"]["d_model"]
    config["model"]["model_parameters"]["dim_feedforward"] = dim_feedforward

    model_config = config["model"].copy()
    model_config.pop("name")
    model = model_class(**model_config)

    if config["tensorboard_logger"] is not None:
        logger = TensorBoardLogger(**config["tensorboard_logger"])
    else:
        logger = None

    model_checkpoint_callback = ModelCheckpoint(**config["model_checkpoint_callback"])

    if config["early_stopping_callback"] is not None:
        early_stopping_callback = EarlyStopping(**config["early_stopping_callback"])
    else:
        early_stopping_callback = None

    trainer = Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=[c for c in [model_checkpoint_callback, early_stopping_callback] if c is not None],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    checkpoint = torch.load(model_checkpoint_callback.best_model_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    test_results = trainer.test(model, dataloaders=test_loader)
    return test_results


def objective(
        trial: Trial,
        config: Dict[str, Any],
        divisble_d_model_num_heads: List[str],
        meta_data: Dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
) -> Any:
    config = deepcopy(config)
    # Model parameters
    # Restore tuple from string
    d_model, num_heads = eval(trial.suggest_categorical("d_model_num_heads", divisble_d_model_num_heads))
    config["model"]["model_parameters"]["d_model"] = d_model
    config["model"]["model_parameters"]["num_heads"] = num_heads
    config["model"]["model_parameters"]["dropout"] = trial.suggest_float(
        "dropout",
        *config["model"]["model_parameters"]["dropout"],
    )
    config["model"]["model_parameters"]["num_layers"] = trial.suggest_int(
        "num_layers",
        *config["model"]["model_parameters"]["num_layers"],
    )
    # config["model"]["model_parameters"]["batch_norm"] = trial.suggest_categorical(
    #     "batch_norm",
    #     config["model"]["model_parameters"]["batch_norm"],
    # )
    config["model"]["model_parameters"]["gnn_type"] = trial.suggest_categorical(
        "gnn_type",
        config["model"]["model_parameters"]["gnn_type"],
    )
    config["model"]["model_parameters"]["k_hop"] = trial.suggest_int(
        "k_hop",
        *config["model"]["model_parameters"]["k_hop"],
    )
    config["model"]["model_parameters"]["global_pool"] = trial.suggest_categorical(
        "global_pool",
        config["model"]["model_parameters"]["global_pool"],
    )
    # Optimizer parameters
    config["model"]["lr"] = trial.suggest_loguniform(
        "lr",
        *config["model"]["lr"],
    )
    config["model"]["weight_decay"] = trial.suggest_loguniform(
        "weight_decay",
        *config["model"]["weight_decay"],
    )

    if config["tensorboard_logger"] is not None:
        config["tensorboard_logger"]["name"] = (
                config["tensorboard_logger"]["name"] +
                f"__trial_id_{trial._trial_id}"
        )

    test_metric = []
    for i_it in range(5):
        test_result = train_surrogate_model(
            config=config,
            meta_data=meta_data,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )
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

    is_pair = False
    model_class = getattr(surrogate_model, config["model"]["name"])
    if model_class.__name__ == "RankingPipelineDatasetSurrogateModel":
        is_pair = True

    train_dataset, val_dataset, test_dataset, meta_data = get_datasets(
        config["dataset_params"]["root_path"],
        is_pair,
    )

    # Not the best solution, that may lead to overfitting, but enables fair comparison with traditional models.
    if len(val_dataset) == 0:
        config["early_stopping_callback"]["monitor"] = "train_loss"
        config["model_checkpoint_callback"]["monitor"] = "train_loss"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_dataloader_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_dataloader_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_dataloader_workers"],
    )

    # To avoid mismatch of arguments for torch attention (`embed_dim must be divisible by num_heads`).
    divisble_d_model_num_heads = find_divisible_pairs(
        list(range(*config["model"]["model_parameters"]["d_model"])),
        list(range(*config["model"]["model_parameters"]["num_heads"]))
    )
    # Hack to enable persistent storage in a database
    divisble_d_model_num_heads = list(map(str, divisble_d_model_num_heads))

    objective_function = partial(
        objective,
        divisble_d_model_num_heads=divisble_d_model_num_heads,
        config=config,
        meta_data=meta_data,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    study.optimize(
        objective_function,
        n_trials=n_trials,
        show_progress_bar=True,
        # catch=(AssertionError,),
    )
