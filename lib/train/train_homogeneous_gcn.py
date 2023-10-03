from typing import Tuple

import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader

from lib.datasets import HomogeneousPipelineDataset
from lib.models import HomogeneousGCN
from lib.pl import LightningModule


def train_homogeneous_gcn(
        dataset_root: str,
        logdir: str,
        edge_direction: str,
        in_channels: int,
        out_channels: int,
        gnn_hidden_channels: int,
        gnn_num_layers: int,
        mlp_hidden_channels: int,
        mlp_num_layers: int,
        aggregation: str,
        clip_output: Tuple[float, float],
        batch_size: int,
        lr: float,
        max_epochs: int,
        use_operations_hyperparameters: bool,
):
    model = LightningModule(
        model=HomogeneousGCN(
            in_channels=in_channels,
            out_channels=out_channels,
            gnn_hidden_channels=gnn_hidden_channels,
            gnn_num_layers=gnn_num_layers,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_channels=mlp_hidden_channels,
            aggregation=aggregation,
            clip_output=clip_output,
        ),
        loss=F.mse_loss,
        lr=lr,
    )
    train_dataset = HomogeneousPipelineDataset(
        root=dataset_root,
        split="train",
        direction=edge_direction,
        use_operations_hyperparameters=use_operations_hyperparameters,
    )
    val_dataset = HomogeneousPipelineDataset(
        root=dataset_root,
        split="val",
        direction=edge_direction,
        use_operations_hyperparameters=use_operations_hyperparameters,
    )
    test_dataset = HomogeneousPipelineDataset(
        root=dataset_root,
        split="test",
        direction=edge_direction,
        use_operations_hyperparameters=use_operations_hyperparameters,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    experiment_name = f"g{gnn_num_layers}_{gnn_hidden_channels}__m{mlp_num_layers}_{mlp_hidden_channels}__" \
                      f"{aggregation}_agg__{'clip' if clip_output is not None else 'no_clip'}__" \
                      f"lr_{lr}__" \
                      f"edge_direction_{edge_direction}__" \
                      f"{'use' if use_operations_hyperparameters else 'no'}_operations_hyperparameters"

    logger = TensorBoardLogger(
        save_dir=logdir,
        name=experiment_name,
    )
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", save_last=True, every_n_epochs=1)

    trainer = Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, ],
        log_every_n_steps=4,
        check_val_every_n_epoch=1,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    train_homogeneous_gcn(
        dataset_root=r"C:\Users\Konstantin\PycharmProjects\NIR\dataset\pipeline_dataset",
        logdir=r"C:\Users\Konstantin\PycharmProjects\NIR\experiments\homogeneous_gcn\pipeline_dataset",
        edge_direction="reversed",
        in_channels=15,
        out_channels=2,
        gnn_hidden_channels=8,
        gnn_num_layers=3,
        mlp_hidden_channels=8,
        mlp_num_layers=2,
        aggregation="sum",
        clip_output=None,  # (0., 1.),
        batch_size=1024,
        lr=1e-3,
        max_epochs=1000,
        use_operations_hyperparameters=False,
    )
