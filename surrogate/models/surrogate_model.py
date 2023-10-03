from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from surrogate.encoders import GraphTransformer, MLPDatasetEncoder


class SurrogateModel(LightningModule):
    def __init__(
            self,
            model_parameters: Dict[str, Any],  # loss_name: str,
            lr: float = 1e-3,
            warmup_steps: int = None,
            abs_pe: Any = None,
    ):
        """loss_name: loss name from torch.nn.functional"""
        super().__init__()

        self.pipeline_encoder = GraphTransformer(**{k: v for k, v in model_parameters.items() if k != "name"})
        self.dataset_encoder = MLPDatasetEncoder(model_parameters['dim_dataset'],
                                                 model_parameters['meta_data'],
                                                 hidden_dim=model_parameters['d_model'],
                                                 output_dim=model_parameters['d_model'])

        self.final_model = nn.Sequential(nn.BatchNorm1d(model_parameters['d_model'] * 2),
                                         nn.Linear(model_parameters['d_model'] * 2, model_parameters['d_model'] * 2),
                                         nn.ReLU(),
                                         nn.Linear(model_parameters['d_model'] * 2, 1))

        self.loss = nn.MSELoss()  # getattr(F, loss_name)
        self.lr = lr
        self.abs_pe = abs_pe
        self.warmup_steps = warmup_steps

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        z_pipeline = self.pipeline_encoder(data)
        z_dataset = self.dataset_encoder(data.d)

        return self.final_model(torch.cat((z_pipeline, z_dataset), 1))

    def training_step(self, batch, batch_idx):
        # sign flip as in Bresson et al. for laplacian PE
        if self.abs_pe == 'lap':
            sign_flip = torch.rand(batch.abs_pe.shape[-1])
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            batch.abs_pe = batch.abs_pe * sign_flip.unsqueeze(0)

        pred = torch.squeeze(self.forward(batch))

        loss = self.loss(pred, batch.y)
        self.log("train_loss", loss, batch_size=batch.y.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        pred = torch.squeeze(self.forward(batch))
        loss = self.loss(pred, batch.y)
        mse_loss = F.mse_loss(pred, batch.y)
        mae_loss = F.l1_loss(pred, batch.y)
        self.log("val_loss", loss, batch_size=batch.y.shape[0])
        self.log("val_mse_loss", mse_loss, batch_size=batch.y.shape[0])
        self.log("val_mae_loss", mae_loss, batch_size=batch.y.shape[0])
        return loss

    # TODO: Egor shoud fix it
    def test_step(self, batch, batch_idx):
        pred = torch.squeeze(self.forward(batch))
        ndcg = 1  # ndcg_fixed(batch.y.cpu().numpy(), pred.cpu().numpy())

        self.log("test_ndcg", ndcg, batch_size=batch.y.shape[0])
        return loss

    # TODO: oprimizewr config + scheduler config
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(list(self.pipeline_encoder.parameters()) +
                                      list(self.dataset_encoder.parameters()) +
                                      list(self.final_model.parameters()),
                                      lr=self.lr,
                                      weight_decay=1e-5)
        return optimizer

        # optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        # if self.warmup_steps is not None:
        #     return [optimizer, ]
        # else:
        #     lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #         optimizer,
        #         mode='min',
        #         factor=0.5,
        #         patience=15,
        #         min_lr=1e-05,
        #         verbose=False,
        #     )
        #     return [optimizer, ], [lr_scheduler, ]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None) -> None:
        if self.warmup_steps is not None:
            if self.trainer.global_step < self.warmup_steps:
                lr_steps = (self.lr - 1e-6) / self.warmup_steps
                lr = 1e-6 + self.trainer.global_step * lr_steps
            else:
                decay_factor = self.lr * self.warmup_steps ** .5
                lr = decay_factor * self.trainer.global_step ** -.5
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

# class WarmupLR:
#     def __init__(self, optimizer: Optimizer, target_lr: float, warmup_steps: int) -> None:
#         self.optimizer = optimizer
#         self.target_lr = target_lr
#         self.step_idx = 0
#         self.warmup_steps = warmup_steps
#         self.lr_steps = (self.target_lr - 1e-6) / self.warmup_steps
#
#     def state_dict(self):
#         """Returns the state of the scheduler as a :class:`dict`.
#
#         It contains an entry for every variable in self.__dict__ which
#         is not the optimizer.
#         """
#         return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
#
#     def load_state_dict(self, state_dict):
#         """Loads the schedulers state.
#
#         Args:
#             state_dict (dict): scheduler state. Should be an object returned
#                 from a call to :meth:`state_dict`.
#         """
#         self.__dict__.update(state_dict)
#
#     def step(self):
#         self.step_idx += 1
#         if self.step_idx < self.warmup_steps:
#             lr = 1e-6 + self.step_idx * self.lr_steps
#         else:
#             lr = self.decay_factor * self.step_idx ** -.5
#
#         for param_group in self.optimizer.param_groups:
#             param_group["lr"] = lr
