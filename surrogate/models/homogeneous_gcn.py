from typing import Dict, Any

import torch.nn.functional as F

from lib.models import HomogeneousGCN as HomogeneousGCN_
from .base_lightning_module import BaseLightningModule


class HomogeneousGCN(BaseLightningModule):

    def __init__(
            self,
            model_parameters: Dict[str, Any],
            loss_name: str,
            lr: float = 1e-3,
    ):
        super().__init__(
            HomogeneousGCN_(**model_parameters),
            getattr(F, loss_name),
            lr,
        )
