# TODO: add description and refactor

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import List, Tuple, Dict
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, n_input: int, n_hidden: List[int], n_output: int):
        super().__init__()
        assert len(n_hidden) > 0
        self.model = nn.Sequential()
        self.model.append(nn.Linear(n_input, n_hidden[0]))
        self.model.append(nn.ReLU())
        for i in range(1, len(n_hidden)):
            self.model.append(nn.Linear(n_hidden[i - 1], n_hidden[i]))
            self.model.append(nn.ReLU())
        self.model.append(nn.Linear(n_hidden[-1], n_output))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, n_input: int, n_hidden: List[int], n_output: int):
        super().__init__()
        assert len(n_hidden) > 0
        self.model = nn.Sequential()
        self.model.append(nn.Linear(n_input, n_hidden[0]))
        self.model.append(nn.ReLU())
        for i in range(1, len(n_hidden)):
            self.model.append(nn.Linear(n_hidden[i - 1], n_hidden[i]))
            self.model.append(nn.ReLU())
        self.model.append(nn.Linear(n_hidden[-1], n_output))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AutoEncoder(nn.Module):
    def __init__(self, n_input, n_hidden: List[int], embedding_dim: int = 8):
        super().__init__()
        self.encoder = Encoder(n_input, n_hidden, embedding_dim)
        self.decoder = Decoder(embedding_dim, n_hidden, n_input)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction


class JoinedAutoEncoders(pl.LightningModule):
    def __init__(self, autoencoders: Dict[str, nn.Module], learnable: Dict[str, nn.Parameter] = None):
        super().__init__()
        self.autoencoders = nn.ModuleDict(autoencoders)

        self.learnable = nn.ParameterDict()
        if learnable is not None:
            self.learnable = nn.ParameterDict(learnable)

    def reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(pred, target)
        return loss

    def embedding_simmilarity_loss(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        losses = []
        for i in range(len(embeddings)):
            anchor = embeddings[i]
            shuffled_indices = np.arange(len(anchor))
            np.random.shuffle(shuffled_indices)
            positives = anchor[shuffled_indices]
            for j in range(len(embeddings)):
                if j != i:
                    negatives = embeddings[j]
                    losses.append(F.triplet_margin_loss(anchor, positives, negatives))
        loss = torch.mean(torch.stack(losses))
        return loss

    def training_step(self, x: Dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        batch_size = x[list(x.keys())[0]].shape[0]

        embeddings = []
        reconstruction_losses = []
        for key in x:
            if key in self.autoencoders:
                data = x[key]
                embedding, reconstruction = self.autoencoders[key](data)
                reconstruction_losses.append(self.reconstruction_loss(reconstruction, data))
                embeddings.append(embedding)
            elif key in self.learnable:
                embeddings.append(self.learnable[key].repeat(batch_size, 1))
        reconstruction_loss = torch.mean(torch.stack(reconstruction_losses))
        embedding_simmilarity_loss = self.embedding_simmilarity_loss(embeddings)
        loss = reconstruction_loss + embedding_simmilarity_loss
        self.log("train/reconstruction_loss", reconstruction_loss)
        self.log("train/embedding_simmilarity_loss", embedding_simmilarity_loss)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, x: Dict[str, torch.Tensor], *args, **kwargs):
        with torch.no_grad():
            reconstruction_losses = []
            for key in x:
                if key in self.autoencoders:
                    data = x[key]
                    _, reconstruction = self.autoencoders[key](data)
                    reconstruction_losses.append(self.reconstruction_loss(reconstruction, data))
        reconstruction_loss = torch.mean(torch.stack(reconstruction_losses))
        self.log("val/reconstruction_loss", reconstruction_loss)

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer


class JoinedAutoencoderHyperparametersEmbedder:
    def __init__(
        self,
        ckpt_path: str,
        embedding_dim: int = 4,
        with_learnable: bool = True,
    ):
        N_HIDDEN = [8, 8]

        nodes_dimensions = {
            "bernb": 0,
            "dt": 3,
            "fast_ica": 2,
            "isolation_forest_class": 3,
            "knn": 3,
            "lgbm": 7,
            "logit": 1,
            "mlp": 0,
            "normalization": 0,
            "pca": 1,
            "poly_features": 2,
            "qda": 0,
            "resample": 3,
            "rf": 5,
            "scaling": 0,
        }

        operations = [
            "bernb",
            "dt",
            "fast_ica",
            "isolation_forest_class",
            "knn",
            "lgbm",
            "logit",
            "mlp",
            "normalization",
            "pca",
            "poly_features",
            "qda",
            "resample",
            "rf",
            "scaling",
        ]

        operations_with_hyperparams = [
            "dt",
            "fast_ica",
            "isolation_forest_class",
            "knn",
            "lgbm",
            "logit",
            "pca",
            "poly_features",
            "resample",
            "rf",
        ]

        operations_without_hyperparams = list(filter(lambda x: x not in operations_with_hyperparams, operations))

        autoencoders = {}
        for key in operations_with_hyperparams:
            n_input = nodes_dimensions[key]
            autoencoder = AutoEncoder(n_input, N_HIDDEN, embedding_dim)
            autoencoders[key] = autoencoder
        learnable = {}
        if with_learnable:
            for key in operations_without_hyperparams:
                learnable[key] = nn.Parameter(data=torch.rand(embedding_dim), requires_grad=True)

        self.model = JoinedAutoEncoders(autoencoders, learnable)
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt["state_dict"])

    def __call__(self, operation_name: str, hyperparameters_vec: np.ndarray) -> np.ndarray:
        hyperparameters_vec = torch.Tensor(hyperparameters_vec).to(self.model.device)
        with torch.no_grad():
            if operation_name in self.model.autoencoders:
                embedding, _ = self.model.autoencoders[operation_name](hyperparameters_vec)
            elif operation_name in self.model.learnable:
                batch_size = hyperparameters_vec.shape[0]
                embedding = self.model.learnable[operation_name].repeat(batch_size, 1)
            elif operation_name == "dataset":
                embedding = torch.Tensor([0.] * 4).reshape(1, -1)  # TODO: add `dataset` node to training of autoencoders
            else:
                raise ValueError(f"Unknown operation_name: {operation_name}")
        return embedding
