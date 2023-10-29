from torch_geometric.nn import MLP


class DatasetEncoder(MLP):
    def __init__(
        self,
        in_size: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        act: str = "relu",
        norm: str = None,
    ):
        super().__init__(
            in_channels=in_size,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            act=act,
            norm=norm,
        )
        self.out_dim = hidden_dim
