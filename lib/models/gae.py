# import torch
# import torch_geometric.nn as tgnn
# import torch.nn as nn
# from typing import Tuple
#
# class Encoder(nn.Module):
# class NodeLabelDecoder(nn.Module)
#     def __init__(self, in_features: int) -> None:
#         super().__init__()
# class NodeEdgeDecoder(nn.Module):
#     def __init__(self, node_decoder: nn.Module) -> None:
#         super().__init__()
#         self.edge_decoder = tgnn.models.InnerProductDecoder()
#         self.node_decoder = node_decoder
#
#     def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         node_data = self.node_decoder.forward(z)
#         edge_index = self.edge_decoder.forward_all(z)
#         return node_data, edge_index
# def gae():
#     return tgnn.models.GAE(
#         encoder=None,  # TODO
#         decoder=Decoder(),  # TODO
#     )