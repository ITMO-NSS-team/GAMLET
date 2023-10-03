# -*- coding: utf-8 -*-
from torch import nn


class MLPDatasetEncoder(nn.Module):
    def __init__(self, input_dim, dict_category,
                 hidden_dim=128, output_dim=64,
                 dropout_in=0.4, dropout=0.2):
        super().__init__()
        
        self.inp_layer = nn.Sequential(nn.BatchNorm1d(input_dim),
                                       nn.Dropout(p=dropout_in),
                                       nn.Linear(input_dim, hidden_dim))

        if dict_category:
            self.emb_layers = nn.ModuleList([nn.Embedding(x, hidden_dim)
                                             for x in dict_category.values()])
        self.n_cat = len(dict_category)

        self.block1 = nn.Sequential(nn.BatchNorm1d(hidden_dim),
                                    nn.Dropout(p=dropout),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU())

        self.block2 = nn.Sequential(nn.BatchNorm1d(hidden_dim),
                                    nn.Linear(hidden_dim, output_dim))

    def forward(self, x_cont):
        z = self.inp_layer(x_cont)
        # for i in range(self.n_cat):
        #     z += self.emb_layers[i](x_cont[:, i])  # TODO: fix!!!!!!!!!!!!!!!!!

        z = self.block1(z)
        z = self.block2(z)
        return z
