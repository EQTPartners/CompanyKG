"""
Copyright (C) eqtgroup.com Ltd 2023
https://github.com/EQTPartners/CompanyKG
License: MIT, https://github.com/EQTPartners/CompanyKG/LICENSE.md
"""

import logging

import GCL.augmentors as A
import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform

from ckg_benchmarks.gcl.base import BaseEncoder

logger = logging.getLogger(__name__)


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class MvgrlEncoder(BaseEncoder):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(MvgrlEncoder, self).__init__(
            {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
            }
        )

        aug1 = A.Identity()
        aug2 = A.PPRDiffusion(alpha=0.2)
        gconv1 = GConv(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers
        )
        gconv2 = GConv(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers
        )

        self.encoder1 = gconv1
        self.encoder2 = gconv2
        self.augmentor = (aug1, aug2)
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def from_hparams(hparams):
        return MvgrlEncoder(
            hparams["input_dim"],
            hparams["hidden_dim"],
            hparams["num_layers"],
        )

    @staticmethod
    def corruption(x, edge_index, edge_weight):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight

    def forward(self, x, edge_index, batch_size, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z1 = self.encoder1(x1, edge_index1, edge_weight1)[:batch_size]
        z2 = self.encoder2(x2, edge_index2, edge_weight2)[:batch_size]
        g1 = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)))
        g2 = self.project(torch.sigmoid(z2.mean(dim=0, keepdim=True)))
        z1n = self.encoder1(*self.corruption(x1, edge_index1, edge_weight1))[
            :batch_size
        ]
        z2n = self.encoder2(*self.corruption(x2, edge_index2, edge_weight2))[
            :batch_size
        ]
        return z1, z2, g1, g2, z1n, z2n

    def encode_batch(self, node_embeddings, edges, batch_size):
        # TODO Add edge weight here
        with torch.no_grad():
            vectors = self.encoder1(node_embeddings, edges)[:batch_size]
        return vectors.detach()
