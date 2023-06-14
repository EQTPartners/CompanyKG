"""
Copyright (C) eqtgroup.com Ltd 2023
https://github.com/EQTPartners/CompanyKG
License: MIT, https://github.com/EQTPartners/CompanyKG/LICENSE.md
"""

import logging

import GCL.augmentors as A
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from ckg_benchmarks.gcl.base import BaseEncoder

logger = logging.getLogger(__name__)


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class GraceEncoder(BaseEncoder):
    def __init__(self, input_dim, hidden_dim, num_layers, proj_dim):
        super(GraceEncoder, self).__init__(
            {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "proj_dim": proj_dim,
            }
        )

        aug1 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
        aug2 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])

        gconv = GConv(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            activation=torch.nn.ReLU,
            num_layers=num_layers,
        )
        self.encoder = gconv
        self.augmentor = (aug1, aug2)

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    @staticmethod
    def from_hparams(hparams):
        return GraceEncoder(
            hparams["input_dim"],
            hparams["hidden_dim"],
            hparams["num_layers"],
            hparams["proj_dim"],
        )

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def encode_batch(self, node_embeddings, edges, batch_size):
        with torch.no_grad():
            vectors = self.encoder(node_embeddings, edges)[:batch_size]
        return vectors.detach()
