"""
Copyright (C) eqtgroup.com Ltd 2023
https://github.com/EQTPartners/CompanyKG
License: MIT, https://github.com/EQTPartners/CompanyKG/LICENSE.md
"""

import dgl
import torch


class GraphSAGE(torch.nn.Module):
    """The GraphSAGE model class"""

    def __init__(
        self, n_layer: int, in_feats: int, h_feats: int, dropout: float
    ) -> None:
        """Initializer of GraphSAGE model.

        Args:
            n_layer (int): the number of GNN layers.
            in_feats (int): the dimension of input feature.
            h_feats (int): the dimension of the graph embedding to be trained.
            dropout (float): the drop out rate of GNN layers.
        """
        super(GraphSAGE, self).__init__()
        self.n_layer = n_layer
        self.h_feats = h_feats
        self.gcn_layers = torch.nn.ModuleList()
        for i in range(n_layer):
            if i == 0:
                self.gcn_layers.append(
                    dgl.nn.SAGEConv(
                        in_feats, h_feats, aggregator_type="gcn", feat_drop=dropout
                    )
                )
            elif i == n_layer - 1:
                self.gcn_layers.append(
                    dgl.nn.SAGEConv(h_feats, h_feats, aggregator_type="gcn")
                )
            else:
                self.gcn_layers.append(
                    dgl.nn.SAGEConv(
                        h_feats, h_feats, aggregator_type="gcn", feat_drop=dropout
                    )
                )

    def forward(self, mfgs: list, x: torch.Tensor) -> torch.Tensor:
        """The forward propagation function of the model.

        Args:
            mfgs (list): the Message-passing Flow Graphs (MFGs).
            x (torch.Tensor): the input feature tensor.

        Returns:
            torch.Tensor: the output tensor of the forward pass.
        """
        for i in range(self.n_layer):
            if i == 0:
                h_dst = x[: mfgs[i].num_dst_nodes()]
                h = self.gcn_layers[i](mfgs[i], (x, h_dst))
                h = torch.nn.functional.leaky_relu(h)
            elif i == self.n_layer - 1:
                h_dst = h[: mfgs[i].num_dst_nodes()]
                h = self.gcn_layers[i](mfgs[i], (h, h_dst))
            else:
                h_dst = h[: mfgs[i].num_dst_nodes()]
                h = self.gcn_layers[i](mfgs[i], (h, h_dst))
                h = torch.nn.functional.leaky_relu(h)
        return h


class DotPredictor(torch.nn.Module):
    """A pairwise predictor implemented with dot product"""

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        """The forward pass of DotPredictor.

        Args:
            g (dgl.DGLGraph): the input graph.
            h (torch.Tensor): the input node feature.

        Returns:
            torch.Tensor: the output scores.
        """
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(dgl.function.u_dot_v("h", "h", "score"))
            return g.edata["score"][:, 0]
