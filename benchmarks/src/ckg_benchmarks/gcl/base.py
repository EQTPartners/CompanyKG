"""
Copyright (C) eqtgroup.com Ltd 2023
https://github.com/EQTPartners/CompanyKG
License: MIT, https://github.com/EQTPartners/CompanyKG/LICENSE.md
"""

import logging

import torch
from torch_geometric.loader import NeighborLoader

logger = logging.getLogger(__name__)


class BaseEncoder(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        # All encoders should have a num_layers param
        self.num_layers = self.hparams["num_layers"]

    def encode_batch(self, node_embeddings, edges, batch_size):
        """
        Pass node embeddings through the GNN to transform to a new
        set of embeddings.

        This processes a single batch. The node embeddings need
        to include all nodes in the subgraph relevant to encoding
        this batch and the edges define the subgraph for the batch.
        The batch_size is given to specify which nodes (at the
        beginning of node_embeddings) we want to get an encoding for.

        :param node_embeddings: Input embeddings for each node in the graph
        :param edges: Edges in the graph
        :param batch_size: Number of nodes we're getting embeddings for
        :return: New embeddings for each node as a PyTorch tensor
        """
        raise NotImplementedError("subclass should implement encode()")

    def encode(self, pyg_graph, batch_size=64, sample_edges=50, device=None):
        # We can't typically load a full graph into memory, so we fetch
        #  batches, making sure we provide the necessary level of neighbours
        #  for each included sample
        data_loader = NeighborLoader(
            pyg_graph,
            num_neighbors=[sample_edges] * self.num_layers,
            batch_size=batch_size,
        )
        encodings = []
        for batch in data_loader:
            if device is not None:
                batch.to(device)
            encodings.append(
                self.encode_batch(batch.x, batch.edge_index, batch_size).detach().cpu()
            )
        all_encodings = torch.cat(encodings)
        return all_encodings
