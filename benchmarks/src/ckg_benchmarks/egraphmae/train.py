"""
Copyright (C) eqtgroup.com Ltd 2023
https://github.com/EQTPartners/CompanyKG
License: MIT, https://github.com/EQTPartners/CompanyKG/LICENSE.md
"""

import argparse
import logging
import os

import dgl
import numpy as np
import torch

from ckg_benchmarks.egraphmae.model import EGraphMAE
from ckg_benchmarks.graphmae.train import GraphMAETrainer
from ckg_benchmarks.utils import ranged_type
from companykg.kg import CompanyKG

logger = logging.getLogger(__name__)


class EGraphMAETrainer(GraphMAETrainer):
    """
    Subclass the GraphMAETrainer to inherit most of the training
    procedure from it. We override certain parts here that are
    specific to eGraphMAE.

    """
    training_method_name = "egraphmae"

    def __init__(self, edge_hidden_dim=32, **kwargs):
        self.edge_hidden_dim = edge_hidden_dim
        super().__init__(**kwargs)

    def load_companykg(self):
        # Override to load with edge weights
        return CompanyKG(
            nodes_feature_type=self.nodes_feature_type,
            load_edges_weights=True,
            data_root_folder=self.data_root_folder,
        )

    def build_graph(self):
        graph = self.comkg.get_dgl_graph(self.work_folder)[0]
        graph.edata["weight"] = graph.edata["weight"] / np.linalg.norm(
            graph.edata["weight"], axis=1, keepdims=True
        )
        graph = dgl.add_reverse_edges(graph, copy_edata=True)
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph, fill_data=1.0)
        # The following code for creating a subgraph for SR evaluation that
        #  can be used in early stopping was in the original implementation,
        #  but wasn't being used.
        # I'm leaving it here so we can revive this efficiency trick if we
        #  need to later
        # But note that it happened before the add_reverse_edges, etc (which
        #  are then applied separately to the subgraph)
        """
        # Create a subgraph for evaluation: EGraphMAE requires significantly more memory
        # hence can only inference on a subgraph on CPU during training.
        sr_df = self.comkg.eval_tasks["sr"][1]
        sr_nids = list(
            set(sr_df.target_node_id.unique())
            .union(set(sr_df.candidate0_node_id.unique()))
            .union(set(sr_df.candidate1_node_id.unique()))
        )
        graph = self.comkg.get_dgl_graph(self.work_folder)[0]
        eval_g = dgl.merge(
            [
                dgl.khop_in_subgraph(graph, sr_nids, k=opts.n_layer)[0],
                dgl.khop_out_subgraph(graph, sr_nids, k=opts.n_layer)[0],
            ]
        )
        eval_g = dgl.add_reverse_edges(eval_g, copy_edata=True)
        eval_g = dgl.remove_self_loop(eval_g)
        eval_g = dgl.add_self_loop(eval_g, fill_data=1.0)
        """
        return graph

    @property
    def hparams_str(self):
        # We have one extra parameter
        return f"{super().hparams_str}_{self.edge_hidden_dim}"

    def inference(self) -> torch.Tensor:
        """
        eGraphMAE inference at the moment only supports CPU inference, so doesn't
        pay attention to self.device.

        """
        # Due to memory limitation we have to save and load to cpu for inference
        tmp_model_path = os.path.join(self.work_folder, "tmp_model.pth")
        torch.save(self.model, tmp_model_path)
        tmp_model = torch.load(tmp_model_path, map_location="cpu")
        tmp_model.eval()
        # Note that embed will return both node and edge embedding
        embed_dense = (
            tmp_model.embed(
                self.graph, self.graph.ndata["feat"], self.graph.edata["weight"]
            )[0].cpu().detach()
        )
        embed_dense / np.linalg.norm(embed_dense, axis=1, keepdims=True)
        if embed_dense.shape[0] == self.comkg.n_nodes:
            return embed_dense.numpy()
        else:
            # The evaluation logic requires full embedding matrix
            embed = torch.zeros(self.comkg.n_nodes, self.embedding_dim)
            for idx, x in enumerate(self.graph.ndata["_ID"].tolist()):
                embed[x, :] = embed_dense[idx, :]
            return embed.numpy()

    def init_model(self, load=None):
        # Initialize model and training
        if load is None:
            # Initialize a new model
            model = EGraphMAE(
                in_dim=self.comkg.nodes_feature_dim,
                num_hidden=self.embedding_dim,
                num_layers=self.n_layer,
                feat_drop=self.dropout_rate + 0.1,
                attn_drop=self.dropout_rate,
                nhead=self.n_heads,
                mask_rate=self.mask_rate,
                drop_edge_rate=self.drop_edge_rate,
                num_edge_features=self.comkg.edges_weight_dim,
                num_edge_hidden=self.edge_hidden_dim,
            )
        else:
            # Load a pre-trained model
            model = torch.load(load)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        lr_schedule = lambda epoch: (1 + np.cos(epoch * np.pi / self.epochs)) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
        return model, optimizer, scheduler

    def training_loss(self, subgraph):
        loss, _ = self.model(subgraph, subgraph.ndata["feat"], subgraph.edata["weight"])
        return loss


train_model = EGraphMAETrainer.train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        default=1000,
        type=ranged_type(int, 1, 2000),
        help="The max number of training epochs",
    )
    parser.add_argument(
        "--nodes-feature-type",
        type=str,
        default="msbert",
        choices=["msbert", "ada2", "simcse", "pause"],
        help="The type of nodes feature: msbert, ada2, simcse or pause",
    )
    parser.add_argument(
        "--device",
        default=0,
        type=int,
        help="The device used to carry out the training: use cpu when less than 0",
    )
    parser.add_argument(
        "--data-root-folder",
        default="./data",
        type=str,
        help="The root folder where the CompanyKG data is downloaded to",
    )
    parser.add_argument(
        "--finetune-from",
        type=str,
        help="The saved model to be finetuned from, ex. ./msbert_1000_2_0.01_0.001_256_0.1_0.1_8_100_42.pth",
    )
    parser.add_argument(
        "--work-folder",
        default="./experiments/egraphmae",
        type=str,
        help="The working folder where models and logs are saved to",
    )
    parser.add_argument(
        "--n-layer",
        default=2,
        type=ranged_type(int, 2, 4),
        help="The number of GNN layers",
    )
    parser.add_argument(
        "--dropout-rate",
        default=0.1,
        type=ranged_type(float, 0.0, 0.3),
        help="The feature dropout rate of GNN layers",
    )
    parser.add_argument(
        "--learning-rate",
        default=0.001,
        type=ranged_type(float, 0.00001, 0.01),
        help="The training learning rate",
    )
    parser.add_argument(
        "--embedding-dim",
        default=64,
        type=ranged_type(int, 8, 512),
        help="The dimension of the node embedding to be learned",
    )
    parser.add_argument(
        "--drop-edge-rate",
        default=0.5,
        type=ranged_type(float, 0.1, 0.8),
        help="The rate of edges to be dropped during training",
    )
    parser.add_argument(
        "--mask-rate",
        default=0.5,
        type=ranged_type(float, 0.1, 0.8),
        help="The rate of nodes feature to be masked during training",
    )
    parser.add_argument(
        "--n-heads",
        default=8,
        type=ranged_type(int, 1, 8),
        help="The number of attention heads",
    )
    parser.add_argument(
        "--n-lives",
        default=100,
        type=ranged_type(int, 10, 200),
        help="The number of training epochs allowed when evaluation metrics are not improving",
    )
    parser.add_argument(
        "--edge-hidden-dim",
        default=32,
        type=ranged_type(int, 4, 64),
        help="The dimension of the edge embedding",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="The seed used to run the experiment",
    )
    parser.add_argument(
        "--disable-metis",
        action="store_true",
        help="Force trainer not to use Metis partitions, even when the embedding size "
             "is large"
    )
    opts = parser.parse_args()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    # Create trainer and run
    logger.info("Initializing trainer")
    trainer = EGraphMAETrainer(
        edge_hidden_dim=opts.edge_hidden_dim,
        nodes_feature_type=opts.nodes_feature_type,
        data_root_folder=opts.data_root_folder,
        embedding_dim=opts.embedding_dim,
        n_layer=opts.n_layer,
        dropout_rate=opts.dropout_rate,
        n_heads=opts.n_heads,
        mask_rate=opts.mask_rate,
        drop_edge_rate=opts.drop_edge_rate,
        learning_rate=opts.learning_rate,
        epochs=opts.epochs,
        n_lives=opts.n_lives,
        seed=opts.seed,
        device=opts.device,
        finetune_from=opts.finetune_from,
        work_folder=opts.work_folder,
        disable_metis=opts.disable_metis,
    )

    logger.info("Starting model training")
    trainer.train_model()
    logger.info("Training complete")
