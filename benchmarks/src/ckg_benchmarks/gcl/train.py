"""
Copyright (C) eqtgroup.com Ltd 2023
https://github.com/EQTPartners/CompanyKG
License: MIT, https://github.com/EQTPartners/CompanyKG/LICENSE.md
"""

import argparse
import logging

import GCL.losses as L
import torch
from GCL.models import DualBranchContrast
from torch.optim import Adam
from torch_geometric.loader import NeighborLoader
from tqdm.auto import tqdm

from ckg_benchmarks.base import BaseTrainer
from ckg_benchmarks.gcl.grace import GraceEncoder
from ckg_benchmarks.gcl.mvgrl import MvgrlEncoder
from ckg_benchmarks.utils import ranged_type
from companykg import CompanyKG

logger = logging.getLogger(__name__)


def grace_train_step(encoder_model, contrast_model, batch, optimizer, batch_size):
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(batch.x, batch.edge_index)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    h1 = h1[:batch_size]
    h2 = h2[:batch_size]
    loss = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()
    return loss.item()


def mvgrl_train_step(encoder_model, contrast_model, data, optimizer, batch_size):
    optimizer.zero_grad()
    z1, z2, g1, g2, z1n, z2n = encoder_model(data.x, data.edge_index, batch_size)
    loss = contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, h3=z1n, h4=z2n)
    loss.backward()
    optimizer.step()
    return loss.item()


class GclTrainer(BaseTrainer):
    def __init__(
        self,
        method: str = "grace",
        edge_weights: bool = False,
        embedding_dim: int = 64,
        n_layer: int = 2,
        sampler_edges: int = 5,
        batch_size: int = 16,
        learning_rate: float = None,
        epochs: int = 500,
        **kwargs,
    ):
        self.method = method
        if self.method not in ["grace", "mvgrl"]:
            raise ValueError("training method should be 'grace' or 'mvgrl'")
        self.training_method_name = self.method

        if "finetune_from" in kwargs:
            raise ValueError(
                "'finetune_from' is not currently supported for GCL methods"
            )
        self.edge_weights = edge_weights
        self.batch_size = batch_size
        self.sampler_edges = sampler_edges
        self.embedding_dim = embedding_dim
        self.n_layer = n_layer
        self.epochs = epochs

        # This will be set by the model init
        self.contrast_model = None

        if learning_rate is None:
            # Use different defaults for GRACE and MVGRL
            learning_rate = 0.01 if self.method == "mvgrl" else 0.001
        self.learning_rate = learning_rate

        if self.method == "grace":
            self.train_step = grace_train_step
        else:
            self.train_step = mvgrl_train_step

        super().__init__(**kwargs)

    def load_companykg(self):
        """
        Load CompanyKG dataset in preparation for training

        """
        return CompanyKG(
            nodes_feature_type=self.nodes_feature_type,
            load_edges_weights=self.edge_weights,
            data_root_folder=self.data_root_folder,
        )

    @property
    def hparams_str(self):
        return "_".join(
            str(x)
            for x in [
                self.comkg.nodes_feature_type,
                self.epochs,
                self.n_layer,
                self.embedding_dim,
                self.sampler_edges,
                self.batch_size,
                self.seed,
            ]
        )

    def build_graph(self):
        # Convert to PyG graph
        return self.comkg.to_pyg()

    def init_model(self, load=None):
        if self.method == "mvgrl":
            model = MvgrlEncoder(
                self.comkg.nodes_feature_dim, self.embedding_dim, self.n_layer
            )
            self.contrast_model = DualBranchContrast(loss=L.JSD(), mode="G2L")
        else:
            model = GraceEncoder(
                self.comkg.nodes_feature_dim,
                self.embedding_dim,
                self.n_layer,
                self.embedding_dim,
            )
            self.contrast_model = DualBranchContrast(
                loss=L.InfoNCE(tau=0.2), mode="L2L", intraview_negs=True
            )

        optimizer = Adam(model.parameters(), lr=self.learning_rate)
        return model, optimizer, None

    def _train_model(self):
        data_loader = NeighborLoader(
            self.graph,
            num_neighbors=[self.sampler_edges] * self.n_layer,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.model.train()
        for epoch in range(self.epochs):
            logger.info(f"Starting epoch {epoch + 1}")
            with tqdm(
                total=len(data_loader),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                disable=None,
            ) as pbar:
                epoch_loss = 0
                for bnum, batch in enumerate(data_loader):
                    batch.to(self.device)
                    loss = self.train_step(
                        self.model,
                        self.contrast_model,
                        batch,
                        self.optimizer,
                        self.batch_size,
                    )
                    epoch_loss += loss

                    pbar.set_postfix({"loss": epoch_loss / (bnum + 1)})
                    pbar.update()
            logger.info(f"Epoch {epoch+1} loss: {epoch_loss}")

    def inference(self) -> torch.Tensor:
        self.model.to(self.device)
        return self.model.encode(
            self.graph, batch_size=self.batch_size, device=self.device
        )


# Alias for initializing and training a model
train_model = GclTrainer.train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        default="grace",
        choices=["grace", "mvgrl"],
        help="The training method to use: grace or mvgrl",
    )
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
        "--work-folder",
        default="./experiments",
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
        "--learning-rate",
        type=ranged_type(float, 0.00001, 0.01),
        help="The training learning rate. Defaults: 0.01 (MVGRL), 0.001 (GRACE)",
    )
    parser.add_argument(
        "--embedding-dim",
        default=64,
        type=ranged_type(int, 8, 512),
        help="The dimension of the node embedding to be learned",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="The seed used to run the experiment",
    )
    parser.add_argument(
        "--batch-size",
        default=16,
        type=int,
        help="Batch size to use for training and inference",
    )
    parser.add_argument(
        "--sampler-edges",
        default=5,
        type=int,
        help="Number of neighbors to sample to build training batches",
    )
    parser.add_argument(
        "--edge-weights",
        action="store_true",
        help="Load edge weights for training",
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
    trainer = GclTrainer(
        nodes_feature_type=opts.nodes_feature_type,
        data_root_folder=opts.data_root_folder,
        method=opts.method,
        edge_weights=opts.edge_weights,
        embedding_dim=opts.embedding_dim,
        n_layer=opts.n_layer,
        sampler_edges=opts.sampler_edges,
        batch_size=opts.batch_size,
        learning_rate=opts.learning_rate,
        epochs=opts.epochs,
        seed=opts.seed,
        device=opts.device,
        work_folder=opts.work_folder,
    )

    logger.info("Starting model training")
    trainer.train_model()
    logger.info("Training complete")
