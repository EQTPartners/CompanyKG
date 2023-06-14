"""
Copyright (C) eqtgroup.com Ltd 2023
https://github.com/EQTPartners/CompanyKG
License: MIT, https://github.com/EQTPartners/CompanyKG/LICENSE.md
"""

import argparse
import logging
import math
import os
from random import randrange
from typing import Union

import dgl
import numpy as np
import torch

from ckg_benchmarks.base import BaseTrainer
from ckg_benchmarks.graphmae.model import GraphMAE
from ckg_benchmarks.utils import ranged_type

logger = logging.getLogger(__name__)


class GraphMAETrainer(BaseTrainer):
    training_method_name = "graphmae"

    def __init__(
            self,
            embedding_dim: int = 64,
            n_layer: int = 2,
            dropout_rate: float = 0.1,
            n_heads: int = 8,
            mask_rate: float = 0.5,
            drop_edge_rate: float = 0.5,
            learning_rate: float = 0.001,
            epochs: int = 500,
            n_lives: int = 100,
            disable_metis: bool = False,
            **kwargs
    ):
        self.disable_metis = disable_metis
        self.embedding_dim = embedding_dim
        self.n_layer = n_layer
        self.dropout_rate = dropout_rate
        self.n_heads = n_heads
        self.mask_rate = mask_rate
        self.drop_edge_rate = drop_edge_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.n_lives = n_lives

        super().__init__(**kwargs)

    def build_graph(self):
        """
        Build a DGL graph from the loaded CompanyKG graph
        and prepare it for training.

        """
        graph = self.comkg.get_dgl_graph(self.work_folder)[0]
        graph = dgl.add_reverse_edges(graph)
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
        return graph

    @property
    def hparams_str(self):
        return "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}".format(
            self.comkg.nodes_feature_type,
            self.epochs,
            self.n_layer,
            self.dropout_rate,
            self.learning_rate,
            self.embedding_dim,
            self.drop_edge_rate,
            self.mask_rate,
            self.n_heads,
            self.n_lives,
            self.seed,
        )

    def init_model(self, load=None):
        # Initialize model and training
        if load is None:
            # Initialize a new model
            model = GraphMAE(
                in_dim=self.comkg.nodes_feature_dim,
                num_hidden=self.embedding_dim,
                num_layers=self.n_layer,
                feat_drop=self.dropout_rate + 0.1,
                attn_drop=self.dropout_rate,
                nhead=self.n_heads,
                mask_rate=self.mask_rate,
                drop_edge_rate=self.drop_edge_rate,
            )
        else:
            # Load a pre-trained model
            model = torch.load(load)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        lr_schedule = lambda epoch: (1 + np.cos(epoch * np.pi / self.epochs)) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
        return model, optimizer, scheduler

    def inference_gpu(self) -> torch.Tensor:
        g = self.graph.to(self.device)
        self.model.eval()
        embed = self.model.embed(g, g.ndata["feat"]).cpu().detach().numpy()
        self.model.train()
        return embed / np.linalg.norm(embed, axis=1, keepdims=True)

    def inference_cpu(self) -> torch.Tensor:
        tmp_model_path = os.path.join(self.work_folder, "tmp_model.pth")
        torch.save(self.model, tmp_model_path)
        tmp_model = torch.load(tmp_model_path, map_location="cpu")
        tmp_model.eval()
        g = self.graph.to("cpu")
        embed = tmp_model.embed(g, g.ndata["feat"]).cpu().detach().numpy()
        return embed / np.linalg.norm(embed, axis=1, keepdims=True)

    def inference(self) -> torch.Tensor:
        try:
            return self.inference_gpu()
        except:
            logger.warning(
                f"GPU inference failed, fall back to CPU inference. Please be patient ..."
            )
            return self.inference_cpu()

    def training_loss(self, subgraph):
        loss, _ = self.model(subgraph, subgraph.ndata["feat"])
        return loss

    def train_epoch(
        self,
        epoch: int,
        train_dataloader: Union[dgl.dataloading.dataloader.DataLoader, list],
        metis: int = 1,
    ):
        self.model.train()
        n_step = len(train_dataloader)
        for step, subgraph in enumerate(train_dataloader):
            subgraph = subgraph.to(self.device)
            loss = self.training_loss(subgraph)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            if metis <= 1:
                logger.info(f"epoch {epoch}: loss={loss.item()}")
            else:
                logger.info(
                    f"epoch {epoch} | metis {metis} | step {step}/{n_step}: loss={loss.item()}"
                )

    def _train_model(self):
        # For high-dim nodes feature, we need to create Metis partitions
        if self.comkg.nodes_feature_type != "pause" and not self.disable_metis:
            logger.info("Using Metis partitions")
            n_metis = [5, 10, 15, 20, 30, 40, 50, 100, 300, 400, 600, 800, 1000]
            train_dataloaders = []

            for p in n_metis:
                sampler = dgl.dataloading.ClusterGCNSampler(
                    self.graph,
                    p,
                    cache_path=os.path.join(
                        self.work_folder, f"{self.comkg.nodes_feature_type}_metis_{p}.pkl"
                    ),
                )
                train_dataloader = dgl.dataloading.DataLoader(
                    graph=self.graph,
                    indices=torch.arange(p),
                    graph_sampler=sampler,
                    batch_size=math.ceil(p / 10),
                    shuffle=True,
                    drop_last=False,
                    num_workers=2,
                )
                train_dataloaders.append(train_dataloader)
            use_metis = True
        else:
            use_metis = False

        # Training Procedure
        best_sr_acc = 0
        n_lives = self.n_lives
        model_save_path = os.path.join(self.work_folder, f"{self.hparams_str}.pth")

        for epoch in range(self.epochs):
            logger.info(f"Starting epoch {epoch+1}/{self.epochs}")
            eval_gap = 1
            if use_metis:
                train_dataloader_idx = randrange(len(n_metis))
                current_n_metis = n_metis[train_dataloader_idx]
                train_dataloader = train_dataloaders[train_dataloader_idx]
                with train_dataloader.enable_cpu_affinity():
                    self.train_epoch(
                        epoch=epoch,
                        train_dataloader=train_dataloader,
                        metis=current_n_metis,
                    )
            else:
                eval_gap = 3
                self.train_epoch(
                    epoch=epoch,
                    train_dataloader=[self.graph],
                )

            # Training SR and SP evaluation
            if epoch % eval_gap == 0:
                # Inference
                embed = self.inference()

                # SR Eval on the validation set
                sr_acc = self.comkg.evaluate_sr(embed=embed, split="validation")
                logger.info(f"SR Accuracy: {sr_acc}, Lives left: {n_lives}")

                # SR task is prioritized here due to it being more challenging
                #  and having a defined validation set
                if sr_acc > best_sr_acc:
                    best_sr_acc = sr_acc
                    n_lives = self.n_lives

                    # Save the best-so-far trained model
                    torch.save(self.model, model_save_path)
                    logger.info(f"Model saved to {model_save_path}")
                else:
                    n_lives -= 1
                    if n_lives < 0:
                        break

        # Load the best model
        self.model = torch.load(model_save_path)
        logger.info(f"Best model loaded from {model_save_path}")


train_model = GraphMAETrainer.train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        default=500,
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
        default="./experiments/graphmae",
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
        help="The dimension of the embedding to be learned",
    )
    parser.add_argument(
        "--drop-edge-rate",
        default=0.5,
        type=ranged_type(float, 0.1, 0.9),
        help="The rate of edges to be dropped during training",
    )
    parser.add_argument(
        "--mask-rate",
        default=0.5,
        type=ranged_type(float, 0.1, 0.9),
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
    trainer = GraphMAETrainer(
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
