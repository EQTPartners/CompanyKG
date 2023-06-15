"""
Copyright (C) eqtgroup.com Ltd 2023
https://github.com/EQTPartners/CompanyKG
License: MIT, https://github.com/EQTPartners/CompanyKG/LICENSE.md
"""

import argparse
import logging
import os

import dgl
import torch

from ckg_benchmarks.base import BaseTrainer
from ckg_benchmarks.graphsage.model import GraphSAGE, DotPredictor
from ckg_benchmarks.utils import ranged_type

logger = logging.getLogger(__name__)


class GraphSageTrainer(BaseTrainer):
    training_method_name = "graphsage"

    def __init__(
            self,
            embedding_dim: int = 64,
            n_layer: int = 2,
            dropout_rate: float = 0.1,
            learning_rate: float = 0.001,
            epochs: int = 2,
            train_batch_size: int = 2048,
            inference_batch_size: int = 2048,
            n_sample_neighbor: int = 8,
            **kwargs
    ):
        if "finetune_from" in kwargs:
            raise ValueError("'finetune_from' is not currently supported for GCL methods")
        self.embedding_dim = embedding_dim
        self.n_layer = n_layer
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.inference_batch_size = inference_batch_size
        self.n_sample_neighbor = n_sample_neighbor
        # This will get set by the model init
        self.predictor = None

        super().__init__(**kwargs)

        # Create training sampler, also used for inference
        self.negative_sampler = dgl.dataloading.negative_sampler.Uniform(1)
        self.sampler = dgl.dataloading.NeighborSampler(
            [self.n_sample_neighbor for _ in range(self.n_layer)]
        )

    @property
    def hparams_str(self):
        return "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}".format(
            self.nodes_feature_type,
            self.epochs,
            self.n_sample_neighbor,
            self.train_batch_size,
            self.n_layer,
            self.dropout_rate,
            self.embedding_dim,
            self.learning_rate,
            self.seed,
        )

    def build_graph(self):
        # Create DGL graph
        graph = self.comkg.get_dgl_graph(self.work_folder)[0]
        graph = dgl.add_reverse_edges(graph)
        return graph

    def init_model(self, load=None):
        # Create the model
        model = GraphSAGE(
            n_layer=self.n_layer,
            in_feats=self.comkg.nodes_feature_dim,
            h_feats=self.embedding_dim,
            dropout=self.dropout_rate,
        )
        logger.info(model)
        predictor = DotPredictor()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()), lr=self.learning_rate
        )
        self.predictor = predictor
        # We don't use a scheduler, so just return None
        return model, optimizer, None

    def _train_model(self):
        train_dataloader = dgl.dataloading.DataLoader(
            self.graph,
            torch.arange(self.graph.number_of_edges()),
            dgl.dataloading.as_edge_prediction_sampler(
                self.sampler, negative_sampler=self.negative_sampler
            ),
            device=self.device,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=2,
        )
        total_steps = len(train_dataloader)

        self.predictor.to(self.device)

        # Start training loop
        for epoch in range(self.epochs):
            logger.info(f"Starting epoch {epoch+1}/{self.epochs}")
            self.model.train()
            for step, (_, pos_graph, neg_graph, mfgs) in enumerate(train_dataloader):
                inputs = mfgs[0].srcdata["feat"]
                outputs = self.model(mfgs, inputs)
                pos_score = self.predictor(pos_graph, outputs)
                neg_score = self.predictor(neg_graph, outputs)
                score = torch.cat([pos_score, neg_score])
                label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(score, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                logger.info(f"{step}/{total_steps} of epoch {epoch}: loss={loss.item()}")

            # Save the trained model
            epoch_snapshot_name = f"{self.hparams_str}_e{epoch}"
            model_save_path = os.path.join(self.work_folder, f"{epoch_snapshot_name}.pth")
            torch.save(self.model, model_save_path)
            logger.info(f"Model saved to {model_save_path}")

    def inference(self) -> torch.Tensor:
        """The inference (prediction) function for GraphSAGE model.

        Args:
            model (GraphSAGE): a GraphSAGE model.
            g (dgl.DGLGraph): the input graph to be predicted.
            sampler (dgl.dataloading.neighbor_sampler.NeighborSampler): a sampler that
                should be exactly the same as the one used in training.
            batch_size (int): the batch size for inference mini-batches.
            device (str): the device (cuda or cpu) to run the inference.

        Returns:
            torch.Tensor: the predicted node embeddings.
        """
        self.model.eval()
        with torch.no_grad():
            _dataloader = dgl.dataloading.DataLoader(
                self.graph,
                torch.arange(self.graph.number_of_nodes()),
                self.sampler,
                batch_size=self.inference_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,  # num_workers > 0 will cause evaluation randomness
                device=self.device,
            )
            result = []
            for _, _, mfgs in _dataloader:
                inputs = mfgs[0].srcdata["feat"]
                result.append(self.model(mfgs, inputs))
            return torch.cat(result)


train_model = GraphSageTrainer.train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        default=2,
        type=ranged_type(int, 1, 100),
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
        "--n-sample-neighbor",
        default=8,
        type=ranged_type(int, 2, 64),
        help="The number of neighbor to be sampled",
    )
    parser.add_argument(
        "--train-batch-size",
        default=2048,
        type=ranged_type(int, 16, 2**15),
        help="The number of samples in each training mini-batch",
    )
    parser.add_argument(
        "--inference-batch-size",
        default=2048,
        type=ranged_type(int, 16, 2**15),
        help="The number of samples in each inference mini-batch",
    )
    parser.add_argument(
        "--n-layer",
        default=2,
        type=ranged_type(int, 2, 4),
        choices=range(2, 4),
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
        default=128,
        type=ranged_type(int, 8, 1024),
        help="The dimension of the embedding to be learned",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="The seed used to run the experiment",
    )
    opts = parser.parse_args()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    logger.info("Initializing trainer")
    trainer = GraphSageTrainer(
        nodes_feature_type=opts.nodes_feature_type,
        data_root_folder=opts.data_root_folder,
        embedding_dim=opts.embedding_dim,
        n_layer=opts.n_layer,
        dropout_rate=opts.dropout_rate,
        learning_rate=opts.learning_rate,
        epochs=opts.epochs,
        seed=opts.seed,
        device=opts.device,
        work_folder=opts.work_folder,
        train_batch_size=opts.train_batch_size,
        inference_batch_size=opts.inference_batch_size,
        n_sample_neighbor=opts.n_sample_neighbor,
    )

    logger.info("Starting model training")
    trainer.train_model()
    logger.info("Training complete")
