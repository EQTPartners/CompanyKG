"""
Copyright (C) eqtgroup.com Ltd 2023
https://github.com/EQTPartners/CompanyKG
License: MIT, https://github.com/EQTPartners/CompanyKG/LICENSE.md
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional

import torch

from ckg_benchmarks.utils import set_random_seed
from companykg import CompanyKG

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    training_method_name = "unknown"

    def __init__(
        self,
        nodes_feature_type: str = "msbert",
        data_root_folder: str = "./data",
        seed: int = 42,
        device: Union[int, str] = 0,
        work_folder: str = "./experiments",
        finetune_from: Optional[str] = None,
        allow_retrain: bool = False,
    ):
        self.allow_retrain = allow_retrain
        self.nodes_feature_type = nodes_feature_type
        self.data_root_folder = data_root_folder
        self.seed = seed
        self.work_folder = Path(work_folder) / self.training_method_name
        self.device = device if device >= 0 else "cpu"
        self.finetune_from = finetune_from

        # File path to which the final model will be saved
        self.eval_results_path = self.work_folder / f"{self.hparams_str}.pkl"
        self.embedding_save_path = self.work_folder / f"{self.hparams_str}.pt"

        # Create CompanyKG object
        self.comkg = self.load_companykg()
        self.comkg.describe()

        # Create DGL graph
        self.graph = self.build_graph()
        logger.info(self.graph)

        self.model, self.optimizer, self.scheduler = self.init_model(
            load=self.finetune_from
        )

        # Set at the end of training based on the final model
        self.embeddings = None

    def load_companykg(self):
        """
        Load CompanyKG dataset in preparation for training

        """
        return CompanyKG(
            nodes_feature_type=self.nodes_feature_type,
            load_edges_weights=False,
            data_root_folder=self.data_root_folder,
        )

    @property
    @abstractmethod
    def hparams_str(self):
        ...

    @abstractmethod
    def build_graph(self):
        ...

    @abstractmethod
    def init_model(self, load=None):
        ...

    @abstractmethod
    def _train_model(self):
        ...

    @abstractmethod
    def inference(self) -> torch.Tensor:
        ...

    def train_model(self):
        set_random_seed(self.seed)
        self.model.to(self.device)

        # Check if the current trial has been run already
        # Allow this check to be overridden
        if not self.allow_retrain and self.eval_results_path.exists():
            logger.info(f"Skip trial {self.hparams_str}: has already been run")
            return

        # Logging: send a copy to the output folder
        os.makedirs(self.work_folder, exist_ok=True)
        log_path = os.path.join(self.work_folder, f"{self.hparams_str}.log")
        logger.info(f"Sending training logs to {log_path}")
        file_handler = logging.FileHandler(log_path, mode="a")
        log_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-5.5s]  %(message)s"
        )
        file_handler.setFormatter(log_formatter)
        # Add handler to the root logger
        logging.getLogger().addHandler(file_handler)

        logger.info("Strating model training")
        self._train_model()
        logger.info("Model training complete")

        # Inference with best model
        logger.info("Projecting full KG using final model")
        embed = self.inference()
        torch.save(embed, self.embedding_save_path)
        logger.info(f"Best embeddings saved to {self.embedding_save_path}")
        # Keep the embeddings for later use
        self.embeddings = embed
        # You can evaluate these using:
        #  results = trainer.comkg.evaluate(embed=trainer.embeddings, silent=True)

    def evaluate(self, silent=False):
        if self.embeddings is None:
            raise RuntimeError(
                "projected embeddings are not available for evaluation: training must be run first"
            )
        results = self.comkg.evaluate(embed=self.embeddings, silent=silent)
        return results

    @classmethod
    def train(cls, **kwargs):
        """
        Initialize a trainer and run the training routine, returning the trainer.
        This also provides the trained model as `trainer.model`.

        :return: trainer instance
        """
        logger.info("Initializing model and trainer")
        trainer = cls(**kwargs)
        logger.info("Starting model training")
        trainer.train_model()
        logger.info("Model training complete")
        return trainer
