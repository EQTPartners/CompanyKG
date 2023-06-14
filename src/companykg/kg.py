"""
Copyright (C) eqtgroup.com Ltd 2023
https://github.com/EQTPartners/CompanyKG
License: MIT, https://github.com/EQTPartners/CompanyKG/LICENSE.md
"""

import logging
import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from scipy import spatial
from sklearn.metrics import roc_auc_score, accuracy_score

from companykg.settings import (
    EDGES_FILENAME,
    ZENODO_DATASET_BASE_URI,
    EDGES_WEIGHTS_FILENAME,
    NODES_FEATURES_FILENAME_TEMPLATE,
    EVAL_TASK_FILENAME_TEMPLATE,
)
from companykg.utils import download_zenodo_file

logger = logging.getLogger(__name__)


class CompanyKG:
    """The CompanyKG class that provides utility functions
    to load data and carry out evaluations.
    """

    def __init__(
        self,
        nodes_feature_type: str = "msbert",
        load_edges_weights: bool = False,
        data_root_folder: str = "./data",
    ) -> None:
        """Initialize a CompanyKG object.

        Args:
            nodes_feature_type (str, optional): the desired note feature type.
                Viable values include "msbert", "pause", "simcse", "ada2". Defaults to "msbert".
            load_edges_weights (bool, optional): load edge weights or not. Defaults to False.
            data_root_folder (str, optional): root folder of downloaded data. Defaults to "./data".
                If the folder does not exist, the latest version of the dataset will be downloaded from
                Zenodo.
        """

        self.data_root_folder = data_root_folder

        # Load nodes feature: only load one type
        self.nodes_feature_type = nodes_feature_type

        # Create a local data directory - NOP if directory already exists
        os.makedirs(data_root_folder, exist_ok=True)

        # Load edges
        # First check if edges file exists - download if it doesn't
        self.edges_file = os.path.join(data_root_folder, EDGES_FILENAME)
        if not os.path.exists(self.edges_file):
            download_zenodo_file(
                os.path.join(ZENODO_DATASET_BASE_URI, EDGES_FILENAME),
                self.edges_file,
            )
        self.edges = torch.load(self.edges_file)
        logger.info(f"[DONE] Loaded {self.edges_file}")

        # Load edge weights [Optional]
        # First check if edge weights file exists - download if it doesn't
        self.load_edges_weights = load_edges_weights
        if load_edges_weights:
            self.edges_weight_file = os.path.join(
                data_root_folder, EDGES_WEIGHTS_FILENAME
            )
            if not os.path.exists(self.edges_weight_file):
                download_zenodo_file(
                    os.path.join(ZENODO_DATASET_BASE_URI, EDGES_WEIGHTS_FILENAME),
                    self.edges_weight_file,
                )
            self.edges_weight = torch.load(self.edges_weight_file).to_dense()
            logger.info(f"[DONE] Loaded {self.edges_weight_file}")

        # Load nodes feaures file
        # Check for nodes features file - download if it doesn't exist
        _nodes_feature_filename = NODES_FEATURES_FILENAME_TEMPLATE.replace(
            "<FEATURE_TYPE>",
            nodes_feature_type,
        )
        self.nodes_feature_file = os.path.join(
            data_root_folder, _nodes_feature_filename
        )
        if not os.path.exists(self.nodes_feature_file):
            download_zenodo_file(
                os.path.join(ZENODO_DATASET_BASE_URI, _nodes_feature_filename),
                self.nodes_feature_file,
            )
        self._load_node_features()
        logger.info(f"[DONE] Loaded {self.nodes_feature_file}")

        # Load evaluation test data
        self.eval_task_types = ("sp", "sr", "cr")
        self.eval_tasks = dict()
        for task_type in self.eval_task_types:
            # Check if evaluation test data exists - otherwise download it
            _eval_task_filename = EVAL_TASK_FILENAME_TEMPLATE.replace(
                "<TASK_TYPE>", task_type
            )
            _eval_task_file = os.path.join(data_root_folder, _eval_task_filename)
            if not os.path.exists(_eval_task_file):
                download_zenodo_file(
                    os.path.join(ZENODO_DATASET_BASE_URI, _eval_task_filename),
                    _eval_task_file,
                )
            self.eval_tasks[task_type] = (
                _eval_task_file,
                pd.read_parquet(_eval_task_file),
            )
            logger.info(f"[DONE] Loaded {_eval_task_file}")

        self.n_edges = self.edges.shape[0]
        if self.load_edges_weights:
            self.edges_weight_dim = self.edges_weight.shape[1]

        # Default Top-K for CR task
        self.eval_cr_top_ks = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

    def _load_node_features(self):
        self.nodes_feature = torch.load(self.nodes_feature_file)
        if self.nodes_feature.dtype is not torch.float32:
            self.nodes_feature = self.nodes_feature.to(dtype=torch.float32)
        # Set Vars
        self.n_nodes = self.nodes_feature.shape[0]
        self.nodes_feature_dim = self.nodes_feature.shape[1]

    def change_feature_type(self, feature_type: str):
        if feature_type != self.nodes_feature_type:
            self.nodes_feature_type = feature_type
            self._load_node_features()

    @property
    def nodes_id(self) -> list:
        """Get an ordered list of node IDs.

        Returns:
            list: an ordered (ascending) list of node IDs.
        """
        return [i for i in range(self.n_nodes)]

    def describe(self) -> None:
        """Print key statistics of loaded data."""
        print(f"data_root_folder={self.data_root_folder}")
        print(f"n_nodes={self.n_nodes}, n_edges={self.n_edges}")
        print(f"nodes_feature_type={self.nodes_feature_type}")
        print(f"nodes_feature_dimension={self.nodes_feature_dim}")
        if self.load_edges_weights:
            print(f"edges_weight_dimension={self.edges_weight_dim}")
        for task_type in self.eval_task_types:
            print(f"{task_type}: {len(self.eval_tasks[task_type][1])} samples")

    def to_pyg(self):
        """
        Build a PyTorch-geometric graph from the loaded CompanyKG.

        """
        try:
            from torch_geometric.data import Data
        except ImportError as e:
            raise ImportError(
                "pytorch-geometric is not installed: please install to produce PyG graph"
            ) from e

        # Incxlude edges going in both directions, since PyG uses directed graphs
        edge_index = torch.concat([self.edges.T, self.edges[:, [1, 0]].T], dim=1)
        return Data(x=self.nodes_feature, edge_index=edge_index)

    def to_igraph(self):
        """
        Build an iGraph graph from the loaded CompanyKG.
        Requires iGraph to be installed.

        """
        try:
            import igraph as ig
        except ImportError as e:
            raise ImportError(
                "python-igraph is not installed: please install to produce iGraph graph"
            ) from e

        g = ig.Graph()
        g.add_vertices(self.n_nodes)
        # Names should be strings
        g.vs["name"] = [str(i) for i in self.nodes_id]

        logger.info("Building iGraph graph from edges")
        if self.load_edges_weights:
            # Convert tensors to Np arrays
            edge_weights = self.edges_weight.numpy()
            edges = self.edges.numpy()

            # Flatten the non-zero weights for each edge so we have a separate edge for each weight type
            nonzeros = np.nonzero(edge_weights)
            # These are just the column indices of the nonzeros
            types = nonzeros[1]
            # The weights for these separated edges are the flattened non-zero values
            weights = edge_weights[nonzeros]
            # The edges themselves are indexed by the row indices of the non-zero values
            # This repeats edges where there are multiple non-zero weight types
            edges = edges[nonzeros[0]]

            # Flatten the non-zero weights for each edge so we have a separate edge for each weight type
            attrs = {
                "type": types,
                "weight": weights,
            }
            g.add_edges(
                edges,
                attributes=attrs,
            )
        else:
            g.add_edges((i, j) for (i, j) in self.edges)
        return g

    def evaluate_sp(self, embed: torch.Tensor) -> float:
        """Evaluate the specified node embeddings on SP task.

        Args:
            embed (torch.Tensor): the node embeddings to be evaluated.

        Returns:
            float: AUC score on SP task.
        """
        test_data = self.eval_tasks["sp"][1]
        gt = test_data["label"].tolist()
        pred = []
        for _, row in test_data.iterrows():
            node_embeds = (embed[row["node_id0"]], embed[row["node_id1"]])
            try:
                with np.errstate(invalid="ignore"):
                    pred.append(
                        1 - spatial.distance.cosine(node_embeds[0], node_embeds[1])
                    )
            except:
                print(row)
                raise
        return roc_auc_score(gt, pred)

    def evaluate_sr(self, embed: torch.Tensor, split: str = "validation") -> float:
        """Evaluate the specified node embeddings on SR task.

        Args:
            embed (torch.Tensor): the node embeddings to be evaluated.
            split (str): the split (validation/test) on which the evaluation will be run.

        Returns:
            float: Accuracy on SR task.
        """
        test_data = self.eval_tasks["sr"][1]
        test_data = test_data[test_data["split"] == split]
        gt = test_data["label"].tolist()
        pred = []
        for _, row in test_data.iterrows():
            query_embed = embed[row["target_node_id"]]
            candidate0_embed = embed[row["candidate0_node_id"]]
            candidate1_embed = embed[row["candidate1_node_id"]]
            with np.errstate(invalid="ignore"):
                _p1 = 1 - spatial.distance.cosine(query_embed, candidate0_embed)
                _p2 = 1 - spatial.distance.cosine(query_embed, candidate1_embed)
            pred.append(0) if _p1 >= _p2 else pred.append(1)
        return accuracy_score(gt, pred)

    @staticmethod
    def search_most_similar(
        target_embed: torch.Tensor, embed: torch.Tensor
    ) -> Tuple[np.array, np.array]:
        """Search top-K most similar nodes to a target node.

        Args:
            target_embed (torch.Tensor): the embedding of the target node.
            embed (torch.Tensor): the node embeddings to be searched from, i.e. candidate nodes.
            K (int, optional): the number of nodes to be returned as search result. Defaults to 50.

        Returns:
            Tuple[np.array, np.array]: the node IDs and the cosine similarity scores.
        """
        with np.errstate(invalid="ignore"):
            sims = np.dot(embed, target_embed) / (
                np.linalg.norm(embed, axis=1) * np.linalg.norm(target_embed)
            )
        # Reverse so the most similar is first
        max_ids = np.argsort(sims)[
            -2::-1
        ]  # remove target company (first element in the reversed array)
        return max_ids, sims[max_ids]

    def cr_top_ks(self, embed: torch.Tensor, ks: List[int]):
        """
        Evaluate CR (Competitor retrieval) as the average recall @ k for a number of
        different values of k.

        :param embed: the node embeddings to be evaluated
        :param ks: list of k-values to evaluate at
        :return:
        """
        test_data = self.eval_tasks["cr"][1]
        target_nids = list(test_data["target_node_id"].unique())
        target_nids.sort()
        # Collect recalls for different ks from each sample
        k_recalls = dict((k, []) for k in ks)

        for nid in target_nids:
            competitor_df = test_data[test_data["target_node_id"] == nid]
            competitor_nids = set(list(competitor_df["competitor_node_id"].unique()))
            # Get as many predictions as we'll need for the highest k
            res_nids, res_dists = CompanyKG.search_most_similar(embed[nid], embed)
            for k in sorted(ks):
                k_res_nids = set(res_nids[:k])
                common_set = k_res_nids & competitor_nids
                recall = len(common_set) / len(competitor_nids)
                k_recalls[k].append(recall)

        # Average the recalls over samples for each k
        recalls = [np.mean(k_recalls[k]) for k in sorted(ks)]
        return recalls

    def cr_top_k(self, embed: torch.Tensor, k: int = 50) -> float:
        """Evaluate CR (Competitor Retrieval) performance using top-K hit rate.
        This function will evaluate each target company in CR test set.

        Args:
            embed (torch.Tensor): the node embeddings to be evaluated
            k (int, optional): the number of nodes to be returned as search result. Defaults to 50.

        Returns:
            Tuple[float, list]: the overall hit rate and the per-target hit rate.
        """
        return self.cr_top_ks(embed, [k])[0]

    def evaluate_cr(self, embed: torch.Tensor) -> list:
        """Evaluate the specified node embeddings on CR task.

        Args:
            embed (torch.Tensor): the node embeddings to be evaluated.

        Returns:
            float: the list of tuples containing the CR results.
                The first element in each tuple is the overall hit rate for top-K.
        """
        return self.cr_top_ks(embed, self.eval_cr_top_ks)

    def evaluate(
        self,
        embeddings_file: str = None,
        embed: torch.Tensor = None,
        silent: bool = False,
    ) -> dict:
        """Evaluate the specified embedding on all evaluation tasks: SP, SR and CR.
        When none parameters provided, it will evaluate the embodied nodes feature.

        Args:
            embeddings_file (str, optional): the path to the embedding file;
                it has highest priority. Defaults to None.
            embed (torch.Tensor, optional): the embedding to be evaluated;
                it has second highest priority. Defaults to None.
            silent (bool): by default, evaluation results are printed to stdout;
                if True, nothing is output, you just get the results in the
                returned dict

        Returns:
            dict: a dictionary of evaluation results.
        """
        if embeddings_file is not None:
            try:
                embed = torch.load(embeddings_file)
            except:
                embed = torch.load(embeddings_file, map_location="cpu")
            result_dict = {"source": embeddings_file}
            if not silent:
                print(f"Evaluate Node Embeddings {embeddings_file}:")
        elif embed is not None:
            result_dict = {"source": f"embed {embed.shape}"}
            if not silent:
                print(f"Evaluate Custom Embeddings:")
        else:
            embed = self.nodes_feature
            result_dict = {"source": self.nodes_feature_type}
            if not silent:
                print(f"Evaluate Node Features {self.nodes_feature_type}:")
        # SP Task
        if not silent:
            print("Evaluate SP ...")
        result_dict["sp_auc"] = self.evaluate_sp(embed)
        if not silent:
            print("SP AUC:", result_dict["sp_auc"])
        # SR Task
        if not silent:
            print("Evaluate SR ...")
        result_dict["sr_validation_acc"] = self.evaluate_sr(embed)
        result_dict["sr_test_acc"] = self.evaluate_sr(embed, split="test")
        if not silent:
            print(
                "SR Validation ACC:",
                result_dict["sr_validation_acc"],
                "SR Test ACC:",
                result_dict["sr_test_acc"],
            )

        # CR Task
        if not silent:
            print(f"Evaluate CR with top-K hit rate (K={self.eval_cr_top_ks}) ...")
        result_dict["cr_topk_hit_rate"] = self.evaluate_cr(embed)
        if not silent:
            print("CR Hit Rates:", result_dict["cr_topk_hit_rate"])

        return result_dict

    def get_dgl_graph(self, work_folder: str) -> list:
        """Obtain a DGL graph. If it has not been built before, a new graph will be constructed,
        otherwise it will simply load from file in the specified working directory.

        Args:
            work_folder (str): the working directory of graph building.

        Returns:
            list: the built graph(s).
        """
        try:
            import dgl
        except ImportError as e:
            raise ImportError(
                "DGL is not installed. Please install to produce DGL graph"
            ) from e

        dgl_file = os.path.join(work_folder, f"dgl_{self.nodes_feature_type}.bin")
        if os.path.isfile(dgl_file):
            return dgl.data.utils.load_graphs(dgl_file)[0]
        else:
            graph_data = {
                ("_N", "_E", "_N"): self.edges.tolist(),
            }
            g = dgl.heterograph(graph_data)
            g.ndata["feat"] = self.nodes_feature
            if self.load_edges_weights:
                g.edata["weight"] = self.edges_weight
            dgl.data.utils.save_graphs(dgl_file, [g])
        return [g]
