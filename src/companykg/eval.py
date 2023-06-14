import argparse
import json
import logging
from pathlib import Path

import torch

from companykg import CompanyKG

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "embeddings_path",
        help="Path to a pyTorch tensor file containing embeddings to be evaluated"
    )
    parser.add_argument(
        "--data-root-folder",
        default="./data",
        type=str,
        help="The root folder where the CompanyKG data is downloaded to",
    )
    parser.add_argument(
        "--output",
        default="./eval_results.json",
        type=str,
        help="File path to output evaluation results to as JSON",
    )
    opts = parser.parse_args()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    # Load the CKG data
    # Node feature type and edge weights, as we only use it for evaluation
    logger.info(f"Loading CompanyKG from {opts.data_root_folder}")
    ckg = CompanyKG(
        data_root_folder=opts.data_root_folder,
    )
    logger.info(f"Loading embeddings for evaluation from {opts.embeddings_path}")
    embed = torch.load(opts.embeddings_path)

    # Check we've got an embedding for every node
    if embed.shape[0] != ckg.n_nodes:
        raise ValueError(f"number of embeddings ({embed.shape[0]}) does not match number of "
                         f"nodes in KG ({ckg.n_nodes})")

    logger.info("Running evaluation")
    # This will output the results to stdout
    results = ckg.evaluate(embed=embed, silent=False)

    eval_results_path = Path(opts.output)
    with eval_results_path.open("w") as f:
        json.dump(results, f)
    logger.info(f"Evaluation results are exported to {eval_results_path}")
