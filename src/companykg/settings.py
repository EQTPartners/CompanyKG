"""
Copyright (C) eqtgroup.com Ltd 2023
https://github.com/EQTPartners/CompanyKG
License: MIT, https://github.com/EQTPartners/CompanyKG/LICENSE.md
"""

CHUNK_SIZE = 1024 * 1024 * 64  # 64 MiB chunk sizes
# This should point to the record number of the latest version
ZENODO_RECORD_NUMBER = "8010239"
ZENODO_DATASET_BASE_URI = f"https://zenodo.org/record/{ZENODO_RECORD_NUMBER}/files/"
EDGES_FILENAME = "edges.pt"
EDGES_WEIGHTS_FILENAME = "edges_weight.pt"
NODES_FEATURES_FILENAME_TEMPLATE = "nodes_feature_<FEATURE_TYPE>.pt"
EVAL_TASK_FILENAME_TEMPLATE = "eval_task_<TASK_TYPE>.parquet.gz"
