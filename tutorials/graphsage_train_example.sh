#!/bin/bash
#
# Copyright (C) eqtgroup.com Ltd 2023
# https://github.com/EQTPartners/CompanyKG
# License: MIT, https://github.com/EQTPartners/CompanyKG/LICENSE.md
#
# This is an example of how to call the GraphSAGE training interface
# Note that the hyperparameters used are extremely limited, so the
#  resulting model will not be good, but can be trained with small memory
python -m ckg_benchmarks.graphsage.train \
    --epochs 1 \
    --n-layer 2 \
    --embedding-dim 8 \
    --data-root-folder ./data \
    --device -1 \
    --train-batch-size 256 \
    --inference-batch-size 256 \
    --n-sample-neighbor 2
