#!/bin/bash
#
# Copyright (C) eqtgroup.com Ltd 2023
# https://github.com/EQTPartners/CompanyKG
# License: MIT, https://github.com/EQTPartners/CompanyKG/LICENSE.md
#
# This is an example of how to call the GCL training interface
#  to train a GNN with GRACE
python -m ckg_benchmarks.gcl.train \
    --device -1 \
    --method grace \
    --n-layer 1 \
    --embedding-dim 8 \
    --epochs 1 \
    --sampler-edges 2 \
    --batch-size 128
