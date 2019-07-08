#!/usr/bin/env bash

# Run from project's root

# Set PYTHONPATH
parentdir="$(dirname "$(pwd)")"
export PYTHONPATH=$PYTHONPATH:$parentdir

# Compute features
echo 'Compute Features'
python3 ./compute_features.py --dataset_name dev-clean
python3 ./compute_features.py --dataset_name test-clean
python3 ./compute_features.py --dataset_name train-clean-100
# python3 ./compute_features.py --dataset_name train-clean-360

