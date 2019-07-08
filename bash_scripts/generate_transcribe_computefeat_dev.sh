#!/usr/bin/env bash

# Run from project's root

# Set PYTHONPATH
parentdir="$(dirname "$(pwd)")"
export PYTHONPATH=$PYTHONPATH:$parentdir

# Generate Noisy Tracks
echo 'Generate Dataset Dev'
python3 ./add_noise_dataset.py --dataset_name dev-clean

# Transcribe Noisy Tracks
echo 'Transcribe Dataset Dev'g
python3 ./transcribe_dataset.py --dataset_name dev-clean

# Compute features
echo 'Compute Features Dev'
python3 ./compute_features.py --dataset_name dev-clean
