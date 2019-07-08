#!/usr/bin/env bash

# Run from project's root

# Set PYTHONPATH
parentdir="$(dirname "$(pwd)")"
export PYTHONPATH=$PYTHONPATH:$parentdir

# Generate Noisy Tracks
echo 'Generate Dataset Train-100'
python3 ./add_noise_dataset.py --dataset_name train-clean-100

# Transcribe Noisy Tracks
echo 'Transcribe Dataset Train-100'
python3 ./transcribe_dataset.py --dataset_name train-clean-100

# Compute features
echo 'Computing Features Train-100'
python3 ./compute_features.py --dataset_name train-clean-100