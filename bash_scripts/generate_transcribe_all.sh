#!/usr/bin/env bash

# Run from project's root

# Set PYTHONPATH
parentdir="$(dirname "$(pwd)")"
export PYTHONPATH=$PYTHONPATH:$parentdir

# Generate Noisy Tracks
echo 'Generate Dataset'
python3 ./add_noise_dataset.py --dataset_name dev-clean
python3 ./add_noise_dataset.py --dataset_name test-clean
python3 ./add_noise_dataset.py --dataset_name train-clean-100
# python3 ./add_noise_dataset.py --dataset_name train-clean-360


# Transcribe Noisy Tracks
echo 'Transcribe Dataset'
python3 ./transcribe_dataset.py --dataset_name dev-clean
python3 ./transcribe_dataset.py --dataset_name test-clean
python3 ./transcribe_dataset.py --dataset_name train-clean-100
# python3 ./transcribe_dataset.py --dataset_name train-clean-360