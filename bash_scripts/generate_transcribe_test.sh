#!/usr/bin/env bash

# Run from project's root

# Set PYTHONPATH
parentdir="$(dirname "$(pwd)")"
export PYTHONPATH=$PYTHONPATH:$parentdir

# Generate Noisy Tracks
echo 'Generate Dataset Test'
python3 ./add_noise_dataset.py --dataset_name test-clean

# Transcribe Noisy Tracks
echo 'Transcribe Dataset Test'
python3 ./transcribe_dataset.py --dataset_name test-clean
