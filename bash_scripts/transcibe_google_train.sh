#!/usr/bin/env bash

# Run from project's root

# Set PYTHONPATH
parentdir="$(dirname "$(pwd)")"
export PYTHONPATH=$PYTHONPATH:$parentdir

# Transcribe Noisy Tracks
echo 'Transcribe Dataset Train'
python3 ./transcribe_dataset.py --dataset_name train-clean-100
