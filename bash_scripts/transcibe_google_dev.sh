#!/usr/bin/env bash

# Run from project's root

# Set PYTHONPATH
parentdir="$(dirname "$(pwd)")"
export PYTHONPATH=$PYTHONPATH:$parentdir

# Transcribe Noisy Tracks
echo 'Transcribe Dataset Dev'
python3 ./transcribe_dataset.py --dataset_name dev-clean
