#!/usr/bin/env bash

# Run from project's root

# Set PYTHONPATH
parentdir="$(dirname "$(pwd)")"
export PYTHONPATH=$PYTHONPATH:$parentdir

# Generate Noisy Tracks
echo 'Collect Features in Dataframe Test'
python3 ./collect_features_in_df.py --dataset_name test-clean --transcriber google
