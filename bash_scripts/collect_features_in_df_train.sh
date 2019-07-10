#!/usr/bin/env bash

# Run from project's root

# Set PYTHONPATH
parentdir="$(dirname "$(pwd)")"
export PYTHONPATH=$PYTHONPATH:$parentdir

# Generate Noisy Tracks
echo 'Collect Features in Dataframe Train'
python3 ./collect_features_in_df.py --dataset_name train-clean-100 --transcriber sphinx
