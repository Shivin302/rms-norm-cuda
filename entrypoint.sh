#!/bin/bash
set -e
# Entrypoint for Docker container

echo "Starting pipeline"

# cd $path
export MPLCONFIGDIR=$(mktemp -d)

python /app/rms_torch.py

echo "Finished pipeline"
