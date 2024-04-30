#!/bin/bash
set -e
# Entrypoint for Docker container

echo "Starting pipeline"

cd /app
python rms_torch.py

echo "Finished pipeline"
