#!/bin/bash
set -e
# Entrypoint for Docker container

echo "Starting pipeline"

cd /app
pytest
python run_rms.py

echo "Finished pipeline"
