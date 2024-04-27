#!/bin/bash
set -e
# Entrypoint for Docker container

echo "Starting pipeline"
# mode_root="/app/appelii/apps/$mode"
# cur_time=$(date +%F_%H-%M-%S)
# path="/app/appelii/experiments/${cur_time}_${mode}_${descriptor}"
# mkdir -p $path
# cp -r ${mode_root}/* ${path}
# echo "Running pipeline in ${path}"

# cd $path
export MPLCONFIGDIR=$(mktemp -d)

python /app/rms_torch.py

echo "Finished pipeline"
