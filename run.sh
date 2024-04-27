#!/bin/bash
set -e # force error if .env doesn't exist, e.g.

export CUR_DIR=$(cd $(dirname "${BASH_SOURCE[0]}")/ && pwd -P)
source $CUR_DIR/docker.env

# Local environment override
source $CUR_DIR/.env
export IMAGE_NAME=$(echo $IMAGE_NAME)
export USER_ID=$(id -u)
export USERNAME=$(id -un)

echo "Running ${mode} in: " $IMAGE_NAME
# echo "Using GPU(s): " ${gpu_num}

docker compose -f docker-compose.yaml run --rm rms_norm
