#!/bin/bash
# cd to the docker folder before calling this script
set -e # force error if .env doesn't exist

export CUR_DIR=$(cd $(dirname "${BASH_SOURCE[0]}")/ && pwd -P)

# Local environment override
source $CUR_DIR/.env
export IMAGE_NAME=$(echo $IMAGE_NAME)
export USER_ID=$(id -u)
export USERNAME=$(id -un)

echo "Running image:" $IMAGE_NAME
echo "Using GPUs:" $CUDA_VISIBLE_DEVICES

docker compose -f docker-compose.yaml run --rm rms_norm
