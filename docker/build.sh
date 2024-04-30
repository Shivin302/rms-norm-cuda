# cd to the docker folder and call this script to build the docker image
source .env
export CUR_DIR=$(cd $(dirname "${BASH_SOURCE[0]}")/ && pwd -P)
export IMAGE_NAME=$(echo $IMAGE_NAME)
export USER_ID=$(id -u)
export USERNAME=$(id -un)
echo $USER_ID $USERNAME $CUR_DIR $CUDA_VISIBLE_DEVICES

docker compose build
