source .env
export CUR_DIR=$(cd $(dirname "${BASH_SOURCE[0]}")/ && pwd -P)
export IMAGE_NAME=$(echo $IMAGE_NAME)
export USER_ID=$(id -u)
export USERNAME=$(id -un)
echo $USER_ID $USERNAME

docker compose build
