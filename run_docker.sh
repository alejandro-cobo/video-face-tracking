#!/bin/bash

# Script to build and run the docker image
# Usage: ./run_docker.sh <PATH>
# Author: Alejandro Cobo (alejandro.cobo@upm.es)
# Last revised: 2025/01/20

readonly ARGS="$@"
readonly IMAGE="video-face-tracking"
readonly VOLUME_PATH="/usr/src/app/input"

build_if_necessary() {
    if [ -z "$(docker images -q "$IMAGE" 2> /dev/null)" ]; then
        docker build . -t video-face-tracking
    fi
}

main() {
    local path=${ARGS[0]}
    local kwargs=${ARGS[@]:1:}

    build_if_necessary

    if [ -f "$path" ]; then
        path=$(realpath -- $path)
        local dirname=$(dirname -- $path)
        local basename=$(basename -- $path)
        docker run -v "$dirname:$VOLUME_PATH" --gpus all "$IMAGE" "$VOLUME_PATH/$basename" $kwargs
    elif [ -d "$path" ]; then
        path=$(realpath -- $path)
        docker run -v "$path:$VOLUME_PATH" --gpus all "$IMAGE" "$VOLUME_PATH" $kwargs
    else
        echo "run_docker.sh error: \"$path\" is not a file or directory" 1>&2
        docker run "$IMAGE" --help
    fi
}
main

