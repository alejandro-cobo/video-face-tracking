#!/bin/bash

# Script to export the necessary paths to make onnxruntime locate CUDA libs
# Usage: source setup_cuda.sh
# Author: Alejandro Cobo (alejandro.cobo@upm.es)
# Last revised: 2025/01/20

main() {
    unset LD_LIBRARY_PATH
    local pip_cmd="pip"
    if ! command -v $pip_cmd &> /dev/null; then
        pip_cmd="uv pip"
    fi
    local root_path=$($pip_cmd show onnxruntime-gpu | awk '$1 ~ /Location/ {print $2}')
    for dir in $(echo $root_path/nvidia/*/lib); do
        export LD_LIBRARY_PATH=$dir:$LD_LIBRARY_PATH
    done
}
main

