#!/bin/bash

# static args
PYTHON_SCRIPT_PATH="/home/saul/comfy/ComfyUI/custom_nodes/comfy_saul_plugin/run"

if [ "$#" -eq 2 ]; then
    # PYTHON_SCRIPT_NAME="gen_smoke_camera.py"
    PYTHON_SCRIPT_NAME="gen_via_api.py"
    yq e '.people = "'"$1"'"' -i "$PYTHON_SCRIPT_PATH/config/default.yaml"
    yq e '.task = "'"$2"'"' -i "$PYTHON_SCRIPT_PATH/config/default.yaml"
else
    exit 1
fi

# source miniconda, need to be excute by BASH.
source ~/miniconda3/etc/profile.d/conda.sh

[ -d "$PYTHON_SCRIPT_PATH" ] && echo "folder exists" || { echo "folder does not exist"; exit 1; }

cd "$PYTHON_SCRIPT_PATH"

[ -f "$PYTHON_SCRIPT_NAME" ] && echo "file exists" || { echo "file does not exist"; exit 1; }

if conda info --envs | grep "comfy"; then
    conda activate comfy
else
    echo "conda env does not exits"
    exit 1
fi

python "$PYTHON_SCRIPT_NAME"

