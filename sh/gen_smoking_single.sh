#!/bin/bash

# static args
PYTHON_SCRIPT_PATH="/home/saul/comfy/ComfyUI/custom_nodes/comfy_saul_plugin/run"
PYTHON_SCRIPT_NAME="gen_fake_cigarette.py"

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

# dyna args
if [ "$#" -eq 2 ]; then
    yq e '.people = "'"$1"'"' -i "$PYTHON_SCRIPT_PATH/config/smoke.yaml"
    yq e '.cigarette = "'"$2"'"' -i "$PYTHON_SCRIPT_PATH/config/smoke.yaml"
elif [ "$#" -eq 1 ]; then
    yq e '.people = "'"$1"'"' -i "$PYTHON_SCRIPT_PATH/config/smoke.yaml"
    yq e '.cigarette = "'"None"'"' -i "$PYTHON_SCRIPT_PATH/config/smoke.yaml"
else
    exit 1
fi

python "$PYTHON_SCRIPT_NAME"

