#!/bin/bash

# static args
PYTHON_SCRIPT_PATH="/home/saul/comfy/ComfyUI/custom_nodes/comfy_saul_plugin/run"


# function
# length of args is 2, means no cigarette in people face, so use Wan2.1+ControlNet to generate data.
# length is 1 means using Wan2.1+Camera to generate data.
if [ "$#" -eq 2 ]; then
    PYTHON_SCRIPT_NAME="gen_smoke_control.py"
    yq e '.people = "'"$1"'"' -i "$PYTHON_SCRIPT_PATH/config/smoke_control.yaml"
    yq e '.cigarette = "'"$2"'"' -i "$PYTHON_SCRIPT_PATH/config/smoke_control.yaml"
elif [ "$#" -eq 1 ]; then
    # PYTHON_SCRIPT_NAME="gen_smoke_camera.py"
    PYTHON_SCRIPT_NAME="gen_smoke_wan2_2_14B.py"
    yq e '.people = "'"$1"'"' -i "$PYTHON_SCRIPT_PATH/config/smoke_camera.yaml"
    # yq e '.cigarette = "'"None"'"' -i "$PYTHON_SCRIPT_PATH/config/smoke_camera.yaml"
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

