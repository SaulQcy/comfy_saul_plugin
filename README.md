# comfy-plugin

## How to install

First u need to instasll ComfyUI, and then cd `/home/xxx/comfy/ComfyUI/custom_nodes`. Then clone this repo and restart ComfyUI.

## Key Plugins

1. `yolov3_smoking_det.py`

This plugin is to auto-label the generated fake images by Diffusion Models (DMs).

2. `fuse_cigarette_people.py`

This plugin patchs a cigarette to a people's mouse according to the face keypoints.

3. `extract_similary_webp.py`

This plugin automatically search the most similar pose video given an image. The similarity is calculated in `kpts_similarity.py`

## How to use in terminal

1. First, u need to run the main in a tmux terminal, e.g., `python main.py`.

2. Second, u can conduct a simple test, i.e., u can try run `comfy_saul_plugin/run/gen_via_api.py`. This file will send HTTP request to `main` to perform AIGC.
More specifically, the `gen_via_api.py` controls `main` by a API file provided in a config file.

3. By the way, the config is `comfy_saul_plugin/run/config/default.yaml`, which has several keys:
- `json_file_path`: the API file path. U can also build your own workflow and export to API format.
- `task`: e.g., `cup`, the images will be saved in `xxx/xxx/cup/xxx`.
- `positive` and `negative`: these are the prompts to guild the DMs generation.
- `people`: this is the input image. The DMs will generate a video whose first frame is this image.

4. Thirdly, if anything is ok, u can run `comfy_saul_plugin/sh/gen_x_single.sh`, which accepts two parameters, i.e., the input image path and the task name.
This shell script will overload some values of keys in `comfy_saul_plugin/run/config/default.yaml`. 
The most import overload values is `people` because we want to generate from different images. 

Examples:
```shell
./gen_x_single.sh /home/lior/datasets/dms/train/Manual_Full/c200_drink/DC2E97CCD830@Phone@1744680486@2@2001E6BB29CDEC7E7F4C76048AEC3FF0.jpg cup
```
5. Finally, if anything is ok, u can open a new termux and run `comfy_saul_plugin/sh/gen_x.sh`, which also accept two args, i.e., the target folder and the task name. This script will scanner the target folder to find all images, then call `comfy_saul_plugin/sh/gen_x_single.sh`.

Example:
```shell
./gen_x.sh /home/saul/Lior/cup/iter_2508 cup
```

OK. If all things right, the next time u want to generate images, u only need to perform Step 1 and Step 5.


