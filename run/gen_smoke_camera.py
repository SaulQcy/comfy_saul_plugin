import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS
from nodes import NODE_CLASS_MAPPINGS
from custom_nodes.comfy_saul_plugin.vitpose import VitPoseProcess
import PIL.Image
import numpy as np
from custom_nodes.comfy_saul_plugin.change_camera_pose import ChangeCameraPose
import hydra
from omegaconf import OmegaConf, DictConfig

@hydra.main(version_base=None, config_path="./config/", config_name="smoke_camera")
def main(config: DictConfig):

    positive_prompt = config.positive
    negative_prompt = config.negative
    input_image_path = config.people

    img = PIL.Image.open(input_image_path)
    cam_w = cam_h = 512

    x_np = np.array(img).astype(float) / 255.
    x_t = torch.from_numpy(x_np).unsqueeze(0)
    x_pose, _, _ = VitPoseProcess().main(x_t)
    camera_pose, = ChangeCameraPose().main(x_pose)

    print(positive_prompt, negative_prompt, camera_pose)
    print(positive_prompt, negative_prompt, camera_pose)
    print(positive_prompt, negative_prompt, camera_pose)

    import_custom_nodes()
    with torch.inference_mode():
        print(positive_prompt, negative_prompt, camera_pose)
        print(positive_prompt, negative_prompt, camera_pose)
        print(positive_prompt, negative_prompt, camera_pose)
        cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        cliploader_38 = cliploader.load_clip(
            clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            type="wan",
            device="default",
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_6 = cliptextencode.encode(
            text=positive_prompt,
            clip=get_value_at_index(cliploader_38, 0),
        )

        cliptextencode_7 = cliptextencode.encode(
            text=negative_prompt,
            clip=get_value_at_index(cliploader_38, 0),
        )

        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_37 = unetloader.load_unet(
            unet_name="wan2.1_fun_camera_v1.1_1.3B_bf16.safetensors",
            weight_dtype="default",
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_39 = vaeloader.load_vae(vae_name="wan_2.1_vae.safetensors")

        clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
        clipvisionloader_49 = clipvisionloader.load_clip(
            clip_name="clip_vision_h.safetensors"
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_52 = loadimage.load_image(image="camera_2_00005_001505.jpg")

        clipvisionencode = NODE_CLASS_MAPPINGS["CLIPVisionEncode"]()
        clipvisionencode_51 = clipvisionencode.encode(
            crop="none",
            clip_vision=get_value_at_index(clipvisionloader_49, 0),
            image=get_value_at_index(loadimage_52, 0),
        )

        wancameraembedding = NODE_CLASS_MAPPINGS["WanCameraEmbedding"]()
        wancameraembedding_57 = wancameraembedding.run(
            camera_pose=camera_pose,
            width=cam_w,
            height=cam_h,
            length=81,
            speed=1,
            fx=0.5,
            fy=0.5,
            cx=0.5,
            cy=0.5,
        )

        wancameraimagetovideo = NODE_CLASS_MAPPINGS["WanCameraImageToVideo"]()
        wancameraimagetovideo_56 = wancameraimagetovideo.encode(
            width=get_value_at_index(wancameraembedding_57, 1),
            height=get_value_at_index(wancameraembedding_57, 2),
            length=get_value_at_index(wancameraembedding_57, 3),
            batch_size=1,
            positive=get_value_at_index(cliptextencode_6, 0),
            negative=get_value_at_index(cliptextencode_7, 0),
            vae=get_value_at_index(vaeloader_39, 0),
            clip_vision_output=get_value_at_index(clipvisionencode_51, 0),
            start_image=get_value_at_index(loadimage_52, 0),
            camera_conditions=get_value_at_index(wancameraembedding_57, 0),
        )

        modelsamplingsd3 = NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        smoking_auto_label = NODE_CLASS_MAPPINGS["Smoking Auto Label"]()
        end_node = NODE_CLASS_MAPPINGS["End Node"]()

        for q in range(1):
            modelsamplingsd3_54 = modelsamplingsd3.patch(
                shift=8, model=get_value_at_index(unetloader_37, 0)
            )

            ksampler_3 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=6,
                sampler_name="uni_pc",
                scheduler="simple",
                denoise=1,
                model=get_value_at_index(modelsamplingsd3_54, 0),
                positive=get_value_at_index(wancameraimagetovideo_56, 0),
                negative=get_value_at_index(wancameraimagetovideo_56, 1),
                latent_image=get_value_at_index(wancameraimagetovideo_56, 2),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampler_3, 0),
                vae=get_value_at_index(vaeloader_39, 0),
            )

            smoking_auto_label_63 = smoking_auto_label.main(
                image_in=get_value_at_index(vaedecode_8, 0)
            )

            end_node_64 = end_node.main(
                any=get_value_at_index(smoking_auto_label_63, 0)
            )


if __name__ == "__main__":
    main()
