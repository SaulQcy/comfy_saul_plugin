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


def main():
    import_custom_nodes()
    with torch.inference_mode():
        cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        cliploader_38 = cliploader.load_clip(
            clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            type="wan",
            device="default",
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_6 = cliptextencode.encode(
            text="The white dragon warrior stands still, eyes full of determination and strength. The camera slowly moves closer or circles around the warrior, highlighting the powerful presence and heroic spirit of the character.",
            clip=get_value_at_index(cliploader_38, 0),
        )

        cliptextencode_7 = cliptextencode.encode(
            text="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            clip=get_value_at_index(cliploader_38, 0),
        )

        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_37 = unetloader.load_unet(
            unet_name="wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
            weight_dtype="default",
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_39 = vaeloader.load_vae(vae_name="wan_2.1_vae.safetensors")

        unetloader_56 = unetloader.load_unet(
            unet_name="wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
            weight_dtype="default",
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_62 = loadimage.load_image(image="example.png")

        wanimagetovideo = NODE_CLASS_MAPPINGS["WanImageToVideo"]()
        wanimagetovideo_63 = wanimagetovideo.encode(
            width=640,
            height=640,
            length=81,
            batch_size=1,
            positive=get_value_at_index(cliptextencode_6, 0),
            negative=get_value_at_index(cliptextencode_7, 0),
            vae=get_value_at_index(vaeloader_39, 0),
            start_image=get_value_at_index(loadimage_62, 0),
        )

        modelsamplingsd3 = NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
        ksampleradvanced = NODE_CLASS_MAPPINGS["KSamplerAdvanced"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        createvideo = NODE_CLASS_MAPPINGS["CreateVideo"]()
        savevideo = NODE_CLASS_MAPPINGS["SaveVideo"]()

        for q in range(1):
            modelsamplingsd3_55 = modelsamplingsd3.patch(
                shift=8, model=get_value_at_index(unetloader_56, 0)
            )

            modelsamplingsd3_54 = modelsamplingsd3.patch(
                shift=8.000000000000002, model=get_value_at_index(unetloader_37, 0)
            )

            ksampleradvanced_57 = ksampleradvanced.sample(
                add_noise="enable",
                noise_seed=random.randint(1, 2**64),
                steps=20,
                cfg=3.5,
                sampler_name="euler",
                scheduler="simple",
                start_at_step=0,
                end_at_step=10,
                return_with_leftover_noise="enable",
                model=get_value_at_index(modelsamplingsd3_54, 0),
                positive=get_value_at_index(wanimagetovideo_63, 0),
                negative=get_value_at_index(wanimagetovideo_63, 1),
                latent_image=get_value_at_index(wanimagetovideo_63, 2),
            )

            ksampleradvanced_58 = ksampleradvanced.sample(
                add_noise="disable",
                noise_seed=random.randint(1, 2**64),
                steps=20,
                cfg=3.5,
                sampler_name="euler",
                scheduler="simple",
                start_at_step=10,
                end_at_step=10000,
                return_with_leftover_noise="disable",
                model=get_value_at_index(modelsamplingsd3_55, 0),
                positive=get_value_at_index(wanimagetovideo_63, 0),
                negative=get_value_at_index(wanimagetovideo_63, 1),
                latent_image=get_value_at_index(ksampleradvanced_57, 0),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampleradvanced_58, 0),
                vae=get_value_at_index(vaeloader_39, 0),
            )

            createvideo_60 = createvideo.create_video(
                fps=16, images=get_value_at_index(vaedecode_8, 0)
            )

            savevideo_61 = savevideo.save_video(
                filename_prefix="video/ComfyUI",
                format="auto",
                codec="auto",
                video=get_value_at_index(createvideo_60, 0),
            )


if __name__ == "__main__":
    main()
