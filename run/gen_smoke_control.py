import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import hydra.version




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



# def get_args():
#     p = argparse.ArgumentParser("smoking workflow argparser")
#     p.add_argument("--people", type=str, default=SMOKING_DICT["PEOPLE"])
#     # "No --cigarette" means the original people image has contained cigarette, no need other cigarette.
#     p.add_argument("--cigarette", type=str, default=SMOKING_DICT["CIGARETTE"])
#     p.add_argument("--positive", type=str, default=SMOKING_DICT["POSITIVE_PROMPT"])
#     p.add_argument("--negative", type=str, default=SMOKING_DICT["NEGATIVE_PROMPT"])
# 
    # return p

@hydra.main(version_base=None, config_path="./config/", config_name="smoke_control")
def main(config: DictConfig):
    print("-" * 20)
    print("Hydra config is:")
    print(config)
    print("-"*20)
    import_custom_nodes()
    print(config)
    with torch.inference_mode():
        cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        cliploader_38 = cliploader.load_clip(
            clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            type="wan",
            device="default",
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        # positive prompt
        cliptextencode_6 = cliptextencode.encode(
            text=config.positive,
            clip=get_value_at_index(cliploader_38, 0),
        )

        cliptextencode_7 = cliptextencode.encode(
            text=config.negative,
            clip=get_value_at_index(cliploader_38, 0),
        )

        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_37 = unetloader.load_unet(
            unet_name="wan2.1_fun_control_1.3B_bf16.safetensors", weight_dtype="default"
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_39 = vaeloader.load_vae(vae_name="wan_2.1_vae.safetensors")

        clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
        clipvisionloader_49 = clipvisionloader.load_clip(
            clip_name="clip_vision_h.safetensors"
        )

        # load the image with people
        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_97 = loadimage.load_image(
            image=config.people
        )

        dwpreprocessor = NODE_CLASS_MAPPINGS["DWPreprocessor"]()
        dwpreprocessor_90 = dwpreprocessor.estimate_pose(
            detect_hand="enable",
            detect_body="enable",
            detect_face="enable",
            resolution=512,
            bbox_detector="yolo_nas_m_fp16.onnx",
            pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt",
            scale_stick_for_xinsr_cn="disable",
            image=get_value_at_index(loadimage_97, 0),
        )

        # determine the first frame
        if config.cigarette == "None":
            print("The original people image has contained cigarette.")
            first_frame = get_value_at_index(loadimage_97, 0)
        else:
            loadimage_96 = loadimage.load_image(
                image=config.cigarette
            )

            fuse_people_and_cigarette = NODE_CLASS_MAPPINGS["Fuse People and Cigarette"]()
            fuse_people_and_cigarette_99 = fuse_people_and_cigarette.main(
                image=get_value_at_index(loadimage_97, 0),
                keypoint=get_value_at_index(dwpreprocessor_90, 1),
                cigarette=get_value_at_index(loadimage_96, 0),
            )
            first_frame = get_value_at_index(fuse_people_and_cigarette_99, 0)
        # print(first_frame)
        clipvisionencode = NODE_CLASS_MAPPINGS["CLIPVisionEncode"]()
        clipvisionencode_51 = clipvisionencode.encode(
            crop="none",
            clip_vision=get_value_at_index(clipvisionloader_49, 0),
            image=first_frame,
        )

        wanfuncontroltovideo = NODE_CLASS_MAPPINGS["WanFunControlToVideo"]()
        wanfuncontroltovideo_55 = wanfuncontroltovideo.encode(
            width=640,
            height=480,
            length=77,
            batch_size=1,
            positive=get_value_at_index(cliptextencode_6, 0),
            negative=get_value_at_index(cliptextencode_7, 0),
            vae=get_value_at_index(vaeloader_39, 0),
            clip_vision_output=get_value_at_index(clipvisionencode_51, 0),
            # start_image=get_value_at_index(fuse_people_and_cigarette_99, 0),
            start_image=first_frame,
            control_video=get_value_at_index(dwpreprocessor_90, 0),
        )

        skiplayerguidancedit = NODE_CLASS_MAPPINGS["SkipLayerGuidanceDiT"]()
        modelsamplingsd3 = NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
        unettemporalattentionmultiply = NODE_CLASS_MAPPINGS[
            "UNetTemporalAttentionMultiply"
        ]()
        cfgzerostar = NODE_CLASS_MAPPINGS["CFGZeroStar"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        smoking_auto_label = NODE_CLASS_MAPPINGS["Smoking Auto Label"]()

        for q in range(1):
            skiplayerguidancedit_65 = skiplayerguidancedit.skip_guidance(
                double_layers="9,10",
                single_layers="9,10",
                scale=3,
                start_percent=0.01,
                end_percent=0.8000000000000002,
                rescaling_scale=0,
                model=get_value_at_index(unetloader_37, 0),
            )

            modelsamplingsd3_67 = modelsamplingsd3.patch(
                shift=5.000000000000001,
                model=get_value_at_index(skiplayerguidancedit_65, 0),
            )

            unettemporalattentionmultiply_68 = unettemporalattentionmultiply.patch(
                self_structural=1,
                self_temporal=1,
                cross_structural=1.2,
                cross_temporal=1.3,
                model=get_value_at_index(modelsamplingsd3_67, 0),
            )

            cfgzerostar_66 = cfgzerostar.patch(
                model=get_value_at_index(unettemporalattentionmultiply_68, 0)
            )

            ksampler_3 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=6,
                sampler_name="uni_pc",
                scheduler="simple",
                denoise=1,
                model=get_value_at_index(cfgzerostar_66, 0),
                positive=get_value_at_index(wanfuncontroltovideo_55, 0),
                negative=get_value_at_index(wanfuncontroltovideo_55, 1),
                latent_image=get_value_at_index(wanfuncontroltovideo_55, 2),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampler_3, 0),
                vae=get_value_at_index(vaeloader_39, 0),
            )

            smoking_auto_label_94 = smoking_auto_label.main(
                image_in=get_value_at_index(vaedecode_8, 0)
            )


if __name__ == "__main__":
    main()
