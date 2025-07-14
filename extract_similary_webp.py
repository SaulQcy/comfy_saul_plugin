from comfy.comfy_types.node_typing import ComfyNodeABC, IO, InputTypeDict, TypedDict
from PIL import Image
import numpy as np
import os
import torch

class ExtractWebpGivenFolder(ComfyNodeABC):
    DESCRIPTION = "Given a folder, extract all the .webp files and their filenames. " \
    "Furthermore, extract the first frame for further processes."
    CATEGORY = "Saul"

    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                'path': (IO.STRING, {}),
            }
        }
    
    RETURN_NAMES = ("path_and_images", )
    RETURN_TYPES = (IO.ANY, )
    OUTPUT_NODE = False
    FUNCTION = "main"

    def main(self, path):
        folder = path
        file_ls = [f'{folder}/{f}' for f in os.listdir(folder) if f.endswith('.webp')]
        print(file_ls)
        print(file_ls)
        ret = {}
        for f in file_ls:
            x = Image.open(f).convert("RGB")
            x = np.array(x).astype(float) / 255.
            x = torch.from_numpy(x)
            ret[f] = x
        print(ret)
        return ret


class FindMostSimilarWebp(ComfyNodeABC):
    DESCRIPTION = "Given the path_and_tensor dict., and a target image," \
    "find the most similar images, and load the image sequence as return."
    CATEGORY = "Saul"

    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "target_image": (IO.IMAGE, {}),
                "dict": (IO.ANY, {}),
            }
        }
    
    OUTPUT_NODE = False
    RETURN_NAMES = ("finded_webp", )
    RETURN_TYPES = (IO.IMAGE, )
    FUNCTION = "main"

    def main(self, target_image, dict):
        pass
