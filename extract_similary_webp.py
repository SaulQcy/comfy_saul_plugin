from comfy.comfy_types.node_typing import ComfyNodeABC, IO, InputTypeDict, TypedDict
from PIL import Image, ImageSequence
import numpy as np
import os
import torch
from custom_nodes.comfy_saul_plugin.kpts_similarity import compute_pose_similarity
from custom_nodes.comfyui_controlnet_aux.node_wrappers.dwpose import DWPose_Preprocessor

class ExtractFilesGivenFolder(ComfyNodeABC):
    DESCRIPTION = "Given a folder, extract all the .webp files and their filenames. " \
    "Furthermore, extract the first frame for further processes."
    CATEGORY = "Saul"

    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                'path': (IO.STRING, {"default": "/home/saul/AIGC/sleep_pose_video/"}),
                'suffix': (IO.STRING, {"default": ".webp"}),
            }
        }
    
    RETURN_NAMES = ("file_name_list", )
    RETURN_TYPES = (IO.ANY, )
    OUTPUT_NODE = False
    FUNCTION = "main"

    def main(self, path, suffix):
        folder = path
        file_ls = [f'{folder}/{f}' for f in os.listdir(folder) if f.endswith(str(suffix))]
        # # print(file_ls)
        # # print(file_ls)
        # ret = {}
        # for f in file_ls:
        #     x = Image.open(f).convert("RGB")
        #     x = np.array(x).astype(float) / 255.
        #     x = torch.from_numpy(x)
        #     ret[f] = x
        # # print(ret)
        return file_ls,


class FindMostSimilarWebp(ComfyNodeABC):
    DESCRIPTION = "Given the path_and_tensor dict., and a target image," \
    "find the most similar images, and load the image sequence as return."
    CATEGORY = "Saul"

    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "target_image": (IO.IMAGE, {}),
                "file_name_ls": (IO.ANY, {}),
            }
        }
    
    OUTPUT_NODE = False
    RETURN_NAMES = ("finded_webp", )
    RETURN_TYPES = (IO.IMAGE, )
    FUNCTION = "main"

    threshold = 0.97

    def main(self, target_image, file_name_ls):
        print(file_name_ls)
        max_similarity = -1
        most_similar_filename = None
        for filename in file_name_ls:
            # load images
            x = Image.open(filename).convert("RGB")
            x = np.array(x).astype(float) / 255.
            x = torch.from_numpy(x)
            if x.shape[0] != 1:
                x = x.unsqueeze(0)

            # calculate pose
            pose_processor = DWPose_Preprocessor()
            pose_A = pose_processor.estimate_pose(x, "disable", "disable", "enable")
            pose_B = pose_processor.estimate_pose(target_image, "disable", "disable", "enable")

            # calculate pose similarity
            # print(pose_A['result'][-1])
            sim_val = compute_pose_similarity(pose_A['result'][-1], pose_B['result'][-1])
            # print(sim_val)

            # update most similar filename
            if sim_val > max_similarity:
                max_similarity = sim_val
                most_similar_filename = filename
        print(f"find the most similar .webp file ({most_similar_filename}), \ncorresponding similarity is {max_similarity: .2f}")  
        if max_similarity < self.threshold:
            import warnings
            warnings.warn(f"Similarity is below {self.threshold}, the video might be not suitable!")
        if most_similar_filename is None:
            raise ValueError("Do not find the video!")
        frames = []
        img = Image.open(most_similar_filename)
        for frame in ImageSequence.Iterator(img):
            frames.append(np.array(frame.convert('RGB')))
        webp_np = np.stack(frames).astype(float) / 255.
        webp_tc = torch.from_numpy(webp_np)
        print(webp_tc.shape)
        return (webp_tc, )
