import subprocess
import os
import time
import PIL.Image
import imageio
import torch
import numpy as np
import PIL
from comfy.comfy_types.node_typing import IO, InputTypeDict
from comfy.comfy_types.node_typing import ComfyNodeABC
from comfy_api.input.video_types import VideoInput
from comfy_api.input_impl.video_types import VideoFromFile



class ComfyVideoCutting(ComfyNodeABC):

    DESCRIPTION = 'Given a video input, output a cutting video (e.g., 50s -> 5s)'
    CATEGORY = 'Saul'

    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                'video_in': (IO.VIDEO, {}),
                't_min': (IO.FLOAT, {'min': 0., 'max': 1., 'step': 0.1, 'default': 0., }),
                't_max': (IO.FLOAT, {'min': 0., 'max': 1., 'step': 0.1, 'default': 0.2, }),
            }
        }
    
    OUTPUT_NODE = False
    RETURN_TYPES = (IO.IMAGE, )
    RETURN_NAMES = ("video_out", )

    FUNCTION = 'cutting_video'

    def cutting_video(self, video_in: VideoFromFile, t_min, t_max):
        all_frames = video_in.get_components().images
        l = len(all_frames)
        t_min = int(l * t_min)
        t_max = int(l * t_max)
        res_frames = all_frames[t_min:t_max]
        print(res_frames.shape)
        return res_frames

