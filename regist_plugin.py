from .video_cutting_node import ComfyVideoCutting
from .vitpose import VitPoseProcess

NODE_CLASS_MAPPINGS = {
    "Cutting Video": ComfyVideoCutting,
    "Get Pose": VitPoseProcess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Saul-Plugin": "Cutting-Video",
    "Saul-Plugin": "VitPose-Default",
}