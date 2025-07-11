from .video_cutting_node import ComfyVideoCutting
from .vitpose import VitPoseProcess
from .yolov3_smoking_det import SmokingAutoLabel
from .fuse_cigarette_people import FuseCigarettePeople

NODE_CLASS_MAPPINGS = {
    "Cutting Video": ComfyVideoCutting,
    "Get Pose": VitPoseProcess,
    "Smoking Auto Label": SmokingAutoLabel,
    "Fuse People and Cigarette": FuseCigarettePeople,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Saul-Plugin": "Cutting-Video",
    "Saul-Plugin": "VitPose-Default",
    "Saul-Plugin": "Smoking Auto Label",
    "Saul-Plugin": "Fuse Cigarette and People",

}