from .video_cutting_node import ComfyVideoCutting
from .vitpose import VitPoseProcess
from .yolov3_smoking_det import SmokingAutoLabel
from .fuse_cigarette_people import FuseCigarettePeople
from .yolov11_pose_det import PatchBox
from .img_tools import ExtractFirstFrame, EndNode, BlendImage
from .kpts_similarity import KeypointsSimilarity
from .extract_similary_webp import ExtractFilesGivenFolder, FindMostSimilarWebp

NODE_CLASS_MAPPINGS = {
    "Cutting Video": ComfyVideoCutting,
    "Get Pose": VitPoseProcess,
    "Smoking Auto Label": SmokingAutoLabel,
    "Fuse People and Cigarette": FuseCigarettePeople,
    "Patch Pose to People": PatchBox,
    "Extract the First Frame": ExtractFirstFrame,
    "Compute Keypoints Similarity": KeypointsSimilarity,
    "End Node": EndNode,
    "Blend Images": BlendImage,
    "Extract .webp from Folder": ExtractFilesGivenFolder,
    "Find the most similar webp": FindMostSimilarWebp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Saul-Plugin": "Cutting-Video",
    "Saul-Plugin": "VitPose-Default",
    "Saul-Plugin": "Smoking Auto Label",
    "Saul-Plugin": "Fuse Cigarette and People",
    "Saul-Plugin": "Patch Pose to People",
    "Saul-Plugin": "Extract the First Frame",
    "Saul-Plugin": "Compute Keypoints Similarity",
    "Saul-Plugin": "End Node",
    "Saul-Plugin": "Blend Images",
    "Saul-Plugin": "Extract .webp from Folder",
    "Saul-Plugin": "Find The Most Similar .webp",
}