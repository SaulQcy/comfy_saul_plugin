from .vitpose_utils import get_pose_result, load_vitpose_default
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
import torch
import numpy as np
import PIL.Image

class VitPoseProcess(ComfyNodeABC):

    DESCRIPTION = 'Given a Image, using VitPose to extract the key-points'
    CATEGORY = 'Saul'

    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            'required': {
                'image_in': (IO.ANY, {}),
            }
        }
    
    RETURN_TYPES = (IO.IMAGE, IO.IMAGE, IO.IMAGE)
    RETURN_NAMES = ('pose', 'pose_with_image', 'image')
    FUNCTION = 'main'

    def main(self, image_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        print(image_in.shape)
        img_np = (np.array(image_in) * 255.).astype(np.uint8)
        pose_ls = []
        pose_img_ls = []
        tmp1, tmp2 = load_vitpose_default()
        for i in range(img_np.shape[0]):
            img_pl = PIL.Image.fromarray(img_np[i])
            pose, pose_img, _ = get_pose_result(img_pl, tmp1, tmp2)
            pose = np.array(pose)
            pose = torch.from_numpy(pose / 255.)
            pose_img = np.array(pose_img)
            pose_img = torch.from_numpy(pose_img / 255.)
            pose_ls.append(pose)
            pose_img_ls.append(pose_img)
        pose = torch.stack(pose_ls, dim=0)
        pose_img = torch.stack(pose_img_ls, dim=0)
        return (pose, pose_img, image_in)
    

