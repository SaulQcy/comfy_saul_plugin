from comfy.comfy_types.node_typing import ComfyNodeABC, IO, InputTypeDict, TypedDict
import torch
import hydra
from omegaconf import OmegaConf, DictConfig

class ChangeCameraPose(ComfyNodeABC):

    DESCRIPTION = "Given a face keypoints image, determine the camera pose"
    CATEGORY = "Saul"

    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "kps_img": (IO.IMAGE, {})
            }
        }
    
    RETURN_NAMES = ()
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "main"
    
    def main(self, kps_img: torch.Tensor):
        ret = 0.
        total = 0
        # print(kps_img.shape)
        n, h, w, c = kps_img.shape

        mask = kps_img > 0.1
        indices = torch.nonzero(mask)
        total = indices.size(0)
        ret = indices[:, 2].float().sum() / total if total > 0 else 0.

        # print(ret)
        cfg_path = '/home/saul/comfy/ComfyUI/custom_nodes/comfy_saul_plugin/run/config/smoke_camera.yaml'
        cfg = OmegaConf.load(cfg_path)
        if ret > w / 2:
            print("Pan Right")
            cfg.camera_pose = "Pan Right"
        else:
            print("Pan Left")
            cfg.camera_pose = "Pan Left"
        OmegaConf.save(cfg, cfg_path)
        return cfg.camera_pose, 