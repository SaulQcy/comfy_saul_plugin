from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
import numpy as np
import cv2
import torch

class FuseCigarettePeople(ComfyNodeABC):
    DESCRIPTION = 'GIven three param., i.e., people face, face keypoints and cigarette. Patch cigarette to people face.'
    CATEGORY = 'Saul'

    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            'required': {
                'image': (IO.IMAGE, {}),
                'keypoint': (IO.ANY, {}),
                'cigarette': (IO.IMAGE, {}),
            }
        }
    
    RETURN_TYPES = (IO.IMAGE, )
    RETURN_NAMES = ('smoking people', )
    FUNCTION = 'main'

    def main(self, image, keypoint, cigarette):
        keypoint = keypoint[0].get('people')[0].get('face_keypoints_2d')
        face_kpts = np.array(keypoint).reshape(-1, 3)
        mouth_kpts = face_kpts[48:68, :2]  # 只取x, y
        xmin = int(np.min(mouth_kpts[:, 0]))
        ymin = int(np.min(mouth_kpts[:, 1]))
        xmax = int(np.max(mouth_kpts[:, 0]))
        ymax = int(np.max(mouth_kpts[:, 1]))
        print(xmin, ymin, xmax, ymax)
        x_center = int((xmax + xmin) / 2)
        y_center = int((ymax + ymin) / 2)


        # 1. 缩放香烟为嘴巴框一半大小
        target_w = int((xmax - xmin) * 0.5)
        target_h = int((ymax - ymin) * 1.2)

        cigarette = cigarette.squeeze().cpu()
        cigarette = (cigarette.numpy() * 255).astype(np.uint8)

        resized_cig = cv2.resize(cigarette, (target_w, target_h), interpolation=cv2.INTER_AREA)

        resized_cig = resized_cig.astype(float) / 255.
        resized_cig = torch.from_numpy(resized_cig)
        resized_cig = resized_cig.unsqueeze(0)

        _, img_h, img_w, _ = image.shape  # Full image dimensions
        _, cig_h, cig_w, _ = resized_cig.shape  # Cigarette image dimensions

        # Compute starting positions (right-top alignment)
        x_start = x_center - cig_w  # Shift left by width of resized_cig
        y_start = y_center          # Keep y_center as the top edge
        x_start = max(0, x_start)
        y_start = max(0, y_start)

        # Overlay resized_cig onto image (only where resized_cig is non-zero)
        mask = (resized_cig != 0)
        image[:, y_start:y_start + cig_h, x_start:x_start + cig_w, :] = torch.where(
            mask,
            resized_cig,
            image[:, y_start:y_start + cig_h, x_start:x_start + cig_w, :]
        )
        if image.shape[0] == 1:
            image = image.unsqueeze(0)
        print(f"fuse node output shape: {image.shape}")
        return image