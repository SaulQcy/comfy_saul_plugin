from ultralytics import YOLO
import PIL.Image
import torch
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize
import cv2
from comfy.comfy_types.node_typing import ComfyNodeABC, IO, InputTypeDict

class PatchBox(ComfyNodeABC):

    DESCRIPTION = 'Given an Image, using yolo11n-pose to box it.'
    CATEGORY = 'Saul'

    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            'required': {
                'image_in': (IO.IMAGE, {})
            }
        }
    
    RETURN_NAMES = ('image_out', )
    RETURN_TYPES = (IO.IMAGE, )
    FUNCTION = 'main'

    def main(self, image_in: torch.Tensor) -> tuple[torch.Tensor, ]:

        image_out, _ = demo(image_in.permute(0, 3, 1, 2))
        return image_out.permute(0, 2, 3, 1),


def demo(x):
    # Load model
    model = YOLO("/home/saul/comfy/ComfyUI/models/saul_model/yolo11n-pose.pt")

    # Load and transform image
    # img = PIL.Image.open('./bus.jpg').convert('RGB')
    # trans = Compose([Resize((640, 640)), ToTensor()])
    # x: torch.Tensor = trans(img).unsqueeze(0)

    # Inference
    b, c, h, w = x.shape
    x = Resize((640, 640))(x)
    results = model(x)

    # Convert image to OpenCV format for visualization (optional)
    img_cv = np.array(x.squeeze().permute(1, 2, 0))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    for result in results:
        keypoints = result.keypoints.data  # [num_people, num_keypoints, 3]
        if keypoints is None or len(keypoints) == 0:
            continue
        
        for person_kpts in keypoints:
            # person_kpts: [num_keypoints, 3] → [:, 0:2] is (x, y)
            visible_kpts = person_kpts[person_kpts[:, 2] > 0][:, 0:2]  # only visible keypoints
            if visible_kpts.shape[0] == 0:
                continue

            # Get bounding box from keypoints
            xmin = int(visible_kpts[:, 0].min())
            ymin = int(visible_kpts[:, 1].min())
            xmax = int(visible_kpts[:, 0].max())
            ymax = int(visible_kpts[:, 1].max())

            print(f"Bounding box: ({xmin}, {ymin}) - ({xmax}, {ymax})")

            # Optional: visualize
            print(img_cv.shape)
            cv2.rectangle(img_cv, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    x_box = torch.from_numpy(img_cv.transpose(2, 0, 1))
    x_box = Resize(((h, w)))(x_box)
    return x_box.unsqueeze(0), (xmin, ymin, xmax, ymax)

