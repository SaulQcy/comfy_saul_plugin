from comfy.comfy_types.node_typing import ComfyNodeABC, IO, InputTypeDict, TypedDict
import torch
from torchvision.transforms.transforms import Resize

class ExtractFirstFrame(ComfyNodeABC):
    DESCRIPTION = 'Extract the first from the frames'
    CATEGORY = 'Saul'

    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            'required': {
                'image_in': (IO.IMAGE, {}),
            }
        }
    
    RETURN_NAMES = ('First Frame', )
    RETURN_TYPES = (IO.IMAGE, )

    FUNCTION = 'main'

    def main(self, image_in: torch.Tensor) -> tuple[torch.Tensor, ]:
        ret = image_in[0]
        return ret.unsqueeze(0), 

class EndNode(ComfyNodeABC):
    DESCRIPTION = 'End Node given any inputs'
    CATEGORY = "Saul"

    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                'any': (IO.ANY, {})
            }
        }
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "main"
    OUTPUT_NODE = True

    def main(self, any):
        print("Received input at EndNode:", any)
        return {}
    

class BlendImage(ComfyNodeABC):
    DESCRIPTION = 'Given two Image, blend them.'
    CATEGORY = "Saul"

    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "image_1": (IO.IMAGE, {}),
                "image_2": (IO.IMAGE, {}),
                "blend_coeff": (IO.FLOAT, {'min': 0., "max": 1., "step": 0.1, "default":1.})
            }
        }
    
    RETURN_NAMES = ("image_out", )
    RETURN_TYPES = (IO.IMAGE, )
    OUTPUT_NODE = False
    FUNCTION = "main"

    def main(self, image_1, image_2, blend_coeff):
        N, H, W, C = image_1.shape
        image_2 = Resize((H, W))(image_2.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        print(image_1.shape)
        print(image_2.shape)
        assert image_1.shape == image_2.shape

        ret = image_1 + image_2 * blend_coeff
        print(ret.shape)
        return ret,


def get_folder_file_name():
    from datetime import datetime
    folder_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return folder_name

