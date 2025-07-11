from comfy.comfy_types.node_typing import ComfyNodeABC, IO, InputTypeDict, TypedDict
import torch

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


