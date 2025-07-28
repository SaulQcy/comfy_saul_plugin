# from custom_nodes.comfyui_controlnet_aux.node_wrappers.dwpose import DWPose_Preprocessor
# import PIL.Image
# import numpy as np
# import torch

# pose_processor = DWPose_Preprocessor()

# img = PIL.Image.open('/home/saul/comfy/ComfyUI/custom_nodes/comfyui_controlnet_aux/examples/comfyui-controlnet-aux-logo.png').convert('RGB')
# x_np = np.array(img).astype(float) / 255.
# x = torch.from_numpy(x_np)
# x = x.unsqueeze(0)

# res = pose_processor.estimate_pose(x, "disable", "disable", "enable")
# x_pose = res['result'][0].shape

d = {
    1: 11,
    2: 22,
    3: 33,
}

for k, v in d.items():
    print(k, v)
