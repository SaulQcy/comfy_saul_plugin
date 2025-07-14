from PIL import Image
import numpy as np
import os
import torch

folder = '/home/saul/AIGC/sleep_pose_video/'
file_ls = [f'{folder}/{f}' for f in os.listdir(folder) if f.endswith('.webp')]
print(file_ls)
ret = {}
for f in file_ls:
    x = Image.open(f).convert("RGB")
    x = np.array(x).astype(float) / 255.
    x = torch.from_numpy(x)
    ret[f] = x

print(ret)