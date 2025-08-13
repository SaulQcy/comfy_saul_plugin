import PIL.Image
import numpy as np

img = PIL.Image.open("/home/saul/Desktop/202237219.gif").convert("RGB")
x_np = np.array(img)
print(x_np.shape, x_np.dtype, x_np.min(), x_np.max())

# x_np = np.ones_like(x_np) * 255 - x_np
x_np = x_np + 2
img = PIL.Image.fromarray(x_np)
PIL.Image.Image.save(img, "/home/saul/Desktop/202237219_inverted.gif")