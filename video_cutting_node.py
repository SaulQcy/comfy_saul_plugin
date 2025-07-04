import subprocess
import os
import time
import PIL.Image
import imageio
import torch
import numpy as np
import PIL

class VideoPathToWebPNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {"default": "./video.mp4"}),
                "start_second": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 9999.0, "step": 0.1}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 600.0, "step": 0.1}),
                "output_name": ("STRING", {"default": "output.webp"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_images",)
    FUNCTION = "convert_to_webp"
    CATEGORY = "Saul-Plugin/Video"

    def convert_to_webp(self, input_path: str, start_second: float, duration: float, output_name: str):
        temp_cut = "/tmp/temp_cut.mp4"
        output_webp = os.path.join("/tmp", output_name)

        # Step 1: Cut video using ffmpeg
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(start_second), "-i", input_path,
            "-t", str(duration), "-c:v", "libx264", "-c:a", "aac", temp_cut
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.1)  # Wait for IO completion

        # Step 2: Convert to .webp animation
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_cut,
            "-vf", "fps=10,scale=320:-1:flags=lanczos", "-loop", "0", output_webp
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.1)

        imgs = PIL.Image.open(output_webp)
        return (imgs_np,)
