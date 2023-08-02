# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, Wed 02 Aug 2023 06:43:47 AM CST
# ***
# ************************************************************************************/
#

import pdb
import os
import time
import torch
import CLIP

from tqdm import tqdm

if __name__ == "__main__":
    model, device = CLIP.get_model("ViT-B-32")

    N = 100
    B, C, H, W = 1, 3, 384, 384

    text = CLIP.tokenize(["a diagram", "a dog", "a cat"])

    mean_time = 0
    progress_bar = tqdm(total=N)
    for count in range(N):
        progress_bar.update(1)

        image = torch.randn(B, C, H, W)
        # print("image: ", image.size())

        start_time = time.time()
        with torch.no_grad():
            y = model(image.to(device), text.to(device))
        torch.cuda.synchronize()
        mean_time += time.time() - start_time

    mean_time /= N
    print(f"Mean spend {mean_time:0.4f} seconds")
    os.system("nvidia-smi | grep python")
