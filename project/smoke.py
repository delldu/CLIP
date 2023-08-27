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
    model, device = CLIP.get_segment_model()

    N = 100
    B, C, H, W = 1, 3, 1024, 1024

    text = CLIP.tokenize(["a diagram", "a dog", "a cat"])

    mean_time = 0
    progress_bar = tqdm(total=N)
    for count in range(N):
        progress_bar.update(1)

        image = torch.randn(B, C, H, W)
        start_time = time.time()
        with torch.no_grad():
            y = model(image.to(device), text.to(device))
        torch.cuda.synchronize()
        mean_time += time.time() - start_time

    mean_time /= N
    print(f"Mean spend {mean_time:0.4f} seconds")
    os.system("nvidia-smi | grep python")


    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as p:
        for ii in range(100):
            image = torch.randn(B, C, H, W)
            start_time = time.time()
            with torch.no_grad():
                y = model(image.to(device), text.to(device))
            torch.cuda.synchronize()
        p.step()

    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    os.system("nvidia-smi | grep python")
    