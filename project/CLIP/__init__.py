"""Image Recognize Anything Model Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, Wed 02 Aug 2023 06:43:47 AM CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
from PIL import Image

import torch
import todos
from torchvision.transforms import Compose, ToTensor

from .clip import CLIP
from .simple_tokenizer import SimpleTokenizer

import pdb

torch.set_printoptions(sci_mode=False)


tokenizer = SimpleTokenizer()


def tokenize(texts, context_length: int = 77, truncate: bool = False):
    # texts = ['a diagram', 'a dog', 'a cat']

    if isinstance(texts, str):
        texts = [texts]

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    # (Pdb) sot_token -- 49406, eot_token -- 49407

    # (Pdb) all_tokens[0] -- [49406, 320, 22697, 49407]
    # (Pdb) all_tokens[1] -- [49406, 320, 1929, 49407]
    # (Pdb) all_tokens[2] -- [49406, 320, 2368, 49407]

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    # result.size() -- [3, 77], !!! context_length -- 77 !!!
    return result


def create_model(version):
    """
    Create model
    """

    model = CLIP(version=version)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    print(f"Running model {version} on {device} ...")

    return model, device


def get_model(version):
    """Load jit script model."""

    model, device = create_model(version)
    # print(model)

    model = torch.jit.script(model)
    todos.data.mkdir("output")
    torch_file_name = f"output/{version}.torch"
    if not os.path.exists(torch_file_name):
        model.save(torch_file_name)

    return model, device


def predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model_version = "ViT-B-32"
    model, device = get_model(model_version)
    transform = Compose([
        lambda image: image.convert("RGB"),
        ToTensor(),
    ])

    # load files
    image_filenames = todos.data.load_files(input_files)

    text = tokenize(["a diagram", "a dog", "a cat"]).to(device)

    # start predict
    results = []
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = model(image, text)

            results.append(f"File name: {filename}")
            results.append("ViT-B-32 Standard: [[0.9927937  0.00421068 0.00299572]]")
            results.append(f"{model_version} Probs: {probs.cpu()}")
            results.append("-" * 128)

    progress_bar.close()

    print("\n".join(results))

    todos.model.reset_device()
