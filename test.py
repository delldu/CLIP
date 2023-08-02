import torch
import clip
from PIL import Image
import numpy as np
import pdb


torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# model, preprocess = clip.load("./download/ViT-B-16.pt", device=device)
model, preprocess = clip.load("./download/ViT-B-32.pt", device=device)
# model, preprocess = clip.load("ViT-L/14", device=device)
model = model.to(device)

model = model.eval()
# model -- CLIP(...)

# model = torch.jit.script(model)
# torch.jit.script(model.visual)
# model = torch.compile(model)

# model = torch.compile(model)



image = preprocess(Image.open("CLIP.png").convert("RGB")).unsqueeze(0).to(device)
# image.size() -- [1, 3, 224, 224]

text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
# text.size() -- [3, 77]

with torch.no_grad():
    image_features = model.encode_image(image)
    # image_features.size() -- [1, 512]
    text_features = model.encode_text(text)
    # text_features.size() -- [3, 512]

    logits_per_image, logits_per_text = model(image, text)
    # logits_per_image -- tensor([[25.5625, 20.0938, 19.7500]], device='cuda:0', dtype=torch.float16)

    # logits_per_text
    # tensor([[25.5625],
    #         [20.0938],
    #         [19.7500]], device='cuda:0', dtype=torch.float16)

    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("ViT-B-32 Standard:", "[[0.9927937  0.00421068 0.00299572]]")
print("Probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


