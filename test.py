import torch
import clip
from PIL import Image
import pdb

# from clip.simple_tokenizer import SimpleTokenizer
# model.byte_encoder


device = "cuda" if torch.cuda.is_available() else "cpu"

# model, preprocess = clip.load("ViT-B/32", device=device)
model, preprocess = clip.load("./download/ViT-B-32.pt", device=device)

model = model.eval()
# model -- CLIP(...)


# model = torch.jit.script(model)
# model = torch.compile(model)

# model.visual
# model.self.transformer


# model = torch.compile(model)

# model -- CLIP
# preprocess -- 
# Compose(
#     Resize(size=224, interpolation=bicubic, max_size=None, antialias=warn)
#     CenterCrop(size=(224, 224))
#     ToTensor()
#     Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
# )

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
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

print("ViT-B-32:", "[[0.9927937  0.00421068 0.00299572]]")
print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
