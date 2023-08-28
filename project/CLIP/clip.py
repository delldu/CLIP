from collections import OrderedDict
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import pdb


def load_weights(model, model_path):
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    if os.path.exists(checkpoint):
        print(f"Loading model weight from {checkpoint} ...")
        model.load_state_dict(torch.load(checkpoint))
    else:
        print("-" * 32, "Warnning", "-" * 32)
        print(f"model weight file '{checkpoint}'' not exist !!!")


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model) # LayerNorm(d_model), support torch.jit.script
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model) # LayerNorm(d_model), support torch.jit.script
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x.to(torch.float32)).to(x.dtype)) # support torch.jit.script
        x = x + self.mlp(self.ln_2(x.to(torch.float32)).to(x.dtype)) # support torch.jit.script
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) 
            for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, 
            stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width) # LayerNorm(width), support torch.jit.script

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = nn.LayerNorm(width) # LayerNorm(width), support torch.jit.script 
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x.to(torch.float32)).to(x.dtype) # support torch.jit.script

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :].to(torch.float32)).to(x.dtype) # support torch.jit.script

        # if self.proj is not None:
        #     x = x @ self.proj

        x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim = 512,
                 # vision
                 image_resolution = 224,
                 vision_layers = 12,
                 vision_width = 768,
                 vision_patch_size = 32,
                 # text
                 context_length = 77,
                 vocab_size = 49408,
                 transformer_width = 512,
                 transformer_heads = 8,
                 transformer_layers = 12,
                 version = "ViT-B-32"
                 ):
        super().__init__()
        if version == "ViT-B-16":
            vision_patch_size = 16
        elif version == "ViT-B-32":
            pass
        elif version == "ViT-L-14":
            embed_dim = 768
            vision_layers = 24
            vision_width = 1024
            vision_patch_size = 14
            transformer_width = 768
            transformer_heads = 12
        else:
            assert False, f"NO Support CLIP {version}"

        self.version = version
        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size # 49408
        self.token_embedding = nn.Embedding(vocab_size, transformer_width) # Embedding(49408, 512)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width)) # [77, 512]
        self.ln_final = nn.LayerNorm(transformer_width) # LayerNorm(transformer_width), support torch.jit.script

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim)) # [512, 512]
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) # 14.285714285714285 --> 2.6593
        # self.logit_scale.exp() -- 100.00 ?
        self.initialize_parameters()
        self.image_normal = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

        load_weights(self, f"models/{self.version}.pth")
        convert_weights(self) # reduce model size, torch.jit.script OK

        for param in self.parameters():
            param.requires_grad = False

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        image = self.image_normal(image)

        return self.visual(image.type(self.dtype)).to(torch.float32)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x.to(torch.float32)).type(self.dtype) # support torch.jit.script

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x.to(torch.float32)

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features, support torch.jit.script
        image_features = image_features / image_features.norm(p=2.0, dim=1, keepdim=True)
        text_features = text_features / text_features.norm(p=2.0, dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp() # 100.00 for ViT-B-32
        logits_per_image = logit_scale * image_features @ text_features.t()

        return logits_per_image.softmax(dim=-1)


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


if __name__ == '__main__':
    model = CLIP()
    model = torch.jit.script(model)
    print(model)
