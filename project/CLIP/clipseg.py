import os
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from CLIP.clip import CLIP
from typing import List, Tuple

import pdb

class CLIPSeg(nn.Module):
    def __init__(
        self,
        version="ViT-B-16",
        extract_layers=(0, 3, 6, 9),
        cond_layer=0,
        reduce_dim=64,
        n_heads=4,
    ):
        super().__init__()
        self.MAX_H = 1024
        self.MAX_W = 1024
        # self.MAX_TIMES = 16

        self.version = version
        self.extract_layers = extract_layers
        self.cond_layer = cond_layer
        self.backbone = CLIP(version=version).eval()

        # film_mul, film_add, reduces, blocks, trans_conv
        self.film_mul = nn.Linear(512, reduce_dim)
        self.film_add = nn.Linear(512, reduce_dim)

        self.reduces = nn.ModuleList([nn.Linear(768, reduce_dim) for _ in range(len(extract_layers) - 1)])
        self.blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=reduce_dim, nhead=n_heads) for _ in range(len(self.extract_layers) - 1)]
        )

        self.token_shape = {"ViT-B-32": (7, 7), "ViT-B-16": (14, 14)}[version]
        trans_conv_ks = {"ViT-B-32": (32, 32), "ViT-B-16": (16, 16)}[version]
        tp_kernels = (trans_conv_ks[0] // 4, trans_conv_ks[0] // 4)
        self.trans_conv = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(reduce_dim, reduce_dim // 2, kernel_size=tp_kernels[0], stride=tp_kernels[0]),
            nn.ReLU(),
            nn.ConvTranspose2d(reduce_dim // 2, 1, kernel_size=tp_kernels[1], stride=tp_kernels[1]),               
        )
        self.image_normal = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

        self.update_weights()

        for param in self.parameters():
            param.requires_grad = False

    def update_weights(self, model_path="models/ViT-B-16-Seg.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path

        if os.path.exists(checkpoint):
            print(f"Loading model weight from {checkpoint} ...")
            update_state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
            target_state_dict = self.state_dict()
            for n, p in update_state_dict.items():
                if n.startswith("reduce."): # skip reduce ...
                    continue

                if n in target_state_dict.keys():
                    target_state_dict[n].copy_(p)
                else:
                    raise KeyError(n)
        else:
            print("-" * 32, "Warnning", "-" * 32)
            print(f"model weight file '{checkpoint}'' not exist !!!")


    def forward(self, image, text):
        ''' 
            text.size() -- [2, 77], text is tokens
        '''
        B, C, H, W = image.size()
        if H > self.MAX_H or W > self.MAX_W:
            image =  F.interpolate(image, size=(self.MAX_H, self.MAX_W), mode="bilinear", align_corners=True)
        else:
            image =  F.interpolate(image, size=(self.MAX_H//2, self.MAX_W//2), mode="bilinear", align_corners=True)

        image = self.image_normal(image)

        bs = text.shape[0]
        if bs > 1:
            image = image.repeat(bs, 1, 1, 1)

        cond = self.backbone.encode_text(text).to(torch.float32) # [2, 512]

        # self.extract_layers -- (0, 3, 6, 9)
        activations: List[torch.Tensor] = self.visual_forward(image)

        activation1 = activations[0]  # size() -- [485, 4, 768]
        activations = activations[1:]  # len(activation) -- 3
        activation2 = activations[::-1]
        # len(activation2) -- 3, len(self.blocks) -- 3, len(self.reduces) -- 3

        a = torch.zeros_like(image)
        for i, (block, reduce) in enumerate(zip(self.blocks, self.reduces)):
            if i == 0:
                a = reduce(activation2[i])
            else:
                a = reduce(activation2[i]) + a
            if i == self.cond_layer:
                a = self.film_mul(cond) * a + self.film_add(cond)
            a = block(a)

        a = a[1:].permute(1, 2, 0)  # rm cls token and -> BS, Feats, Tokens
        size = int(math.sqrt(a.shape[2]))
        a = a.view(bs, a.shape[1], size, size)
        a = self.trans_conv(a)

        masks = torch.sigmoid(a)  # size() -- [4, 1, 352, 352]

        masks =  F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=True)

        return masks


    def rescaled_pos_emb(self, new_size: Tuple[int, int]):
        # assert len(new_size) == 2
        # self.token_shape -- (14, 14)
        # self.backbone.visual.positional_embedding.size() -- [197, 768]
        a = self.backbone.visual.positional_embedding[1:].T.view(1, 768, 
            self.token_shape[0], self.token_shape[1])
        b = (
            F.interpolate(a, new_size, mode="bicubic", align_corners=False)
            .squeeze(0)
            .view(768, new_size[0] * new_size[1])
            .T
        )
        return torch.cat([self.backbone.visual.positional_embedding[:1], b])

    def visual_forward(self, image) -> List[torch.Tensor]:
        # (Pdb) image.size(), image.min(), image.max() -- [2, 3, 768, 768], -2.1179, 2.5911

        x = self.backbone.visual.conv1(image.type(self.backbone.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # self.backbone.visual.class_embedding.size() -- [768]
        x = torch.cat(
            [
                self.backbone.visual.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )

        standard_n_tokens = 197  # ==> 197 ?
        # x.shape -- torch.Size([2, 2305, 768])
        if x.shape[1] != standard_n_tokens:
            new_shape = int(math.sqrt(x.shape[1] - 1))
            x = x + self.rescaled_pos_emb((new_shape, new_shape)).to(x.dtype)[None, :, :]
        else:
            x = x + self.backbone.visual.positional_embedding.to(x.dtype)

        # ==> x.dtype = torch.float16
        x = self.backbone.visual.ln_pre(x.type(torch.float32))
        x = x.permute(1, 0, 2)  # NLD -> LND

        activations = []
        # len(self.backbone.visual.transformer.resblocks) -- 12
        for i, res_block in enumerate(self.backbone.visual.transformer.resblocks):
            # x = forward_multihead_attention(x, res_block)
            x_ = res_block.ln_1(x)

            q, k, v = F.linear(x_.type(self.backbone.dtype), res_block.attn.in_proj_weight, 
                res_block.attn.in_proj_bias).chunk(3, dim=-1)
            tgt_len, bsz, embed_dim = q.size()

            head_dim = embed_dim // res_block.attn.num_heads
            scaling = float(head_dim) ** -0.5

            q = q.contiguous().view(tgt_len, bsz * res_block.attn.num_heads, res_block.attn.head_dim).transpose(0, 1)
            k = k.contiguous().view(-1, bsz * res_block.attn.num_heads, res_block.attn.head_dim).transpose(0, 1)
            v = v.contiguous().view(-1, bsz * res_block.attn.num_heads, res_block.attn.head_dim).transpose(0, 1)

            q = q * scaling

            attn_output_weights = torch.bmm(q, k.transpose(1, 2))  #  n_heads * batch_size, tokens^2, tokens^2
            attn_output_weights = torch.softmax(attn_output_weights, dim=-1)

            attn_output = torch.bmm(attn_output_weights, v)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
            attn_output = res_block.attn.out_proj(attn_output)

            x = x + attn_output
            # ==> x.dtype == torch.float32
            x = x + res_block.mlp(res_block.ln_2(x).type(self.backbone.dtype)).to(torch.float32)

            # x.size() -- [2305, 2, 768]
            # attn_output_weights.size() -- [24, 2305, 2305]
            if i in self.extract_layers:
                activations += [x.to(torch.float32)]

        # x = x.permute(1, 0, 2)  # LND -> NLD
        # x = self.backbone.visual.ln_post(x[:, 0, :])

        # # self.backbone.visual.proj.size() -- [768, 512]
        # # if self.backbone.visual.proj is not None:  # True
        # #     x = x @ self.backbone.visual.proj
        # x = x @ self.backbone.visual.proj

        # # x.size() -- [2, 512]
        # len(activations) -- 4, all elements size() is [2305, 2, 768]
        return activations


if __name__ == '__main__':
    model = CLIPSeg()
    model = torch.jit.script(model)
    # print(model)
