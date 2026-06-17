from functools import partial
import math
from typing import Callable, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .davit_layers.attention import MemEffAttention
from .davit_layers.block import Block
from .davit_layers.mlp import Mlp
from .davit_layers.patch_embed import PatchEmbed
from .davit_layers.swiglu_ffn import SwiGLUFFNFused


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def named_apply(
    fn: Callable,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for block in self:
            x = block(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer in ("swiglu", "swiglufused"):
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            def f(*args, **kwargs):
                return nn.Identity()
            ffn_layer = f
        else:
            raise NotImplementedError(ffn_layer)

        blocks = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunk_size = depth // block_chunks
            for i in range(0, depth, chunk_size):
                chunked_blocks.append([nn.Identity()] * i + blocks[i : i + chunk_size])
            self.blocks = nn.ModuleList([BlockChunk(chunk) for chunk in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, width, height):
        prev_dtype = x.dtype
        n_patch = x.shape[1] - 1
        n_ref = self.pos_embed.shape[1] - 1
        if n_patch == n_ref and width == height:
            return self.pos_embed

        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        width = width // self.patch_size + self.interpolate_offset
        height = height // self.patch_size + self.interpolate_offset
        sqrt_ref = math.sqrt(n_ref)
        scale = (float(width) / sqrt_ref, float(height) / sqrt_ref)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_ref), int(sqrt_ref), dim).permute(0, 3, 1, 2),
            scale_factor=scale,
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(prev_dtype)

    def prepare_tokens(self, x):
        batch_size, _, width, height = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(batch_size, -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, width, height)
        if self.register_tokens is not None:
            x = torch.cat(
                (x[:, :1], self.register_tokens.expand(batch_size, -1, -1), x[:, 1:]),
                dim=1,
            )
        return x

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens(x)
        output = []
        total_len = len(self.blocks)
        blocks_to_take = range(total_len - n, total_len) if isinstance(n, int) else n
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in blocks_to_take:
                output.append(x)
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            batch_size, _, width, height = x.shape
            outputs = [
                out.reshape(batch_size, width // self.patch_size, height // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)


class DAViT(nn.Module):
    def __init__(self, encoder: str = "vitl", ckpt: str = None):
        super().__init__()
        if encoder != "vitl":
            raise ValueError(f"Only vitl is supported for now, got {encoder}")

        self.pretrained = DinoVisionTransformer(
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
            img_size=518,
            num_register_tokens=0,
            interpolate_antialias=False,
            interpolate_offset=0.1,
            block_fn=partial(Block, attn_class=MemEffAttention),
        )
        if ckpt:
            state_dict = torch.load(ckpt, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            valid_dict = {}
            for key, value in state_dict.items():
                if "depth_head" in key or "mask_token" in key:
                    continue
                key = key.replace(
                    "agent.vadv2_model._backbone.image_encoder.pretrained",
                    "pretrained",
                )
                valid_dict[key] = value
            self.load_state_dict(valid_dict, strict=False)

    def forward(self, x, return_class_token: bool = False):
        return self.pretrained.get_intermediate_layers(
            x,
            1,
            return_class_token=return_class_token,
            reshape=True,
        )
