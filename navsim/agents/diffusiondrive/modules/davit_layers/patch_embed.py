# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0.

from typing import Callable, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_hw = make_2tuple(img_size)
        patch_hw = make_2tuple(patch_size)
        patch_grid = (image_hw[0] // patch_hw[0], image_hw[1] // patch_hw[1])

        self.img_size = image_hw
        self.patch_size = patch_hw
        self.patches_resolution = patch_grid
        self.num_patches = patch_grid[0] * patch_grid[1]
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_hw, stride=patch_hw
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, height, width = x.shape
        patch_h, patch_w = self.patch_size
        assert height % patch_h == 0
        assert width % patch_w == 0

        x = self.proj(x)
        out_h, out_w = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, out_h, out_w, self.embed_dim)
        return x
