import math
from typing import Tuple, Optional, Dict, Union, Literal

import torch
from torch import Tensor
from torch.nn import Module

from gsplat import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from dataclasses import dataclass, field


def compute_sh_degree(k: int) -> int:
    """Compute the SH degree for a given number of harmonics.
        Args:
            k: number of harmonics.
        Returns:
            The SH degree.
    """
    return int(math.sqrt(k) - 1)


@dataclass  
class GaussianSplattingConfig:
    antialiased: bool = False
    """Whether to use antialiasing."""
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    packed: bool = False
    """Use packed mode for rasterization, this leads to less memory usage but slightly slower."""
    sparse_grad: bool = False
    """Use sparse gradients for optimization. (experimental)"""
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    """Camera model."""
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB"
    """Render mode."""


class GaussianSplattingRender(Module):
    config: GaussianSplattingConfig

    def __init__(self, config: GaussianSplattingConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def forward(
        self,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor,
        colors: Tensor,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        need_active: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        if need_active:
            scales = torch.exp(scales)  # [N,]
            opacities = torch.sigmoid(opacities)  # [N,]

        rasterize_mode = "antialiased" if self.config.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            sh_degree=compute_sh_degree(colors.shape[-2]),
            packed=self.config.packed,
            absgrad=(
                self.config.strategy.absgrad
                if isinstance(self.config.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.config.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=False,
            camera_model=self.config.camera_model,
            render_mode=self.config.render_mode,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info