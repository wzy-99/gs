from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, List, Literal
import torch
import json
import os
from PIL import Image
import numpy as np

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.scripts.render import get_path_from_json


@dataclass
class BaseRenderConfig(InstantiateConfig):
    """Configuration for rendering point clouds."""

    _target: Type = field(default_factory=lambda: BaseRender)
    
    camera_path_filename: Path = Path("camera_path.json")
    """Filename of the camera path to render."""

    output_path: Path = Path("render")
    """Path to the output image file."""

    output_format: Literal["images", "video", "both"] = "both"
    """How to save output data."""

    image_format: Literal["jpeg", "png"] = "jpeg"
    """Image format"""


class BaseRender:
    """Class for rendering point clouds."""
    def __init__(self, config: BaseRenderConfig):
        self.config = config
        os.makedirs(self.config.output_path, exist_ok=True)
        self.path = self.config.camera_path_filename

    def load_tracks(self, path: Path) -> Cameras:
        """Loads the tracks from the camera path file."""
        with open(path, "r", encoding="utf-8") as f:
            camera_path = json.load(f)
        camera_path = get_path_from_json(camera_path)
        return camera_path
    
    def save_image(self, img: np.ndarray, name: str):
        """Saves the image to the output path.
        Args:
            img (np.ndarray): Image to save. Has shape (H, W, 3).
        """
        if self.config.output_format in ["images", "both"]:
            if img.dtype == np.float32:
                img = (img * 255).astype(np.uint8)
            output_path = self.config.output_path / f"{name}.{self.config.image_format}"
            Image.fromarray(img).save(output_path)
    
    def save_video(self, imgs: List[torch.Tensor], name: str):
        ...