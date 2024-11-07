from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, List, Literal
import torch
import json
import os
from PIL import Image
import numpy as np
import imageio

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
    
    def save_image(self, image: np.ndarray, name: str):
        """Saves the image to the output path.
        Args:
            image (np.ndarray): Image to save. Has shape (H, W, 3).
        """
        if self.config.output_format in ["images", "both"]:
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)
            output_path = self.config.output_path / 'images'
            os.makedirs(output_path, exist_ok=True)
            Image.fromarray(image).save(output_path / f'{name}.{self.config.image_format}')
    
    def save_video(self, images: List[np.ndarray], name: str):
        """Saves the video to the output path.
        Args:
            images (List[np.ndarray]): List of images to save. Each image has shape (H, W, 3).
            path (Path): Path to the output video file.
        """
        if self.config.output_format in ["video", "both"]:
            if images[0].dtype == np.float32:
                images = [image * 255 for image in images]
            output_path = self.config.output_path / 'videos'
            os.makedirs(output_path, exist_ok=True)
            imageio.mimsave(output_path / name, images, fps=30)