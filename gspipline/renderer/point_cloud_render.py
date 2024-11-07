from dataclasses import dataclass, field
import numpy as np
import torch
from pathlib import Path
from typing import Type, List, Literal

from nerfstudio.cameras.cameras import Cameras
from gspipline.renderer.base_render import BaseRender, BaseRenderConfig


@dataclass
class PointCloudRenderConfig(BaseRenderConfig):
    """Configuration for rendering point clouds."""

    _target: Type = field(default_factory=lambda: PointCloudRender)
    
    point_cloud_path: Path = Path("points.ply")
    """Path to the point cloud file to be rendered."""

    background_color: str | List[float] = "white"
    """Background color of the rendered image."""

    point_size: float = 0.01
    """Size of points in the rendered point cloud."""

    engine: Literal["pytorch3d"] = "pytorch3d"
    """Rendering engine to use for rendering the point cloud. Currently only "pytorch3d" is supported."""
    # TODO: Add support Open3D and other rendering engines.


def nerfstudio2pytorch3d(c2w: torch.Tensor):
    """Convert NerfStudio camera to PyTorch3D camera."""
    R_c2w = c2w[..., :3, :3]
    T_c2w = c2w[..., :3, 3]
    R_w2c = R_c2w.transpose(-1, -2)
    T_w2c = -R_w2c @ T_c2w.unsqueeze(-1)
    # RUB -> LUF
    Tr = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], device=c2w.device, dtype=c2w.dtype)
    R_w2c = Tr @ R_w2c
    T_w2c = Tr @ T_w2c
    R = R_w2c.transpose(-1, -2)
    T = T_w2c.squeeze(-1)
    return R, T


class PointCloudRender(BaseRender):
    """Class for rendering point clouds."""
    config: PointCloudRenderConfig
    device: torch.device

    def __init__(self, config: PointCloudRenderConfig):
        super().__init__(config)
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def render(self, points3D_xyz: torch.Tensor, points3D_rgb: torch.Tensor):
         """Render the point cloud."""
         if self.config.engine == "pytorch3d":
            from pytorch3d.structures import Pointclouds
            from pytorch3d.renderer import (
                PointsRasterizationSettings,
                PointsRenderer,
                PointsRasterizer,
                AlphaCompositor,
                PerspectiveCameras,
            )
            points3D_rgb = points3D_rgb / 255.0
            pcd = Pointclouds(points=[points3D_xyz], features=[points3D_rgb]).to(self.device)
            cameras = self.load_tracks(self.config.camera_path_filename)
            frames = []
            for i in range(len(cameras)):
                R, T = nerfstudio2pytorch3d(cameras[i].camera_to_worlds)

                cams = PerspectiveCameras(
                    focal_length=torch.tensor([[cameras[i].fx, cameras[i].fy]], device=self.device, dtype=torch.float32),
                    principal_point=torch.tensor([[cameras[i].cx, cameras[i].cy]], device=self.device, dtype=torch.float32),
                    in_ndc=False,
                    image_size=torch.tensor([[cameras[i].height, cameras[i].width]], device=self.device),
                    R=R.unsqueeze(0).to(self.device),
                    T=T.unsqueeze(0).to(self.device),
                    device=self.device,
                )

                raster_settings = PointsRasterizationSettings(
                    image_size=(cameras[i].height.numpy().tolist()[0], cameras[i].width.numpy().tolist()[0]),
                    radius=self.config.point_size,
                    points_per_pixel=10,
                    bin_size=0
                )

                renderer = PointsRenderer(
                    rasterizer=PointsRasterizer(cameras=cams, raster_settings=raster_settings),
                    compositor=AlphaCompositor()
                ).to(self.device)

                images: torch.Tensor = renderer(pcd) # B x H x W x 3

                image = images[0, ..., :3].cpu().numpy()
                frames.append(image)
                self.save_image(image, f"frame_{i:04d}")
            
            self.save_video(frames, "vedio.mp4")


    def main(self):
        """Render the point cloud."""
        cameras = self.load_tracks(self.config.camera_path_filename)
        if self.config.engine == "pytorch3d":
            from pytorch3d.renderer import (
                PointsRasterizationSettings,
                PointsRenderer,
                PointsRasterizer,
                AlphaCompositor,
                PerspectiveCameras,
            )
            from pytorch3d.io import (
                IO
            )
            pcd = IO().load_pointcloud(self.config.point_cloud_path, device=self.device)
            frames = []
            for i in range(len(cameras)):
                R, T = nerfstudio2pytorch3d(cameras[i].camera_to_worlds)

                cams = PerspectiveCameras(
                    focal_length=torch.tensor([[cameras[i].fx, cameras[i].fy]], device=self.device, dtype=torch.float32),
                    principal_point=torch.tensor([[cameras[i].cx, cameras[i].cy]], device=self.device, dtype=torch.float32),
                    in_ndc=False,
                    image_size=torch.tensor([[cameras[i].height, cameras[i].width]], device=self.device),
                    R=R.unsqueeze(0).to(self.device),
                    T=T.unsqueeze(0).to(self.device),
                    device=self.device,
                )

                raster_settings = PointsRasterizationSettings(
                    image_size=(cameras[i].height.numpy().tolist()[0], cameras[i].width.numpy().tolist()[0]),
                    radius=self.config.point_size,
                    points_per_pixel=10,
                    bin_size=0
                )

                renderer = PointsRenderer(
                    rasterizer=PointsRasterizer(cameras=cams, raster_settings=raster_settings),
                    compositor=AlphaCompositor()
                ).to(self.device)

                images = renderer(pcd)

                image = images[0, ..., :3].cpu().numpy()
                self.save_image(image, f"frame_{i:04d}")
                frames.append(image)
            
            self.save_video(frames, "vedio.mp4")