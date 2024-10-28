import time
import numpy as np
from pathlib import Path
from typing import Dict, Type, List, Optional
from PIL import Image

import viser
import viser.transforms as vtf

from gspipline.viewer.render_panel import populate_render_tab
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO
from nerfstudio.viewer_legacy.server.viewer_utils import get_free_port


from dataclasses import dataclass, field


@dataclass
class TrackViewerConfig(InstantiateConfig):
    """ Configuration for Track Viewer """
    _target: Type = field(default_factory=lambda: TrackViewer)

    # For ViserServer
    websocket_port: Optional[int] = None
    """The websocket port to connect to. If None, find an available port."""
    websocket_port_default: int = 7007
    """The default websocket port to connect to if websocket_port is not specified"""
    websocket_host: str = "0.0.0.0"
    """The host address to bind the websocket server to."""

    # For Render Panel
    datapath: Path = Path(".")
    """output directory for camera track"""

    # For camera
    H: int = 1080
    """height of the camera"""
    W: int = 1920
    """width of the camera"""
    focal_degree: float = 50.0
    """focal degree of the camera"""

    camera_frustum_scale: float = 0.1
    """Scale for the camera frustums in the viewer."""
    scale_factor: float = 1.0
    """Scale factor for the images."""
    point_size: float = 0.1
    """Size of the points in the point cloud."""


class TrackViewer:
    ready = False
    def __init__(self, config: TrackViewerConfig):
        self.need_update = False
        self.config = config
        self.scale_factor = config.scale_factor
        if self.config.websocket_port is None:
            websocket_port = get_free_port(default_port=self.config.websocket_port_default)
        else:
            websocket_port = self.config.websocket_port
        self.viser_server = viser.ViserServer(host=config.websocket_host, port=websocket_port)
        tabs = self.viser_server.gui.add_tab_group()
        with tabs.add_tab("Render", viser.Icon.CAMERA):
            self.render_tab_state = populate_render_tab(self.viser_server, datapath=config.datapath, H=config.H, W=config.W, fov_degrees=config.focal_degree)
        with tabs.add_tab("Scene", viser.Icon.SETTINGS):
            self.gui_point_size = self.viser_server.gui.add_number(
                "Point Size",
                self.config.point_size,
                min=0.01,
                max=10.0,
                step=0.01,
            )
            self.need_update_point_cloud = False
            @self.gui_point_size.on_update
            def _(_) -> None:
                self.need_update_point_cloud = True
        
    def init_scene(
        self,
        dataparser_outputs: DataparserOutputs,
    ) -> None:
        """Draw some images and the scene aabb in the viewer.

        Args:
            data (DataparserOutputs): The output of the dataparser.
        """
        self.dataparser_outputs = dataparser_outputs
        self.set_cameras(self.dataparser_outputs.cameras, self.dataparser_outputs.image_filenames)
        if 'points3D_xyz' in self.dataparser_outputs.metadata:
            self.set_point_cloud(self.dataparser_outputs.metadata['points3D_xyz'].cpu().numpy(),
                                self.dataparser_outputs.metadata['points3D_rgb'].cpu().numpy())
        self.ready = True

    def update_scene(self):
        self.viser_server.scene.reset()
        self.set_cameras(self.dataparser_outputs.cameras, self.dataparser_outputs.image_filenames)
        if 'points3D_xyz' in self.dataparser_outputs.metadata:
            self.set_point_cloud(self.dataparser_outputs.metadata['points3D_xyz'].cpu().numpy(),
                                self.dataparser_outputs.metadata['points3D_rgb'].cpu().numpy())
        self.need_update = True

    def get_numpy_image(self, image_filename: Path) -> np.ndarray:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        pil_image = Image.open(image_filename)
        if self.scale_factor != 1.0: 
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.Resampling.BILINEAR)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def set_cameras(self, cameras: Cameras, image_filenames: List[Path]):
        self.camera_handles: Dict[int, viser.CameraFrustumHandle] = {}
        for idx, (camera, image_filename) in enumerate(zip(cameras, image_filenames)):
            image = self.get_numpy_image(image_filename)
            R = vtf.SO3.from_matrix(camera.camera_to_worlds[:3, :3])
            R = R @ vtf.SO3.from_x_radians(np.pi)
            camera_handle = self.viser_server.scene.add_camera_frustum(
                name=f"/cameras/camera_{idx:05d}",
                fov=float(2 * np.arctan(camera.cx / camera.fx[0])),
                scale=self.config.camera_frustum_scale,
                aspect=float(camera.cx[0] / camera.cy[0]),
                image=image,
                wxyz=R.wxyz,
                position=camera.camera_to_worlds[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO,
            )
            @camera_handle.on_click 
            def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz

    def set_point_cloud(self, points3D_xyz, points3D_rgb):
        self.point_cloud_handle = self.viser_server.scene.add_point_cloud(
            "/gaussian_splatting_initial_points",
            points=points3D_xyz * VISER_NERFSTUDIO_SCALE_RATIO,
            colors=points3D_rgb,
            point_size=self.gui_point_size.value,
            point_shape="circle",
            visible=True,  # Hidden by default.
        )

    def run(self):
        while True:
            if self.need_update_point_cloud:
                self.point_cloud_handle.remove()
                self.set_point_cloud(self.dataparser_outputs.metadata['points3D_xyz'].cpu().numpy(),
                                     self.dataparser_outputs.metadata['points3D_rgb'].cpu().numpy())
                self.need_update_point_cloud = False
            time.sleep(1e-3)