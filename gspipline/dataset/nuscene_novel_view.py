import numpy as np
import torch

from gspipline.dataset.nuscene import nuSceneConfig, nuSceneDataset
from gspipline.dataset.utils import center_crop_image, sample_n, align_coordinate, resize_image, normalize_extrinsics

from dataclasses import dataclass

@dataclass
class nuSceneNovelViewConfig(nuSceneConfig):
    num_target_views: int = 1
    """Number of target views to be used for training."""
    num_context_views: int = 2
    """Number of context views to be used for training."""
    max_interval: int = 10
    """Maximum interval between the context views."""
    center_crop: bool = True
    """Whether to center crop the images."""
    normalize_extrinsics: bool = True
    """Whether to normalize the extrinsics."""
    image_size: int | tuple[int, int] = (256, 256)
    """Image size to be used for training, if -1, use the original image size."""



class nuSceneNovelViewDataset(nuSceneDataset):
    config: nuSceneNovelViewConfig

    def __init__(self, config: nuSceneNovelViewConfig):
        assert config.num_context_views == 2, "Currently, we only support two context views."
        if isinstance(config.image_size, int):
            config.image_size = (config.image_size, config.image_size)
        super().__init__(config)

    def __len__(self) -> int:
        return len(self.nusc.scene) * len(self.cam_list)

    def __getitem__(self, idx: int) -> dict:
        scene = self.nusc.scene[idx // len(self.cam_list)]
        cam = self.cam_list[idx % len(self.cam_list)]
        sample = self.load_sample(scene, cam)
        return sample

    def load_sample(self, sample_data: dict, cam_name: str) -> dict:
        """ Load a sample.
            Args:
                sample_data (dict): sample data from nuscenes-api
                cam_name (str): camera name, e.g. CAM_FRONT
            Returns:
                A dictionary containing the sample data.
                    1. 'image': [v, C, H, W]
                    2. 'extrinsics': [v, 4, 4]
                    3. 'intrinsics': [v, 3, 3]
                    4. 'novel_image': [n, C, H, W]
                    5. 'novel_extrinsics': [n, 4, 4]
                    6. 'novel_intrinsics': [n, 3, 3]
        """
        all_samples = self.load_all_samples(sample_data)

        assert len(all_samples) >= self.config.num_target_views + self.config.num_context_views, \
            f"Not enough views in {sample_data['token']} for {self.config.num_target_views + self.config.num_context_views} views."

        samples = sample_n(all_samples, self.config.num_target_views + self.config.num_context_views, self.config.max_interval)

        samples = [samples[0]] + [samples[-1]] + samples[1:1+self.config.num_target_views]

        images = [self.load_sample_image(sample, cam_name) for sample in samples]
        images = np.stack([np.array(image['pil_image']) for image in images])

        calibs = [self.load_sample_calib(sample, cam_name) for sample in samples]
        extrinsics = [calib['extrinsics_cam_to_world'] for calib in calibs]
        intrinsics = [calib['intrinsics'] for calib in calibs]
        extrinsics = align_coordinate(extrinsics, extrinsics[0])
        extrinsics = np.stack(extrinsics)
        intrinsics = np.stack(intrinsics)

        if self.config.center_crop:
            images, intrinsics = center_crop_image(images, intrinsics)

        if self.config.image_size[0] > 0 and (self.config.image_size != images.shape[1] or self.config.image_size != images.shape[2]):
            images, intrinsics = resize_image(images, intrinsics, self.config.image_size)

        if self.config.normalize_extrinsics:
            extrinsics = normalize_extrinsics(extrinsics, extrinsics[0], extrinsics[1])

        images = (torch.from_numpy(images).permute(0, 3, 1, 2) / 255.0).float() # [v, C, H, W]
        extrinsics = torch.from_numpy(extrinsics).float()
        intrinsics = torch.from_numpy(intrinsics).float()

        # Image to np
        return {
            'image': images[:self.config.num_context_views],
            'novel_image': images[self.config.num_context_views:],
            'extrinsics': extrinsics[:self.config.num_context_views],
            'novel_extrinsics': extrinsics[self.config.num_context_views:],
            'intrinsics': intrinsics[:self.config.num_context_views],
            'novel_intrinsics': intrinsics[self.config.num_context_views:],
            'width': images.shape[3],
            'height': images.shape[2],
        }