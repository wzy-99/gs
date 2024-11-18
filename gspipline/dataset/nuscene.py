import os
import numpy as np
from PIL import Image
from typing import Type

from torch.utils.data import Dataset

from pyquaternion import Quaternion

from nuscenes import NuScenes

from gspipline.dataset.base_dataset import BaseDataset, BaseDatasetConfig

from dataclasses import dataclass, field

@dataclass
class nuSceneConfig(BaseDatasetConfig):
    _target: Type = field(default_factory=lambda: nuSceneDataset)
    
    root_path: str = field(default='./data/nuScenes')
    """ The root path of the nuScenes dataset. """
    version: str = field(default='v1.0-mini')
    """ The version of the nuScenes dataset. """
    verbose: bool = False
    """ Whether to print verbose messages. """


class nuSceneDataset(BaseDataset):
    config: nuSceneConfig
    def __init__(self, config: nuSceneConfig):
        super().__init__(config)
        self.nusc = NuScenes(version=config.version, dataroot=config.root_path, verbose=config.verbose)
        self.cam_list = [          # {frame_idx}_{cam_id}.jpg
            "CAM_FRONT",        # "xxx_0.jpg"
            "CAM_FRONT_LEFT",   # "xxx_1.jpg"
            "CAM_FRONT_RIGHT",  # "xxx_2.jpg"
            "CAM_BACK_LEFT",    # "xxx_3.jpg"
            "CAM_BACK_RIGHT",   # "xxx_4.jpg"
            "CAM_BACK"          # "xxx_5.jpg"
        ]

    def __len__(self):
        return len(self.nusc.scene)

    def load_all_samples(self, scene_data: dict):
        """ Load all samples in a scene. 
            Args:
                scene_data: scene data from nuscenes-api
            Returns:
                A list of dictionaries containing the sample data.
        """
        first_sample_token, last_sample_token = scene_data['first_sample_token'], scene_data['last_sample_token']
        curr_sample_record = self.nusc.get('sample', first_sample_token)
        recs = []
        while True:
            recs.append(curr_sample_record)
            if curr_sample_record['next'] == '' or curr_sample_record['token'] == last_sample_token:
                break
            curr_sample_record = self.nusc.get('sample', curr_sample_record['next'])
        return recs
    
    def load_sample_image(self, sample_data: dict, cam_name: str):
        """ Load all image in a sample.
            Args:
                sampple_data (dict): sample data from nuscenes-api
                cam_name (str): camera name, e.g. CAM_FRONT
            Returns:
                A dictionary containing the image data.
                    - filename (str): the path to the image file.
                    - filepath (str): the absolute path to the image file.
                    - pil_image (PIL.Image): the image data in PIL format.
        """
        cam_data = self.nusc.get('sample_data', sample_data['data'][cam_name])
        source_path = os.path.join(self.nusc.dataroot, cam_data['filename'])
        return {
            'filename': cam_data['filename'],
            'filepath': source_path,
            'pil_image': Image.open(source_path)
        }
    
    def load_sample_calib(self, sample_data : dict, cam_name : str):
        """ Load the calibration data for a sample.
            Args:
                sample_data (dict): sample data from nuscenes-api
                cam_name (str): camera name, e.g. CAM_FRONT
            Returns:
                A dictionary containing the calibration data.
                    - extrinsics_cam_to_world (np.ndarray): 4x4 transformation matrix from camera to world coordinates.
                    - Ks (list): intrinsics of the camera.
        """
        cam_data = self.nusc.get('sample_data', sample_data['data'][cam_name])
        calib_data = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        # Extrinsics (camera to ego)
        extrinsics_cam_to_ego = np.eye(4)
        extrinsics_cam_to_ego[:3, :3] = Quaternion(calib_data['rotation']).rotation_matrix
        extrinsics_cam_to_ego[:3, 3] = np.array(calib_data['translation'])
        # Get ego pose (ego to world)
        ego_pose_data = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        ego_to_world = np.eye(4)
        ego_to_world[:3, :3] = Quaternion(ego_pose_data['rotation']).rotation_matrix
        ego_to_world[:3, 3] = np.array(ego_pose_data['translation'])
        # Transform camera extrinsics to world coordinates
        extrinsics_cam_to_world = ego_to_world @ extrinsics_cam_to_ego
        # Intrinsics
        intrinsics = np.array(calib_data['camera_intrinsic'])
        return {
            'extrinsics_cam_to_world': extrinsics_cam_to_world,
            'intrinsics': intrinsics
        }