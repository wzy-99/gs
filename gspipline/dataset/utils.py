import numpy as np
from typing import Any
from PIL import Image

def normalize_intrinsics(intrinsics: np.ndarray, image_shape: tuple[int, int]):
    """ Normalize the intrinsics.
        Args:
            intrinsics (np.ndarray): a numpy array containing the intrinsics, which can be either (..., 4, 4).
            image_shape (tuple): a tuple containing the shape of the image, which can be either (H, W).
        Returns:
            A numpy array containing the normalized intrinsics.
        """
    intrinsics[..., 0, 0] /= image_shape[1]
    intrinsics[..., 1, 1] /= image_shape[0]
    intrinsics[..., 0, 2] /= image_shape[1]
    intrinsics[..., 1, 2] /= image_shape[0]
    return intrinsics


def denormalize_intrinsics(intrinsics: np.ndarray, image_shape: tuple[int, int]):
    """ Denormalize the intrinsics.
        Args:
            intrinsics (np.ndarray): a numpy array containing the intrinsics, which can be either (..., 4, 4).
            image_shape (tuple): a tuple containing the shape of the image, which can be either (H, W).
        Returns:
            A numpy array containing the denormalized intrinsics.
        """
    intrinsics[..., 0, 0] *= image_shape[1]
    intrinsics[..., 1, 1] *= image_shape[0]
    intrinsics[..., 0, 2] *= image_shape[1]
    intrinsics[..., 1, 2] *= image_shape[0]
    return intrinsics


def center_crop_image(images: np.ndarray, intrinsics: np.ndarray | None = None):
    """ Crop the image to the square shape.
        Args:
            images (np.ndarray): a numpy array containing the images, which can be either (B, H, W, C) or (H, W, C) or (H, W).
            intrinsics (np.ndarray): a numpy array containing the intrinsics, which can be either (..., 4, 4).
        Returns:
            A numpy array containing the cropped images and a numpy array containing the updated intrinsics.
        """
    if len(images.shape) == 3:
        height, width, _ = images.shape
    elif len(images.shape) == 4:
        _, height, width, _ = images.shape
    elif len(images.shape) == 2:
        height, width = images.shape
    else:
        raise ValueError("The shape of the images is not valid.")
    min_size = min(height, width)
    top = (height - min_size) // 2
    left = (width - min_size) // 2
    bottom = top + min_size
    right = left + min_size
    if len(images.shape) == 3:
        images = images[top:bottom, left:right, :]
    elif len(images.shape) == 4:
        images = images[:, top:bottom, left:right, :]
    else:
        images = images[top:bottom, left:right]
    if intrinsics is not None:
        intrinsics[..., 0, 2] -= left
        intrinsics[..., 1, 2] -= top
    return images, intrinsics


def resize_image(images: np.ndarray, intrinsics: np.ndarray | None = None, size: int | tuple[int, int] = 256):
    """ Resize the image to the given size.
        Args:
            images (np.ndarray): a numpy array containing the images, which can be either (B, H, W, C) or (H, W, C) or (H, W).
            intrinsics (np.ndarray): a numpy array containing the intrinsics, which can be either (..., 4, 4).
            size (int): the size of the resized image.
        Returns:
            A numpy array containing the resized images and a numpy array containing the updated intrinsics.
    """
    if isinstance(size, int):
        size = (size, size)
    if len(images.shape) == 4:
        # (B, H, W, C)
        batch_size, original_height, original_width, channels = images.shape
    elif len(images.shape) == 3:
        # (H, W, C)
        original_height, original_width, channels = images.shape
        batch_size = 1
        images = images[np.newaxis, ...]
    elif len(images.shape) == 2:
        # (H, W)
        original_height, original_width = images.shape
        channels = 1
        batch_size = 1
        images = images[np.newaxis, ..., np.newaxis]
    else:
        raise ValueError("Unsupported image shape. Must be (B, H, W, C), (H, W, C), or (H, W).")
    
    resized_images = []

    for i in range(batch_size):
        img = Image.fromarray(images[i])
        img_resized = img.resize(size)
        resized_images.append(np.array(img_resized))

    resized_images = np.array(resized_images)

    if intrinsics is not None:
        scale_factor_h = size[1] / original_height
        scale_factor_w = size[0] / original_width

        intrinsics[..., 0, 0] *= scale_factor_w
        intrinsics[..., 1, 1] *= scale_factor_h
        intrinsics[..., 0, 2] *= scale_factor_w
        intrinsics[..., 1, 2] *= scale_factor_h

    return resized_images, intrinsics


def align_coordinate(cam_to_worlds: list[np.ndarray], reference_cam_to_world: np.ndarray):
    """ Align the views to the reference view.
        Args:
            cam_to_worlds (list): a list of dictionaries containing the extrinsics of the views.
            reference_cam_to_world (np.ndarray): a array containing the extrinsics of the reference view.
        Returns:
            A list of dictionaries containing the aligned views.
        """
    aligned_cam_to_worlds = []
    world_to_ref_cam = np.linalg.inv(reference_cam_to_world)
    for cam_to_world in cam_to_worlds:
        cam_to_ref_cam = np.dot(world_to_ref_cam, cam_to_world)
        aligned_cam_to_worlds.append(cam_to_ref_cam)
    return aligned_cam_to_worlds


def sample_n(sequence: list[Any], n: int, max_range: int, min_range: int = 1, sort: bool = True) -> list[int]:
    """ Sample n integers from m with a maximum interval.
        Args:
            sequence (list): a list of any objects.
            n (int): the number of integers to sample.
            max_range (int): the maximum interval.
            min_range (int): the minimum interval.
        Returns:
            A list of n integers.
        """
    if n > len(sequence):
        raise ValueError("n should be less than or equal to m.")
    if max_range < min_range:
        raise ValueError("max range should be greater than or equal to min range.")
    if max_range < n:
        raise ValueError("max range should be greater than or equal to n.")
    
    min_range = max(min_range, n)
    intervals = np.random.randint(min_range, max_range + 1, 1)
    start = np.random.randint(0, len(sequence) - intervals + 1)
    end = start + intervals
    indices = np.random.choice(np.arange(start, end), n, replace=False)
    if sort:
        indices.sort()
    return [sequence[i] for i in indices]


def normalize_extrinsics(extrinsics: np.ndarray, extrinsics_A: np.ndarray, extrinsics_B: np.ndarray) -> np.ndarray:
    """ Normalize the extrinsics.
        Args:
            extrinsics (np.ndarray): a numpy array containing the extrinsics, which can be either (..., 4, 4).
            extrinsics_A (np.ndarray): a numpy array containing the extrinsics of view A, which can be either (4, 4).
            extrinsics_B (np.ndarray): a numpy array containing the extrinsics of view B, which can be either (4, 4).
        Returns:
            A numpy array containing the normalized extrinsics.
        """
    pose_A = extrinsics_A[:3, 3]
    pose_B = extrinsics_B[:3, 3]
    scale = np.linalg.norm(pose_B - pose_A)
    extrinsics[..., :3, 3] -= pose_A
    extrinsics[..., :3, 3] /= scale
    return extrinsics