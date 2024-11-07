import gradio as gr
import numpy as np
from dataclasses import dataclass, field
from typing import Type, List, Tuple, Generator
from PIL import Image
import cv2
import torch
from pathlib import Path

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.build_sam import build_sam2_video_predictor

from nerfstudio.configs.base_config import InstantiateConfig


# points color and marker
COLORS = [(255, 0, 0), (0, 255, 0)]
POINT_SIZE = 5

SUPPORT_IMAGE_FORMATS = [".jpg", ".jpeg", ".png"]


@dataclass
class SegmentAnythingConfig:
    """Configuration for rendering point clouds."""

    checkpoint: str = "checkpoints/sam2.1_hiera_large.pt"
    """Path to the SAM2 checkpoint."""

    model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    """Path to the SAM2 model configuration file."""


@dataclass
class SegmentViewerConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: SegmentViewer)

    data_path: Path = Path("data/images")
    """The path to the directory containing the images."""

    output_path: Path = Path("data/mask")
    """The path to the directory to save the output."""

    predictor: SegmentAnythingConfig = field(default_factory=SegmentAnythingConfig)
    """The configuration of the segmentor."""

    server_port: int = 7860
    """The port of the server."""
    server_name: str = "0.0.0.0"
    """The name of the server."""


class SegmentViewer:
    """A Gradio interface for the Segment Anything model."""
    def __init__(self, config: SegmentViewerConfig):
        self.config = config
        self.config.output_path.mkdir(exist_ok=True, parents=True)
        images = sorted(list(self.config.data_path.glob("*")))
        self.image_paths = [img for img in images if img.suffix.lower() in SUPPORT_IMAGE_FORMATS]
        self.display_image = np.array(Image.open(self.image_paths[0]))
        self.display_image_idx = 0
        self.object_id = 1
        self.select_points: List[tuple[tuple[int, int], int, int, int]] = []
        self.setup()

    def setup(self):
        """Load the segmentor and initialize the display image."""
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self.predictor: SAM2VideoPredictor = build_sam2_video_predictor(self.config.predictor.model_cfg, self.config.predictor.checkpoint)
        self.inference_state: dict = self.predictor.init_state(str(self.config.data_path))

    def run(self) -> Generator[Tuple[Image.Image, List[Tuple[np.ndarray, str]]], None, None]:
        self.predictor.reset_state(self.inference_state)
        for (x, y), label, frame_idx, object_id in self.select_points:
            self.predictor.add_new_points_or_box(self.inference_state, 
                                                 frame_idx=frame_idx, 
                                                 points=np.array([[x, y]], dtype=np.float32), labels=np.array([label], dtype=np.int32), obj_id=object_id)
        
        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            self.save_mask(video_segments[out_frame_idx], self.image_paths[out_frame_idx].stem)
            yield (Image.open(self.image_paths[out_frame_idx]), [(mask.astype(np.uint8), str(obj_id)) for obj_id, mask in video_segments[out_frame_idx].items()])

    def get_mask(self, frame_idx: int) -> dict[int, np.ndarray]:
        """ Get the mask for the given frame index."""
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state, start_frame_idx=frame_idx, max_frame_num_to_track=1):
            return {
                out_obj_id: (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    def add_points(self, frame_idx: int, input_points: tuple[int, int], label: int) -> dict[int, np.ndarray]:
        """
        :param frame_idx int
        :param input_points (np array) (N, 2)
        return (H, W) mask, (H, W) logits
        """
        assert self.predictor is not None

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=1,
                points=np.array([[input_points[0], input_points[1]]], dtype=np.float32),
                labels=np.array([label]),
            )

        return {
                out_obj_id: (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
    
    def save_mask(self, masks: dict[str, np.ndarray], name: Path):
        """Save the mask to the output path."""
        for obj_id, mask in masks.items():
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            # mask_img.save(self.config.output_path / f"{name}_{obj_id}.png")
            mask_img.save(self.config.output_path / f"{name}.png")

    def on_select_point(self, point_type: str, evt: gr.SelectData) -> tuple[np.ndarray, List[tuple[Path, str]]]:
        """When the user clicks on the image, show points and update the mask."""
        if point_type == 'foreground_point':
            self.select_points.append((evt.index, 1, self.display_image_idx, self.object_id))   # append the foreground_point
        elif point_type == 'background_point':
            self.select_points.append((evt.index, 0, self.display_image_idx, self.object_id))    # append the background_point
        else:
            self.select_points.append((evt.index, 1, self.display_image_idx, self.object_id))    # default foreground_point
        return self.draw(), (self.display_image, self.draw_mask())

    def on_undo_point(self) -> np.ndarray:
        """Undo the last point selection."""
        if len(self.select_points) > 0:
            _, _, frame_idx, _ = self.select_points.pop()
            if frame_idx != self.display_image_idx:
                self.display_image_idx = frame_idx
                self.display_image = np.array(Image.open(self.image_paths[frame_idx]))
        return self.draw(), (self.display_image, self.draw_mask()), self.display_image_idx

    def on_slider_change(self, index: int) -> np.ndarray:
        """When the user changes the image index, update the display image."""
        self.display_image_idx = index
        self.display_image = np.array(Image.open(self.image_paths[index]))
        return self.draw(), (self.display_image, self.draw_mask())
    
    def on_image_select(self, evt: gr.SelectData) -> tuple[np.ndarray, List[tuple[Path, str]], int]:
        """When the user selects an image, update the display image."""
        self.display_image_idx = evt.index
        self.display_image = np.array(Image.open(self.image_paths[self.display_image_idx]))
        return self.draw(), (self.display_image, self.draw_mask()), self.display_image_idx
    
    def draw(self) -> np.ndarray:
        display_image = self.display_image.copy()
        """Draw the points in the image."""
        for ((x, y), point_type, frame_idx, _) in self.select_points:
            if frame_idx == self.display_image_idx:
                cv2.circle(display_image, (x, y), POINT_SIZE, COLORS[point_type], -1)
        return display_image
    
    def draw_mask(self) -> List[tuple[np.ndarray, str]]:
        """Draw the mask in the image."""
        self.predictor.reset_state(self.inference_state)
        if len(self.select_points) == 0:
            return []
        for ((x, y), point_type, frame_idx, _) in self.select_points:
            masks = self.add_points(frame_idx, (x, y), point_type)
        if frame_idx != self.display_image_idx:
            masks = self.get_mask(self.display_image_idx)
        return [(mask.astype(np.uint8), str(obj_id)) for obj_id, mask in masks.items()]

    def gui(self) -> gr.Interface:
        with gr.Blocks() as demo:
            # Segment image
            with gr.Row():
                with gr.Column():
                    # input image
                    input_images = gr.Gallery(type="filepath", label='Input images', value=self.image_paths)
                    display_image = gr.Image(type="numpy", label='Display image', value=self.display_image)
                    image_slider = gr.Slider(minimum=0, maximum=len(self.image_paths)-1, label='Image index')
                    # point prompt
                    undo_button = gr.Button('Undo')
                    with gr.Row():
                        fg_bg_radio = gr.Radio(
                            ['foreground_point', 'background_point'],
                            info="Select foreground or background point",
                            value='foreground_point',
                            label='Point label')
                    gr.Markdown('You can click on the image to select points prompt. '
                                'Default: `foreground_point`.')
                    run_button = gr.Button('Run')

                # show only mask
                with gr.Column():
                    output_mask = gr.AnnotatedImage()
        
            input_images.select(self.on_image_select, None, [display_image, output_mask, image_slider])
            display_image.select(self.on_select_point, fg_bg_radio, [display_image, output_mask])
            image_slider.change(self.on_slider_change, image_slider, [display_image, output_mask])
            undo_button.click(self.on_undo_point, None, [display_image, output_mask, image_slider])
            run_button.click(self.run, None, output_mask)

        return demo
    
    def main(self):
        iface = self.gui()
        iface.launch(server_port=self.config.server_port, server_name=self.config.server_name)