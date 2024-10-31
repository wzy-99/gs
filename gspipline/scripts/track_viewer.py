from dataclasses import dataclass, field

from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.base_dataparser import DataParser
from gspipline.viewer.track_viewer import TrackViewer, TrackViewerConfig

import tyro
import math


@dataclass
class ViewerLauncherConfig:
    """Configuration for the viewer launcher."""
    viewer: TrackViewerConfig = field(default_factory=TrackViewerConfig)
    """The viewer configuration to use for displaying data."""
    dataparser: AnnotatedDataParserUnion = field(default_factory=NerfstudioDataParserConfig)
    """The data parser to use for loading data."""


def main(config: ViewerLauncherConfig):
    dataparser: DataParser = config.dataparser.setup()
    outputs = dataparser.get_dataparser_outputs()
    if config.viewer.H == -1:
        config.viewer.H = outputs.cameras[0].height.cpu().numpy().tolist()[0]
    if config.viewer.W == -1:
        config.viewer.W = outputs.cameras[0].width.cpu().numpy().tolist()[0]
    if config.viewer.focal_degree == -1:
        fx = outputs.cameras[0].fx.cpu().numpy().tolist()[0]
        config.viewer.focal_degree = 2 * (180 / math.pi) * math.atan(outputs.cameras[0].width.cpu().numpy().tolist()[0] / (2 * fx))
    viewer: TrackViewer = config.viewer.setup()
    viewer.init_scene(outputs)
    viewer.run()


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            ViewerLauncherConfig,
        )
    )


if __name__ == "__main__":
    entrypoint()

