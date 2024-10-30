from dataclasses import dataclass, field

from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.base_dataparser import DataParser
from gspipline.renderer.point_cloud_render import PointCloudRenderConfig, PointCloudRender

import tyro


@dataclass
class RenderLauncherConfig:
    """Configuration for the viewer launcher."""
    renderer: PointCloudRenderConfig = field(default_factory=PointCloudRenderConfig)
    dataparser: AnnotatedDataParserUnion | None = field(default=None)
    """The data parser to use for loading data."""


def main(config: RenderLauncherConfig):
    dataparser: DataParser = config.dataparser.setup()
    outputs = dataparser.get_dataparser_outputs()
    renderer: PointCloudRender = config.renderer.setup()
    assert 'points3D_xyz' in outputs.metadata and 'points3D_rgb' in outputs.metadata, "The dataparser must output 'points3D_xyz' and 'points3D_rgb' in the metadata."
    renderer.render(outputs.metadata['points3D_xyz'], outputs.metadata['points3D_rgb'])

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            RenderLauncherConfig,
        )
    )


if __name__ == "__main__":
    entrypoint()
