from dataclasses import dataclass, field


from gspipline.viewer.segment_viewer import SegmentViewerConfig, SegmentViewer

import tyro


@dataclass
class SegmentViewerLaunchConfig:
    """Configuration for the viewer launcher."""
    viewer: SegmentViewerConfig = field(default_factory=SegmentViewerConfig)
    """Configuration for the segmenter."""


def main(config: SegmentViewerLaunchConfig):
    """Main function for the viewer launcher."""
    viewer: SegmentViewer = config.viewer.setup()
    viewer.main()


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            SegmentViewerLaunchConfig,
        )
    )


if __name__ == "__main__":
    entrypoint()
