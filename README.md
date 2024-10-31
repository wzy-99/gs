# Track Viewer

```bash
python gspipline/scripts/track_viewer.py nerfstudio-data --data data/spann3r/output/bicycle --load_3D_points --center-method none --no-auto-scale-poses --orientation-method none
```

# 3D Point Cloud Renderer

```bash
python gspipline/scripts/point_cloud_render.py --renderer.camera_path_filename camera_paths/cameras.json nerfstudio-data --data data/spann3r/output/ship --load_3D_points --no-auto-scale-poses --orientation-method none --center-method none
```