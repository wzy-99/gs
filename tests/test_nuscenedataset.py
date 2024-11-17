from gspipline.dataset.nuscene_novel_view import nuSceneNovelViewDataset, nuSceneNovelViewConfig

config = nuSceneNovelViewConfig(
    root_path='/mnt/data/wangziyi/data/nuscenes',
)

dataset = nuSceneNovelViewDataset(config)

for i in range(10):
    data = dataset[i]
    print(data)