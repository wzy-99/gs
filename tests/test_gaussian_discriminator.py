import hydra
from PIL import Image
import numpy as np
import torch

from gspipline.datset.nuscene_novel_view import nuSceneNovelViewDataset, nuSceneNovelViewConfig
from gspipline.pipeline.gaussian_discriminator import GaussianDiscriminatorPipeline, GaussianDiscriminatorPipelineConfig

from noposplat.model.encoder import EncoderCfg

from omegaconf import OmegaConf

from dacite import Config, from_dict

from lightning import seed_everything

seed_everything(42)

@hydra.main(config_path="../configs/noposplat/encoder", config_name="noposplat")
def main(config):
    config = from_dict(data_class=EncoderCfg, data=OmegaConf.to_container(config))
    
    config = GaussianDiscriminatorPipelineConfig(
        encoder_pretrained_ckpt='/mnt/data/wangziyi/codes/gspipline/data/noposplat/re10k.ckpt',
        gaussian_encoder=config
    )

    gs_pipeline = GaussianDiscriminatorPipeline(config)

    gs_pipeline = gs_pipeline.cuda()

    config = nuSceneNovelViewConfig(
        root_path='/mnt/data/wangziyi/data/nuscenes',
    )

    dataset = nuSceneNovelViewDataset(config)

    for i in range(1):
        data = dataset[i]

        images = data['image'].cuda().float().unsqueeze(0)
        # save
        for i in range(images.shape[0]):
            for j in range(images.shape[1]):
                image = images[i, j].detach().permute(1, 2, 0).cpu().numpy()
                image = (image + 1.0) * 127.5
                image = image.astype(np.uint8)
                Image.fromarray(image).save(f'/mnt/data/wangziyi/codes/gspipline/tests/gt_image{i+1}_view{j+1}.png')
        novel_images = data['novel_image'].cuda().float().unsqueeze(0).unsqueeze(0)
        # save
        for i in range(novel_images.shape[0]):
            for j in range(novel_images.shape[1]):
                image = novel_images[i, j].detach().permute(1, 2, 0).cpu().numpy()
                image = (image + 1.0) * 127.5
                image = image.astype(np.uint8)
                Image.fromarray(image).save(f'/mnt/data/wangziyi/codes/gspipline/tests/novel_image{i+1}_view{j+1}.png')
        intrinsics = data['intrinsics'].cuda().float().unsqueeze(0)

        with torch.no_grad():
            gaussians = gs_pipeline._predict_gaussian(
                batch={
                    'image': images,
                    'intrinsics': intrinsics
                }
            )

        batch = {}
        # batch['camtoworlds'] = torch.eye(4).unsqueeze(0).unsqueeze(0).cuda().float() # (1, 1, 4, 4)
        batch['camtoworlds'] = data['novel_extrinsics'].unsqueeze(0).unsqueeze(0).cuda().float() # (1, 1, 4, 4)
        # batch['camtoworlds'] = data['extrinsics'].unsqueeze(0).cuda().float() # (1, 1, 4, 4)
        a, b = data['extrinsics'][0, :3, 3], batch['camtoworlds'][-1, :3, 3]
        scale = (a - b).norm()
        batch['camtoworlds'][:, :, :3, 3] /= scale
        # batch['Ks'] = torch.tensor([[256, 0, 128], [0, 256, 128], [0, 0, 1]]).unsqueeze(0).unsqueeze(0).cuda().float() # (1, 1, 3, 3)
        batch['Ks'] = (data['novel_intrinsics'] * 256).unsqueeze(0).unsqueeze(0).cuda().float()
        # batch['Ks'] = (data['intrinsics'] * 256).unsqueeze(0).cuda().float()
        batch['width'] = 256
        batch['height'] = 256
        with torch.no_grad():
            renderings = gs_pipeline._render_gaussian(gaussians, batch)

        images = renderings['images']
        for i in range(images.shape[0]):
            for j in range(images.shape[1]):
                image = images[i, j].detach().cpu().numpy()
                image = (image * 255.0).astype(np.uint8)
                Image.fromarray(image).save(f'/mnt/data/wangziyi/codes/gspipline/tests/image{i+1}_gaussian{j+1}.png')

if __name__ == "__main__":
    main()