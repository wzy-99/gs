import hydra
from PIL import Image
import numpy as np
import torch

from gspipline.dataset.nuscene_novel_view import nuSceneNovelViewDataset, nuSceneNovelViewConfig
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

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    for data in dataloader:

        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.cuda().float()
        
        # images = data['image']
        # for i in range(images.shape[0]):
        #     for j in range(images.shape[1]):
        #         image = images[i, j].detach().permute(1, 2, 0).cpu().numpy()
        #         image = image * 255.0
        #         image = image.astype(np.uint8)
        #         Image.fromarray(image).save(f'/mnt/data/wangziyi/codes/gspipline/tests/gt_image{i+1}_view{j+1}.png')
        # novel_images = data['novel_image']
        # for i in range(novel_images.shape[0]):
        #     for j in range(novel_images.shape[1]):
        #         image = novel_images[i, j].detach().permute(1, 2, 0).cpu().numpy()
        #         image = image * 255.0
        #         image = image.astype(np.uint8)
        #         Image.fromarray(image).save(f'/mnt/data/wangziyi/codes/gspipline/tests/novel_image{i+1}_view{j+1}.png')

        ret = gs_pipeline.training_step(data, 0)

        images = ret['images']
        for i in range(images.shape[0]):
            image = images[i].detach().permute(1, 2, 0).cpu().numpy()
            image = image * 255.0
            image = image.astype(np.uint8)
            Image.fromarray(image).save(f'/mnt/data/wangziyi/codes/gspipline/tests/image{i+1}.png')

        print(ret['loss'])

        # images = renderings['images']
        # for i in range(images.shape[0]):
        #     for j in range(images.shape[1]):
        #         image = images[i, j].detach().cpu().numpy()
        #         image = (image * 255.0).astype(np.uint8)
        #         Image.fromarray(image).save(f'/mnt/data/wangziyi/codes/gspipline/tests/image{i+1}_gaussian{j+1}.png')

if __name__ == "__main__":
    main()