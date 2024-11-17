import hydra
from PIL import Image
import numpy as np
import torch

from gspipline.pipeline.gaussian_discriminator import GaussianDiscriminatorPipeline, GaussianDiscriminatorPipelineConfig

from noposplat.model.encoder import EncoderCfg

from omegaconf import OmegaConf

from dacite import Config, from_dict


@hydra.main(config_path="../configs/noposplat/encoder", config_name="noposplat")
def main(config):
    config = from_dict(data_class=EncoderCfg, data=OmegaConf.to_container(config))
    
    config = GaussianDiscriminatorPipelineConfig(
        encoder_pretrained_ckpt='/home/wzy/data/noposplat/re10k.ckpt',
        gaussian_encoder=config
    )

    gs_pipeline = GaussianDiscriminatorPipeline(config)

    gs_pipeline = gs_pipeline.cuda()

    # print(gs_pipeline)

    img1 = Image.open('/home/wzy/code/gspipline/data/gspipline/test_images/image1.png')
    img1 = img1.resize((256, 256))
    img1 = np.array(img1)
    img2 = Image.open('/home/wzy/code/gspipline/data/gspipline/test_images/image2.png')
    img2 = img2.resize((256, 256))
    img2 = np.array(img2)

    images = np.stack([img1, img2], axis=0) # (2, 3, H, W)
    images = images.transpose(0, 3, 1, 2) # (2, W, H, 3)
    images = images.astype(np.float32) / 255.0
    images = images * 2.0 - 1.0 # (2, W, H, 3)
    images = torch.from_numpy(images).cuda()
    images = images.reshape(-1, 2, 3, images.shape[2], images.shape[3]) # (1, 2, 3, H, W)

    intrinsics = torch.tensor([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]]).cuda().float()
    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0) # (1, 1, 3, 3)
    intrinsics = intrinsics.repeat(images.shape[0], images.shape[1], 1, 1) # (1, 2, 3, 3)

    with torch.no_grad():
        gaussians = gs_pipeline._predict_gaussian(
            batch={
                'image': images,
                'intrinsics': intrinsics
            }
        )

    batch = {}
    batch['camtoworlds'] = torch.eye(4).unsqueeze(0).unsqueeze(0).cuda().float() # (1, 1, 4, 4)
    batch['Ks'] = torch.tensor([[512, 0, 256], [0, 512, 256], [0, 0, 1]]).unsqueeze(0).unsqueeze(0).cuda().float() # (1, 1, 3, 3)
    batch['width'] = 512
    batch['height'] = 512
    with torch.no_grad():
        renderings = gs_pipeline._render_gaussian(gaussians, batch)

    images = renderings['images']
    for i in range(images.shape[0]):
        for j in range(images.shape[1]):
            image = images[i, j].detach().cpu().numpy()
            image = (image * 255.0).astype(np.uint8)
            Image.fromarray(image).save(f'/home/wzy/code/gspipline/tests/image{i+1}_gaussian{j+1}.png')

if __name__ == "__main__":
    main()