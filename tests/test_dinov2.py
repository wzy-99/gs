import torch

# DINOv2
dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

x = torch.randn(1, 3, 224, 224)

dino = dino.cuda()
x = x.cuda()

with torch.no_grad():
    out = dino(x)

print(out.shape)