import torch

from torchvision.utils import make_grid
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from models import Generator
from datasets import ImageDataset

opt = {
    'epoch': 0,
    'n_epochs': 200,
    'batchSize': 1,
    'dataroot': './datasets/cf-leaf/',
    'lr': 0.0002,
    'decay_epoch': 100,
    'size': 224,
    'input_nc': 3,
    'output_nc': 3,
    'cuda': torch.cuda.is_available(),
    'n_cpu': 8,
}

device = torch.device("cuda" if opt['cuda'] else "cpu")

# Initialize generator
netG_A2B = Generator(opt['input_nc'], opt['output_nc'])
netG_B2A = Generator(opt['output_nc'], opt['input_nc'])

if opt['cuda']:
    netG_A2B.to(torch.device("cuda"))
    netG_B2A.to(torch.device("cuda"))
    


# Load trained models
netG_A2B.load_state_dict(torch.load('output/netG_A2B_epoch_199.pth'))
netG_B2A.load_state_dict(torch.load('output/netG_B2A_epoch_199.pth'))

netG_A2B.eval()
netG_B2A.eval()

# Dataset for testing
transforms_ = [
    transforms.Resize(opt['size'], Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
dataloader = DataLoader(
    ImageDataset(opt['dataroot'], transforms_=transforms_, mode='test'),
    batch_size=opt['batchSize'],
    shuffle=False,
    num_workers=opt['n_cpu'],
)

# Visualize test results
def visualize_results(real_A, real_B, fake_A, fake_B):
    images = torch.cat((real_A.cpu().data, fake_B.cpu().data, real_B.cpu().data, fake_A.cpu().data), 0)
    grid = make_grid(images, nrow=4, normalize=True, scale_each=True)
    plt.figure(figsize=(15, 5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

# Testing loop
for i, batch in enumerate(dataloader):
    # real_A = Variable(batch['A'].type(Tensor))
    # real_B = Variable(batch['B'].type(Tensor))

    real_A = batch['A'].to(device, dtype=torch.float32)
    real_B = batch['B'].to(device, dtype=torch.float32)
    fake_B = netG_A2B(real_A)
    fake_A = netG_B2A(real_B)

    # Visualize results
    print(f"Visualizing sample {i+1}/{len(dataloader)}")
    visualize_results(real_A, real_B, fake_A, fake_B)
