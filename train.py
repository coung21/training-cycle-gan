# Training settings
import torch
from models import Generator, Discriminator
from datasets import ImageDataset
from utils import ReplayBuffer, LambdaLR, weights_init_normal
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
import itertools


opt = {
    'epoch': 0,
    'n_epochs': 100,
    'batchSize': 1,
    'dataroot': './datasets/cf-leaf/',
    'lr': 0.0002,
    'decay_epoch': 50,
    'size': 224,
    'input_nc': 3,
    'output_nc': 3,
    'cuda': torch.cuda.is_available(),
    'n_cpu': 1,
}

# Initialize networks
netG_A2B = Generator(opt['input_nc'], opt['output_nc'])
netG_B2A = Generator(opt['output_nc'], opt['input_nc'])
netD_A = Discriminator(opt['input_nc'])
netD_B = Discriminator(opt['output_nc'])

if opt['cuda']:
    print('Using CUDA')
    netG_A2B.to(torch.device("cuda"))
    netG_B2A.to(torch.device("cuda"))
    netD_A.to(torch.device("cuda"))
    netD_B.to(torch.device("cuda"))

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt['lr'], betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt['lr'], betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt['lr'], betas=(0.5, 0.999))

# Learning rate schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt['n_epochs'], opt['epoch'], opt['decay_epoch']).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt['n_epochs'], opt['epoch'], opt['decay_epoch']).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt['n_epochs'], opt['epoch'], opt['decay_epoch']).step)

# Inputs & targets
# Initialize tensors
device = torch.device('cuda' if opt['cuda'] else 'cpu')

input_A = torch.zeros(opt['batchSize'], opt['input_nc'], opt['size'], opt['size'], device=device, dtype=torch.float32)
input_B = torch.zeros(opt['batchSize'], opt['output_nc'], opt['size'], opt['size'], device=device, dtype=torch.float32)

# Correct the size of target tensors
target_real = torch.ones((opt['batchSize'], 1), device=device, dtype=torch.float32)
target_fake = torch.zeros((opt['batchSize'], 1), device=device, dtype=torch.float32)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [
    transforms.Resize(int(opt['size'] * 1.12), Image.BICUBIC),
    transforms.RandomCrop(opt['size']),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
dataloader = DataLoader(
    ImageDataset(opt['dataroot'], transforms_=transforms_, mode="train"),
    batch_size=opt['batchSize'],
    shuffle=True,
    num_workers=opt['n_cpu'],
)

# Training loop
for epoch in range(opt['epoch'], opt['n_epochs']):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        optimizer_G.step()

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()
        optimizer_D_A.step()

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()
        optimizer_D_B.step()
        print(f'Training epoch {epoch}')

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    print(f'Done epoch {epoch}')
    # Save models
    torch.save(netG_A2B.state_dict(), f'./output/netG_A2B_epoch_{epoch}.pth')
    torch.save(netG_B2A.state_dict(), f'./output/netG_B2A_epoch_{epoch}.pth')
    torch.save(netD_A.state_dict(), f'./output/netD_A_epoch_{epoch}.pth')
    torch.save(netD_B.state_dict(), f'./output/netD_B_epoch_{epoch}.pth')
