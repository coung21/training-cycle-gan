from torchvision.datasets.folder import default_loader
from torchvision import transforms
import torch
import os

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms_, mode='train'):
        # Compose the transforms into a single callable transform
        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted([os.path.join(root, mode, 'A', x) for x in os.listdir(os.path.join(root, mode, 'A')) if os.path.isfile(os.path.join(root, mode, 'A', x))]) # Build full file paths for domain A
        self.files_B = sorted([os.path.join(root, mode, 'B', x) for x in os.listdir(os.path.join(root, mode, 'B')) if os.path.isfile(os.path.join(root, mode, 'B', x))]) # Build full file paths for domain B
        self.len_dataset = min(len(self.files_A), len(self.files_B))

    def __getitem__(self, index):
        img_A = default_loader(self.files_A[index % self.len_dataset])
        img_B = default_loader(self.files_B[index % self.len_dataset])
        return {'A': self.transform(img_A), 'B': self.transform(img_B)}

    def __len__(self):
        return self.len_dataset