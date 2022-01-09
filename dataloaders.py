import torch
from torch.utils.data import dataloader
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Grayscale, Compose, CenterCrop
from torchvision.utils import make_grid
from pathlib import Path
import numpy as np


def road_segment_dataset():
    data_path = 'data/road_segments/'
    transforms = Compose([Grayscale(), CenterCrop(128), ToTensor()])
    return torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transforms
    )


class NPZLoader(dataloader.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.files = list(Path(path).glob('*/*.npz'))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        numpy_array = np.load(str(self.files[item]))['arr_0']
        torch_array = torch.from_numpy(numpy_array)
        if self.transform is not None:
            torch_array = self.transform(torch_array)
        return torch_array, 0


train_dataset = NPZLoader('data/road_sdfs', transform=Compose([CenterCrop(128)]))

sdf_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    num_workers=0,
    shuffle=True)


for batch_idx, (data, target) in enumerate(sdf_loader):
    plt.imshow(make_grid(data.unsqueeze(1), normalize=True).permute(1, 2, 0), cmap='jet')
    plt.pause(1.0)


road_segment_loader = torch.utils.data.DataLoader(
    road_segment_dataset(),
    batch_size=64,
    num_workers=0,
    shuffle=True)


for batch_idx, (data, target) in enumerate(road_segment_loader):
    plt.imshow(make_grid(data, normalize=True).permute(1, 2, 0))
    plt.pause(1.0)