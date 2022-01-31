import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import dataloader, random_split
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Grayscale, Compose, CenterCrop, Resize
from torchvision.utils import make_grid
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
from typing import Optional
from math import ceil
from torch.nn.functional import one_hot
import cv2
from scipy.ndimage import distance_transform_edt


class ToSegment:
    def __call__(self, segment):
        segment = torch.from_numpy(np.asarray(segment).astype(np.int64))
        image = one_hot(segment).permute(2, 0, 1)
        return image


def gradient(segment):
    gradient_x = cv2.Sobel(segment, 3, dx=1, dy=0)
    gradient_y = cv2.Sobel(segment, 3, dx=0, dy=1)
    return cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0.0)


class ToSDF:
    def __call__(self, segment):
        segment = np.asarray(segment)

        # Signed Distance Field
        gradient_segment = gradient(segment.astype(float))
        sdf = distance_transform_edt(gradient_segment == 0)
        sign = segment.astype(float) * 2 - 1
        sdf = sdf * sign
        return sdf


class TruncateAndNormalize:
    def __init__(self, min, max):
        self.min, self.max = min, max

    def __call__(self, sdf):
        return (np.clip(sdf, self.min, self.max) - self.min) / (self.max - self.min)


def road_segment_dataset(data_path):
    transforms = Compose([Grayscale(), CenterCrop(128), Resize(32), ToSegment()])
    return torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transforms
    )


def road_sdf_dataset(data_path):
    transforms = Compose([Grayscale(), CenterCrop(128), Resize(32),
                          ToSDF(), ToTensor()])
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


class RoadSegmentDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train, self.val = None, None

    def setup(self, stage: Optional[str] = None):
        full = road_segment_dataset(self.data_dir)
        train_size, val_size = len(full) * 9 // 10, ceil(len(full) / 10)
        self.train, self.val = random_split(full, [train_size, val_size])

    def train_dataloader(self):
        return dataloader.DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return dataloader.DataLoader(self.val, batch_size=self.batch_size)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError

    def test_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError


class RoadSDFDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir",
                 batch_size: int = 64,
                 num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train, self.val = None, None

    def setup(self, stage: Optional[str] = None):
        full = road_sdf_dataset(data_path=self.data_dir)
        train_size, val_size = len(full) * 9 // 10, ceil(len(full) / 10)
        self.train, self.val = random_split(full, [train_size, val_size])

    def train_dataloader(self):
        return dataloader.DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return dataloader.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError

    def test_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError


if __name__ == '__main__':

    # train_dataset = NPZLoader('data/road_sdfs', transform=Compose([CenterCrop(128)]))

    # sdf_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=64,
    #     num_workers=0,
    #     shuffle=True)
    #
    # for batch_idx, (data, target) in enumerate(sdf_loader):
    #     plt.imshow(make_grid(data.unsqueeze(1), normalize=True).permute(1, 2, 0), cmap='jet')
    #     plt.pause(1.0)

    # road_segment_loader = torch.utils.data.DataLoader(
    #     road_segment_dataset(data_path='data/road_segments/'),
    #     batch_size=64,
    #     num_workers=0,
    #     shuffle=True)
    #
    # for batch_idx, (data, target) in enumerate(road_segment_loader):
    #     plt.imshow(make_grid(data, normalize=True).permute(1, 2, 0))
    #     plt.pause(1.0)

    road_sdf_loader = torch.utils.data.DataLoader(
        road_sdf_dataset(data_path='data/road_segments/'),
        batch_size=64,
        num_workers=0,
        shuffle=True)

    for batch_idx, (data, target) in enumerate(road_sdf_loader):
        plt.imshow(data[0].squeeze())
        plt.pause(5.0)
        plt.imshow(make_grid(data, normalize=True).permute(1, 2, 0))
        plt.pause(5.0)