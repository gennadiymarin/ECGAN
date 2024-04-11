from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import Cityscapes
from torchvision.transforms import InterpolationMode
from utils import get_palette
import torch
import numpy as np


def get_dataset(name, src_data, train=True, img_size=(128, 256)):
    if name == 'cityscapes':
        return CityScapesDataSet(src_data, train, img_size)
    if name == 'ade20k':
        return ADE20kDataset(src_data, train, img_size)
    raise ValueError


class CityScapesDataSet(Dataset):
    def __init__(self, src_data: str, train=True, img_size=(128, 256)):
        split = 'train' if train else 'val'
        self.dataset = Cityscapes(src_data, split=split, mode='fine',
                                  target_type='color',
                                  transform=transforms.Compose([transforms.ToTensor(),
                                                                transforms.Resize(img_size[0],
                                                                                  interpolation=InterpolationMode.NEAREST),
                                                                transforms.CenterCrop(img_size),
                                                                transforms.Normalize(0.5, 0.5)]),
                                  target_transform=transforms.Compose([transforms.PILToTensor(),
                                                                       transforms.Resize(img_size[0],
                                                                                         interpolation=InterpolationMode.NEAREST),
                                                                       transforms.CenterCrop(img_size),
                                                                       ])
                                  )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item: int):
        img, seg = self.dataset[item]
        return img, seg[:3]


class ADE20kDataset(Dataset):
    def __init__(self, src_data: str, train=True, img_size=(256, 256)):
        self.split = 'train' if train else 'val'
        self.img_dir = f'{src_data}/ADEChallengeData2016/images/' + ('training' if train else 'validation')
        self.ann_dir = f'{src_data}/ADEChallengeData2016/annotations/' + ('training' if train else 'validation')
        self.length = 20210 if train else 2000
        self.img_size = img_size
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize(img_size),
                                             transforms.Normalize(0.5, 0.5)])
        self.seg_transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize(img_size, interpolation=InterpolationMode.NEAREST),
        ])
        self.palette = get_palette('ade20k')

    def __len__(self):
        return self.length

    def __getitem__(self, item: int):
        img_name = 'ADE_{}_{:08d}'.format(self.split, item+1)
        img = self.transform(Image.open(f'{self.img_dir}/{img_name}.jpg'))

        seg = self.seg_transform(Image.open(f'{self.ann_dir}/{img_name}.png'))
        seg = seg.repeat(3, 1, 1)
        return img, seg
