from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import Cityscapes
from torchvision.transforms import InterpolationMode


# class CityScapesDataSet(Dataset):
#     def __init__(self, src_data: str, train=True):
#         self.path = src_data
#         self.path += '/train/' if train else '/val/'
#         self.length = 2975 if train else 500
#         self.transform = transforms.Compose([transforms.ToTensor(),
#                                              transforms.Normalize(0.5, 0.5)])
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, item: int):
#         img = self.transform(Image.open(self.path + str(item + 1) +'.jpg'))
#         return img[:,:,:256], img[:,:,256:]

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

# train_set = CityScapesDataSet('/kaggle/input/cityscapes-image-pairs/cityscapes_data')
# val_set = CityScapesDataSet('/kaggle/input/cityscapes-image-pairs/cityscapes_data', train=False)
#
# train_loader = DataLoader(train_set, batch_size=CFG.batch_size, shuffle=True)
# val_loader = DataLoader(val_set, batch_size=CFG.batch_size, shuffle=False)
