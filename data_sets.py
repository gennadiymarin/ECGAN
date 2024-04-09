from torch.utils.data import Dataset
from PIL import Image
import torch.transforms as transforms


class CityScapesDataSet(Dataset):
    def __init__(self, src_data: str, train=True):
        self.path = src_data
        self.path += '/train/' if train else '/val/'
        self.length = 2975 if train else 500
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(0.5, 0.5)])

    def __len__(self):
        return self.length

    def __getitem__(self, item: int):
        img = self.transform(Image.open(self.path + str(item + 1) +'.jpg'))
        return img[:,:,:256], img[:,:,256:]



# train_set = CityScapesDataSet('/kaggle/input/cityscapes-image-pairs/cityscapes_data')
# val_set = CityScapesDataSet('/kaggle/input/cityscapes-image-pairs/cityscapes_data', train=False)
#
# train_loader = DataLoader(train_set, batch_size=CFG.batch_size, shuffle=True)
# val_loader = DataLoader(val_set, batch_size=CFG.batch_size, shuffle=False)
