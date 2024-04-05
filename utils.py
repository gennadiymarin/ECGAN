import torch
from torch import nn
import cv2 as cv
import numpy as np
import torchvision.transforms as T


class Canny(nn.Module):
    def __init__(self, t_low=100, t_high=200):
        super().__init__()
        self.t_low = t_low
        self.t_high = t_high

    def forward(self, x):
        img = x.detach().numpy()
        img = (img * 255).astype('uint8')
        edges = cv.Canny(img, self.t_low, self.t_high)
        edges = T.ToTensor()(edges)
        return edges.expand(3, *edges.shape[1:])

