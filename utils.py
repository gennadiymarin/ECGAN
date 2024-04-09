import torch
from torch import nn
import cv2 as cv
import numpy as np
import torchvision.transforms as T


def toRGB(x):
    return ((x + 1) / 2 * 255).to(torch.uint8)

def RGB2n(x, colors):  #x: batch x 3 x h x w, colors: 35 x 3
    x1 = x.repeat(35, 1, 1, 1, 1).permute(1, 3, 4, 0, 2).int()  # B x h x w x 35 x 3
    return torch.all(x1 == colors, dim=-1).permute(0, 3, 1, 2)  # B x 35 x H x W


class Canny(nn.Module):
    def __init__(self, t_low=100, t_high=200):
        super().__init__()
        self.t_low = t_low
        self.t_high = t_high

    def forward(self, x):
        """
        x: (B, C, H, W) in (-1, 1)
        """
        B, C, H, W = x.shape
        img = x.permute((0, 2, 3, 1))
        img = toRGB(img).detach().numpy()
        img = img.reshape(B * H, W, C)
        edges = cv.Canny(img, self.t_low, self.t_high)
        edges = torch.tensor(edges.reshape(B, H, W)).unsqueeze(1)
        edges = edges.repeat(1, 3, 1, 1)
        return edges
