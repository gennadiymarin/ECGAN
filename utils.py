import torch
from torch import nn
import cv2 as cv
import numpy as np
import torchvision.transforms as T


def toRGB(x):
    return ((x + 1) / 2 * 255).to(torch.uint8)


def RGB2n(img_seg, labels):
    """
    img_seg: Bx3xHxW in 0..255
    labels: 19x3
    """
    x = img_seg.permute(0, 2, 3, 1)  # BxHxWx3
    res = (x.unsqueeze(-2).repeat(1, 1, 1, labels.size(0), 1) == labels)  # BxHxWx19x3
    return torch.all(res, dim=-1).permute(0, 3, 1, 2).float()


class Canny(nn.Module):
    def __init__(self, t_low=100, t_high=200):
        super().__init__()
        self.t_low = t_low
        self.t_high = t_high

    @torch.no_grad()
    def forward(self, x):
        """
        x: (B, C, H, W) in (-1, 1)
        """
        device = x.device
        B, C, H, W = x.shape
        img = x.permute((0, 2, 3, 1))
        img = toRGB(img).detach().cpu().numpy()
        img = img.reshape(B * H, W, C)
        edges = cv.Canny(img, self.t_low, self.t_high)
        edges = torch.tensor(edges.reshape(B, H, W)).unsqueeze(1)
        edges = edges.repeat(1, 3, 1, 1).to(device).float() / 255
        return T.Normalize(0.5, 0.5)(edges)


def logits2seg(logits, palette):
    """
    :param logits: Bx19xHxW
    :param palette: 19x3
    :return: Bx3xHxW
    """
    predicted_segmentation_maps = logits.argmax(dim=1).numpy()  # B x H x W

    color_segs = np.zeros((predicted_segmentation_maps.shape[0],
                           predicted_segmentation_maps.shape[1],
                           predicted_segmentation_maps.shape[2], 3), dtype=np.uint8)  # B x H x W x 3

    palette = np.array(palette)
    for label, color in enumerate(palette):
        color_segs[predicted_segmentation_maps == label, :] = color

    color_segs = torch.tensor(color_segs).permute(0, 3, 1, 2)  # B x 3 x H x W

    return color_segs
