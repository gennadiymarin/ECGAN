import numpy as np
import torch


def pixel_contrastive_loss(color_batch, f, labels):
    '''

    :param color_batch: (B, 3, H, W)
    :param f: (B, C, H, W)
    :param labels:
    :return:
    '''
    total_loss = 0

    color_batch = torch.permute(
        color_batch, (0, 2, 3, 1)).flatten(end_dim=-2)  # (-1 , 3)
    f = torch.permute(f, (0, 2, 3, 1)).flatten(end_dim=-2)  # (-1 , C)

    for label in labels:
        mask = torch.all(color_batch == torch.tensor(label), axis=-1)
        if mask.sum() == 0 :
            continue

        pos = np.random.choice(len(f[mask]), min(
            10000, len(f[mask])), replace=False)
        neg = np.random.choice(len(f[~mask]), min(
            10000, len(f[~mask])), replace=False)

        pos_batch = f[mask][pos] / \
            (f[mask][pos] ** 2).sum(axis=1).sqrt().reshape(-1, 1)
        neg_batch = f[~mask][neg] / \
            (f[~mask][neg] ** 2).sum(axis=1).sqrt().reshape(-1, 1)

        loss = torch.exp(pos_batch @ pos_batch.T)
        sum_neg = torch.exp(pos_batch @ neg_batch.T).sum(axis=1)
        loss = -torch.log(loss / (loss + sum_neg)).sum() / \
            mask.sum() / pos_batch.size(0)
        total_loss += loss

    return total_loss


def reconstructive_loss(img_true, img_pred):
    '''
    :param img_true: (B, 3, H, W)
    :param img_pred: (B, 3, H, W)
    :return:
    '''

    return abs(img_true - img_pred).sum()
