import numpy as np
import torch
# import torchvision.transforms.functional as F
import torch.nn.functional as F
import torchvision


def pixel_contrastive_loss(img_seg, f, labels):
    '''
    :param img_seg: (B, 3, H, W)
    :param f: (B, C, H, W)
    :param labels:
    :return:
    '''

    total_loss = 0

    img_seg = torch.permute(
        img_seg, (0, 2, 3, 1)).flatten(end_dim=-2)  # (-1 , 3)
    f = torch.permute(f, (0, 2, 3, 1)).flatten(end_dim=-2)  # (-1 , C)

    for label in labels:
        mask = torch.all(img_seg == label, dim=-1)
        if mask.sum() == 0:
            continue

        pos = np.random.choice(len(f[mask]), min(
            1000, len(f[mask])), replace=False)
        neg = np.random.choice(len(f[~mask]), min(
            1000, len(f[~mask])), replace=False)

        pos_batch = f[mask][pos] / \
                    (f[mask][pos] ** 2).sum(dim=1).sqrt().reshape(-1, 1)
        neg_batch = f[~mask][neg] / \
                    (f[~mask][neg] ** 2).sum(dim=1).sqrt().reshape(-1, 1)

        loss = torch.exp(pos_batch @ pos_batch.T)
        sum_neg = torch.exp(pos_batch @ neg_batch.T).sum(dim=1)
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

    return F.l1_loss(img_true, img_pred, reduction='sum')


def generator_mma_loss(edge_fake_logits, img_fake1_logits, img_fake2_logits, lmbd=2):
    edge_fake_loss = F.binary_cross_entropy_with_logits(edge_fake_logits, torch.ones_like(edge_fake_logits))
    img_fake_loss1 = F.binary_cross_entropy_with_logits(img_fake1_logits, torch.ones_like(img_fake1_logits))
    img_fake_loss2 = F.binary_cross_entropy_with_logits(img_fake2_logits, torch.ones_like(img_fake2_logits))
    return lmbd * img_fake_loss2 + img_fake_loss1 + edge_fake_loss


def discr_mma_loss(edge_real_logits, img_real_logits, edge_fake_logits, img_fake1_logits, img_fake2_logits, lmbd=2):
    edge_real_loss = F.binary_cross_entropy_with_logits(edge_real_logits, torch.ones_like(edge_real_logits))
    edge_fake_loss = F.binary_cross_entropy_with_logits(edge_fake_logits.detach(), torch.zeros_like(edge_fake_logits))

    img_real_loss = F.binary_cross_entropy_with_logits(img_real_logits, torch.ones_like(img_real_logits))
    img_fake_loss1 = F.binary_cross_entropy_with_logits(img_fake1_logits.detach(), torch.zeros_like(img_fake1_logits))
    img_fake_loss2 = F.binary_cross_entropy_with_logits(img_fake2_logits.detach(), torch.zeros_like(img_fake2_logits))

    return (lmbd + 1) * img_real_loss + lmbd * img_fake_loss2 + img_fake_loss1 + \
        edge_real_loss + edge_fake_loss


# https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, input, target):
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            # print(x.shape, y.shape)
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss


def similarity_loss(x, y):  # B x N x H x W
    x1 = x.reshape(x.shape[0], x.shape[1], -1)  # B x N x M     M = HW
    x_s = x1 @ x1.permute(0, 2, 1)  # B x M x M

    y1 = y.reshape(y.shape[0], y.shape[1], -1)
    y_s = y1 @ y1.permute(0, 2, 1)

    return F.binary_cross_entropy_with_logits(x_s, y_s)


def disc_feature_loss(x, y):
    return F.l1_loss(x, y)


class GANLossFactory:
    def __init__(self, config):
        self.perc = VGGPerceptualLoss().to(config.device)
        self._coefs = config.loss_coefs
        self._losses = {
            'mma_G': generator_mma_loss,
            'mma_D': discr_mma_loss,
            'pix_contr': pixel_contrastive_loss,
            'L1': reconstructive_loss,
            'sim': similarity_loss,
            'perc': self.perc,
            'discr_f': disc_feature_loss,
        }

    def __getitem__(self, key):
        return lambda *args, **kwargs: self._coefs[key] * self._losses[key](*args, **kwargs)
