from tqdm import tqdm
import torch
import torchvision.transforms.functional as F
from loss import pixel_contrastive_loss


def train_epoch(model, optimizers, train_loader, device):
    loss_log = []
    acc_log = []
    model.train()

    for img, img_seg in tqdm(train_loader, desc='Training'):
        s = RGB2n(img_seg, labels)  # TODO

        img = img.to(device)
        img_seg = img_seg.to(device)

        # train D
        optimizers['D'].zero_grad()

        img_edge = model.canny(img)
        f, out_edge, out_img1, out_img2, pred_labels = model(s, img)

        edge_real = model.discriminator(img_edge, s)
        edge_fake = model.discriminator(out_edge, s)
        img_real = model.discriminator(img, s)
        img_fake1 = model.discriminator(out_img1, s)
        img_fake2 = model.discriminator(out_img2, s)

        edge_real_loss = F.binary_cross_entropy_with_logits(edge_real, torch.ones_like(edge_real))
        edge_fake_loss = F.binary_cross_entropy_with_logits(out_edge.detach(), torch.zeros_like(edge_fake))

        img_real_loss = F.binary_cross_entropy_with_logits(img, torch.ones_like(img_real))
        img_fake_loss1 = F.binary_cross_entropy_with_logits(out_img1.detach(), torch.zeros_like(img_fake1))
        img_fake_loss2 = F.binary_cross_entropy_with_logits(out_img2.detach(), torch.zeros_like(img_fake2))

        loss = edge_real_loss + edge_fake_loss + img_real_loss + img_fake_loss1 + img_fake_loss2 + ...
        loss.backward()

        optimizers['D'].step()

        # train G
        optimizers['G'].zero_grad()
        f, out_edge, out_img1, out_img2, labels = model(s, img)

        edge_fake = model.discriminator(out_edge, s)

        img_fake1 = model.discriminator(out_img1, s)
        img_fake2 = model.discriminator(out_img2, s)

        edge_fake_loss = F.binary_cross_entropy_with_logits(out_fake, torch.ones_like(edge_fake))
        img_fake_loss1 = F.binary_cross_entropy_with_logits(out_img1, torch.ones_like(img_fake1))
        img_fake_loss2 = F.binary_cross_entropy_with_logits(out_img2, torch.ones_like(img_fake2))

        mma_loss = edge_real_loss + edge_fake_loss + img_real_loss + img_fake_loss1 + img_fake_loss2
        sim_loss = similarity_loss()  # TODO
        contrastive_loss = pixel_contrastive_loss(img_seg, f, labels)

        loss = mma_loss + sim_loss + contrastive_loss + ...

        loss.backward()

        optimizers['G'].step()

    return ...  # TODO


def train(model, optimizers, n_epochs, train_loader, device='cuda'):
    for epoch in range(epoch_bias + 1, n_epochs + epoch_bias + 1):
        ... = train_epoch(model, optimizers, train_loader, device) #TODO
    return ...
