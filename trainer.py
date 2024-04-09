from tqdm import tqdm
import torch
import torchvision.transforms.functional as F
from loss import GANLossFactory
from networks import ECGAN
from data_sets import CityScapesDataSet
from torch.utils.data import DataLoader
from config import TrainingConfig


class Trainer:
    def __init__(self, config):
        self.config = config

        self.device = self.config.device
        self.model = ECGAN(self.config).to(self.device)
        self.optimizers = {
            'G': torch.optim.Adam(self.model.generator.parameters(), lr=self.config.lr,
                                  betas=(self.config.beta1., self.config.beta2)),
            'D': torch.optim.Adam(self.model.discriminator.parameters(), lr=self.config.lr,
                                  betas=(self.config.beta1, self.config.beta2)),
        }

        dataset = CityScapesDataSet(self.config.data_path, train=True)
        self.loader = DataLoader(dataset, batch_size=self.config.batch_size,
                                 shuffle=True, num_workers=0)

        self.losses = GANLossFactory(self.config)
        self.labels = get_dataset_labels(self.config.dataset)  # TODO
        self.epoch = 0

    def train_epoch(self):
        for img, img_seg in tqdm(self.loader, desc='Training'):
            img = img.to(self.device)
            img_seg = img_seg.to(self.device)

            s = RGB2n(img_seg, self.labels)  # TODO заимпортить это

            loss_D = self.update_D(img, s)
            loss_G = self.update_G(img, img_seg, s)

    def update_D(self, img, s):
        self.optimizers['D'].zero_grad()

        img_edge = self.model.canny(img)
        f, out_edge, out_img1, out_img2, pred_labels = self.model(s, img)

        edge_real_logits = self.model.discriminator(img_edge, s)
        edge_fake_logits = self.model.discriminator(out_edge, s)
        img_real_logits = self.model.discriminator(img, s)
        img_fake1_logits = self.model.discriminator(out_img1, s)
        img_fake2_logits = self.model.discriminator(out_img2, s)

        loss = self.losses['mma_D'](edge_real_logits, img_real_logits, edge_fake_logits, img_fake1_logits,
                                    img_fake2_logits)

        loss.backward()

        self.optimizers['D'].step()

    def update_G(self, img, img_seg, s):
        self.optimizers['G'].zero_grad()
        f, out_edge, out_img1, out_img2, pred_labels = self.model(s, img)

        img_edge = self.model.canny(img)

        edge_fake_logits = self.model.discriminator(out_edge, s)
        img_fake1_logits = self.model.discriminator(out_img1, s)
        img_fake2_logits = self.model.discriminator(out_img2, s)

        loss = self.losses['mma_G'](edge_fake_logits, img_fake1_logits, img_fake2_logits) \
               + self.losses['pix_contr'](img_seg, f, self.labels) \
               + self.losses['L1'](img, out_img1) \
               + self.losses['sim'](img, out_img2) \
               + self.losses['perc'](img_edge, out_edge) \
               + self.losses['perc'](img, out_img1) \
               + self.losses['perc'](img, out_img2) \
               + self.losses['discr_f'](img_edge, out_edge) \
               + self.losses['discr_f'](img, out_img1) \
               + self.losses['discr_f'](img, out_img2)

        loss.backward()

        self.optimizers['G'].step()

    def train(self):
        self.model.train()
        for self.epoch in range(1, self.config.n_epochs + 1):
            self.train_epoch()
