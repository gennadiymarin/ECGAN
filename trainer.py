from tqdm import tqdm
import torch
from loss import GANLossFactory
from networks import ECGAN
from data_sets import get_dataset
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from utils import RGB2n, toRGB, get_palette
import wandb


class Trainer:
    def __init__(self, config):
        self.config = config

        self.device = self.config.device
        self.model = ECGAN(self.config).to(self.device)
        self.optimizers = {
            'G': torch.optim.Adam(self.model.generator.parameters(), lr=self.config.lr,
                                  betas=(self.config.beta1, self.config.beta2)),
            'D': torch.optim.Adam(self.model.discriminator.parameters(), lr=self.config.lr,
                                  betas=(self.config.beta1, self.config.beta2)),
        }

        dataset = get_dataset(self.config.dataset, self.config.data_path, train=True,
                              img_size=(self.config.H, self.config.W))

        self.loader = DataLoader(dataset, batch_size=self.config.batch_size,
                                 shuffle=True, num_workers=0)

        self.losses = GANLossFactory(self.config)
        self.labels = torch.tensor(get_palette(config.dataset), device=self.device)
        self.epoch = 0

        val_data = get_dataset(self.config.dataset, self.config.data_path, train=False,
                               img_size=(self.config.H, self.config.W))
        self.log_batch = next(iter(DataLoader(val_data, batch_size=self.config.batch_size,
                                              shuffle=False, num_workers=0)))

        wandb.init(project=config.project_name)

    def train_epoch(self):
        self.model.train()
        total_loss_D, total_loss_G = 0, 0
        for iter, (img, img_seg) in enumerate(tqdm(self.loader, desc='Training')):
            img = img.to(self.device)
            img_seg = img_seg.to(self.device)
            s = RGB2n(img_seg, self.labels)

            loss_D = self.update_D(img, s)
            loss_G = self.update_G(img, img_seg, s)

            total_loss_D += loss_D
            total_loss_G = loss_G

            if iter % 50 == 0:
                self.wandb_log_img()
                self.wandb_log_losses(total_loss_D, total_loss_G)

        self.wandb_log_img()
        #         self.wandb_log_losses(total_loss_D/len(self.loader), total_loss_G/len(self.loader))

        self.wandb_log_losses(total_loss_D, total_loss_G)

    @torch.no_grad()
    def wandb_log_img(self):
        imgs, segs = self.log_batch
        s = RGB2n(segs.to(self.device), self.labels)

        self.model.eval()
        f, out_edge, out_img1, out_img2, _ = self.model(s)

        res = torch.cat([toRGB(imgs).cpu(),
                         segs.cpu(),
                         toRGB(out_img2).cpu(), toRGB(out_img1).cpu(), toRGB(out_edge).cpu()], dim=-2)

        res = [wandb.Image(to_pil_image(res[i])) for i in range(4)]

        wandb.log({"res": res})

    @torch.no_grad()
    def wandb_log_losses(self, loss_D, loss_G):
        wandb.log({'loss_D': loss_D, 'loss_G': loss_G})

    def update_D(self, img, s):
        self.optimizers['D'].zero_grad()

        img_edge = self.model.canny(img)
        f, out_edge, out_img1, out_img2, _ = self.model(s)

        edge_real_logits = self.model.discriminator(img_edge, s)
        edge_fake_logits = self.model.discriminator(out_edge, s)
        img_real_logits = self.model.discriminator(img, s)
        img_fake1_logits = self.model.discriminator(out_img1, s)
        img_fake2_logits = self.model.discriminator(out_img2, s)

        loss = self.losses['mma_D'](edge_real_logits, img_real_logits, edge_fake_logits, img_fake1_logits,
                                    img_fake2_logits)
        loss.backward()

        self.optimizers['D'].step()

        return loss

    def update_G(self, img, img_seg, s):
        self.optimizers['G'].zero_grad()
        f, out_edge, out_img1, out_img2, label_logits = self.model(s)
        # s_fake = RGB2n(pred_labels, self.labels)
        img_edge = self.model.canny(img)

        edge_real_logits = self.model.discriminator(img_edge, s).detach()
        edge_fake_logits = self.model.discriminator(out_edge, s)
        img_real_logits = self.model.discriminator(img, s).detach()
        img_fake1_logits = self.model.discriminator(out_img1, s)
        img_fake2_logits = self.model.discriminator(out_img2, s)

        loss_dict = {
            'mma_G': self.losses['mma_G'](edge_fake_logits, img_fake1_logits, img_fake2_logits),
            'pix_contr': self.losses['pix_contr'](img_seg, f, self.labels),
            'L1_img1': self.losses['L1'](img, out_img1),
            # 'L1_img2': self.losses['L1'](img, out_img2),
            'sim': self.losses['sim'](label_logits, s),
            'perc_edge': self.losses['perc'](img_edge.float(), out_edge.float()),
            'perc_img1': self.losses['perc'](img, out_img1),
            'perc_img2': self.losses['perc'](img, out_img2),
            'discr_f_edge': self.losses['discr_f'](edge_real_logits, edge_fake_logits),
            'discr_f_img1': self.losses['discr_f'](img_real_logits, img_fake1_logits),
            'discr_f_img2': self.losses['discr_f'](img_real_logits, img_fake2_logits)
        }

        loss = sum(loss_dict.values())

        loss.backward()

        self.optimizers['G'].step()

        return loss_dict

    def train(self):
        for self.epoch in range(1, self.config.n_epochs + 1):
            self.train_epoch()
            if self.epoch % 5 == 0:
                torch.save({
                    'model': self.model.state_dict(),
                    'optimizerG': self.optimizers['G'].state_dict(),
                    'optimizerD': self.optimizers['D'].state_dict()
                }, self.config.ckpt_path + f'/{self.config.dataset}_{self.epoch}ep.pt')
