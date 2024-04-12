import torch
import numpy as np
from torch import nn
from torch.nn.utils import spectral_norm
from utils import Canny, toRGB, get_palette
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


class ResBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()

        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels_out)

        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels_out)

        self.relu = nn.ReLU()

        self.projection = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1)
        )

    def forward(self, x):
        x_in = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.projection(x_in) + x
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()

        self.net = nn.Sequential(
            ResBlock(channels_in, channels_out // 8),
            ResBlock(channels_out // 8, channels_out // 4),
            ResBlock(channels_out // 4, channels_out // 2),
            ResBlock(channels_out // 2, channels_out),
        )

    def forward(self, x):
        return self.net(x)


class ConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv = spectral_norm(nn.Conv2d(channels_in, channels_out, 3, padding=1))

    def forward(self, x):
        x = self.conv(x)
        return x


class Generator(nn.Module):
    def __init__(self, channels_in, c_hidden, channels_out, n=3):
        super().__init__()
        self.n = n

        self.encoder = Encoder(channels_in, c_hidden)

        convs_edge = []
        convs_img = []

        for _ in range(n):
            convs_edge.append(ConvBlock(c_hidden, c_hidden // 2))
            convs_img.append(ConvBlock(c_hidden, c_hidden // 2))
            c_hidden //= 2

        self.convs_edge = nn.ModuleList(convs_edge)
        self.convs_img = nn.ModuleList(convs_img)

        self.edge_final = nn.Sequential(
            ConvBlock(c_hidden, channels_out),
            nn.Tanh()
        )

        self.img_final = nn.Sequential(
            ConvBlock(c_hidden, channels_out),
            nn.Tanh()
        )

        self.sigmoid = nn.Sigmoid()

        self.canny = Canny()

    def forward(self, x):

        f = self.encoder(x)

        out_edge = f
        out_img = f

        for conv_e, conv_i in zip(self.convs_edge, self.convs_img):
            out_edge = conv_e(out_edge)
            out_img = conv_i(out_img)

            out_edge = torch.nn.functional.relu(out_edge)
            out_img = torch.nn.functional.relu(out_img)

            out_img = out_img + self.sigmoid(out_edge) * out_img

        out_edge = self.edge_final(out_edge)
        out_edge = self.canny(out_edge)

        out_img = self.img_final(out_img)
        out_img = out_img + self.sigmoid(out_edge) * out_img

        out_img = torch.tanh(out_img)

        return f, out_edge, out_img


class SemanticPreserveModule(nn.Module):
    def __init__(self, CFG):
        super().__init__()

        self.conv1 = spectral_norm(nn.Conv2d(in_channels=CFG.c_hidden + CFG.semantic_classes + 6,
                                             out_channels=CFG.semantic_classes,
                                             kernel_size=CFG.kernel_size, padding=CFG.padding))
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = spectral_norm(nn.Conv2d(in_channels=CFG.semantic_classes,
                                             out_channels=CFG.c_hidden + CFG.semantic_classes + 6,
                                             kernel_size=CFG.kernel_size, padding=CFG.padding))

        self.final = spectral_norm(nn.Conv2d(in_channels=CFG.c_hidden + CFG.semantic_classes + 6,
                                             out_channels=3,
                                             kernel_size=CFG.kernel_size, padding=CFG.padding))
        self.tanh = nn.Tanh()

    def forward(self, x):
        identity = self.conv1(x)

        out = self.avg_pool(identity)
        out = self.sigmoid(self.avg_pool(out))
        out = out * identity + identity

        out = self.conv2(out)

        out = self.tanh(self.final(out))
        return out


class Discriminator(nn.Module):
    def __init__(self, CFG):
        super(Discriminator, self).__init__()

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)  # 256 -> 128
        self.layers = nn.ModuleList([
            self._downsample_block(3 + CFG.semantic_classes, 64),  # 128 -> 64
            self._downsample_block(64, 128),  # 64 -> 32
            self._downsample_block(128, 256),  # 32 -> 16
            self._downsample_block(256, 512),  # 16 -> 8
        ]
        )
        self.conv = nn.Conv2d(512, 64, kernel_size=1)
        self.fc1 = nn.Sequential(
            spectral_norm(nn.Linear(64 * CFG.H // 32 * CFG.W // 32, 128)),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2))
        self.fc2 = nn.Linear(128, 1)

    def _downsample_block(self, input_channels, output_channels):
        return nn.Sequential(
            spectral_norm(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=2)),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x, y):  # B x C x H x W
        intermediate_features = []
        x1 = torch.concat([x, y], dim=1)  # B x C + N x H x W
        x1 = self.avg_pool(x1)
        for block in self.layers:
            x1 = block(x1)
        x1 = self.conv(x1).reshape(x1.shape[0], -1)
        x1 = self.fc1(x1)
        intermediate_features.append(x1)
        x1 = self.fc2(x1)
        return x1, intermediate_features


class LabelGenerator(nn.Module):
    def __init__(self, CFG):
        super(LabelGenerator, self).__init__()

        self.processor = SegformerImageProcessor(do_resize=False)
        self.model = SegformerForSemanticSegmentation.from_pretrained(CFG.LG_model_name)
        self.device = CFG.device
        self.palette = get_palette(CFG.dataset)
        self.model.to(self.device)

    def get_resized_logits(self, logits, target_sizes):
        resized_logits = []
        for idx in range(len(logits)):
            resized_logits.append(torch.nn.functional.interpolate(
                logits[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
            ))
        return torch.cat(resized_logits)

    def forward(self, imgs):
        pixel_values = self.processor(toRGB(imgs), return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values)
            logits = outputs.logits

        target_sizes = [imgs.shape[2:] for _ in range(imgs.shape[0])]
        resized_logits = self.get_resized_logits(logits, target_sizes)
        return resized_logits  # RGB non-normalized


class ECGAN(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = Generator(config.semantic_classes, config.c_hidden, 3)
        self.discriminator = Discriminator(config)
        self.semantic_preserving_module = SemanticPreserveModule(config)
        self.label_generator = LabelGenerator(config)
        self.canny = Canny()

    def forward(self, s):
        f, out_edge, out_img1 = self.generator(s)

        out_img2 = self.semantic_preserving_module(torch.cat([s, f, out_edge, out_img1], dim=1))

        label_logits = self.label_generator(out_img2)

        return f, out_edge, out_img1, out_img2, label_logits
