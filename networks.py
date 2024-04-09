import torch
from torch import nn
from torch.nn.utils import spectral_norm
from utils import Canny


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
    def __init__(self, channels_in, c_hidden, channels_out, n):
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

        x_in = x

        f = self.encoder(x)

        out_edge = f
        out_img = f

        for conv_e, conv_i in zip(self.convs_edge,self.convs_img):
            out_edge = conv_e(out_edge)
            out_img = conv_i(out_img)
            out_img += self.sigmoid(out_edge) * out_img

        out_edge = self.edge_final(out_edge)
        out_edge = self.canny(out_edge) #TODO

        out_img = self.img_final(out_img)
        out_img += self.sigmoid(out_edge) * out_img

        return f, out_edge, out_img


class ECGAN(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = Generator(config.labels_cnt, config.c_hidden, 3)
        self.discriminator = Discrimintator(config.labels_cnt)
        self.semantic_preserving_module = SemanticPresevingModule(config)
        self.label_generator = LabelGenerator(config)


    def forward(self, s, img):
        f, out_edge, out_img1 = self.generator(s)

        out_img2 = self.semantic_preserving_module(torch.cat([s, f, out_edge, out_img1],dim=1))

        labels = self.label_generator(out_img2)

        return f, out_edge, out_img1, out_img2, labels











