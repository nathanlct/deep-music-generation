
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.z_dim = 100

        self.fc1 = nn.Linear(100, 1024)
        self.fc2 = nn.Linear(1024, 513)

        self.tconv1 = nn.ConvTranspose2d(171, 171, (2,1), stride=2) # 171 = 513/3
        self.tconv2 = nn.ConvTranspose2d(171, 171, (2,1), stride=2)
        self.tconv3 = nn.ConvTranspose2d(171, 171, (2,1), stride=2)
        self.tconv4 = nn.ConvTranspose2d(171, 171, (2,1), stride=2)
        self.tconv5 = nn.ConvTranspose1d(171, 1, (1,128))

        self.fc_block = nn.Sequential(
            self.fc1, nn.BatchNorm1d(1024), nn.ReLU(),
            self.fc2, nn.BatchNorm1d(513), nn.ReLU(),
        )

        self.conv_block = nn.Sequential(
            self.tconv1, nn.BatchNorm2d(171), nn.ReLU(),
            self.tconv2, nn.BatchNorm2d(171), nn.ReLU(),
            self.tconv3, nn.BatchNorm2d(171), nn.ReLU(),
            self.tconv4, nn.BatchNorm2d(171), nn.ReLU(),
            self.tconv5, nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.fc_block(z)
        out = out.view(z.shape[0], -1, 3, 1)
        out = self.conv_block(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(1, 14, (2,128), stride=2)
        self.conv2 = nn.Conv2d(14, 77, (12,1), stride=6)

        self.fc1 = nn.Linear(231, 1024) # 231 = 77*3
        self.fc2 = nn.Linear(1024, 1)

        self.conv_block = nn.Sequential(
            self.conv1, nn.BatchNorm2d(14), nn.LeakyReLU(),
            self.conv2, nn.BatchNorm2d(77), nn.LeakyReLU(),
        )

        self.fc_block = nn.Sequential(
            self.fc1, nn.BatchNorm1d(1024), nn.LeakyReLU(),
            self.fc2, nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv_block(x)
        out = out.view(x.shape[0], -1)
        out = self.fc_block(out)
        return out
