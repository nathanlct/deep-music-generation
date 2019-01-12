import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.latent_dim = 100

        self.fc1 = nn.Linear(100, 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.tconv1 = nn.ConvTranspose2d(256, 256, (2,1), stride=(2,2))
        self.tconv2 = nn.ConvTranspose2d(256, 256, (2,1), stride=(2,2))
        self.tconv3 = nn.ConvTranspose2d(256, 256, (2,1), stride=(2,2))
        self.tconv4 = nn.ConvTranspose1d(256, 256, (1,128))

        self.fc_block = nn.Sequential(
            self.fc1, nn.ReLU(),
            self.fc2, nn.ReLU(),
        )

        self.conv_block = nn.Sequential(
            self.tconv1, nn.ReLU(),
            self.tconv2, nn.ReLU(),
            self.tconv3, nn.ReLU(),
            self.tconv4, nn.Sigmoid(),
        )

    def forward(self, z, batch_size):
        out = self.fc_block(z)
        out = out.view(batch_size, -1, 2, 1)
        out = self.conv_block(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(256, 14, (2,128), stride=2)
        self.conv2 = nn.Conv2d(14, 77, (4,1), stride=2)

        self.fc1 = nn.Linear(231, 1024) # 231 = 77*3
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x, batch_size):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = out.view(batch_size, -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))

        return out
