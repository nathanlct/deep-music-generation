import model

import torch


def load_data():
    pass


generator = model.Generator()
discriminator = model.Discriminator()

batch_size = 32

z = torch.randn(batch_size, generator.latent_dim)
out = generator.forward(z, batch_size) ; print(out.shape)
out = discriminator.forward(out, batch_size) ; print(out.shape)
