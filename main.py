import model
import torch
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    pass


use_gpu = torch.cuda.is_available()
print("Using " + ("GPU" if use_gpu else "CPU"))

def gpu(tensor, gpu=use_gpu):
    if gpu:
        return tensor.cuda()
    else:
        return tensor

netG = gpu(model.Generator())
netD = gpu(model.Discriminator())

def testdims():
    batch_size = 32
    z = torch.randn(batch_size, netG.z_dim)
    out = netG.forward(z) ; print(out.shape)
    out = netD.forward(out) ; print(out.shape)
testdims()

batch_size = 64
lr = 1e-3
n_epochs = 20

test_data = np.random.randn(64 * 10, 1, 48, 128)
X = test_data
# len(X) must be a multiple of batch_size
# (otherwise, modify torch.split() so the last incomplete batch is not returned)

print("Size of data: {} ({} batches)".format(len(X), len(X) / batch_size))
print("Params: batch_size={}, lr={}, n_epochs={}".format(batch_size, lr, n_epochs))

optimizer_G = torch.optim.Adam(netG.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(netD.parameters(), lr=lr)

lossG = []
lossD = []

for e in range(n_epochs):

    np.random.shuffle(X)
    real_samples = torch.from_numpy(X).type(torch.FloatTensor)

    lossG_epoch = 0
    lossD_epoch = 0

    print("epoch {}/{}".format(e+1, n_epochs))

    for real_batch in real_samples.split(batch_size):

        # improve discriminator
        z = gpu(torch.randn(batch_size, netG.z_dim))
        fake_batch = netG(z)

        D_scr_on_real = netD(gpu(real_batch))
        D_scr_on_fake = netD(fake_batch)

        loss = - torch.mean(torch.log(1 - D_scr_on_fake) + torch.log(D_scr_on_real))
        optimizer_D.zero_grad()
        loss.backward()
        optimizer_D.step()

        lossD_epoch += loss

        # improve generator twice
        for _ in range(2):
            z = gpu(torch.randn(batch_size, netG.z_dim))
            fake_batch = netG(z)

            D_scr_on_fake = netD(fake_batch)

            loss = - torch.mean(torch.log(D_scr_on_fake))
            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()

            lossG_epoch += loss

    lossG.append(lossG_epoch)
    lossD.append(lossD_epoch)


plt.figure()
epochs = list(range(1, n_epochs+1))
plt.plot(epochs, lossG, label='Generator loss')
plt.plot(epochs, lossD, label='Discriminator loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (non-normalized)')
plt.legend()
plt.show()
