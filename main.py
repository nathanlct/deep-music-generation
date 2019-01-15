import model
import os
import torch
import numpy as np
#import matplotlib.pyplot as plt
import encoder

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
#testdims()


batch_size = 32
lr = 2e-4
n_epochs = 50

# test
test_data = np.random.uniform(low=0.8,high=1.0,size=(32*50, 1, 16, 128))
# test_data = np.random.randn(32*3, 1, 16, 128)

bars = []
for i in range(1,257):
    x = encoder.file_to_dictionary('data/Bach+Johann/' + str(i) + '.mid')['Voice 1']
    bars += x
bars = np.array(bars, dtype=float)
bars += np.random.randn(bars.shape[0],bars.shape[1],bars.shape[2])/10
bars[bars >= 1] = 1
bars[bars <= 0] = 0
bars = bars.reshape(-1, 1, 48, 128)[:3968]
X = bars

# len(X) must be a multiple of batch_size
# (otherwise, modify torch.split() so the last incomplete batch is not returned)

print("Size of data: {} ({} batches)".format(len(X), len(X) / batch_size))
print("Params: batch_size={}, lr={}, n_epochs={}".format(batch_size, lr, n_epochs))

optimizer_G = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

lossG = []
lossD = []

for epoch in range(1, 21):

    np.random.shuffle(X)
    real_samples = torch.from_numpy(X).type(torch.FloatTensor)

    lossG_epoch = 0
    lossD_epoch = 0

    print("D only ; epoch {}/{}".format(epoch, 20))

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

    lossG.append(-5)
    lossD.append(lossD_epoch)

    print("LossD: {}".format(lossD))



for epoch in range(1, n_epochs+1):

    np.random.shuffle(X)
    real_samples = torch.from_numpy(X).type(torch.FloatTensor)

    lossG_epoch = 0
    lossD_epoch = 0

    print("D and 4G ; epoch {}/{}".format(epoch, n_epochs))

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
        for _ in range(4):
            z = gpu(torch.randn(batch_size, netG.z_dim))
            fake_batch = netG(z)
            D_scr_on_fake = netD(fake_batch)
            loss = -torch.mean(torch.log(D_scr_on_fake))
            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()
            lossG_epoch += loss

    lossG.append(lossG_epoch)
    lossD.append(lossD_epoch)

    print("LossG: {}, LossD: {}".format(lossG, lossD))



    if epoch == n_epochs:
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save({
            'epoch': epoch,
            'model_state_dict': netG.state_dict(),
            'optimizer_state_dict': optimizer_G.state_dict(),
            'loss': lossG,
        }, "models/netG_epoch_{}.pt".format(epoch))

        torch.save({
            'epoch': epoch,
            'model_state_dict': netD.state_dict(),
            'optimizer_state_dict': optimizer_D.state_dict(),
            'loss': lossD,
        }, "models/netD_epoch_{}.pt".format(epoch))


# plt.figure()
# epochs = list(range(1, 20+n_epochs+1))
# plt.plot(epochs, lossG, label='Generator loss')
# plt.plot(epochs, lossD, label='Discriminator loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss (non-normalized)')
# plt.legend()
# plt.show()


# z = torch.randn(batch_size, netG.z_dim)
# out = netG.forward(z)
# print(out)
# print(out[3])
