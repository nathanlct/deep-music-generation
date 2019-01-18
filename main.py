import model
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import encoder, encodings
# from encodings import debug
import torch.nn as nn
from scipy.misc import imshow

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
    batch_size = 2
    z = torch.randn(batch_size, netG.z_dim)
    out = netG.forward(z) ; print(out)
    out = netD.forward(out) ; print(out.shape)
# testdims()




batch_size = 32
lr = 2e-4
n_epochs = 50

# test
test_data = np.random.uniform(low=0.8,high=1.0,size=(32*50, 1, 16, 128))
# test_data = np.random.randn(32*3, 1, 16, 128)

bars = []
d = encoder.file_to_dictionary('data/Bach+Johann/10.mid')
plt.imshow(encodings.change_encoding(d,0,2)['Voice 1'][0])
for i in range(10,11):#257):
    x = encoder.file_to_dictionary('data/Bach+Johann/' + str(i) + '.mid')['Voice 1']
    bars += x
    print(x)
bars = np.array(bars, dtype=float)
# bars += np.random.randn(bars.shape[0],bars.shape[1],bars.shape[2])/10
bars[bars >= 1] = 1
bars[bars <= 0] = 0
bars = bars.reshape(-1, 1, 48, 128)[:32]
X = bars
print("bars echentillon : ",bars[0])
plt.imshow(bars[10][0])
plt.show()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


netG.apply(weights_init)
netD.apply(weights_init)

# print(netG)
# print(netD)



# len(X) must be a multiple of batch_size
# (otherwise, modify torch.split() so the last incomplete batch is not returned)

print("Size of data: {} ({} batches)".format(len(X), len(X) / batch_size))
print("Params: batch_size={}, lr={}, n_epochs={}".format(batch_size, lr, n_epochs))

optimizer_G = torch.optim.Adam(netG.parameters())
optimizer_D = torch.optim.Adagrad(netD.parameters())

lossG = []
lossD = []

def gamble(tensor):
    # for i in range(len(tensor)) :
    # print(tensor)
    # print(tensor.size())
    tensor = torch.reshape(tensor,[32,48*128])
    # print(tensor)
    tensor = torch.nn.functional.gumbel_softmax(tensor)
    tensor = torch.reshape(tensor,[32,1,48,128])
    return tensor

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

        # print(fake_batch.size)

        # m = torch.nn.Softmax()
        # fake_batch = m(fake_batch)
        fake_batch = gamble(fake_batch)

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

# init
criterion = nn.BCELoss()
fixed_noise = torch.randn(64, netG.z_dim, 1, 1)
real_label = 1
fake_label = 0


for epoch in range(1, n_epochs+1):

    np.random.shuffle(X)
    real_samples = torch.from_numpy(X).type(torch.FloatTensor)

    lossG_epoch = 0
    lossD_epoch = 0

    print("D and 4G ; epoch {}/{}".format(epoch, n_epochs))

    for real_batch in real_samples.split(batch_size):

        ###
        # improve discriminator
        ###
        # netD.zero_grad()
        # # on real batch
        # real = gpu(real_batch)
        # label = gpu(torch.full((batch_size,), real_label))
        # output = netD(real).view(-1)
        # errD_real = criterion(output, label)
        # errD_real.backward()
        # D_x = output.mean().item()
        #
        z = gpu(torch.randn(batch_size, netG.z_dim))
        fake_batch = netG(z)

        #
        # m = torch.nn.Softmax()
        # fake_batch = m(fake_batch)
        fake_batch = gamble(fake_batch)

        D_scr_on_real = netD(gpu(real_batch))
        D_scr_on_fake = netD(fake_batch)

        loss = - torch.mean(torch.log(1 - D_scr_on_fake) + torch.log(D_scr_on_real))
        optimizer_D.zero_grad()
        loss.backward()

        # z = gpu(torch.randn(batch_size, netG.z_dim))
        # fake = netG(z)
        # label.fill_(fake_label)
        # output = netD(fake.detach()).view(-1)
        # errD_fake = criterion(output, label)
        # errD_fake.backward()
        # optimizer_D.step()

        lossD_epoch += loss
        #
        # errD = errD_real + errD_fake
        # lossD_epoch += errD

        # improve generator twice
        for _ in range(4):

            fake_batch = netG(z)

            # m = torch.nn.Softmax()
            # fake_batch = m(fake_batch)

            D_scr_on_fake = netD(fake_batch)
            loss = -torch.mean(torch.log(D_scr_on_fake))
            optimizer_G.zero_grad()
            loss.backward()

            optimizer_G.step()
            lossG_epoch += loss

            # netG.zero_grad()
            # label.fill_(real_label)
            # z = gpu(torch.randn(batch_size, netG.z_dim))
            # fake = netG(z)
            # output = netD(fake).view(-1)
            # errG = criterion(output, label)
            # errG.backward()
            # optimizer_G.step()
            #
            # lossG_epoch += errG

    lossG.append(lossG_epoch)
    lossD.append(lossD_epoch)

    print("LossG: {}, LossD: {}".format(lossG_epoch, lossD_epoch))
    # print("fake_batch",fake_batch)



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

for i in range(20):
    lossG[i] = lossG[20]

plt.figure()
epochs = list(range(1, 20+n_epochs+1))
plt.plot(epochs, lossG, label='Generator loss')
plt.plot(epochs, lossD, label='Discriminator loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (non-normalized)')
plt.legend()
plt.show()

z = torch.randn(batch_size, netG.z_dim)
out = netG.forward(z)


plt.imshow(out[3][0].detach().numpy())
plt.show()

# m = torch.nn.functional.gumbel_softmax()
thing = torch.nn.functional.gumbel_softmax(out[3][0])
print(out[3][0].detach().numpy())
print(thing.detach().numpy())

# print("Sortie sur le générateur : ",out)
# print("Sortie sur le générateur : ",out[3][0].detach().numpy())

plt.imshow(thing.detach().numpy())
plt.show()
