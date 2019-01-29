import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import torch
import torch.nn as nn
import sys

import encoder
import reencodings
import model
from downloader import PATH




use_gpu = torch.cuda.is_available()
print("Using " + ("GPU" if use_gpu else "CPU"))

def gpu(tensor, gpu=use_gpu):
    if gpu:
        return tensor.cuda()
    else:
        return tensor


"""
DATA LOADING
"""

SCORES_PATH = os.path.join(PATH, 'scores')
LABELS_PATH = os.path.join(PATH, 'labels')

def load_data():
    if os.path.isfile(SCORES_PATH) and os.path.isfile(LABELS_PATH):
        return 0
    else:
        scores = []
        labels = []
        for dir in os.listdir(PATH):
            print(dir)
            dir_path = os.path.join(PATH, dir)
            for file in os.listdir(dir_path):
                scores.append(encoder.file_to_dictionary(os.path.join(dir_path, file)))
                labels.append(dir)

    raise AttributeError

def show_bar(bar, threshold=None):
    if threshold:
        bar = bar.numpy()
        bar[bar <= threshold] = 0
        bar[bar > threshold] = 1
    if use_gpu:
        print(bar)
    else:
        plt.imshow(bar)
        plt.show()



batch_size = 16 * 2 # all voices are 18 tabs (or not)


bars = []
d = encoder.file_to_dictionary('data/Bach+Johann/10.mid')

N = 100
for i in range(1, N):
    x = encoder.file_to_dictionary('data/Bach+Johann/' + str(i) + '.mid')
    x = reencodings.change_encoding(x, 0, 1)['Voice 1']
    bars += x

X = np.array(bars, dtype=float)

X = X.reshape(X.shape[0], 1, 128, 48)
X = np.transpose(X, (0,1,3,2)) # X: (n_batches, 1, 48, 128)

X = X[:len(X)-(len(X)%batch_size)]
# len(X) must be a multiple of batch_size
# (gotta modify torch.split() so the last incomplete batch is not returned)

"""
Initialization
"""

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

lr = 2e-4
n_epochs = 30

G_training_ratio = 2 # number of times the generator is trained at each iteration

netG = gpu(model.Generator())
netD = gpu(model.Discriminator())

netG.apply(weights_init)
netD.apply(weights_init)

# print(netG)
# print(netD)

print("Size of data: {} ({} batches)".format(len(X), len(X) / batch_size))
print("Params: batch_size={}, lr={}, n_epochs={}, G_training_ratio={}".format(batch_size, lr, n_epochs, G_training_ratio))

optimizer_G = torch.optim.Adam(netG.parameters())
optimizer_D = torch.optim.Adagrad(netD.parameters())

lossG = []
lossD = []

def gamble(tensor):
    tensor = torch.reshape(tensor, [batch_size, 48*128])
    tensor = torch.nn.functional.gumbel_softmax(tensor)
    tensor = torch.reshape(tensor, [batch_size,1,48,128])
    return tensor

fixed_noise = gpu(torch.randn(batch_size, netG.z_dim))

"""
TRAINING
"""

criterion = nn.BCELoss()

for epoch in range(1, n_epochs+1):

    # shuffle training data (no conditioner for the moment)

    np.random.shuffle(X)
    real_samples = torch.from_numpy(X).type(torch.FloatTensor)

    lossG_epoch = 0
    lossD_epoch = 0
    n_batch = 0

    # train

    for real_batch in real_samples.split(batch_size):

        n_batch += 1

        # improve discriminator

        optimizer_D.zero_grad()

        # on real
        real = gpu(real_batch)
        real_label = gpu(torch.full((batch_size,), 1))
        output_real = netD(real).view(-1)
        errD_real = criterion(output_real, real_label)
        errD_real.backward()

        # on fake
        z = gpu(torch.randn(batch_size, netG.z_dim))
        fake = netG(z)
        fake_label = gpu(torch.full((batch_size,), 0))
        output_fake = netD(fake.detach()).view(-1)
        errD_fake = criterion(output_fake, fake_label)
        errD_fake.backward()

        optimizer_D.step()

        lossD_epoch += errD_real + errD_fake

        # improve generator

        for _ in range(G_training_ratio):

            optimizer_G.zero_grad()

            z = gpu(torch.randn(batch_size, netG.z_dim))
            fake = netG(z)
            fake_label = gpu(torch.full((batch_size,), 0))
            output = netD(fake).view(-1)
            errG = criterion(output, fake_label)
            errG.backward()

            optimizer_G.step()

            lossG_epoch += errG


    lossG.append(lossG_epoch / (G_training_ratio * n_batch))
    lossD.append(lossD_epoch / n_batch)

    print("End of epoch {}/{}; lossG={}, lossD={}".format(epoch, n_epochs, lossG[-1], lossD[-1]))

    # if epoch == 1 or epoch % 5 == 0:
    #     print("Generator sample")
    #     gen_output = gamble(netG(fixed_noise))
    #     show_bar(gen_output[0][0].detach(), threshold=0.1)

    # save networks states

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




plt.figure()
epochs = list(range(1, n_epochs+1))
plt.plot(epochs, lossG, label='Generator loss')
plt.plot(epochs, lossD, label='Discriminator loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


gen_output = netG(fixed_noise)
show_bar(gen_output[0][0].detach(), threshold=0.5)
show_bar(gen_output[1][0].detach(), threshold=0.5)
show_bar(gen_output[2][0].detach(), threshold=0.5)
show_bar(gen_output[3][0].detach(), threshold=0.1)
show_bar(gen_output[4][0].detach(), threshold=0.2)
show_bar(gen_output[5][0].detach(), threshold=0.3)
show_bar(gen_output[6][0].detach(), threshold=0.4)
show_bar(gen_output[7][0].detach(), threshold=0.6)
show_bar(gen_output[8][0].detach(), threshold=0.7)
show_bar(gen_output[9][0].detach(), threshold=0.8)
show_bar(gen_output[10][0].detach(), threshold=0.9)
show_bar(gen_output[11][0].detach())
