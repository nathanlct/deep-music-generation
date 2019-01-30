import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
# import ipdb
import torchvision.utils as vutils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import *
from ops import *
import sys
import os
import encoder
import reencodings
import time


def main():
    is_train = 1
    is_draw = 0
    is_sample = 0

    batch_size = 288 * 2
    epochs = 1000
    lr = 0.0002

    check_range_st = 0
    check_range_ed = 129
    pitch_range = check_range_ed - check_range_st-1

    device = torch.device('cuda')#('cpu')

    """ DATA LOADING """
    bars = []
    # N = 10#313

    print("Loading data")
    for dir in [""]:#os.listdir("./data"):
        if dir != ".DS_Store":
            dir_path = os.path.join("../data", dir)
            for file in os.listdir(dir_path):
                if file[-1] == 'd':
                    print(os.path.join(dir_path, file))
                    x = encoder.file_to_dictionary(os.path.join(dir_path, file))
                    # x = encoder.file_to_dictionary('../data/Bach+Johann/' + str(i) + '.mid')
                    if len(x) > 0:
                        x = reencodings.change_encoding(x, 0, 1)['Voice 1']
                        if len(x)>0:
                            bars.append((np.zeros((128, 16)), x[0][:,:16]))
                            bars.append((x[0][:,:16], x[0][:,16:32]))
                            bars.append((x[0][:,16:32], x[0][:,32:]))
                            for j in range(len(x)-1):
                                bars.append((x[j][:,32:], x[j+1][:,:16]))
                                bars.append((x[j+1][:,:16], x[j+1][:,16:32]))
                                bars.append((x[j+1][:,16:32], x[j+1][:,32:]))
    X = np.array(bars, dtype=float)
    X = X.reshape(X.shape[0], 2, 1, 128, 16)
    X = np.transpose(X, (0,1,2,4,3)) # X: (n_batches, 2, 1, 48, 128)
    X = X[:len(X)-(len(X)%batch_size)]

    print("Dataset size:", len(X))




    if is_train == 1 :
        netG = generator(pitch_range).to(device)
        netD = discriminator(pitch_range).to(device)


        # netG.load_state_dict(torch.load('models/netG_epoch_9.pth'))
        # netD.load_state_dict(torch.load('models/netD_epoch_9.pth'))

        netD.train()
        netG.train()
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))


        nz = 100
        fixed_noise = torch.randn(batch_size, nz, device=device)
        real_label = 1
        fake_label = 0
        average_lossD = 0
        average_lossG = 0
        average_D_x   = 0
        average_D_G_z = 0

        lossD_list =  []
        lossD_list_all = []
        lossG_list =  []
        lossG_list_all = []
        D_x_list = []
        D_G_z_list = []
        for epoch in range(epochs):
            t = time.time()
            sum_lossD = 0
            sum_lossG = 0
            sum_D_x   = 0
            sum_D_G_z = 0
            average_lossD = 0
            average_lossG = 0
            average_D_x   = 0
            average_D_G_z = 0

            np.random.shuffle(X)
            dataset = torch.from_numpy(X).type(torch.FloatTensor)

            i = 0
            for i, data_tmp in enumerate(dataset.split(batch_size), 0):
                prev_data = data_tmp[:,0]
                data = data_tmp[:,1]

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                netD.zero_grad()
                real_cpu = data.to(device)
                prev_data_cpu = prev_data.to(device)

                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), real_label, device=device)
                D, D_logits, fm = netD(real_cpu,batch_size,pitch_range)

                #####loss
                d_loss_real = reduce_mean(sigmoid_cross_entropy_with_logits(D_logits, 0.9*torch.ones_like(D)))
                d_loss_real.backward(retain_graph=True)
                D_x = D.mean().item()
                sum_D_x += D_x

                # train with fake
                noise = torch.randn(batch_size, nz, device=device)
                fake = netG(noise,prev_data_cpu,batch_size,pitch_range)
                label.fill_(fake_label)
                D_, D_logits_, fm_ = netD(fake.detach(),batch_size,pitch_range)
                d_loss_fake = reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, torch.zeros_like(D_)))

                d_loss_fake.backward(retain_graph=True)
                D_G_z1 = D_.mean().item()
                errD = d_loss_real + d_loss_fake
                errD = errD.item()
                lossD_list_all.append(errD)
                sum_lossD += errD
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                D_, D_logits_, fm_= netD(fake,batch_size,pitch_range)

                ###loss
                g_loss0 = reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, torch.ones_like(D_)))
                #Feature Matching
                features_from_g = reduce_mean_0(fm_)
                features_from_i = reduce_mean_0(fm)
                fm_g_loss1 =torch.mul(l2_loss(features_from_g, features_from_i), 0.1)

                mean_image_from_g = reduce_mean_0(fake)
                smean_image_from_i = reduce_mean_0(real_cpu)
                fm_g_loss2 = torch.mul(l2_loss(mean_image_from_g, smean_image_from_i), 0.01)

                errG = g_loss0 + fm_g_loss1 + fm_g_loss2
                errG.backward(retain_graph=True)
                D_G_z2 = D_.mean().item()
                optimizerG.step()

                ############################
                # (3) Update G network again: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                D_, D_logits_, fm_ = netD(fake,batch_size,pitch_range)

                ###loss
                g_loss0 = reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, torch.ones_like(D_)))
                #Feature Matching
                features_from_g = reduce_mean_0(fm_)
                features_from_i = reduce_mean_0(fm)
                loss_ = nn.MSELoss(reduction='sum')
                feature_l2_loss = loss_(features_from_g, features_from_i)/2
                fm_g_loss1 =torch.mul(feature_l2_loss, 0.1)

                mean_image_from_g = reduce_mean_0(fake)
                smean_image_from_i = reduce_mean_0(real_cpu)
                mean_l2_loss = loss_(mean_image_from_g, smean_image_from_i)/2
                fm_g_loss2 = torch.mul(mean_l2_loss, 0.01)
                errG = g_loss0 + fm_g_loss1 + fm_g_loss2
                sum_lossG +=errG
                errG.backward()
                lossG_list_all.append(errG.item())

                D_G_z2 = D_.mean().item()
                sum_D_G_z += D_G_z2
                optimizerG.step()

                if epoch % 5 == 0 and i % 50 == 0:
                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                          % (epoch, epochs, i, int(len(dataset)/batch_size),
                             errD, errG, D_x, D_G_z1, D_G_z2))

#                if i % 100 == 0:
 #                   vutils.save_image(real_cpu,
  #                           '%s/real_samples.png' % 'file',
   #                          normalize=True)
    #                fake = netG(fixed_noise,prev_data_cpu,batch_size,pitch_range)
     #               vutils.save_image(fake.detach(),
      #                      '%s/fake_samples_epoch_%03d.png' % ('file', epoch),
       #                     normalize=True)

            average_lossD = (sum_lossD / len(dataset))
            average_lossG = (sum_lossG / len(dataset))
            average_D_x = (sum_D_x / len(dataset))
            average_D_G_z = (sum_D_G_z / len(dataset))

            lossD_list.append(average_lossD)
            lossG_list.append(average_lossG)
            D_x_list.append(average_D_x)
            D_G_z_list.append(average_D_G_z)

            tt = time.time()
            print('==> Epoch: {} Average lossD: {:.10f} average_lossG: {:.10f},average D(x): {:.10f},average D(G(z)): {:.10f} ,time: {:.1f} '.format(
              epoch, average_lossD,average_lossG,average_D_x, average_D_G_z, tt-t))

            del average_lossD
            del average_lossG
            del average_D_x
            del average_D_G_z
            del sum_lossD
            del sum_lossG
            del sum_D_x
            del sum_D_G_z

            
		if epoch%100 == 0:
		    np.save('lossD_list.npy',lossD_list)
		    np.save('lossG_list.npy',lossG_list)
		    np.save('lossD_list_all.npy',lossD_list_all)
		    np.save('lossG_list_all.npy',lossG_list_all)
		    np.save('D_x_list.npy',D_x_list)
		    np.save('D_G_z_list.npy',D_G_z_list)

		# do checkpointing
		    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % ('./models', epoch))
		    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % ('./models', epoch))

	    np.save('lossD_list.npy',lossD_list)
	    np.save('lossG_list.npy',lossG_list)
	    np.save('lossD_list_all.npy',lossD_list_all)
	    np.save('lossG_list_all.npy',lossG_list_all)
	    np.save('D_x_list.npy',D_x_list)
	    np.save('D_G_z_list.npy',D_G_z_list)

	    
	    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % ('./models', epoch))
	    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % ('./models', epoch))

    if is_draw == 1:
        lossD_print = np.load('lossD_list.npy')
        lossG_print = np.load('lossG_list.npy')
        length = lossG_print.shape[0]

        x = np.linspace(0, length-1, length)
        x = np.asarray(x)
        plt.figure()
        plt.plot(x, lossD_print,label=' lossD',linewidth=1.5)
        plt.plot(x, lossG_print,label=' lossG',linewidth=1.5)

        plt.legend(loc='upper right')
        plt.xlabel('data')
        plt.ylabel('loss')
        plt.savefig('where you want to save/lr='+ str(lr) +'_epoch='+str(epochs)+'.png')

    if is_sample == 1:
        batch_size = 8
        nz = 100
        n_bars = 7
        X_te = np.load('yourtestingx')
        prev_X_te = np.load('yourtestingprevx')
        prev_X_te = prev_X_te[:,:,check_range_st:check_range_ed,:]
        y_te    = np.load('yourdchord')

        test_iter = get_dataloader(X_te,prev_X_te,y_te)
        kwargs = {'num_workers': 8, 'pin_memory': True}# if args.cuda else {}
        test_loader = DataLoader(test_iter, batch_size=batch_size, shuffle=False, **kwargs)

        netG = generator(pitch_range)
        netG.load_state_dict(torch.load('models/netG_epoch_199.pth'))

        output_songs = []
        output_chords = []
        for i, (data,prev_data,chord) in enumerate(test_loader, 0):
            list_song = []
            first_bar = data[0].view(1,1,16,128)
            list_song.append(first_bar)

            list_chord = []
            first_chord = chord[0].view(1,13).numpy()
            list_chord.append(first_chord)
            noise = torch.randn(batch_size, nz)

            for bar in range(n_bars):
                z = noise[bar].view(1,nz)
                y = chord[bar].view(1,13)
                if bar == 0:
                    prev = data[0].view(1,1,16,128)
                else:
                    prev = list_song[bar-1].view(1,1,16,128)
                sample = netG(z, prev, y, 1,pitch_range)
                list_song.append(sample)
                list_chord.append(y.numpy())

            print('num of output_songs: {}'.format(len(output_songs)))
            output_songs.append(list_song)
            output_chords.append(list_chord)
        np.save('output_songs.npy',np.asarray(output_songs))
        np.save('output_chords.npy',np.asarray(output_chords))

        print('creation completed, check out what I make!')


if __name__ == "__main__" :

    main()
