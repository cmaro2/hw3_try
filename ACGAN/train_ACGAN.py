from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import numpy as np
from ACGAN.Gen_ACGAN import Generator
from ACGAN.Disc_ACGAN import Discriminator
from ACGAN.parser_2 import arg_parse
import matplotlib.pyplot as plt
from ACGAN import data_ACGAN
from PIL import Image

if __name__ == '__main__':
    #load args
    args = arg_parse()

    def save_model(model, save_path):
        torch.save(model.state_dict(), save_path)

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    real_faces = torch.utils.data.DataLoader(data_ACGAN.DATA(args),
                                             batch_size=args.train_batch,
                                             num_workers=args.workers,
                                             shuffle=True)

    # Get random seed
    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    device = torch.device('cuda')
    g_mult = 1
    # Create the generator
    netG = Generator(nz=args.nz).cuda()
    # Create the Discriminator
    netD = Discriminator().cuda()

    # Apply random weights to the Discriminator and the Generator
    netD.apply(weights_init)
    netG.apply(weights_init)

    # Set criterion as BCELoss function and auxiliary loss
    criterion = nn.BCELoss().cuda()
    auxiliary_loss = nn.CrossEntropyLoss().cuda()

    noise = torch.FloatTensor(args.train_batch, args.nz, 1, 1).cuda()
    eval_noise = torch.FloatTensor(args.train_batch, args.nz, 1, 1).normal_(0, 1).cuda()
    label = torch.FloatTensor(args.train_batch).cuda()
    label2 = torch.FloatTensor(args.train_batch*g_mult).cuda()
    aux_label = torch.LongTensor(args.train_batch).cuda()

    # Used to visualize the progression of the generator
    fixed_noise = torch.randn(10, args.nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    print_num = (40000/5)/args.train_batch

    # define variables
    noise = Variable(noise)
    eval_noise = Variable(eval_noise)
    label = Variable(label)
    label2 = Variable(label2)
    aux_label = Variable(aux_label)
    # noise for evaluation
    eval_noise_ = np.random.normal(0, 1, (args.train_batch, args.nz))
    eval_label = np.random.randint(0, 2, args.train_batch)
    eval_onehot = np.zeros((args.train_batch, 2))
    eval_onehot[np.arange(args.train_batch), eval_label] = 1
    eval_noise_[np.arange(args.train_batch), :2] = eval_onehot[np.arange(args.train_batch)]
    eval_noise_ = (torch.from_numpy(eval_noise_))
    eval_noise.data.copy_(eval_noise_.view(args.train_batch, args.nz, 1, 1))

    # Setup Adam optimizers for Discriminator and the Generator
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta, 0.999), weight_decay=args.weight_decay)

    real_faces = torch.utils.data.DataLoader(data_ACGAN.DATA(args),
                                             batch_size=args.train_batch,
                                             num_workers=args.workers,
                                             shuffle=True)

    #Training loop __________________________________________________

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    max_batch_losses = 0
    iters = 0
    torch.save(fixed_noise, 'seed.pt')

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(args.epoch):
        batch_losses = 0
        avg_loss_d = 0
        avg_loss_g = 0
        avg_dx = 0
        avg_dgz1 = 0
        avg_dgz2 = 0
        avg_smile = 0
        avg_smile_d = 0
        for i, (images, smile) in enumerate(real_faces):

            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            image_batch = images.cuda()
            batch_size = image_batch.size(0)
            label.fill_(real_label)
            aux_label.data.resize_(batch_size).copy_(smile.view(-1))
            #smile = Variable(smile.type(torch.cuda.LongTensor)).view(-1)

            ## Train with all-real batch
            # Format batch
            netD.zero_grad()
            output, smile_D = netD(image_batch)
            #output = output.view(-1)
            #smile_D = smile_D.view(-1)
            # Calculate loss on real
            errD_real_rf = criterion(output, label)
            errD_real_sm = auxiliary_loss(smile_D, aux_label)
            errD_real = errD_real_rf + errD_real_sm
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()


            ## Train with all-fake batch
            # Generate fake image batch with G
            noise.data.resize_(batch_size, args.nz, 1, 1).normal_(0, 1)
            fake_smile = np.random.randint(0, 2, batch_size)
            noise_ = np.random.normal(0, 1, (batch_size, args.nz))
            class_onehot = np.zeros((batch_size, 2))
            class_onehot[np.arange(batch_size), fake_smile] = 1
            noise_[np.arange(batch_size), :2] = class_onehot[np.arange(batch_size)]
            noise_ = (torch.from_numpy(noise_))
            noise.data.copy_(noise_.view(batch_size, args.nz, 1, 1))
            aux_label.data.resize_(batch_size).copy_(torch.from_numpy(fake_smile))

            fake = netG(noise)

            label.fill_(fake_label)
            # Classify all fake batch with D
            output, smile_D = netD(fake.detach())
            errD_fake_rf = criterion(output, label)
            errD_fake_sm = auxiliary_loss(smile_D, aux_label)
            errD_fake = errD_fake_rf + errD_fake_sm
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()


            # (2) Update G network: maximize log(D(G(z)))_______________________________
            """
            noise.data.resize_(batch_size*g_mult, args.nz, 1, 1).normal_(0, 1)
            fake_smile = np.random.randint(0, 2, batch_size*2)
            noise_ = np.random.normal(0, 1, (batch_size*g_mult, args.nz))
            class_onehot = np.zeros((batch_size*g_mult, 2))
            class_onehot[np.arange(batch_size*g_mult), fake_smile] = 1
            noise_[np.arange(batch_size*g_mult), :2] = class_onehot[np.arange(batch_size*g_mult)]
            noise_ = (torch.from_numpy(noise_))
            noise.data.copy_(noise_.view(batch_size*g_mult, args.nz, 1, 1))
            aux_label.data.resize_(batch_size*g_mult).copy_(torch.from_numpy(fake_smile))

            fake = netG(noise)
"""
            netG.zero_grad()
            label2.fill_(real_label)
            # Perform another forward pass of all-fake batch through D
            output, smile_f = netD(fake)
            # Calculate G's loss based on this output
            errG_rf = criterion(output, label2)
            errG_sm = auxiliary_loss(smile_f, aux_label)
            errG = errG_rf + errG_sm
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            avg_loss_d += errD.item()
            avg_loss_g += errG.item()
            avg_dx += D_x
            avg_dgz1 += D_G_z1
            avg_dgz2 += D_G_z2
            avg_smile += errD_real_sm.item()
            avg_smile_d += errD_fake_sm.item()

            # Output training stats
            if (i+1) % print_num == 0:
                print('Epoch: %d/%d\tbatches: %d/%d\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tSmile error: %.4f / %.4f'
                      % (epoch, args.epoch, i+1, len(real_faces),
                         avg_loss_d/print_num, avg_loss_g/print_num, avg_dx/print_num, avg_dgz1/print_num, avg_dgz2/print_num, avg_smile/print_num, avg_smile_d/print_num))
                avg_loss_d = 0
                avg_loss_g = 0
                avg_dx = 0
                avg_dgz1 = 0
                avg_dgz2 = 0
                avg_smile = 0
                avg_smile_d = 0
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            batch_losses += D_G_z2
            iters += 1
        print('Max losses: ' + str(max_batch_losses) + ', this batch losses: ' + str(batch_losses))
        if(batch_losses > max_batch_losses):
            max_batch_losses = batch_losses
            save_model(netG, os.path.join(args.save_dir, 'netG_best.pth.tar'))

        save_model(netG, os.path.join(args.save_dir, 'net_G_{}.pth.tar'.format(epoch)))
        with torch.no_grad():
            fake = fake.detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        img_only = np.transpose(img_list[-1], (1, 2, 0)).numpy()
        img_only = (img_only * 255)
        img_array = np.array(img_only, dtype=np.uint8)
        result = Image.fromarray(img_array)
        result.save('images/' + 'image_3_{}.png'.format(epoch))

    #save_model(netG, os.path.join(args.save_dir, 'netG.pth.tar'))
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()