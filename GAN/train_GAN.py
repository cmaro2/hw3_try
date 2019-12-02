from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
from GAN.Gen_GAN import Generator
from GAN.Disc_GAN import Discriminator
from parser_2 import arg_parse
import matplotlib.pyplot as plt
from GAN import data_GAN

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


    # Get random seed
    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device('cuda')

    # Create the generator
    netG = Generator(args.ngpu).to(device)
    # Create the Discriminator
    netD = Discriminator(args.ngpu).to(device)

    # Apply random weights to the Discriminator and the Generator
    netD.apply(weights_init)
    netG.apply(weights_init)

    # Set criterion as BCELoss function
    criterion = nn.BCELoss()

    # Used to visualize the progression of the generator
    fixed_noise = torch.randn(32, 100, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0
    print_num = 1000

    # Setup Adam optimizers for Discriminator and the Generator
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta, 0.999), weight_decay=args.weight_decay)
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta, 0.999), weight_decay=args.weight_decay)

    real_faces = torch.utils.data.DataLoader(data_GAN.DATA(args),
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
        for i, images in enumerate(real_faces):

            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            image_batch = images.to(device)
            batch_size = image_batch.size(0)
            label = torch.full((batch_size,), real_label, device=device)
            output = netD(image_batch).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate fake image batch with G
            label2 = torch.full((batch_size,), fake_label, device=device)
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake = netG(noise)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label2)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            # (2) Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            label2.fill_(real_label)
            # Perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label2)
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

            # Output training stats
            if i % print_num == 0:
                print('Epoch: %d/%d\tbatches: %d/%d\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args.epoch, i, len(real_faces),
                         avg_loss_d/print_num, avg_loss_g/print_num, avg_dx/print_num, avg_dgz1/print_num, avg_dgz2/print_num))
                avg_loss_d = 0
                avg_loss_g = 0
                avg_dx = 0
                avg_dgz1 = 0
                avg_dgz2 = 0
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            batch_losses += D_G_z2
            iters += 1
        print('Max losses: ' + str(max_batch_losses) + ', this batch losses: ' + str(batch_losses))
        if(batch_losses > max_batch_losses):
            max_batch_losses = batch_losses
            save_model(netG, os.path.join(args.save_GAN, 'netG_best.pth.tar'))

        save_model(netG, os.path.join(args.save_GAN, 'net_G_{}.pth.tar'.format(epoch)))



    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    #save_model(netG, os.path.join(args.save_dir, 'netG.pth.tar'))
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()