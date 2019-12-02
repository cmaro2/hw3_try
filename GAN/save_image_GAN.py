import torch
from parser_2 import arg_parse
import torchvision.utils as vutils
from GAN.Gen_GAN import Generator
import os
import numpy as np
from PIL import Image


device = torch.device('cuda')

#t = torch.randn(32, 100, 1, 1, device=device)
#torch.save(t, 'seed_GAN.pt')
t = torch.load('seed_GAN.pt')

args = arg_parse()

netG = Generator(args.ngpu).cuda()

#plt.figure()
#plt.subplot(1, 2, 2)
#plt.axis("off")

for i in range(53):
    model_std = torch.load(os.path.join(args.save_GAN, 'net_G_{}.pth.tar'.format(i)))
    netG.load_state_dict(model_std)
    img_list = []
    with torch.no_grad():
        fake = netG(t).detach().cpu()
    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
    img_only = np.transpose(img_list[-1], (1, 2, 0)).numpy()
    img_only = (img_only *255)
    img_array = np.array(img_only, dtype=np.uint8)
    result = Image.fromarray(img_array)
    result.save('images/' + 'image_1_{}.png'.format(i))
    #plt.title("Fake Images from model: " + str(i))
    #plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    #plt.show()

