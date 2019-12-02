import torch
import os
import torchvision.utils as vutils
import numpy as np
from PIL import Image

from parser_2 import arg_parse
from GAN.Gen_GAN import Generator as gen_GAN
from ACGAN.Gen_ACGAN import Generator as gen_ACGAN

if __name__ == '__main__':
    # Load parser
    args = arg_parse()

    # Load noises from files
    noise_GAN = torch.load('seed_GAN.pt')
    noise_ACGAN = torch.load('seed_ACGAN.pt')

    # Open models and load states
    GAN_G = gen_GAN(1).cuda()
    ACGAN_G = gen_ACGAN(10).cuda()

    GAN_G.load_state_dict(torch.load('model_GAN.pth.tar'))
    ACGAN_G.load_state_dict(torch.load('model_ACGAN.pth.tar'))

    img_GAN = []
    img_ACGAN = []

    # Use generators on noise
    with torch.no_grad():
        fake_GAN = GAN_G(noise_GAN).detach().cpu()
        fake_ACGAN = ACGAN_G(noise_ACGAN).detach().cpu()

    # Save GAN image
    print('Saving GAN image')
    img_GAN.append(vutils.make_grid(fake_GAN, padding=2, normalize=True, nrow=8))
    img_only = np.transpose(img_GAN[-1], (1, 2, 0)).numpy()
    img_only = (img_only *255)
    img_array = np.array(img_only, dtype=np.uint8)
    result = Image.fromarray(img_array)
    result.save(os.path.join(args.save_dir,'fig1_2.png'))

    # Save ACGAN image
    print('Saving ACGAN image')
    img_ACGAN.append(vutils.make_grid(fake_ACGAN, padding=2, normalize=True, nrow=10))
    img_only = np.transpose(img_ACGAN[-1], (1, 2, 0)).numpy()
    img_only = (img_only *255)
    img_array = np.array(img_only, dtype=np.uint8)
    result = Image.fromarray(img_array)
    result.save(os.path.join(args.save_dir,'fig2_2.png'))

    print('DONE')


