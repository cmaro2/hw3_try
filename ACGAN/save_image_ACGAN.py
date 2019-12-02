import torch
import torchvision.utils as vutils
from ACGAN.Gen_ACGAN import Generator
import os
import numpy as np
from torch.autograd import Variable
from PIL import Image


#t = torch.load('seed.pt')
device = torch.device('cuda')
nz = 10

if(1):
    a = np.zeros(10)
    a.fill(1)
    b = np.zeros(10)
    noise = torch.FloatTensor(10, nz, 1, 1).cuda()
    noise = Variable(noise)
    noise2 = torch.FloatTensor(10, nz, 1, 1).cuda()
    noise2 = Variable(noise2)

    noise_x = np.random.normal(0, 1, (10, nz))

    noise_ = noise_x
    class_onehot = np.zeros((10, 2))
    class_onehot[np.arange(10), a.astype(int)] = 1
    noise_[np.arange(10), :2] = class_onehot[np.arange(10)]
    noise_ = (torch.from_numpy(noise_))
    noise.data.copy_(noise_.view(10, nz, 1, 1))

    noise_ = noise_x
    class_onehot = np.zeros((10, 2))
    class_onehot[np.arange(10), b.astype(int)] = 1
    noise_[np.arange(10), :2] = class_onehot[np.arange(10)]
    noise_ = (torch.from_numpy(noise_))
    noise2.data.copy_(noise_.view(10, nz, 1, 1))

    noise_A = torch.cat((noise, noise2),0)

    torch.save(noise_A, 'seed_ACGAN.pt')

else:
    noise_A = torch.load('seed_ACGAN.pt')


netG = Generator(nz=nz).cuda()
#plt.figure()
#plt.subplot(1, 2, 2)
#plt.axis("off")

for i in range(50):
    model_std = torch.load(os.path.join('./log/net_G_{}.pth.tar'.format(i)))
    netG.load_state_dict(model_std)
    img_list = []
    with torch.no_grad():
        fake = netG(noise_A).detach().cpu()
    img_list.append(vutils.make_grid(fake, padding=2, normalize=True, nrow=10))
    img_only = np.transpose(img_list[-1], (1, 2, 0)).numpy()
    img_only = (img_only *255)
    img_array = np.array(img_only, dtype=np.uint8)
    result = Image.fromarray(img_array)
    result.save('images/' + 'image_2_{}.png'.format(i))

