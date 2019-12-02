import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz= 10):
        super(Generator, self).__init__()
        self.nz = nz
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(nz, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        tconv2 = self.tconv2(input)
        tconv3 = self.tconv3(tconv2)
        tconv4 = self.tconv4(tconv3)
        tconv5 = self.tconv5(tconv4)
        tconv5 = self.tconv6(tconv5)

        return tconv5
