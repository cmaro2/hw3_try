import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

#Set mean and STD values
MEAN=[0.5, 0.5, 0.5]
STD=[0.5, 0.5, 0.5]

class DATA(Dataset):
    def __init__(self, args):

        ''' set up basic parameters for dataset '''
        self.dir = '../hw3_data/face/train'
        self.data_dir = os.listdir(self.dir)
        self.img_dir = [self.dir + '/' + photo for photo in self.data_dir]

        # Read Smiling column of the csv file
        df = pd.read_csv('../hw3_data/face/train.csv', usecols=['Smiling'])
        self.smile = df.values

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):

        # get data
        img_path = self.img_dir[idx]
        # read image
        img = Image.open(img_path).convert('RGB')
        # get smiling value
        sml = self.smile[idx]

        return self.transform(img), sml