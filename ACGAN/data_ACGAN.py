import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

#MEAN=[0.485, 0.456, 0.406]
#STD=[0.229, 0.224, 0.225]

MEAN=[0.5, 0.5, 0.5]
STD=[0.5, 0.5, 0.5]

class DATA(Dataset):
    def __init__(self, args):

        ''' set up basic parameters for dataset '''
        self.dir = '../hw3_data/face/train'
        self.data_dir = os.listdir(self.dir)
        self.img_dir = [self.dir + '/' + photo for photo in self.data_dir]

        df = pd.read_csv('../hw3_data/face/train.csv', usecols=['Smiling'])

        self.smile = df.values

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])# (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB



    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):

        #get data
        img_path = self.img_dir[idx]
        #read image
        img = Image.open(img_path).convert('RGB')
        #get smiling value
        sml = self.smile[idx]

        return self.transform(img), sml