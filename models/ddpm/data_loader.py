import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import path2arr
import glob
import numpy as np

class ImagePreprocess(Dataset):
    def __init__( self, 
                  data_dir,
                  img_size=128,
                  hole_ratio=0.2,
                ):

        self.x_paths = sorted(glob.glob(f'{data_dir}/*jpg'))
        self.img_size = img_size
        self.hole_ratio = hole_ratio

    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):

        # img preprocess
        x = path2arr(self.x_paths[idx], img_size=self.img_size)
        x = x.astype(np.float16) / 127.5 - 1.
        #x = fill_hole(x, hole_ratio=self.hole_ratio)
        
        # to tensor
        x = torch.tensor(x, dtype=torch.float16).unsqueeze(0)
        return x

    def get_loader(self, batch_size=8, shuffle=True):
        return  DataLoader(self, batch_size=batch_size, shuffle=shuffle)

def main():
    
    pp = WheelPreprocess('../db_pattern/DB/wheel_org/')
    loader = pp.get_loader()

    for x in loader:
        print(x.shape)
        print('x sample: ', x[0,:,20,:])
        break
        
if __name__ == "__main__":
    main()