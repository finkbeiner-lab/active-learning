from PIL import Image
import torch 
from torchvision.datasets import VisionDataset
import numpy as np
import pdb

class AmyBDataset(VisionDataset):
    def __init__(self, X, Y, transforms=None):      
        self.X = X
        self.Y = Y
        self.transforms = transforms
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = Image.open(self.X[idx]).convert("RGB")
       
        mask = Image.open(self.Y[idx]).convert('P')
        # image = Image.open("/media/vivek/Samsung_T5/vivek/CS8K/video01/video01_00080/frame_80_endo.png")
        # mask = Image.open("/media/vivek/Samsung_T5/vivek/CS8K/video01/video01_00080/frame_80_endo_mask.png")

        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        if self.transforms:
            image, mask = self.transforms(image, mask)
        
        C = 4 # num semantic classes for AmyB
        
        masks = torch.zeros((C, image.shape[1], image.shape[2]))
        for c in range(0, C):
            # masks[c, :, :] = torch.where(mask == c, 1, 0)[:, :, 0].float()
            masks[c, :, :] = torch.where(mask == 0, 1, 0).float()


        return image, masks


