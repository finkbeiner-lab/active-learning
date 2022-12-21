import os
import glob
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from datamodules.cs8k_dataset import CS8KDataset
from datamodules.amyb_dataset import AmyBDataset
import datamodules.transforms as T
import pdb


class AmyBDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.root = '/mnt/new-nas/work/data/npsad_data/vivek/Datasets/amyb_wsi'
        
        # Augmentations for train data
        self.train_trans = T.Compose([
            T.PILToTensor(),
        ])
        
        # Only resizing for validation and test augmentations
        self.val_trans = T.Compose([
            T.PILToTensor(),
        ])
        self.test_trans = self.val_trans
        
    def prepare_data(self):
        self.X_tr = np.asarray(glob.glob(f'{self.root}/train/images/**/*.png', recursive=True))
        self.Y_tr = np.asarray(glob.glob(f'{self.root}/train/labels/**/*.png', recursive=True))
        
        self.X_val = np.asarray(glob.glob(f'{self.root}/val/images/**/*.png', recursive=True))
        self.Y_val = np.asarray(glob.glob(f'{self.root}/val/labels/**/*.png', recursive=True))
        
        self.X_te = np.asarray(glob.glob(f'{self.root}/test/images/**/*.png', recursive=True))
        self.Y_te = np.asarray(glob.glob(f'{self.root}/test/labels/**/*.png', recursive=True))
        
        self.X_val = self.X_val[0:self.args.val_size]
        self.Y_val = self.Y_val[0:self.args.val_size]
        
        self.X_te = self.X_te[0:self.args.test_size]
        self.Y_te = self.Y_te[0:self.args.test_size]
        
    def setup(self, stage=None):
        self.all = AmyBDataset(self.X_tr, self.Y_tr, self.train_trans) # train + unlb
        self.val = AmyBDataset(self.X_val, self.Y_val, self.val_trans)
        self.test = AmyBDataset(self.X_te, self.Y_te, self.test_trans)
        
    def update_pool(self, inc_idxs):
        # Train dataset should only include inputs for current active learning round
        self.lb = AmyBDataset(self.X_tr[inc_idxs], self.Y_tr[inc_idxs], self.train_trans)
        
        # Unlabeled used for querying samples for next round
        # exc_idxs = np.full(len(self.X_tr), False, dtype=bool)
        # exc_idxs[inc_idxs] = True
        exc_idxs = ~inc_idxs

        self.unlb = AmyBDataset(self.X_tr[exc_idxs], self.Y_tr[exc_idxs], self.train_trans)   

    def train_dataloader(self):
        return DataLoader(self.lb, batch_size=self.args.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=1, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=1, shuffle=False)
    
    def unlb_dataloader(self):
        return DataLoader(self.unlb, batch_size=1, shuffle=False)
    
    def all_dataloader(self):
        return DataLoader(self.all, batch_size=1, shuffle=False)
    
    def compute_lb_dist(self):
        dist = torch.zeros(4)
        lb_dataloader = self.train_dataloader()
        
        for i, (x, y) in enumerate(lb_dataloader):
            for j in range(4):
                dist[j] += y[:, j, :, :].sum()
            
        return dist
    
    