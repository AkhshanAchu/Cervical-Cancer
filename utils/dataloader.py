import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from tools.helper import *

class CellSegDataset(Dataset):
    def __init__(self, samples, out_size=256):
        """
        samples: list of (img_path, cyt_path, nuc_path)
        """
        self.samples = samples
        self.out_size = out_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cyt_path, nuc_path = self.samples[idx]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        h, w, _ = img.shape

        cyt_poly = load_polygon(cyt_path)
        nuc_poly = load_polygon(nuc_path)

        cyt_mask = polygon_to_mask(cyt_poly, (h, w))
        nuc_mask = polygon_to_mask(nuc_poly, (h, w))

        img, cyt_mask, nuc_mask = crop_from_polygon(
            img, cyt_mask, nuc_mask, cyt_poly, margin=32
        )

        img = cv2.resize(img, (self.out_size, self.out_size))
        cyt_mask = cv2.resize(cyt_mask, (self.out_size, self.out_size), interpolation=cv2.INTER_NEAREST)
        nuc_mask = cv2.resize(nuc_mask, (self.out_size, self.out_size), interpolation=cv2.INTER_NEAREST)

        img = torch.from_numpy(img).permute(2, 0, 1)
        cyt_mask = torch.from_numpy(cyt_mask).unsqueeze(0).float()
        nuc_mask = torch.from_numpy(nuc_mask).unsqueeze(0).float()

        return img, cyt_mask, nuc_mask
