# data/dataset.py
import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

def extract_timestamp(filename):
    # "1445.599225344.png" → 1445.599225344
    return float(os.path.splitext(filename)[0])


def match_gps_timestamp(img_ts, gps_timestamps):
    """
    Find index of closest GPS timestamp
    """
    idx = np.argmin(np.abs(gps_timestamps - img_ts))
    return idx

# data/dataset.py (continued)
class DroneSatDataset(Dataset):
    def __init__(self, image_dir, gt_csv, indices):
        self.image_dir = image_dir
        self.images = sorted(os.listdir(image_dir))

        self.gt = pd.read_csv(gt_csv)
        self.gps_ts = self.gt["timestamp"].values

        self.indices = indices

        self.tf = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)

        img_ts = extract_timestamp(img_name)
        gps_idx = match_gps_timestamp(img_ts, self.gps_ts)

        row = self.gt.iloc[gps_idx]
        lat = row["latitude"]
        lon = row["longitude"]

        return img, lat, lon, img_name
