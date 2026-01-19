# data/loaders.py

import os
import numpy as np
from torch.utils.data import DataLoader
from data.dataset import DroneSatDataset


def get_dataloaders(
    dataset_root,
    batch_size=8,
    val_ratio=0.2,
    num_workers=4
):
    """
    Returns train and validation dataloaders for drone-satellite registration.
    """
    image_dir = os.path.join(dataset_root, "task_cv_model/train_data/drone_images")
    gt_csv = os.path.join(dataset_root, "task_cv_model/train_data/ground_truth.csv")

    all_images = [f for f in sorted(os.listdir(image_dir)) if f.endswith(('.png', '.jpg', '.jpeg'))]
    n = len(all_images)
    indices = np.arange(n)
    np.random.shuffle(indices)
    split = int((1 - val_ratio) * n)
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = DroneSatDataset(image_dir, gt_csv, train_idx)
    val_ds = DroneSatDataset(image_dir, gt_csv, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=1,
        shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader

# SuperPoint-based alignment function
def align_drone_with_satellite(drone_img, satellite_img, superpoint_model, superglue_model, geo_transform, src_crs):
    """
    Align a drone image with a satellite image using SuperPoint and SuperGlue.
    Returns the estimated GPS coordinates (lat, lon) for the drone image center.
    """
    # 1. Extract features from both images using SuperPoint
    # 2. Match features using SuperGlue
    # 3. Estimate homography or transformation
    # 4. Map drone image center to satellite pixel, then to GPS using geo_transform and src_crs
    # (This is a stub; actual implementation will require model inference code)
    # Example return: (latitude, longitude)
    return None, None
