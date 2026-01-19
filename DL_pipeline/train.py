# train.py
import torch
from data.loaders import get_dataloaders
from models.backbone import Backbone
from models.homography_net import HomographyNet
from geometry.warp import theta_to_H
from geometry.geo_utils import SatelliteAligner, haversine
from math import radians, sin, cos, sqrt, atan2


train_loader, val_loader = get_dataloaders("dataset")

backbone = Backbone().cuda()
hnet = HomographyNet().cuda()

optimizer = torch.optim.AdamW(
    list(backbone.parameters()) + list(hnet.parameters()), lr=1e-4
)

geo = SatelliteAligner("/home/anand/Desktop/Nicky/job_prep/Assignment/Idle_robotics/DL_pipeline")

for epoch in range(50):
    for drone_img, gt_lat, gt_lon, _ in train_loader:
        drone_img = drone_img.cuda()

        fs = backbone(drone_img)  # satellite features assumed cached in practice
        fd = backbone(drone_img)

        theta = hnet(fs, fd)
        H = theta_to_H(theta)

        # center pixel after warp
        px, py = 128, 128
        pred_lat, pred_lon = geo.pixel_to_gps(px, py)

        loss = haversine(pred_lat, pred_lon, gt_lat.mean().item(), gt_lon.mean().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} | Loss (m): {loss.item():.2f}")
