# validate.py
import torch
from data.loaders import get_dataloaders
from models.backbone import Backbone
from models.homography_net import HomographyNet
from geometry.geo_utils import SatelliteAligner
from inference_utils import infer
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


_, val_loader = get_dataloaders("dataset")

device = "cuda" if torch.cuda.is_available() else "cpu"

backbone = Backbone().to(device).eval()
hnet = HomographyNet().to(device).eval()
geo = SatelliteAligner("dataset/satellite/map.tif")

errors = []
inliers = 0

with torch.no_grad():
    for img, gt_lat, gt_lon, _ in val_loader:
        img = img.to(device)

        pred_lat, pred_lon = infer(
            img,
            backbone,
            hnet,
            geo,
            device
        )

        err = haversine(
            pred_lat, pred_lon,
            gt_lat.item(), gt_lon.item()
        )

        errors.append(err)
        if err < 20:
            inliers += 1

print(f"Mean Error (m): {sum(errors) / len(errors):.2f}")
print(f"<20m Ratio: {inliers / len(errors):.3f}")
