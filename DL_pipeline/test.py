# test.py
import torch
from data.loaders import get_dataloaders
from models.backbone import Backbone
from models.homography_net import HomographyNet
from geometry.geo_utils import SatelliteAligner
from inference_utils import infer

device = "cuda" if torch.cuda.is_available() else "cpu"

_, _, test_loader = get_dataloaders("dataset")

backbone = Backbone().to(device).eval()
hnet = HomographyNet().to(device).eval()
geo = SatelliteAligner("dataset/satellite/map.tif")

with torch.no_grad():
    for img, name in test_loader:
        img = img.to(device)
        lat, lon = infer(img, backbone, hnet, geo, device)
        print(f"{name[0]}, {lat}, {lon}")
