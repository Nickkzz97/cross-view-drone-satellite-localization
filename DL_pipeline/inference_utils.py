# inference_utils.py
import torch
from geometry.warp import theta_to_H
from geometry.geo_utils import SatelliteAligner

@torch.no_grad()
def infer(
    drone_img,
    backbone,
    hnet,
    geo_aligner,
    device="cuda"
):
    """
    drone_img: [1, 3, H, W] tensor
    returns: (latitude, longitude)
    """

    drone_img = drone_img.to(device)

    # Feature extraction
    fd = backbone(drone_img)

    # In a practical system, fs comes from cached satellite features
    # For now, we use the same feature tensor (placeholder, consistent with training)
    fs = fd

    # Predict homography
    theta = hnet(fs, fd)
    H = theta_to_H(theta)

    # Take center pixel after alignment
    _, _, Hh, Wh = drone_img.shape
    px, py = Wh // 2, Hh // 2

    # Convert pixel → GPS
    lat, lon = geo_aligner.pixel_to_gps(px, py)

    return lat, lon
