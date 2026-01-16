import os
import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from pyproj import Transformer
from tqdm import tqdm
from utils import *

# -----------------------------
# CONFIG
# -----------------------------
TEST_IMAGE_DIR = "task_cv_model/test_data/drone_images/"
SAT_MAP_PATH = "task_cv_model/map.tif"
OUTPUT_CSV = "outputs/test_output.csv"

TILE_SIZE = 512
MIN_MATCHES = 25
MIN_INLIERS = 8

# -----------------------------
# Load GeoTIFF metadata
# -----------------------------
with rasterio.open(SAT_MAP_PATH) as src:
    geo_transform = src.transform
    src_crs = src.crs
    sat_img = src.read()[:3].transpose(1, 2, 0)

# -----------------------------
# CRS transformer
# -----------------------------
transformer = Transformer.from_crs(
    src_crs, "EPSG:4326", always_xy=True
)

sat_gray = cv2.cvtColor(sat_img, cv2.COLOR_RGB2GRAY)
sat_tiles = tile_image(sat_gray)

# Precompute SIFT features for satellite tiles
sat_features = []
for t in sat_tiles:
    kp, des = extract_sift(t["tile"])
    if des is not None:
        sat_features.append({
            "kp": kp,
            "des": des,
            "offset": t["offset"]
        })

# -----------------------------
# Test inference loop
# -----------------------------
results = []

drone_imgs = load_drone_images(TEST_IMAGE_DIR)

# Preprocess drone images
drone_imgs_proc = preprocess_drone_images(drone_imgs)

test_images = sorted(os.listdir(TEST_IMAGE_DIR))

# for drone_gray in tqdm(drone_imgs_proc, desc="Processing test images"):
for name, drone_gray in tqdm(drone_imgs_proc.items(),
                             desc="Processing drone images",
                             total=len(drone_imgs_proc)):
    # img_path = os.path.join(TEST_IMAGE_DIR, img_name)

    # img = cv2.imread(img_path)
    # if img is None:
    #     continue

    # drone_gray = preprocess_drone_images(img)
    # try:
    kp_d, des_d = extract_sift(drone_gray)
    #     print('exrtact sift success')
    # except: print('error in SIFT extraction:', drone_gray.shape)

    if des_d is None:
        continue

    best = {"inliers": 0}

    for tile in sat_features:
        matches = match_keypoints(des_d, tile["des"])
        if len(matches) < MIN_MATCHES:
            continue

        H, inliers = ransac_homography(
            kp_d, tile["kp"], matches
        )

        if H is not None and inliers > best["inliers"]:
            best = {
                "H": H,
                "offset": tile["offset"],
                "inliers": inliers
            }

    if best["inliers"] < MIN_INLIERS:
        continue

    # -----------------------------
    # Drone center → satellite pixel
    # -----------------------------
    h, w = drone_gray.shape
    center = np.array([[w/2, h/2, 1]]).T
    pt = best["H"] @ center
    pt /= pt[2]

    x_sat = pt[0, 0] + best["offset"][0]
    y_sat = pt[1, 0] + best["offset"][1]

    lat, lon = sat_pixel_to_gps(x_sat, y_sat)

    # Timestamp from image name
    timestamp = float(os.path.splitext(img_name)[0])

    results.append({
        "timestamp": timestamp,
        "latitude": lat,
        "longitude": lon,
        "altitude": np.nan
    })

# -----------------------------
# Save test_output.csv
# -----------------------------
test_output = pd.DataFrame(results)
test_output = test_output[
    ["timestamp", "latitude", "longitude", "altitude"]
]

test_output.to_csv(OUTPUT_CSV, index=False)

print(f"Saved {len(test_output)} predictions to {OUTPUT_CSV}")
