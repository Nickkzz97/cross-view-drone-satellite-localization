import rasterio
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from utils import *

# Load satellite map
sat_img, sat_transform, sat_crs = load_satellite_map("task_cv_model/map.tif")

# Load drone images
drone_imgs = load_drone_images("task_cv_model/train_data/drone_images/")

# Preprocess drone images
drone_imgs_proc = preprocess_drone_images(drone_imgs)

sat_gray = satellite_to_gray(sat_img)
sat_gray = normalize_uint8(sat_gray)

sat_tiles = tile_image(sat_gray)
vis = cv2.cvtColor(sat_gray, cv2.COLOR_GRAY2RGB)

for t in sat_tiles[::50]:
    x, y = t["offset"]
    cv2.rectangle(vis, (x,y), (x+512,y+512), (255,0,0), 8)


sat_features = []

for t in sat_tiles:
    kp, des = extract_sift(t["tile"])
    if des is not None:
        sat_features.append({
            "kp": kp,
            "des": des,
            "offset": t["offset"],
            "tile": t["tile"]
        })
# name = list(drone_imgs.keys())[200]
# sample_tile = sat_features[0]["tile"]
# kp, des = extract_sift(sample_tile)
# kp_d, des_d = extract_sift(drone_imgs_proc[name])

# kp_img = cv2.drawKeypoints(
#     sample_tile, kp, None,
#     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# )
# kp_drone = cv2.drawKeypoints(
#     drone_imgs_proc[name], kp_d, None,
#     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# )  

# print("Satellite Keypoints:", len(kp))
# print("Drone Keypoints:", len(kp_d))

# tile = sat_features[0]
# matches, good_matches = match_keypoints(des_d, tile["des"])
# match_plot = cv2.drawMatches(drone_imgs_proc[name], kp_d, tile["tile"], tile["kp"], good_matches[:100], None, flags=2)

# H, inliers = ransac_homography(kp_d, tile["kp"], good_matches)
# print("RANSAC Inliers:", inliers)
# print('matches, good matches, inliers:', len(matches), len(good_matches), inliers)

results = {}

for name, drone_gray in tqdm(drone_imgs_proc.items(),
                             desc="Processing drone images",
                             total=len(drone_imgs_proc)):

    kp_d, des_d = extract_sift(drone_gray)

    if des_d is None:
        continue

    best = {"inliers": 0}

    for tile in sat_features:
        matches, good_matches = match_keypoints(des_d, tile["des"])
        if len(matches) < 25:
            continue

        H, inliers = ransac_homography(kp_d, tile["kp"], good_matches)

        if inliers > best["inliers"]:
            best = {
                "offset": tile["offset"],
                "H": H,
                "inliers": inliers
            }
    # print(f"{name}: Best inliers = {best['inliers']}")
    results[name] = best

rows = []

for name, res in results.items():
    if res["inliers"] == 0 or res["H"] is None:
        continue

    rows.append({
        "image_name": name,
        "tile_x": res["offset"][0],
        "tile_y": res["offset"][1],
        "inliers": res["inliers"]
    })

df = pd.DataFrame(rows)
os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/best_matches.csv", index=False)
# H_dict = {
#     name: res["H"]
#     for name, res in results.items()
#     if res["H"] is not None
# }

# np.save("outputs/homographies.npy", H_dict)

# os.makedirs("outputs", exist_ok=True)

H_store = {}

for name, res in results.items():
    if res["H"] is None:
        continue

    H_store[name] = {
        "H": res["H"],
        "offset": res["offset"],
        "inliers": res["inliers"]
    }

np.save("outputs/best_homographies.npy", H_store)