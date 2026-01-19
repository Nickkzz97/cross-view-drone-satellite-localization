import os
import cv2
import torch
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
from sklearn.model_selection import train_test_split
from pyproj import Transformer
from tqdm import tqdm
from pathlib import Path
import time
from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from SuperGluePretrainedNetwork.models.superglue import SuperGlue
from utils import *


with rasterio.open("task_cv_model/map.tif") as src:
    src_crs = src.crs
    sat_img = src.read([1,2,3]).transpose(1,2,0)
    geo_transform = src.transform

to_wgs84 = Transformer.from_crs(
    src_crs, "EPSG:4326", always_xy=True
)

gt = pd.read_csv("task_cv_model/train_data/ground_truth.csv")
gt = gt.sort_values("timestamp").reset_index(drop=True)

samples = []

for name in os.listdir("task_cv_model/train_data/drone_images"):
    ts = float(os.path.splitext(name)[0])
    gt_row = align_timestamp(ts, gt)

    samples.append({
        "path": os.path.join("task_cv_model/train_data/drone_images", name),
        "timestamp": ts,
        "lat": gt_row.latitude,
        "lon": gt_row.longitude
    })
 
def evaluate(dataset, desc="Evaluating", csv_path=None):
    errors = []
    success = 0
    times = []   # ⬅ store per-image time

    tp_5m = tp_20m = 0
    fn_5m = fn_20m = 0

    total = len(dataset)

    for i,sample in enumerate(tqdm(dataset, desc=desc, total=len(dataset))):
        drone = cv2.imread(sample["path"])
        t0 = time.time()
        if i %50 == 0: pred = align_drone_to_satellite(drone, sat_img, sample["path"], geo_transform, i, vis = True)
        pred = align_drone_to_satellite(drone, sat_img, sample["path"], geo_transform, i,vis = False)
        t1 = time.time()

        times.append(t1 - t0)
        basename = Path(sample["path"]).stem
        if pred is None:
            record = {
                "image": basename,
                "status": "failed",
                "pred_lat": None,
                "pred_lon": None,
                "gt_lat": sample["lat"],
                "gt_lon": sample["lon"],
                "error_m": None
            }
        else:

            print('dtype pred, gt:', type(pred[0]),type(pred[1]),type(sample["lat"]))
            pred_lat = float(pred[0])
            pred_lon = float(pred[1])
            gt_lat = float(sample["lat"])
            gt_lon = float(sample["lon"])
            err = haversine(
                pred_lat, pred_lon,
                gt_lat, gt_lon)
            errors.append(err)
            if err <= 5:
                tp_5m += 1
            else:
                fn_5m += 1

            if err <= 20:
                tp_20m += 1
            else:
                fn_20m += 1
            success += 1

            record = {
                "image": basename,
                "status": "success",
                "pred_lat": pred[0],
                "pred_lon": pred[1],
                "gt_lat": sample["lat"],
                "gt_lon": sample["lon"],
                "error_m": err
            }

            print(record)

        # 🔹 APPEND RECORD IMMEDIATELY
        if csv_path is not None:
            pd.DataFrame([record]).to_csv(
                csv_path,
                mode="a",
                header=False,
                index=False
            )

    errors = np.array(errors)
    times = np.array(times)

    # --- Statistics ---
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    rmse_error = np.sqrt(np.mean(errors ** 2))
    max_error = np.max(errors)

    err_std = np.std(errors)
    err_var = np.var(errors)

    recall_5m = 100 * tp_5m / total
    recall_20m = 100 * tp_20m / total

    # avg_inliers = np.mean(inlier_counts)

    avg_time = times.mean()
    # --- Print ---
    print(f"Error Std Dev : {err_std:.2f} m")
    print(f"Error Variance: {err_var:.2f} m^2")

    print(f"Recall @ 5m   : {recall_5m:.2f}%")
    print(f"Recall @ 20m  : {recall_20m:.2f}%")

    print(f"Mean error    : {mean_error:.2f} m")
    print(f"Median error  : {median_error:.2f} m")
    print(f"RMSE          : {rmse_error:.2f} m")
    print(f"Max error     : {max_error:.2f} m")

    # print(f"Average Inliers (proxy): {avg_inliers:.2f}")
    print(f"Average time per image : {avg_time:.3f} s")

    return {
        "mean": mean_error,
        "median": median_error,
        "rmse": rmse_error,
        "max": max_error,
        "recall_5m": recall_5m,
        "recall_20m": recall_20m,
        # "avg_inliers": avg_inliers,
        "avg_time_sec": avg_time,
        "num_samples": success
    }

if __name__ == "__main__":
    train_metrics = evaluate(
    samples,
    desc="Train set",
    csv_path="superpoint_prediction.csv"
)