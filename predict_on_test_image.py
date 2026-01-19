from tqdm import tqdm
import pandas as pd
import cv2
from pathlib import Path
import time
import rasterio
from pyproj import Transformer
from utils import *

with rasterio.open("task_cv_model/map.tif") as src:
    src_crs = src.crs
    sat_img = src.read([1,2,3]).transpose(1,2,0)
    geo_transform = src.transform

to_wgs84 = Transformer.from_crs(
    src_crs, "EPSG:4326", always_xy=True
)

def estimate_gps_testset(
    image_paths,
    csv_path="test_predictions.csv",
    desc="Estimating GPS (test set)"
):
    times = []

    for i, img_path in enumerate(tqdm(image_paths, desc=desc)):
        drone = cv2.imread(img_path)
        basename = Path(img_path).stem

        t0 = time.time()
        if i % 50 == 0:
            pred = align_drone_to_satellite(
                drone, sat_img, img_path, i, vis=True
            )
        else:
            pred = align_drone_to_satellite(
                drone, sat_img, img_path, i, vis=False
            )
        t1 = time.time()

        elapsed = t1 - t0
        times.append(elapsed)

        if pred is None:
            record = {
                "image": basename,
                "status": "failed",
                "pred_lat": None,
                "pred_lon": None,
                "time_sec": elapsed
            }
        else:
            pred_lat = float(pred[0])
            pred_lon = float(pred[1])

            record = {
                "image": basename,
                "status": "success",
                "pred_lat": pred_lat,
                "pred_lon": pred_lon,
                "time_sec": elapsed
            }

        # Save incrementally
        write_header = not Path(csv_path).exists()
        pd.DataFrame([record]).to_csv(
            csv_path,
            mode="a",
            header=write_header,
            index=False
        )

    avg_time = sum(times) / max(len(times), 1)
    print(f"Average time per image: {avg_time:.3f} s")

    return {
        "num_images": len(image_paths),
        "avg_time_sec": avg_time
    }

if __name__ == "__main__":
    test_image_dir = "task_cv_model/test_images"
    test_image_paths = list(Path(test_image_dir).glob("*.png"))

    results = estimate_gps_testset(
        [str(p) for p in test_image_paths],
        csv_path="output/test_predictions.csv",
        desc="Estimating GPS for test images"
    )

    print("Done. Summary:")
    print(results)
