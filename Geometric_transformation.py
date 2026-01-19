import numpy as np
import rasterio
from rasterio.transform import xy
import pandas as pd

from utils import *

H_store = np.load("outputs/best_homographies.npy", allow_pickle=True).item()
# with rasterio.open("task_cv_model/map.tif") as src:
#     print(src.crs)

# x_proj, y_proj = xy(transform, y, x)
# with rasterio.open("task_cv_model/map.tif") as src:
#     geo_transform = src.transform 

with rasterio.open("task_cv_model/map.tif") as src:
    geo_transform = src.transform
    src_crs = src.crs

# Load drone images
drone_imgs = load_drone_images("task_cv_model/train_data/drone_images/")

# Preprocess drone images
drone_imgs_proc = preprocess_drone_images(drone_imgs)

geo_results = []

for name in H_store.keys():
    res = geoposition_drone_image(
        name,
        H_store,
        drone_imgs_proc,
        geo_transform, src_crs
    )
    if res is not None:
        geo_results.append(res)

df_geo = pd.DataFrame(geo_results)
df_geo.to_csv("outputs/predicted_gps.csv", index=False)

print('Sanity Check.....')
print(df_geo[["latitude", "longitude"]].head())
assert df_geo["latitude"].between(-90, 90).all()
assert df_geo["longitude"].between(-180, 180).all()
print(df_geo[["latitude", "longitude"]].head())