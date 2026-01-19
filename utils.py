import cv2
import os
import rasterio
import numpy as np
import rasterio
import time
from rasterio.transform import xy
from pyproj import Transformer

def load_satellite_map(tif_path):
    dataset = rasterio.open(tif_path)
    
    # Read image (C, H, W) → (H, W, C)
    image = dataset.read()
    image = np.transpose(image, (1, 2, 0))
    
    transform = dataset.transform      # pixel ↔ geo transform
    crs = dataset.crs                  # coordinate reference system
    
    return image, transform, crs

def load_drone_images(folder):
    images = {}
    for fname in os.listdir(folder):
        if fname.endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            images[fname] = img
    return images

def preprocess_image(img, size=(512, 512)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resized = cv2.resize(gray, size)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def preprocess_drone_images(drone_images):
    processed = {}
    for name, img in drone_images.items():
        processed[name] = preprocess_image(img)
    return processed

def satellite_to_gray(sat_img):
    """
    sat_img: (H, W, C) GeoTIFF image
    """
    # If more than 3 bands (common in satellite images)
    if sat_img.shape[2] >= 3:
        rgb = sat_img[:, :, :3].astype(np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = sat_img[:, :, 0]

    return gray

def normalize_uint8(img):
    img = img.astype(np.float32)
    img = 255 * (img - img.min()) / (img.max() - img.min() + 1e-6)
    return img.astype(np.uint8)

# functions
def tile_image(img, tile_size=512, stride=256):
    tiles = []
    h, w = img.shape
    for y in range(0, h - tile_size, stride):
        for x in range(0, w - tile_size, stride):
            tiles.append({
                "tile": img[y:y+tile_size, x:x+tile_size],
                "offset": (x, y)
            })
    return tiles

# def extract_orb(img):
#     return orb.detectAndCompute(img, None)

# def match_features(des_d, des_s, ratio=0.75):
#     matches = bf.knnMatch(des_d, des_s, k=2)
#     return [m for m,n in matches if m.distance < ratio * n.distance]

sift = cv2.SIFT_create(nfeatures=4000)
def extract_sift(img):
    return sift.detectAndCompute(img, None)

def ratio_test(matches, ratio=0.75):
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

def match_keypoints(desc_1, desc_2, ratio=0.85):
    FLANN_INDEX_KDTREE = 1  

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    desc_1 = np.asarray(desc_1, np.float32)
    desc_2 = np.asarray(desc_2, np.float32)

    matches = flann.knnMatch(desc_1, desc_2, k=2)
    good = ratio_test(matches, ratio)
    return matches, good

def ransac_verify(kp_d, kp_s, matches):
    if len(matches) < 10:
        return None, 0

    src = np.float32([kp_d[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst = np.float32([kp_s[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return H, inliers

def ransac_homography(kp_d, kp_s, matches):
    if len(matches) < 10:
        return None, 0

    src = np.float32([kp_d[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst = np.float32([kp_s[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    H, mask = cv2.findHomography(
        src, dst,
        cv2.RANSAC,
        ransacReprojThreshold=6
    )

    inliers = int(mask.sum()) if mask is not None else 0
    
    if inliers<8:
        return None, inliers
    return H, mask.ravel(),inliers

# def ransac_homography(kp_d, kp_s, matches, thresh=6):
    # if len(matches) < 8:
    #     return None, None, 0

    # src = np.float32([kp_d[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    # dst = np.float32([kp_s[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    # H, mask = cv2.findHomography(src, dst, cv2.RANSAC, thresh)

    # if mask is None:
    #     return None, None, 0

    # inliers = int(mask.sum())
    # return H, mask.ravel(), inliers

def drone_center_to_sat_pixel(H, tile_offset, drone_shape):
    """
    H: 3x3 homography (drone → tile)
    tile_offset: (x0, y0)
    drone_shape: (H, W)
    """
    h, w = drone_shape

    # Drone center pixel (homogeneous)
    pt_drone = np.array([[w/2, h/2, 1]]).T

    # Map to satellite tile
    pt_tile = H @ pt_drone
    pt_tile /= pt_tile[2]

    x_tile, y_tile = pt_tile[0,0], pt_tile[1,0]

    # Map to full satellite image
    x_sat = x_tile + tile_offset[0]
    y_sat = y_tile + tile_offset[1]

    return x_sat, y_sat



def projected_to_latlon(x, y, src_crs):
    """
    x, y: projected coordinates (meters)
    src_crs: CRS from GeoTIFF
    """
    transformer = Transformer.from_crs(
        src_crs,
        "EPSG:4326",
        always_xy=True
    )

    lon, lat = transformer.transform(x, y)
    return lat, lon

def sat_pixel_to_gps(transform, x, y, src_crs):
    x_proj, y_proj = xy(transform, y, x)
    lat, lon = projected_to_latlon(x_proj, y_proj, src_crs)
    return lat, lon

def geoposition_drone_image(
    image_name,
    H_store,
    drone_imgs_proc,
    geo_transform, src_crs
):
    if image_name not in H_store:
        return None

    data = H_store[image_name]
    H = data["H"]
    offset = data["offset"]

    drone_img = drone_imgs_proc[image_name]
    h, w = drone_img.shape[:2]

    x_sat, y_sat = drone_center_to_sat_pixel(
        H, offset, (h, w)
    )

    # lat, lon = sat_pixel_to_gps(
    #     geo_transform, x_sat, y_sat
    # )

    lat, lon = sat_pixel_to_gps(
        geo_transform,
        x_sat,
        y_sat,
        src_crs
    )
    return {
        "image": image_name,
        "sat_x": x_sat,
        "sat_y": y_sat,
        "latitude": lat,
        "longitude": lon,
        "inliers": data["inliers"]
    }

#Compare different feature extractor methods

def match_orb(img1, img2):
    orb = cv2.ORB_create(nfeatures=3000)

    t0 = time.time()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good = ratio_test(matches, 0.75)
    inliers = ransac_verify(kp1, kp2, good)
    t = time.time() - t0

    return {
        "method": "ORB",
        "keypoints": (len(kp1), len(kp2)),
        "raw_matches": len(matches),
        "good_matches": len(good),
        "inliers": inliers,
        "time": t
    }

def match_sift(img1, img2):
    sift = cv2.SIFT_create(nfeatures=4000)

    t0 = time.time()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=50)
    )

    matches = flann.knnMatch(des1.astype(np.float32),
                             des2.astype(np.float32), k=2)

    good = ratio_test(matches, 0.85)
    inliers = ransac_verify(kp1, kp2, good)
    t = time.time() - t0

    return {
        "method": "SIFT",
        "keypoints": (len(kp1), len(kp2)),
        "raw_matches": len(matches),
        "good_matches": len(good),
        "inliers": inliers,
        "time": t
    }

def match_surf(img1, img2):
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)

    t0 = time.time()
    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=50)
    )

    matches = flann.knnMatch(des1, des2, k=2)
    good = ratio_test(matches, 0.8)
    inliers = ransac_verify(kp1, kp2, good)
    t = time.time() - t0

    return {
        "method": "SURF",
        "keypoints": (len(kp1), len(kp2)),
        "raw_matches": len(matches),
        "good_matches": len(good),
        "inliers": inliers,
        "time": t
    }

