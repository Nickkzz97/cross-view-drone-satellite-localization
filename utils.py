import torch
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import rasterio
from pyproj import Transformer
from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from SuperGluePretrainedNetwork.models.superglue import SuperGlue

# Initialize SuperPoint and SuperGlue models
sp_config = {
    "descriptor_dim": 256,
    "nms_radius": 4,
    "keypoint_threshold": 0.001,
    "max_keypoints": 2048,
    "remove_borders": 4
}
sg_config = {
    "weights": "outdoor",
    "sinkhorn_iterations": 20,
    "match_threshold": 0.2
}
device = "cuda" if torch.cuda.is_available() else "cpu"
print('device:', device)
superpoint = SuperPoint(sp_config).to(device).eval()
superglue = SuperGlue(sg_config).to(device).eval()

def align_timestamp(img_ts, gt_df):
    idx = np.argmin(np.abs(gt_df["timestamp"].values - img_ts))
    return gt_df.iloc[idx]

def extract_superpoint(img, superpoint=superpoint):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inp = torch.from_numpy(gray/255.).float()[None,None].to(device)

    with torch.no_grad():
        out = superpoint({"image": inp})

    return (
        out["keypoints"][0],
        out["descriptors"][0],
        out["scores"][0]
    )

def superglue_match(img1, img2):
    k1, d1, s1 = extract_superpoint(img1)
    k2, d2, s2 = extract_superpoint(img2)

    data = {
        "keypoints0": k1[None],
        "keypoints1": k2[None],
        "descriptors0": d1[None],
        "descriptors1": d2[None],
        "scores0": s1[None],
        "scores1": s2[None],
        "image0": torch.empty((1,1,*img1.shape[:2]), device=device),
        "image1": torch.empty((1,1,*img2.shape[:2]), device=device),
    }

    with torch.no_grad():
        pred = superglue(data)

    matches = pred["matches0"][0].cpu().numpy()
    valid = matches > -1

    return (
        k1[valid].cpu().numpy(),
        k2[matches[valid]].cpu().numpy()
    )

def visualize_matches(img1, pts1, img2, pts2, max_lines=50):
    """
    img1, img2 : BGR images
    pts1, pts2 : (N,2) matched points
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2

    n = min(len(pts1), max_lines)

    for i in range(n):
        p1 = tuple(pts1[i].astype(int))
        p2 = (int(pts2[i][0] + w1), int(pts2[i][1]))

        color = tuple(np.random.randint(0,255,3).tolist())
        cv2.circle(canvas, p1, 3, color, -1)
        cv2.circle(canvas, p2, 3, color, -1)
        cv2.line(canvas, p1, p2, color, 1)

    return canvas

def is_homography_valid(H, img_shape):
    h, w = img_shape[:2]

    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1]
    ]).T

    warped = H @ corners
    warped /= warped[2]

    xs = warped[0]
    ys = warped[1]

    # Reject if extreme distortion
    if np.any(np.isnan(xs)) or np.any(np.isnan(ys)):
        return False

    area = cv2.contourArea(
        np.vstack([xs, ys]).T.astype(np.float32)
    )

    if area < 0.05 * (w * h):
        return False

    if area > 10 * (w * h):
        return False

    return True

def estimate_best_satellite_patch(
    drone_img, sat_img,
    window=512, stride=256
):
    best = None
    best_inliers = 0
    drone_img = cv2.resize(drone_img, (window, window))
    for y in range(0, sat_img.shape[0]-window, stride):
        for x in range(0, sat_img.shape[1]-window, stride):
            patch = sat_img[y:y+window, x:x+window]

            mk0, mk1 = superglue_match(drone_img, patch)
            if len(mk0) < 12:
                continue

            H, mask = cv2.findHomography(mk0, mk1, cv2.RANSAC, 5.0)
            # H, mask = cv2.findHomography(mk0, mk1, cv2.RANSAC, 5.0)
            if H is None:
                continue

            if not is_homography_valid(H, drone_img.shape):
                continue

            inliers = int(mask.sum())
            mask = mask.ravel().astype(bool)
            if inliers > best_inliers:
                best_inliers = inliers
                best = {
                    "H": H,
                    "x": x,
                    "y": y,
                    "mk0":mk0,
                    "mk1":mk1,
                    "patch": patch,
                    "inlier_mask":mask,
                    "inliers": inliers
                }

    return best

def save_best_inlier_npz(best, save_dir, image_id):
    os.makedirs(save_dir, exist_ok=True)

    np.savez(
        os.path.join(save_dir, f"{image_id}_best_inlier.npz"),
        H=best["H"],
        x=best["x"],
        y=best["y"],
        patch=best["patch"],
        mk0=best["mk0"],
        mk1=best["mk1"],
        inliersk=best["inliers"],
        # num_inliers=best["num_inliers"]
    )

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def pixel_to_latlon(px, py, transform, transformer):
    # Pixel → projected coordinates
    x_proj, y_proj = rasterio.transform.xy(
        transform, py, px, offset="center"
    )

    # Projected → lat/lon
    lon, lat = transformer.transform(x_proj, y_proj)
    return lat, lon

def estimate_gps_from_inliers(best, geo_transform, transformer):
    """
    Estimate GPS using centroid of satellite inlier points
    """
    mask = best["inlier_mask"]

    sat_pts = best["mk1"][mask]

    sat_pts = np.asarray(sat_pts)

    if sat_pts.ndim == 3:
        sat_pts = sat_pts.squeeze()

    if sat_pts.ndim != 2 or sat_pts.shape[1] != 2:
        print("Invalid sat_pts shape:", sat_pts.shape)
        return None

    if len(sat_pts) < 3:
        return None

    px = sat_pts[:, 0].mean() + best["x"]
    py = sat_pts[:, 1].mean() + best["y"]

    return pixel_to_latlon(px, py, geo_transform, transformer)

def visualize_gps_on_satellite(i,
    sat_img, px, py, patch_info=None, title="GPS estimate"
):
    plt.figure(figsize=(8,8))
    plt.imshow(sat_img)
    plt.scatter(px, py, c="red", s=60, marker="x", label="Estimated GPS")

    if patch_info is not None:
        x, y, w = patch_info
        rect = plt.Rectangle(
            (x, y), w, w,
            edgecolor="lime", facecolor="none", linewidth=2,
            label="Matched patch"
        )
        plt.gca().add_patch(rect)

    plt.legend()
    plt.title(title+f'_i')
    plt.axis("off")
    plt.show()

def save_gps_on_satellite(i,
    sat_img, px, py, best, out_dir
):
    ensure_dir(out_dir)

    vis = sat_img.copy()

    # GPS point
    cv2.drawMarker(
        vis,
        (int(px), int(py)),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=100,
        thickness=10
    )

    # Patch bounding box
    x, y = best["x"], best["y"]
    h, w = best["patch"].shape[:2]
    cv2.rectangle(
        vis,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        20
    )

    cv2.imwrite(os.path.join(out_dir, f"gps_on_satellite_{i}.png"), vis)

def align_drone_to_satellite(drone_img, sat_img, path, geo_transform, i,vis = False):
    window = 512
    drone_img = cv2.resize(drone_img, (window, window))
    best = estimate_best_satellite_patch(drone_img, sat_img)
    image_id = os.path.splitext(os.path.basename(path))[0]
    save_best_inlier_npz(best, "output_dir", image_id)
    if best is None:
        return None

    h, w = drone_img.shape[:2]
    center = np.array([[w/2, h/2, 1]]).T
    mapped = best["H"] @ center
    mapped /= mapped[2]

    px = mapped[0] + best["x"]
    py = mapped[1] + best["y"]
    # print("mk1 shape:", best["mk1"].shape)
    # print("mask shape:", best["inlier_mask"].shape)
    # Inlier satellite points (pixel space)
    sat_inliers = best["mk1"][best["inlier_mask"]]

    px = sat_inliers[:,0].mean() + best["x"]
    py = sat_inliers[:,1].mean() + best["y"]

    # px = sat_pts[:,0].mean() + best["x"]
    # py = sat_pts[:,1].mean() + best["y"]

    if vis: visualize_gps_on_satellite(i,
        sat_img,
        px, py,
        patch_info=(best["x"], best["y"], best["patch"].shape[0]),
        title="Estimated GPS location on satellite"
    )
    out_dir = 'output'
    save_gps_on_satellite(i,sat_img, px, py, best, out_dir)
    lat, lon  = estimate_gps_from_inliers(
        best,
        geo_transform,
        to_wgs84
    )
    # return gps
    # lat, lon = estimate_gps_from_inliers
    # pixel_to_latlon(
    # px, py,
    # geo_transform,
    # to_wgs84
    # )
    return lat, lon


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = radians(lat2-lat1)
    dlon = radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*atan2(sqrt(a), sqrt(1-a))
