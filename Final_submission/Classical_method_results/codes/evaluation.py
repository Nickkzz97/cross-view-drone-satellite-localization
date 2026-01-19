import os
import numpy as np
import pandas as pd

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute Haversine distance in meters.
    """
    R = 6371000.0  # Earth radius in meters

    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

def image_to_timestamp(image_name):
    # Example: "1709123456.png" → 1709123456
    return float(os.path.splitext(image_name)[0])

# pred["timestamp"] = pred["image"].apply(
#     lambda x: float(os.path.splitext(x)[0])
# )


pred = pd.read_csv("outputs/predicted_gps.csv")
gt = pd.read_csv("task_cv_model/train_data/ground_truth.csv")

pred["timestamp"] = pred["image"].apply(image_to_timestamp)
gt["timestamp"] = gt["timestamp"].astype(float)

total_images = len(pred['image'])

pred = pred.sort_values("timestamp").reset_index(drop=True)
gt = gt.sort_values("timestamp").reset_index(drop=True)

df = pd.merge_asof(
    pred,
    gt,
    on="timestamp",
    direction="nearest",
    tolerance=5.0   # seconds (adjust if needed)
)

# print(f"Number of predictions {pred.iloc[470:480][['image', 'timestamp']]} Gt {gt['timestamp'][200:210],}:", len(pred),len(gt),  pred.keys(), gt.keys())
print("Total predictions:", len(pred), df.keys(), gt.keys())
print("Matched rows     :", df["latitude_y"].notna().sum())
# df = pred.merge(
#     gt,
#     on="image",
#     suffixes=("_pred", "_gt")
# ) 
# 79  1468.127342304.png  1468.127342304 Gt (450    1513.568992
# 451    1513.828804
# 452    1514.089399

df["error_m"] = haversine_distance(
    df["latitude_x"],
    df["longitude_x"],
    df["latitude_y"],
    df["longitude_y"]
)
df.to_csv("outputs/localization_errors.csv", index=False)
mean_error = df["error_m"].mean()
median_error = df["error_m"].median()
rmse_error = np.sqrt(np.mean(df["error_m"]**2))
max_error = df["error_m"].max()

registered = df["latitude_x"].notna().sum()

registration_success_rate = (registered / len(pred)) * 100
print(f"Registration Success Rate: {registration_success_rate:.2f}%")

recall_5m = (df["error_m"] <= 5).mean() * 100
recall_20m = (df["error_m"] <= 20).mean() * 100
registered_df = df.copy()  # df already has only registered rows

precision_5m = (registered_df["error_m"] <= 5).mean() * 100
precision_20m = (registered_df["error_m"] <= 20).mean() * 100
avg_inliers = df["inliers"].mean()

err_std = registered_df["error_m"].std()
err_var = registered_df["error_m"].var()

print(f"Error Std Dev : {err_std:.2f} m")
print(f"Error Variance: {err_var:.2f} m^2")

print(f"Recall @ 5m : {recall_5m:.2f}%")
print(f"Recall @ 20m: {recall_20m:.2f}%")
print(f"Precision @ 5m : {precision_5m:.2f}%")
print(f"Precision @ 20m: {precision_20m:.2f}%")
print(f"Mean error   : {mean_error:.2f} m")
print(f"Median error : {median_error:.2f} m")
print(f"RMSE         : {rmse_error:.2f} m")
print(f"Max error    : {max_error:.2f} m")
print(f"Average Inliers (proxy): {avg_inliers:.2f}")

summary = {
    "Total Images Processed": total_images,
    "Registration Success Rate (%)": registration_success_rate,
    "Recall @ 5m (%)": recall_5m,
    "Recall @ 20m (%)": recall_20m,
    "Precision @ 5m (%)": precision_5m,
    "Precision @ 20m (%)": precision_20m,
    "Median Localization Error (m)": median_error,
    "Error Std Dev (m)": err_std,
    "Avg Inlier Ratio": avg_inliers if "inlier_ratio" in df else avg_inliers,
    # "Avg Execution Time (s/image)": avg_time
}

summary_df = pd.DataFrame(summary.items(), columns=["Metric", "Value"])
print(summary_df)

for t in [5, 10, 20, 50]:
    acc = (df["error_m"] <= t).mean() * 100
    print(f"Accuracy @ {t}m: {acc:.1f}%")
corr = df["error_m"].corr(df["inliers"])
print("Correlation (error vs inliers):", corr)
