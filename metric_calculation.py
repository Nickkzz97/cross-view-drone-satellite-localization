import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_results(csv_path):
    columns = [
        "image",
        "status",
        "pred_lat",
        "pred_lon",
        "gt_lat",
        "gt_lon",
        "error_m"
    ]

    df = pd.read_csv(csv_path, header=None, names=columns)

    # Convert numeric columns safely
    numeric_cols = ["pred_lat", "pred_lon", "gt_lat", "gt_lon", "error_m"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def compute_metrics_from_csv(df):
    total = len(df)

    # Successful registrations
    success_df = df[df["status"] == "success"].copy()
    failed_df = df[df["status"] != "success"]

    success = len(success_df)

    errors = success_df["error_m"].values

    # --- Basic error stats ---
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    rmse_error = np.sqrt(np.mean(errors ** 2))
    max_error = np.max(errors)

    err_std = np.std(errors)
    err_var = np.var(errors)

    # --- Threshold logic ---
    tp_5m = np.sum(errors <= 5)
    tp_20m = np.sum(errors <= 20)

    fn_5m = total - tp_5m
    fn_20m = total - tp_20m

    recall_5m = 100 * tp_5m / total
    recall_20m = 100 * tp_20m / total

    precision_5m = 100 * tp_5m / max(success, 1)
    precision_20m = 100 * tp_20m / max(success, 1)

    registration_rate = 100 * success / total

    # --- Print exactly like your pipeline ---
    print(f"Error Std Dev        : {err_std:.2f} m")
    print(f"Error Variance       : {err_var:.2f} m^2")

    print(f"Recall @ 5m          : {recall_5m:.2f}%")
    print(f"Recall @ 20m         : {recall_20m:.2f}%")

    print(f"Precision @ 5m       : {precision_5m:.2f}%")
    print(f"Precision @ 20m      : {precision_20m:.2f}%")

    print(f"Mean error           : {mean_error:.2f} m")
    print(f"Median error         : {median_error:.2f} m")
    print(f"RMSE                 : {rmse_error:.2f} m")
    print(f"Max error            : {max_error:.2f} m")

    print(f"Registration rate    : {registration_rate:.2f}%")

    return {
        "mean": mean_error,
        "median": median_error,
        "rmse": rmse_error,
        "max": max_error,
        "std": err_std,
        "var": err_var,
        "recall_5m": recall_5m,
        "recall_20m": recall_20m,
        "precision_5m": precision_5m,
        "precision_20m": precision_20m,
        "registration_rate": registration_rate,
        "num_success": success,
        "num_total": total
    }

df = load_results("output/train_errors_final.csv")
metrics = compute_metrics_from_csv(df)

df_success = df[df.status == "success"]

plt.hist(df_success["error_m"], bins=30)
plt.xlabel("Localization Error (m)")
plt.ylabel("Count")
plt.title("Error Distribution")
plt.show()
