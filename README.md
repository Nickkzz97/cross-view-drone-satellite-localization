# SuperPoint-Based Cross-View Drone Localization

This repository implements a **vision-based cross-view localization pipeline** for estimating the geographic location (**latitude, longitude**) of drone images by registering them with a **georeferenced satellite map**.  
The approach leverages **SuperPoint-based feature extraction**, a **sliding-window satellite patch search**, **homography estimation**, and **geospatial coordinate conversion** to perform visual localization.

The pipeline supports:
- Localization with ground truth GPS (for evaluation)
- Localization on unseen test images without ground truth
- Saving predictions and metrics to CSV files

---

##  Repository Overview

Key scripts in this repository:

- `Estimate_latlon.py` – Runs localization and generates predictions (with or without ground truth)
- `metric_calculation.py` – Computes evaluation metrics from saved CSV results
- `predict_on_test_image.py` – Performs GPS estimation on test images without ground truth
- `utils.py` – Helper functions for matching, homography, and coordinate conversion
- `visualization.ipynb` – Qualitative visualization and debugging
- `requirements.txt` – Required Python dependencies
- `output/` – Stores generated CSV files and visualizations

---

##  Installation

### 1. Clone the Repository
```bash
git clone <repository_url>
cd <repository_name>

### 2. Install Dependencies
pip install -r requirements.txt

## Localization with Ground Truth (Training / Validation)

### Step 1: Run Localization: If ground truth GPS coordinates are available for drone images, you can estimate locations and evaluate performance as follows:

python Estimate_latlon.py
  - Registers each drone image with the satellite map
  - Estimates latitude and longitude
  - Saves predictions and errors to a CSV file

### Step 2: Compute Evaluation Metrics

python metric_calculation.py

##Localization on Test Images (No Ground Truth)
     python predict_on_test_image.py

## Qualitative verification of alignment and GPS estimation can be performed using:
    visualization.ipynb

    Visualizations include:
        - Selected satellite patch
        - Warped drone image alignment
        - Estimated GPS location overlaid on the satellite image
