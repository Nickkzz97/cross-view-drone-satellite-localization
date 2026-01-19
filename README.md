SuperPoint-Based Cross-View Drone Localization
==============================================

This repository implements a vision-based cross-view localization pipeline for estimating
the geographic location (latitude, longitude) of drone images by registering them with a
georeferenced satellite map.

The approach leverages SuperPoint-based feature extraction, a sliding-window satellite
patch search, homography estimation, and geospatial coordinate conversion to perform
visual localization.

The pipeline supports:
- Localization with ground truth GPS (for evaluation)
- Localization on unseen test images without ground truth
- Saving predictions and evaluation metrics to CSV files


Repository Overview
-------------------

Key scripts and files in this repository:

- Estimate_latlon.py
  Runs the localization pipeline and generates GPS predictions with or without ground truth.

- metric_calculation.py
  Computes evaluation metrics (e.g., localization error) from saved CSV results.

- predict_on_test_image.py
  Performs GPS estimation on test images without ground truth GPS data.

- utils.py
  Contains helper functions for feature matching, homography estimation,
  and coordinate conversion.

- visualization.ipynb
  Provides qualitative visualization and debugging utilities.

- requirements.txt
  Lists required Python dependencies.

- output/
  Directory for storing generated CSV files and visualizations.


Installation
------------

1. Clone the repository:

   git clone <repository_url>
   cd <repository_name>

2. Install dependencies:

   pip install -r requirements.txt


Localization with Ground Truth (Training / Validation)
------------------------------------------------------

If ground truth GPS coordinates are available for drone images, the pipeline can be
run in evaluation mode.

Step 1: Run localization

   python Estimate_latlon.py

This script:
- Registers each drone image with the satellite map
- Estimates latitude and longitude
- Saves predictions and localization errors to a CSV file

Step 2: Compute evaluation metrics

   python metric_calculation.py


Localization on Test Images (No Ground Truth)
--------------------------------------------

For unseen drone images without ground truth GPS:

   python predict_on_test_image.py

This script estimates GPS coordinates and saves predictions to a CSV file.


Visualization and Qualitative Analysis
--------------------------------------

Qualitative verification of alignment and GPS estimation can be performed using:

   visualization.ipynb

The notebook provides visualizations such as:
- Selected satellite patches
- Warped drone image alignment
- Estimated GPS locations overlaid on the satellite image
