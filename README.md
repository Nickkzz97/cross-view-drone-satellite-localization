Cross-View Drone-to-Satellite Localization (Classical Methods)
==============================================================

This repository contains an initial implementation of a classical computer
vision pipeline for cross-view drone-to-satellite image localization.
The focus of this implementation is on feature-based matching, geometric
transformation, and evaluation for registering drone imagery with
georeferenced satellite maps.

The project primarily explores classical feature extraction and geometric
alignment techniques as a baseline for cross-view image stitching and
localization.

Installation
------------

1. Clone the repository:

   git clone <repository_url>
   cd <repository_name>

2. Install required dependencies:

   pip install -r requirements.txt


Usage
-----

The pipeline can be executed in stages depending on the task:

1. Feature extraction and comparison:
   Use Compare_feature_extraction.ipynb to analyze the behavior of different
   classical feature descriptors under cross-view conditions.

2. Drone-to-satellite matching:
   Run ***find_drone_match.py***: identify candidate satellite regions that
   correspond to a given drone image.

3. Geometric alignment:
   ***Geometric_transformation.py***: estimate geometric transformations
   between matched drone and satellite images.

4. Evaluation:
   Use ***evaluation.py***: compute performance metrics and assess localization accuracy.

5. Testing:
   Execute ***test_output.py*** : validate the end-to-end pipeline and inspect generated outputs.


Visualization and Debugging
---------------------------

Qualitative analysis and debugging can be performed using:
- ***Compare_feature_extraction.ipynb***

Modify the file paths in the above commands according to your directory structure.
