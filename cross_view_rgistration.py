import cv2
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import math
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# === GLOBAL CAMERA PARAMETERS ===
K = np.array([[456.46871015134053, 0.0, 643.3599454303429],
              [0.0, 455.40127946882507, 357.51076963739786],
              [0.0, 0.0, 1.0]])
D = np.array([0.03299031731836506, -0.03150792611905064, 
              -0.0017902177017069096, 0.00027220443810142304, 0.0])

class CrossViewGeolocalizer:
    def __init__(self, geotiff_path, gt_csv=None, imu_csv=None):
        self.geotiff_path = geotiff_path
        self.sat_img, self.transform, self.bounds = self._load_satellite()
        self.gt_interps = None
        self.imu_df = None
        
        if gt_csv and os.path.exists(gt_csv):
            self._load_gt_imu(gt_csv, imu_csv)
    
    def _load_satellite(self):
        """Load satellite GeoTIFF once."""
        with rasterio.open(self.geotiff_path) as src:
            sat_bands = src.read()
            if len(sat_bands) == 3:
                sat_img = np.moveaxis(sat_bands, 0, -1).astype(np.uint8)
            else:
                sat_img = np.repeat(sat_bands[0][:,:,np.newaxis], 3, axis=2)
            return sat_img, src.transform, rasterio.transform.array_bounds(
                src.height, src.width, src.transform)
    
    def _load_gt_imu(self, gt_csv, imu_csv):
        """Load and interpolate ground truth."""
        gt_df = pd.read_csv(gt_csv)
        self.gt_interps = (
            interp1d(gt_df['timestamp'], gt_df['latitude'], 'linear', fill_value='extrapolate'),
            interp1d(gt_df['timestamp'], gt_df['longitude'], 'linear', fill_value='extrapolate'),
            interp1d(gt_df['timestamp'], gt_df['altitude'], 'linear', fill_value='extrapolate')
        )
        
        if imu_csv and os.path.exists(imu_csv):
            self.imu_df = pd.read_csv(imu_csv)
        print(f"Loaded GT: {len(gt_df)} samples, IMU: {len(self.imu_df) if self.imu_df is not None else 0}")
    
    # def process_folder(self, drone_folder, output_dir='results', batch_size=5):
    #     """Process entire folder of drone images."""
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     # Find all images
    #     drone_images = glob.glob(os.path.join(drone_folder, "*.png")) + \
    #                   glob.glob(os.path.join(drone_folder, "*.jpg")) + \
    #                   glob.glob(os.path.join(drone_folder, "*.jpeg"))
    #     drone_images.sort()
        
    #     results = []
    #     for i in range(0, len(drone_images), batch_size):
    #         batch = drone_images[i:i+batch_size]
    #         print(f"Processing batch {i//batch_size + 1}/{(len(drone_images)+batch_size-1)//batch_size}")
    #         result_template = {
    #             'filename': '',
    #             'status': 'UNKNOWN',
    #             'lat_pred': np.nan,
    #             'lon_pred': np.nan,
    #             'lat_gt': np.nan, 
    #             'lon_gt': np.nan,
    #             'error_m': np.nan,
    #             'confidence': np.nan
    #         }

    #         # In your processing:
    #         result = result_template.copy()
    #         # result.update(computed_values)  # Safe merge
    #         # results.append(result)
    #         for img_path in batch:
    #             try:
    #                 result = self.process_single_image(img_path)
    #                 if result is None:
    #                     result = {'filename': img_path, 'status': 'FAILED', 'error_m': np.nan}
    #                 else:
    #                     result.setdefault('status', 'SUCCESS')  # Default to success if missing
    #                 results.append(result)
    #                 self.save_result(img_path, result, output_dir)
    #                 print(f"✓ {Path(img_path).name}: {result.get('error_m', 'N/A')}m")
    #             except Exception as e:
    #                 print(f"✗ Failed {img_path}: {e}")
    #                 # Add failed result with error flag
    #                 results.append({
    #                     'image': Path(img_path).name,
    #                     'timestamp': 0.0,
    #                     'est_lat': 0.0, 'est_lon': 0.0,
    #                     'gt_lat': 0.0, 'gt_lon': 0.0, 'gt_alt': 0.0,
    #                     'error_m': float('nan'),
    #                     'matches': 0,
    #                     'patch_offset': (0, 0),
    #                     'status': 'FAILED'
    #                 })
        
    #     self.save_results_summary(results, output_dir)
    #     return pd.DataFrame(results)

    def process_folder(self, drone_folder, output_dir):
        """Enhanced debugging version"""
        self.drone_folder = drone_folder  # Store for use in process_single_image
        # DEBUG: Check folder contents
        if not os.path.exists(drone_folder):
            print(f"❌ ERROR: Folder not found: {drone_folder}")
            return pd.DataFrame()
        
        image_files = [f for f in os.listdir(drone_folder) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
        
        print(f"📁 Folder: {drone_folder}")
        print(f"🔍 Found {len(image_files)} images: {image_files[:5]}...")  # Show first 5
        
        if len(image_files) == 0:
            print("❌ NO IMAGES FOUND! Check:")
            print("  1. Folder path: task_cv_model/drone_images/")
            print("  2. File extensions: .png, .jpg, .jpeg, .tiff")
            print("  3. Permissions: chmod -R 755 task_cv_model/")
            return pd.DataFrame()
        
        results = []
        for i, img_file in enumerate(image_files[:3]):
            print(f"Processing {i+1}/{len(image_files)}: {img_file}")
            try:
                result = self.process_single_image(img_file)  # Pass FULL PATH
                # Ensure result dict always has required fields
                if result is None:
                    result = {'filename': img_file, 'status': 'FAILED_NO_PRED', 'error_m': np.nan}
                result.setdefault('status', 'SUCCESS')
                result.setdefault('error_m', np.nan)
                results.append(result)
                print(f"✅ {img_file}: {result.get('status', 'UNKNOWN')}")
            except Exception as e:
                print(f"❌ {img_file}: {str(e)[:100]}")
                results.append({
                    'filename': img_file, 'status': 'ERROR', 
                    'error': str(e), 'error_m': np.nan
                })
                break
        
        print(f"🎉 Processed {len(results)} images")
        self.save_results_summary(results, output_dir)
        return pd.DataFrame(results)
    
    def process_single_image(self, img_path):
        """Full pipeline for single image."""
        # image_name = Path(img_path).stem
        img_path = os.path.join(self.drone_folder, img_path)  # ADD THIS LINE
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image_name = Path(img_path).stem
        image_ts = self._parse_timestamp(image_name)
        
        # Get ground truth
        gt_lat, gt_lon, gt_alt = self._get_gt_at_timestamp(image_ts)
        
        # Load and preprocess image
        drone_rgb = cv2.imread(img_path)
        if drone_rgb is None:
            raise ValueError(f"Failed to load: {img_path}")
        if drone_rgb is None:
            raise ValueError("Could not load image")
        drone_rgb = cv2.cvtColor(drone_rgb, cv2.COLOR_BGR2RGB)
        drone_undist = self._undistort_image(drone_rgb[:,:,0])  # Use grayscale
        drone_gray = cv2.cvtColor(drone_undist, cv2.COLOR_RGB2GRAY)
        # Feature matching pipeline
        drone_kp, drone_des = self._extract_features(drone_gray)
        best_patch, best_matches, best_offset, H = self._find_best_patch(drone_kp, drone_des)
        
        # Geolocalization
        center_sat_px = self._warp_center(H, drone_undist.shape)
        est_lat, est_lon = self._pixel_to_gps(center_sat_px)
        
        # Error calculation
        error_m = self._haversine_distance(gt_lat, gt_lon, est_lat, est_lon)
        
        return {
            'image': image_name,
            'timestamp': image_ts,
            'est_lat': est_lat, 'est_lon': est_lon,
            'gt_lat': gt_lat, 'gt_lon': gt_lon, 'gt_alt': gt_alt,
            'error_m': error_m,
            'matches': len(best_matches),
            'patch_offset': best_offset,
            'status': 'SUCCESS'
        }
    
    def _parse_timestamp(self, image_name):
        """Parse timestamp like '1445.599225344' from filename."""
        try:
            return float(image_name.split('.')[0])
        except:
            return 0.0
    
    def _get_gt_at_timestamp(self, ts):
        """Interpolate GT or return defaults."""
        if self.gt_interps is None:
            return 17.3850, 78.4867, 100.0  # Hyderabad default
        
        try:
            nearest_ts = ts
            return (self.gt_interps[0](nearest_ts), 
                   self.gt_interps[1](nearest_ts), 
                   self.gt_interps[2](nearest_ts))
        except:
            return 17.3850, 78.4867, 100.0
    
    def _undistort_image(self, gray_img):
        """Apply camera undistortion."""
        h, w = gray_img.shape
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
        undistorted = cv2.undistort(gray_img, K, D, None, new_K)
        return undistorted
    
    def _extract_features(self, img, max_features=3000):
        """Optimized SIFT extraction."""
        sift = cv2.SIFT_create(nfeatures=max_features)
        kp, des = sift.detectAndCompute(img, None)
        return kp, des
    
    def _find_best_patch(self, drone_kp, drone_des, window_size=768, stride=128):
        """Efficient pyramid sliding window search."""
        sat_gray = cv2.cvtColor(self.sat_img, cv2.COLOR_RGB2GRAY)
        best_score, best_matches, best_patch, best_offset = 0, [], None, (0, 0)
        best_H = None
        
        sift_fast = cv2.SIFT_create(nfeatures=1000)
        
        for scale in [1.0, 0.75, 0.5]:  # Pyramid
            scaled_sat = cv2.resize(sat_gray, None, fx=scale, fy=scale)
            ws = int(window_size * scale)
            sat_h, sat_w = scaled_sat.shape
            
            for y in range(0, sat_h-ws, stride):
                for x in range(0, sat_w-ws, stride):
                    patch = scaled_sat[y:y+ws, x:x+ws]
                    patch_kp, patch_des = sift_fast.detectAndCompute(patch, None)
                    
                    if len(patch_des) < 10: continue
                    
                    matches = self._match_features(drone_des, patch_des)
                    score = len(matches)
                    
                    if score > best_score and score > 15:
                        H, inliers = self._compute_homography(drone_kp, patch_kp, matches)
                        if inliers > 8:
                            best_score = score
                            best_matches = matches
                            best_patch = patch
                            best_offset = (x/scale, y/scale)
                            best_H = H
        
        return best_patch, best_matches, best_offset, best_H
    
    def _match_features(self, des1, des2, ratio_thresh=0.7):
        """FLANN + Lowe ratio test."""
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
        return good
    
    def _compute_homography(self, kp1, kp2, matches):
        """RANSAC homography."""
        if len(matches) < 4: return None, 0
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
        return H, np.sum(mask.ravel()==1)
    
    def _warp_center(self, H, img_shape):
        """Map image center through homography."""
        if H is None:
            return (self.sat_img.shape[1]//2, self.sat_img.shape[0]//2)
        h, w = img_shape
        center = np.array([w/2, h/2, 1]).reshape(3,1)
        center_warped = H @ center
        return (int(center_warped[0,0]/center_warped[2,0]), 
                int(center_warped[1,0]/center_warped[2,0]))
    
    def _pixel_to_gps(self, row, col):
        """Convert pixel to lat/lon."""
        lon, lat = rasterio.transform.xy(self.transform, row, col)
        return lat, lon
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Distance in meters."""
        R = 6371000
        phi1, phi2 = np.radians([lat1, lat2])
        dphi = np.radians(lat2-lat1)
        dlambda = np.radians(lon2-lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    def save_result(self, img_path, result, output_dir):
        """Save visualizations and results."""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        sat_overview = cv2.resize(self.sat_img, (800, 800))
        oy, ox = int(result['patch_offset'][1]), int(result['patch_offset'][0])
        oh, ow = 100, 100
        cv2.rectangle(sat_overview, (ox, oy), (ox+ow, oy+oh), (0,255,0), 3)
        plt.imshow(sat_overview)
        plt.title('Satellite + Best Patch')
        
        plt.subplot(132)
        plt.text(0.1, 0.5, f'Image: {result["image"]}\n'
                           f'Est: {result["est_lat"]:.6f}, {result["est_lon"]:.6f}\n'
                           f'GT: {result["gt_lat"]:.6f}, {result["gt_lon"]:.6f}\n'
                           f'Error: {result["error_m"]:.1f}m\n'
                           f'Matches: {result["matches"]}', 
                 transform=plt.gca().transAxes, fontsize=12)
        plt.axis('off')
        plt.title('Localization Result')
        
        plt.subplot(133)
        error_val = result.get('error_m', float('nan'))
        plt.bar(['Camera', 'GT'], [error_val, 0], color=['red', 'green'])
        plt.ylabel('Error (m)')
        plt.title('Accuracy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{Path(img_path).stem}_result.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def save_results_summary(self, results, output_dir):
        if not results:
            print("⚠️  No results to save - empty processing")
            pd.DataFrame([{'status': 'NO_IMAGES', 'error': 'No images processed'}]).to_csv(
                os.path.join(output_dir, 'summary.csv'), index=False)
            return
        
        df = pd.DataFrame(results)
        print(f"Available columns: {df.columns.tolist()}")
        
        # Ensure required columns exist
        required_cols = ['status', 'error_m', 'filename']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 'UNKNOWN' if col == 'status' else np.nan
        
        success_mask = (df['status'] == 'SUCCESS') & df['error_m'].notna()
        num_success = success_mask.sum()
        mean_error = df.loc[success_mask, 'error_m'].mean() if num_success > 0 else np.nan
        
        print(f"=== SUMMARY ===")
        print(f"Total images: {len(df)}")
        print(f"Successful: {num_success}")
        print(f"Mean error (success): {mean_error:.2f}m" if not np.isnan(mean_error) else "No successful localizations")
        
        # Save detailed results
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
        
        # Save summary stats
        summary = pd.DataFrame({
            'total': [len(df)], 'success': [num_success], 
            'success_rate': [num_success/len(df)*100],
            'mean_error_m': [mean_error]
        })
        summary.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
        print("Processing complete! Check folder for CSVs.")

    
    # def save_results_summary(self, results, output_dir):
    #     """Save CSV summary with error handling."""
    #     df = pd.DataFrame(results)
    #     print("Available columns:", df.columns.tolist())
    #     print("First few rows:\n", df.head())
    #     print("Sample data types:\n", df.dtypes)
        
    #     # Ensure all required columns exist
    #     required_cols = ['image', 'timestamp', 'est_lat', 'est_lon', 'gt_lat', 'gt_lon', 'gt_alt', 'error_m', 'matches']
    #     for col in required_cols:
    #         if col not in df.columns:
    #             df[col] = np.nan
        
    #     # Filter successful results for statistics
    #     # success_mask = (df['status'] == 'SUCCESS') & df['error_m'].notna()
    #     if 'status' in df.columns and 'error_m' in df.columns:
    #         success_mask = (df['status'] == 'SUCCESS') & df['error_m'].notna()
    #     else:
    #         print("Warning: Missing 'status' or 'error_m' columns. Creating default mask.")
    #         success_mask = pd.Series([True] * len(df), index=df.index)
        
    #     num_success = success_mask.sum()
    #     mean_error_success = df.loc[success_mask, 'error_m'].mean() if num_success > 0 else np.nan
    #     successful_df = df[success_mask]
        
    #     df.to_csv(os.path.join(output_dir, 'geolocalization_results.csv'), index=False)
        
    #     print(f"\n=== SUMMARY ===")
    #     print(f"Total images: {len(results)}")
    #     print(f"Successful: {len(successful_df)}")
    #     if len(successful_df) > 0:
    #         print(f"Mean Error: {successful_df['error_m'].mean():.1f}m")
    #         print(f"Median Error: {successful_df['error_m'].median():.1f}m")
    #         print(f"Best: {successful_df['error_m'].min():.1f}m")
    #         print(f"Worst: {successful_df['error_m'].max():.1f}m")
    #     else:
    #         print("No successful localizations")

# === USAGE ===
if __name__ == "__main__":
    # Initialize (replace paths)
    geoloc = CrossViewGeolocalizer(
        geotiff_path='task_cv_model/map.tif',
        gt_csv='task_cv_model/train_data/ground_truth.csv',  # Optional: timestamp,latitude,altitude,longitude
        imu_csv='task_cv_model/train_data/imu_data.csv'      # Optional
    )
    
    # Process entire folder
    results_df = geoloc.process_folder(
        drone_folder='task_cv_model/train_data/drone_images/',
        output_dir='task_cv_model/geoloc_results'
    )
    
    print("Processing complete! Check task_cv_model/geoloc_results/ folder.")
