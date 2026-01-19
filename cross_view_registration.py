import numpy as np
import cv2
import rasterio
from rasterio.transform import Affine
import pyproj
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path
import gc
import warnings
warnings.filterwarnings('ignore')

class MemoryEfficientGeoTIFFLoader:
    def __init__(self, geotiff_path, max_pixels=2000*2000):
        """Memory-efficient GeoTIFF loader"""
        self.dataset = rasterio.open(geotiff_path)
        
        # Calculate scale factor to limit memory
        total_pixels = self.dataset.width * self.dataset.height
        print(f"Original size: {self.dataset.width}x{self.dataset.height} ({total_pixels:,} pixels)")
        
        if total_pixels > max_pixels:
            self.scale_factor = np.sqrt(max_pixels / total_pixels)
            new_width = int(self.dataset.width * self.scale_factor)
            new_height = int(self.dataset.height * self.scale_factor)
            print(f"Downsampling to: {new_width}x{new_height}")
            
            # Read downsampled
            data = self.dataset.read(
                out_shape=(self.dataset.count, new_height, new_width),
                resampling=rasterio.enums.Resampling.bilinear
            )
            self.image = np.transpose(data, (1, 2, 0))
        else:
            self.image = np.transpose(self.dataset.read([1, 2, 3]), (1, 2, 0))
            self.scale_factor = 1.0
        
        print(f"Loaded size: {self.image.shape[1]}x{self.image.shape[0]}")
        
        # Convert to uint8 to save memory
        if self.image.dtype != np.uint8:
            self.image = (self.image / 256).astype(np.uint8)
        
        # Adjust transform for scaling
        self.transform = self.dataset.transform
        if self.scale_factor != 1.0:
            self.transform = Affine(
                self.transform.a / self.scale_factor,
                self.transform.b,
                self.transform.c,
                self.transform.d,
                self.transform.e / self.scale_factor,
                self.transform.f
            )
        
        self.crs = self.dataset.crs
        self.transformer_to_wgs84 = pyproj.Transformer.from_crs(
            self.crs, 'EPSG:4326', always_xy=True
        )
        
        # Estimate GSD
        self.gsd = self.estimate_gsd()
        print(f"Estimated GSD: {self.gsd:.3f} m/px")
        
        # Clear memory
        gc.collect()
    
    def estimate_gsd(self):
        """Estimate ground sampling distance"""
        try:
            # From transform
            pixel_size_x = abs(self.transform.a)
            pixel_size_y = abs(self.transform.e)
            
            # If in degrees, convert to meters
            if pixel_size_x > 0.001:
                meters_per_degree = 111320
                pixel_size_x *= meters_per_degree
                pixel_size_y *= meters_per_degree
            
            return max(0.1, min((pixel_size_x + pixel_size_y) / 2, 10.0))
        except:
            return 0.5
    
    def pixel_to_latlon(self, x, y):
        """Convert pixel to lat/lon"""
        # Adjust for scaling
        adj_x = x / self.scale_factor if self.scale_factor != 1.0 else x
        adj_y = y / self.scale_factor if self.scale_factor != 1.0 else y
        
        map_x, map_y = self.transform * (adj_x, adj_y)
        lon, lat = self.transformer_to_wgs84.transform(map_x, map_y)
        return lat, lon
    
    def latlon_to_pixel(self, lat, lon):
        """Convert lat/lon to pixel"""
        transformer_from_wgs84 = pyproj.Transformer.from_crs(
            'EPSG:4326', self.crs, always_xy=True
        )
        map_x, map_y = transformer_from_wgs84.transform(lon, lat)
        inv_transform = ~self.transform
        x, y = inv_transform * (map_x, map_y)
        
        # Adjust for scaling
        if self.scale_factor != 1.0:
            x *= self.scale_factor
            y *= self.scale_factor
        
        return x, y

class LightweightLocalizer:
    def __init__(self, camera_params, geotiff_path, use_imu=False):
        """Lightweight localizer with memory management"""
        self.camera_matrix = np.array(camera_params['camera_matrix']['data']).reshape(3, 3)
        self.dist_coeffs = np.array(camera_params['distortion_coeffs']['data']).flatten()
        self.image_width = camera_params['image_width']
        self.image_height = camera_params['image_height']
        self.use_imu = use_imu
        
        # Load satellite map efficiently
        print("Loading satellite map (memory-efficient)...")
        self.satellite = MemoryEfficientGeoTIFFLoader(geotiff_path, max_pixels=1500*1500)
        
        # Preprocess satellite map once
        self.preprocess_satellite_map()
        
        # Initialize tracking
        self.last_position = None
        self.trajectory = []
        self.altitude_estimate = 50  # meters
        
        # Feature cache (empty initially, loaded on demand)
        self.sat_features = None
        
        print("Localizer initialized (memory-efficient mode)")
    
    def preprocess_satellite_map(self):
        """Minimal preprocessing to save memory"""
        # Convert to grayscale
        if len(self.satellite.image.shape) == 3:
            self.sat_gray = cv2.cvtColor(self.satellite.image, cv2.COLOR_RGB2GRAY)
        else:
            self.sat_gray = self.satellite.image.copy()
        
        # Simple contrast enhancement
        self.sat_enhanced = cv2.equalizeHist(self.sat_gray)
        
        # Create edge map (small)
        self.sat_edges = cv2.Canny(self.sat_enhanced, 50, 150)
        
        # Clear intermediate arrays
        del self.satellite.image  # Free original image memory
        gc.collect()
    
    def extract_satellite_features_lazy(self):
        """Extract satellite features only when needed"""
        if self.sat_features is None:
            print("Extracting satellite features (lazy loading)...")
            
            # Use ORB instead of SIFT for memory efficiency
            orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2)
            self.sat_keypoints, self.sat_descriptors = orb.detectAndCompute(self.sat_enhanced, None)
            
            self.sat_features = {
                'keypoints': self.sat_keypoints,
                'descriptors': self.sat_descriptors
            }
            
            print(f"Extracted {len(self.sat_keypoints)} ORB features")
        
        return self.sat_features
    
    def preprocess_drone_image(self, drone_image):
        """Minimal drone image preprocessing"""
        # Resize drone image to save memory
        scale = 0.5
        h, w = drone_image.shape[:2]
        resized = cv2.resize(drone_image, (int(w * scale), int(h * scale)))
        
        # Undistort (simplified)
        if self.dist_coeffs.any():
            undistorted = cv2.undistort(resized, self.camera_matrix, self.dist_coeffs)
        else:
            undistorted = resized
        
        # Convert to grayscale
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        
        # Simple enhancement
        enhanced = cv2.equalizeHist(gray)
        
        # Extract ORB features (memory efficient)
        orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2)
        keypoints, descriptors = orb.detectAndCompute(enhanced, None)
        
        return {
            'image': undistorted,
            'gray': gray,
            'enhanced': enhanced,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'scale': scale
        }
    
    def template_match_fast(self, drone_data, search_center=None):
        """Fast template matching"""
        # Create template from drone edges
        drone_edges = cv2.Canny(drone_data['enhanced'], 50, 150)
        
        # Estimate template size based on altitude
        # Simple scaling: assume 1 pixel in drone ≈ 0.5m at 50m altitude
        base_size = 100  # pixels at 50m
        template_size = int(base_size * (50 / self.altitude_estimate))
        template_size = max(50, min(template_size, 200))
        
        # Resize template
        template = cv2.resize(drone_edges, (template_size, template_size))
        
        # Define search area
        if search_center:
            search_radius = 400  # pixels
            x_min = max(0, int(search_center[0] - search_radius))
            x_max = min(self.sat_edges.shape[1], int(search_center[0] + search_radius))
            y_min = max(0, int(search_center[1] - search_radius))
            y_max = min(self.sat_edges.shape[0], int(search_center[1] + search_radius))
            
            search_area = self.sat_edges[y_min:y_max, x_min:x_min + search_radius*2]
        else:
            # First frame: search in a grid pattern
            h, w = self.sat_edges.shape
            grid_size = 3
            cell_w = w // grid_size
            cell_h = h // grid_size
            
            best_match = None
            best_confidence = 0
            
            for i in range(grid_size):
                for j in range(grid_size):
                    x_min = j * cell_w
                    x_max = min((j + 1) * cell_w, w)
                    y_min = i * cell_h
                    y_max = min((i + 1) * cell_h, h)
                    
                    search_area = self.sat_edges[y_min:y_max, x_min:x_max]
                    
                    if template.shape[0] > search_area.shape[0] or template.shape[1] > search_area.shape[1]:
                        continue
                    
                    result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                    
                    if max_val > best_confidence:
                        best_confidence = max_val
                        match_x = x_min + max_loc[0] + template.shape[1] // 2
                        match_y = y_min + max_loc[1] + template.shape[0] // 2
                        best_match = (match_x, match_y)
            
            return best_match, best_confidence
        
        # Template matching
        if template.shape[0] > search_area.shape[0] or template.shape[1] > search_area.shape[1]:
            return None, 0.0
        
        result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val < 0.25:  # Low confidence threshold
            return None, max_val
        
        # Calculate position
        match_x = x_min + max_loc[0] + template.shape[1] // 2
        match_y = y_min + max_loc[1] + template.shape[0] // 2
        
        return (match_x, match_y), max_val
    
    def feature_match_fast(self, drone_data, search_center=None):
        """Fast feature matching"""
        sat_features = self.extract_satellite_features_lazy()
        
        if drone_data['descriptors'] is None or sat_features['descriptors'] is None:
            return None, 0
        
        # BFMatcher for ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(drone_data['descriptors'], sat_features['descriptors'])
        
        if len(matches) < 10:
            return None, len(matches)
        
        # Filter by distance
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:min(50, len(matches))]
        
        # Get matched points
        drone_pts = np.float32([drone_data['keypoints'][m.queryIdx].pt for m in good_matches])
        sat_pts = np.float32([sat_features['keypoints'][m.trainIdx].pt for m in good_matches])
        
        # Adjust for drone image scaling
        drone_pts = drone_pts / drone_data['scale']
        
        # Find affine transformation
        if len(good_matches) >= 3:
            try:
                H, inliers = cv2.estimateAffinePartial2D(drone_pts, sat_pts, method=cv2.RANSAC)
                
                if H is not None and np.sum(inliers) >= 5:
                    # Convert to homography
                    H_homog = np.eye(3)
                    H_homog[:2, :] = H
                    
                    # Map center
                    drone_center = np.array([[drone_data['image'].shape[1] / 2, 
                                            drone_data['image'].shape[0] / 2]], 
                                          dtype=np.float32).reshape(-1, 1, 2)
                    
                    sat_center = cv2.perspectiveTransform(drone_center, H_homog)
                    return (sat_center[0][0][0], sat_center[0][0][1]), np.sum(inliers)
            except:
                pass
        
        return None, len(good_matches)
    
    def refine_position(self, initial_pos, confidence):
        """Simple position refinement"""
        if initial_pos is None or confidence < 0.3:
            return initial_pos, confidence
        
        # Small local search
        search_radius = 50
        x, y = initial_pos
        
        # Create small search window
        x_min = max(0, int(x - search_radius))
        x_max = min(self.sat_enhanced.shape[1], int(x + search_radius))
        y_min = max(0, int(y - search_radius))
        y_max = min(self.sat_enhanced.shape[0], int(y + search_radius))
        
        if x_max <= x_min or y_max <= y_min:
            return initial_pos, confidence
        
        # Extract region for visualization
        self.last_refined_region = self.sat_enhanced[y_min:y_max, x_min:x_max]
        
        return initial_pos, confidence
    
    def localize(self, drone_image, imu_data=None, ground_truth=None, visualize=False):
        """Fast localization with memory management"""
        print(f"\nLocalizing image {len(self.trajectory) + 1}")
        
        # Preprocess drone image
        drone_data = self.preprocess_drone_image(drone_image)
        print(f"  Drone features: {len(drone_data['keypoints'])}")
        
        # Get search center
        search_center = None
        if self.use_imu and imu_data and self.last_position:
            dt = 0.1
            velocity_x = imu_data.get('lin_acc_x', 0) * dt
            velocity_y = imu_data.get('lin_acc_y', 0) * dt
            
            # Convert to pixels
            velocity_px = (velocity_x / self.satellite.gsd, velocity_y / self.satellite.gsd)
            search_center = (self.last_position[0] + velocity_px[0],
                           self.last_position[1] + velocity_px[1])
        
        # Method 1: Fast template matching
        print("  Method 1: Template matching...")
        template_pos, template_conf = self.template_match_fast(drone_data, search_center)
        
        if template_pos and template_conf > 0.3:
            print(f"    Template match: confidence {template_conf:.3f}")
            estimated_pos = template_pos
            confidence = template_conf
            method = "template"
        else:
            # Method 2: Feature matching
            print("  Method 2: Feature matching...")
            feature_pos, feature_matches = self.feature_match_fast(drone_data, search_center)
            
            if feature_pos and feature_matches >= 10:
                print(f"    Feature match: {feature_matches} matches")
                estimated_pos = feature_pos
                confidence = min(1.0, feature_matches / 50.0)
                method = f"features ({feature_matches})"
            else:
                print(f"    Feature match failed: {feature_matches if feature_matches else 0} matches")
                
                # Fallback: use last position or center
                if self.last_position:
                    estimated_pos = self.last_position
                    confidence = 0.1
                    method = "fallback (last position)"
                    print("    Using last known position")
                else:
                    # First frame: use image center
                    estimated_pos = (self.sat_enhanced.shape[1] // 2, 
                                   self.sat_enhanced.shape[0] // 2)
                    confidence = 0.05
                    method = "fallback (center)"
                    print("    Using image center (first frame)")
        
        # Refine position
        estimated_pos, confidence = self.refine_position(estimated_pos, confidence)
        
        # Convert to GPS
        lat, lon = self.satellite.pixel_to_latlon(estimated_pos[0], estimated_pos[1])
        
        # Update tracking
        self.last_position = estimated_pos
        self.trajectory.append({
            'timestamp': len(self.trajectory),
            'pixel': estimated_pos,
            'gps': (lat, lon),
            'confidence': confidence,
            'method': method
        })
        
        # Calculate error if ground truth available
        error = None
        if ground_truth and 'latitude' in ground_truth:
            error = self.haversine_distance(
                lat, lon,
                ground_truth['latitude'], ground_truth['longitude']
            )
            print(f"  Error: {error:.2f}m")
        
        print(f"  Result: {method}, Confidence: {confidence:.3f}")
        print(f"  Position: ({estimated_pos[0]:.0f}, {estimated_pos[1]:.0f})")
        print(f"  GPS: ({lat:.6f}, {lon:.6f})")
        
        if visualize and len(self.trajectory) <= 3:  # Only visualize first few
            self.visualize_minimal(drone_data, estimated_pos, confidence, method)
        
        # Clear memory
        gc.collect()
        
        return (lat, lon), estimated_pos, confidence, method, error
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Fast distance calculation"""
        R = 6371000
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    def visualize_minimal(self, drone_data, estimated_pos, confidence, method):
        """Minimal visualization to save memory"""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Drone image
        axes[0].imshow(cv2.cvtColor(drone_data['image'], cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Drone\n{drone_data["image"].shape[1]}x{drone_data["image"].shape[0]}')
        axes[0].axis('off')
        
        # Satellite with position
        axes[1].imshow(self.sat_enhanced, cmap='gray')
        axes[1].scatter(estimated_pos[0], estimated_pos[1],
                       c='red', s=50, marker='x', linewidths=2)
        axes[1].set_title(f'Satellite\nMethod: {method}')
        axes[1].axis('off')
        
        # Zoomed view if available
        if hasattr(self, 'last_refined_region'):
            axes[2].imshow(self.last_refined_region, cmap='gray')
            center_x = self.last_refined_region.shape[1] // 2
            center_y = self.last_refined_region.shape[0] // 2
            axes[2].scatter(center_x, center_y, c='red', s=30, marker='x')
            axes[2].set_title(f'Refined\nConf: {confidence:.3f}')
        else:
            axes[2].text(0.3, 0.5, f'Confidence: {confidence:.3f}\nMethod: {method}',
                        fontsize=10, ha='center')
            axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

class EfficientDataLoader:
    def __init__(self, image_folder, imu_csv_path=None, gt_csv_path=None):
        """Efficient data loader that handles missing files"""
        self.image_folder = Path(image_folder)
        
        # Load images
        self.image_files = sorted(self.image_folder.glob("*.png"))
        self.image_timestamps = []
        
        for img_file in self.image_files:
            try:
                timestamp = float(img_file.stem)
                self.image_timestamps.append(timestamp)
            except:
                continue
        
        print(f"Found {len(self.image_files)} images")
        
        # Load IMU data if available
        self.imu_df = None
        if imu_csv_path and Path(imu_csv_path).exists():
            self.imu_df = pd.read_csv(imu_csv_path)
            print(f"Loaded {len(self.imu_df)} IMU records")
        else:
            print("IMU data not available")
        
        # Load ground truth if available
        self.gt_df = None
        if gt_csv_path and Path(gt_csv_path).exists():
            self.gt_df = pd.read_csv(gt_csv_path)
            print(f"Loaded {len(self.gt_df)} ground truth records")
        else:
            print("Ground truth not available")
        
        # Create interpolators
        self._create_interpolators()
        self.image_cache = {}
    
    def _create_interpolators(self):
        """Create interpolators for available data"""
        self.imu_interpolator = None
        self.gt_interpolator = None
        
        # IMU interpolator
        if self.imu_df is not None and len(self.imu_df) > 0:
            imu_times = self.imu_df.iloc[:, 0].values
            imu_data = self.imu_df.iloc[:, 1:].values
            
            self.imu_interpolator = interp1d(
                imu_times, imu_data, axis=0,
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )
        
        # Ground truth interpolator
        if self.gt_df is not None and len(self.gt_df) > 0:
            gt_times = self.gt_df.iloc[:, 0].values
            gt_data = self.gt_df.iloc[:, 1:].values
            
            self.gt_interpolator = interp1d(
                gt_times, gt_data, axis=0,
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )
    
    def get_image_data(self, idx):
        """Get image by index with caching"""
        if idx >= len(self.image_files):
            raise IndexError(f"Index {idx} out of range")
        
        if idx not in self.image_cache:
            img = cv2.imread(str(self.image_files[idx]))
            if img is None:
                raise ValueError(f"Could not load image {idx}")
            self.image_cache[idx] = img
        
        return self.image_cache[idx], self.image_timestamps[idx]
    
    def get_imu_data_at_time(self, timestamp):
        """Get IMU data at timestamp"""
        if self.imu_interpolator is None:
            return {}
        
        try:
            data = self.imu_interpolator(timestamp)
            return {
                'ang_vel_x': float(data[0]),
                'ang_vel_y': float(data[1]),
                'ang_vel_z': float(data[2]),
                'lin_acc_x': float(data[3]),
                'lin_acc_y': float(data[4]),
                'lin_acc_z': float(data[5])
            }
        except:
            return {}
    
    def get_gt_data_at_time(self, timestamp):
        """Get ground truth at timestamp"""
        if self.gt_interpolator is None:
            return {}
        
        try:
            data = self.gt_interpolator(timestamp)
            return {
                'latitude': float(data[0]),
                'longitude': float(data[1]),
                'altitude': float(data[2]) if len(data) > 2 else 0.0
            }
        except:
            return {}
    
    def get_batch(self, start_idx, batch_size=5):
        """Get a batch of data for processing"""
        end_idx = min(start_idx + batch_size, len(self.image_files))
        
        batch = []
        for idx in range(start_idx, end_idx):
            try:
                image, timestamp = self.get_image_data(idx)
                imu_data = self.get_imu_data_at_time(timestamp)
                gt_data = self.get_gt_data_at_time(timestamp)
                
                batch.append({
                    'index': idx,
                    'timestamp': timestamp,
                    'image': image,
                    'imu': imu_data,
                    'ground_truth': gt_data
                })
            except Exception as e:
                print(f"Warning: Skipping image {idx}: {e}")
        
        return batch

def main():
    # Camera configuration
    camera_config = {
        'camera_matrix': {
            'cols': 3,
            'data': [456.46871015134053, 0.0, 643.3599454303429,
                     0.0, 455.40127946882507, 357.51076963739786,
                     0.0, 0.0, 1.0],
            'rows': 3
        },
        'distortion_coeffs': {
            'cols': 5,
            'data': [0.03299031731836506, -0.03150792611905064, 
                     -0.0017902177017069096, 0.00027220443810142304, 0.0],
            'rows': 1
        },
        'image_width': 1280,
        'image_height': 720
    }
    
    # Paths
    image_folder = "/home/anand/Desktop/Nicky/job_prep/Assignment/Idle_robotics/task_cv_model/train_data/drone_images"
    imu_csv_path = "/home/anand/Desktop/Nicky/job_prep/Assignment/Idle_robotics/task_cv_model/train_data/imu_data.csv"
    gt_csv_path = "/home/anand/Desktop/Nicky/job_prep/Assignment/Idle_robotics/task_cv_model/train_data/ground_truth.csv"
    geotiff_path = "/home/anand/Desktop/Nicky/job_prep/Assignment/Idle_robotics/task_cv_model/map.tif"
    
    # Check essential files
    essential_paths = [image_folder, geotiff_path]
    for path in essential_paths:
        if not Path(path).exists():
            print(f"Error: Required path not found: {path}")
            return []
    
    print("="*60)
    print("MEMORY-EFFICIENT LOCALIZATION PIPELINE")
    print("="*60)
    
    # Load data efficiently
    data_loader = EfficientDataLoader(image_folder, imu_csv_path, gt_csv_path)
    
    # Test configurations
    configs = [
        {"use_imu": False, "name": "Visual-only"},
        {"use_imu": True, "name": "IMU-assisted"}
    ]
    
    all_results = []
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"{'='*60}")
        
        # Initialize localizer for this configuration
        localizer = LightweightLocalizer(camera_config, geotiff_path, use_imu=config['use_imu'])
        
        # Process in small batches to save memory
        batch_size = 5
        total_images = len(data_loader.image_files)
        results = []
        
        for batch_start in range(0, total_images, batch_size):
            print(f"\nProcessing batch {batch_start//batch_size + 1}/{(total_images + batch_size - 1)//batch_size}")
            
            batch = data_loader.get_batch(batch_start, batch_size)
            
            for data in batch:
                print(f"\nImage {data['index'] + 1}/{total_images} (t={data['timestamp']:.3f}s)")
                
                # Localize
                estimated_gps, estimated_pixel, confidence, method, error = localizer.localize(
                    data['image'],
                    data['imu'] if config['use_imu'] else None,
                    data['ground_truth'],
                    visualize=(len(results) < 2)  # Visualize first 2
                )
                
                results.append({
                    'config': config['name'],
                    'index': data['index'],
                    'timestamp': data['timestamp'],
                    'estimated_gps': estimated_gps,
                    'estimated_pixel': estimated_pixel,
                    'ground_truth': data['ground_truth'],
                    'error_meters': error,
                    'confidence': confidence,
                    'method': method
                })
            
            # Clear memory between batches
            gc.collect()
        
        # Analyze this configuration
        successful = [r for r in results if r['error_meters'] is not None]
        if successful:
            errors = [r['error_meters'] for r in successful]
            print(f"\n{config['name']} Summary:")
            print(f"  Success rate: {len(successful)}/{len(results)}")
            print(f"  Mean error: {np.mean(errors):.2f}m")
            print(f"  Median error: {np.median(errors):.2f}m")
            print(f"  Min error: {np.min(errors):.2f}m")
            print(f"  Max error: {np.max(errors):.2f}m")
            
            # Show method distribution
            methods = [r['method'] for r in results]
            from collections import Counter
            method_counts = Counter(methods)
            print(f"  Methods used: {dict(method_counts)}")
        
        all_results.extend(results)
    
    # Final analysis
    print(f"\n{'='*60}")
    print("FINAL ANALYSIS")
    print("="*60)
    
    # Save results
    results_df = pd.DataFrame([{
        'config': r['config'],
        'index': r['index'],
        'timestamp': r['timestamp'],
        'estimated_lat': r['estimated_gps'][0] if r['estimated_gps'] else None,
        'estimated_lon': r['estimated_gps'][1] if r['estimated_gps'] else None,
        'gt_lat': r['ground_truth'].get('latitude', None),
        'gt_lon': r['ground_truth'].get('longitude', None),
        'error_meters': r['error_meters'],
        'confidence': r['confidence'],
        'method': r['method']
    } for r in all_results])
    
    results_df.to_csv('lightweight_results.csv', index=False)
    print("\nResults saved to 'lightweight_results.csv'")
    
    # Plot overall error distribution
    all_errors = [r['error_meters'] for r in all_results if r['error_meters'] is not None]
    if all_errors:
        plt.figure(figsize=(10, 6))
        plt.hist(all_errors, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(np.mean(all_errors), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_errors):.2f}m')
        plt.axvline(np.median(all_errors), color='green', linestyle='--',
                   label=f'Median: {np.median(all_errors):.2f}m')
        plt.xlabel('Localization Error (meters)')
        plt.ylabel('Frequency')
        plt.title('Overall Error Distribution (Memory-Efficient Pipeline)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return all_results

if __name__ == "__main__":
    # Set memory limits
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, hard))  # 4GB limit
    
    results = main()