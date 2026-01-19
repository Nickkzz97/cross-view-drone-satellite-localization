# geometry/geo_utils.py
import rasterio
from math import radians, sin, cos, sqrt, atan2

class SatelliteAligner:
    def __init__(self, geotiff_path):
        self.src = rasterio.open(geotiff_path)
        self.transform = self.src.transform

    def pixel_to_gps(self, x, y):
        lon, lat = self.transform * (x, y)
        return lat, lon

# geometry/geo_utils.py (continued)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))
