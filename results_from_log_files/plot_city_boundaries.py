import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box

# Define the bounding box coordinates for each city
cities = {
    "Mumbai": [19.270177, 72.979731, 18.893957, 72.776333],
    "Auckland": [-36.681247, 174.925937, -36.965932, 174.63532],
    "Istanbul": [41.671, 29.9581, 40.7289, 27.9714],
    "Mexico City": [19.592757, -98.940303, 19.048237, -99.364924],
    "Bogota": [4.837015, -73.996423, 4.4604, -74.223689],
    "New York City": [40.916178, -73.700181, 40.477399, -74.25909],
    "Cape Town": [-34.462, 18.1107, -33.3852, 19.0926]
}

# Define size of new bounding box in kilometers (half side for simplicity)
new_bbox_size_km = 25  # 50x50 sqkm so half side is 25 km
istanbul_size_km = 37.5  # 75x75 sqkm so half side is 37.5 km

# Earth's radius in kilometers (approx)
R = 6371

# Function to calculate new bounding box
def calc_new_bbox(lat, lon, size_km):
    lat_change = size_km / R
    lon_change = size_km / (R * np.cos(np.pi * lat / 180))

    # Calculate new bounding box with given size
    new_lat_min = lat - np.degrees(lat_change)
    new_lat_max = lat + np.degrees(lat_change)
    new_lon_min = lon - np.degrees(lon_change)
    new_lon_max = lon + np.degrees(lon_change)
    
    return new_lat_min, new_lon_min, new_lat_max, new_lon_max

# Calculate the new bounding boxes centered around each city's midpoint
new_bboxes = {}
for city, coords in cities.items():
    lat_min, lon_max, lat_max, lon_min = coords
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    
    # Istanbul has a different size for the bounding box
    if city == "Istanbul":
        new_bboxes[city] = calc_new_bbox(center_lat, center_lon, istanbul_size_km)
    else:
        new_bboxes[city] = calc_new_bbox(center_lat, center_lon, new_bbox_size_km)

new_bboxes

