import os.path
import sys

import geopy.distance

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import config

import folium

# Assuming rn_city_wise_bboxes is defined as in the original snippet provided by the user.

rn_city_wise_bboxes = config.rn_city_wise_bboxes

import folium
import geopy.distance
from folium import Map, Rectangle

def add_grid_lines_to_map(city_map, ne_corner, sw_corner, square_side_in_kms, city):
    # Calculate the number of lines needed based on the square side in kms
    if city == "Istanbul":
        num_lines = 75
    else:
        num_lines = 50

    # Add vertical grid lines (longitudes)
    for i in range(0, num_lines, 2):
        start_point = geopy.distance.distance(kilometers=i).destination(sw_corner, bearing=90)
        end_point = geopy.distance.distance(kilometers=i).destination(ne_corner, bearing=270)
        folium.PolyLine([(start_point.latitude, start_point.longitude), (end_point.latitude, start_point.longitude)],
                        color="black", weight=1, opacity=0.4).add_to(city_map)

    # Add horizontal grid lines (latitudes)
    for i in range(0, num_lines, 2):
        start_point = geopy.distance.distance(kilometers=i).destination(sw_corner, bearing=0)
        end_point = geopy.distance.distance(kilometers=i).destination(ne_corner, bearing=180)
        folium.PolyLine([(start_point.latitude, start_point.longitude), (start_point.latitude, end_point.longitude)],
                        color="black", weight=1, opacity=0.4).add_to(city_map)

def plot_bboxes_on_map(city_bboxes, shift_tiles=0):
    maps = {}
    for city, bbox in city_bboxes.items():
        # Calculate the original center of the bounding box
        center = geopy.Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        square_side_in_kms = 75 if city == 'Istanbul' else 50
        # Your shift logic here (omitted for brevity)

        # Calculate new NE and SW corners from the center
        half_side = square_side_in_kms / 2 ** 0.5
        ne_corner = geopy.distance.distance(kilometers=half_side).destination(center, bearing=45)
        sw_corner = geopy.distance.distance(kilometers=half_side).destination(center, bearing=225)

        # Create a map centered at the original center
        city_map = Map(location=[center.latitude, center.longitude], zoom_start=12)

        # Add a rectangle for the calculated bounding box
        Rectangle(
            bounds=[[sw_corner.latitude, sw_corner.longitude], [ne_corner.latitude, ne_corner.longitude]],
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.1  # Reduced opacity for the new bounding box
        ).add_to(city_map)

        # Add a rectangle for the original bounding box
        Rectangle(
            bounds=[[bbox[2], bbox[3]], [bbox[0], bbox[1]]],
            color="green",
            fill=True,
            fill_color="green",
            fill_opacity=0.1  # Reduced opacity for the original bounding box
        ).add_to(city_map)

        # Calculate the intersection of the two bounding boxes if any
        intersect_sw_lat = max(sw_corner.latitude, bbox[2])
        intersect_sw_lon = max(sw_corner.longitude, bbox[3])
        intersect_ne_lat = min(ne_corner.latitude, bbox[0])
        intersect_ne_lon = min(ne_corner.longitude, bbox[1])

        if intersect_ne_lat > intersect_sw_lat and intersect_ne_lon > intersect_sw_lon:
            # Add a rectangle for the intersection
            Rectangle(
                bounds=[[intersect_sw_lat, intersect_sw_lon], [intersect_ne_lat, intersect_ne_lon]],
                color="purple",
                fill=True,
                fill_color="purple",
                fill_opacity=0.2  # Higher opacity for the intersection
            ).add_to(city_map)

        # maps[city] = city_map

        add_grid_lines_to_map(city_map, ne_corner, sw_corner, square_side_in_kms, city)

        maps[city] = city_map

    return maps


# Assuming rn_city_wise_bboxes is already defined in the config
city_maps = plot_bboxes_on_map(rn_city_wise_bboxes)

# Save the maps as HTML files to open in a browser
for city, city_map in city_maps.items():
    city_map.save(f'{city}_bbox.html')
