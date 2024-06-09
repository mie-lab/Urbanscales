import glob
import pickle
import sys
import time

import numpy as np

import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import config


from smartprint import smartprint as sprint
import numpy as np
import matplotlib.pyplot as plt
from shapely.wkt import loads
import skimage
from urbanscales.io.speed_data import SpeedData, Segment, SegmentList # Segment and SegmentList might be greyed out in imports but
                                                                        # but they are needed while reading the pickle
                                                                        # files
from tqdm import tqdm



# Geographic boundaries
north, west, south, east = 51.595537326042965, -0.26768329676877245, 51.37083422088536, 0.092207656541023

# Function to map geographic coordinates to array indices
def map_coords_to_indices(lat, lon, north, south, east, west, array_shape):
    """
    Maps geographic coordinates to indices in a numpy array.

    Args:
        lat (float): Latitude.
        lon (float): Longitude.
        north (float): Northern boundary of the area.
        south (float): Southern boundary of the area.
        east (float): Eastern boundary of the area.
        west (float): Western boundary of the area.
        array_shape (tuple): Shape of the numpy array.

    Returns:
        tuple: (row index, column index, status) where status indicates if the point is inside or outside the array bounds.
    """
    lat_range = north - south
    lon_range = east - west

    # Normalize latitude and longitude values
    norm_lat = (north - lat) / lat_range
    norm_lon = (lon - west) / lon_range

    # Map to array indices
    row = int(norm_lat * (array_shape[0] - 1))
    col = int(norm_lon * (array_shape[1] - 1))

    status = "point_inside"
    try:
        assert norm_lat > 0 and norm_lon > 0
        assert norm_lat < 1 and norm_lon < 1
    except:
        status = "point_outside"

    return row, col, status

# Example WKT string
city_name = "London"
sd = SpeedData(city_name, config.sd_raw_speed_data_gran, config.sd_target_speed_data_gran)
wktstring_list = list(sd.segment_jf_map.keys())

for shape_ in [5000]:# [100, 500, 2500, 5000]:
    count_outside = 0
    array = np.random.rand(shape_, shape_) + np.nan
    for counter, wkt_string in enumerate(tqdm(wktstring_list, "Iterating over linestrings to plot on the matrix")):
        # wkt_string = 'MULTILINESTRING Z ((0.36577 51.66201 0, 0.36577 51.6617 0, 0.36576 51.66139 0))'

        # Parse WKT string and plot on the array
        multiline = loads(wkt_string)

        for line in multiline.geoms:
            points = list(line.coords)
            for i in range(len(points) - 1):
                start_0, start_1, status_start = map_coords_to_indices(points[i][1], points[i][0], north, south, east, west, array.shape)
                end_0, end_1, status_end = map_coords_to_indices(points[i+1][1], points[i+1][0], north, south, east, west, array.shape)
                assert status_end in ["point_inside", "point_outside"]\
                        and status_start in ["point_inside", "point_outside"] # to ensure we don't have a new error message by mistake

                # we will ignore those end points, specifically those that exist at the boudnaries of the entire crop (50 sq.km. in this case)
                if "point_outside" in [status_start, status_end]:
                    count_outside += 1
                    break

                rr, cc = skimage.draw.line(start_0, start_1, end_0, end_1)
                array[rr, cc] = np.nanmean(list(sd.segment_jf_map.values())[counter])  # Assign a value (1) to the line pixels

    # Display the array
    plt.imshow(array, interpolation='none')
    sprint (np.nansum(array))
    plt.title(str(array.shape))
    plt.colorbar()
    plt.savefig(os.path.join(config.BASE_FOLDER, config.network_folder, city_name) +
                str(array.shape) + ".png", dpi=300)
    plt.show(block=False)
