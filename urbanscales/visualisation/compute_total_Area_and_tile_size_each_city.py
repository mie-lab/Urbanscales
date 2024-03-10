import csv
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import sys

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from xgboost import XGBRegressor
from shapely.wkt import loads
from shapely.wkt import loads
from shapely.geometry import Polygon as ShapelyPolygon
from sklearn.metrics import make_scorer, explained_variance_score as explained_variance_scorer, r2_score
import geopandas as gpd
import contextily as ctx
from shapely.wkt import loads
from pyproj import Geod
from shapely.geometry import Polygon as ShapelyPolygon

from shapely.wkt import loads
from pyproj import Geod
from shapely.geometry import Polygon as ShapelyPolygon
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import config
config.MASTER_VISUALISE_EACH_STEP = True


sys.path.append("../../../")
from urbanscales.preprocessing.train_data import TrainDataVectors
import pickle
import copy
from sklearn.model_selection import KFold
from smartprint import smartprint as sprint

current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.join(current_dir, '..')
os.chdir(current_dir)
sprint (os.getcwd())



class CustomUnpicklerTrainDataVectors(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module, name)



if __name__ == "__main__":
    list_of_cities = "Singapore|Zurich|Mumbai|Auckland|Istanbul|MexicoCity|Bogota|NewYorkCity|Capetown|London".split("|")
    list_of_cities_list_of_list = [
                                    # list_of_cities[:2],
                                    # list_of_cities[2:]
                                    [list_of_cities[0]],
                                    [list_of_cities[1]],
                                    [list_of_cities[2]],
                                    [list_of_cities[3]],
                                    [list_of_cities[4]],
                                    [list_of_cities[5]],
                                    [list_of_cities[6]],
                                    [list_of_cities[7]],
                                    [list_of_cities[8]],
                                    [list_of_cities[9]],
                                    # list(config.rn_city_wise_bboxes.keys())
                                ]

    tod_list_of_list = config.ps_tod_list

    common_features = [
        'betweenness',
        'circuity_avg',
        'global_betweenness',
        'k_avg',
        'lane_density',
        # 'm',
        'metered_count',
        # 'n',
        # 'non_metered_count',
        'street_length_total',
        # 'streets_per_node_count_5',
        'total_crossings'
    ]

    scale_list = config.scl_list_of_seeds

    for list_of_cities in list_of_cities_list_of_list:
        for tod_list in tod_list_of_list:
            for scale in scale_list:
                for city in list_of_cities:

                    tod = tod_list
                    x = []
                    y = []
                    fname = os.path.join(config.BASE_FOLDER, config.network_folder, city, f"_scale_{scale}_train_data_{tod}.pkl")
                    try:
                        temp_obj = CustomUnpicklerTrainDataVectors(open(fname, "rb")).load()
                        if isinstance(temp_obj.X, pd.DataFrame):
                            filtered_X = temp_obj.X[list(common_features)]
                    except:
                        # print ("Error in :")
                        # sprint(list_of_cities, tod_list, scale, city)
                        continue

                    # After processing each city and time of day, concatenate data
                    x.append(temp_obj.X)
                    y.append(temp_obj.Y)

                    # Concatenate the list of DataFrames in x and y
                    assert len(x) == 1
                    X = pd.concat(x, ignore_index=True)
                    # Convert any NumPy arrays in the list to Pandas Series
                    y_series = [pd.Series(array) if isinstance(array, np.ndarray) else array for array in y]

                    # Concatenate the list of Series and DataFrames
                    Y = pd.concat(y_series, ignore_index=True)

                    # sprint (city, scale, tod, config.shift_tile_marker, X.shape, Y.shape)

                    from shapely.geometry import box

                    bboxes = [list(i.keys())[0] for i in temp_obj.bbox_X]
                    # Compute the smallest rectangle that includes all bounding boxes
                    min_lon = min(b[3] for b in bboxes)
                    max_lon = max(b[2] for b in bboxes)
                    min_lat = min(b[1] for b in bboxes)
                    max_lat = max(b[0] for b in bboxes)
                    overall_bbox = box(min_lon, min_lat, max_lon, max_lat)


                    def compute_polygon_properties(polygon_wkt):
                        # Load the polygon from the WKT string
                        polygon = loads(polygon_wkt)

                        # Check if the polygon is valid
                        if not polygon.is_valid:
                            raise ValueError("Polygon is not valid")
                        if not isinstance(polygon, ShapelyPolygon):
                            raise TypeError("Input must be a shapely Polygon")

                        # Define WGS84 as CRS
                        geod = Geod(ellps="WGS84")

                        # Extract coordinates from the polygon and ensure it is closed
                        coords = list(polygon.exterior.coords)
                        if coords[0] != coords[-1]:
                            coords.append(coords[0])  # Close the polygon if not already closed

                        lons, lats = zip(*coords)

                        # Compute the area and perimeter
                        area, perim = geod.polygon_area_perimeter(lons, lats)
                        area_km2 = abs(area) / 1e6  # Convert from square meters to square kilometers

                        # Compute the lengths of the sides of the polygon
                        side_lengths_km = []
                        for i in range(len(coords) - 1):
                            p1 = coords[i]
                            p2 = coords[i + 1]
                            _, _, distance = geod.inv(p1[0], p1[1], p2[0], p2[1])
                            side_lengths_km.append(distance / 1000)  # Convert from meters to kilometers

                        return area_km2, side_lengths_km

                    # Compute the area of the polygon
                    area_km2, side_lengths = compute_polygon_properties(overall_bbox.wkt)

                    print(f"{city}, {scale}, Area of the polygon: {area_km2} km²")
                    print(f"{city}, {scale}, Lengths of the polygon sides: {np.mean(side_lengths)} km")

                    ar_list = []
                    sl_list = []
                    for b in bboxes:
                         #  since our order of bbox coordinates is NSEW
                        min_lon = b[3]
                        max_lon = b[2]
                        min_lat = b[1]
                        max_lat = b[0]
                        overall_bbox = box(min_lon, min_lat, max_lon, max_lat)
                        ar, sl = compute_polygon_properties(overall_bbox.wkt)
                        sl_list.extend(sl)
                        ar_list.append(ar)

                    print(f"{city}, {scale}, Mean area of bboxes {np.mean(ar_list)} km²")
                    print(f"{city}, {scale}, Lengths of the bboxes sides: {np.mean(sl_list)} km")





                    ## The output from above can be parsed and then used to create the latex table using the script below

                    ################################
                # data = """
                #    Istanbul	37	Mean area of bboxes 4.115571533348708 km²
                #    Istanbul	37	Lengths of the bboxes sides: 2.0286879125877797 km
                #    Istanbul	75	Mean area of bboxes 1.001800738276205 km²
                #    Istanbul	75	Lengths of the bboxes sides: 1.0009004187814945 km
                #    Istanbul	150	Mean area of bboxes 0.2504943918339809 km²
                #    Istanbul	150	Lengths of the bboxes sides: 0.5004944191963677 km
                #    Singapore	25	Mean area of bboxes 3.9999472154442994 km²
                #    Singapore	25	Lengths of the bboxes sides: 1.999986787354066 km
                #    Singapore	50	Mean area of bboxes 0.9999875977249344 km²
                #    Singapore	50	Lengths of the bboxes sides: 0.9999937967932919 km
                #    Singapore	100	Mean area of bboxes 0.2499970733312681 km²
                #    Singapore	100	Lengths of the bboxes sides: 0.4999970730699433 km
                #    Zurich	25	Mean area of bboxes 3.999844066642591 km²
                #    Zurich	25	Lengths of the bboxes sides: 1.9999610219341264 km
                #    Zurich	50	Mean area of bboxes 0.999991080306609 km²
                #    Zurich	50	Lengths of the bboxes sides: 0.9999955431492096 km
                #    Zurich	100	Mean area of bboxes 0.24998406139519627 km²
                #    Zurich	100	Lengths of the bboxes sides: 0.4999840626977518 km
                #    Mumbai	25	Mean area of bboxes 3.998999775241331 km²
                #    Mumbai	25	Lengths of the bboxes sides: 1.9997499349260275 km
                #    Mumbai	50	Mean area of bboxes 0.9997508102755499 km²
                #    Mumbai	50	Lengths of the bboxes sides: 0.999875406316389 km
                #    Mumbai	100	Mean area of bboxes 0.24994225888793573 km²
                #    Mumbai	100	Lengths of the bboxes sides: 0.49994226009676945 km
                #    Auckland	25	Mean area of bboxes 3.9979781272229733 km²
                #    Auckland	25	Lengths of the bboxes sides: 1.9994945478039157 km
                #    Auckland	50	Mean area of bboxes 0.9994917400354257 km²
                #    Auckland	50	Lengths of the bboxes sides: 0.9997458809795093 km
                #    Auckland	100	Mean area of bboxes 0.24987403742651654 km²
                #    Auckland	100	Lengths of the bboxes sides: 0.4998740432551434 km
                #    MexicoCity	25	Mean area of bboxes 3.9984515660881503 km²
                #    MexicoCity	25	Lengths of the bboxes sides: 1.9996128845379397 km
                #    MexicoCity	50	Mean area of bboxes 0.9995676887805294 km²
                #    MexicoCity	50	Lengths of the bboxes sides: 0.9997838463940458 km
                #    MexicoCity	100	Mean area of bboxes 0.24988796238330008 km²
                #    MexicoCity	100	Lengths of the bboxes sides: 0.49988796397351337 km
                #    Bogota	25	Mean area of bboxes 3.9998734198308337 km²
                #    Bogota	25	Lengths of the bboxes sides: 1.9999683389943226 km
                #    Bogota	50	Mean area of bboxes 0.9999702686055804 km²
                #    Bogota	50	Lengths of the bboxes sides: 0.9999851324295892 km
                #    Bogota	100	Mean area of bboxes 0.24999397302772647 km²
                #    Bogota	100	Lengths of the bboxes sides: 0.4999939728491314 km
                #    NewYorkCity	25	Mean area of bboxes 3.99797419798653 km²
                #    NewYorkCity	25	Lengths of the bboxes sides: 1.9994935670480556 km
                #    NewYorkCity	50	Mean area of bboxes 0.9992842079438672 km²
                #    NewYorkCity	50	Lengths of the bboxes sides: 0.99964211687863 km
                #    NewYorkCity	100	Mean area of bboxes 0.24980921819644158 km²
                #    NewYorkCity	100	Lengths of the bboxes sides: 0.4998092248170411 km
                #    Capetown	25	Mean area of bboxes 3.9996438793951294 km²
                #    Capetown	25	Lengths of the bboxes sides: 1.9999109829815307 km
                #    Capetown	50	Mean area of bboxes 0.9998632139456818 km²
                #    Capetown	50	Lengths of the bboxes sides: 0.9999316164422386 km
                #    Capetown	100	Mean area of bboxes 0.24994442440881634 km²
                #    Capetown	100	Lengths of the bboxes sides: 0.49994442902862585 km
                #    London	25	Mean area of bboxes 3.9996729932276485 km²
                #    London	25	Lengths of the bboxes sides: 1.9999183095043407 km
                #    London	50	Mean area of bboxes 0.9998357140348424 km²
                #    London	50	Lengths of the bboxes sides: 0.9999178830020676 km
                #    London	100	Mean area of bboxes 0.2499487804280171 km²
                #    London	100	Lengths of the bboxes sides: 0.499948791791743 km
                #    """
                #
                # # Split the data into lines for processing
                # lines = data.strip().split('\n')
                #
                # # Set up the LaTeX document
                # latex_content = r"""\documentclass{article}
                #    \usepackage{booktabs}
                #    \begin{document}
                #    \begin{table}[h]
                #    \centering
                #    \caption{Mean area and side lengths of bboxes for selected cities}
                #    \label{tab:cities}
                #    \begin{tabular}{lcc}
                #    \toprule
                #    City & Bboxes & Mean Area (km\textsuperscript{2}) & Mean Side Length (km) \\
                #    \midrule
                #    """
                #
                # # Process the data and add it to the LaTeX content
                # for line in lines:
                #     # Split each line into parts
                #     parts = line.split('\t')
                #     # Check if the line is about area or side length and process accordingly
                #     if 'Mean area' in parts[2]:
                #         city = parts[0]
                #         bboxes = parts[1]
                #         area = parts[2].split(' ')[4]
                #     else:
                #         # Side length information, we create a new row in the LaTeX table
                #         side_length = parts[2].split(' ')[5]
                #         latex_content += f"{city} & {bboxes} & {round(float(area), 3)} & {round(float(side_length), 3)} \\\\\n"
                #
                # # Finish the LaTeX document
                # latex_content += r"""\bottomrule
                #    \end{tabular}
                #    \end{table}
                #    \end{document}
                #    """
                #
                # # Output LaTeX content
                # print(latex_content)
                #
