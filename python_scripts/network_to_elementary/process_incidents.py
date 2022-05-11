import time
import pandas as pd
import matplotlib.patches
import osmnx as ox
import pickle
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import sys
from shapely import geometry
import numpy as np
import geopandas as gpd
from pyproj import Geod
from shapely.geometry import Point, LineString
import sys
import taxicab as tc


sys.path.insert(0, "/Users/nishant/Documents/GitHub/WCS/python_scripts/network_to_elementary")
from osm_to_tiles import line_to_bbox_list


def create_bbox_to_CCT(
    csv_file_name, read_from_pickle=True, N=100, folder_path="", use_route_path=False, graph_with_edge_travel_time=None
):
    """

    :param csv_file_name:
    :param read_from_pickle:
    :param N:
    :param folder_path:
    :return:
    """
    incident_data = pd.read_csv(folder_path + csv_file_name)
    print(incident_data.head())
    o_lat, o_lon, d_lat, d_lon = (
        incident_data["o_lat"].to_numpy(),
        incident_data["o_lon"].to_numpy(),
        incident_data["d_lat"].to_numpy(),
        incident_data["d_lon"].to_numpy(),
    )

    dict_bbox_to_CCT = {}
    if read_from_pickle:
        bbox_list = []
        with open(folder_path + "osm_tiles_stats_dict" + str(N) + ".pickle", "rb") as handle:
            osm_tiles_stats_dict = pickle.load(handle)

        for keyval in osm_tiles_stats_dict:
            try:
                # need to fix this messy way to read dictionary @Nishant
                key, val = list(keyval.keys())[0], list(keyval.values())[0]
                assert val != "EMPTY_STATS"
                bbox_list.append(key)
            except:
                continue

    else:
        print("This part has not been written yet")
        print("Please use the previously saved dictionary to get BBs")
        sys.exit()

    path_missing_counter = 0
    path_present_counter = 0
    for i in range(incident_data.shape[0]):
        if not use_route_path:
            bbox_intersecting = line_to_bbox_list(
                bbox_list,
                [o_lat[i], o_lon[i], d_lat[i], d_lon[i]],
                plotting_enabled=False,
                use_route_path=use_route_path,
            )
        else:
            try:

                route_linestring = tc.distance.shortest_path(
                    graph_with_edge_travel_time, (o_lat[i], o_lon[i]), (d_lat[i], d_lon[i])
                )
                bbox_intersecting = line_to_bbox_list(
                    bbox_list,
                    [o_lat[i], o_lon[i], d_lat[i], d_lon[i]],
                    plotting_enabled=False,
                    use_route_path=use_route_path,
                    route_linestring=route_linestring,
                )
                path_present_counter += 1
            except:
                print("Route not found! ")
                path_missing_counter += 1
                continue

        for bbox in bbox_intersecting:
            if bbox in dict_bbox_to_CCT:
                dict_bbox_to_CCT[bbox].append(pd.to_timedelta(incident_data["lasting_time"][i]).total_seconds())
            else:
                dict_bbox_to_CCT[bbox] = [pd.to_timedelta(incident_data["lasting_time"][i]).total_seconds()]

        if use_route_path:
            print("Route not found: ", path_missing_counter * 100 / (path_missing_counter + path_present_counter), "%")

    if use_route_path:
        print(
            "Total route not found: ", path_missing_counter * 100 / (path_missing_counter + path_present_counter), "%"
        )

    return dict_bbox_to_CCT


if __name__ == "__main__":

    dict_bbox_to_CCT = create_bbox_to_CCT(
        csv_file_name="combined_incidents_13_days.csv",
        read_from_pickle=True,
        N=50,
        folder_path="/Users/nishant/Documents/GitHub/WCS/python_scripts/network_to_elementary/",
    )

    do_nothing = True
