import time
import warnings
import networkx
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
from tqdm import tqdm


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

    for i, tq in zip(range(incident_data.shape[0]), tqdm(range(incident_data.shape[0]))):
        if not use_route_path:
            bbox_intersecting = line_to_bbox_list(
                bbox_list,
                [o_lat[i], o_lon[i], d_lat[i], d_lon[i]],
                plotting_enabled=False,
                use_route_path=use_route_path,
            )
        else:
            try:

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="Geometry is in a geographic CRS. Results from 'distance'"
                    )
                    route = tc.distance.shortest_path(
                        graph_with_edge_travel_time, (o_lat[i], o_lon[i]), (d_lat[i], d_lon[i])
                    )
                x_main, y_main = [], []
                x_first, y_first = [], []
                x_last, y_last = [], []
                if route[1]:
                    # for the meaning of route[1], route[2], route[3]
                    # see taxicab repository readMe. it is straighforward
                    #  three parts: one main, and two extensions
                    for u, v in zip(route[1][:-1], route[1][1:]):
                        # if there are parallel edges, select the shortest in length
                        data = min(graph_with_edge_travel_time.get_edge_data(u, v).values(), key=lambda d: d["length"])
                        if "geometry" in data:
                            # if geometry attribute exists, add all its coords to list
                            xs, ys = data["geometry"].xy
                            x_main.extend(xs)
                            y_main.extend(ys)
                        else:
                            # otherwise, the edge is a straight line from node to node
                            x_main.extend(
                                (graph_with_edge_travel_time.nodes[u]["x"], graph_with_edge_travel_time.nodes[v]["x"])
                            )
                            y_main.extend(
                                (graph_with_edge_travel_time.nodes[u]["y"], graph_with_edge_travel_time.nodes[v]["y"])
                            )

                # process partial edge first one
                if route[2]:
                    x_first, y_first = zip(*route[2].coords)

                # process partial edge last one
                if route[3]:
                    x_last, y_last = zip(*route[3].coords)

                lon_list = x_main + list(x_first) + list(x_last)
                lat_list = y_main + list(y_first) + list(y_last)

                XYcoord = (np.column_stack((lat_list, lon_list)).flatten()).tolist()
                # at this point, we should get lat, lon, lat, lon, ......  and so on

                assert len(XYcoord) % 2 == 0
                route_linestring = LineString(list(zip(XYcoord[0::2], XYcoord[1::2])))

                bbox_intersecting = line_to_bbox_list(
                    bbox_list,
                    [o_lat[i], o_lon[i], d_lat[i], d_lon[i]],
                    plotting_enabled=False,
                    use_route_path=use_route_path,
                    route_linestring=route_linestring,
                )
                path_present_counter += 1
            except (UnboundLocalError, IndexError, networkx.exception.NetworkXNoPath) as e:
                print("Route not found! Straight line used ")
                bbox_intersecting = line_to_bbox_list(
                    bbox_list,
                    [o_lat[i], o_lon[i], d_lat[i], d_lon[i]],
                    plotting_enabled=False,
                    use_route_path=False,
                    route_linestring=None,
                )
                path_missing_counter += 1



        for bbox in bbox_intersecting:
            if bbox in dict_bbox_to_CCT:
                dict_bbox_to_CCT[bbox].append(pd.to_timedelta(incident_data["lasting_time"][i]).total_seconds())
            else:
                dict_bbox_to_CCT[bbox] = [pd.to_timedelta(incident_data["lasting_time"][i]).total_seconds()]

        # if use_route_path:
        #     print(
        #         "Route not found counter: ",
        #         path_missing_counter * 100 / (path_missing_counter + path_present_counter),
        #         "%",
        #     )
        pass

    if use_route_path:
        print(
            "Outside loop: Total route not found: ",
            path_missing_counter * 100 / (path_missing_counter + path_present_counter),
            "%",
        )
        time.sleep(5)

    return dict_bbox_to_CCT


if __name__ == "__main__":

    dict_bbox_to_CCT = create_bbox_to_CCT(
        csv_file_name="combined_incidents_13_days.csv",
        read_from_pickle=True,
        N=50,
        folder_path="/Users/nishant/Documents/GitHub/WCS/python_scripts/network_to_elementary/",
    )

    do_nothing = True
