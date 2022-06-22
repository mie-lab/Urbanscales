import math
import multiprocessing
import os
import pickle
import sys
import warnings
import csv
import matplotlib.patches
import matplotlib.pyplot as plt
import networkx
import numpy as np
import pandas as pd
import taxicab as tc
from shapely.geometry import LineString
from smartprint import smartprint as sprint
from tqdm import tqdm

from python_scripts.network_to_elementary.osm_to_tiles import line_to_bbox_list
import config


def base_from_N(N):
    if N <= 2 or N % 2 != 0:
        return N
    return base_from_N(N // 2)


def create_bbox_to_CCT(
    csv_file_name,
    N=100,
    folder_path="",
    use_route_path=False,
    graph_with_edge_travel_time=None,
    plotting_enabled=config.plotting_enabled,
    suppress_warning=False,
):
    """

    :param csv_file_name:
    :param read_from_pickle:
    :param N:
    :param folder_path:
    :return:
    """
    incident_data = pd.read_csv(folder_path + csv_file_name)
    sprint("Before removal", incident_data.shape)

    # we noticed some incidents had the same lat-lon locaiton
    # the overall number is small (~400 out of ~7900; see Issue #53 for details), so we just drop them
    incident_data.drop(incident_data[incident_data["dist (km)"] == 0].index, inplace=True)

    # filter the lat, lons which lie outside the study region
    min_lon, max_lon, min_lat, max_lat = graph_with_edge_travel_time[1]
    incident_data.drop(incident_data[incident_data["o_lat"] > max_lat].index, inplace=True)
    incident_data.drop(incident_data[incident_data["o_lon"] > max_lon].index, inplace=True)
    incident_data.drop(incident_data[incident_data["d_lat"] > max_lat].index, inplace=True)
    incident_data.drop(incident_data[incident_data["d_lon"] > max_lon].index, inplace=True)

    incident_data.drop(incident_data[incident_data["o_lat"] < min_lat].index, inplace=True)
    incident_data.drop(incident_data[incident_data["o_lon"] < min_lon].index, inplace=True)
    incident_data.drop(incident_data[incident_data["d_lat"] < min_lat].index, inplace=True)
    incident_data.drop(incident_data[incident_data["d_lon"] < min_lon].index, inplace=True)

    sprint("After two rounds of removal", incident_data.shape)

    try:
        o_lat, o_lon, d_lat, d_lon = (
            incident_data["o_lat"].to_numpy(),
            incident_data["o_lon"].to_numpy(),
            incident_data["d_lat"].to_numpy(),
            incident_data["d_lon"].to_numpy(),
        )
    except KeyError:
        print("Header missing in pandas datafrme;\n to be specific the incidents.csv file")
        sys.exit()

    fname = config.intermediate_files_path + "osm_tiles_stats_dict" + str(N) + ".pickle"
    if not os.path.isfile(fname):
        # importing here so that we avoid the circular imports error;
        # this is to-do, should be removed at a later stage
        from python_scripts.network_to_elementary.tiles_to_elementary import (
            step_1_osm_tiles_to_features as step_1_osm_tiles_to_features,
        )

        _ = step_1_osm_tiles_to_features(
            N=N,
            plotting_enabled=False,
            n_threads=config.num_threads,
            generate_for_perfect_fit=True,
            base_N=base_from_N(N),
            debug_multi_processing_error=False,
        )

        # sys.exit()

    bbox_list = []
    with open(fname, "rb") as handle:
        osm_tiles_stats_dict = pickle.load(handle)

    for keyval in osm_tiles_stats_dict:
        try:
            # need to fix this messy way to read dictionary @Nishant
            key, val = list(keyval.keys())[0], list(keyval.values())[0]
            assert val != "EMPTY_STATS"
            bbox_list.append(key)
        except:
            continue

    paramlist = []

    for i in range(incident_data.shape[0]):

        paramlist.append(
            (
                i,
                use_route_path,
                bbox_list,
                o_lat,
                o_lon,
                d_lat,
                d_lon,
                use_route_path,
                graph_with_edge_travel_time[0],
                incident_data,
                config.use_route_path_curved,
                plotting_enabled,
                (min_lon, max_lon, min_lat, max_lat),
                suppress_warning,
            )
        )
    if config.use_route_path_curved:
        os.system("rm " + config.temp_files + "dict_*.t " + config.temp_files + "combined_file.txt")
    else:
        os.system(
            "rm "
            + config.temp_files
            + "dict_*.t "
            + config.temp_files
            + "curved_*.pickle "
            + config.temp_files
            + "combined_file.txt"
        )
        os.system("rm " + config.temp_files + "false_ODs*.txt " + config.temp_files + "true_ODs*.txt")

    # r = p_map(helper_box_to_CCT, paramlist, num_cpus=55)
    # r = process_map(helper_box_to_CCT, paramlist, max_workers=45, chunksize=1)
    #     # p = Pool(45)
    #     tqdm(p.map(helper_box_to_CCT, paramlist), total=len(paramlist))

    with multiprocessing.Pool(config.num_threads) as p:
        p.map(helper_box_to_CCT, paramlist)

    # single threaded for debug
    # for param in paramlist:
    #     helper_box_to_CCT(param)

    # combine files from each thread to a single csv file
    os.system("cat " + config.temp_files + "dict_*.t > " + config.temp_files + "combined_file.txt")
    os.system("cat " + config.temp_files + "false_ODs*.txt > " + config.temp_files + "combined_file_false_ODs")
    os.system("cat " + config.temp_files + "true_ODs*.txt > " + config.temp_files + "combined_file_true_ODs")

    if plotting_enabled:
        plot_scatter_for_true_and_false_incidents(
            config.temp_files + "combined_file_true_ODs", config.temp_files + "combined_file_false_ODs", bbox_list
        )

    dict_bbox_to_CCT = {}
    list_of_dates = []

    with open(config.temp_files + "combined_file.txt") as f:
        for row in f:
            listed = row.strip().split(";")
            bbox = eval(listed[0].strip())
            CCT = float(eval(listed[1].strip()))
            incident_start_hour = int(eval(listed[2].strip()))
            incident_start_date = str((listed[3].strip()))

            key = (bbox, incident_start_hour, incident_start_date)
            if key in dict_bbox_to_CCT:
                dict_bbox_to_CCT[key].append(CCT)
            else:
                dict_bbox_to_CCT[key] = [CCT]

            list_of_dates.append(incident_start_date)

    unique_dates_list = list(set(list_of_dates))

    # retain only the maximum for each key
    for key in dict_bbox_to_CCT:
        dict_bbox_to_CCT[key] = max(dict_bbox_to_CCT[key])

    dict_bbox_to_CCT = convert_bbox_to_CCT_new_format(dict_bbox_to_CCT, unique_dates_list)

    return dict_bbox_to_CCT, unique_dates_list


def plot_scatter_for_true_and_false_incidents(truefile, falsefile, bb_list):

    # od_lat_lon_line =  [o_lat[i], o_lon[i], d_lat[i], d_lon[i]] passed

    # bb_list: list of all bounding boxes (for each grid in the city)
    for bbox in bb_list:
        lat1, lon1, lat2, lon2 = bbox
        centre_lon = 0.5 * (lon1 + lon2)
        centre_lat = 0.5 * (lat1 + lat2)
        plt.scatter(centre_lon, centre_lat, s=3, color="black")
        # plt.gca().add_patch(
        #     matplotlib.patches.Rectangle((lon1, lat1), lon2 - lon1, lat2 - lat1, lw=0.8, alpha=0.5, color="yellow")
        # )

    with open(truefile) as f:
        for row in f:
            o_lat, o_lon, d_lat, d_lon = row.strip().split(",")
            o_lat, o_lon, d_lat, d_lon = [float(x) for x in [o_lat, o_lon, d_lat, d_lon]]
            plt.scatter(o_lon, o_lat, color="green", s=2)
            plt.scatter(d_lon, d_lat, color="green", s=2)

    with open(falsefile) as f:
        for row in f:
            o_lat, o_lon, d_lat, d_lon = row.strip().split(",")
            o_lat, o_lon, d_lat, d_lon = [float(x) for x in [o_lat, o_lon, d_lat, d_lon]]
            plt.scatter(o_lon, o_lat, color="red", s=2)
            plt.scatter(d_lon, d_lat, color="red", s=2)

    plt.xticks([])
    plt.yticks([])
    plt.show()


def convert_bbox_to_CCT_new_format(dict_bbox_to_CCT, unique_dates_list):
    """
    key: bbox
    val = 2D array ( 24 * len(unique_dates_list) )
    :return:
    """
    dict_bbox_to_CCT_new = {}

    for key in dict_bbox_to_CCT:
        # initialise each matrix with -1
        bbox, incident_start_hour, incident_start_date = key
        dict_bbox_to_CCT_new[bbox] = pd.DataFrame(
            np.random.rand(24, len(unique_dates_list)) * 0 - 1, columns=unique_dates_list
        )

    list_of_bboxes = []
    for key in dict_bbox_to_CCT:
        bbox, incident_start_hour, incident_start_date = key
        list_of_bboxes.append(bbox)
        dict_bbox_to_CCT_new[bbox].iloc[incident_start_hour][incident_start_date] = dict_bbox_to_CCT[key]

    assert len(dict_bbox_to_CCT_new) == len(set(list_of_bboxes))

    sprint(dict_bbox_to_CCT_new[list(dict_bbox_to_CCT_new.keys())[0]].shape)
    sprint(len(dict_bbox_to_CCT_new))
    sprint(len(dict_bbox_to_CCT))
    # import time
    # time.sleep(1000)

    return dict_bbox_to_CCT_new


def helper_box_to_CCT(params):
    (
        i,
        use_route_path,
        bbox_list,
        o_lat,
        o_lon,
        d_lat,
        d_lon,
        use_route_path,
        graph_with_edge_travel_time,
        incident_data,
        read_curved_paths_from_pickle,
        plotting_enabled,
        (min_lon, max_lon, min_lat, max_lat),
        suppress_warning,
    ) = params
    if not use_route_path:
        bbox_intersecting = line_to_bbox_list(
            bbox_list,
            [o_lat[i], o_lon[i], d_lat[i], d_lon[i]],
            plotting_enabled=False,
            use_route_path=use_route_path,
            gca_patch=False,
        )
        if plotting_enabled:
            lat_list = []
            lon_list = []
            for lat1, lon1, lat2, lon2 in bbox_intersecting:
                lat_list.append((lat1 + lat2) / 2)
                lon_list.append((lon1 + lon2) / 2)
            plt.scatter(lon_list, lat_list, s=0.5, color="blue")
            plt.scatter(o_lon[i], o_lat[i], s=3, color="green")
            plt.scatter(d_lon[i], d_lat[i], s=3, color="red")
            plt.xlim(min_lon, max_lon)
            plt.ylim(min_lat, max_lat)
            plt.show()

    elif use_route_path:
        try:
            # if np.random.rand() < 0.8:
            #     return

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS. Results from 'distance'")
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

            if plotting_enabled:
                plt.scatter(lon_list, lat_list, s=0.5, color="blue")
                plt.scatter(lon_list[0], lat_list[0], s=3, color="green")
                plt.scatter(lon_list[-1], lat_list[-1], s=3, color="red")
                plt.xlim(min_lon, max_lon)
                plt.ylim(min_lat, max_lat)
                plt.show()

            XYcoord = (np.column_stack((lat_list, lon_list)).flatten()).tolist()
            # at this point, we should get lat, lon, lat, lon, ......  and so on

            assert len(XYcoord) % 2 == 0

            picklefilename = config.temp_files + "curved_" + str(i) + ".pickle"
            if read_curved_paths_from_pickle:
                with open(picklefilename, "rb") as handle:
                    i, route_linestring = pickle.load(handle)
            else:
                route_linestring = LineString(list(zip(XYcoord[0::2], XYcoord[1::2])))
                with open(picklefilename, "wb") as f:
                    pickle.dump((i, route_linestring), f, protocol=4)

            bbox_intersecting = line_to_bbox_list(
                bbox_list,
                [o_lat[i], o_lon[i], d_lat[i], d_lon[i]],
                plotting_enabled=False,
                use_route_path=use_route_path,
                route_linestring=route_linestring,
            )

        except (UnboundLocalError, IndexError, networkx.exception.NetworkXNoPath, KeyError) as e:
            print("Route not found! Straight line used ")
            bbox_intersecting = line_to_bbox_list(
                bbox_list,
                [o_lat[i], o_lon[i], d_lat[i], d_lon[i]],
                plotting_enabled=False,
                use_route_path=False,
                route_linestring=None,
            )

    try:
        assert len(bbox_intersecting) >= 1
        with open(config.temp_files + "true_ODs" + str(i) + ".txt", "w") as f2:
            csvwriter = csv.writer(f2)
            csvwriter.writerow([o_lat[i], o_lon[i], d_lat[i], d_lon[i]])

    except:
        debug_pitstop = True

        if not suppress_warning:
            warnings.warn("Some lines ignored: i value: " + str(i))

        with open(config.temp_files + "false_ODs" + str(i) + ".txt", "w") as f2:
            csvwriter = csv.writer(f2)
            csvwriter.writerow([o_lat[i], o_lon[i], d_lat[i], d_lon[i]])

    for bbox in bbox_intersecting:
        with open(config.temp_files + "dict_" + str(i) + ".t", "w") as f:
            pandas_dt = pd.to_datetime(incident_data["start_time"][i]).tz_localize("utc").tz_convert("Singapore")
            f.write(
                str(bbox)
                + ";"
                + str(pd.to_timedelta(incident_data["lasting_time"][i]).total_seconds())
                + ";"
                + str(pandas_dt.hour)
                + ";"
                + str(pandas_dt._date_repr)
                + "\n"
            )
        do_nothing = True  # debug pit stop


if __name__ == "__main__":

    dict_bbox_to_CCT = create_bbox_to_CCT(
        csv_file_name="../../intermediate_files/combined_incidents_13_days.csv",
        N=50,
        folder_path=config.intermediate_files_path,
        plotting_enabled=True,
    )

    do_nothing = True
