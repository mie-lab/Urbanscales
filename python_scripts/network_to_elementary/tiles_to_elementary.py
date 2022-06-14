import multiprocessing

import numpy as np

from get_sg_osm import get_sg_poly, get_poly_from_bbox
from python_scripts.network_to_elementary.osm_to_tiles import (
    fetch_road_network_from_osm_database,
    split_poly_to_bb,
    is_point_in_bounding_box,
)
import sys, os


import os
from multiprocessing import Pool
from tqdm.auto import tqdm

# from p_tqdm import p_map
import multiprocessing as mp
import matplotlib
import matplotlib.pyplot as plt
import os
import networkx as nx
import pickle
from osmnx import truncate, utils_graph
from osmnx import stats as osxstats
import osmnx as ox


def get_box_to_nodelist_map(G_osm: ox.graph, bbox_list, scale, N, read_from_pickle=False):
    """
    This funciton is O(n^2) it can be made faster using hashing
    (depending on need)
    """
    filename = "bbox_to_points_map_" + str(scale) + ".pickle"
    if read_from_pickle:
        with open(filename, "rb") as handle:
            bbox_to_points_map = pickle.load(handle)
        return bbox_to_points_map

    else:
        bbox_to_points_map = {}
        total_nodes = len(list(G_osm.nodes))
        for node, tq in zip(
            G_osm.nodes,
            tqdm(
                range(total_nodes),
                position=((N - 170) // 10),
                desc=str(N) + "" * 30 * ((N - 170) // 10),
                leave=True,
            ),
        ):
            # for node, tq in zip(G_osm.nodes, (range(total_nodes))):

            # y is the lat, x is the lon (Out[20]: {'y': 1.2952316, 'x': 103.872544, 'street_count': 3})
            lat, lon = G_osm.nodes[node]["y"], G_osm.nodes[node]["x"]

            for bbox in bbox_list:
                if is_point_in_bounding_box(lat, lon, bb=bbox):

                    if bbox in bbox_to_points_map:
                        bbox_to_points_map[bbox].append(node)
                    else:
                        bbox_to_points_map[bbox] = [node]

        with open(filename, "wb") as f:
            pickle.dump(bbox_to_points_map, f, protocol=pickle.HIGHEST_PROTOCOL)

        return bbox_to_points_map


def get_OSM_tiles(bbox_list, osm_graph, read_from_pickle, N):
    """
    :param bbox_list:
    :param osm_graph:
    :return:
    """
    osm_graph_tiles = {}
    empty_count = 0
    non_empty_count = 0

    bbox_to_points_map = get_box_to_nodelist_map(
        osm_graph, bbox_list=bbox_list, scale=len(bbox_list), N=N, read_from_pickle=read_from_pickle
    )

    for bbox in bbox_list:

        if bbox in bbox_to_points_map:
            list_of_nodes_in_tile = bbox_to_points_map[bbox]
        else:
            # cases of bbox outside singapore
            empty_count += 1
            continue

        try:
            truncated_graph = (osm_graph.subgraph(list_of_nodes_in_tile)).copy()
            non_empty_count += 1

        except:
            empty_count += 1
            truncated_graph = "EMPTY"

        osm_graph_tiles[bbox] = truncated_graph

    percentage_of_empty_graphs = round(empty_count / (empty_count + non_empty_count) * 100, 2)
    return osm_graph_tiles, percentage_of_empty_graphs

    # '["highway"~"motorway|motorway_link|primary"]'


def get_stats_for_one_tile(input):
    """

    :param input: tuple of osm, its corresponding bounding box
    :return:
    """

    osm, bbox = input

    if osm == "EMPTY":
        stats = "EMPTY_STATS"

    else:
        spn = osxstats.count_streets_per_node(osm)
        nx.set_node_attributes(osm, values=spn, name="street_count")
        try:
            stats = ox.basic_stats(osm)
        except:
            print("stats = ox.basic_stats(osm): ", " ERROR\n Probably no edge in graph")
            stats = "EMPTY_STATS"
    return {bbox: stats}


def tile_stats_to_images(output_path: str, list_of_dict_bbox_to_stats, N):
    """

    :param output_path:
    :param dict_bbox_to_stats:
    :return:


    stats look as shown below:

    {'n': 13,
    'm': 14,
    'k_avg': 2.1538461538461537,
    'edge_length_total': 911.3389999999999,
     'edge_length_avg': 65.09564285714285,
     'streets_per_node_avg': 2.1538461538461537,
     'streets_per_node_counts': {0: 2, 1: 0, 2: 7, 3: 2, 4: 2},
     'streets_per_node_proportions': {0: 0.15384615384615385, 1: 0.0, 2: 0.5384615384615384, 3: 0.15384615384615385,
                                      4: 0.15384615384615385},
     'intersection_count': 11,
     'street_length_total': 911.3389999999998,
     'street_segment_count': 14,
     'street_length_avg': 65.09564285714285,
     'circuity_avg': 0.999997647629181,
     'self_loop_proportion': 0.0}
    """

    for kv in list_of_dict_bbox_to_stats:
        for k in kv:
            stats = kv[k]
            if stats == "EMPTY_STATS":
                continue
            metric_list = list(stats.keys())

    cmap = plt.get_cmap("gist_rainbow")
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    for metric in metric_list:
        maxVal = 0
        for kv in list_of_dict_bbox_to_stats:

            # get max val of this metric for color encoding
            for k in kv:
                stats = kv[k]
                if stats == "EMPTY_STATS":
                    continue
                metric_stats = stats[metric]

                # # managing cases of value being dict/list
                try:
                    iter(metric_stats)
                    metric_stats = len(metric_stats)
                except:
                    # do nothing
                    metric_stats = metric_stats

                maxVal = max(maxVal, metric_stats)

        cmap = plt.get_cmap("YlGnBu")

        plt.clf()
        for kv in list_of_dict_bbox_to_stats:
            for k in kv:
                stats = kv[k]
                if stats == "EMPTY_STATS":
                    continue
                bbox = k
                lat1, lon1, lat2, lon2 = bbox
                metric_stats = stats[metric]

                # # managing cases of value being dict/list
                try:
                    iter(metric_stats)
                    metric_stats = len(metric_stats)
                except:
                    # do nothing
                    metric_stats = metric_stats

                plt.scatter((lon1 + lon2) / 2, (lat1 + lat2) / 2, s=0.3, color="black")
                plt.gca().add_patch(
                    matplotlib.patches.Rectangle(
                        (lon1, lat1), lon2 - lon1, lat2 - lat1, lw=0.8, alpha=0.5, color=cmap(metric_stats / maxVal)
                    )
                )

        plt.title(metric)
        plt.clim(0, maxVal)
        plt.colorbar()
        plt.gca().set_aspect(0.66)
        plt.savefig(output_path + metric + "_scale_" + str(N) + "+.png", dpi=300)
        # plt.show(block=False)


def step_1_osm_tiles_to_features(
    read_G_from_pickle=True,
    read_osm_tiles_stats_from_pickle=False,
    n_threads=7,
    N=50,
    plotting_enabled=True,
    generate_for_perfect_fit=False,
    base_N=-1,
    debug_multi_processing_error=False,
):
    """

    :param read_G_from_pickle:
    :param read_osm_tiles_stats_from_pickle:
    :param n_threads:
    :param N:
    :param plotting_enabled:
    :return:
    """
    if generate_for_perfect_fit:
        # must be accompanied by the base value
        if base_N == -1:
            print("Fatal error in step_1_osm_tiles_to_features!\n Wrong argument combination provided")
            sys.exit(0)

    if read_G_from_pickle:
        with open("G_OSM_extracted.pickle", "rb") as handle:
            G_OSM = pickle.load(handle)
    else:
        G_OSM = fetch_road_network_from_osm_database(polygon=get_sg_poly(), network_type="drive", custom_filter=None)
        with open("G_OSM_extracted.pickle", "wb") as f:
            pickle.dump(G_OSM, f, protocol=pickle.HIGHEST_PROTOCOL)

    G_OSM_dict, _error_ = get_OSM_tiles(
        osm_graph=G_OSM,
        bbox_list=split_poly_to_bb(
            get_sg_poly(), N, plotting_enabled=False, generate_for_perfect_fit=True, base_N=base_N
        ),
        N=N,
        read_from_pickle=read_osm_tiles_stats_from_pickle,
    )

    if read_osm_tiles_stats_from_pickle:
        with open("osm_tiles_stats_dict" + str(N) + ".pickle", "rb") as handle:
            osm_tiles_stats_dict = pickle.load(handle)

    else:
        inputs = []
        for osm_tile in G_OSM_dict:
            inputs.append((G_OSM_dict[osm_tile], osm_tile))

        # multithreaded
        pool = mp.Pool(n_threads)
        osm_tiles_stats_dict_multithreaded = pool.map(get_stats_for_one_tile, inputs)

        # single threaded
        osm_tiles_stats_dict_single_threaded = []
        for i in range(len(inputs)):
            osm_tiles_stats_dict_single_threaded.append({inputs[i][1]: get_stats_for_one_tile(inputs[i])[inputs[i][1]]})

        if debug_multi_processing_error:
            assert len(osm_tiles_stats_dict_single_threaded) == len(osm_tiles_stats_dict_multithreaded)
            # osm_tiles_stats_dict_single_threaded and multi-threaded both are list of dicts with single value each

            for i in range(len(osm_tiles_stats_dict_single_threaded)):

                for j in range(len(osm_tiles_stats_dict_multithreaded)):
                    if (
                        list(osm_tiles_stats_dict_single_threaded[i].keys())[0]
                        == list(osm_tiles_stats_dict_multithreaded[i].keys())[0]
                    ):
                        dict_1 = list(osm_tiles_stats_dict_single_threaded[i].values())
                        dict_2 = list(osm_tiles_stats_dict_multithreaded[i].values())

                        if type(dict_1) is list and dict_1[0] == "EMPTY_STATS":
                            # case: "EMPTY_STATS"
                            assert dict_1[0] == dict_2[0]
                        else:
                            for key in dict_1:
                                print(dict_1)
                                if type(dict_1[key]) is not dict:
                                    assert dict_1[key] == dict_2[key]
                                # this is still not complete, because some values are dicts again!!!
                                # but no need for overkill right now

        osm_tiles_stats_dict = osm_tiles_stats_dict_multithreaded

        with open("osm_tiles_stats_dict" + str(N) + ".pickle", "wb") as f:
            pickle.dump(osm_tiles_stats_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    if plotting_enabled:
        tile_stats_to_images("output_images/tilestats/", osm_tiles_stats_dict, N)


def generate_one_grid_size(N, generate_for_perfect_fit=False, base_N=-1):

    if generate_for_perfect_fit:
        # must be accompanied by the base value
        if base_N == -1:
            print("Fatal error in generate_one_grid_size!\n Wrong argument combination provided")
            sys.exit(0)

    step_1_osm_tiles_to_features(
        read_G_from_pickle=True,
        read_osm_tiles_stats_from_pickle=False,
        N=N,
        plotting_enabled=False,
        n_threads=35,
        generate_for_perfect_fit=generate_for_perfect_fit,
        base_N=base_N,
        debug_multi_processing_error=True,
    )


if __name__ == "__main__":
    # with multiprocessing.Pool(10) as p:
    #     p.map(generate_one_grid_size, list(range(170, 300, 10)))

    # for base in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:  # , 6, 7, 8, 9, 10]:
    #     for i in range(5):  # :range(60, 120, 10):
    #         scale = base * (2 ** i)
    #         if scale > 150:
    #             continue
    #         generate_one_grid_size(N=scale, generate_for_perfect_fit=True, base_N=base)

    generate_one_grid_size(N=17, generate_for_perfect_fit=True, base_N=9)

    last_line = "dummy"
