import os
import pickle
import sys
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
from tqdm.auto import tqdm
from shapely import geometry
import config

from multiprocessing import pool as mp

from connectedregions.analysis.feature_extraction.static_features.network_features.graph_features import (
    get_stats_for_one_tile,
)
from connectedregions.preprocessing.osm.OSM_cities import (
    download_OSM_for_cities,
    get_poly_from_list_of_coords,
    get_poly,
)
from connectedregions.preprocessing.tiles.helper_files import (
    is_bounding_box_in_polygon,
    is_point_in_bounding_box,
    calculate_ground_distance,
)


def split_poly_to_bb(poly: geometry.Polygon, n, plotting_enabled=False, generate_for_perfect_fit=False, base_N=-1):
    """
    :param poly: shapely polygon
    :param n: ise used to create a list of bounding boxes; total
           number of such boxes = (n X (aspect_ratio * n) ); scaled_n is calculated in this function
    :return:
    """
    if generate_for_perfect_fit:
        # must be accompanied by the base value
        if base_N == -1:
            print("Fatal error in generate for perfect fit!\n wrong argument combination base_N not provided")
            sys.exit(0)

    min_lon, min_lat, max_lon, max_lat = poly.bounds

    vertical = calculate_ground_distance(min_lat, min_lon, max_lat, min_lon)
    horizontal = calculate_ground_distance(min_lat, min_lon, min_lat, max_lon)
    print("vertical ", vertical // 1000, " km")
    print("horizontal ", horizontal // 1000, " km")
    aspect_ratio = vertical / horizontal
    print("Aspect ratio ", aspect_ratio)

    if not generate_for_perfect_fit:
        delta_x = (max_lat - min_lat) / n
        delta_y = (max_lon - min_lon) / (n / aspect_ratio)

    elif generate_for_perfect_fit:
        delta_x = (max_lat - min_lat) / base_N
        delta_y = (max_lon - min_lon) / (base_N / aspect_ratio)
        delta_x /= n / base_N
        delta_y /= n / base_N

    bbox_list = []
    i = min_lat
    while i + delta_x <= max_lat:
        j = min_lon
        while j + delta_y <= max_lon:
            bbox_list.append((i, j, i + delta_x, j + delta_y))
            j += delta_y
        i += delta_x

    # round off everything to 5 decimal points
    for i in range(len(bbox_list)):
        bbox_list[i] = tuple([round(xx, 5) for xx in bbox_list[i]])

    if plotting_enabled:
        for bbox in bbox_list:
            lat1, lon1, lat2, lon2 = bbox
            centre_lon = 0.5 * (lon1 + lon2)
            centre_lat = 0.5 * (lat1 + lat2)
            plt.scatter(centre_lon, centre_lat, s=0.3, color="red")

            # plot rectangle
            if is_bounding_box_in_polygon(poly, bbox):
                color = "green"
            else:
                color = "red"
            plt.gca().add_patch(
                matplotlib.patches.Rectangle((lon1, lat1), lon2 - lon1, lat2 - lat1, lw=0.8, alpha=0.5, color=color)
            )

        plt.xlim([min_lon, max_lon])
        plt.ylim([min_lat, max_lat])
        plt.xlabel("latitude")
        plt.ylabel("longitude")
        plt.savefig("output_images/network_graphs/bbox_inside_polygoin.png", dpi=400)
        plt.show(block=False)

    return bbox_list


def get_box_to_nodelist_map(G_osm: ox.graph, bbox_list, scale, N):
    """
    This funciton is O(n^2) it can be made faster using hashing
    (depending on need)
    """
    filename = os.path.join(config.intermediate_files_path, "bbox_to_points_map_" + str(scale) + ".pickle")
    if os.path.isfile(filename):
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
            pickle.dump(bbox_to_points_map, f, protocol=4)

        return bbox_to_points_map


def get_OSM_tiles(bbox_list, osm_graph, N):
    """
    :param bbox_list:
    :param osm_graph:
    :return:
    """
    osm_graph_tiles = {}
    empty_count = 0
    non_empty_count = 0

    bbox_to_points_map = get_box_to_nodelist_map(
        osm_graph,
        bbox_list=bbox_list,
        scale=len(bbox_list),
        N=N,
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
                # this is not required any more because we are not using those features
                # try:
                #     iter(metric_stats)
                # except:
                #     metric_stats = len(metric_stats)

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
        plt.savefig(os.path.join(config.outputfolder, metric + "_scale_" + str(N) + ".png"), dpi=300)
        # plt.show(block=False)


def step_1_osm_tiles_to_features(
    single_city, N=50, base_N=-1, generate_for_perfect_fit=False, debug_multi_processing_error=None
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

    fname = os.path.join(
        config.intermediate_files_path, "multiple_cities", "raw_graphs_from_OSM_pickles", single_city + ".pickle"
    )
    if os.path.isfile(fname):
        with open(fname, "rb") as handle:
            G_OSM = pickle.load(handle)
    else:
        download_OSM_for_cities()

    G_OSM_dict, _error_ = get_OSM_tiles(
        osm_graph=G_OSM,
        bbox_list=split_poly_to_bb(get_poly_from_list_of_coords(get_poly(single_city)), n=N),
        N=base_N,
    )

    fname = os.path.join(config.intermediate_files_path, "osm_tiles_stats_dict" + str(N) + ".pickle")
    if os.path.isfile(fname):
        with open(fname, "rb") as handle:
            osm_tiles_stats_dict = pickle.load(handle)

    else:
        inputs = []
        for osm_tile in G_OSM_dict:
            inputs.append((G_OSM_dict[osm_tile], osm_tile))

        # multithreaded
        pool = mp.Pool(config.num_threads)
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
                        print("----------------")
                        dict_1 = list(osm_tiles_stats_dict_single_threaded[i].values())
                        dict_2 = list(osm_tiles_stats_dict_multithreaded[i].values())
                        print(dict_1)
                        print(dict_2)
                        print("----------------")
                        assert repr(dict_1) == repr(dict_2)

        osm_tiles_stats_dict = osm_tiles_stats_dict_multithreaded

        with open(fname, "wb") as f:
            pickle.dump(osm_tiles_stats_dict, f, protocol=4)

    if config.plotting_enabled:
        tile_stats_to_images(os.path.join(config.outputfolder + "output_images", "tilestats"), osm_tiles_stats_dict, N)

    return osm_tiles_stats_dict


def base_from_N(N):
    if N <= 2 or N % 2 != 0:
        return N
    return base_from_N(N // 2)


def generate_tiles_for_one_N(single_city, N, generate_for_perfect_fit=True):
    osm_tiles_stats_dict = step_1_osm_tiles_to_features(
        single_city=single_city,
        N=N,
        generate_for_perfect_fit=generate_for_perfect_fit,
        base_N=base_from_N(N),
        debug_multi_processing_error=False,
    )
    return osm_tiles_stats_dict


if __name__ == "__main__":
    # with multiprocessing.Pool(10) as p:
    #     p.map(generate_one_grid_size, list(range(170, 300, 10)))

    for base in config.base_list:
        for i in range(config.hierarchies):  # :range(60, 120, 10):
            scale = base * (2 ** i)
            if scale > 120:
                continue
            generate_tiles_for_one_N(single_city="Singapore", N=scale, generate_for_perfect_fit=True)

    last_line = "dummy"
