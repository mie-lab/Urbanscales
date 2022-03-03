from get_sg_osm import get_sg_poly, get_poly_from_bbox
from python_scripts.network_to_elementary.osm_to_tiles import (
    fetch_road_network_from_osm_database,
    split_poly_to_bb,
    is_point_in_bounding_box,
)
import multiprocessing as mp
import os
import networkx as nx
import pickle
from osmnx import truncate, utils_graph
import osmnx as ox


def get_box_to_nodelist_map(G_osm: ox.graph, bbox_list, scale, read_from_pickle=False):
    """
    This funciton is O(n^2) it can be made faster using hashing
    (depending on need)
    :param G_osm:
    :param bbox_list:
    :param scale:
    :return:
    """
    filename = "bbox_to_points_map_" + str(scale) + ".pickle"
    if read_from_pickle:
        with open(filename, "rb") as handle:
            bbox_to_points_map = pickle.load(handle)
        return bbox_to_points_map

    else:
        bbox_to_points_map = {}
        count = 0
        total = len(list(G_osm.nodes))
        for node in G_osm.nodes:

            print(round(count / total * 100, 2), "%")
            count += 1

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


def get_OSM_tiles(bbox_list, osm_graph, read_from_pickle):
    """
    :param bbox_list:
    :param osm_graph:
    :return:
    """
    osm_graph_tiles = {}
    empty_count = 0
    non_empty_count = 0

    bbox_to_points_map = get_box_to_nodelist_map(osm_graph, bbox_list=bbox_list, scale=len(bbox_list), read_from_pickle=read_from_pickle)

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
        spn = utils_graph.count_streets_per_node(osm)
        nx.set_node_attributes(osm, values=spn, name="street_count")
        try:
            stats = ox.basic_stats(osm)
        except:
            print ("stats = ox.basic_stats(osm): ", " ERROR\n Probably no edge in graph")
            stats="EMPTY_STATS"
    return {bbox:stats}


def tile_to_feature(osm: ox.graph):
    if osm == "EMPTY":
        stats = "EMPTY_STATS"

    else:
        spn = ox.utils_graph.count_streets_per_node(osm)
        nx.set_node_attributes(osm, values=spn, name="street_count")
        stats = ox.basic_stats(osm)
    return stats


if __name__ == "__main__":
    read_from_pickle = True
    if read_from_pickle:
        with open("G_OSM_extracted.pickle", "rb") as handle:
            G_OSM = pickle.load(handle)
    else:
        G_OSM = fetch_road_network_from_osm_database(polygon=get_sg_poly(), network_type="drive", custom_filter=None)
        with open("G_OSM_extracted.pickle", "wb") as f:
            pickle.dump(G_OSM, f, protocol=pickle.HIGHEST_PROTOCOL)

    G_OSM_dict, _error_ = get_OSM_tiles(
        osm_graph=G_OSM, bbox_list=split_poly_to_bb(get_sg_poly(), 25, plotting_enabled=False), read_from_pickle=read_from_pickle
    )

    osm_tiles_stats_dict = {}
    inputs = []
    for osm_tile in G_OSM_dict:
        inputs.append((G_OSM_dict[osm_tile], osm_tile))

    # multithreaded
    pool = mp.Pool(7)
    osm_tiles_stats_dict = (pool.map(get_stats_for_one_tile, inputs))

    # single threaded
    # for i in inputs:
    #     osm_tiles_stats_dict[i] = f(i)

    with open("osm_tiles_stats_dict.pickle", "wb") as f:
        pickle.dump(osm_tiles_stats_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
