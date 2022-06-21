# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 16:39:45 2022

@author: yatzhang
"""
import config
import copy
import pickle
import os, sys
import random
import time
import multiprocessing as mp
import osmnx as ox
import networkx as nx
from smartprint import smartprint as sprint
import numpy as np
import shapely
from osgeo import ogr, osr, gdal
from scipy import spatial
from tqdm import tqdm
import matplotlib.pyplot as plt
from shapely import geometry
import os.path
import csv
from python_scripts.network_to_elementary.get_sg_osm import get_sg_poly
from python_scripts.network_to_elementary.osm_to_tiles import (
    fetch_road_network_from_osm_database,
    is_point_in_bounding_box,
    is_bounding_box_in_polygon,
)
from python_scripts.four_steps.steps_combined import (
    convert_to_single_statistic_by_removing_minus_1,
    generate_bbox_CCT_from_file,
)
from python_scripts.network_to_elementary.tiles_to_elementary import get_stats_for_one_tile

#

#


means = np.array(
    [
        2507.205311,
        4745.48239,
        3.323964347,
        470434.5476,
        86.78318726,
        2.497561,
        2156.193431,
        354369.9667,
        3566.963522,
        86.83845717,
        1.000000062,
        0.003754518,
    ]
)
stds = np.array(
    [
        2423.187613,
        4587.052383,
        0.832626468,
        466054.288,
        34.53542935,
        0.598464541,
        2076.314683,
        349919.1807,
        3441.744589,
        34.75293407,
        5.30583e-06,
        0.028030059,
    ]
)


def from_ogr_to_shapely_plot(list_of_three_polys, seed_i, count, convex_hull=False):
    # Creating a copy of the input OGR geometry. This is done in order to
    # ensure that when we drop the M-values, we are only doing so in a
    # local copy of the geometry, not in the actual original geometry.
    # ogr_geom_copy = ogr.CreateGeometryFromWkb(ogr_geom.ExportToIsoWkb())
    plt.clf()
    plt_set = [[seed_i, "deepskyblue", "solid"], ["epoch" + count, "tomato", "dotted"]]

    fig, ax = create_base_map(osmfilename="G_OSM.pickle")

    for i in range(len(list_of_three_polys)):
        poly = list_of_three_polys[i]
        ogr_geom_copy = ogr.CreateGeometryFromWkb(poly.ExportToWkb())

        # Dropping the M-values
        ogr_geom_copy.SetMeasured(False)

        # Generating a new shapely geometry
        shapely_geom = shapely.wkt.loads(ogr_geom_copy.ExportToWkt())

        # ax.scatter(103.82668627004688, 1.3534468386817158, s=10)
        # fig.savefig("with_basemap2.png", dpi=300)

        if shapely_geom.type == "Polygon":
            x, y = shapely_geom.exterior.xy
            if convex_hull:
                x, y = shapely_geom.convex_hull.exterior.coords.xy

            ax.plot(x, y, label=plt_set[i][0], color=plt_set[i][1], linestyle=plt_set[i][2])
        elif shapely_geom.type == "MultiPolygon":
            for m in range(len(shapely_geom)):
                _x, _y = shapely_geom[m].exterior.xy

                if convex_hull:
                    _x, _y = shapely_geom[m].convex_hull.exterior.coords.xy

                if m == 0:
                    ax.plot(_x, _y, label=plt_set[i][0], color=plt_set[i][1], linestyle=plt_set[i][2])
                else:
                    ax.plot(_x, _y, label="_" + plt_set[i][0], color=plt_set[i][1], linestyle=plt_set[i][2])
    ax.legend(loc="upper right")
    fig.savefig(config.outputfolder + "merge_plots/epoch_" + seed_i + str(count) + ".png", dpi=300)


def create_base_map(osmfilename=config.intermediate_files_path + "G_OSM_extracted.pickle"):
    with open(osmfilename, "rb") as handle:
        G = pickle.load(handle)

    fig, ax = ox.plot.plot_graph(
        G,
        ax=None,
        figsize=(12, 8),
        bgcolor="white",
        node_color="black",
        node_size=0.1,
        node_alpha=None,
        node_edgecolor="none",
        node_zorder=1,
        edge_color="black",
        edge_linewidth=0.1,
        edge_alpha=None,
        show=True,
        close=False,
        save=False,
        bbox=None,
    )
    return fig, ax


def from_ogr_to_shapely_plot_multiseeds(dict_merge, epoch, criteria_thre, convex_hull=False, base_map_enabled=False):
    # Creating a copy of the input OGR geometry. This is done in order to
    # ensure that when we drop the M-values, we are only doing so in a
    # local copy of the geometry, not in the actual original geometry.
    # ogr_geom_copy = ogr.CreateGeometryFromWkb(ogr_geom.ExportToIsoWkb())
    plt.clf()
    plt_set = [["tomato", "dotted"]]
    colors_pad = plt.cm.Set3(np.linspace(0, 1, len(dict_merge)))

    # shuffle the list of colors, so that nearby boxes are not perceptually similar colors
    random.shuffle(colors_pad)

    if base_map_enabled:
        fig, ax = create_base_map(osmfilename=config.intermediate_files_path + "G_OSM_extracted.pickle")
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    for i in range(len(dict_merge)):
        poly = dict_merge[list(dict_merge)[i]]
        ogr_geom_copy = ogr.CreateGeometryFromWkb(poly.ExportToWkb())

        # Dropping the M-values
        ogr_geom_copy.SetMeasured(False)

        # Generating a new shapely geometry
        shapely_geom = shapely.wkt.loads(ogr_geom_copy.ExportToWkt())

        if shapely_geom.type == "Polygon":
            x, y = shapely_geom.exterior.xy
            if convex_hull:
                x, y = shapely_geom.convex_hull.exterior.coords.xy

            # plt.plot(x, y, label=list(dict_merge)[i],) #color=colors_pad[i])
            ax.fill(x, y, label=list(dict_merge)[i], alpha=0.7, color=colors_pad[i])

        elif shapely_geom.type == "MultiPolygon":
            for m in range(len(shapely_geom.geoms)):
                _x, _y = shapely_geom.geoms[m].exterior.xy

                if convex_hull:
                    _x, _y = shapely_geom.geoms[m].convex_hull.exterior.coords.xy

                if m == 0:
                    # ax.plot(_x, _y, label=list(dict_merge)[i], color=colors_pad[i])
                    ax.fill(_x, _y, label=list(dict_merge)[i], alpha=0.7, color=colors_pad[i])
                else:
                    # ax.plot(_x, _y, label="_" + list(dict_merge)[i], color=colors_pad[i])
                    ax.fill(_x, _y, label="_" + list(dict_merge)[i], alpha=0.7, color=colors_pad[i])
    # plt.legend(loc="upper right", fontsize=4)
    # plt.title("epoch: " + str(epoch))
    # plt.xlim(103.6, 104.1)
    # plt.ylim(1.26, 1.45)
    # plt.gca().set_aspect(21/56)
    fig.savefig(config.outputfolder + "merge_plots/thre" + str(criteria_thre) + "_epoch" + str(epoch) + ".png", dpi=300)


def read_shpfile_SGboundary(shpfile):
    ds = ogr.Open(shpfile, 0)
    iLayer = ds.GetLayerByIndex(0)
    iFeature = iLayer.GetNextFeature()
    geometry_wkt = -1
    while iFeature is not None:
        geometry_wkt = iFeature.GetGeometryRef()
    return geometry_wkt


def is_point_in_polygon(lat, lon, shapely_poly):
    """

    Args:
        lat:
        lon:
        gdal_poly:

    Returns:

    """

    point = geometry.Point(lon, lat, 0)
    return shapely_poly.contains(point)


def create_bboxmap_file(bboxmap_file, G_OSM):
    print("bbox_to_nodes_map file NOT found; creating ...... ")
    max_lat = -1
    max_lon = -1
    min_lat = 99999999999
    min_lon = 99999999999
    for node in G_OSM.nodes:
        # y is the lat, x is the lon (Out[20]: {'y': 1.2952316, 'x': 103.872544, 'street_count': 3})
        lat, lon = G_OSM.nodes[node]["y"], G_OSM.nodes[node]["x"]
        min_lon = min(min_lon, lon)
        max_lon = max(max_lon, lon)
        min_lat = min(min_lat, lat)
        max_lat = max(max_lat, lat)

    # create list of bboxes
    delta_x = (max_lon - min_lon) / bbox_split
    delta_y = (max_lat - min_lat) / bbox_split
    bbox_list = []
    i = min_lon
    while i + delta_x <= max_lon:
        j = min_lat
        while j + delta_y <= max_lat:
            bbox_list.append((i, j, i + delta_x, j + delta_y))
            # (i, j, i + delta_x, j + delta_y) : lon_min, lat_min, lon_max, lat_max
            #  lon_max should not be confused with glbal (throughout City poly) max_lat, max_lon above

            j += delta_y
        i += delta_x

    bbox_to_nodes_map = {}
    for node in G_OSM.nodes:
        # y is the lat, x is the lon (Out[20]: {'y': 1.2952316, 'x': 103.872544, 'street_count': 3})
        lat, lon = G_OSM.nodes[node]["y"], G_OSM.nodes[node]["x"]
        for bbox in bbox_list:
            lon_min, lat_min, lon_max, lat_max = bbox
            if is_point_in_bounding_box(lat, lon, [lat_min, lon_min, lat_max, lon_max]):
                if bbox in bbox_to_nodes_map:
                    bbox_to_nodes_map[bbox].append(node)
                else:
                    bbox_to_nodes_map[bbox] = [node]

    with open(bboxmap_file, "wb") as f:
        pickle.dump(bbox_to_nodes_map, f, protocol=4)

    print("bbox_to_nodes_map file created! :)")


def convert_gdal_poly_to_shapely_poly(poly_gdal):
    """

    Args:
        poly_gdal:

    Returns:

    """
    # convert gdal to shapely polygon
    ogr_geom_copy = ogr.CreateGeometryFromWkb(poly_gdal.ExportToWkb())

    # Dropping the M-values
    ogr_geom_copy.SetMeasured(False)

    # Generating a new shapely geometry
    shapely_polygon = shapely.wkt.loads(ogr_geom_copy.ExportToWkt())

    return shapely_polygon


def get_OSM_subgraph_in_poly_fast(G_OSM, polygon_from_gdal):
    # [[[BB1_lon1, BB1_lat1], [BB1_lon2, BB1_lat2]], [[BB2_lon1, BB2_lat1], .... ]

    if os.path.isfile(bboxmap_file):
        # with open(bboxmap_file, "rb") as handle1:
        #     bbox_to_nodes_map = pickle.load(handle1)
        bbox_to_nodes_map = global_bbox_to_nodes_map
        # remap_parameters = global_remap_parameters

    else:
        create_bboxmap_file(bboxmap_file, G_OSM)

    # reduced search space
    nodes_for_subgraph = []

    shapely_polygon = convert_gdal_poly_to_shapely_poly(polygon_from_gdal)

    bounds = shapely_polygon.bounds
    lon_min_poly, lat_min_poly, lon_max_poly, lat_max_poly = bounds

    for bbox in bbox_to_nodes_map:
        lon_min, lat_min, lon_max, lat_max = bbox

        #  first we reduce search space by reducing the polygon to BBox
        # classic range overlap problem; we extend it to 2-D
        if not (
            (lon_min <= lon_max_poly and lon_min_poly <= lon_max)
            and (lat_min <= lat_max_poly and lat_min_poly <= lat_max)
        ):
            continue

        contains_result = is_bounding_box_intersecting_polygon(shapely_polygon, bbox)
        if contains_result == "partial":
            # same order inside is_bounding_box_in_polygon
            # i.e. lon_min, lat_min, lon_max, lat_max = bbox

            for node in bbox_to_nodes_map[bbox]:
                lat, lon = G_OSM.nodes[node]["y"], G_OSM.nodes[node]["x"]

                # if is_point_in_bounding_box(lat, lon, bb=[lat_min, lon_min, lat_max, lon_max]):
                if is_point_in_polygon(lat, lon, shapely_polygon):
                    nodes_for_subgraph.append(node)

        elif contains_result == "all":
            # no need to check specific nodes; we can just add all of them
            # and expand the list of nodes
            nodes_for_subgraph = nodes_for_subgraph + bbox_to_nodes_map[bbox]

    subgraph = G_OSM.subgraph(nodes_for_subgraph).copy()
    return subgraph, nodes_for_subgraph


def is_bounding_box_intersecting_polygon(shapely_polygon, bbox):
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    """
    lon_min, lat_min, lon_max, lat_max = bbox

    # complex case, when BB corner is not inside the polygon
    bb_polygon = geometry.Polygon(
        [[lon_min, lat_min], [lon_max, lat_min], [lon_max, lat_max], [lon_min, lat_max], [lon_min, lat_min]]
    )
    if shapely_polygon.contains(bb_polygon):
        return "all"
    elif bb_polygon.intersects(shapely_polygon):
        return "partial"
    else:
        return False


def get_OSM_subgraph_in_poly(G_OSM, polygon_from_gdal):
    # [[[BB1_lon1, BB1_lat1], [BB1_lon2, BB1_lat2]], [[BB2_lon1, BB2_lat1], .... ]

    nodes_for_subgraph = []

    shapely_poly = convert_gdal_poly_to_shapely_poly(poly_gdal=polygon_from_gdal)

    for node in G_OSM.nodes:
        # y is the lat, x is the lon (Out[20]: {'y': 1.2952316, 'x': 103.872544, 'street_count': 3})
        lat, lon = G_OSM.nodes[node]["y"], G_OSM.nodes[node]["x"]

        # if is_point_in_bounding_box(lat, lon, bb=[lat_min, lon_min, lat_max, lon_max]):
        if is_point_in_polygon(lat, lon, shapely_poly=shapely_poly):
            nodes_for_subgraph.append(node)

    subgraph = G_OSM.subgraph(nodes_for_subgraph).copy()
    return subgraph


def convert_stats_to_vector(stats):
    if stats == "EMPTY_STATS":
        return stats
    single_vector = []
    for key_2 in (
        "n",
        "m",
        "k_avg",
        "edge_length_total",
        "edge_length_avg",
        "streets_per_node_avg",
        "intersection_count",
        "street_length_total",
        "street_segment_count",
        "street_length_avg",
        "circuity_avg",
        "self_loop_proportion",
    ):
        single_vector.append(stats[key_2])
    single_vector = np.array(single_vector)

    assert single_vector.shape == (12,)
    return single_vector


def compute_local_criteria(
    polygon_1,
    polygon_2,
    read_G_osm_from_pickle=True,
    bbox_to_points_map=None,
    a=0.5,
    b=0.5,
    loss_merge="sum",
    debug=False,
):
    """
    compute the local criteria between two neighbouring polygons
        Assume f = a*similarity + b*connectivity, where a and b are constants
        similarity: distance of feature variable vectors between two polygons
        connectivity: whether these two polygones are connected via OSM network

    Parameters
    ----------
    polygon_1 : @Nishant has assumed polygon1 and polygon2 to be a similar format as discussed before
     (i.e. list of BBs)
                # format of list_of_BBs;
                # [[[BB1_lon1, BB1_lat1], [BB1_lon2, BB1_lat2]], [[BB2_lon1, BB2_lat1], ......... ]
                polygon_1 is the seed zone
                polygon_2 is being assessed for merging/ not merging
    polygon_2 : TYPE
        DESCRIPTION.

    loss_merge: default "sum"

    Returns
    -------
    criteria_value : float
        local criteria value between two polygons
        the smaller, the better

    """
    if read_G_osm_from_pickle:
        G_OSM = global_G_OSM
    else:
        G_OSM = fetch_road_network_from_osm_database(polygon=get_sg_poly(), network_type="drive", custom_filter=None)
        with open(config.intermediate_files_path + "G_OSM_extracted.pickle", "wb") as f:
            pickle.dump(G_OSM, f, protocol=4)

    global global_map_of_polygon_to_features

    if debug:
        starttime = time.time()
        subgraph_1_slow = get_OSM_subgraph_in_poly(G_OSM, polygon_1)
        subgraph_2_slow = get_OSM_subgraph_in_poly(G_OSM, polygon_2)
        old_time = time.time() - starttime

        starttime = time.time()
        subgraph_1_fast, _ = get_OSM_subgraph_in_poly_fast(G_OSM, polygon_1)
        subgraph_2_fast, _ = get_OSM_subgraph_in_poly_fast(G_OSM, polygon_2)
        new_time = time.time() - starttime

        print("Speed-up: ", round(old_time / new_time, 2), "X faster")

        # sprint(len(list(subgraph_1_fast.nodes)))
        # sprint(len(list(subgraph_1_slow.nodes)))
        assert nx.is_isomorphic(subgraph_1_slow, subgraph_1_fast)
        assert nx.is_isomorphic(subgraph_2_slow, subgraph_2_fast)

    if polygon_1 in global_map_of_polygon_to_features:
        stats_vector_1, nodes_1 = global_map_of_polygon_to_features[polygon_1]
    else:
        subgraph_1_fast, nodes_1 = get_OSM_subgraph_in_poly_fast(G_OSM, polygon_1)
        stats_vector_1 = convert_stats_to_vector(get_stats_for_one_tile([subgraph_1_fast]))
        global_map_of_polygon_to_features[polygon_1] = stats_vector_1, nodes_1

    if polygon_2 in global_map_of_polygon_to_features:
        stats_vector_2, nodes_2 = global_map_of_polygon_to_features[polygon_2]
    else:
        subgraph_2_fast, nodes_2 = get_OSM_subgraph_in_poly_fast(G_OSM, polygon_2)
        stats_vector_2 = convert_stats_to_vector(get_stats_for_one_tile([subgraph_2_fast]))
        global_map_of_polygon_to_features[polygon_2] = stats_vector_2, nodes_2

    subgraph_combined = G_OSM.subgraph(nodes_1 + nodes_2).copy()
    stats_vector_combined = convert_stats_to_vector(get_stats_for_one_tile([subgraph_combined]))

    # This extra test needed because we have some extra bboxes (empty graphs in the get_OSM_subgraph_in_poly_fast (FAST))
    # function
    if type(stats_vector_1) == str or type(stats_vector_2) == str or type(stats_vector_combined) == str:
        return 1  # 1 implies merge not going ahead

    # add NaN filter
    if np.isnan(stats_vector_1).any() or np.isnan(stats_vector_2).any() or np.isnan(stats_vector_combined).any():
        return 1

    # FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
    # disagreement between numpy as string comparison

    # with open("save_vectors.csv", "a") as f:
    #     csvwriter = csv.writer(f)
    #     csvwriter.writerow(stats_vector_1.flatten().tolist())
    #     csvwriter.writerow(stats_vector_2.flatten().tolist())
    #     csvwriter.writerow(stats_vector_combined.flatten().tolist())

    assert stats_vector_1.shape == stats_vector_2.shape == stats_vector_combined.shape == (12,)

    # https://github.com/gboeing/osmnx/blob/997facb88ac566ccf79227a13b86f2db8642d04a/osmnx/stats.py#L339
    # m refers to edge count; It is the second value in our vector
    edge_count_1 = stats_vector_1[1]
    edge_count_2 = stats_vector_2[1]
    edge_count_combined = stats_vector_combined[1]

    new_edges = edge_count_combined - (edge_count_1 + edge_count_2)

    if debug:
        sprint(edge_count_1, edge_count_2, edge_count_combined, new_edges)

    if new_edges == 0:
        return 1

    # normalise using the mean and variance for each column;
    # the means and stds were computed using around 14K values during previous runs
    stats_vector_1 = (stats_vector_1 - means) / stds
    stats_vector_2 = (stats_vector_2 - means) / stds

    ###########################
    # a = 0
    a = 0.75
    f_sim = a * spatial.distance.cosine(stats_vector_1, stats_vector_2)
    f_conn = b * (1 / new_edges)

    with open("save_f_conn_f_sim.csv", "a") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow([f_conn, f_sim, f_conn / f_sim])

    if loss_merge == "sum":
        if debug:
            sprint(f_sim, f_conn)
        criteria_value = f_sim + f_conn

    return criteria_value


def get_combined_bbox_dict(scales=[], folder_path=""):
    """
    creates a single dictionary with key = bbox
    from all bbox, hour, date to CCT
    """
    if len(scales) == 0:
        print("Scale list is empty; Wrong input; exiting code")
        sys.exit(0)

    combined_dict = {}
    for scale in scales:
        fname = folder_path + "dict_bbox_hour_date_to_CCT" + str(scale) + ".pickle"
        if not os.path.isfile(fname):
            generate_bbox_CCT_from_file(N=scale, use_route_path=config.use_route_path_curved)

        with open(fname, "rb") as f1:
            dict_bbox_hour_date_to_CCT = pickle.load(f1)

        combined_dict = {**combined_dict, **dict_bbox_hour_date_to_CCT}

    return combined_dict


def get_single_Y_for_polygon(combined_dict, polygon_bbox_list, timefilter, method_for_single_statistic):
    """
    combined_dict, polygon, timefilter, method_for_single_statistic
    """

    # polygon must be a list of bboxes

    polygon_list_of_bbox = []
    for bbox_wkt in polygon_bbox_list:
        if bbox_wkt == []:
            continue
        shapely_geom = shapely.wkt.loads(bbox_wkt.ExportToWkt())
        lat_list = shapely_geom.exterior.xy[1]
        lon_list = shapely_geom.exterior.xy[0]
        key = min(lat_list), min(lon_list), max(lat_list), max(lon_list)
        polygon_list_of_bbox.append(key)

    # polysubset_dict = [combined_dict[bbox] for bbox in polygon_list_of_bbox]

    polysubset_dict = {}
    for key in polygon_list_of_bbox:
        if key in combined_dict:
            polysubset_dict[key] = combined_dict[key]

    polysubset_dict_k_bbox_value_scalar = convert_to_single_statistic_by_removing_minus_1(
        dict_bbox_hour_date_to_CCT=polysubset_dict,
        timefilter=timefilter,
        method_for_single_statistic=method_for_single_statistic,
    )

    return np.sum(list(polysubset_dict_k_bbox_value_scalar.values()))


def get_single_X_for_polygon(polygon_from_gdal, G_OSM):
    """
    polygon_from_gdal, G_OSM
    """
    subgraph = get_OSM_subgraph_in_poly(G_OSM, polygon_from_gdal)
    X = convert_stats_to_vector(get_stats_for_one_tile([subgraph]))

    return X


def compute_local_criteria_random(polygon_1, polygon_2):  # tbd
    """
    compute the local criteria between two neighbouring polygons
        Assume f = a*similarity + b*connectivity, where a and b are constants
        similarity: distance of feature variable vectors between two polygons
        connectivity: whether these two polygones are connected via OSM network

    Parameters
    ----------
    polygon_1 : TYPE
        DESCRIPTION.
    polygon_2 : TYPE
        DESCRIPTION.

    Returns
    -------
    criteria_value : float
        local criteria value between two polygons
        the smaller, the better

    """
    # criteria_value = 0
    # return criteria_value
    return random.random()


def bbox_ogr_polygon(bbox_coords):
    """
    convert the two coordinates of bbox into ogr.wkbPolygon

    Parameters
    ----------
    bbox_coords : list of coordinates
        list[[lng1, lat1],[lng2, lat2]].

    Returns
    -------
    ogr_polygon : ogr.wkbPolygon
        in this format, it can be directly used in topological computation

    """
    # print('output:', bbox_coords)
    # print('test')
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(bbox_coords[0][0], bbox_coords[0][1])  # lng1, lat1
    ring.AddPoint(bbox_coords[0][0], bbox_coords[1][1])  # lng1, lat2
    ring.AddPoint(bbox_coords[1][0], bbox_coords[1][1])  # lng2, lat2
    ring.AddPoint(bbox_coords[1][0], bbox_coords[0][1])  # lng2, lat1
    ring.AddPoint(bbox_coords[0][0], bbox_coords[0][1])  # lng1, lat1
    ogr_polygon = ogr.Geometry(ogr.wkbPolygon)
    ogr_polygon.AddGeometry(ring)
    return ogr_polygon


def identify_bbox_usage(dict_bbox_select):
    """
    identify whether all selected bboxes has been marked as False

    Parameters
    ----------
    dict_bbox_select : dict{bbox_ogrstring: bool flag}
        dictionary of all bboxes that need to be considered in the merge process.

    Returns
    -------
    bool
        bool flag that whether all selected bboxes has been marked as False

    """
    for bbox_i in dict_bbox_select:
        if dict_bbox_select[bbox_i]:
            return True
    return False


def identify_bbox_usage_num(dict_bbox_select):
    num = 1
    for bbox_i in dict_bbox_select:
        if dict_bbox_select[bbox_i]:
            num += 1
    return num


def hierarchical_region_merging_oneseed(
    bbox_file, island_file, seed_file, merged_shpfile, criteria_thre
):  # input_file is not used here
    """
    implement hierarchial region merging process for each tree, each region is limited by its boundary

    Returns
    -------
    merged results of each island for each tree

    """

    # read bbox_file, a set of bbox file in multi-hierarchies
    # dict_bbox = {'hierarchy_1':[[]], 'hierarchy_2':[[]], 'hierarchy_3':[[]]}
    with open(bbox_file, "rb") as handle1:
        dict_bbox = pickle.load(handle1)

    # The keys of these three dictionaries are the same
    # read the list of bbox in each island in the best-fit hierarchy
    # dict_islands = {'island_1':[], 'island_2':[], 'island_3':[], 'island_4':[]}
    with open(island_file, "rb") as handle2:
        dict_islands = pickle.load(handle2)
    # read start-seed (bbox) in each island in the best-fit hierarchy
    # dict_seeds = {'island_1':[], 'island_2':[], 'island_3':[], 'island_4':[]}
    with open(seed_file, "rb") as handle3:
        dict_seeds = pickle.load(handle3)
    # store the merge result
    dict_merge = {}

    # implement hierarchial region merge for each seed
    for seed_i in tqdm(dict_seeds, desc="Processing island"):
        seed_bh = dict_seeds[seed_i]
        island_bh = dict_islands[seed_i]

        # merge the separate bbox in this island into a whole polygon
        whole_island = ogr.Geometry(ogr.wkbPolygon)

        for bbox_i in island_bh:
            bbox_ogr = bbox_ogr_polygon(bbox_i)
            whole_island = whole_island.Union(bbox_ogr)

        # identify all bbox in multi-hierarchies that within with the boundary in the best-fit hierarchy
        dict_bbox_select = {}  # bbox: flag
        for hierarchy_i in dict_bbox:
            bbox_eh = dict_bbox[hierarchy_i]
            for bbox_i in bbox_eh:
                # convert bbox_eh to ogr_string
                bbox_ogr = bbox_ogr_polygon(bbox_i)

                if whole_island.Contains(bbox_ogr):
                    dict_bbox_select[bbox_ogr] = True
        print("total number:", len(dict_bbox_select))

        # begin region merging
        seed_zone = bbox_ogr_polygon(seed_bh)
        from_ogr_to_shapely_plot([whole_island, seed_zone], seed_i, "_" + str(0))

        # stop merging when all bbox has been marked as False
        count = 0
        while identify_bbox_usage(dict_bbox_select):
            count += 1
            touch_count = 0
            # find all neibhouring bbox intersects with seed region, labeled as False
            for select_i in dict_bbox_select:
                if dict_bbox_select[select_i] == True and seed_zone.Overlaps(select_i):
                    dict_bbox_select[select_i] = False

            # find the minimum local_criteria and its bbox among all bboxes that it touches
            min_criteria = sys.maxsize
            min_merge_bbox = []
            _flag = False
            for select_i in dict_bbox_select:
                if dict_bbox_select[select_i] == True and seed_zone.Touches(select_i):
                    touch_count += 1
                    tmp_criteria = compute_local_criteria(seed_zone, select_i, read_G_osm_from_pickle=False)
                    # if tmp_criteria is too large, indicate as false
                    if tmp_criteria > criteria_thre:
                        dict_bbox_select[select_i] = False
                        continue
                    if tmp_criteria < min_criteria:
                        min_criteria = tmp_criteria
                        min_merge_bbox = select_i
                        _flag = True

            # merge seed_zone with min_merge_bbox
            if _flag:
                dict_bbox_select[min_merge_bbox] = False
                seed_zone = seed_zone.Union(min_merge_bbox)
                from_ogr_to_shapely_plot([whole_island, seed_zone], seed_i, "_" + str(count))
                print(
                    "Epcoh: {}. Available bbox: {}. Touch bbox: {}.".format(
                        count, identify_bbox_usage_num(dict_bbox_select), touch_count
                    )
                )
            # stop iterating when no bbox touching with seed_zone
            if touch_count == 0:
                print(
                    "Epcoh: {}. Available bbox: {}. Touch bbox: {}.".format(
                        count, identify_bbox_usage_num(dict_bbox_select), touch_count
                    )
                )
                break

        # merge result
        dict_merge[seed_i] = seed_zone

    # output it as shapefile result
    os.environ["SHAPE_ENCODING"] = "utf-8"
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.access(merged_shpfile, os.F_OK):
        driver.DeleteDataSource(merged_shpfile)
    newds = driver.CreateDataSource(merged_shpfile)
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    layernew = newds.CreateLayer("line", srs, ogr.wkbPolygon)

    field_PC = ogr.FieldDefn("island", ogr.OFTString)
    field_PC.SetWidth(30)
    layernew.CreateField(field_PC)

    for i_zone in dict_merge:
        feat = ogr.Feature(layernew.GetLayerDefn())
        feat.SetGeometry(dict_merge[i_zone])
        feat.SetField("island", i_zone)
        layernew.CreateFeature(feat)
        feat.Destroy()
    newds.Destroy()


def output_epoch_shp(dict_merge, merged_shapefile, epoch):
    tmp_shapefile = merged_shapefile[:-4] + "_epoch" + str(epoch) + ".shp"

    # output it as shapefile result
    os.environ["SHAPE_ENCODING"] = "utf-8"
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.access(tmp_shapefile, os.F_OK):
        driver.DeleteDataSource(tmp_shapefile)
    newds = driver.CreateDataSource(tmp_shapefile)
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    layernew = newds.CreateLayer("line", srs, ogr.wkbPolygon)

    field_PC = ogr.FieldDefn("island", ogr.OFTString)
    field_PC.SetWidth(30)
    layernew.CreateField(field_PC)

    for i_zone in dict_merge:
        feat = ogr.Feature(layernew.GetLayerDefn())
        feat.SetGeometry(dict_merge[i_zone])
        feat.SetField("island", i_zone)
        layernew.CreateFeature(feat)
        feat.Destroy()
    newds.Destroy()


def hierarchical_region_merging_multiseeds(
    bbox_file, island_file, seed_file, merged_shpfile, criteria_thre, shp_epoch
):  # input_file is not used here
    """
    implement hierarchial region merging process for each tree, multi-seeds growing together, no island boundary limitation

    Returns
    -------
    merged results of each island for each tree

    """

    # compute GoF for merged islands
    combined_dict = get_combined_bbox_dict(
        scales=[
            5,
            10,
            20,
        ],
        folder_path=config.intermediate_files_path,
    )
    timefilter = [5, 6, 7, 8]

    # read bbox_file, a set of bbox file in multi-hierarchies
    with open(bbox_file, "rb") as handle1:
        dict_bbox = pickle.load(handle1)

    with open(island_file, "rb") as handle2:
        dict_islands_details_test = pickle.load(handle2)

    # merge island bboxes
    dict_islands = {}
    dict_islands_details = {}
    for seed_i in dict_islands_details_test:
        one_island = dict_islands_details_test[seed_i]
        one_island_ogr = []
        whole_island = ogr.Geometry(ogr.wkbPolygon)
        for bbox_i in one_island:
            bbox_ogr = bbox_ogr_polygon(bbox_i)
            one_island_ogr.append(bbox_ogr)
            whole_island = whole_island.Union(bbox_ogr)
        dict_islands[seed_i] = whole_island
        dict_islands_details[seed_i] = one_island_ogr

    X = []
    Y = []
    for seed_i in dict_islands_details:

        stats = get_single_X_for_polygon(dict_islands[seed_i], global_G_OSM)
        if type(stats) == str and stats == "EMPTY_STATS":
            continue
        Y.append(
            get_single_Y_for_polygon(
                combined_dict,
                timefilter=timefilter,
                polygon_bbox_list=dict_islands_details[seed_i],
                method_for_single_statistic="sum",
            )
        )
        X.append((get_single_X_for_polygon(dict_islands[seed_i], global_G_OSM)).flatten().tolist())
    print(X)
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape, Y.shape)
    with open("islands_X_Y", "wb") as f:
        pickle.dump([X, Y], f, protocol=4)

    # The keys of two dictionaries are the same
    with open(seed_file, "rb") as handle3:
        dict_seeds = pickle.load(handle3)

    print(dict_seeds)

    # store the merge result
    dict_merge = copy.deepcopy(dict_seeds)
    dict_merge_details = copy.deepcopy(dict_seeds)

    for seed_i in dict_merge:
        seed_bh = dict_merge[seed_i]
        dict_merge[seed_i] = bbox_ogr_polygon(seed_bh)
        dict_merge_details[seed_i] = [bbox_ogr_polygon(seed_bh)]

    # identify all bbox in multi-hierarchies
    dict_bbox_select = {}  # bbox: flag
    for hierarchy_i in dict_bbox:
        bbox_eh = dict_bbox[hierarchy_i]
        for bbox_i in bbox_eh:
            # convert bbox_eh to ogr_string
            bbox_ogr = bbox_ogr_polygon(bbox_i)
            dict_bbox_select[bbox_ogr] = True
    print("total number:", len(dict_bbox_select))

    for seed_i in dict_merge:
        seed_zone = dict_merge[seed_i]
        # find all neibhouring bbox Overlapping with seed region, labeled as False
        for select_i in dict_bbox_select:
            if dict_bbox_select[select_i] == True and (
                seed_zone.Overlaps(select_i)
                or seed_zone.Crosses(select_i)
                or seed_zone.Contains(select_i)
                or seed_zone.Within(select_i)
            ):
                dict_bbox_select[select_i] = False

    # implement hierarchial region merge
    epoch = 0
    from_ogr_to_shapely_plot_multiseeds(dict_merge, epoch, criteria_thre)
    while identify_bbox_usage(dict_bbox_select):
        epoch += 1
        touch_count = 0
        for seed_i in dict_merge:
            seed_zone = dict_merge[seed_i]

            # find all neibhouring bbox Overlapping with seed region, labeled as False
            for select_i in dict_bbox_select:
                if dict_bbox_select[select_i] == True and (
                    seed_zone.Overlaps(select_i)
                    or seed_zone.Crosses(select_i)
                    or seed_zone.Contains(select_i)
                    or seed_zone.Within(select_i)
                ):
                    dict_bbox_select[select_i] = False

            # find the minimum local_criteria and its bbox among all bboxes that it touches
            min_criteria = sys.maxsize
            min_merge_bbox = []
            _flag = False
            for select_i in dict_bbox_select:
                if dict_bbox_select[select_i] == True and seed_zone.Touches(select_i):
                    touch_count += 1
                    tmp_criteria = compute_local_criteria(seed_zone, select_i, read_G_osm_from_pickle=True)
                    # if tmp_criteria is too large, indicate as false
                    if tmp_criteria >= criteria_thre:
                        dict_bbox_select[select_i] = False
                        continue
                    if tmp_criteria < min_criteria:
                        min_criteria = tmp_criteria
                        min_merge_bbox = select_i
                        _flag = True

            # merge seed_zone with min_merge_bbox
            if _flag:
                dict_bbox_select[min_merge_bbox] = False
                seed_zone = seed_zone.Union(min_merge_bbox)
            # merge result
            dict_merge[seed_i] = seed_zone
            dict_merge_details[seed_i].append(min_merge_bbox)

            # find all neibhouring bbox Overlapping with seed region, labeled as False
            for select_i in dict_bbox_select:
                if dict_bbox_select[select_i] == True and (
                    seed_zone.Overlaps(select_i)
                    or seed_zone.Crosses(select_i)
                    or seed_zone.Contains(select_i)
                    or seed_zone.Within(select_i)
                ):
                    dict_bbox_select[select_i] = False

        print(
            "Epoch: {}. Available bbox: {}. Touch bbox: {}.".format(
                epoch, identify_bbox_usage_num(dict_bbox_select), touch_count
            )
        )
        from_ogr_to_shapely_plot_multiseeds(dict_merge, epoch, criteria_thre)
        # stop iterating when no bbox touching with seed_zone
        if touch_count == 0:
            print(
                "Epoch: {}. Available bbox: {}. Touch bbox: {}.".format(
                    epoch, identify_bbox_usage_num(dict_bbox_select), touch_count
                )
            )
            break

        if epoch % shp_epoch == 0:
            output_epoch_shp(dict_merge, merged_shpfile, epoch)

    # output it as shapefile result

    method_for_single_statistic = "sum"
    X = []
    Y = []

    for seed_i in dict_merge_details:
        Y.append(
            get_single_Y_for_polygon(
                combined_dict,
                timefilter=timefilter,
                polygon_bbox_list=dict_merge_details[seed_i],
                method_for_single_statistic="sum",
            )
        )
        X.append(get_single_X_for_polygon(dict_merge[seed_i], global_G_OSM))

    X = np.array(X)
    Y = np.array(Y)
    print(X.shape, Y.shape)
    with open("post_merge_X_Y", "wb") as f:
        pickle.dump([X, Y], f, protocol=4)

    os.environ["SHAPE_ENCODING"] = "utf-8"
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.access(merged_shpfile, os.F_OK):
        driver.DeleteDataSource(merged_shpfile)
    newds = driver.CreateDataSource(merged_shpfile)
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    layernew = newds.CreateLayer("line", srs, ogr.wkbPolygon)

    field_PC = ogr.FieldDefn("island", ogr.OFTString)
    field_PC.SetWidth(30)
    layernew.CreateField(field_PC)

    for i_zone in dict_merge:
        feat = ogr.Feature(layernew.GetLayerDefn())
        feat.SetGeometry(dict_merge[i_zone])
        feat.SetField("island", i_zone)
        layernew.CreateFeature(feat)
        feat.Destroy()
    newds.Destroy()


def main_func(thre):
    hierarchical_region_merging_multiseeds(
        config.intermediate_files_path + "dict_bbox_" + str(config.base) + "_.pickle",
        config.intermediate_files_path + "dict_islands_" + str(config.best_fit_hierarchy) + "_.pickle",
        config.intermediate_files_path + "dict_seeds_" + str(config.best_fit_hierarchy) + "_.pickle",
        config.outputfolder + "output_thre" + str(thre) + ".shp",
        thre,
        10,
    )


global_map_of_polygon_to_features = {}

with open(config.intermediate_files_path + "G_OSM_extracted.pickle", "rb") as handle:
    global_G_OSM = pickle.load(handle)

bboxmap_file = config.intermediate_files_path + "bbox_to_OSM_nodes_map.pickle"
bbox_split = config.bbox_split_for_merge_optimisation

if os.path.isfile(bboxmap_file):
    with open(bboxmap_file, "rb") as handle1:
        global_bbox_to_nodes_map = pickle.load(handle1)
else:
    create_bboxmap_file(bboxmap_file, global_G_OSM)


if __name__ == "__main__":
    # np.seterr(divide='ignore', invalid='ignore')  # used to ignore runtime running
    # os.environ['PROJ_LIB'] = r'C:\Users\yatzhang\Anaconda3\envs\trafficenv\Library\share\proj'
    # os.environ['GDAL_DATA'] = r'C:\Users\yatzhang\Anaconda3\envs\trafficenv\Library\share'

    # hierarchical_region_merging_oneseed('./urban_merge/dict_bbox_5_.pickle',
    #                                     './urban_merge/dict_islands_2_.pickle',
    #                                     './urban_merge/dict_seeds_2_.pickle',
    #                                     './urban_merge/output.shp',
    #                                     0.75)

    os.system("rm -rf " + config.outputfolder + "merge_plots")
    os.system("mkdir " + config.outputfolder + "merge_plots")

    os.system("rm " + config.outputfolder + "save_vectors.csv")
    os.system("rm " + config.outputfolder + "save_f_conn_f_sim.csv")

    # single threads
    # thre = 0.75
    # Tip: the final parameter means to output shpfile every n epochs
    # hierarchical_region_merging_multiseeds('dict_bbox_5_.pickle',
    #                                        'dict_seeds_2_.pickle',
    #                                        'output_thre' + str(thre) + '.shp',
    #                                        thre, 10)

    # multi threads
    thre_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    # pool = mp.Pool(7)
    # pool.map(main_func, thre_list)
    main_func(0.9)
