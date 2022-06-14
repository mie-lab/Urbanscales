# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 16:39:45 2022

@author: yatzhang
"""
import copy
import pickle
import os, sys
import random

import numpy as np
import shapely
from osgeo import ogr, osr, gdal
from tqdm import tqdm
import geopandas as gp
import matplotlib as mpl
import matplotlib.pyplot as plt

import osmnx as ox

from python_scripts.network_to_elementary.get_sg_osm import get_sg_poly
from python_scripts.network_to_elementary.osm_to_tiles import (
    fetch_road_network_from_osm_database,
    is_point_in_bounding_box,
)
from python_scripts.network_to_elementary.tiles_to_elementary import get_box_to_nodelist_map, get_stats_for_one_tile



def from_ogr_to_shapely_plot(list_of_three_polys, seed_i, count):
    # Creating a copy of the input OGR geometry. This is done in order to
    # ensure that when we drop the M-values, we are only doing so in a
    # local copy of the geometry, not in the actual original geometry.
    # ogr_geom_copy = ogr.CreateGeometryFromWkb(ogr_geom.ExportToIsoWkb())
    plt.clf()
    plt_set = [[seed_i, 'deepskyblue', 'solid'],
               ['epoch' + count, 'tomato', 'dotted']]
    for i in range(len(list_of_three_polys)):
        poly = list_of_three_polys[i]
        ogr_geom_copy = ogr.CreateGeometryFromWkb(poly.ExportToWkb())

        # Dropping the M-values
        ogr_geom_copy.SetMeasured(False)

        # Generating a new shapely geometry
        shapely_geom = shapely.wkb.loads(ogr_geom_copy.ExportToIsoWkb())

        if shapely_geom.type == 'Polygon':
            x, y = shapely_geom.exterior.xy
            plt.plot(x, y, label=plt_set[i][0], color=plt_set[i][1], linestyle=plt_set[i][2])
        elif shapely_geom.type == 'MultiPolygon':
            for m in range(len(shapely_geom)):
                _x, _y = shapely_geom[m].exterior.xy
                if m==0:
                    plt.plot(_x, _y, label=plt_set[i][0], color=plt_set[i][1], linestyle=plt_set[i][2])
                else:
                    plt.plot(_x, _y, label='_'+plt_set[i][0], color=plt_set[i][1], linestyle=plt_set[i][2])
    plt.legend(loc='upper right')
    plt.savefig("./urban_merge/epoch_"+ seed_i + str(count)+".png", dpi=300)


def from_ogr_to_shapely_plot_multiseeds(dict_merge, epoch):
    # Creating a copy of the input OGR geometry. This is done in order to
    # ensure that when we drop the M-values, we are only doing so in a
    # local copy of the geometry, not in the actual original geometry.
    # ogr_geom_copy = ogr.CreateGeometryFromWkb(ogr_geom.ExportToIsoWkb())
    plt.clf()
    plt_set = [['tomato', 'dotted']]
    colors_pad = plt.cm.rainbow(np.linspace(0, 1, len(dict_merge)))
    for i in range(len(dict_merge)):
        poly = dict_merge[list(dict_merge)[i]]
        ogr_geom_copy = ogr.CreateGeometryFromWkb(poly.ExportToWkb())

        # Dropping the M-values
        ogr_geom_copy.SetMeasured(False)

        # Generating a new shapely geometry
        shapely_geom = shapely.wkb.loads(ogr_geom_copy.ExportToIsoWkb())

        if shapely_geom.type == 'Polygon':
            x, y = shapely_geom.exterior.xy
            plt.plot(x, y, label=list(dict_merge)[i], color=colors_pad[i])
        elif shapely_geom.type == 'MultiPolygon':
            for m in range(len(shapely_geom)):
                _x, _y = shapely_geom[m].exterior.xy
                if m==0:
                    plt.plot(_x, _y, label=list(dict_merge)[i], color=colors_pad[i])
                else:
                    plt.plot(_x, _y, label='_'+list(dict_merge)[i], color=colors_pad[i])
    plt.legend(loc='upper right')
    plt.title('epoch: ' + str(epoch))
    plt.xlim(103.6, 104.1)
    plt.ylim(1.26, 1.45)
    plt.savefig("./urban_merge/epoch_" + str(epoch) + ".png", dpi=300)


def read_shpfile_SGboundary(shpfile):
    ds = ogr.Open(shpfile, 0)
    iLayer = ds.GetLayerByIndex(0)
    iFeature = iLayer.GetNextFeature()
    geometry_wkt = -1
    while iFeature is not None:
        geometry_wkt = iFeature.GetGeometryRef()
    return geometry_wkt

def get_OSM_subgraph_in_poly(G_OSM, polygon_as_bb_list):
    # [[[BB1_lon1, BB1_lat1], [BB1_lon2, BB1_lat2]], [[BB2_lon1, BB2_lat1], .... ]

    nodes_for_poly = []
    for node in G_OSM.nodes:
        for bbox in polygon_as_bb_list:
            # y is the lat, x is the lon (Out[20]: {'y': 1.2952316, 'x': 103.872544, 'street_count': 3})
            lat, lon = G_OSM.nodes[node]["y"], G_OSM.nodes[node]["x"]

            # format of bb in is_point_in_bounding_box:
            #     lat_min, lon_min = [bb[0], bb[1]]
            #     lat_max, lon_max = [bb[2], bb[3]]
            [BB1_lon1, BB1_lat1], [BB1_lon2, BB1_lat2] = bbox

            # reassign for existing is_point_in_bounding_box function
            lat_min = min(BB1_lat1, BB1_lat2)
            lat_max = max(BB1_lat1, BB1_lat2)
            lon_min = min(BB1_lon1, BB1_lon2)
            lon_max = max(BB1_lon1, BB1_lon2)

            if is_point_in_bounding_box(lat, lon, bb=[lat_min, lon_min, lat_max, lon_max]):
                nodes_for_poly.append(node)
    subgraph = G_OSM.subgraph(nodes_for_poly).copy()
    return subgraph


def convert_stats_to_vector(stats):
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

    assert single_vector.shape == (1, 12)
    return single_vector


def compute_local_criteria(
    polygon_1, polygon_2, read_G_osm_from_pickle=True, bbox_to_points_map=None, a=0.5, b=0.5, loss_merge="sum"
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
        with open("G_OSM_extracted.pickle", "rb") as handle:
            G_OSM = pickle.load(handle)
    else:
        G_OSM = fetch_road_network_from_osm_database(polygon=get_sg_poly(), network_type="drive", custom_filter=None)
        # with open("G_OSM_extracted.pickle", "wb") as f:
        #     # not needed
        #     pickle.dump(G_OSM, f, protocol=pickle.HIGHEST_PROTOCOL)

    stats_vector_1 = convert_stats_to_vector(get_stats_for_one_tile([get_OSM_subgraph_in_poly(G_OSM, polygon_1)]))
    stats_vector_2 = convert_stats_to_vector(get_stats_for_one_tile([get_OSM_subgraph_in_poly(G_OSM, polygon_2)]))
    stats_vector_combined = convert_stats_to_vector(
        get_stats_for_one_tile([get_OSM_subgraph_in_poly(G_OSM, polygon_1 + polygon_2)])
    )

    # https://github.com/gboeing/osmnx/blob/997facb88ac566ccf79227a13b86f2db8642d04a/osmnx/stats.py#L339
    # m refers to edge count
    edge_count_1 = stats_vector_1[1]
    edge_count_2 = stats_vector_2[1]
    edge_count_combined = stats_vector_combined[1]
    new_edges = edge_count_combined - (edge_count_1 + edge_count_2)
    if new_edges == 0:
        return 1

    f_sim = a * spatial.distance.cosine(stats_vector_1, stats_vector_2)
    f_conn = b * (1 / new_edges)

    if loss_merge == "sum":
        criteria_value = f_sim + f_conn

    return criteria_value


def compute_local_criteria_random(polygon_1, polygon_2): #tbd
    '''
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

    '''
    # criteria_value = 0
    # return criteria_value
    return random.random()


def bbox_ogr_polygon(bbox_coords):
    '''
    convert the two coordinates of bbox into ogr.wkbPolygon

    Parameters
    ----------
    bbox_coords : list of coordinates
        list[[lng1, lat1],[lng2, lat2]].

    Returns
    -------
    ogr_polygon : ogr.wkbPolygon
        in this format, it can be directly used in topological computation

    '''
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
    '''
    identify whether all selected bboxes has been marked as False

    Parameters
    ----------
    dict_bbox_select : dict{bbox_ogrstring: bool flag}
        dictionary of all bboxes that need to be considered in the merge process.

    Returns
    -------
    bool
        bool flag that whether all selected bboxes has been marked as False

    '''
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


def hierarchical_region_merging_oneseed(bbox_file, island_file, seed_file, merged_shpfile, criteria_thre):  # input_file is not used here
    '''
    implement hierarchial region merging process for each tree, each region is limited by its boundary

    Returns
    -------
    merged results of each island for each tree

    '''

    # read bbox_file, a set of bbox file in multi-hierarchies
    # dict_bbox = {'hierarchy_1':[[]], 'hierarchy_2':[[]], 'hierarchy_3':[[]]}
    with open(bbox_file, 'rb') as handle1:
        dict_bbox = pickle.load(handle1)
    
    # The keys of these three dictionaries are the same
    # read the list of bbox in each island in the best-fit hierarchy
    # dict_islands = {'island_1':[], 'island_2':[], 'island_3':[], 'island_4':[]}
    with open(island_file, 'rb') as handle2:
        dict_islands = pickle.load(handle2)
    # read start-seed (bbox) in each island in the best-fit hierarchy
    # dict_seeds = {'island_1':[], 'island_2':[], 'island_3':[], 'island_4':[]}
    with open(seed_file, 'rb') as handle3:
        dict_seeds = pickle.load(handle3)
    # store the merge result
    dict_merge = {}
    
    # implement hierarchial region merge for each seed
    for seed_i in tqdm(dict_seeds, desc='Processing island'):
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
        print('total number:', len(dict_bbox_select))
                    
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
                if dict_bbox_select[select_i]==True and seed_zone.Overlaps(select_i):
                    dict_bbox_select[select_i] = False
            
            # find the minimum local_criteria and its bbox among all bboxes that it touches
            min_criteria = sys.maxsize
            min_merge_bbox = []
            _flag = False
            for select_i in dict_bbox_select:
                if dict_bbox_select[select_i]==True and seed_zone.Touches(select_i):
                    touch_count += 1
                    tmp_criteria = compute_local_criteria(seed_zone, select_i)
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
                from_ogr_to_shapely_plot([whole_island, seed_zone], seed_i, "_"+ str(count))
                print('Epcoh: {}. Available bbox: {}. Touch bbox: {}.'.
                      format(count, identify_bbox_usage_num(dict_bbox_select), touch_count))
            # stop iterating when no bbox touching with seed_zone
            if touch_count == 0:
                print('Epcoh: {}. Available bbox: {}. Touch bbox: {}.'.
                      format(count, identify_bbox_usage_num(dict_bbox_select), touch_count))
                break

        # merge result
        dict_merge[seed_i] = seed_zone

    # output it as shapefile result
    os.environ['SHAPE_ENCODING'] = "utf-8"
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.access( merged_shpfile, os.F_OK ):
        driver.DeleteDataSource( merged_shpfile )
    newds = driver.CreateDataSource(merged_shpfile)
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    layernew = newds.CreateLayer('line',srs,ogr.wkbPolygon)
    
    field_PC = ogr.FieldDefn("island", ogr.OFTString)
    field_PC.SetWidth(30)
    layernew.CreateField(field_PC)
    
    for i_zone in dict_merge:
        feat = ogr.Feature(layernew.GetLayerDefn())
        feat.SetGeometry(dict_merge[i_zone])
        feat.SetField('island', i_zone)
        layernew.CreateFeature(feat)
        feat.Destroy()
    newds.Destroy()


def hierarchical_region_merging_multiseeds(bbox_file, seed_file, merged_shpfile, criteria_thre):  # input_file is not used here
    '''
    implement hierarchial region merging process for each tree, multi-seeds growing together, no island boundary limitation

    Returns
    -------
    merged results of each island for each tree

    '''

    # read bbox_file, a set of bbox file in multi-hierarchies
    with open(bbox_file, 'rb') as handle1:
        dict_bbox = pickle.load(handle1)

    # The keys of two dictionaries are the same
    with open(seed_file, 'rb') as handle3:
        dict_seeds = pickle.load(handle3)
    print(dict_seeds)
    # store the merge result
    dict_merge = copy.deepcopy(dict_seeds)
    for seed_i in dict_merge:
        seed_bh = dict_merge[seed_i]
        dict_merge[seed_i] = bbox_ogr_polygon(seed_bh)

    # identify all bbox in multi-hierarchies
    dict_bbox_select = {}  # bbox: flag
    for hierarchy_i in dict_bbox:
        bbox_eh = dict_bbox[hierarchy_i]
        for bbox_i in bbox_eh:
            # convert bbox_eh to ogr_string
            bbox_ogr = bbox_ogr_polygon(bbox_i)
            dict_bbox_select[bbox_ogr] = True
    print('total number:', len(dict_bbox_select))

    for seed_i in dict_merge:
        seed_zone = dict_merge[seed_i]
        # find all neibhouring bbox Overlapping with seed region, labeled as False
        for select_i in dict_bbox_select:
            if dict_bbox_select[select_i] == True and (seed_zone.Overlaps(select_i) or seed_zone.Crosses(select_i) or seed_zone.Contains(select_i) or seed_zone.Within(select_i)):
                dict_bbox_select[select_i] = False

    # implement hierarchial region merge
    epoch = 0
    from_ogr_to_shapely_plot_multiseeds(dict_merge, epoch)
    while identify_bbox_usage(dict_bbox_select):
        epoch += 1
        touch_count = 0
        for seed_i in dict_merge:
            seed_zone = dict_merge[seed_i]

            # find all neibhouring bbox Overlapping with seed region, labeled as False
            for select_i in dict_bbox_select:
                if dict_bbox_select[select_i] == True and (seed_zone.Overlaps(select_i) or seed_zone.Crosses(select_i) or seed_zone.Contains(select_i) or seed_zone.Within(select_i)):
                    dict_bbox_select[select_i] = False

            # find the minimum local_criteria and its bbox among all bboxes that it touches
            min_criteria = sys.maxsize
            min_merge_bbox = []
            _flag = False
            for select_i in dict_bbox_select:
                if dict_bbox_select[select_i] == True and seed_zone.Touches(select_i):
                    touch_count += 1
                    tmp_criteria = compute_local_criteria(seed_zone, select_i)
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
            # merge result
            dict_merge[seed_i] = seed_zone

            # find all neibhouring bbox Overlapping with seed region, labeled as False
            for select_i in dict_bbox_select:
                if dict_bbox_select[select_i] == True and (seed_zone.Overlaps(select_i) or seed_zone.Crosses(select_i) or seed_zone.Contains(select_i) or seed_zone.Within(select_i)):
                    dict_bbox_select[select_i] = False

        print('Epcoh: {}. Available bbox: {}. Touch bbox: {}.'.
              format(epoch, identify_bbox_usage_num(dict_bbox_select), touch_count))
        from_ogr_to_shapely_plot_multiseeds(dict_merge, epoch)
        # stop iterating when no bbox touching with seed_zone
        if touch_count == 0:
            print('Epcoh: {}. Available bbox: {}. Touch bbox: {}.'.
                  format(epoch, identify_bbox_usage_num(dict_bbox_select), touch_count))
            break

    # output it as shapefile result
    os.environ['SHAPE_ENCODING'] = "utf-8"
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.access(merged_shpfile, os.F_OK):
        driver.DeleteDataSource(merged_shpfile)
    newds = driver.CreateDataSource(merged_shpfile)
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    layernew = newds.CreateLayer('line', srs, ogr.wkbPolygon)

    field_PC = ogr.FieldDefn("island", ogr.OFTString)
    field_PC.SetWidth(30)
    layernew.CreateField(field_PC)

    for i_zone in dict_merge:
        feat = ogr.Feature(layernew.GetLayerDefn())
        feat.SetGeometry(dict_merge[i_zone])
        feat.SetField('island', i_zone)
        layernew.CreateFeature(feat)
        feat.Destroy()
    newds.Destroy()


if __name__ == '__main__': 
    print('test')
    os.environ['PROJ_LIB'] = r'C:\Users\yatzhang\Anaconda3\envs\trafficenv\Library\share\proj'
    os.environ['GDAL_DATA'] = r'C:\Users\yatzhang\Anaconda3\envs\trafficenv\Library\share'
    
    # hierarchical_region_merging_oneseed('./urban_merge/dict_bbox_5_.pickle',
    #                                     './urban_merge/dict_islands_2_.pickle',
    #                                     './urban_merge/dict_seeds_2_.pickle',
    #                                     './urban_merge/output.shp',
    #                                     0.75)

    hierarchical_region_merging_multiseeds('./urban_merge/dict_bbox_5_.pickle',
                                           './urban_merge/dict_seeds_2_.pickle',
                                           './urban_merge/output.shp',
                                           1)
