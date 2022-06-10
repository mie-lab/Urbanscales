# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 16:39:45 2022

@author: yatzhang
"""

import os, sys
from osgeo import ogr, osr
import geopandas as gp
import matplotlib.pyplot as plt


def compute_feature_bbox(polygon):  #tbd
    '''
    compute the feature vectors of each polygon, including bbox and irregular polygon (merged version)

    Parameters
    ----------
    polygon : TYPE
        DESCRIPTION.

    Returns
    -------
    recompute_feature : list (float vector)
        eight-dimensional feature variable vector

    '''
    recompute_feature = 0
    return recompute_feature


def compute_local_criteria(polygon_1, polygon_2): #tbd
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
    criteria_value = 0
    return criteria_value


def bbox_ogr_polygon(bbox_coords):
    '''
    convert the two coordinates of bbox into ogr.wkbPolygon

    Parameters
    ----------
    bbox_coords : list of coordinates
        list[[lat1, lng1],[lat2, lng2]].

    Returns
    -------
    ogr_polygon : ogr.wkbPolygon
        in this format, it can be directly used in topological computation

    '''
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(bbox_coords[0][0], bbox_coords[0][1])  # lat1, lng1
    ring.AddPoint(bbox_coords[0][0], bbox_coords[1][1])  # lat1, lng2
    ring.AddPoint(bbox_coords[1][0], bbox_coords[1][1])  # lat2, lng2
    ring.AddPoint(bbox_coords[1][0], bbox_coords[0][1])  # lat2, lng1
    ring.AddPoint(bbox_coords[0][0], bbox_coords[0][1])  # lat1, lng1
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


def hierarchical_region_merging_onetree(input_file, merged_shpfile):  # input_file is not used here
    '''
    implement hierarchial region merging process for each tree

    Returns
    -------
    merged results of each island for each tree

    '''
    # read bbox_file, a set of bbox file in multi-hierarchies
    dict_bbox = {'hierarchy_1':[[]], 'hierarchy_2':[[]], 'hierarchy_3':[[]]}
    
    # The keys of these three dictionaries are the same
    # read the list of bbox in each island in the best-fit hierarchy
    dict_islands = {'island_1':[], 'island_2':[], 'island_3':[], 'island_4':[]}
    # read start-seed (bbox) in each island in the best-fit hierarchy
    dict_seeds = {'island_1':[], 'island_2':[], 'island_3':[], 'island_4':[]}
    # store the merge result
    dict_merge = {}
    
    # implement hierarchial region merge for each seed
    for seed_i in dict_seeds:
        seed_bh = dict_seeds[seed_i]
        island_bh = dict_islands[seed_i]
        
        # merge the separate bbox in this island into a whole polygon
        whole_island = ogr.Geometry(ogr.wkbPolygon)
        for bbox_i in island_bh:
            bbox_ogr = bbox_ogr_polygon(bbox_i)
            whole_island = whole_island.Union(bbox_ogr)

        # identify all bbox in multi-hierarchies that interests with the boundary in the best-fit hierarchy
        dict_bbox_select = {}  # bbox: flag
        for hierarchy_i in dict_bbox:
            bbox_eh = dict_bbox[hierarchy_i]
            for bbox_i in bbox_eh:
                # convert bbox_eh to ogr_string
                bbox_ogr = bbox_ogr_polygon(bbox_i)
                
                if whole_island.Intersects(bbox_ogr):
                    dict_bbox_select[bbox_ogr] = True
                    
        # begin region merging
        seed_zone = bbox_ogr_polygon(seed_bh)
        # stop merging when all bbox has been marked as False
        while identify_bbox_usage(dict_bbox_select):
            # find all neibhouring bbox intersects with seed region, labeled as False
            for select_i in dict_bbox_select:
                if seed_zone.Intersect(select_i):
                    dict_bbox_select[select_i] = False
            
            # find the minimum local_criteria and its bbox among all bboxes that it touches
            min_criteria = sys.maxsize
            min_merge_bbox = []
            for select_i in dict_bbox_select:
                if dict_bbox_select[select_i]==True and seed_zone.Touches(select_i):
                    tmp_criteria = compute_local_criteria(seed_zone, select_i)
                    if tmp_criteria < min_criteria:
                        min_criteria = tmp_criteria
                        min_merge_bbox = select_i
            
            # merge seed_zone with min_merge_bbox
            seed_zone = seed_zone.Union(min_merge_bbox)
        
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
    
    # visualization
    shp_geod = gp.GeoDataFrame.from_file(merged_shpfile)
    shp_geod.plot()
    plt.show()
        

if __name__ == '__main__': 
    print('test')
