import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import LineString

import config
from connectedregions.preprocessing.tiles.helper_files import is_bounding_box_intersecting_linestring


def read_speed_files():
    rid_speed_jf = pd.read_csv(os.path.join(config.data_folder, config.var_jf))
    rid_speed_ci = pd.read_csv(os.path.join(config.data_folder, config.var_ci))
    rid_speed_linestring = pd.read_csv(os.path.join(config.data_folder, config.road_linestring))


    merged_jf = pd.merge(rid_speed_jf, rid_speed_linestring, on="NID", how="inner")
    merged_jf = merged_jf.drop(columns=["Length", "Description", "PC", "NID"])

    merged_ci = pd.merge(rid_speed_ci, rid_speed_linestring, on="NID", how="inner")
    merged_ci = merged_ci.drop(columns=["Length", "Description", "PC", "NID"])

    return {"JF": merged_jf, "CI": merged_ci}


def create_intersecting_bbox_map(rid_speed_linestring, scale, bbox_list_at_this_scale):
    """

    Args:
        rid_speed_linestring:
        scale:
        bbox_list_at_this_scale:     Each bbox must be of this format: tuple/list both are fine
         lon_min, lat_min, lon_max, lat_max = bbox

    Returns:

    """
    fname = os.path.join(config.intermediate_files_path, "speed_ls_bbox_map_scale" + str(scale) + ".pickle")
    if os.path.exists(fname):
        with open(fname, "rb") as handle:
            ls_bbox_map = pickle.load(handle)
    else:
        ls_bbox_map = {}
        for bbox in bbox_list_at_this_scale:
            for i in range(rid_speed_linestring.shape[0]):
                tuples = tuples(eval(rid_speed_linestring.Linestring.iloc[i].replace("MULTILINESTRING ","").replace(" ",",").replace(",0)",")").replace(",0,","),(")))
                shapely_linestring = LineString(tuples)
                if is_bounding_box_intersecting_linestring(shapely_linestring, bbox):
                    if tuples in ls_bbox_map:
                        ls_bbox_map[tuples].append(tuple(bbox))
                    else:
                        ls_bbox_map[tuples] = [tuple(bbox)]

                if config.plotting_enabled_speed_data_preprocess:
                    plt.plot(*shapely_linestring.xy, color="blue")


            if config.plotting_enabled_speed_data_preprocess:
                plt.show()

            with open(fname, "wb") as f:
                pickle.dump(ls_bbox_map, f, protocol=4)


    return ls_bbox_map