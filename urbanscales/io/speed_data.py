import pickle
import sys

import numpy as np

import config
import os
import shapely.wkt
from shapely import geometry
import pandas as pd
import copy
from tqdm import tqdm
from smartprint import smartprint as sprint


class SpeedData:
    def __init__(self, city_name, time_gran_minutes_raw, time_gran_minutes_target):
        """

        Args:
            city_name:
            time_gran_minutes_raw:
            time_gran_minutes_target:
        """
        self.city_name = city_name
        self.time_gran_minutes_raw = time_gran_minutes_raw
        self.time_gran_minutes_target = time_gran_minutes_target

        self.road_segments = None
        self.NIDs = None
        self.NID_road_segment_map = {}
        self.nid_jf_map = {}
        self.segment_jf_map = {}

        self.set_road_segments()
        self.set_segment_jf_map()

    def set_road_segments(self):
        if not os.path.exists(os.path.join(config.sd_base_folder_path, self.city_name)):
            if not os.path.exists(config.sd_base_folder_path):
                os.mkdir(config.sd_base_folder_path)
            os.mkdir(os.path.join(config.sd_base_folder_path, self.city_name))

        if not os.path.exists(
            os.path.join(config.sd_base_folder_path, self.city_name, config.sd_seg_file_path_within_city)
        ):
            raise Exception("Error in here data; data file SEG missing")

        df = pd.read_csv(os.path.join(config.sd_base_folder_path, self.city_name, config.sd_seg_file_path_within_city))
        df = df[["NID", "Linestring"]].copy()
        self.road_segments = SegmentList(df.Linestring.to_list()).list_of_linestrings
        self.NIDs = df.NID.to_list()
        self.NID_road_segment_map = dict(zip(self.NIDs, self.road_segments))
        debug_stop = True

    def set_segment_jf_map(self):
        if not os.path.exists(
            os.path.join(config.sd_base_folder_path, self.city_name, config.sd_jf_file_path_within_city)
        ):
            raise Exception("Error in here data; data file JF missing")
        """
        NID	2022-02-28T23:59:27	2022-03-01T00:01:27	2022-03-01T00:03:27	2022-03-01T00:05:27
        0	0.0	0.18115	0.18115	0.18115
        1	0.0	0.0	0.0	0.0
        2	0.0	0.0	0.0	0.0
        3	0.0	0.0	0.0	0.0
        4	1.47707	1.54911	1.20348	0.99297
        """
        df = pd.read_csv(os.path.join(config.sd_base_folder_path, self.city_name, config.sd_jf_file_path_within_city))

        assert self.road_segments is not None, "list_of_linestrings not set"

        self.num_timesteps_in_data = 0

        for i in tqdm(range(len(self.NIDs)), desc=" Reading JF file"):
            seg_nid = self.NIDs[i]
            jf_list = df.loc[df["NID"] == seg_nid].values.flatten().tolist()

            jf_list = self._aggregation(jf_list, self.time_gran_minutes_target // self.time_gran_minutes_raw)
            if self.num_timesteps_in_data == 0:
                self.num_timesteps_in_data = len(jf_list)
            else:
                # all segments should have the same number of time steps in data
                assert self.num_timesteps_in_data == len(jf_list)

            self.nid_jf_map[seg_nid] = copy.deepcopy(jf_list)
            self.segment_jf_map[hash(self.NID_road_segment_map[seg_nid])] = copy.deepcopy(jf_list)

        fname = os.path.join("network", self.city_name, "_speed_data_object.pkl")
        if not os.path.exists(fname):
            with open(fname, "wb") as f:
                pickle.dump(self, f, protocol=config.pickle_protocol)

    def get_object(cityname):
        """

        Args:
            scale:

        Returns: (Saved) Object of this class (Scale)

        """
        fname = os.path.join("network", cityname, "_speed_data_object.pkl")
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                obj = pickle.load(f)
        return obj

    def _aggregation(self, jf_list, combine_how_many_t_steps):
        """

        Args:
            jf_list:
            combine_how_many_t_steps:

        Returns:
            shortened JF list

        """
        if config.sd_temporal_combination_method == "mean":
            agg_func = np.mean
        elif config.sd_temporal_combination_method == "max":
            agg_func = np.max

        a = []
        for i in range(0, len(jf_list), combine_how_many_t_steps):
            a.append(agg_func(jf_list[i : i + combine_how_many_t_steps]))
        return a


class Segment:
    def __init__(self, linestring):
        """

        Args:
            linestring: format of linestring
                        MULTILINESTRING ((103.81404 1.32806 0,103.81401 ....... 103.81363 1.32444 0,103.81353 1.32388 0
                        ,103.81344 1.32346 0))
        """
        self.polygon = shapely.wkt.loads(linestring)

    def get_shapely_poly(self):
        return self.polygon

    def seg_hash(shapely_poly):
        """

        Args:
            shapely_poly: Shapely polygon
                    if it is in string (wkt), no need for this function, but no harm as well
                                            just make sure that it is the same throughout
                                            don't mix with and without hash

        Returns:
            hash

        """
        assert (
            isinstance(shapely_poly, str)
            or isinstance(shapely_poly, geometry.Polygon)
            or isinstance(shapely_poly, geometry.MultiLineString)
        )
        if isinstance(shapely_poly, geometry.Polygon) or isinstance(shapely_poly, geometry.MultiLineString):
            return hash(shapely_poly.wkt)
        else:
            return hash(shapely_poly)


class SegmentList:
    def __init__(self, list_of_linestrings):
        """

        Args:
            list_of_linestrings:
        """
        self.list_of_linestrings = []
        for ls in list_of_linestrings:
            self.list_of_linestrings.append(Segment(ls))


if __name__ == "__main__":
    sd = SpeedData("Singapore", 2, 10)
    sprint(sd.num_timesteps_in_data)