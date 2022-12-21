import copy
import os
import pickle
import sys
import time
from multiprocessing import Pool

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


import shapely.wkt
import numpy as np
import shapely.ops
import config
from urbanscales.preprocessing.prep_network import Scale
from urbanscales.io.road_network import RoadNetwork
from urbanscales.io.speed_data import SpeedData, Segment
from tqdm import tqdm
from smartprint import smartprint as sprint
import shapely.geometry

import pickle

# All custom unpicklers are due to SO user Pankaj Saini's answer:  https://stackoverflow.com/a/51397373/3896008
class CustomUnpicklerScaleJF(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "ScaleJF":
            return ScaleJF
        if name == "Scale":
            from urbanscales.preprocessing.prep_network import Scale

            return Scale
        return super().find_class(module, name)


class ScaleJF:
    """
    Attributes:
        self.bbox_segment_map = {}
        self.bbox_jf_map = {}
        self.Scale = scale
        self.SpeedData = speed_data
    """

    def __init__(self, scale: Scale, speed_data: SpeedData, tod: int):
        assert scale.RoadNetwork.city_name == speed_data.city_name
        fname = os.path.join(
            config.network_folder,
            scale.RoadNetwork.city_name,
            "_scale_" + str(scale) + "_prep_speed_" + str(tod) + ".pkl",
        )
        if config.ps_delete_existing_pickle_objects:
            if os.path.exists(fname):
                os.remove(fname)

        if os.path.exists(fname):
            with open(fname, "rb") as f:
                temp = copy.deepcopy(pickle.load(f))
                self.__dict__.update(temp.__dict__)

        else:
            self.bbox_segment_map = {}
            self.bbox_jf_map = {}
            self.Scale = scale
            self.SpeedData = speed_data
            self.tod = tod
            self.set_bbox_segment_map()
            self.set_bbox_jf_map()

    def set_bbox_segment_map(self):
        # Step 1: iterate over segments
        # Step 2: iterate over bboxes
        # Then within the loop populate the dict if both are intersecting
        fname = os.path.join(
            config.network_folder,
            self.Scale.RoadNetwork.city_name,
            "_scale_" + str(self.Scale.scale) + "_prep_speed_" + str(self.tod) + ".pkl",
        )

        if os.path.exists(fname):
            # @P/A Ask 2
            self.bbox_segment_map = copy.deepcopy(
                (ScaleJF.get_object(self.Scale.RoadNetwork.city_name, self.Scale.scale, self.tod)).bbox_segment_map,
            )
            return
        else:
            count_ = len(self.SpeedData.NID_road_segment_map) * len(self.Scale.dict_bbox_to_subgraph)
            pbar = tqdm(
                total=count_,
                desc="Setting bbox Segment map...",
            )
            for segment in self.SpeedData.segment_jf_map:
                print("Inside loop 1")
                seg_poly = shapely.wkt.loads(segment)

                for bbox in self.Scale.dict_bbox_to_subgraph.keys():
                    print("                   Inside loop 2")
                    N, S, E, W = bbox
                    bbox_shapely = shapely.geometry.box(W, S, E, N, ccw=True)
                    if seg_poly.intersection(bbox_shapely):
                        if bbox in self.bbox_segment_map:
                            self.bbox_segment_map[bbox].append(segment)
                        else:
                            self.bbox_segment_map[bbox] = [segment]

                    pbar.update(1)
            sprint(len(self.bbox_segment_map))

            with open(fname, "wb") as f:
                pickle.dump(self, f, protocol=config.pickle_protocol)
                print("Pickle saved! ")

    def set_bbox_jf_map(self):
        """
        Input: object of class Scale JF
        For the current tod, this function sets its Y (JF mean/median) depending on config
        """
        pbar = tqdm(total=len(self.bbox_segment_map), desc="Setting bbox JF map...")
        for bbox in self.bbox_segment_map:
            val = []
            for segment in self.bbox_segment_map[bbox]:
                try:
                    val.append(self.SpeedData.segment_jf_map[segment][self.tod])
                except:
                    debug_stop = True
                    # sprint("Error in segment_jf_map; length ", len(self.SpeedData.segment_jf_map[segment]))
                    sys.exit(0)

            if config.ps_spatial_combination_method == "mean":
                agg_func = np.mean
            elif config.ps_spatial_combination_method == "max":
                agg_func = np.max

            self.bbox_jf_map[bbox] = agg_func(val)
            pbar.update(1)

    def get_object(cityname, scale, tod):
        """
        This function uses the idea that the network and its maps are the same for all tod's
        Hence, it reads the static part (bbox to segment map) from the pickle and then
        create bbox_jf_map for the particular tod of this object
        Returns: (Saved) Object of this class (Scale)

        """
        fname = os.path.join(
            config.network_folder, cityname, "_scale_" + str(scale) + "_prep_speed_" + str(tod) + ".pkl"
        )
        if os.path.exists(fname):
            # ScaleJF.preprocess_different_tods([tod], Scale.get_object_at_scale(cityname, scale), SpeedData.get_object(cityname))
            try:
                obj = CustomUnpicklerScaleJF(open(fname, "rb")).load()
            except EOFError:
                sprint(fname)
                raise Exception("Error! Corrupted pickle file:\n Filename:  " + fname)
        else:
            raise Exception("Error! trying to read file that does not exist: Filename: " + fname)

        obj.tod = tod
        obj.set_bbox_jf_map()
        return obj

    def preprocess_different_tods(range_element, scl: Scale, sd: SpeedData):
        for tod in tqdm(range_element, desc="Processing for different ToDs"):
            scl_jf = ScaleJF(scl, sd, tod=tod)
            fname = os.path.join(
                config.network_folder,
                scl.RoadNetwork.city_name,
                "_scale_" + str(scl.scale) + "_prep_speed_" + str(tod) + ".pkl",
            )
            with open(fname, "wb") as f:
                pickle.dump(scl_jf, f, protocol=config.pickle_protocol)
                print("Pickle saved! ")
                # with open(fname, "rb") as f:
                #     obj = pickle.load(f)

    @staticmethod
    def helper_parallel(params):
        city, seed, depth = params
        sd = SpeedData(city, config.sd_raw_speed_data_gran, config.sd_target_speed_data_gran)
        scl = Scale(RoadNetwork(city), seed ** depth)
        ScaleJF.preprocess_different_tods(config.ps_tod_list, scl, sd)

    @staticmethod
    def connect_speed_and_nw_data_for_all_cities():
        list_of_parallel_items = []
        for city in config.scl_master_list_of_cities:
            for seed in config.scl_list_of_seeds:
                for depth in config.scl_list_of_depths:
                    list_of_parallel_items.append((city, seed, depth))

        if config.ppl_parallel_overall > 1:
            p = Pool(config.ppl_parallel_overall)
            print(p.map(ScaleJF.helper_parallel, list_of_parallel_items))
        else:
            for params in list_of_parallel_items:
                ScaleJF.helper_parallel(params)


if __name__ == "__main__":
    sys.path.append(config.home_folder_path)
    ScaleJF.connect_speed_and_nw_data_for_all_cities()

    debug_stop = 2
