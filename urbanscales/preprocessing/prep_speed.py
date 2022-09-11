import copy
import os
import pickle
import shapely.wkt
import numpy as np
import shapely.ops
import config
from prep_network import Scale
from urbanscales.io.road_network import RoadNetwork
from urbanscales.io.speed_data import SpeedData, Segment
from tqdm import tqdm
from smartprint import smartprint as sprint
import shapely.geometry


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
            "network", self.Scale.RoadNetwork.city_name, "_scale_" + str(self.Scale.scale) + "_prep_speed.pkl"
        )
        if os.path.exists(fname):
            # @P/A Ask 2
            self.bbox_segment_map = copy.deepcopy((ScaleJF.get_object_at_scale(self.Scale.RoadNetwork.city_name, self.Scale.scale)).bbox_segment_map)
            return
        else:
            pbar = tqdm(
                total=len(self.SpeedData.NID_road_segment_map) * len(self.Scale.dict_bbox_to_subgraph),
                desc="Setting bbox Segment map...",
            )
            for segment in self.SpeedData.segment_jf_map:

                seg_poly = shapely.wkt.loads(segment)

                for bbox in self.Scale.dict_bbox_to_subgraph.keys():
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
        pbar = tqdm(total=len(self.bbox_segment_map), desc="Setting bbox JF map...")
        for bbox in self.bbox_segment_map:
            val = []
            for segment in self.bbox_segment_map[bbox]:
                val.append(self.SpeedData.segment_jf_map[segment][self.tod])

            if config.ps_spatial_combination_method == "mean":
                agg_func = np.mean
            elif config.ps_spatial_combination_method == "max":
                agg_func = np.max

            self.bbox_jf_map[bbox] = agg_func(val)
            pbar.update(1)

    def get_object_at_scale(cityname, scale):
        """

        Args:
            scale:

        Returns: (Saved) Object of this class (Scale)

        """
        fname = os.path.join("network", cityname, "_scale_" + str(scale) + "_prep_speed.pkl")
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                obj = pickle.load(f)
        return obj


if __name__ == "__main__":
    sd = SpeedData.get_object("Singapore")
    scl = Scale.get_object_at_scale("Singapore", 9)
    scl_jf = ScaleJF(scl, sd, tod=7)
    scl_jf = ScaleJF.get_object_at_scale("Singapore", 9,)
    debug_stop = 2
