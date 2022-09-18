import os
import pickle

import numpy as np

import config
from urbanscales.io.speed_data import SpeedData
from urbanscales.preprocessing.prep_network import Scale
from urbanscales.preprocessing.prep_speed import ScaleJF
from urbanscales.preprocessing.tile import Tile
import pandas as pd

from urbanscales.io.speed_data import Segment  # this line if not present gives

# an error while depickling a file.


class TrainDataVectors:
    def __init__(self, city_name, scale, tod):
        """

        Args:
            city_name & scale: Trivial
            tod: single number based on granularity
        """
        self.X = []
        self.Y = []
        self.tod = tod
        self.city_name = city_name
        self.scale = scale

        self.set_X_and_Y()

    def set_X_and_Y(self):
        sd = SpeedData.get_object(self.city_name)
        scl = Scale.get_object_at_scale(self.city_name, self.scale)
        # scl_jf = ScaleJF(scl, sd )
        scl_jf = ScaleJF.get_object(self.city_name, self.scale, self.tod)
        assert isinstance(scl_jf, ScaleJF)
        for bbox in scl_jf.bbox_segment_map:
            # assert bbox in scl_jf.bbox_jf_map
            assert isinstance(scl, Scale)
            subg = scl.dict_bbox_to_subgraph[bbox]
            if isinstance(subg, str):
                if subg == "Empty":
                    # we skip creating X and Y for this empty tile
                    # which does not have any roads OR
                    # is outside the scope of the administrative area
                    continue

            assert isinstance(subg, Tile)

            self.X.append(subg.get_vector_of_features())
            self.Y.append(scl_jf.bbox_jf_map[bbox])

        fname = os.path.join(
            "network", scl.RoadNetwork.city_name, "_scale_" + str(scl.scale) + "_train_data_" + str(self.tod) + ".pkl"
        )
        if not os.path.exists(fname):
            with open(fname, "wb") as f:
                pickle.dump({"X": self.X, "Y": self.Y}, f, protocol=config.pickle_protocol)
                print("Pickle saved! ")

        debug_stop = 2

    def get_object(cityname, scale, tod):
        fname = os.path.join("network", cityname, "_scale_" + str(scale) + "_train_data_" + str(tod) + ".pkl")
        assert os.path.exists(fname)
        with open(fname, "rb") as f:
            obj = pickle.load(f)
        nparrayX = np.array(obj.X)
        nparrayY = np.array(obj.Y)

        obj.X = pd.DataFrame(data=nparrayX, columns=Tile.get_feature_names())
        obj.Y = pd.DataFrame(data=nparrayY)

        return obj


if __name__ == "__main__":
    # td = TrainDataVectors("Singapore", 3 ** 2, 3600 * 7 // 600)
    # sg = SpeedData.get_object("Singapore")
    for tod in range(24):
        for seed in [2, 3]:  # , 4, 5, 6, 7]:
            for depth in range(2, 4):
                TrainDataVectors("Singapore", seed ** depth, tod)
    debug_stop = 1

    debug_stop = 3
