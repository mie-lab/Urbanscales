import copy
import os
import pickle
import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

import config
from urbanscales.io.road_network import RoadNetwork
from urbanscales.preprocessing.prep_network import Scale
from urbanscales.preprocessing.prep_speed import ScaleJF
from urbanscales.preprocessing.tile import Tile
import pandas as pd

# from smartprint import smartprint as sprint
from slugify import slugify

# from urbanscales.io.speed_data import Segment  # this line if not present gives
# # an error while depickling a file.

# All custom unpicklers are due to SO user Pankaj Saini's answer:  https://stackoverflow.com/a/51397373/3896008
class CustomUnpicklerTrainDataVectors(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module, name)


class TrainDataVectors:
    def __init__(self, city_name, scale, tod):
        """

        Args:
            city_name & scale: Trivial
            tod: single number based on granularity
        """
        fname = os.path.join(
            config.BASE_FOLDER,
            config.network_folder,
            city_name,
            "_scale_" + str(scale) + "_train_data_" + str(tod) + ".pkl",
        )
        if config.td_delete_existing_pickle_objects:
            if os.path.exists(fname):
                os.remove(fname)

        if os.path.exists(fname):
            # with open(fname, "rb") as f:
            temp = copy.deepcopy(CustomUnpicklerTrainDataVectors(open(fname, "rb")).load())
            self.__dict__.update(temp.__dict__)
            nparrayX = np.array(self.X)
            nparrayY = np.array(self.Y)
            print(nparrayX.shape, nparrayY.shape)

        else:
            self.X = []
            self.Y = []
            self.tod = tod
            self.city_name = city_name
            self.scale = scale
            self.empty_train_data = True
            self.set_X_and_Y()

    def set_X_and_Y(self):
        # sd = SpeedData(self.city_name, c)
        # rn = RoadNetwork.get_object(self.city_name)
        scl = Scale.get_object(self.city_name, self.scale)
        # scl_jf = ScaleJF(scl, sd )

        for tod_ in config.td_tod_list:
            scl_jf = ScaleJF.get_object(self.city_name, self.scale, tod_)
            assert isinstance(scl_jf, ScaleJF)
            for bbox in scl_jf.bbox_segment_map:
                # assert bbox in scl_jf.bbox_jf_map
                assert isinstance(scl, Scale)
                subg = scl.dict_bbox_to_subgraph[bbox]
                if isinstance(subg, str):
                    if subg == config.rn_no_stats_marker:
                        # we skip creating X and Y for this empty tile
                        # which does not have any roads OR
                        # is outside the scope of the administrative area
                        continue

                assert isinstance(subg, Tile)

                self.X.append(subg.get_vector_of_features())
                self.Y.append(scl_jf.bbox_jf_map[bbox])

        fname = os.path.join(
            config.BASE_FOLDER,
            config.network_folder,
            scl.RoadNetwork.city_name,
            "_scale_" + str(scl.scale) + "_train_data_" + str(self.tod) + ".pkl",
        )
        if not os.path.exists(fname):
            nparrayX = np.array(self.X)
            nparrayY = np.array(self.Y)

            if not nparrayY.size < 30:  # we ignore cases with less than 100 data points
                self.empty_train_data = False

                self.X = pd.DataFrame(data=nparrayX, columns=Tile.get_feature_names())
                self.Y = pd.DataFrame(data=nparrayY)

                self.X, self.Y = TrainDataVectors.filter_infs(self.X, self.Y)
                if config.td_plot_raw_variance_before_scaling:
                    df = pd.DataFrame(self.X, columns=Tile.get_feature_names())
                    df.var().to_csv(
                        os.path.join(
                            config.BASE_FOLDER,
                            config.results_folder,
                            slugify(
                                "pre-norm-feat-variance-" + self.city_name + "-" + str(self.scale) + "-" + str(self.tod)
                            ),
                        )
                        + ".csv"
                    )

                self.Y = self.Y.values.reshape(self.Y.shape[0])

                with open(fname, "wb") as f:
                    pickle.dump(self, f, protocol=config.pickle_protocol)
                    print("Pickle saved! ")

            else:
                print("\n\n")
                print(self.city_name, " has less than 30 data points...skipping\n\n")

        debug_stop = 2

    def viz_y_hist(self):
        plt.clf()
        if isinstance(self.Y, list):
            # case when number of data points less than 30; Training data not generated.
            return

        plt.hist(self.Y.to_numpy().flatten(), bins=list(np.arange(0, 11, 0.3)))
        ttl = (
            self.city_name
            + "_tod-"
            + str(self.tod)
            + "_scale"
            + str(self.scale)
            + "_agg-"
            + config.sd_temporal_combination_method
            + config.ps_spatial_combination_method
        )
        plt.title(ttl)
        plt.savefig(os.path.join(config.sd_base_folder_path, ttl + ".png"), dpi=300)

    # @staticmethod
    # def get_object(cityname, scale, tod):
    #     fname = os.path.join("network", cityname, "_scale_" + str(scale) + "_train_data_" + str(tod) + ".pkl")
    #     assert os.path.exists(fname)
    #     with open(fname, "rb") as f:
    #         obj = pickle.load(f)
    #     nparrayX = np.array(obj.X)
    #     nparrayY = np.array(obj.Y)
    #
    #     obj.X = pd.DataFrame(data=nparrayX, columns=Tile.get_feature_names())
    #     obj.Y = pd.DataFrame(data=nparrayY)
    #
    #     return obj

    @staticmethod
    def filter_infs(df1, df2):
        """

        Args:
            df1, df2: Pandas dataframe
            refer to X and Y respectively

        Returns: dataframe with rows removed

        """
        initial_numrows = df1.shape[0]

        df2 = df2[np.isfinite(df1).all(1)]
        df1 = df1[np.isfinite(df1).all(1)]

        final_numrows = df1.shape[0]
        if initial_numrows != final_numrows:
            print(
                "Number & percentage of rows removed ",
                initial_numrows - final_numrows,
                (final_numrows - initial_numrows) / initial_numrows * 100,
                "%",
            )

        return df1, df2

    @staticmethod
    def compute_training_data_for_all_cities():
        for city in config.scl_master_list_of_cities:
            for seed in config.scl_list_of_seeds:
                for depth in config.scl_list_of_depths:
                    for tod in config.td_tod_list:
                        startime = time.time()
                        tdv = TrainDataVectors(city, seed ** depth, tod)
                        if config.td_viz_y_hist == True:
                            tdv.viz_y_hist()
                        print(time.time() - startime)
                        print(city, seed, depth, tod)


if __name__ == "__main__":
    # this chdir might not be needed;
    # tgere was some trouble with paths in my case.
    os.chdir(config.home_folder_path)

    TrainDataVectors.compute_training_data_for_all_cities()
