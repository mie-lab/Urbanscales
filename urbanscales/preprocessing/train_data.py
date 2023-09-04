import copy
import os
import pickle
import time
import glob
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import config
from urbanscales.io.road_network import RoadNetwork
from urbanscales.preprocessing.prep_network import Scale
from urbanscales.preprocessing.prep_speed import ScaleJF
from urbanscales.preprocessing.tile import Tile
import pandas as pd
from tqdm import tqdm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from smartprint import smartprint as sprint
from slugify import slugify
import seaborn as sns

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

        alternate_filename = os.path.join(
            config.BASE_FOLDER,
            config.network_folder,
            city_name, "_scale_" + str(scale) + "_train_data_" + str(tod) + ".pkl",
        )  # for the case when all files are present at the same folder with city name prefixes
        print(alternate_filename)

        if config.td_reuse_Graph_features:
            # get the list of any training files
            list_of_existing_all_td_for_any_tod_files = glob.glob(os.path.join(
                                                config.BASE_FOLDER,
                                                config.network_folder,
                                                city_name,
                                                "_scale_" + str(scale) + "_train_data_*.pkl",
                                            ))

            if len(list_of_existing_all_td_for_any_tod_files) > 0:
                # choose the first file
                alternate_filename = list_of_existing_all_td_for_any_tod_files[0]

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

        elif os.path.exists(alternate_filename):
            # with open(fname, "rb") as f:
            temp = copy.deepcopy(CustomUnpicklerTrainDataVectors(open(alternate_filename, "rb")).load())
            self.__dict__.update(temp.__dict__)
            nparrayX = np.array(self.X)
            nparrayY = np.array(self.Y)
            sprint(nparrayX.shape, nparrayY.shape)

            # empty the self.Y and then refill
            # empty
            self.Y = []
            self.bbox_Y = []
            self.tod = tod

            print ("Updating only the Y component")
            # refill
            self.set_X_and_Y(process_X=False)

            self.empty_train_data = False
        else:


            self.X = []
            self.Y = []
            self.bbox_X = []
            self.bbox_Y = []
            self.tod = tod
            self.city_name = city_name
            self.scale = scale

            print("Updating both X and Y component")
            self.set_X_and_Y(process_X=True)

            self.empty_train_data = False

    def plot_collinearity_heatmap(self):
        """
        Plots a heatmap of the collinearity between features of self.X
        Returns:
        - A heatmap plot.
        """
        df = self.X
        # Compute the correlation matrix
        corr_matrix = df.corr()

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(20, 20))

        # Draw the heatmap with a color map
        sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, linewidths=.5, ax=ax)

        plt.tight_layout()
        # Show the plot
        plt.show()

    def set_X_and_Y(self, process_X=True):
        scl = Scale.get_object(self.city_name, self.scale)
        scl_jf = ScaleJF.get_object(self.city_name, self.scale, self.tod)
        assert isinstance(scl_jf, ScaleJF)

        with open(os.path.join(config.BASE_FOLDER, config.network_folder,
                               self.city_name + "_mean_betweenness_centrality.pkl"), "rb") as f_between:
            dict_between = pickle.load(f_between)

        sprint(len(scl_jf.bbox_segment_map))
        for bbox in tqdm(
                scl_jf.bbox_segment_map,
                desc="Training vectors for city, scale, tod: "
                     + self.city_name + "-" + str(self.scale) + "-" + str(self.tod)):
            assert isinstance(scl, Scale)
            subg = scl.dict_bbox_to_subgraph[bbox]
            if isinstance(subg, str) and subg == config.rn_no_stats_marker:
                continue
            assert isinstance(subg, Tile)

            if process_X:
                self.X.append(subg.get_features() + [subg.get_betweenness_centrality_global(dict_between)])
                self.bbox_X.append({bbox: self.X[-1]})

            self.Y.append(scl_jf.bbox_jf_map[bbox])
            self.bbox_Y.append({bbox: self.Y[-1]})

        sprint(len(self.bbox_Y), len(self.bbox_X))

        fname = os.path.join(
            config.BASE_FOLDER, config.network_folder, scl.RoadNetwork.city_name,
            "_scale_" + str(scl.scale) + "_train_data_" + str(self.tod) + ".pkl")

        if not os.path.exists(fname):
            self.empty_train_data = False

            if process_X:
                self.X = pd.DataFrame(data=np.array(self.X), columns=Tile.get_feature_names() + ["global_betweenness"])
                # Drop columns with more than 10% NaN values
                self.X = self.X.loc[:, self.X.isnull().mean() <= 0.1]
                # Fill NaN values with the mean of the respective column
                self.X.fillna(self.X.mean(), inplace=True)


                if len(config.td_drop_feature_lists) > 0:
                    for feat in config.td_drop_feature_lists:
                        self.X.drop(feat, axis=1, inplace=True)

                assert not self.X.isna().any().any(), "The DataFrame self.X contains NaN values."

                self.plot_collinearity_heatmap()



            self.Y = pd.DataFrame(data=np.array(self.Y))
            self.Y = self.Y.values.reshape(self.Y.shape[0])
            assert not np.isnan(self.Y).any(), "The array self.Y contains NaN values."

            with open(fname, "wb") as f:
                pickle.dump(self, f, protocol=config.pickle_protocol)
                print("Pickle saved!")

        debug_stop = 2

    # def set_Y_only(self):
    #     # sd = SpeedData(self.city_name, c)
    #     # rn = RoadNetwork.get_object(self.city_name)
    #     scl = Scale.get_object(self.city_name, self.scale)
    #     # scl_jf = ScaleJF(scl, sd )
    #
    #     scl_jf = ScaleJF.get_object(self.city_name, self.scale, self.tod)
    #
    #     assert isinstance(scl_jf, ScaleJF)
    #
    #     # get same list of bbox as the precomputed X since the ordering insdie the keys of the dict
    #     # self_jf.bbox_segment_map is not guaranteed :)
    #     bbox_list = [bbox for bbox_dict in self.bbox_X for bbox in bbox_dict.keys()]
    #
    #     for bbox in tqdm(
    #         bbox_list,  # this time we iterature through existing bbox_list instead of scl_jf.bbox_segment_map,
    #         #                                                           in the func set_X_and_Y
    #         desc="Recomputing only Y vectors for city, scale, tod: "
    #         + self.city_name
    #         + "-"
    #         + str(self.scale)
    #         + "-"
    #         + str(self.tod),
    #     ):
    #         # assert bbox in scl_jf.bbox_jf_map
    #         assert isinstance(scl, Scale)
    #         subg = scl.dict_bbox_to_subgraph[bbox]
    #         if isinstance(subg, str):
    #             if subg == config.rn_no_stats_marker:
    #                 # we skip creating X and Y for this empty tile
    #                 # which does not have any roads OR
    #                 # is outside the scope of the administrative area
    #                 continue
    #
    #         assert isinstance(subg, Tile)
    #
    #         # We don't update X this time
    #         # self.X.append(subg.get_vector_of_features())
    #
    #         self.Y.append(scl_jf.bbox_jf_map[bbox])
    #
    #         # we don't update X this time, just keep the same order using self.bbox_X
    #         # self.bbox_X.append({bbox: self.X[-1]})
    #
    #         self.bbox_Y.append({bbox: self.Y[-1]})
    #
    #     fname = os.path.join(
    #         config.BASE_FOLDER,
    #         config.network_folder,
    #         scl.RoadNetwork.city_name,
    #         "_scale_" + str(scl.scale) + "_train_data_" + str(self.tod) + ".pkl",
    #     )
    #     sprint (fname)
    #     if not os.path.exists(fname):
    #         nparrayX = np.array(self.X)
    #         nparrayY = np.array(self.Y)
    #         print  ("Reached here!! ")
    #         # if not nparrayY.size < 30:  # we ignore cases with less than 100 data points
    #         self.empty_train_data = False
    #
    #         self.X = pd.DataFrame(data=nparrayX, columns=Tile.get_feature_names())
    #         self.Y = pd.DataFrame(data=nparrayY)
    #
    #         self.X, self.Y = TrainDataVectors.filter_infs(self.X, self.Y)
    #         if config.td_plot_raw_variance_before_scaling:
    #             df = pd.DataFrame(self.X, columns=Tile.get_feature_names())
    #             if not os.path.exists(os.path.join(config.BASE_FOLDER, config.results_folder)):
    #                 os.mkdir(os.path.join(config.BASE_FOLDER, config.results_folder))
    #
    #             df.var().to_csv(
    #                 os.path.join(
    #                     config.BASE_FOLDER,
    #                     config.results_folder,
    #                     slugify(
    #                         "pre-norm-feat-variance-" + self.city_name + "-" + str(self.scale) + "-" + str(self.tod)
    #                     ),
    #                 )
    #                 + ".csv"
    #             )
    #
    #         self.Y = self.Y.values.reshape(self.Y.shape[0])
    #
    #         with open(fname, "wb") as f:
    #             pickle.dump(self, f, protocol=config.pickle_protocol)
    #             print("Pickle saved! ")
    #         print  ("Pickle saved!! ")
    #     debug_stop = 2

    def viz_y_hist(self):
        plt.clf()
        if isinstance(self.Y, list):
            # case when number of data points less than 30; Training data not generated.
            return

        plt.hist(self.Y.flatten(), bins=list(np.arange(0, 11, 0.3)))
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

        try:
            assert df2.shape[0] == df1.shape[0]
        except:
            sprint (df2.shape, df1.shape)
            raise AssertionError

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
                        tdv = TrainDataVectors(city, seed**depth, tod)
                        if config.td_viz_y_hist == True:
                            tdv.viz_y_hist()
                        print(time.time() - startime)
                        print("Inside train_data.py ", city, seed, depth, tod)

    def __repr__(self):
        x_shape = self.X.shape if hasattr(self.X, 'shape') else 'N/A'
        y_shape = self.Y.shape if hasattr(self.Y, 'shape') else 'N/A'

        return (
            f"<TrainDataVectors(city_name={self.city_name!r}, scale={self.scale!r}, tod={self.tod!r}, "
            f"empty_train_data={self.empty_train_data!r}, X_shape={x_shape}, Y_shape={y_shape})>"
        )


if __name__ == "__main__":
    # this chdir might not be needed;
    # tgere was some trouble with paths in my case.
    os.chdir(config.home_folder_path)

    TrainDataVectors.compute_training_data_for_all_cities()
