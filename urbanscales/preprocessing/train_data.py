import copy
import glob
import os
import pickle
import sys
import time
import matplotlib
matplotlib.use('Agg')

from random import random

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import config

# RoadNetwork is greyed out in IDEs like PyCharm hinting that the import is not being used but it is needed for pickle loadings
from urbanscales.io.road_network import RoadNetwork   # Remember not to comment out; specifically need to be careful with
# automatic Optimise Imports button in pycharm. That will delete this line and result in a crash.

from urbanscales.preprocessing.prep_network import Scale
from urbanscales.preprocessing.prep_speed import ScaleJF
from urbanscales.preprocessing.tile import Tile
import pandas as pd
from tqdm import tqdm
from smartprint import smartprint as sprint
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import contextily as ctx
from slugify import slugify

# from urbanscales.io.speed_data import Segment  # this line if not present gives
# # an error while depickling a file.


# All custom unpicklers are due to SO user Pankaj Saini's answer:  https://stackoverflow.com/a/51397373/3896008
class CustomUnpicklerTrainDataVectors(pickle.Unpickler):
    """
    A custom unpickler that extends the standard pickle.Unpickler to provide additional functionality or restrictions
    during the unpickling process. It is designed to handle specific class loading behaviors, ensuring that module
    dependencies are respected correctly.

    Methods:
        find_class(module, name): Redirects the loading process for classes to the superclass's method, ensuring
        compatibility and security during the unpickling process.
    """
    def find_class(self, module, name):
        """`
        Ensures that the class specified is correctly loaded from the correct module, leveraging the super class's
        functionality. This method can be customized to handle specific class loading rules or to implement security
        measures.

        Parameters:
            module (str): The module where the class is located.
            name (str): The name of the class to be loaded.

        Returns:
            type: The class type that is to be loaded.
        """
        return super().find_class(module, name)


class TrainDataVectors:
    """
    Represents training data vectors for a given city, scale, and time of day. This class manages the loading,
    processing, and manipulation of training data from stored pickle files or by generating new training vectors.

    Attributes:
        X (DataFrame or list): The feature vectors for the training data.
        Y (DataFrame or list): The target values associated with the feature vectors.
        bbox_X (list): Bounding boxes associated with each feature vector in X.
        bbox_Y (list): Bounding boxes associated with each target value in Y.
        city_name (str): The name of the city for which the training data is generated.
        scale (int): The scale or resolution of the data.
        tod (int): Time of day identifier for the training data.
        empty_train_data (bool): Flag to indicate if the training data is empty.

    Methods:
        __init__(city_name, scale, tod): Initializes the TrainDataVectors object, potentially loading existing data or
                                        generating new vectors.
        plot_collinearity_heatmap(): Plots a heatmap representing the collinearity between features in X.
        set_X_and_Y(process_X=True): Processes and sets the X and Y attributes based on provided or existing data.
        viz_y_hist(): Visualizes the histogram of the Y values, useful for understanding the distribution of target values.

        compute_training_data_for_all_cities(): Static method to compute training data for all configured cities,
                                                scales, and times of day.
    """

    def __init__(self, city_name, scale, tod):
        """
        Initializes the TrainDataVectors object by either loading existing training data from pickle files or by
        generating new training vectors based on the specified city, scale, and time of day.

        Parameters:
            city_name (str): The city for which the training data is to be generated or loaded.
            scale (int): The scale or resolution of the data.
            tod (int): Time of day identifier, used to specify or load different sets of training data.
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
        Generates and displays a heatmap of the collinearity between features in the dataset. This helps in identifying
        highly correlated features that might affect model performance.
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

        if not os.path.exists(os.path.join(config.BASE_FOLDER, config.results_folder)):
            os.mkdir(os.path.join(config.BASE_FOLDER, config.results_folder))
        plt.savefig(os.path.join(config.BASE_FOLDER, config.results_folder, "collinearity_heatmap" +
                                 slugify(str((config.CONGESTION_TYPE, self.city_name, self.scale, self.tod))) + ".png"))
        plt.show(block=False);
        plt.close()

    def set_X_and_Y(self, process_X=True):
        """
        Processes and sets the X and Y attributes by loading or calculating feature vectors and target values. This method
        can either process both X and Y or just update Y based on the 'process_X' flag.

        Parameters:
            process_X (bool): If True, processes both X and Y; otherwise, only processes Y.
        """
        scl = Scale.get_object(self.city_name, self.scale)
        scl_jf = ScaleJF.get_object(self.city_name, self.scale, self.tod)
        assert isinstance(scl_jf, ScaleJF)

        if config.MASTER_VISUALISE_EACH_STEP:
            # Plot the bboxes from scl_jf
            # Example list of bounding boxes
            # bboxes = list(scl_jf.bbox_segment_map.keys())
            bboxes = list (scl.dict_bbox_to_subgraph.keys())

            # Create a GeoDataFrame with these bounding boxes
            gdf = gpd.GeoDataFrame({
                'geometry': [Polygon([(lon1, lat1), (lon1, lat2), (lon2, lat2), (lon2, lat1)]) for
                             lat1, lat2, lon1, lon2 in bboxes]
            }, crs="EPSG:4326")  # EPSG:4326 is WGS84 latitude-longitude projection

            # Convert the GeoDataFrame to the Web Mercator projection (used by most web maps)
            gdf_mercator = gdf.to_crs(epsg=3857)

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 10))
            gdf_mercator.boundary.plot(ax=ax, color='red')
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
            ax.set_axis_off()
            plt.show(block=False)

        betweenness_fname = os.path.join(config.BASE_FOLDER, config.network_folder,
                                         self.city_name + "_mean_betweenness_centrality.pkl")
        if os.path.exists(betweenness_fname):
            with open(betweenness_fname, "rb") as f_between:
                dict_between = pickle.load(f_between)
        else:
            K = 20
            iterations = 10
            initial_bc = nx.betweenness_centrality(scl.RoadNetwork.G_osm, k=K)
            all_nodes = list(initial_bc.keys())

            # Dictionary to store the mean betweenness centrality values for each node up to each iteration
            mean_values_per_node = {node: [] for node in all_nodes}

            for i in tqdm(range(iterations), desc="Computing betweenness using iterations.."):
                bc = nx.betweenness_centrality(scl.RoadNetwork.G_osm, k=K)
                for node in all_nodes:
                    if i == 0:
                        current_mean = bc[node]
                    else:
                        current_mean = (mean_values_per_node[node][-1] * i + bc[node]) / (i + 1)
                    mean_values_per_node[node].append(current_mean)
            overall_mean_bc_per_node = {node: sum(values) / len(values) for node, values in
                                        mean_values_per_node.items()}

            dict_between = overall_mean_bc_per_node
            rand_pickle_marker = os.path.join(config.temp_folder_for_robust_pickle_files,
                                              str(int(np.random.rand() * 100000000000000)))
            with open(rand_pickle_marker, "wb") as f_between:
                pickle.dump(dict_between, f_between, protocol=config.pickle_protocol)
                print("Pickle saved!")
            os.rename(rand_pickle_marker, betweenness_fname)

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


                if len(config.td_drop_feature_lists) > 0:
                    for feat in config.td_drop_feature_lists:
                        self.X.drop(feat, axis=1, inplace=True)


                self.plot_collinearity_heatmap()



            self.Y = pd.DataFrame(data=np.array(self.Y))
            self.Y = self.Y.values.reshape(self.Y.shape[0])
            assert not np.isnan(self.Y).any(), "The array self.Y contains NaN values."

            rand_pickle_marker = os.path.join(config.temp_folder_for_robust_pickle_files,
                                              str(int(np.random.rand() * 100000000000000)))
            with open(rand_pickle_marker, "wb") as f:
                pickle.dump(self, f, protocol=config.pickle_protocol)
                print("Pickle saved!")
            os.rename(rand_pickle_marker, fname)

        debug_stop = 2


    def viz_y_hist(self):
        """
        Visualizes a histogram of the Y values to provide insights into the distribution of target values across the dataset.
        Useful for initial data analysis and to check data quality.
        """
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



    @staticmethod
    def compute_training_data_for_all_cities():
        """
        Computes and stores training data for all configured cities, scales, and times of day. This method orchestrates
        the entire process, leveraging other methods in the class to generate or update training data as necessary.
        """
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
                        print (tdv)

    def __repr__(self):
        """
        Returns a formal string representation of the TrainDataVectors object, providing a quick summary of its main
        attributes. This representation includes the city name, scale, time of day, whether the training data is empty,
        and the shapes of the X and Y datasets if they are available.

        Returns:
            str: A string representation of the TrainDataVectors object, which can be useful for debugging and logging
                 purposes to quickly identify the state of the object.
        """
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
