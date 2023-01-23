import asyncio
import copy
import csv
import glob
import multiprocessing
import os
import pickle
import shutil
import sys
import numpy as np
from multiprocessing import Pool
import threading
from osmnx import utils_graph
from urbanscales.preprocessing.smart_truncate_gpd import smart_truncate

from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import time
import warnings

import networkx as nx
import osmnx as ox
from geopy.distance import geodesic

# from smartprint import smartprint as sprint
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import config
from urbanscales.io.road_network import RoadNetwork
from urbanscales.preprocessing.tile import Tile

import pickle

# All custom unpicklers are due to SO user Pankaj Saini's answer:  https://stackoverflow.com/a/51397373/3896008
class CustomUnpicklerScale(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Scale":
            return Scale
        return super().find_class(module, name)


class Scale:
    def __init__(self, RoadNetwork, scale):
        fname = os.path.join(
            config.BASE_FOLDER, config.network_folder, RoadNetwork.city_name, "_scale_" + str(scale) + ".pkl"
        )
        if config.scl_delete_existing_pickle_objects:
            if os.path.exists(fname):
                os.remove(fname)

        if os.path.exists(fname):
            with open(fname, "rb") as f:
                temp = copy.deepcopy(pickle.load(f))
                self.__dict__.update(temp.__dict__)
        else:
            self.RoadNetwork = RoadNetwork
            self.scale = scale
            self.tile_area = (config.rn_square_from_city_centre ** 2) / (scale ** 2)

            # the following assert ensures that the scale * scale gives the correct number of tiles
            # as this is valid only if our city is a square
            assert config.rn_square_from_city_centre != -1 and config.rn_percentage_of_city_area == 100

            self.set_bbox_sub_G_map()

    def set_bbox_sub_G_map(self, save_to_pickle=True):
        fname = os.path.join(
            config.BASE_FOLDER, config.network_folder, self.RoadNetwork.city_name, "_scale_" + str(self.scale) + ".pkl"
        )
        if os.path.exists(fname):
            # do nothing
            return
        self._set_list_of_bbox()
        self.dict_bbox_to_subgraph = {}

        inputs = list(range(len(self.list_of_bbox)))
        starttime = time.time()
        if not os.path.exists(config.warnings_folder):
            os.mkdir(config.warnings_folder)

        if config.ppl_parallel_overall > 1:
            if config.scl_n_jobs_parallel > 1:
                print("Remove one level of parallelisation")
                raise Exception("AssertionError: daemonic processes are not allowed to have children")

        if config.scl_n_jobs_parallel > 1:
            print("Processing in parallel: ")

            print("File counter running in background; clearing temp files")
            print("Cleaning temp files folder")
            shutil.rmtree(os.path.join(config.BASE_FOLDER, "temp", self.RoadNetwork.city_name), ignore_errors=True)
            os.mkdir(os.path.join(config.BASE_FOLDER, "temp", self.RoadNetwork.city_name))
            print("Cleaned the temp folder")

            stop_background_thread = threading.Event()

            self.keep_countin_state = True
            filecounter = threading.Thread(target=self.keep_counting)
            filecounter.start()

            with Pool(config.scl_n_jobs_parallel) as p:
                list_of_tuples = p.map(self._helper_create_dict_in_parallel, list(self.list_of_bbox))
            self.keep_countin_state = False
            stop_background_thread.clear()

        elif config.scl_n_jobs_parallel == 1:
            # single threaded
            list_of_tuples = []
            for i in tqdm(
                range(len(inputs)),
                desc="Single threaded performance " + self.scale.RoadNetwork.city_name + self.scale.scale,
            ):  # input_ in inputs:
                input_, _ = self.list_of_bbox[i]
                k, v = self._helper_create_dict_in_parallel(input_)
                list_of_tuples.append((k, v))
        else:
            raise Exception("Wrong number of threads specified in config file.")

        self.dict_bbox_to_subgraph = dict(list_of_tuples)
        print(time.time() - starttime, "seconds using", config.scl_n_jobs_parallel, "threads")

        print("Before removing empty bboxes", len(self.list_of_bbox))

        empty_bboxes = []
        for key in self.dict_bbox_to_subgraph:
            if self.dict_bbox_to_subgraph[key] == config.rn_no_stats_marker:
                empty_bboxes.append(key)

        for key in empty_bboxes:
            del self.dict_bbox_to_subgraph[key]

        self.list_of_bbox = list(set(self.list_of_bbox) - set(empty_bboxes))
        print("After removing empty bboxes", len(self.list_of_bbox))

        if not os.path.exists(fname):
            with open(fname, "wb") as f:
                pickle.dump(self, f, protocol=config.pickle_protocol)

    def _set_list_of_bbox(self):
        self.list_of_bbox = []

        # X: lon
        # Y: Lat

        self._helper_compute_deltas()

        srn = self.RoadNetwork
        self.delta_x

        start_y = srn.min_y
        num_tiles = int((srn.max_y - srn.min_y) / self.delta_y * int((srn.max_x - srn.min_x) / self.delta_x))
        pbar = tqdm(total=num_tiles, desc="Creating list of bboxes.. ")

        while start_y + self.delta_y <= srn.max_y:
            start_x = srn.min_x
            while start_x + self.delta_x <= srn.max_x:
                # bbox_poly = geometry.box(start_x, start_x + self.delta_x, start_y, start_y + self.delta_y, ccw=True)

                N = start_y + self.delta_y
                S = start_y
                E = start_x + self.delta_x
                W = start_x
                # self.dict_bbox_to_subgraph[(N,S,E,W)] = ox.truncate.truncate_graph_bbox(srn.G_osm, N, S, E, W)
                self.list_of_bbox.append([N, S, E, W])
                pbar.update(1)

                start_x += self.delta_x

            start_y += self.delta_y

        debug_stop = 2
        len_ = len(self.list_of_bbox)
        for i in range(len_):
            self.list_of_bbox[i] = tuple(self.list_of_bbox[i] + [len_])

    def _helper_compute_deltas(self):
        srn = self.RoadNetwork
        diagonal = geodesic((srn.max_y, srn.max_x), ((srn.min_y, srn.min_x))).meters
        y_edge_1 = geodesic((srn.max_y, srn.max_x), ((srn.min_y, srn.max_x))).meters
        x_edge_1 = geodesic((srn.max_y, srn.max_x), ((srn.max_y, srn.min_x))).meters
        y_edge_2 = geodesic((srn.min_y, srn.max_x), ((srn.max_y, srn.max_x))).meters
        x_edge_2 = geodesic((srn.min_y, srn.max_x), ((srn.min_y, srn.min_x))).meters

        # assert < 0.1 % error in distance computation
        err = config.scl_error_percentage_tolerance
        try:
            assert (((y_edge_1 ** 2 + x_edge_1 ** 2) ** 0.5 - diagonal) / diagonal * 100) < err
            assert ((y_edge_1 - y_edge_2) / y_edge_1 * 100) < err
            assert ((x_edge_1 - x_edge_2) / x_edge_1 * 100) < err
        except:
            print((((y_edge_1 ** 2 + x_edge_1 ** 2) ** 0.5 - diagonal) / diagonal * 100))
            print((y_edge_1 - y_edge_2) / y_edge_1 * 100)
            print((x_edge_1 - x_edge_2) / x_edge_1 * 100)
            sys.exit(0)

        self.x_edge = (x_edge_1 + x_edge_2) / 2
        self.y_edge = (y_edge_1 + y_edge_2) / 2
        self.aspect_y_by_x = self.y_edge / self.x_edge

        self.delta_y = (srn.max_y - srn.min_y) / self.scale
        self.delta_x = (srn.max_x - srn.min_x) / self.scale * self.aspect_y_by_x

    def create_file_marker(self):
        with open(
            os.path.join(
                config.BASE_FOLDER,
                "temp",
                self.RoadNetwork.city_name,
                (
                    "temp-"
                    + self.RoadNetwork.city_name
                    + str(self.scale)
                    + "-"
                    + str(int(np.random.rand() * 10000000))
                    + ".marker"
                ),
            ),
            "w",
        ) as f:
            f.write("Done")

    def keep_counting(self):
        oldcount = 0
        with tqdm(total=self.list_of_bbox[0][-1], desc="Counting files ") as pbar:
            while self.keep_countin_state:
                count = len(
                    glob.glob(
                        os.path.join(
                            config.BASE_FOLDER,
                            "temp",
                            self.RoadNetwork.city_name,
                            ("temp-" + self.RoadNetwork.city_name + str(self.scale) + "*.marker"),
                        )
                    )
                )
                assert self.list_of_bbox[0][-1] == len(self.list_of_bbox)
                if config.scl_temp_file_counter:
                    pbar.update(count - oldcount)
                time.sleep(1)
                oldcount = count

    def _helper_create_dict_in_parallel(self, key):
        if config.scl_temp_file_counter:
            self.create_file_marker()
        N, S, E, W, total = key

        def debug_by_plotting_bboxes(color, small_G=None):

            fig, ax = ox.plot.plot_graph(
                self.RoadNetwork.G_osm,
                ax=None,
                figsize=(10, 10),
                bgcolor="white",
                node_color="red",
                node_size=5,
                node_alpha=None,
                node_edgecolor="none",
                node_zorder=1,
                edge_color="black",
                edge_linewidth=0.1,
                edge_alpha=None,
                show=False,
                close=False,
                save=False,
                bbox=None,
            )
            # if color == "green":
            #     ox.plot.plot_graph(
            #     small_G,
            #     ax=ax,
            #     bgcolor="white",
            #     node_color="green",
            #     node_size=10,
            #     node_alpha=None,
            #     node_edgecolor="none",
            #     node_zorder=1,
            #     edge_color="black",
            #     edge_linewidth=0.1,
            #     edge_alpha=None,
            #     show=False,
            #     close=False,
            #     save=False,
            #     bbox=None,
            # )
            rect = plt.Rectangle((W, S), E - W, N - S, facecolor=color, alpha=0.3, edgecolor=None)
            ax.add_patch(rect)
            plt.savefig(
                os.path.join(config.BASE_FOLDER, config.results_folder, "empty_tiles", self.RoadNetwork.city_name)
                + str(self.scale)
                + color
                + str(int(np.random.rand() * 100000000))
                + ".png"
            )
            # plt.show()

        try:
            truncated_graph = smart_truncate(self.RoadNetwork.G_osm, N, S, E, W)
            tile = Tile(
                truncated_graph,
                self.tile_area,
            )
            # if config.verbose >= 2:
            #     with open(os.path.join(config.warnings_folder, "empty_graph_tiles.txt"), "a") as f:
            #         csvwriter = csv.writer(f)
            #         csvwriter.writerow(["ValueError at i: " + str(i) + " " + self.RoadNetwork.city_name])
            debug_by_plotting_bboxes("green", tile.G)
        except (ValueError, Exception):  #
            # if config.verbose >= 1:
            #     with open(os.path.join(config.warnings_folder, "empty_graph_tiles.txt"), "a") as f:
            #         csvwriter = csv.writer(f)
            #         csvwriter.writerow(["ValueError at i: " + str(i) + " " + self.RoadNetwork.city_name])

            debug_by_plotting_bboxes("red")
            return (key, config.rn_no_stats_marker)
        except nx.exception.NetworkXPointlessConcept:
            debug_by_plotting_bboxes("blue")
            return (key, config.rn_no_stats_marker)
            # pass
        return (key, tile)
        # pass

    @staticmethod
    def generate_scales_for_all_cities():
        for city in config.scl_master_list_of_cities:
            for seed in config.scl_list_of_seeds:
                for depth in config.scl_list_of_depths:
                    Scale(RoadNetwork(city), seed ** depth)
                    print(city, seed, depth)
                    loaded_scale = Scale.get_object(cityname=city, scale=seed ** depth)

    @staticmethod
    def get_object(cityname, scale):
        """

        Args:
            scale:

        Returns: (Saved) Object of this class (Scale)

        """
        fname = os.path.join(config.BASE_FOLDER, config.network_folder, cityname, "_scale_" + str(scale) + ".pkl")
        if os.path.exists(fname):
            obj = CustomUnpicklerScale(open(fname, "rb")).load()
        else:
            raise Exception(fname + " not present \n Run speed_data.py and prep_speed.py before running this function")
        return obj


if __name__ == "__main__":
    Scale.generate_scales_for_all_cities()
    debug_stop = 1
