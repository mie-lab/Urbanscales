import os
import pickle
import time
import warnings

import networkx as nx
import osmnx as ox
from geopy.distance import geodesic
from smartprint import smartprint as sprint
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import config
from urbanscales.io.road_network import RoadNetwork
from urbanscales.preprocessing.tile import Tile


class Scale:
    def __init__(self, RoadNetwork, scale):
        self.RoadNetwork = RoadNetwork
        self.scale = scale
        self.set_bbox_sub_G_map()

    def set_bbox_sub_G_map(self, save_to_pickle=True):
        fname = os.path.join("network", self.RoadNetwork.city_name, "_scale_" + str(self.scale) + ".pkl")
        if os.path.exists(fname):
            # do nothing
            return

        self._set_list_of_bbox()
        self.dict_bbox_to_subgraph = {}

        inputs = list(range(len(self.list_of_bbox)))
        starttime = time.time()
        list_of_tuples = process_map(
            self._helper_create_dict_in_parallel, inputs, max_workers=config.scl_n_jobs_parallel
        )
        self.dict_bbox_to_subgraph = dict(list_of_tuples)
        print(time.time() - starttime, "seconds using", config.scl_n_jobs_parallel, "threads")

        sprint(len(self.list_of_bbox))

        empty_bboxes = []
        for key in self.dict_bbox_to_subgraph:
            if self.dict_bbox_to_subgraph[key] == "Empty":
                empty_bboxes.append(key)

        for key in empty_bboxes:
            del self.dict_bbox_to_subgraph[key]

        self.list_of_bbox = list(set(self.list_of_bbox) - set(empty_bboxes))
        sprint(len(self.list_of_bbox))

        if not os.path.exists(fname):
            with open(fname, "wb") as f:
                pickle.dump(self, f, protocol=config.pickle_protocol)

    def get_object_at_scale(cityname, scale):
        """

        Args:
            scale:

        Returns: (Saved) Object of this class (Scale)

        """
        fname = os.path.join("network", cityname, "_scale_" + str(scale) + ".pkl")
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                obj = pickle.load(f)
        else:
            raise Exception(fname + " not present ")
        return obj

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
                self.list_of_bbox.append((N, S, E, W))
                pbar.update(1)

                start_x += self.delta_x

            start_y += self.delta_y

        debug_stop = 2

    def _helper_compute_deltas(self):
        srn = self.RoadNetwork
        diagonal = geodesic((srn.max_y, srn.max_x), ((srn.min_y, srn.min_x))).meters
        y_edge_1 = geodesic((srn.max_y, srn.max_x), ((srn.min_y, srn.max_x))).meters
        x_edge_1 = geodesic((srn.max_y, srn.max_x), ((srn.max_y, srn.min_x))).meters
        y_edge_2 = geodesic((srn.min_y, srn.max_x), ((srn.max_y, srn.max_x))).meters
        x_edge_2 = geodesic((srn.min_y, srn.max_x), ((srn.min_y, srn.min_x))).meters

        # assert < 0.1 % error in distance computation
        assert (((y_edge_1 ** 2 + x_edge_1 ** 2) ** 0.5 - diagonal) / diagonal * 100) < 0.1
        assert ((y_edge_1 - y_edge_2) / y_edge_1 * 100) < 0.1
        assert ((x_edge_1 - x_edge_2) / x_edge_1 * 100) < 0.1

        self.x_edge = (x_edge_1 + x_edge_2) / 2
        self.y_edge = (y_edge_1 + y_edge_2) / 2
        self.aspect_y_by_x = self.y_edge / self.x_edge

        self.delta_y = (srn.max_y - srn.min_y) / self.scale
        self.delta_x = (srn.max_x - srn.min_x) / self.scale * self.aspect_y_by_x

    def _helper_create_dict_in_parallel(self, i):
        key = self.list_of_bbox[i]

        N, S, E, W = key
        try:
            tile = Tile(ox.truncate.truncate_graph_bbox(self.RoadNetwork.G_osm, N, S, E, W))
            # self.dict_bbox_to_subgraph[key] =

        except (ValueError, nx.exception.NetworkXPointlessConcept):
            warnings.warn("ValueError at i: " + str(i))
            print("ValueError at i: " + str(i))
            # self.dict_bbox_to_subgraph[key] = "Empty"
            # empty_bboxes.append(key)
            # continue
            return (key, "Empty")
        return (key, tile)


if __name__ == "__main__":

    for seed in [2, 3]:  # , 4, 5, 6, 7]:
        for depth in range(2, 4):
            Scale(RoadNetwork("Singapore"), seed ** depth)
            loaded_scale = Scale.get_object_at_scale(cityname="Singapore", scale=seed ** depth)
    debug_stop = 1
