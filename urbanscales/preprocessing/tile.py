import sys
import warnings

import networkx as nx
import numpy as np
from osmnx import stats as osxstats
import osmnx as ox

import config


class Tile:
    def __init__(self, G: nx.MultiDiGraph):
        if not isinstance(G, nx.MultiDiGraph):
            warnings.warn("Allowing empty graph tile")
            self.G = None

        self.G = G

        if self.G is not None:
            # cuz we use some functions of this class for
            # other purposes as well; For example, for empty
            # graphs as well to simply view the list of features
            self.set_stats_for_tile()

    def get_stats_for_tile(self):
        return ox.stats.basic_stats(self.G)

    def set_betweenness_centrality(self):
        self.betweenness = np.mean(nx.betweenness_centrality(self.G).values())

    def set_number_of_lanes(self):
        edges = ox.graph_to_gdfs(self.G, nodes=False, edges=True)
        self.mean_lanes = np.mean((edges["lanes"].value_counts()))

    def set_lane_density(self, tile_area):
        """
        Our tile element does not have the area information directly. It has the corresponding G_OSM, hence the area
            can be computed but that will be a repetition of finding limiting X and Y and compute areas using
            pyproj; Instead we will use the area element using the Scale class (since scale class knows the complete
            area, we can just split it into smaller regions using the global estimate)

        Args:
            tile_area:

        Returns:

        """
        self.lane_density = self.mean_lanes / tile_area

    def set_mean_speed_and_tt(self):
        pass

    def get_vector_of_features(self):
        X = []
        stats = self.get_stats_for_tile()
        for key in [
            "circuity_avg",
            "edge_length_avg",
            "intersection_count",
            "k_avg",
            "m",
            "n",
            "self_loop_proportion",
            "street_length_avg",
            "street_segment_count",
            "streets_per_node_avg",
        ]:
            X.append(stats[key])

        for key in ["streets_per_node_counts", "streets_per_node_proportions"]:
            for i in range(6):
                if i in stats[key]:
                    X.append(stats[key][i])
                else:
                    X.append(0)
        if config.tls_betweenness_features:
            self.set_betweenness_centrality()
            X.append(self.betweenness)
        if config.tls_number_of_lanes:
            self.set_number_of_lanes()
            X.append(self.mean_lanes)
        if config.tls_add_edge_speed_and_tt:
            self.set_average_edge_speed()
            X.append(self.mean_speed)
            self.set_average_tt()
            X.append(self.mean_tt)
        return X

    @staticmethod
    def get_feature_names():
        f = [
            "circuity_avg",
            "edge_length_avg",
            "intersection_count",
            "k_avg",
            "m",
            "n",
            "self_loop_proportion",
            "street_length_avg",
            "street_segment_count",
            "streets_per_node_avg",
        ]
        f = (
            f
            + ["streets_per_node_counts_" + str(i) for i in range(6)]
            + ["streets_per_node_proportions" + str(i) for i in range(6)]
        )
        if config.tls_betweenness_features:
            f.append("betweenness")
        if config.tls_number_of_lanes:
            f.append("mean_lanes")
        if config.tls_add_edge_speed_and_tt:
            f.append("mean_speed")
            f.append("mean_tt")
        return f

    def set_stats_for_tile(self):
        self.basic_features = self.get_stats_for_tile()
        return self.basic_features

    def get_list_of_features(self):
        point = 40.70443736361541, -73.93851957710785
        dist = 300
        G = ox.graph_from_point(point, dist=dist, network_type="drive")
        tile = Tile(G)
        return list(((tile.get_stats_for_tile().keys())))
        # def savefig


if __name__ == "__main__":

    point = 40.70443736361541, -73.93851957710785
    dist = 300
    G = ox.graph_from_point(point, dist=dist, network_type="drive")

    tile = Tile(G)
    print(tile.get_stats_for_tile())
