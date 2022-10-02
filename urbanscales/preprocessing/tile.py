import sys
import warnings

import networkx as nx
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
        return X

    def get_feature_names(self):
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
        return f

    def set_stats_for_tile(self):
        self.features = self.get_stats_for_tile()

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
