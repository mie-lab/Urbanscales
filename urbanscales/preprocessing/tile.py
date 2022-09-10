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
        self.features = None

    def get_stats_for_tile(self):
        return ox.stats.basic_stats(self.G)

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
