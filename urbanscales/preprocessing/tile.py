import csv
import sys

from tqdm import tqdm

import warnings

import networkx as nx
import numpy as np
from osmnx import stats as osxstats
import osmnx as ox
import os
import config
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import config

from smartprint import smartprint as sprint
from slugify import slugify


class Tile:

    # static variable FEATURE_NAMES below
    FEATURE_NAMES = [
        'n', 'm', 'k_avg', 'edge_length_total', 'edge_length_avg',
        'streets_per_node_avg', 'street_length_total', 'street_segment_count',
        'street_length_avg', 'circuity_avg', 'self_loop_proportion',
        'metered_count', 'non_metered_count', 'total_crossings',
        'betweenness', 'mean_lanes', 'lane_density'
    ]

    # Initializing feature names for streets_per_node_counts and streets_per_node_proportions
    for i in range(1, 7):
        FEATURE_NAMES.append(f'streets_per_node_count_{i}')
        FEATURE_NAMES.append(f'streets_per_node_proportion_{i}')

    def __init__(self, G: nx.MultiDiGraph, tile_area):
        """

        Args:
            G:
            tile_area: is needed to compute lane density
        """
        if not isinstance(G, nx.MultiDiGraph):
            warnings.warn("Allowing empty graph tile")
            self.G = None
            self.tile_area = tile_area

        self.G = G

        if self.G is not None:
            # because we use some functions of this class for
            # other purposes as well; For example, for empty
            # graphs as well to simply view the list of features
            if not config.tls_garbage_Test_Speed:
                self.set_basic_stats_for_tile()

                if config.tls_betweenness_features:
                    self.set_betweenness_centrality_local()
                if config.tls_add_edge_speed_and_tt:
                    self.set_average_edge_speed()
                if config.tls_number_of_lanes:
                    self.set_number_of_lanes()
                if config.tls_add_metered_intersections:
                    self.set_intersection_count()

                self.tile_area = tile_area
                self.set_lane_density()

            else:
                for feature in self.FEATURE_NAMES:
                    # eval ("self." + feature + " = 0")
                    setattr(self, feature, "garbage_value")
                    print ("Garbage Tile values set to 0 throughout for all features")

            # print ("All values fixed!")

    def set_basic_stats_for_tile(self):
        basic_stats = ox.stats.basic_stats(self.G)

        # Map the basic_features output to class attributes
        self.n = basic_stats.get('n', np.nan)
        self.m = basic_stats.get('m', np.nan)
        self.k_avg = basic_stats.get('k_avg', np.nan)
        self.edge_length_total = basic_stats.get('edge_length_total', np.nan)
        self.edge_length_avg = basic_stats.get('edge_length_avg', np.nan)
        self.streets_per_node_avg = basic_stats.get('streets_per_node_avg', np.nan)
        self.streets_per_node_counts = [
            basic_stats['streets_per_node_counts'].get(i, 0) for i in range(6)
        ]
        self.streets_per_node_proportions = [
            basic_stats['streets_per_node_proportions'].get(i, 0) for i in range(6)
        ]
        self.street_length_total = basic_stats.get('street_length_total', np.nan)
        self.street_segment_count = basic_stats.get('street_segment_count', np.nan)
        self.street_length_avg = basic_stats.get('street_length_avg', np.nan)
        self.circuity_avg = basic_stats.get('circuity_avg', np.nan)
        self.self_loop_proportion = basic_stats.get('self_loop_proportion', np.nan)

    def set_intersection_count(self):
        # Extract node attributes
        data = list(self.G.nodes().items())

        # Lists to store colors and sizes for each node
        node_colors = []
        node_sizes = []

        for _, attributes in data:
            if 'highway' in attributes:
                if attributes['highway'] == 'traffic_signals':
                    node_colors.append('red')
                    node_sizes.append(100)  # size for traffic signal
                elif attributes['highway'] == 'crossing':
                    node_colors.append('blue')
                    node_sizes.append(100)  # size for crossings
                else:
                    node_colors.append('gray')  # default node color
                    node_sizes.append(25)  # default size
            else:
                node_colors.append('gray')  # default node color
                node_sizes.append(25)  # default size

        # Count the occurrences of each color
        metered_count = node_colors.count('red')  # traffic_signals are considered as "metered"
        non_metered_count = node_colors.count('blue')  # crossings are considered as "non-metered"

        self.metered_count = metered_count
        self.non_metered_count = non_metered_count
        self.total_crossings = metered_count + non_metered_count

    def set_betweenness_centrality_local(self):
        if self.G.__len__() > 20:
            K = 20
            iterations = 20
            initial_bc = nx.betweenness_centrality(self.G, k=K)
            all_nodes = list(initial_bc.keys())

            # Dictionary to store the mean betweenness centrality values for each node up to each iteration
            mean_values_per_node = {node: [] for node in all_nodes}

            for i in tqdm(range(iterations), desc="Computing betweenness using iterations.."):
                bc = nx.betweenness_centrality(self.G, k=K)
                for node in all_nodes:
                    if i == 0:
                        current_mean = bc[node]
                    else:
                        current_mean = (mean_values_per_node[node][-1] * i + bc[node]) / (i + 1)
                    mean_values_per_node[node].append(current_mean)
            overall_mean_bc_per_node = {node: sum(values) / len(values) for node, values in
                                        mean_values_per_node.items()}

            dict_between = overall_mean_bc_per_node
            self.betweenness = np.mean(list(dict_between.values()))
        else:
            lict_of_centralities = list(nx.betweenness_centrality(self.G).values())
            self.betweenness = np.mean(lict_of_centralities)

    def get_betweenness_centrality_global(self, dict_node_to_bc):
        # Filter out nodes from the dict that are also in self.G
        valid_bc_values = [bc for node, bc in dict_node_to_bc.items() if node in self.G.nodes()]
        return np.mean(valid_bc_values)

    def set_number_of_lanes(self):
        edges = ox.graph_to_gdfs(self.G, nodes=False, edges=True)
        if not os.path.exists(config.warnings_folder):
            os.mkdir(config.warnings_folder)
        try:
            self.mean_lanes = np.mean(list(edges["lanes"].value_counts()))
            if config.verbose >= 2:
                with open(os.path.join(config.warnings_folder, "lane_present.txt"), "a") as f:
                    csvwriter = csv.writer(f)
                    csvwriter.writerow(["present"])
        except:
            if config.verbose >= 1:
                with open(os.path.join(config.warnings_folder, "lane_absent.txt"), "a") as f:
                    csvwriter = csv.writer(f)
                    csvwriter.writerow(["missing; set to 2"])
                    self.mean_lanes = config.tls_missing_lanes_default_value

    def set_lane_density(self):
        """
        Our tile element does not have the area information directly. It has the corresponding G_OSM, hence the area
            can be computed but that will be a repetition of finding limiting X and Y and compute areas using
            pyproj; Instead we will use the area element using the Scale class (since scale class knows the complete
            area, we can just split it into smaller regions using the global estimate)

        Args:
            tile_area:

        Returns:

        """
        self.lane_density = self.mean_lanes / self.tile_area

    def set_average_edge_speed(self):
        pass

    def get_features(self):
        """
        This function returns a list of the computed features.
        """
        features = [
            self.n,
            self.m,
            self.k_avg,
            self.edge_length_total,
            self.edge_length_avg,
            self.streets_per_node_avg,
            self.street_length_total,
            self.street_segment_count,
            self.street_length_avg,
            self.circuity_avg,
            self.self_loop_proportion,
            self.metered_count,
            self.non_metered_count,
            self.total_crossings,
            self.betweenness,
            self.mean_lanes,
            self.lane_density
            # Add other features as needed
        ]

        # Appending streets_per_node_counts as individual features
        for count in self.streets_per_node_counts:
            features.append(count)

        # Appending streets_per_node_proportions as individual features
        for proportion in self.streets_per_node_proportions:
            features.append(proportion)

        return features



    @classmethod
    def get_feature_names(cls):
        """
        This function returns a list of names of the computed features.
        """
        return cls.FEATURE_NAMES

    def __repr__(self):
        return (
            f"Tile("
            f"circuity_avg={self.circuity_avg}, "
            f"edge_length_avg={self.edge_length_avg}, "
            f"k_avg={self.k_avg}, "
            f"m={self.m}, "
            f"n={self.n}, "
            f"self_loop_proportion={self.self_loop_proportion}, "
            f"street_length_avg={self.street_length_avg}, "
            f"street_segment_count={self.street_segment_count}, "
            f"streets_per_node_avg={self.streets_per_node_avg}, "
            f"streets_per_node_counts={self.streets_per_node_counts}, "
            f"streets_per_node_proportions={self.streets_per_node_proportions}, "
            f"betweenness={self.betweenness}, "
            f"mean_lanes={self.mean_lanes}, "
            f"metered_count={self.metered_count}, "
            f"non_metered_count={self.non_metered_count}, "
            f"total_crossings={self.total_crossings})"
        )

if __name__ == "__main__":
    point = 40.70443736361541, -73.93851957710785
    dist = 300
    Gu = ox.graph_from_point(point, dist=dist, network_type="drive", simplify=False)

    tile = Tile(Gu, config.rn_square_from_city_centre**2)
    print(tile)
    sprint (tile.get_feature_names())
    sprint (len(tile.get_features()))
    print ("Now trying out the static functionality")
    sprint (Tile.get_feature_names())
    sprint(len(Tile.get_feature_names()))
