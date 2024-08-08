import csv
import os.path
import pickle
import copy
import osmnx as ox
from osmnx import utils_graph
import contextily as ctx
import geopandas as gpd

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import config
import matplotlib

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from smartprint import smartprint as sprint
from urbanscales.preprocessing.tile import Tile
import geojson
from shapely.geometry import shape
import numpy as np
from shapely.geometry.polygon import Polygon
import time
from shapely.ops import unary_union
import pickle
import geopy.distance as gpy_dist
from shapely.geometry import box

# All custom unpicklers are due to SO user Pankaj Saini's answer:  https://stackoverflow.com/a/51397373/3896008


import geopy.distance
from geopy.point import Point


class SquareFilterShifting:
    """
    A class that adjusts a geographical bounding box by shifting its center or resizing it based on given parameters.

    Attributes:
        N (float): Northern latitude of the bounding box.
        E (float): Eastern longitude of the bounding box.
        S (float): Southern latitude of the bounding box.
        W (float): Western longitude of the bounding box.
    """
    def __init__(self, N, E, S, W):
        """
        Initializes the SquareFilterShifting with the boundaries of a geographic area.

        Parameters:
            N (float): Northern latitude.
            E (float): Eastern longitude.
            S (float): Southern latitude.
            W (float): Western longitude.
        """
        self.N = N
        self.E = E
        self.S = S
        self.W = W

    def filter_square_from_road_network(self, square_side_in_kms, shift_tiles=0):
        """
        Adjusts the center of the bounding box and recalculates its boundaries to form a square with a specified side length.

        Parameters:
            square_side_in_kms (float): The side length of the square in kilometers.
            shift_tiles (int): An option to shift the center of the bounding box.
                               Values range from 0 (no shift) to 8, with each number representing different directions and magnitudes of shift.
        """
        # Calculate the original center of the bounding box
        center = Point((self.N + self.S) / 2, (self.E + self.W) / 2)

        assert shift_tiles in [0,1,2,3,4,5]
        # Shift the center based on shift_tiles value
        if shift_tiles == 1:
            # Shift north by 2/3 km
            new_center = geopy.distance.distance(kilometers=2 / 3).destination(center, bearing=0)
        elif shift_tiles == 2:
            # Shift south by 2/3 km
            new_center = geopy.distance.distance(kilometers=2 / 3).destination(center, bearing=180)
        elif shift_tiles == 3:
            # Shift east by 2/3 km
            new_center = geopy.distance.distance(kilometers=2 / 3).destination(center, bearing=90)
        elif shift_tiles == 4:
            # Shift west by 2/3 km
            new_center = geopy.distance.distance(kilometers=2 / 3).destination(center, bearing=270)
        elif shift_tiles == 5:
            # Shift north by 2/3 km
            new_center = geopy.distance.distance(kilometers=2 / 7).destination(center, bearing=0)
        elif shift_tiles == 6:
            # Shift south by 2/3 km
            new_center = geopy.distance.distance(kilometers=2 / 7).destination(center, bearing=180)
        elif shift_tiles == 7:
            # Shift east by 2/3 km
            new_center = geopy.distance.distance(kilometers=2 / 7).destination(center, bearing=90)
        elif shift_tiles == 8:
            # Shift west by 2/3 km
            new_center = geopy.distance.distance(kilometers=2 / 7).destination(center, bearing=270)
        elif shift_tiles == 0:
            # No shift
            new_center = center

        center = new_center

        # Calculate half the side's length in kilometers
        half_side = square_side_in_kms / 2 ** 0.5

        # Find new NE and SW corners by moving from the center
        # North-East corner
        ne_corner = geopy.distance.distance(kilometers=half_side).destination(center, bearing=45)
        self.N, self.E = ne_corner.latitude, ne_corner.longitude

        # South-West corner
        sw_corner = geopy.distance.distance(kilometers=half_side).destination(center, bearing=225)
        self.S, self.W = sw_corner.latitude, sw_corner.longitude


class CustomUnpicklerRoadNetworkShifting(pickle.Unpickler):
    """
    A custom unpickler for loading objects of RoadNetworkShifting class, allowing for safe loading of pickle files.

    Methods:
        find_class: Overrides the default method to ensure the correct class is used during unpickling.
    """
    def find_class(self, module, name):
        """
        Ensures that during unpickling, the correct class is used instead of a potentially harmful or incorrect class.

        Parameters:
            module (str): The module from which to load the class.
            name (str): The name of the class to be loaded.

        Returns:
            type: The class type that matches the given name, specifically checking for 'RoadNetwork'.
        """
        if name == "RoadNetwork":
            return RoadNetworkShifting
        return super().find_class(module, name)


class RoadNetworkShifting:
    """
    Manages the road network data for a specific city, including loading, modifying, and saving the network.

    Attributes:
        rn_fname (str): The filename where the road network data is stored.
        osm_pickle (str): The filename suffix for the OpenStreetMap pickle data.
        city_name (str): The name of the city for which the road network is managed.
        N, E, S, W (float): Geographic coordinates defining the bounding box of the road network.
        G_osm (networkx.MultiDiGraph): The graph object representing the road network.
    """
    def __init__(self, cityname, mode_of_retreival="bbox"):
        """
        Initializes the RoadNetworkShifting with city name and retrieval mode.

        Parameters:
            cityname (str): The name of the city.
            mode_of_retrieval (str): The mode of data retrieval; 'place' to use place names, 'bbox' to use a bounding box.
        """
        self.rn_fname = os.path.join(
            config.BASE_FOLDER, config.network_folder, cityname, config.rn_post_fix_road_network_object_file
        )
        self.osm_pickle = config.rn_post_fix_road_network_object_file
        if config.rn_delete_existing_pickled_objects:
            try:
                os.remove(self.rn_fname)
            except OSError:
                pass

        if os.path.exists(self.rn_fname):
            with open(self.rn_fname, "rb") as f:
                # temp = copy.deepcopy(pickle.load(f))
                ss = time.time()
                temp = copy.deepcopy(CustomUnpicklerRoadNetworkShifting(open(self.rn_fname, "rb")).load())
                self.__dict__.update(temp.__dict__)
                print("Road network loading time: ", time.time() - ss)

        else:
            self.city_name = cityname
            self.N, self.E, self.S, self.W = config.rn_city_wise_bboxes[cityname]

            sprint(self.N, self.E, self.S, self.W)
            if config.rn_percentage_of_city_area != 100:
                self.filter_a_patch_from_road_network(config.rn_percentage_of_city_area)

            if config.rn_square_from_city_centre != -1:
                self.filter_a_square_from_road_network(config.rn_square_from_city_centre)

            sprint(self.N, self.E, self.S, self.W)

            self.G_osm = None

            assert mode_of_retreival.lower() in ["place", "bbox"]
            if mode_of_retreival == "place":
                self.set_get_osm = self.get_osm_from_place
            elif mode_of_retreival == "bbox":
                self.set_get_osm = self.get_osm_from_bbox

            self.set_get_osm()

            self.G_OSM_nodes, self.G_OSM_edges = utils_graph.graph_to_gdfs(self.G_osm)

            if not config.rn_do_not_filter:
                if self.city_name not in config.rn_do_not_filter_list:
                    self.filter_OSM()

            if config.rn_compute_full_city_features:
                self.set_graph_features()

            self.set_boundaries_x_y()
            if config.rn_add_edge_speed_and_tt:
                self.G_osm = ox.speed.add_edge_speeds(self.G_osm)
                self.G_osm = ox.speed.add_edge_travel_times(self.G_osm)
            self.save_road_network_object()

    def save_road_network_object(self):
        """
         Saves the current state of the road network to a file using a secure temporary naming scheme to prevent data corruption.
         """
        rand_pickle_marker = os.path.join(config.temp_folder_for_robust_pickle_files,
                                          str(int(np.random.rand() * 100000000000000)))
        with open(rand_pickle_marker, "wb") as f:
            pickle.dump(self, f, protocol=config.pickle_protocol)
        os.rename(rand_pickle_marker, self.rn_fname)

    def get_osm_from_place(self):
        """
        Retrieves the OpenStreetMap data for the specified city using its place name.

        Returns:
            networkx.MultiDiGraph: The graph object representing the road network.
        """
        if self.G_osm == None:
            fname = os.path.join(config.BASE_FOLDER, config.network_folder, self.city_name, self.osm_pickle)
            if os.path.isfile(fname):
                with open(fname, "rb") as f1:
                    self.G_osm = pickle.load(f1)
            else:
                self.G_osm = ox.graph_from_place(
                    self.city_name, network_type="drive", simplify=config.rn_simplify, retain_all=True
                )
                rand_pickle_marker = os.path.join(config.temp_folder_for_robust_pickle_files,
                                                  str(int(np.random.rand() * 100000000000000)))
                with open(rand_pickle_marker, "wb") as f:
                    pickle.dump(self.G_osm, f, protocol=config.pickle_protocol)
                os.rename(rand_pickle_marker, fname)

        return self.G_osm

    def get_osm_from_bbox(self):
        """
        Retrieves the OpenStreetMap data for the specified city using a bounding box defined by the class attributes N, E, S, and W.

        Returns:
            networkx.MultiDiGraph: The graph object representing the road network.
        """
        if self.G_osm == None:
            fname = os.path.join(config.BASE_FOLDER, config.network_folder, self.city_name, self.osm_pickle)

            # create the directory structure if it doesn't exists
            if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, self.city_name)):
                if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder)):
                    os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder))
                os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder, self.city_name))

            if os.path.isfile(fname) and not config.rn_delete_existing_pickled_objects:
                with open(fname, "rb") as f1:
                    self.G_osm = pickle.load(f1)
            else:
                try:
                    self.G_osm = ox.graph_from_bbox(
                        self.N,
                        self.S,
                        self.E,
                        self.W,
                        network_type="drive",
                        simplify=config.rn_simplify,
                        retain_all=True,
                    )
                except Exception as e:
                    sprint(self.city_name, self.N, self.S, self.E, self.W)
                    # raise Exception(e)
                    print("Exiting execution; graph not fetched from OSM")
                    # sys.exit(0)

                rand_pickle_marker = os.path.join(config.temp_folder_for_robust_pickle_files,
                                                  str(int(np.random.rand() * 100000000000000)))
                with open(rand_pickle_marker, "wb") as f:
                    pickle.dump(self.G_osm, f, protocol=config.pickle_protocol)
                os.rename(rand_pickle_marker, fname)


        return self.G_osm

    def get_osm_from_address(self):
        """
        Retrieves the OpenStreetMap data for the specified city using an address string.

        Returns:
            networkx.MultiDiGraph: The graph object representing the road network.
        """
        fname = os.path.join(config.BASE_FOLDER, config.network_folder, self.city_name, self.osm_pickle)

        # create the directory structure if it doesn't exists
        if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, self.city_name)):
            if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder)):
                os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder))
            os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder, self.city_name))

        if os.path.isfile(fname) and not config.rn_delete_existing_pickled_objects:
            with open(fname, "rb") as f1:
                self.G_osm = pickle.load(f1)
        else:
            self.G_osm = ox.graph_from_address(self.city_name, network_type="drive")
            rand_pickle_marker = os.path.join(config.temp_folder_for_robust_pickle_files,
                                              str(int(np.random.rand() * 100000000000000)))
            with open(rand_pickle_marker, "wb") as f:
                pickle.dump(self.G_osm, f, protocol=config.pickle_protocol)
            os.rename(rand_pickle_marker, fname)

        return self.G_osm

    def set_graph_features(self):
        """
        Computes and assigns statistical features to the road network graph, such as connectivity and centrality measures.
        """
        assert self.G_osm is not None
        self.graph_features = Tile(self.G_osm).set_stats_for_tile()

    def get_graph_features_as_list(self):
        """
        Compiles and returns a list of statistical features of the road network for analytical or reporting purposes.

        Returns:
            list: A list of feature values associated with the road network.
        """
        assert config.rn_square_from_city_centre != -1
        list_of_graph_features = Tile(None, config.rn_square_from_city_centre**2).get_list_of_features()
        list_of_values = [self.city_name]
        for key in list_of_graph_features:
            list_of_values.append(self.graph_features[key])
        return list_of_values

    def plot_basemap(self):
        """
        Generates and saves a graphical representation of the road network with a basemap for visualization.
        """
        if not config.rn_plotting_enabled:
            return

        filepath = os.path.join(config.BASE_FOLDER, config.network_folder, self.city_name, config.rn_base_map_filename)
        ox.plot.plot_graph(
            self.G_osm,
            ax=None,
            figsize=(12, 8),
            bgcolor="white",
            node_color="black",
            node_size=0.1,
            node_alpha=None,
            node_edgecolor="none",
            node_zorder=1,
            edge_color="black",
            edge_linewidth=0.1,
            edge_alpha=None,
            show=True,
            close=False,
            save=True,
            bbox=None,
            filepath=filepath,
            dpi=300,
        )

    def set_boundaries_x_y(self):
        """
        Updates the geographic boundaries of the road network based on the extents of the nodes in the graph.
        """
        min_x = 99999999
        min_y = 99999999
        max_x = -99999999
        max_y = -99999999
        for node in list(self.G_osm.nodes._nodes.values()):
            min_x, max_x = min(min_x, node["x"]), max(max_x, node["x"])
            min_y, max_y = min(min_y, node["y"]), max(max_y, node["y"])
        print(min_x, max_x, min_y, max_y)

        # self.min_x, self.max_x, self.min_y, self.max_y = (min_x, max_x, min_y, max_y)
        self.min_x, self.max_x, self.min_y, self.max_y = (self.W, self.E, self.S, self.N)

    def get_geojson_file_to_single_polygon(self):
        """
        Converts a GeoJSON file of the city's boundaries into a single Shapely Polygon object for use in spatial operations.

        Returns:
            shapely.geometry.Polygon: A polygon representing the city's geographic boundaries.
        """
        fname = os.path.join(
            config.BASE_FOLDER,
            config.network_folder,
            config.rn_prefix_geojson_files + self.city_name.lower() + config.rn_postfix_geojson_files,
        )
        with open(fname) as f:
            geojs = geojson.load(f)

        # x, y = shape(geojs["features"][0]["geometry"]).convex_hull.boundary.xy
        list_of_geoms = []
        for i in range(len(geojs["features"])):
            list_of_geoms.append(shape(geojs["features"][i]["geometry"]))
        x, y = unary_union(list_of_geoms).convex_hull.boundary.xy

        x = x.tolist()
        y = y.tolist()
        x = list(np.array(x).astype(float))
        y = list(np.array(y).astype(float))

        plt.plot(x, y)
        plt.savefig(
            os.path.join(config.BASE_FOLDER, config.network_folder, self.city_name, "polygon_convex.png"), dpi=300
        )

        convex_hull_poly = Polygon([[p[0], p[1]] for p in zip(x, y)])

        return convex_hull_poly

    def filter_OSM(self):
        """
        Applies a geographic filter to the road network, truncating it to fit within a defined polygon representing the city's boundaries.

        Returns:
            networkx.MultiDiGraph: The filtered road network graph.
        """
        try:
            self.G_osm = ox.truncate.truncate_graph_polygon(self.G_osm, self.get_geojson_file_to_single_polygon())
        except ValueError:
            with open("ERROR_getting_smaller_network.txt", "a") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow([self.city_name, "Reverting_to_bigger_network"])
        return self.G_osm

    def filter_a_patch_from_road_network(self, percentage):
        """
        Reduces the road network to a specified percentage of its original geographic extent, centered on the middle of the bounding box.

        Parameters:
            percentage (float): The percentage of the original dimensions to retain, specified as a percentage of each side's length.
        """
        ns = self.N - self.S
        ns_center = self.S + ns / 2
        ew = self.E - self.W
        ew_center = self.W + ew / 2

        self.N = ns_center + ns * 0.5 * (percentage / 100)
        self.S = ns_center - ns * 0.5 * (percentage / 100)

        self.E = ew_center + ew * 0.5 * (percentage / 100)
        self.W = ew_center - ew * 0.5 * (percentage / 100)

    def filter_a_square_from_road_network(self, square_side_in_kms):
        """
        Reduces the road network to a specified percentage of its original geographic extent, centered on the middle of the bounding box.

        Parameters:
            percentage (float): The percentage of the original dimensions to retain, specified as a percentage of each side's length.
        """
        square_filter = SquareFilterShifting(N=self.N, E=self.E, S=self.S, W=self.W)
        square_filter.filter_square_from_road_network(square_side_in_kms)
        self.N, self.S, self.E, self.W = square_filter.N, square_filter.S, square_filter.E, square_filter.W

    def __repr__(self):
        """
        Provides a string representation of the RoadNetworkShifting object, detailing its current state.

        Returns:
            str: A string representation of the RoadNetworkShifting object.
        """
        class_attrs = vars(self)
        repr_str = f"{self.__class__.__name__}(\n"
        for attr, value in class_attrs.items():
            repr_str += f"  {attr}: {value}\n"
        repr_str += ")"
        return repr_str



    @staticmethod
    def generate_road_nw_object_for_all_cities():
        """
        Generates and saves road network objects and their graphical features for all cities specified in the configuration.
        This method handles directory creation, road network retrieval, feature calculation, and data logging for each city.
        """
        if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder)):
            os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder))

        with open(os.path.join(config.BASE_FOLDER, config.network_folder, "all_cities_graph_features.csv"), "w") as f:
            csvwriter = csv.writer(f)
            # if config.rn_compute_full_city_features:
            assert config.rn_square_from_city_centre != -1
            list_of_graph_features = Tile(None, config.rn_square_from_city_centre**2).get_feature_names()
            csvwriter.writerow(["city"] + list_of_graph_features)

            for city in config.rn_master_list_of_cities:
                if not os.path.exists(os.path.join(config.BASE_FOLDER, config.network_folder, city)):
                    os.mkdir(os.path.join(config.BASE_FOLDER, config.network_folder, city))

                starttime = time.time()

                print(city)
                rn = RoadNetworkShifting(city, "bbox")
                if config.rn_plot_basemap:
                    rn.plot_basemap()

                if config.rn_compute_full_city_features:
                    csvwriter.writerow(rn.get_graph_features_as_list())

                print(time.time() - starttime)



    @staticmethod
    def get_object(city):
        """
        Retrieves an instance of the RoadNetworkShifting for a specific city. This method ensures that a road network object
        is consistently retrieved with the latest configuration and updates.

        Parameters:
            city (str): The name of the city for which to retrieve the road network object.

        Returns:
            RoadNetworkShifting: An instance of the RoadNetworkShifting class loaded with the road network data for the specified city.
        """
        rn = RoadNetworkShifting(city)
        return rn


if __name__ == "__main__":
    RoadNetworkShifting.generate_road_nw_object_for_all_cities()
