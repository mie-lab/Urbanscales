import csv
import os.path
import pickle
import copy
import osmnx as ox
import config
import matplotlib.pyplot as plt
from smartprint import smartprint as sprint
from urbanscales.preprocessing.tile import Tile
import geojson
from shapely.geometry import shape
import numpy as np
from shapely.geometry.polygon import Polygon
import time
from shapely.ops import unary_union


class RoadNetwork:
    def __init__(self, cityname, mode_of_retreival="bbox"):
        """
        Example: SG = RoadNetwork("Singapore")
        mode_of_retreival = "place"/"bbox"
        """
        self.rn_fname = os.path.join("network", cityname, config.rn_post_fix_road_network_object_file)
        self.osm_pickle = "_OSM_pickle"
        if config.rn_delete_existing_pickled_objects:
            try:
                os.remove(self.rn_fname)
            except OSError:
                pass

        if os.path.exists(self.rn_fname):
            with open(self.rn_fname, "rb") as f:
                temp = copy.deepcopy(pickle.load(f))
                self.__dict__.update(temp.__dict__)

        else:
            self.city_name = cityname
            self.N, self.E, self.S, self.W = config.rn_city_wise_bboxes[cityname]
            self.G_osm = None

            assert mode_of_retreival.lower() in ["place", "bbox"]
            if mode_of_retreival == "place":
                self.set_get_osm = self.get_osm_from_place
            elif mode_of_retreival == "bbox":
                self.set_get_osm = self.get_osm_from_bbox

            self.set_get_osm()
            if self.city_name not in config.rn_do_not_filter_list:
                self.filter_OSM()
            self.set_graph_features()
            self.set_boundaries_x_y()
            self.save_road_network_object()

    def save_road_network_object(self):
        with open(self.rn_fname, "wb") as f:
            pickle.dump(self, f, protocol=config.pickle_protocol)

    def get_osm_from_place(self):
        if self.G_osm == None:
            fname = os.path.join("network", self.city_name, self.osm_pickle)
            if os.path.isfile(fname):
                with open(fname, "rb") as f1:
                    self.G_osm = pickle.load(f1)
            else:
                self.G_osm = ox.graph_from_place(self.city_name, network_type="drive")
                with open(fname, "wb") as f:
                    pickle.dump(self.G_osm, f, protocol=config.pickle_protocol)
        return self.G_osm
        # G = ox.speed.add_edge_speeds(G)
        # G = ox.speed.add_edge_travel_times(G)

    def get_osm_from_bbox(self):
        if self.G_osm == None:
            fname = os.path.join("network", self.city_name, self.osm_pickle)
            if os.path.isfile(fname):
                with open(fname, "rb") as f1:
                    self.G_osm = pickle.load(f1)
            else:

                self.G_osm = ox.graph_from_bbox(
                    self.N, self.S, self.E, self.W, network_type="drive", simplify=True, retain_all=False
                )

                with open(fname, "wb") as f:
                    pickle.dump(self.G_osm, f, protocol=config.pickle_protocol)
        return self.G_osm

    def get_osm_from_address(self):
        fname = os.path.join("network", self.city_name, self.osm_pickle)
        if os.path.isfile(fname):
            with open(fname, "rb") as f1:
                self.G_osm = pickle.load(f1)
        else:
            self.G_osm = ox.graph_from_address(self.city_name, network_type="drive")
            with open(fname, "wb") as f:
                pickle.dump(self.G_osm, f, protocol=config.pickle_protocol)
        return self.G_osm

    def set_graph_features(self):
        assert self.G_osm is not None
        self.graph_features = Tile(self.G_osm).set_stats_for_tile()

    def plot_basemap(self):
        if not config.rn_plotting_enabled:
            return

        filepath = os.path.join("network", self.city_name, config.rn_base_map_filename)
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
            dpi =300
        )


    def set_boundaries_x_y(self):
        min_x = 99999999
        min_y = 99999999
        max_x = -99999999
        max_y = -99999999
        for node in list(self.G_osm.nodes._nodes.values()):
            min_x, max_x = min(min_x, node["x"]), max(max_x, node["x"])
            min_y, max_y = min(min_y, node["y"]), max(max_y, node["y"])
        print(min_x, max_x, min_y, max_y)
        self.min_x, self.max_x, self.min_y, self.max_y = (min_x, max_x, min_y, max_y)

    def get_geojson_file_to_single_polygon(self):
        fname = os.path.join(config.network_folder,config.rn_prefix_geojson_files+self.city_name.lower()+config.rn_postfix_geojson_files)
        with open(fname) as f:
            geojs = geojson.load(f)

        # x, y = shape(geojs["features"][0]["geometry"]).convex_hull.boundary.xy
        list_of_geoms = []
        for i in range(len(geojs["features"])):
            list_of_geoms.append(shape(geojs["features"][i]["geometry"]))
        x, y = unary_union(list_of_geoms).convex_hull.boundary.xy

        x = x.tolist(); y = y.tolist()
        x = list(np.array(x).astype(float))
        y = list(np.array(y).astype(float))

        plt.plot(x,y)
        plt.savefig(os.path.join(config.network_folder, self.city_name,"polygon_convex.png"), dpi=300)

        convex_hull_poly = Polygon([[p[0], p[1]] for p in zip(x,y)])

        return convex_hull_poly


    def filter_OSM(self):
        try:
            self.G_osm = ox.truncate.truncate_graph_polygon(self.G_osm, self.get_geojson_file_to_single_polygon())
        except ValueError:
            with open("ERROR_getting_smaller_network.txt","a") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow([self.city_name, "Reverting_to_bigger_network"])

        return self.G_osm

if __name__ == "__main__":

    list_of_graph_features = Tile(None).get_list_of_features()

    if not os.path.exists("network"):
        os.mkdir("network")

    with open(os.path.join("network", "all_cities_graph_features.csv"), "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["city"] + list_of_graph_features)

        for city in config.rn_master_list_of_cities:
            if not os.path.exists(os.path.join("network", city)):
                os.mkdir(os.path.join("network", city))

            starttime = time.time()

            rn = RoadNetwork(city)
            rn.plot_basemap()

            if config.rn_compute_graph_features:
                features = Tile(rn.G_osm).get_stats_for_tile()
                list_of_values = [city]
                for key in list_of_graph_features:
                    list_of_values.append(features[key])
                csvwriter.writerow(list_of_values)

            sprint (city)
            sprint(time.time() - starttime)
