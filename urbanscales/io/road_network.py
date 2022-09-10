import csv
import os.path
import pickle
import osmnx as ox
import config
import matplotlib.pyplot as plt
from smartprint import smartprint as sprint
from urbanscales.preprocessing import tile


class RoadNetwork:
    def __init__(self, cityname):
        """
        Example: SG = RoadNetwork("Singapore")
        """
        self.city_name = cityname
        self.G_osm = None
        self.osm_pickle = "_OSM.pkl"

        fname = os.path.join("network", self.city_name, "_road_network_object.pkl")
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                temp = pickle.load(f)
                self.G_osm = temp.G_osm
                self.graph_features = temp.graph_features
                self.min_x, self.max_x, self.min_y, self.max_y = temp.min_x, temp.max_x, temp.min_y, temp.max_y
        else:

            if not os.path.isdir(os.path.join("network", self.city_name)):
                if not os.path.isdir("network"):
                    os.mkdir("network")
                os.mkdir(os.path.join("network", self.city_name))
            self.G_osm = self.get_osm_from_place()
            self.graph_features = self.set_graph_features()
            self.set_boundaries_x_y()

            with open(fname, "wb") as f:
                pickle.dump(self, f, protocol=config.pickle_protocol)

    def get_osm_from_place(self):
        if self.G_osm == None:
            fname = os.path.join("network", self.city_name, self.osm_pickle)
            if os.path.isfile(fname):
                with open(fname, "rb") as f1:
                    self.G_osm = pickle.load(f1)
            else:
                self.G_osm = ox.graph_from_place(
                    {"city": self.city_name, "country": "South Africa"}, network_type="drive"
                )
                with open(fname, "wb") as f:
                    pickle.dump(self.G_osm, f, protocol=config.pickle_protocol)
        return self.G_osm
        # G = ox.speed.add_edge_speeds(G)
        # G = ox.speed.add_edge_travel_times(G)

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
        self.graph_features = tile.Tile(self.G_osm).set_stats_for_tile()

    def plot_basemap(self):
        if not config.rn_plotting_enabled:
            return

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
            save=False,
            bbox=None,
        )
        plt.savefig(os.path.join("network", self.city_name, "_base_osm.png"), dpi=300)

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


if __name__ == "__main__":

    import time

    list_of_graph_features = tile.Tile(None).get_list_of_features()
    with open(os.path.join("network", "all_cities_graph_features.csv"), "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["city"] + list_of_graph_features)

        for city in config.rn_master_list_of_cities:
            rn = RoadNetwork(city)

            starttime = time.time()
            try:
                rn.get_osm_from_place()
            except ValueError:
                sprint(city, " Error;  ..; Trying by address")
                rn.get_osm_from_address()
                continue
            sprint(time.time() - starttime)

            starttime = time.time()
            sprint(time.time() - starttime)
            rn.get_osm_from_place()
            rn.plot_basemap()

            if config.rn_compute_graph_features:
                features = tile.Tile(rn.G_osm).get_stats_for_tile()
                list_of_values = [city]
                for key in list_of_graph_features:
                    list_of_values.append(features[key])
                csvwriter.writerow(list_of_values)
