import os

import config
from urbanscales.preprocessing.prep_network import Scale
import matplotlib.pyplot as plt
import osmnx as ox


class VizSpatial:
    def __init__(self, cityname, scale, tod):
        """

        Args:
            cityname:
            scale:
            tod: used only for y plots
        """
        self.cityname = cityname
        self.scale = scale
        self.tod = tod
        self.scl = Scale.get_object(cityname=self.cityname, scale=scale)
        self.features_names_valid = self.filter_features(
            self.scl.dict_bbox_to_subgraph[list(self.scl.dict_bbox_to_subgraph.keys())[0]].get_feature_names()
        )
        self.plot_x_features()

    def filter_features(self, feature_names):
        a = []
        for featur_ in feature_names:
            if featur_ not in config.td_drop_feature_lists:
                a.append(featur_)
        return a

    def plot_x_features(self):

        for valid_feature in self.features_names_valid:
            long_list = []
            lat_list = []
            colorlist = []
            for key in self.scl.dict_bbox_to_subgraph.keys():
                #  sample: (1.3064703999999998, 1.3026463, 103.82636672474786, 103.8225672),
                bbox = key  # self.scl.dict_bbox_to_subgraph[key]

                features = self.scl.dict_bbox_to_subgraph[key].get_vector_of_features()
                names = self.scl.dict_bbox_to_subgraph[key].get_feature_names()
                assert len(features) == len(names)

                feature_filtered = None
                for f, n in zip(features, names):
                    if n == valid_feature:
                        feature_filtered = f

                lat_N, lat_S, lon_E, lon_W = bbox
                centre_lat = (lat_N + lat_S) / 2
                centre_lon = (lon_E + lon_W) / 2
                lat_list.append(centre_lat)
                long_list.append(centre_lon)
                colorlist.append(feature_filtered)

                # plt.figure(figsize=(10, 10))
            # plt.clf()
            fig, ax = ox.plot.plot_graph(
                self.scl.RoadNetwork.G_osm,
                ax=None,
                figsize=(12, 12),
                bgcolor="white",
                node_color="black",
                node_size=0.1,
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
            plt.scatter(
                long_list, lat_list, marker="s", c=colorlist, cmap="viridis", alpha=0.7, s=max(20 * 30 // self.scale, 8)
            )
            plt.colorbar().set_label(valid_feature)
            plt.xlabel("longitude of bbox centre", fontsize=12)
            plt.ylabel("latitude of bbox centre", fontsize=12)
            plt.title((self.cityname + "-" + str(self.scale) + "-" + valid_feature))
            if not os.path.exists(os.path.join(config.results_folder, "spatial-dist")):
                os.mkdir(os.path.join(config.results_folder, "spatial-dist"))
            plt.savefig(
                os.path.join(
                    config.results_folder,
                    "spatial-dist",
                    (self.cityname + "-" + str(self.scale) + "-" + valid_feature) + ".png",
                ),
                dpi=300,
            )

    @staticmethod
    def generate_spatial_plots_for_all_cities():
        for city in config.scl_master_list_of_cities:
            for seed in config.scl_list_of_seeds[::-1]:
                for depth in config.scl_list_of_depths:
                    for tod in config.td_tod_list:
                        VizSpatial(city, seed ** depth, tod)


if __name__ == "__main__":
    VizSpatial.generate_spatial_plots_for_all_cities()
