import os

import config
from urbanscales.preprocessing.prep_network import Scale
import matplotlib as matpl
import matplotlib.patches as ptch
import matplotlib.pyplot as plt
import osmnx as ox
from tqdm import tqdm


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
        cmap = matpl.cm.get_cmap("viridis")
        dict_sugb_to_features = {}  # so that we compute all features in the first iteration itself

        for valid_feature in self.features_names_valid:
            long_list = []
            lat_list = []

            widthlist = []
            heighlist = []
            feature_max = -1000000
            feature_min = 1000000000000
            feature_val_list = []
            list_ = list(self.scl.dict_bbox_to_subgraph.keys())
            for i in tqdm(range(len(list_)), desc="Computing G features for bboxes: "):
                key = list_[i]
                #  sample: (1.3064703999999998, 1.3026463, 103.82636672474786, 103.8225672),
                bbox = key  # self.scl.dict_bbox_to_subgraph[key]

                if key in dict_sugb_to_features:
                    features = dict_sugb_to_features[key]
                else:
                    features = self.scl.dict_bbox_to_subgraph[key].get_vector_of_features()
                    dict_sugb_to_features[key] = features

                names = self.scl.dict_bbox_to_subgraph[key].get_feature_names()
                assert len(features) == len(names)

                feature_filtered = None
                for f, n in zip(features, names):
                    if n == valid_feature:
                        feature_filtered = f
                feature_val_list.append(feature_filtered)

                assert feature_filtered is not None
                assert not isinstance(feature_filtered, str)  # checking for empty case

                feature_max = max(feature_max, feature_filtered)
                feature_min = min(feature_min, feature_filtered)

                lat_N, lat_S, lon_E, lon_W, _unused_len = bbox
                centre_lat = (lat_N + lat_S) / 2
                centre_lon = (lon_E + lon_W) / 2

                widthlist.append(lon_E - lon_W)
                heighlist.append(lat_N - lat_S)

                assert widthlist[-1] > 0
                assert heighlist[-1] > 0

                lat_list.append(centre_lat)
                long_list.append(centre_lon)

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
            for xy_index in tqdm(
                range(len(long_list)), desc="Plotting " + valid_feature + self.cityname + str(self.scale)
            ):
                xy = long_list[xy_index], lat_list[xy_index]

                # Params of plt.Rectange: (xy, width, height, ..)
                # The rectangle extends from xy[0] to xy[0] + width in x-direction and from xy[1]
                # to xy[1] + height in y-direction.
                rect = plt.Rectangle(
                    xy,
                    widthlist[xy_index],
                    heighlist[xy_index],
                    color=cmap((feature_val_list[xy_index] - feature_min) / feature_max),
                    alpha=0.7,
                )

                ax.add_patch(rect)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(feature_min, feature_max))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label=valid_feature)

            plt.xlabel("longitude of bbox centre", fontsize=12)
            plt.ylabel("latitude of bbox centre", fontsize=12)
            plt.title((self.cityname + "-" + str(self.scale) + "-" + valid_feature))
            print("Saving image!")
            if not os.path.exists(os.path.join(config.BASE_FOLDER, config.results_folder, "spatial-dist")):
                os.mkdir(os.path.join(config.BASE_FOLDER, config.results_folder, "spatial-dist"))
            plt.savefig(
                os.path.join(
                    config.BASE_FOLDER,
                    config.results_folder,
                    "spatial-dist",
                    (self.cityname + "-" + str(self.scale) + "-" + valid_feature) + ".png",
                ),
                dpi=300,
            )
            print("Image saved successfully!")

    @staticmethod
    def generate_spatial_plots_for_all_cities():
        for city in config.scl_master_list_of_cities:
            for seed in config.scl_list_of_seeds:
                for depth in config.scl_list_of_depths:
                    for tod in config.td_tod_list:
                        VizSpatial(city, seed ** depth, tod)


if __name__ == "__main__":
    VizSpatial.generate_spatial_plots_for_all_cities()
