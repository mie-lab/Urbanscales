import os
import sys


sys.path.append("../../../")
import pickle
import copy
from sklearn.model_selection import KFold
from smartprint import smartprint as sprint
from scipy import stats

current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.join(current_dir, '..')
os.chdir(current_dir)
sprint (os.getcwd())
import geopandas as gpd
import contextily as ctx
from shapely.geometry import box
import geopy.distance as gpy_dist

# for the custom unpickler
from urbanscales.preprocessing.train_data import TrainDataVectors

mean_max = "max"
ZIPPEDfoldername = "train_data_three_scales_" + mean_max
PLOTS_foldername = mean_max + "_case_corr_analysis/"

if "mean" in ZIPPEDfoldername:
    assert "mean" in PLOTS_foldername
if "max" in ZIPPEDfoldername:
    assert "max" in PLOTS_foldername

os.system("tar -xf ../" + ZIPPEDfoldername + ".tar.gz")

class CustomUnpicklerTrainDataVectors(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module, name)

class CorrelationAnalysis:
    def __init__(self, X_data=None, Y_data=None, fname=None):
        if fname:
            temp = copy.deepcopy(CustomUnpicklerTrainDataVectors(open(fname, "rb")).load())
            self.__dict__.update(temp.__dict__)
            self.nparrayX = np.array(self.X)
            self.nparrayY = np.array(self.Y)
            self.columns = self.X.columns.tolist()
        elif X_data is not None and Y_data is not None:
            if isinstance(X_data, pd.DataFrame):
                self.nparrayX = X_data.to_numpy()
                self.columns = X_data.columns.tolist()
            else:  # assume it's a numpy array
                self.nparrayX = X_data
                self.columns = [f"feature_{i}" for i in range(X_data.shape[1])]  # generic names
            self.nparrayY = Y_data
        else:
            raise ValueError("Either provide a filename or both X_data and Y_data arrays for initialization.")
        print(self.nparrayX.shape, self.nparrayY.shape)


    def random_forest_feature_importance(self, common_features):
        n_splits = 7

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        total_feature_importances = np.zeros(len(common_features))

        for train_index, test_index in kfold.split(self.nparrayX, self.nparrayY):
            X_train, X_test = self.nparrayX[train_index], self.nparrayX[test_index]
            Y_train, Y_test = self.nparrayY[train_index], self.nparrayY[test_index]
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, Y_train)
            total_feature_importances += rf.feature_importances_
        total_feature_importances /= n_splits

        return total_feature_importances

    def plot_feature_importances(self, importances, scale, common_features, list_of_cities):
        plt.title(f"Feature Importances RF " + list_of_cities)
        plt.plot(range(len(common_features)), importances, marker='o', linestyle='-', label=f"Scale: {scale}")
        plt.xticks(range(len(common_features)), common_features, rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.legend()




#
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import contextily as ctx
# from shapely.geometry import box
# import geopy.distance as gpy_dist
# import numpy as np

def plot_bboxes_for_debugging(temp_obj, identifier):
    bbox_list = temp_obj.bbox_X  # Assuming this is the correct attribute

    # Create a list of box geometries
    geometries = []
    area_list = []
    x_distance_list = []
    y_distance_list = []
    for bbox in bbox_list:
        # Assuming bbox is structured as (N, S, E, W, id)
        N, S, E, W, _ = list(bbox.keys())[0]
        geometries.append(box(W, S, E, N))

        x_distance = gpy_dist.geodesic((N, E), (N, W)).km
        y_distance = gpy_dist.geodesic((N, E), (S, E)).km
        area = x_distance * y_distance

        area_list.append(area)
        x_distance_list.append(x_distance)
        y_distance_list.append(y_distance)


    # Create GeoDataFrame from the geometries
    gdf = gpd.GeoDataFrame({'geometry': geometries}, crs='EPSG:4326')

    # Convert to Web Mercator for basemap
    gdf = gdf.to_crs(epsg=3857)
    fig, ax = plt.subplots(figsize=(15, 10))
    gdf.boundary.plot(ax=ax, color='blue', linewidth=2)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    plt.savefig("Bboxes_final_After_feature_Extraction_" + identifier + ".png", dpi=300)
    plt.show()



    mean_area = np.mean(area)
    median_area = np.median(area)
    mean_x = np.mean(x_distance)
    mean_y = np.mean(y_distance)
    median_x = np.median(x_distance)
    median_y = np.median(y_distance)
    with open("Mean_area_of_tiles.txt", "a") as f:
        f.write(f"{identifier} Mean Geodesic Area: {mean_area} km^2; {mean_x} km; {mean_y} km\n")
        f.write(f"{identifier} Median Geodesic Area: {median_area} km^2; {median_x} km; {median_y} km\n")

# Example usage
# plot_bboxes_for_debugging(temp_obj, "_identifier_")


# Example usage
# plot_bboxes_for_debugging(temp_obj, "_identifier_")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from scipy.cluster.hierarchy import dendrogram, linkage

if __name__ == "__main__":
    os.system("rm Mean_area_of_tiles.txt")
    list_of_cities = "Singapore|Zurich|Mumbai|Auckland|Istanbul|MexicoCity|Bogota|NewYorkCity|Capetown|London".split("|")
    list_of_cities_list_of_list = [
                                    # list_of_cities[:2],
                                    # list_of_cities[2:]
                                    [list_of_cities[0]],
                                    [list_of_cities[1]],
                                    [list_of_cities[2]],
                                    [list_of_cities[3]],
                                    [list_of_cities[4]],
                                    [list_of_cities[5]],
                                    [list_of_cities[6]],
                                    [list_of_cities[7]],
                                    [list_of_cities[8]],
                                    [list_of_cities[9]],

                                ]


    common_features = [
        'betweenness',
        'circuity_avg',
        'global_betweenness',
        'k_avg',
        'lane_density',
        # 'm',
        'metered_count',
        # 'n',
        'non_metered_count',
        'street_length_total',
        'streets_per_node_count_5',
        'total_crossings'
    ]

    scale_list = [25, 50, 100]

    results = {}

    for scale in scale_list:
        all_cities_data = []

        for city in list_of_cities:
            combined_X_for_city = []

            for tod in range(24):
                fname = f"{ZIPPEDfoldername}/{city}/_scale_{scale}_train_data_{tod}.pkl"
                try:
                    temp_obj = CustomUnpicklerTrainDataVectors(open(fname, "rb")).load()

                    if isinstance(temp_obj.X, pd.DataFrame):
                        filtered_X = temp_obj.X[list(common_features)]
                        combined_X_for_city.append(filtered_X)

                        # We need to do this only for one tod; since the graph does not change across all tod's
                        if tod == 0:
                            plot_bboxes_for_debugging(temp_obj, identifier=f"_{city}_scale_{scale}_train_data_{tod}_")

                except FileNotFoundError:
                    print("Error in fname:", fname)

            # Aggregate data for the city over different times of day
            city_avg = pd.concat(combined_X_for_city, axis=0).mean()
            all_cities_data.append(city_avg)

        # Create a dataframe where each row represents a city's average feature values over different times of day
        X = pd.DataFrame(all_cities_data, index=list_of_cities)

        # Perform hierarchical clustering
        Z = linkage(X,
                    'ward')  # 'ward' is one of the methods. You can choose another depending on your data characteristics.

        # Plot the dendrogram
        plt.figure(figsize=(10, 7))
        plt.title(f'Hierarchical Clustering Dendrogram for scale {scale}')
        plt.xlabel('City')
        plt.ylabel('Distance (Ward)')
        dendrogram(
            Z,
            labels=X.index.tolist(),
            leaf_rotation=90.,
            leaf_font_size=8.,
        )
        plt.tight_layout()
        plt.savefig("hierarchical_clustering/" + f'Hierarchical Clustering Dendrogram for scale {scale}'
                    + ".png", dpi=300)
        plt.show()


    # 1. Calculate the mean feature values for each city across scales
    aggregate_data_across_scales = {city: [] for city in list_of_cities}

    combined_Y_for_city_all_scales = {}
    for city in list_of_cities:
        for scale in scale_list:
            combined_Y_for_city_all_scales[city, scale] = []



    for scale in scale_list:
        for city in list_of_cities:
            combined_X_for_city = []

            for tod in range(24):
                fname = f"{ZIPPEDfoldername}/{city}/_scale_{scale}_train_data_{tod}.pkl"
                try:
                    temp_obj = CustomUnpicklerTrainDataVectors(open(fname, "rb")).load()

                    if isinstance(temp_obj.X, pd.DataFrame):
                        filtered_X = temp_obj.X[list(common_features)]
                        combined_X_for_city.append(filtered_X)
                        combined_Y_for_city_all_scales[city, scale].extend(temp_obj.Y.flatten().tolist())
                except FileNotFoundError:
                    print("Error in fname:", fname)

            # Aggregate data for the city over different times of day for the current scale
            city_avg_for_scale = pd.concat(combined_X_for_city, axis=0).mean()
            aggregate_data_across_scales[city].append(city_avg_for_scale)

    for city in list_of_cities:
        plt.clf()
        BINS = 30
        s = 0
        ditributions = []
        for scale in scale_list:
            hist = np.histogram(combined_Y_for_city_all_scales[city, scale], bins=BINS)
            plt.plot(np.array(hist[0]), label="scale@" + str(scale))
            s += np.mean(combined_Y_for_city_all_scales[city, scale])
            ditributions.append(hist[0])

        crr = []
        for dist_1 in ditributions:
            for dist_2 in ditributions:
                if stats.pearsonr(dist_1.flatten(), dist_2.flatten()) != 1:
                    crr.append(stats.pearsonr(dist_1, dist_2)[0])
        # sprint (crr)
        print (city, " corr between scales: ", np.mean(crr))

        tick_locations = range(0, BINS, 3)  # Actual locations
        tick_labels = [str(x) for x in (np.array(tick_locations)/3).tolist()]  # Labels you want to display
        plt.xticks(tick_locations, tick_labels)
        sprint (city, s/3)
        plt.title(city)
        plt.legend()
        plt.savefig(city + "-" + mean_max + "hist_Y_all_scales.png")


    ##
    # Compute the mean across scales for each city
    all_cities_data_aggregated = []
    for city, data_list in aggregate_data_across_scales.items():
        mean_data_across_scales = pd.concat(data_list, axis=1).mean(axis=1)
        all_cities_data_aggregated.append(mean_data_across_scales)

    # 2. Perform hierarchical clustering on the averaged data
    X_aggregated = pd.DataFrame(all_cities_data_aggregated, index=list_of_cities)

    Z_aggregated = linkage(X_aggregated, 'ward')

    # Plot the dendrogram for aggregated data
    plt.figure(figsize=(10, 7))
    plt.title('Hierarchical Clustering Dendrogram (Aggregated Across Scales)')
    plt.xlabel('City')
    plt.ylabel('Distance (Ward)')
    dendrogram(
        Z_aggregated,
        labels=X_aggregated.index.tolist(),
        leaf_rotation=90.,
        leaf_font_size=8.,
    )
    plt.tight_layout()
    plt.savefig("hierarchical_clustering/" + 'Hierarchical Clustering Dendrogram (Aggregated Across Scales)'
                + ".png", dpi=300)
    plt.show()