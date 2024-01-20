import copy

from matplotlib import pyplot as plt
import pandas as pd
from skimage import filters
import numpy as np
from smartprint import smartprint as sprint

def read_and_aggregate_data(city, feature, mean_max, scales):
    aggregated_data = {}
    for scale in scales:
        file_name = f"MEAN_MAX_{mean_max}_FI_Mean_{city}_Scale_{scale}_TOD_0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23.csv"
        column_names = ['feature', 'scale', 'city', 'tod', 'FI_mean']  # List your column names here
        try:
            df = pd.read_csv(file_name, names=column_names)
            df = df[1:]
        except:
            print ("Missing filename: ", file_name, "\n ignored")
            continue
        aggregated_data[scale] = df[df['feature'] == feature]['FI_mean'].mean()
    return aggregated_data

import matplotlib.colors as mcolors
common_features_colors = {
    'betweenness': '#1f77b4',  # blue
    'circuity_avg': '#ff7f0e',  # orange
    'global_betweenness': '#2ca02c',  # green
    'k_avg': '#d62728',  # red
    'lane_density': '#9467bd',  # purple
    'metered_count': '#8c564b',  # brown
    'non_metered_count': '#e377c2',  # pink
    'street_length_total': '#7f7f7f',  # gray
    'm': '#bcbd22',  # lime
    'total_crossings': '#17becf',  # cyan
    "n": "black"
}
city_colors = {
    "Auckland": "#1f77b4",  # Muted blue
    "Bogota": "#ff7f0e",    # Safety orange
    "Capetown": "#2ca02c",  # Cooked asparagus green
    "Istanbul": "#d62728",  # Brick red
    "London": "#9467bd",    # Muted purple
    "MexicoCity": "#8c564b",# Chestnut brown
    "Mumbai": "#e377c2",    # Raspberry yogurt pink
    "NewYorkCity": "#7f7f7f",# Middle gray
    "Singapore": "#bcbd22", # Curry yellow-green
    "Zurich": "#17becf"     # Blue-teal
}



########################################################################
#########################    Feature-specific    ##########################
########################################################################
"""
feature_list = list(common_features_colors.keys())
city_list = list(city_colors.keys())

filter_important_features_recurrent = {}
filter_important_features_non_recurrent = {}

for scale_list in [[25], [50], [100]]:
    filter_important_features_recurrent[scale_list[0]] = {}
    filter_important_features_non_recurrent[scale_list[0]] = {}

    for feature in feature_list:
        # Aggregate data across cities and features
        data_recurrent = {key: [] for key in city_list}
        data_nonrecurrent = {key: [] for key in city_list}

        for city in (city_list):
            data_recurrent[city] = read_and_aggregate_data(city, feature, "mean", scale_list)[scale_list[0]]
            data_nonrecurrent[city] = read_and_aggregate_data(city, feature, "max", scale_list)[scale_list[0]]

        otsu_thres = filters.threshold_otsu(np.array(list(data_recurrent.values())) + 1e-8)
        sprint (scale_list, feature, otsu_thres)
        plt.scatter(city_list, list(data_recurrent.values()), color="tab:blue", marker='o', alpha=0.2)
        plt.plot(city_list, np.array([otsu_thres] * len(city_list)) , color="black", linewidth=1.5)

        filter_important_features_recurrent[scale_list[0]][feature] = []
        filter_important_features_non_recurrent[scale_list[0]][feature] = []

        cities_with_higher_than_otsu = []
        corresponding_values = []
        for city in city_list:
            if data_recurrent[city] > otsu_thres:
                cities_with_higher_than_otsu.append(city)
                corresponding_values.append(data_recurrent[city])
                filter_important_features_recurrent[scale_list[0]][feature].append(city)

        plt.scatter(cities_with_higher_than_otsu,corresponding_values, color="tab:blue")
        plt.title("Recurrent_" + feature + " Scale: " + "_".join([str(x) for x in scale_list]))
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("Recurrent_" + feature + " Scale: " + "_".join([str(x) for x in scale_list]) + ".png", dpi=300)
        plt.show(block=False)



        otsu_thres = filters.threshold_otsu(np.array(list(data_nonrecurrent.values())) + 1e-8)
        sprint (scale_list, feature, otsu_thres)
        plt.scatter(city_list, list(data_nonrecurrent.values()), color="tab:blue", marker='o', alpha=0.2)
        plt.plot(city_list, np.array([otsu_thres] * len(city_list)) , color="black", linewidth=1.5)

        cities_with_higher_than_otsu = []
        corresponding_values = []
        for city in city_list:
            if data_nonrecurrent[city] > otsu_thres:
                cities_with_higher_than_otsu.append(city)
                corresponding_values.append(data_nonrecurrent[city])
                filter_important_features_non_recurrent[scale_list[0]][feature].append(city)

        plt.scatter(cities_with_higher_than_otsu,corresponding_values, color="tab:blue")
        plt.title("Non_Recurrent_" + feature + " Scale: " + "_".join([str(x) for x in scale_list]))
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("Non_Recurrent_" + feature + " Scale: " + "_".join([str(x) for x in scale_list]) + ".png", dpi=300)
        plt.show(block=False)

filter_important_features_recurrent_feature_wise = copy.copy(filter_important_features_recurrent)
# sprint (filter_important_features_recurrent)
filter_important_features_non_recurrent_feature_wise = copy.copy(filter_important_features_non_recurrent)
# sprint (filter_important_features_non_recurrent)

"""


########################################################################
#########################    City-specific    ##########################
########################################################################
feature_list = list(common_features_colors.keys())
city_list = list(city_colors.keys())

filter_important_features_recurrent = {}
filter_important_features_non_recurrent = {}

for scale_list in [[25], [50], [100]]:
    filter_important_features_recurrent[scale_list[0]] = {}
    filter_important_features_non_recurrent[scale_list[0]] = {}

    for city in (city_list):
        filter_important_features_recurrent[scale_list[0]][city] = []
        filter_important_features_non_recurrent[scale_list[0]][city] = []

        # Aggregate data across cities and features
        data_recurrent = {key: [] for key in feature_list}
        data_nonrecurrent = {key: [] for key in feature_list}

        for feature in feature_list:
            data_recurrent[feature] = read_and_aggregate_data(city, feature, "mean", scale_list)[scale_list[0]]
            data_nonrecurrent[feature] = read_and_aggregate_data(city, feature, "max", scale_list)[scale_list[0]]

        otsu_thres = filters.threshold_otsu(np.array(list(data_recurrent.values())) + 1e-8)
        sprint (scale_list, feature, otsu_thres)
        plt.scatter(feature_list, list(data_recurrent.values()), color="tab:blue", marker='o', alpha=0.2)
        plt.plot(feature_list, np.array([otsu_thres] * len(feature_list)) , color="black", linewidth=1.5)



        features_with_higher_than_otsu = []
        corresponding_values = []
        for feature in feature_list:
            if data_recurrent[feature] > otsu_thres:
                features_with_higher_than_otsu.append(feature)
                corresponding_values.append(data_recurrent[feature])
                filter_important_features_recurrent[scale_list[0]][city].append(feature)

        plt.scatter(features_with_higher_than_otsu,corresponding_values, color="tab:blue")
        plt.title("OTSU_Recurrent_" + city + " Scale: " + "_".join([str(x) for x in scale_list]))
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("OTSU_Recurrent_" + city + " Scale: " + "_".join([str(x) for x in scale_list]) + ".png", dpi=300)
        plt.show(block=False)


        otsu_thres = filters.threshold_otsu(np.array(list(data_nonrecurrent.values())) + 1e-8)
        sprint (scale_list, feature, otsu_thres)
        plt.scatter(feature_list, list(data_nonrecurrent.values()), color="tab:blue", marker='o', alpha=0.2)
        plt.plot(feature_list, np.array([otsu_thres] * len(feature_list)) , color="black", linewidth=1.5)


        features_with_higher_than_otsu = []
        corresponding_values = []
        for feature in feature_list:
            if data_nonrecurrent[feature] > otsu_thres:
                features_with_higher_than_otsu.append(feature)
                corresponding_values.append(data_nonrecurrent[feature])
                filter_important_features_non_recurrent[scale_list[0]][city].append(feature)

        plt.scatter(features_with_higher_than_otsu,corresponding_values, color="tab:blue")
        plt.title("OTSU_Non Recurrent_" + city + " Scale: " + "_".join([str(x) for x in scale_list]))
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("OTSU_Non Recurrent_" + city + " Scale: " + "_".join([str(x) for x in scale_list]) + ".png", dpi=300)
        plt.show(block=False)


filter_important_features_recurrent_city_wise = copy.copy(filter_important_features_recurrent)
# sprint (filter_important_features_recurrent)
filter_important_features_non_recurrent_city_wise = copy.copy(filter_important_features_non_recurrent)
# sprint (filter_important_features_non_recurrent)

sprint (filter_important_features_recurrent_city_wise)
sprint (filter_important_features_non_recurrent_city_wise)

