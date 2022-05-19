# First, a uniform segmentation of the urban space into spatial grids is done and eight graph-based features are extracted for each grid.
import os

import sys
import csv
import pandas
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

server_path = "/home/niskumar/WCS/python_scripts/network_to_elementary/"
local_path = "/Users/nishant/Documents/GitHub/WCS/python_scripts/network_to_elementary/"
sys.path.insert(0, server_path)

import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

from python_scripts.network_to_elementary.elf_to_clusters import osm_tiles_states_to_vectors
from python_scripts.network_to_elementary.osm_to_tiles import fetch_road_network_from_osm_database
from python_scripts.network_to_elementary.process_incidents import create_bbox_to_CCT
from python_scripts.network_to_elementary.tiles_to_elementary import step_1_osm_tiles_to_features
import pickle
from shapely.geometry import Polygon
import numpy as np
import sys
import osmnx as ox

# import os
# os.system("pip install -r requirements.txt")

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# step_1_osm_tiles_to_features( read_G_from_pickle=True, read_osm_tiles_stats_from_pickle=False, n_threads=7, N=50, plotting_enabled=True)


def auxiliary_func_G_for_curved_paths():
    geo = {
        "type": "Polygon",
        "coordinates": [
            [
                [103.96078535200013, 1.39109935100015],
                [103.98568769600007, 1.38544342700007],
                [103.99952233200003, 1.38031647300005],
                [104.00342858200003, 1.374172268000066],
                [103.99187259200011, 1.354925848000036],
                [103.97486412900014, 1.334458726000065],
                [103.95435631600009, 1.318101304000052],
                [103.93189537900008, 1.311468817000076],
                [103.90723717500009, 1.308742580000114],
                [103.88770592500003, 1.301255601000136],
                [103.85271243600005, 1.277289130000085],
                [103.84693444100009, 1.271918036000045],
                [103.84408613400012, 1.268500067000034],
                [103.83887780000003, 1.266262111000046],
                [103.82601972700007, 1.264308986000089],
                [103.80160566500007, 1.264797268000081],
                [103.78956139400003, 1.26788971600007],
                [103.78443444100003, 1.273871161000088],
                [103.77588951900009, 1.287583726000108],
                [103.75513756600003, 1.297105210000012],
                [103.73015384200011, 1.302923895000063],
                [103.70875084700003, 1.305243231000119],
                [103.66529381600009, 1.304103908000087],
                [103.6476343110001, 1.308417059000092],
                [103.64039147200003, 1.322251695000091],
                [103.64470462300005, 1.338039455000043],
                [103.67457116000003, 1.38031647300005],
                [103.67888431100005, 1.399237372000073],
                [103.68384850400008, 1.40989817900001],
                [103.69507897200009, 1.421332098000065],
                [103.70834394600013, 1.429388739000089],
                [103.7179468110001, 1.430975653000118],
                [103.73975670700008, 1.428127346000082],
                [103.76221764400009, 1.430975653000118],
                [103.79004967500003, 1.444281317000048],
                [103.80494225400008, 1.448635158000045],
                [103.83155358200003, 1.447088934000092],
                [103.85718834700009, 1.438706773000135],
                [103.93246504000007, 1.401109117000132],
                [103.96078535200013, 1.39109935100015],
            ]
        ],
    }
    poly = Polygon([tuple(l) for l in geo["coordinates"][0]])

    # G_proj = osm.project_graph(G)
    # fig, ax = osm.plot_graph(G_proj)
    # , "trunk","trunk_link", "motorway_link","primary","secondary"]
    # custom_filter=["motorway", "motorway_link","motorway_junction","highway"],

    G = fetch_road_network_from_osm_database(
        polygon=poly, network_type="drive", custom_filter='["highway"~"motorway|motorway_link|primary"]'
    )

    # orig = (1.34294, 103.74631)
    # dest = (1.34499, 103.74022)

    G = ox.speed.add_edge_speeds(G)
    G = ox.speed.add_edge_travel_times(G)
    return G


# step 2
def step_2(N, folder_path):

    dict_bbox_to_CCT_and_start_time = create_bbox_to_CCT(
        csv_file_name="combined_incidents_45_days.csv",
        read_from_pickle=True,
        N=N,
        folder_path=folder_path,
        graph_with_edge_travel_time=auxiliary_func_G_for_curved_paths(),
        use_route_path=True,
        read_curved_paths_from_pickle=False,
    )

    with open(folder_path + "osm_tiles_stats_dict" + str(N) + ".pickle", "rb") as f:
        osm_tiles_stats_dict = pickle.load(f)

    keys_bbox_list, vals_vector_array = osm_tiles_states_to_vectors(osm_tiles_stats_dict)

    X_t = {}
    Y_t = {}
    for hour in range(24):
        Y = []
        X = []
        count_present = 0
        count_absent = 0
        for i, key in enumerate(keys_bbox_list):
            if (key, hour) in dict_bbox_to_CCT_and_start_time:

                Y.append(dict_bbox_to_CCT_and_start_time[key, hour])
                X.append(vals_vector_array[i])
                count_present += 1
                print(dict_bbox_to_CCT_and_start_time[key, hour])

            else:
                count_absent += 1
        X_t[hour] = X
        Y_t[hour] = Y
        print("2nd step: Count present in CCT file: @hour", hour, count_present)
        print("2nd step: Count absent in CCT file: @hour", hour, count_absent)
    return X_t, Y_t


def step_2a_extend_the_vectors(X, Y, expand_type="duplicate"):
    """
    We need to create copies of incidents when several incidents happen in the
    same grid, Our first stab at this is to create multiple copies of the vector

    :param X:
    :param Y:
    :param expand_type: "duplicate" or "max"
    :return:
    """
    y = []
    x = []
    for i in range(len(Y)):
        if expand_type == "duplicate":
            for j in range(len(Y[i])):  # Y is a list of lists
                x.append(X[i])
                y.append(Y[i][j])
        elif expand_type == "max":
            x.append(X[i])
            y.append(max(Y[i]))
        elif expand_type == "mean":
            x.append(X[i])
            y.append(np.mean(Y[i]))
    return np.array(x), np.array(y)


def step_2b_calculate_GOF(X, Y, model="regression"):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, random_state=int(time.time()))

        # ('scaler', StandardScaler()),
        # ("pca", PCA(n_components=5))
        pipe = Pipeline([("scaler", StandardScaler()), ("RF", RandomForestRegressor())])
        # The pipeline can be used as any other estimator
        # and avoids leaking the test set into the train set
        print(pipe.fit(X_train, y_train))
        print(pipe.score(X_test, y_test), " GoF measure")

        # scores = cross_val_score(pipe, X, Y, cv=7)
        # print("CV GoF measure: ", scores)
        # print("Mean CV (GoF):", np.mean(scores))
        # return np.mean(scores)
        # return pipe.score(X_test, y_test)
        y_pred = pipe.predict(X_test)
        return mean_squared_error(y_test, y_pred)
    except:
        return -99999


def step_3(min_, max_, step_, multiple_runs=1, use_saved_vectors=False):
    """

    :param min_:
    :param max_:
    :param step_:
    :return:
    """
    mean_cv_score_dict = {}
    for N in range(min_, max_, step_):

        if use_saved_vectors:
            with open("temp_files/" + "X_t_Y_t_" + str(N) + ".pickle", "rb") as f:
                X_t, Y_t = pickle.load(f)
        else:
            X_t, Y_t = step_2(N, folder_path=server_path)
            with open("temp_files/" + "X_t_Y_t_" + str(N) + ".pickle", "wb") as f:
                pickle.dump((X_t, Y_t), f, protocol=pickle.HIGHEST_PROTOCOL)

        for hour in range(24):
            X, Y = step_2a_extend_the_vectors(X_t[hour], Y_t[hour], expand_type="mean")
            mean_cv_score_dict[N, hour] = []
            for m in range(multiple_runs):
                cv_score = step_2b_calculate_GOF(X, Y, "regression")
                mean_cv_score_dict[N, hour].append(cv_score)

            with open("temp_files/final_results.csv", "a") as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow([N, hour] + mean_cv_score_dict[N, hour] + list(X.shape) + list(Y.shape))

    return mean_cv_score_dict


def box_plot_of_CCT_vs_hours():
    """
    get all Y
    :return:
    """


if __name__ == "__main__":
    starttime = time.time()

    RUN_MODE = "PLOTTING"  # ["RUNNING", "PLOTTING"]:
    MULTIPLE_RUN = 10
    names_of_multiple_run_cols = ["run_" + str(i) for i in range(1, MULTIPLE_RUN + 1)]

    if RUN_MODE == "RUNNING":

        with open("temp_files/final_results.csv", "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(
                ["grid_size", "hour"]
                + names_of_multiple_run_cols
                + ["X_shape_0", "X_Shape_1", "Y_Shape_0", "Y_Shape_1"]
            )

        mean_cv_score_dict = step_3(10, 175, 10, multiple_runs=MULTIPLE_RUN, use_saved_vectors=True)

        # csvwriter.writerow(["Repeat after all runs, same as above"])
        with open("temp_files/final_results_after_complete.csv", "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(["grid_size", "hour"] + ["run_" + str(x) for x in range(1, MULTIPLE_RUN + 1)])
            for grid_size, hour in mean_cv_score_dict:
                csvwriter.writerow([grid_size, hour] + mean_cv_score_dict[grid_size, hour])

        print(round(time.time() - starttime, 2), " seconds")

    elif RUN_MODE == "PLOTTING":
        data = pandas.read_csv("temp_files/final_results.csv")
        print(pandas.read_csv("temp_files/final_results.csv"))

        data = data.dropna(subset=names_of_multiple_run_cols)

        for col in names_of_multiple_run_cols:
            print(col)
            data = data[data.eval(col) != -99999]

        # invert all values (to get the mse)
        #  not needed if not using pipeline
        # data[names_of_multiple_run_cols] = -data[names_of_multiple_run_cols]

        for hour_ in range(24):
            hourly_data = data[data.hour == hour_]
            if hourly_data.shape[0] == 0:
                continue
            print(data.shape, hourly_data.shape)
            plotting_dict = {}
            plotting_dict["max"] = hourly_data[names_of_multiple_run_cols].max(axis=1)
            plotting_dict["min"] = hourly_data[names_of_multiple_run_cols].min(axis=1)
            plotting_dict["median"] = hourly_data[names_of_multiple_run_cols].median(axis=1)

            for col in ["median"]:  # min", "max",
                print(hourly_data["grid_size"])
                print(plotting_dict[col])
                plt.plot(hourly_data["grid_size"], plotting_dict[col], label=col + "- hour" + str(hour_))
            plt.grid(True)
            plt.xlabel("Scale")
            plt.legend(fontsize=8, loc="upper right")
            # plt.yscale("log")
            # plt.fill_between(hourly_data["grid_size"], plotting_dict["min"], plotting_dict["max"], color="yellow", alpha=0.4)

        plt.show()
        print(data)
