# First, a uniform segmentation of the urban space into spatial grids is done and eight graph-based features are extracted for each grid.
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from python_scripts.network_to_elementary.elf_to_clusters import osm_tiles_states_to_vectors
from python_scripts.network_to_elementary.osm_to_tiles import fetch_road_network_from_osm_database
from python_scripts.network_to_elementary.process_incidents import create_bbox_to_CCT
from python_scripts.network_to_elementary.tiles_to_elementary import step_1_osm_tiles_to_features
import pickle
from shapely.geometry import Polygon
import numpy as np
import sys
import osmnx as ox
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, "/Users/nishant/Documents/GitHub/WCS/python_scripts/network_to_elementary")
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

    dict_bbox_to_CCT = create_bbox_to_CCT(
        csv_file_name="combined_incidents_13_days.csv",
        read_from_pickle=True,
        N=N,
        folder_path=folder_path,
        graph_with_edge_travel_time=auxiliary_func_G_for_curved_paths(),
        use_route_path=True,
    )

    with open(folder_path + "osm_tiles_stats_dict" + str(N) + ".pickle", "rb") as f:
        osm_tiles_stats_dict = pickle.load(f)

    keys_bbox_list, vals_vector_array = osm_tiles_states_to_vectors(osm_tiles_stats_dict)

    Y = []
    X = []
    count_present = 0
    count_absent = 0
    for count, key in enumerate(keys_bbox_list):
        if key in dict_bbox_to_CCT:
            Y.append(dict_bbox_to_CCT[key])
            X.append(vals_vector_array[count])
            count_present += 1
        else:
            count_absent += 1
    print("2nd step: Count present in CCT file: ", count_present)
    print("2nd step: Count absent in CCT file: ", count_absent)
    return X, Y


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
    return np.array(x), np.array(y)


def step_2b_calculate_GOF(X, Y, model="regression"):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, random_state=0)

    # ('scaler', StandardScaler())
    pipe = Pipeline([("scaler", StandardScaler()), ("RF", RandomForestRegressor())])
    # The pipeline can be used as any other estimator
    # and avoids leaking the test set into the train set
    print(pipe.fit(X_train, y_train))
    print(pipe.score(X_test, y_test), " GoF measure")

    scores = cross_val_score(pipe, X, Y, cv=5)
    print("CV GoF measure: ", scores)
    print("Mean CV (GoF):", np.mean(scores))
    return np.mean(scores)


def step_3(min_, max_, step_):
    """

    :param min_:
    :param max_:
    :param step_:
    :return:
    """
    mean_cv_score_dict = {}
    for N in range(min_, max_, step_):
        X, Y = step_2(N, "/Users/nishant/Documents/GitHub/WCS/python_scripts/network_to_elementary/")
        X, Y = step_2a_extend_the_vectors(X, Y, expand_type="max")
        cv_score = step_2b_calculate_GOF(X, Y, "regression")
        mean_cv_score_dict[N] = cv_score
    return mean_cv_score_dict


if __name__ == "__main__":
    mean_cv_score_dict = step_3(10, 100, 5)
    for key in mean_cv_score_dict:
        print(key, mean_cv_score_dict[key])
