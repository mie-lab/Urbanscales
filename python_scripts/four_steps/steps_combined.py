# First, a uniform segmentation of the urban space into spatial grids is done and eight graph-based features are extracted for each grid.
import os
# from python_scripts.network_to_elementary.tiles_to_elementary import step_1_osm_tiles_to_features
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from shapely.geometry import Polygon
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from smartprint import smartprint as sprint

import config
from python_scripts.network_to_elementary.elf_to_clusters import osm_tiles_states_to_vectors
from python_scripts.network_to_elementary.osm_to_tiles import fetch_road_network_from_osm_database
from python_scripts.network_to_elementary.process_incidents import create_bbox_to_CCT


def auxiliary_func_G_for_curved_paths(read_osm_from_file):
    sg_bbox_poly = [
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
    geo = {
        "type": "Polygon",
        "coordinates": [sg_bbox_poly],
    }

    array_bbox_poly = np.array(sg_bbox_poly)
    min_lon, max_lon, min_lat, max_lat = (
        array_bbox_poly[:, 0].min(),
        array_bbox_poly[:, 0].max(),
        array_bbox_poly[:, 1].min(),
        array_bbox_poly[:, 1].max(),
    )

    fname_osm = config.intermediate_files_path + "G_OSM_extracted.pickle"
    if os.path.isfile(fname_osm):
        with open(fname_osm, "rb") as f1:
            G = pickle.load(f1)
    else:
        poly = Polygon([tuple(l) for l in geo["coordinates"][0]])

        # G_proj = osm.project_graph(G)
        # fig, ax = osm.plot_graph(G_proj)

        G = fetch_road_network_from_osm_database(polygon=poly, network_type="drive", custom_filter=config.custom_filter)
        with open(fname_osm, "wb") as f2:
            pickle.dump(G, f2, protocol=4)

        # orig = (1.34294, 103.74631)
        # dest = (1.34499, 103.74022)

    G = ox.speed.add_edge_speeds(G)
    G = ox.speed.add_edge_travel_times(G)

    return G, (min_lon, max_lon, min_lat, max_lat)


def generate_bbox_CCT_from_file(N, use_route_path=True):
    """

    :param N:
    :param folder_path:
    :param read_bbox_CCT_from_file:
    :return:
    """
    # Shape:     # dict_bbox_to_CCT[bbox, incident_start_hour, incident_start_date].append(CCT)
    dict_bbox_hour_date_to_CCT, unique_dates = create_bbox_to_CCT(
        csv_file_name=config.combined_incidents_file,
        N=N,
        folder_path=config.intermediate_files_path,
        graph_with_edge_travel_time=auxiliary_func_G_for_curved_paths(read_osm_from_file=True),
        use_route_path=use_route_path,
        plotting_enabled=config.plotting_enabled,
        suppress_warning=True,
    )
    yahan_pahuch_Gaye = "True"
    with open(config.intermediate_files_path + "dict_bbox_hour_date_to_CCT" + str(N) + ".pickle", "wb") as f2:
        pickle.dump(dict_bbox_hour_date_to_CCT, f2, protocol=4)


def convert_to_single_statistic_by_removing_minus_1(
    dict_bbox_hour_date_to_CCT, timefilter, method_for_single_statistic
):
    """

    Args:
        methdict_bbox_hour_date_to_CCT:
        timefilter:
        method_for_single_statistic:

    Returns:

    """
    dict_bbox_hour_date_to_CCT_copy = {}
    for key in dict_bbox_hour_date_to_CCT:
        array_24_x_dates = dict_bbox_hour_date_to_CCT[key]
        if timefilter != -1:
            assert (type(timefilter) == list) or type(timefilter) == tuple
            assert max(timefilter) <= array_24_x_dates.shape[0]
            assert array_24_x_dates.shape[0] == 24
            array_24_x_dates = array_24_x_dates.to_numpy()
            array_24_x_dates = array_24_x_dates[timefilter, :]

        if method_for_single_statistic == "median_across_all":
            if np.count_nonzero(array_24_x_dates != -1) == 0:
                # if empty slice in the line below, we just skip this one.
                # if there is no value that is -1, we ignore this key
                continue
            dict_bbox_hour_date_to_CCT_copy[key] = np.median(array_24_x_dates[array_24_x_dates != -1])
        elif method_for_single_statistic == "sum":
            if np.count_nonzero(array_24_x_dates != -1) == 0:
                # if empty slice in the line below, we just skip this one.
                # if there is no value that is -1, we ignore this key
                continue
            dict_bbox_hour_date_to_CCT_copy[key] = np.sum(array_24_x_dates[array_24_x_dates != -1])
        else:
            print("Wrong parameter for method_for_single_statistic: ", method_for_single_statistic)
            sys.exit(0)
    return dict_bbox_hour_date_to_CCT_copy


# step 2
def step_2(
    N,
    folder_path,
    read_bbox_CCT_from_file,
    plot_bboxes_on_route=False,
    generate_incidents_routes=False,
    method_for_single_statistic="median_across_all",
    timefilter=[5, 6, 7, 8],
):
    """

    :param N:
    :param folder_path:
    :param read_bbox_CCT_from_file:
    :param plot_bboxes_on_route:
    :param generate_incidents_routes:
    :param method_for_single_statistic:
    :param timefilter: timefilter=[5,6,7,8] implies 6AM to 10AM, NOT 5AM TO 9AM; cuz of zero indexing; if timefilter
                                            is -1, we can ignore the timefilter (bascially take all the 24 hours)
    :return:
    """
    if not read_bbox_CCT_from_file:
        generate_bbox_CCT_from_file(N, folder_path, generate_incidents_routes, use_route_path=False)

        with open(folder_path + "dict_bbox_hour_date_to_CCT" + str(N) + ".pickle", "rb") as f1:
            dict_bbox_hour_date_to_CCT = pickle.load(f1)
    elif read_bbox_CCT_from_file:
        if not os.path.isfile(folder_path + "dict_bbox_hour_date_to_CCT" + str(N) + ".pickle"):
            generate_bbox_CCT_from_file(N, use_route_path=config.use_route_path_curved)
        with open(folder_path + "dict_bbox_hour_date_to_CCT" + str(N) + ".pickle", "rb") as f1:
            dict_bbox_hour_date_to_CCT = pickle.load(f1)



    else:
        print("Something wrong in step 2; Exiting execution")
        sys.exit()

    with open(folder_path + "osm_tiles_stats_dict" + str(N) + ".pickle", "rb") as f3:
        osm_tiles_stats_dict = pickle.load(f3)

    dict_bbox_to_vectors = osm_tiles_states_to_vectors(osm_tiles_stats_dict, verbose=False)

    dict_bbox_hour_date_to_CCT = convert_to_single_statistic_by_removing_minus_1(
        dict_bbox_hour_date_to_CCT, timefilter, method_for_single_statistic
    )

    X = []
    Y = []

    # the bbox file might have more grids than the incidents,
    # but it cannot be the other way round; Plus the remaining boxes
    # should be present in both dictionaries, hence the assert statement below
    # remaining implies the grids/ bboxes with at least one incident throughout the 45 days
    assert len(set(set(dict_bbox_to_vectors.keys() - dict_bbox_hour_date_to_CCT.keys()))) == len(
        dict_bbox_to_vectors
    ) - len(dict_bbox_hour_date_to_CCT)

    for bbox in dict_bbox_hour_date_to_CCT:
        X.append(dict_bbox_to_vectors[bbox])
        Y.append(dict_bbox_hour_date_to_CCT[bbox])

    # normalise Y; hard code to 2 hours
    Y = np.array(Y)
    max_Y = np.max(Y)
    # Y = [y / 7200 for y in Y]
    Y = Y/max_Y

    if plot_bboxes_on_route:
        lon_centre = []
        lat_centre = []
        for bbox in dict_bbox_hour_date_to_CCT.keys():
            lon_centre.append((bbox[1] + bbox[3]) / 2)
            lat_centre.append((bbox[0] + bbox[2]) / 2)

        plt.scatter(lon_centre, lat_centre, marker="s", s=100 / N)
        plt.show()

    sprint(N, len(dict_bbox_to_vectors))
    sprint(len(X), len(Y))

    # test homogeniety
    shape = X[0].shape
    assert len(X) == len(Y)
    for row in X:
        assert row.shape == shape

    for row in Y:
        assert isinstance(row, float)

    return X, Y


def step_2b_calculate_GOF(X, Y, model=None):
    if model == None:
        print("model not passed; Exiting code!")
        sys.exit(0)

    # try:
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, random_state=int(time.time()))

    X_train = X
    y_train = Y
    X_test = X
    y_test = Y


    # ('scaler', StandardScaler()),
    ("pca", PCA(n_components=8))
    # ("pca", PCA(n_components=12))
    pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=2)), ("LinR", model)])


    # The pipeline can be used as any other estimator
    # and avoids leaking the test set into the train set
    # print(pipe.fit(X_train, y_train))
    # print(pipe.score(X_test, y_test), " GoF measure")

    pipe.fit(X_train, y_train)
    pipe.score(X_train, y_train)

    # scores = cross_val_score(pipe, X, Y, cv=7)
    # print("CV GoF measure: ", scores)
    # print("Mean CV (GoF):", np.mean(scores))
    # return np.mean(scores)
    # return pipe.score(X_test, y_test)
    # y_pred = pipe.predict(X_test)
    # return mean_squared_error(y_test, y_pred)

    y_pred = pipe.predict(X_test)
    return mean_squared_error(y_test, y_pred)

    # except:
    #     print("Something wrong in model fitting")
    #     sys.exit(0)
    # return -99999


def step_3(
    multiple_runs=1,
    use_saved_vectors=False,
    read_bbox_CCT_from_file=True,
    plot_bboxes_on_route=False,
    generate_incidents_routes=False,
):
    """

    :param min_:
    :param max_:
    :param step_:
    :return:
    """
    mean_cv_score_dict = {}
    # for N in range(min_, max_, step_):

    # if use_saved_vectors:
    #     with open("temp_files/" + "X_t_Y_t_" + str(N) + ".pickle", "rb") as f:
    #         X_t, Y_t = pickle.load(f)
    # else:
    model_name = ["LR", "RF", "GBM"]
    X_len = {}
    X_len_multiple_plots = {}

    base_list = [5]
    hierarchy_max = 7

    for base in base_list:  # , 6, 7]:  # , 6, 7]:  # [5, 6, 7, 8, 9, 10]
        for m_i in range(3):
            if model_name[m_i] == "LR":
                model = LinearRegression()
            elif model_name[m_i] == "RF":
                model = RandomForestRegressor()
            elif model_name[m_i] == "GBM":
                model = GradientBoostingRegressor()

            # timefilter = list(range(0, 24))  #  [5, 6, 7, 8]
            timefilter = [5, 6, 7, 8, 9]
            # X_len = {}

            for i in range(hierarchy_max):  # :range(60, 120, 10):

                scale = base * (2 ** i)
                if scale > 200:
                    continue

                X, Y = step_2(
                    N=scale,
                    folder_path=config.intermediate_files_path,
                    read_bbox_CCT_from_file=read_bbox_CCT_from_file,
                    plot_bboxes_on_route=plot_bboxes_on_route,
                    generate_incidents_routes=generate_incidents_routes,
                    method_for_single_statistic="sum",
                    timefilter=timefilter,
                )
                X_len[scale] = len(X)
                X_len_multiple_plots[scale, tuple(timefilter)] = len(X)

                mean_cv_score_dict[model_name[m_i], scale, tuple(timefilter)] = []

                # no need to plot for different models, they are the same plot
                if m_i == 0:
                    X = np.array(X)
                    pca = PCA().fit(X)
                    plt.plot(np.cumsum(pca.explained_variance_ratio_), label="Scale" + str(scale))
                    plt.grid()
                    plt.xlabel('number of components')
                    plt.ylabel('cumulative explained variance')
                    plt.legend()
                    plt.savefig(config.outputfolder + "PCA_variance_scale" + str(scale) + ".png", dpi=300)
                    print("scale, number_of_data_points, mean_cvscores, mean_std")

                for m in range(multiple_runs):
                    cv_score = step_2b_calculate_GOF(X, Y, model=model)
                    mean_cv_score_dict[model_name[m_i], scale, tuple(timefilter)].append(cv_score)

                mean = np.mean(mean_cv_score_dict[model_name[m_i], scale, tuple(timefilter)])
                std = np.std(mean_cv_score_dict[model_name[m_i], scale, tuple(timefilter)])

                # append the mean in the end
                mean_cv_score_dict[model_name[m_i], scale, tuple(timefilter)].append(mean)

                # append std
                mean_cv_score_dict[model_name[m_i], scale, tuple(timefilter)].append(std)



    plt.clf()
    for base in base_list:  # , 6, 7]:  # [5, 6, 7, 8, 9, 10]
        for m_i in range(3):
            if model_name[m_i] == "LR":
                model = LinearRegression()
            elif model_name[m_i] == "RF":
                model = RandomForestRegressor()
            elif model_name[m_i] == "GBM":
                model = GradientBoostingRegressor()

            x_axis = []
            y_axis = []
            z_axis = []

            for i in range(hierarchy_max):  # :range(60, 120, 10):

                scale = base * (2 ** i)
                if scale > 200:
                    continue

                for key in mean_cv_score_dict:
                    if key[1] != scale:
                        continue
                    if key[0] != model_name[m_i]:
                        continue

                    print(key[1], X_len_multiple_plots[key[1], key[2]], mean_cv_score_dict[key][-1], sep=",")
                    x_axis.append(key[1])
                    y_axis.append(mean_cv_score_dict[key][-2])
                    z_axis.append(mean_cv_score_dict[key][-1])

            xyz = zip(x_axis, y_axis, z_axis)
            xyz = sorted(xyz, key=lambda x: x[0])
            # print(list(xyz))
            x_list = []
            y_list = []
            z_list = []
            for x, y, z in xyz:
                x_list.append(x)
                y_list.append(y)
                z_list.append(z)
            plt.plot(x_list, y_list, label=model_name[m_i] + " base-" + str(base))

        # plt.plot(x_list, z_list, label=model_name[m_i] + "std")
    plt.legend(fontsize=10)
    plt.ylabel("MSE")
    plt.xlabel("Scale")
    # plt.title("Base: "+str(base))
    # plt.ylim(0, 0.012)
    # plt.yscale("log")
    plt.savefig(config.outputfolder + "All_bases_all_models_" + str(base) + ".png", dpi=300)

    plt.show()

    #
    # with open("temp_files/final_results.csv", "a") as f:
    #     csvwriter = csv.writer(f)
    #     csvwriter.writerow([N, hour] + mean_cv_score_dict[N, hour] + list(X.shape) + list(Y.shape))
    do_nothing = True

    # return mean_cv_score_dict


if __name__ == "__main__":
    starttime = time.time()

    RUN_MODE = "RUNNING"  # ["RUNNING", "PLOTTING"]:
    MULTIPLE_RUN = 1

    if RUN_MODE == "RUNNING":

        mean_cv_score_dict = step_3(
            multiple_runs=MULTIPLE_RUN,
            use_saved_vectors=False,
            read_bbox_CCT_from_file=True,
            plot_bboxes_on_route=False,
            generate_incidents_routes=False,
        )

        print(round(time.time() - starttime, 2), " seconds")

"""
if __name__ == "__main__":
    for base in [5]:  # [5, 6, 7, 8, 9, 10]
        for i in range(3):  # :range(60, 120, 10):
            scale = base * (2 ** i)

            generate_bbox_CCT_from_file(N=scale, use_route_path=False)
"""

# elif RUN_MODE == "PLOTTING":
#     data = pandas.read_csv("temp_files/final_results.csv")
#     print(pandas.read_csv("temp_files/final_results.csv"))
#
#     data = data.dropna(subset=names_of_multiple_run_cols)
#
#     for col in names_of_multiple_run_cols:
#         print(col)
#         data = data[data.eval(col) != -99999]
#
#     # invert all values (to get the mse)
#     #  not needed if not using pipeline
#     # data[names_of_multiple_run_cols] = -data[names_of_multiple_run_cols]
#
#     for hour_ in range(24):
#         hourly_data = data[data.hour == hour_]
#         if hourly_data.shape[0] == 0:
#             continue
#         print(data.shape, hourly_data.shape)
#         plotting_dict = {}
#         plotting_dict["max"] = hourly_data[names_of_multiple_run_cols].max(axis=1)
#         plotting_dict["min"] = hourly_data[names_of_multiple_run_cols].min(axis=1)
#         plotting_dict["median"] = hourly_data[names_of_multiple_run_cols].median(axis=1)
#
#         for col in ["median"]:  # min", "max",
#             print(hourly_data["grid_size"])
#             print(plotting_dict[col])
#             plt.plot(hourly_data["grid_size"], plotting_dict[col], label=col + "- hour" + str(hour_))
#         plt.grid(True)
#         plt.xlabel("Scale")
#         plt.legend(fontsize=8, loc="upper right")
#         # plt.yscale("log")
#         # plt.fill_between(hourly_data["grid_size"], plotting_dict["min"], plotting_dict["max"], color="yellow", alpha=0.4)
#
#     plt.show()
#     print(data)
