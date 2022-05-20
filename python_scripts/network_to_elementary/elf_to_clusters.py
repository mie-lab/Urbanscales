import pickle
import numpy as np
from sklearn.cluster import DBSCAN, KMeans

# from yellowbrick.cluster.elbow import kelbow_visualizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def get_gof_clustering(
    keys_bbox_list,
    vals_vector_array,
    clustering_algo="dbscan",
    scaling=True,
    pca=True,
    pca_comp=3,
    k_means_cluster_number_max=10,
):
    X = vals_vector_array

    if scaling:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    if pca:
        pca_ = PCA(n_components=pca_comp)
        X = pca_.fit_transform(X)
        print("pca_.explained_variance_ratio_", pca_.explained_variance_ratio_)

    if clustering_algo == "dbscan":
        clustering = DBSCAN(
            eps=1,
            min_samples=5,
            metric="euclidean",
            metric_params=None,
            algorithm="auto",
            leaf_size=30,
            p=None,
            n_jobs=None,
        ).fit(X)
    elif clustering_algo == "kmeans":

        # kelbow_visualizer(KMeans(random_state=0), X, k=(2, 15))
        plt.savefig("elbow_plot.png", dpi=300)

        distortion = []

        N = k_means_cluster_number_max
        for num_clusters in range(2, N):

            clustering = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
            distortion.append(clustering.inertia_)

            # plotting
            cmap = plt.get_cmap("tab10")
            centre_lat_list, centre_lon_list = [], []
            counter = 0
            for bbox in keys_bbox_list:
                lat1, lon1, lat2, lon2 = bbox
                centre_lat_list.append((lat1 + lat2) / 2)
                centre_lon_list.append((lon1 + lon2) / 2)

                number_of_clusters = len(np.unique(clustering.labels_))
                # clustering.labels_ + 1 to set the indices from 0, otherwise no corresponding color present
                plt.scatter(
                    lon1, lat1, s=3, marker="s", color=cmap((clustering.labels_[counter] + 1) / number_of_clusters)
                )
                counter += 1
            plt.savefig("clusters_num_clusters_" + str(num_clusters) + ".png", dpi=300)
            plt.show(block=False)

        plt.plot(range(2, N), distortion, linewidth=3)
        plt.xlabel("Number of clusters")
        plt.ylabel("Distortion")
        # plt.yscale("log")
        plt.savefig("k_means_elbow_plot.png", dpi=300)
        plt.show(block=False)

    cmap = plt.get_cmap("tab10")
    centre_lat_list, centre_lon_list = [], []
    counter = 0
    for bbox in keys_bbox_list:
        lat1, lon1, lat2, lon2 = bbox
        centre_lat_list.append((lat1 + lat2) / 2)
        centre_lon_list.append((lon1 + lon2) / 2)

        number_of_clusters = len(np.unique(clustering.labels_))
        # clustering.labels_ + 1 to set the indices from 0, otherwise no corresponding color present
        plt.scatter(lon1, lat1, s=3, marker="s", color=cmap((clustering.labels_[counter] + 1) / number_of_clusters))
        counter += 1
    plt.savefig("clusters.png", dpi=300)
    plt.show(block=False)
    # print (clustering.labels_)


def impute_with_mean(vals_vector_array):
    print("Input features shape: ", vals_vector_array.shape)
    print(
        "Number of nan's:",
        np.count_nonzero(np.isnan(vals_vector_array)),
        "\n At the following locations ",
        np.argwhere(np.isnan(vals_vector_array)),
    )
    print("we replace the missing values with the mean of that column")
    nan_index = np.argwhere(np.isnan(vals_vector_array))
    impute_mean = np.nanmean(vals_vector_array[:, 10])
    vals_vector_array[nan_index] = impute_mean

    print("After imputing\n")
    print(
        "Number of nan's:",
        np.count_nonzero(np.isnan(vals_vector_array)),
        "\n At the following locations ",
        np.argwhere(np.isnan(vals_vector_array)),
        "\n\n",
    )
    return vals_vector_array


def osm_tiles_states_to_vectors(osm_tiles_stats_dict):
    """
    {'n': 13,
    'm': 14,
    'k_avg': 2.1538461538461537,
    'edge_length_total': 911.3389999999999,
     'edge_length_avg': 65.09564285714285,
     'streets_per_node_avg': 2.1538461538461537,
     'streets_per_node_counts': {0: 2, 1: 0, 2: 7, 3: 2, 4: 2},
     'streets_per_node_proportions': {0: 0.15384615384615385, 1: 0.0, 2: 0.5384615384615384, 3: 0.15384615384615385,
                                      4: 0.15384615384615385},
     'intersection_count': 11,
     'street_length_total': 911.3389999999998,
     'street_segment_count': 14,
     'street_length_avg': 65.09564285714285,
     'circuity_avg': 0.999997647629181,
     'self_loop_proportion': 0.0}
    """

    keys_bbox_list = []
    vals_vector_list = []
    for keyval in osm_tiles_stats_dict:
        try:
            # need to fix this messy way to read dictionary @Nishant
            key, val = list(keyval.keys())[0], list(keyval.values())[0]
            assert val != "EMPTY_STATS"
        except:
            continue

        single_vector = []
        for key_2 in (
            "n",
            "m",
            "k_avg",
            "edge_length_total",
            "edge_length_avg",
            "streets_per_node_avg",
            "intersection_count",
            "street_length_total",
            "street_segment_count",
            "street_length_avg",
            "circuity_avg",
            "self_loop_proportion",
        ):
            single_vector.append(val[key_2])
        keys_bbox_list.append(key)
        vals_vector_list.append(single_vector)

    vals_vector_array = np.array(vals_vector_list)
    vals_vector_array = impute_with_mean(vals_vector_array)
    return dict(zip(keys_bbox_list, vals_vector_array))


if __name__ == "__main__":
    with open("osm_tiles_stats_dict.pickle", "rb") as f:
        osm_tiles_stats_dict = pickle.load(f)

    keys_bbox_list, vals_vector_array = osm_tiles_states_to_vectors(osm_tiles_stats_dict)
    get_gof_clustering(keys_bbox_list, vals_vector_array, clustering_algo="kmeans", pca=True, pca_comp=3)


do_nothing = True
