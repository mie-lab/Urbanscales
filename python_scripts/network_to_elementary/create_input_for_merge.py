import pickle
from shapely import geometry
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from python_scripts.four_steps.steps_combined import convert_to_single_statistic_by_removing_minus_1
from smartprint import smartprint as sprint
from scipy.ndimage.measurements import label


def get_bbox(osm_tiles_stats_dict):
    bbox_list = []
    for keyval in osm_tiles_stats_dict:
        try:
            # need to fix this messy way to read dictionary @Nishant
            key, val = list(keyval.keys())[0], list(keyval.values())[0]
            assert val != "EMPTY_STATS"
            bbox_list.append(key)
        except:
            continue
    return bbox_list


def get_bbox_as_list_of_list(scale):
    # extra step because by default we have list of tuples in the osm_tiles_stats_dict_XX.pickle files
    with open(
        "/Users/nishant/Documents/GitHub/WCS/python_scripts/network_to_elementary/osm_tiles_stats_dict"
        + str(scale)
        + ".pickle",
        "rb",
    ) as handle:
        bbox_list = get_bbox(pickle.load(handle))

    bbox_lol = []
    for lat1, lon1, lat2, lon2 in bbox_list:
        # revert for homogeiniety in gdal
        bbox_lol.append([[lon1, lat1], [lon2, lat2]])
    return bbox_lol


def create_hierarchy_dict(base_level, number_of_hierarchy):
    # @yatao's comment: read bbox_file, a set of bbox file in multi-hierarchies
    # dict_bbox = {'hierarchy_1':[[]], 'hierarchy_2':[[]], 'hierarchy_3':[[]]}

    dict_bbox = {}

    for i in range(number_of_hierarchy):
        scale = base_level * (2 ** i)
        dict_bbox["hierarchy_" + str(i + 1)] = get_bbox_as_list_of_list(scale)
    return dict_bbox


def get_isl_and_seeds_bboxes_for_best_fit_hierarchy(bbox_list):
    """

    :param bbox_list:
    :return:
    """
    # two handmade polygons
    poly_1 = [
        [103.86955261230469, 1.3439853145503564],
        [103.86611938476561, 1.3426124009441485],
        [103.86474609375, 1.334718132769963],
        [103.86268615722656, 1.3285399921660488],
        [103.87676239013672, 1.3299129136379466],
        [103.88259887695311, 1.336434280186183],
        [103.86955261230469, 1.3439853145503564],
    ]

    poly_2 = [
        [103.86955261230469, 1.3439853145503564],
        [103.86611938476561, 1.3426124009441485],
        [103.86474609375, 1.334718132769963],
        [103.86268615722656, 1.3285399921660488],
        [103.87676239013672, 1.3299129136379466],
        [103.88259887695311, 1.336434280186183],
        [103.86955261230469, 1.3439853145503564],
    ]

    poly_1 = geometry.Polygon([[lon, lat] for lon, lat in poly_1])
    poly_2 = geometry.Polygon([[lon, lat] for lon, lat in poly_2])

    island_1 = []
    island_2 = []

    for key in bbox_list:
        [lon1, lat1], [lon2, lat2] = key
        bb_poly = geometry.Polygon(
            [
                [lon1, lat1],
                [lon1, lat2],
                [lon2, lat2],
                [lon2, lat1],
            ]
        )
        if poly_1.intersects(bb_poly):
            island_1.append([[lon1, lat1], [lon2, lat2]])
        if poly_2.intersects(bb_poly):
            island_2.append([[lon1, lat1], [lon2, lat2]])

    # The keys of these three dictionaries are the same
    # read the list of bbox in each island in the best-fit hierarchy
    # dict_islands = {'island_1':[], 'island_2':[], 'island_3':[], 'island_4':[]}

    dict_islands = {}
    dict_islands["island_1"] = island_1
    dict_islands["island_2"] = island_2

    # seed file:
    # read start-seed (bbox) in each island in the best-fit hierarchy
    # dict_seeds = {'island_1':[], 'island_2':[], 'island_3':[], 'island_4':[]}
    dict_seeds = {}

    dict_seeds["island_1"] = island_1[int(len(island_1) * np.random.rand())]
    dict_seeds["island_2"] = island_2[int(len(island_2) * np.random.rand())]

    return dict_islands, dict_seeds


def create_islands_two_methods(
    N,
    time_filter=[5, 6, 7, 8],
    method_for_single_statistic="median_across_all",
    island_method="conn_comp",
    plotting_enabled=True,
):
    """

    Args:
        N:
        time_filter:
        method_for_single_statistic:
        island_method: "conn_comp" or "dbscan"
        plotting_enabled:

    Returns:

    """
    assert island_method in ["conn_comp", "dbscan"]

    with open(
        "/Users/nishant/Documents/GitHub/WCS/python_scripts/network_to_elementary/dict_bbox_hour_date_to_CCT"
        + str(N)
        + ".pickle",
        "rb",
    ) as f1:
        dict_bbox_hour_date_to_CCT = pickle.load(f1)

    dict_bbox_hour_date_to_CCT = convert_to_single_statistic_by_removing_minus_1(
        dict_bbox_hour_date_to_CCT, time_filter, method_for_single_statistic
    )

    if plotting_enabled:
        plt.clf()
        lon_centre = []
        lat_centre = []
        for bbox in dict_bbox_hour_date_to_CCT.keys():
            lon_centre.append((bbox[1] + bbox[3]) / 2)
            lat_centre.append((bbox[0] + bbox[2]) / 2)
            plt.gca().add_patch(
                matplotlib.patches.Rectangle(
                    (bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0], lw=0.8, alpha=0.5, color="yellow"
                )
            )
        plt.scatter(lon_centre, lat_centre, marker="s", s=100 / N)

        plt.show()

    if island_method == "conn_comp":
        a = np.random.rand(N, (int(N * 1.5))) * 0

        lon1, lat1, lon2, lat2 = list(dict_bbox_hour_date_to_CCT.keys())[0]
        sprint(lon1, lat1, lon2, lat2)
        lon1, lat1, lon2, lat2 = list(dict_bbox_hour_date_to_CCT.keys())[1]
        sprint(lon1, lat1, lon2, lat2)
        max_lat = -1
        max_lon = -1
        min_lat = 99999
        min_lon = 99999
        for bbox in dict_bbox_hour_date_to_CCT.keys():
            lon1, lat1, lon2, lat2 = bbox

            max_lat = max(lat1, max_lat)
            max_lat = max(lat2, max_lat)
            max_lon = max(lon1, max_lon)
            max_lon = max(lon2, max_lon)

            min_lat = min(lat1, min_lat)
            min_lat = min(lat2, min_lat)
            min_lon = min(lon1, min_lon)
            min_lon = min(lon2, min_lon)

        sprint(a.shape)
        delta_x = (max_lon - min_lon) / a.shape[0]
        delta_y = (max_lat - min_lat) / a.shape[1]

        for bbox in dict_bbox_hour_date_to_CCT.keys():
            lon1, lat1, lon2, lat2 = bbox
            sprint(int((lon1 - min_lon) / delta_x), int((lat1 - min_lat) / delta_y))
            a[int((lon1 - min_lon) / delta_x), int((lat1 - min_lat) / delta_y)] = 1

        a = np.array(a, dtype=np.int)
        structure = np.ones((3, 3), dtype=np.int)
        labeled, ncomponents = label(a, structure)
        plt.imshow(a, origin="lower")
        plt.show()

        indices = np.indices(a.shape).T[:, :, [1, 0]]
        sprint(indices)

        unique = []
        for i in range(labeled.shape[0]):
            for j in range(labeled.shape[1]):
                unique.append(labeled[i, j])
        unique = list(set(unique))

        list.sort(unique)

        map_sequential = {}
        for i in range(len(unique)):
            map_sequential[unique[i]] = i
        sprint(unique)
        for key in map_sequential:
            print(key, map_sequential[key])

        for i in range(labeled.shape[0]):
            for j in range(labeled.shape[1]):
                labeled[i, j] = map_sequential[labeled[i, j]]

        cmap = matplotlib.cm.get_cmap("nipy_spectral")
        plt.imshow(labeled, origin="lower", cmap=cmap)
        sprint(ncomponents)
        plt.show()

        for row in labeled:
            print(row)

        # sprint(labeled)
        # plt.hist(a.flatten(), 10)
        # plt.show()
        #
        # plt.hist(labeled.flatten(), 10)
        # plt.show()


if __name__ == "__main__":
    base_level = 5
    dict_bbox = create_hierarchy_dict(base_level, 6)

    best_fit_hierarchy = 5
    dict_islands, dict_seeds = get_isl_and_seeds_bboxes_for_best_fit_hierarchy(
        dict_bbox["hierarchy_" + str(best_fit_hierarchy)]
    )

    with open("dict_bbox_" + str(base_level) + "_.pickle", "wb") as f:
        pickle.dump(dict_bbox, f, protocol=4)
    with open("dict_islands_" + str(best_fit_hierarchy) + "_.pickle", "wb") as f:
        pickle.dump(dict_islands, f, protocol=4)
    with open("dict_seeds_" + str(best_fit_hierarchy) + "_.pickle", "wb") as f:
        pickle.dump(dict_seeds, f, protocol=4)

    create_islands_two_methods(40)

    do_nothing = True
