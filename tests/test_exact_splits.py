import os
import pickle
import matplotlib.pyplot as plt
import pytest
from smartprint import smartprint as sprint


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


def extract_lat_list_lon_list(bbox_list):
    lat_list = []
    lon_list = []
    for lat1, lon1, lat2, lon2 in bbox_list:
        lat_list.append(lat1)
        lat_list.append(lat2)
        lon_list.append(lon1)
        lon_list.append(lon2)
    return lat_list, lon_list


# @pytest.fixture
def ssubset(base_N):
    """
    Issue #58 step 1
    """
    splitted_N = base_N * 2
    with open(
        "../python_scripts/network_to_elementary/" + "osm_tiles_stats_dict" + str(base_N) + ".pickle", "rb"
    ) as handle:
        base_N_osm_bbox_list = get_bbox(pickle.load(handle))
    sprint(os.getcwd())
    with open(
        "../python_scripts/network_to_elementary/" + "osm_tiles_stats_dict" + str(splitted_N) + ".pickle", "rb"
    ) as handle:
        splitted_N_osm_bbox_list = get_bbox(pickle.load(handle))

    sprint(len((base_N_osm_bbox_list)))
    sprint(len((splitted_N_osm_bbox_list)))

    lat_list_1, lon_list_1 = extract_lat_list_lon_list(base_N_osm_bbox_list)
    lat_list_2, lon_list_2 = extract_lat_list_lon_list(splitted_N_osm_bbox_list)

    set_lat_1 = set(lat_list_1)
    set_lat_2 = set(lat_list_2)
    set_lon_1 = set(lon_list_1)
    set_lon_2 = set(lon_list_2)

    # plot 1
    plt.scatter(lon_list_1, lat_list_1, color="green", s=4, label="base")
    plt.scatter(lon_list_2, lat_list_2, color="red", s=1, label="split")
    plt.legend()
    plt.savefig("plot1.png", dpi=400)
    plt.show(block=False)

    # plot 2
    plt.scatter(lon_list_1, lat_list_1, color="green", s=5, label="base")
    plt.scatter(lon_list_2, lat_list_2, color="red", s=2, label="split")
    points_1 = set(zip(lon_list_1, lat_list_1))
    points_2 = set(zip(lon_list_2, lat_list_2))
    intersect_points = set.intersection(points_1, points_2)
    lat_list_common = []
    lon_list_common = []

    for lon, lat in intersect_points:
        lat_list_common.append(lat)
        lon_list_common.append(lon)

    plt.scatter(
        lon_list_common, lat_list_common, marker="s", color="yellow", s=10, label="matching", edgecolors=(0, 0, 0, 1)
    )
    plt.legend()
    plt.savefig("plot2.png", dpi=400)
    plt.show(block=False)

    # plot 3
    for R in [2, 3, 4, 5]:
        plt.clf()
        lat_list_1, lon_list_1 = extract_lat_list_lon_list(base_N_osm_bbox_list)
        lat_list_2, lon_list_2 = extract_lat_list_lon_list(splitted_N_osm_bbox_list)
        lat_list_1 = [round(float(i), R) for i in lat_list_1]
        lat_list_2 = [round(float(i), R) for i in lat_list_2]
        lon_list_1 = [round(float(i), R) for i in lon_list_1]
        lon_list_2 = [round(float(i), R) for i in lon_list_2]
        plt.scatter(lon_list_1, lat_list_1, color="green", s=5, label="base")
        plt.scatter(lon_list_2, lat_list_2, color="red", s=2, label="split")
        points_1 = set(zip(lon_list_1, lat_list_1))
        points_2 = set(zip(lon_list_2, lat_list_2))
        intersect_points = set.intersection(points_1, points_2)
        lat_list_common = []
        lon_list_common = []

        for lon, lat in intersect_points:
            lat_list_common.append(lat)
            lon_list_common.append(lon)

        plt.scatter(
            lon_list_common,
            lat_list_common,
            marker="s",
            color="yellow",
            s=10,
            label="matching",
            edgecolors=(0, 0, 0, 1),
        )
        plt.legend()
        plt.title("rounding digits: " + str(R))
        plt.savefig("rounding_" + str(R) + ".png", dpi=400)
        plt.show(block=False)

    assert (set_lat_1 in set_lat_2) and (set_lon_1 in set_lon_2)


if __name__ == "__main__":
    for N in [10]:  # , 10, 30]:
        ssubset(base_N=N)
