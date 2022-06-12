import pickle
from shapely import geometry
import numpy as np


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
    with open("osm_tiles_stats_dict" + str(scale) + ".pickle", "rb") as handle:
        bbox_list = get_bbox(pickle.load(handle))

    bbox_lol = []
    for lat1, lon1, lat2, lon2 in bbox_list:
        # revert for homogeiniety in gdal
        bbox_lol.append([[lon1, lat1], [lon2, lat2]])
    return bbox_list


def create_hierarchy_dict(base_level, number_of_hierarchy):
    # @yatao's comment: read bbox_file, a set of bbox file in multi-hierarchies
    # dict_bbox = {'hierarchy_1':[[]], 'hierarchy_2':[[]], 'hierarchy_3':[[]]}

    dict_bbox = {}

    for i in range(number_of_hierarchy):
        scale = base_level * (2 ** i)
        dict_bbox["hierarchy_" + str(i + 1)] = get_bbox_as_list_of_list(scale)
    return dict_bbox


def get_isl_and_seeds_bboxes_for_best_fit_hierarchy(bbox_list):
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

    for lat1, lon1, lat2, lon2 in bbox_list:
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
    dict_seeds["island_2"] = island_1[int(len(island_2) * np.random.rand())]

    return dict_islands


if __name__ == "__main__":
    base_level = 5
    dict_bbox = create_hierarchy_dict(base_level, 4)
    with open("dict_bbox_" + str(base_level) + "_.pickle", "wb") as f:
        pickle.dump(dict_bbox, f, protocol=4)

    best_fit_hierarchy = 2
    dict_islands = get_isl_and_seeds_bboxes_for_best_fit_hierarchy(dict_bbox["hierarchy_" + str(best_fit_hierarchy)])
    with open("dict_islands_" + str(best_fit_hierarchy) + "_.pickle", "wb") as f:
        pickle.dump(dict_bbox, f, protocol=4)

    do_nothing = True
