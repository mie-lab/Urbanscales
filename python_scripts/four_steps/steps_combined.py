# First, a uniform segmentation of the urban space into spatial grids is done and eight graph-based features are extracted for each grid.
from python_scripts.network_to_elementary.elf_to_clusters import osm_tiles_states_to_vectors
from python_scripts.network_to_elementary.process_incidents import create_bbox_to_CCT
from python_scripts.network_to_elementary.tiles_to_elementary import step_1_osm_tiles_to_features
import pickle
import sys

sys.path.insert(0, "/Users/nishant/Documents/GitHub/WCS/python_scripts/network_to_elementary")
# step_1_osm_tiles_to_features( read_G_from_pickle=True, read_osm_tiles_stats_from_pickle=False, n_threads=7, N=50, plotting_enabled=True)

# step 2
def step_2(N, folder_path):

    dict_bbox_to_CCT = create_bbox_to_CCT(
        csv_file_name="combined_incidents_13_days.csv",
        read_from_pickle=True,
        N=N,
        folder_path=folder_path,
    )

    with open(folder_path + "osm_tiles_stats_dict" + str(N) + ".pickle", "rb") as f:
        osm_tiles_stats_dict = pickle.load(f)

    keys_bbox_list, vals_vector_array = osm_tiles_states_to_vectors(osm_tiles_stats_dict)

    X = vals_vector_array

    Y = []
    for key in keys_bbox_list:
        Y.append(dict_bbox_to_CCT[key])

    debug_wait = True


if __name__ == "__main__":
    step_2(30, "/Users/nishant/Documents/GitHub/WCS/python_scripts/network_to_elementary/")
