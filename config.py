pickle_protocol = 5


network_folder = "network"

####################################
######   ROAD NETWORK Class   ######
####################################

# format: city,location, N, E, S, W
rn_city_wise_bboxes = {
    "Singapore": [1.51316, 104.135278, 1.130361, 103.566667],
    "Zurich": [47.434666, 8.625441, 47.32022, 8.448006],
    "Mumbai": [19.270177, 72.979731, 18.893957, 72.776333],
    "Auckland": [-35.6984, 175.9032, -37.3645, 173.8963],
    "Istanbul": [41.671, 29.9581, 40.7289, 27.9714],
    "MexicoCity": [19.592757, -98.940303, 19.048237, -99.364924],
    "Bogota": [4.837015, -73.996423, 4.4604, -74.223689],
    "NewYorkCity": [40.916178, -73.700181, 40.477399, -74.25909],
    "Capetown": [-34.462, 18.1107, -33.3852, 19.0926],
    "London": [51.28676, -0.510375, 51.691874, 0.334015],
    "Tokyo": [35.0721, 139.1704, 35.9707, 140.5547],
    "TokyoCore":[35.0721, 139.1704, 35.9707, 140.5547],
}
rn_master_list_of_cities = list(rn_city_wise_bboxes.keys())

rn_do_not_filter_list = [] # ["Zurich"]
rn_plotting_enabled = True
rn_prefix_geojson_files = "gdam_410_"
rn_postfix_geojson_files = ".geojson"
rn_post_fix_road_network_object_file = "_road_network_object_small.pkl"
rn_base_map_filename = "_base_osm_truncated.png"
rn_delete_existing_pickled_objects = False



####################################
#########   Scale Class   ##########
####################################
scl_n_jobs_parallel = 5


####################################
######   Speed Data Class   ########
####################################
sd_base_folder_path = "speed_data"
sd_seg_file_path_within_city = "segments.csv"
sd_jf_file_path_within_city = "jf.csv"
sd_temporal_combination_method = "mean"
assert sd_temporal_combination_method in ["mean", "max"]


####################################
######   PreProcess Speed   ########
####################################
ps_spatial_combination_method = "mean"
assert ps_spatial_combination_method in ["mean", "max"]


intermediate_files_path = "/Users/nishant/Documents/GitHub/WCS/intermediate_files/"
outputfolder = "/Users/nishant/Documents/GitHub/WCS/output_folder/"
temp_files = "/Users/nishant/Documents/GitHub/WCS/temp_files/"
multiple_cities_graphs = (
    "/Users/nishant/Documents/GitHub/WCS/intermediate_files/multiple_cities/raw_graphs_from_OSM_pickles"
)


# incident file configs
combined_incidents_file = "combined_incidents_45_days.csv"


# speed data configs
data_folder = "/Users/nishant/Documents/GitHub/WCS/data/here_speed_demo"
road_linestring = "here_road_linestring_01032022.csv"
var_jf = "here_speed_sgtime_jf.csv"
var_ci = "here_speed_sgtime_cn.csv"
plotting_enabled_speed_data_preprocess = True


custom_filter = None  # '["highway"~"motorway|motorway_link|primary"]'
# , "trunk","trunk_link", "motorway_link","primary","secondary"]
# custom_filter=["motorway", "motorway_link","motorway_junction","highway"],
# '["highway"~"motorway|motorway_link|primary"]'
#'["highway"~"motorway"]'


city_list = [
    "Auckland",
    "Bogota",
    "Cape Town",
    "Istanbul",
    "London",
    "Mexico City",
    "Mumbai",
    "New York City",
    "Singapore",
    "Zurich",
]


num_threads = 7
hierarchies = 5
best_fit_hierarchy = 5
base_for_merge = 6


# graph features
stats_type = "basic_stats"


use_route_path_curved = False
plotting_enabled = False

bbox_split_for_merge_optimisation = 32
reorder_dict_for_better_viz = True

merge_param_a = 0.5
merge_param_b = 0.5

# steps_combined.py
base_list = [5]  # list(range(2, 11))


# urban_merge.py file
thresh_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
single_threshold = 0.9
run_multiple_thresholds_in_parallel = False
method_for_single_statistic = "sum"

# compute_GOF_before_and_after
pca_components = 3
