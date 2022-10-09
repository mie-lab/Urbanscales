pickle_protocol = 5
CV_splits = 5

verbose = 2

network_folder = "network"
warnings_folder = "warnings"
results_folder = "results"

master_delete_all = True  # (one of [True, False, -1])

####################################
######   ROAD NETWORK Class   ######
####################################

# format: city,location, N, E, S, W
rn_city_wise_bboxes = {
    "Singapore": [1.51316, 104.135278, 1.130361, 103.566667],
    "Zurich": [47.434666, 8.625441, 47.32022, 8.448006],
    # "Mumbai": [19.270177, 72.979731, 18.893957, 72.776333],
    # "Auckland": [-35.6984, 175.9032, -37.3645, 173.8963],
    # "Istanbul": [41.671, 29.9581, 40.7289, 27.9714],
    # "MexicoCity": [19.592757, -98.940303, 19.048237, -99.364924],
    # "Bogota": [4.837015, -73.996423, 4.4604, -74.223689],
    "NewYorkCity": [40.916178, -73.700181, 40.477399, -74.25909],
    # "Capetown": [-34.462, 18.1107, -33.3852, 19.0926],
    # "London": [51.28676, -0.510375, 51.691874, 0.334015],
    # "Tokyo": [35.0721, 139.1704, 35.9707, 140.5547],  # @Tokyo removed because no data present in here-api at the time of our study
    # "TokyoCore": [35.0721, 139.1704, 35.9707, 140.5547],
}
rn_master_list_of_cities = list(rn_city_wise_bboxes.keys())

rn_do_not_filter_list = []  # ["Zurich"]
rn_do_not_filter = True
if rn_do_not_filter:
    assert len(rn_do_not_filter_list) == 0

rn_plotting_enabled = False
rn_prefix_geojson_files = "gdam_410_"
rn_postfix_geojson_files = ".geojson"
rn_post_fix_road_network_object_file = "_road_network_object_square.pkl"
rn_base_map_filename = "_base_osm_truncated.png"
rn_compute_full_city_features = False
rn_add_edge_speed_and_tt = True
rn_delete_existing_pickled_objects = False

rn_percentage_of_city_area = 100
if rn_percentage_of_city_area != 100:
    assert rn_post_fix_road_network_object_file == "_road_network_object_extra_small.pkl"

rn_square_from_city_centre = 15  # 10 implies 10X10 sq.km.
if rn_square_from_city_centre != -1:
    assert rn_percentage_of_city_area == 100  # we cannot have two filtering techniques
    # basically it is not needed
    assert rn_post_fix_road_network_object_file == "_road_network_object_square.pkl"


####################################
#########   Scale Class   ##########
####################################
scl_n_jobs_parallel = 5
scl_master_list_of_cities = rn_master_list_of_cities
scl_list_of_depths = [1]
scl_list_of_seeds = [5, 10, 15, 20, 25]

scl_error_percentage_tolerance = 0.2
scl_delete_existing_pickle_objects = False


####################################
######## Tile Class configs ########
####################################
tls_betweenness_features = True
tls_number_of_lanes = True
tls_add_edge_speed_and_tt = rn_add_edge_speed_and_tt
tls_missing_lanes_default_value = 2


####################################
######   Speed Data Class   ########
####################################
sd_base_folder_path = "speed_data"
sd_seg_file_path_within_city = "segments.csv"
sd_jf_file_path_within_city = "jf.csv"
sd_raw_speed_data_gran = 10
sd_target_speed_data_gran = 60
sd_temporal_combination_method = "mean"
assert sd_temporal_combination_method in ["mean", "max"]
sd_delete_existing_pickle_objects = False


####################################
######   PreProcess Speed   ########
####################################
ps_spatial_combination_method = "mean"
assert ps_spatial_combination_method in ["mean", "max"]
ps_tod_list = [8]  # list(range(24))
assert isinstance(ps_tod_list, list)
ps_delete_existing_pickle_objects = False


####################################
##### TRAIN DATA Class configs #####
####################################
td_delete_existing_pickle_objects = False
td_tod_list = ps_tod_list
td_standard_scaler = True

# speed and tt code not working; so we set them to 1;
# @TO-DO: Need to populate the function
td_dummy_speed_and_tt = True


if master_delete_all != -1:
    td_delete_existing_pickle_object = (
        sps_delete_existing_pickle_object
    ) = (
        ssd_delete_existing_pickle_objects
    ) = scl_delete_existing_pickle_objects = rn_delete_existing_pickled_objects = master_delete_all


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


num_threads = 1
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
