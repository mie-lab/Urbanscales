import os

pickle_protocol = 5


verbose = 2
debug_ = True


BASE_FOLDER_local = "/Users/nishant/Documents/GitHub/WCS"
BASE_FOLDER_server = "/home/niskumar/WCS"

cur_dir = os.getcwd()
if cur_dir.split("/")[1] == "home":
    BASE_FOLDER = BASE_FOLDER_server
elif cur_dir.split("/")[1] == "Users":
    BASE_FOLDER = BASE_FOLDER_local

home_folder_path = BASE_FOLDER

master_delete_all = -1  # (one of [True, False, -1])
# -1 implies this master_config_is_not_being_used


####################################
######  DELETE FILES CONFIG ########
####################################
rn_delete_existing_pickled_objects = False
scl_delete_existing_pickle_objects = False
sd_delete_existing_pickle_objects = False
ps_delete_existing_pickle_objects = False
td_delete_existing_pickle_objects = True


#####################################
##############  PLOTS  ################
#####################################
ppl_smallest_sample = -1
ppl_use_all = True
if ppl_use_all:
    assert ppl_smallest_sample == -1

ppl_plot_FI = True
ppl_CV_splits = 7
ppl_plot_corr = True
ppl_hist = False
ppl_hist_bins = 10
ppl_scaling_for_EDA = (
    2  # 0: None; 1: Divide by Max; 2: StandardScaler(); 3: Divide by max; followed by StandardScaler()
)
assert ppl_scaling_for_EDA in [0, 1, 2, 3]
ppl_list_of_baselines = ["Lasso()", "LinearRegression()", "Ridge()"]

# ppl_feature_importance_via_coefficients = False
# if ppl_feature_importance_via_coefficients:
#     assert model in ["LR", "RIDGE", "LASSO"]

ppl_feature_importance_via_NL_models = False
# if ppl_feature_importance_via_NL_models:
#     assert model in ["RFR", "GBM"]

ppl_list_of_NL_models = ["RandomForestRegressor()", "GradientBoostingRegressor()"]

# assert (ppl_feature_importance_via_NL_models != ppl_feature_importance_via_coefficients) or (
#     ppl_feature_importance_via_coefficients == False and ppl_feature_importance_via_NL_models == False
# )

ppl_list_of_correlations = ["pearson"]  # , "kendall", "spearman"]

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
    # "Tokyo": [35.0721, 139.1704, 35.9707, 140.5547],  # @Tokyo removed because no data present in here-api at the time of our study
    # "TokyoCore": [35.0721, 139.1704, 35.9707, 140.5547],
}
rn_city_wise_tz_code = {
    "Singapore": "Asia/Singapore",
    "Zurich": "Europe/Zurich",
    "Mumbai": "Asia/Kolkata",
    "Auckland": "Pacific/Auckland",
    "Istanbul": "Europe/Istanbul",
    "MexicoCity": "America/Mexico_City",
    "Bogota": "America/Bogota",
    "NewYorkCity": "America/New_York",
    "Capetown": "Africa/Johannesburg",
    "London": "Europe/London",
    "Tokyo": "Asia/Tokyo"
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

rn_percentage_of_city_area = 100
if rn_percentage_of_city_area != 100:
    assert rn_post_fix_road_network_object_file == "_road_network_object_small.pkl"

rn_square_from_city_centre = 40  # 15 implies 15X15 sq.km.
if rn_square_from_city_centre != -1:
    assert rn_percentage_of_city_area == 100  # we cannot have two filtering techniques
    # basically it is not needed
    assert rn_post_fix_road_network_object_file == "_road_network_object_square.pkl"

####################################
#########   Scale Class   ##########
####################################
if BASE_FOLDER == BASE_FOLDER_server:
    scl_n_jobs_parallel = 15
else:
    scl_n_jobs_parallel = 7
scl_master_list_of_cities = rn_master_list_of_cities
scl_list_of_depths = [1]
# scl_list_of_seeds = [
#     15,
#     30,
#     50,
#     60,
#     80,
#     110,
#     130,
#     # 150,
#     # 170
# ]  # 40, 45, 50, 55, 60, 65, 70, 80, 85, 90, 95, 100, 120]
scl_list_of_seeds = list(range(15, 350, 10))

scl_error_percentage_tolerance = 0.2


####################################
######## Tile Class configs ########
####################################
tls_betweenness_features = True
tls_number_of_lanes = True
tls_add_edge_speed_and_tt = False

if tls_add_edge_speed_and_tt == True:
    # the data must be present in the road network to begin with
    assert rn_add_edge_speed_and_tt == True

tls_missing_lanes_default_value = 2


####################################
######   Speed Data Class   ########
####################################
sd_base_folder_path = "speed_data"
sd_seg_file_path_within_city = "segments.csv"
sd_jf_file_path_within_city = "jf.csv"
sd_raw_speed_data_gran = 10
sd_target_speed_data_gran = 60
sd_temporal_combination_method = "max"
assert sd_temporal_combination_method in ["mean", "max"]
sd_start_datetime_str = "2022-07-31T18:04:05"
sd_end_datetime_str = "2022-08-03T18:20:05"


####################################
######   PreProcess Speed   ########
####################################
ps_spatial_combination_method = "max"
assert ps_spatial_combination_method in ["mean", "max"]
ps_tod_list = [6]  # list(range(24))
assert isinstance(ps_tod_list, list)


####################################
##### TRAIN DATA Class configs #####
####################################
td_tod_list = ps_tod_list
td_standard_scaler = True
td_min_max_scaler = False
td_plot_raw_variance_before_scaling = True
td_viz_y_hist = True

## maybe we should simply remove this
td_drop_feature_lists = [
    "streets-per-node-proportions0",
    "streets-per-node-proportions1",
    "streets-per-node-proportions3",
    "streets-per-node-proportions4",
    "streets-per-node-proportions5",
    "edge-length-avg",
    "street-segment-count",
    "streets-per-node-counts-0",
]
td_drop_collinear_features = True

if master_delete_all != -1:
    td_delete_existing_pickle_object = (
        sps_delete_existing_pickle_object
    ) = (
        ssd_delete_existing_pickle_objects
    ) = scl_delete_existing_pickle_objects = rn_delete_existing_pickled_objects = master_delete_all


network_folder = "network"  # -tmax-smax"
warnings_folder = "warnings"
results_folder = "results_" + ("full" if ppl_smallest_sample == -1 else str(ppl_smallest_sample)) + "_data"


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
