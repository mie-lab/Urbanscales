import os
import osmnx as ox

pickle_protocol = 5

verbose = 2

DEBUG = False
DEBUG_TRUNCATE = False
MASTER_VISUALISE_EACH_STEP = False

CONGESTION_TYPE = "RECURRENT"
assert CONGESTION_TYPE in ["RECURRENT", "NON-RECURRENT"]

BASE_FOLDER_local = "/Users/nishant/Documents/GitHub/WCS"
BASE_FOLDER_server = "/home/niskumar/WCS"
delete_results_folder = True
cur_dir = os.getcwd()
RUNNING_ON_LOCAL = True
RUNNING_ON_SERVER = False

if cur_dir.split("/")[1] == "home":
    BASE_FOLDER = BASE_FOLDER_server
    RUNNING_ON_LOCAL = False
    RUNNING_ON_SERVER = True

elif cur_dir.split("/")[1] == "Users":
    BASE_FOLDER = BASE_FOLDER_local
    RUNNING_ON_LOCAL = True
    RUNNING_ON_SERVER = False

home_folder_path = BASE_FOLDER

osmnx_cache_folder = os.path.join(BASE_FOLDER, "cache_osmnx")
log_file = os.path.join(BASE_FOLDER, "log-main.txt")
LOGGING_ENABLED = False

ox.settings.use_cache = True
ox.settings.cache_folder = osmnx_cache_folder

####################################
######  DELETE FILES CONFIG ########
####################################
rn_delete_existing_pickled_objects = False
scl_delete_existing_pickle_objects = False
sd_delete_existing_pickle_objects = False
ps_delete_existing_pickle_objects = False
td_delete_existing_pickle_objects = False


#####################################
##############  PLOTS  ################
#####################################
ppl_smallest_sample = -1
ppl_use_all = True
if ppl_use_all:
    assert ppl_smallest_sample == -1
ppl_parallel_overall = 1
ppl_plot_FI = True
ppl_CV_splits = 7
ppl_plot_corr = False
ppl_hist = False
ppl_hist_bins = 10
ppl_scaling_for_EDA = (
    2  # 0: None; 1: Divide by Max; 2: StandardScaler(); 3: Divide by max; followed by StandardScaler()
)
assert ppl_scaling_for_EDA in [0, 1, 2, 3]
ppl_list_of_baselines = [] # ["Lasso()", "LinearRegression()", "Ridge()"], "LinearRegression()"

# ppl_feature_importance_via_coefficients = False
# if ppl_feature_importance_via_coefficients:
#     assert model in ["LR", "RIDGE", "LASSO"]

ppl_feature_importance_via_NL_models = False
# if ppl_feature_importance_via_NL_models:
#     assert model in ["RFR", "GBM"]

ppl_list_of_NL_models = ["RandomForestRegressor()"]  # , "GradientBoostingRegressor()" #  ["RandomForestRegressor()"] # , "GradientBoostingRegressor()"]

# assert (ppl_feature_importance_via_NL_models != ppl_feature_importance_via_coefficients) or (
#     ppl_feature_importance_via_coefficients == False and ppl_feature_importance_via_NL_models == False
# )

ppl_list_of_correlations = ["pearson", "spearman"]  # , "kendall", "spearman"]

####################################
######   ROAD NETWORK Class   ######
####################################

# format: city,location, N, E, S, W
if RUNNING_ON_SERVER:
    rn_city_wise_bboxes = {
        "Singapore": [1.51316, 104.135278, 1.130361, 103.566667],
        "Zurich": [47.434666, 8.625441, 47.32022, 8.448006],
        "Mumbai": [19.270177, 72.979731, 18.893957, 72.776333],
        "Auckland": [-36.681247, 174.925937, -36.965932, 174.63532],
        "Istanbul": [41.671, 29.9581, 40.7289, 27.9714],
        "MexicoCity": [19.592757, -98.940303, 19.048237, -99.364924],
        "Bogota": [4.837015, -73.996423, 4.4604, -74.223689],
        "NewYorkCity": [40.916178, -73.700181, 40.477399, -74.25909],
        "Capetown": [-34.462, 18.1107, -33.3852, 19.0926],
        "London": [51.28676, -0.510375, 51.691874, 0.334015],
        # "Tokyo": [35.0721, 139.1704, 35.9707, 140.5547],  # @Tokyo removed because no data present in here-api at the time of our study
        # "TokyoCore": [35.0721, 139.1704, 35.9707, 140.5547],
    }
    single_city = "Zurich"
    rn_city_wise_bboxes = {single_city : rn_city_wise_bboxes[single_city]}


elif RUNNING_ON_LOCAL:
    rn_city_wise_bboxes = {
        "Singapore": [1.51316, 104.135278, 1.130361, 103.566667],
        "Zurich": [47.434666, 8.625441, 47.32022, 8.448006],
        "Mumbai": [19.270177, 72.979731, 18.893957, 72.776333],
        "Auckland": [-36.681247, 174.925937, -36.965932, 174.63532],
        "Istanbul": [41.671, 29.9581, 40.7289, 27.9714],
        "MexicoCity": [19.592757, -98.940303, 19.048237, -99.364924],
        "Bogota": [4.837015, -73.996423, 4.4604, -74.223689],
        "NewYorkCity": [40.916178, -73.700181, 40.477399, -74.25909],
        "Capetown": [-34.462, 18.1107, -33.3852, 19.0926],
        "London": [51.28676, -0.510375, 51.691874, 0.334015],
        # "Tokyo": [35.0721, 139.1704, 35.9707, 140.5547],  # @Tokyo removed because no data present in here-api at the time of our study
        # "TokyoCore": [35.0721, 139.1704, 35.9707, 140.5547],
    }
    single_city = "Zurich"
    rn_city_wise_bboxes = {single_city : rn_city_wise_bboxes[single_city]}

if RUNNING_ON_SERVER:
    # rn_city_wise_tz_code can be commented out since we don't need to recompute the speed_data_object everytime.
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
        # "Tokyo": "Asia/Tokyo"
        # "TokyoCore": [35.0721, 139.1704, 35.9707, 140.5547],
    }
elif RUNNING_ON_LOCAL:
    # rn_city_wise_tz_code can be commented out since we don't need to recompute the speed_data_object everytime.
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
    }

rn_master_list_of_cities = list(rn_city_wise_bboxes.keys())
rn_basemap_zoom_level = 13
rn_do_not_filter_list = []  # ["Zurich"]
rn_do_not_filter = True
if rn_do_not_filter:
    assert len(rn_do_not_filter_list) == 0

rn_plotting_enabled = False
rn_plotting_for_truncated_graphs = False
rn_truncate_method = "GPD_CUSTOM"
assert rn_truncate_method in ["GPD_DUMMY_NODES_SMART_TRUNC", "OSMNX_RETAIN_EDGE", "GPD_CUSTOM"]
rn_prefix_geojson_files = "gdam_410_"
rn_postfix_geojson_files = ".geojson"
rn_post_fix_road_network_object_file = "_road_network_object_square.pkl"
rn_base_map_filename = "_base_osm_truncated.png"
rn_compute_full_city_features = False
rn_add_edge_speed_and_tt = True
rn_no_stats_marker = "Empty"
rn_plot_basemap = False

rn_percentage_of_city_area = 100
if rn_percentage_of_city_area != 100:
    assert rn_post_fix_road_network_object_file == "_road_network_object_small.pkl"

if single_city == "Istanbul":
    rn_square_from_city_centre = 75  # 15 implies 15X15 sq.km.
else:
    rn_square_from_city_centre = 50 #   # 15 implies 15X15 sq.km.

if rn_square_from_city_centre != -1:
    assert rn_percentage_of_city_area == 100  # we cannot have two filtering techniques
    # basically it is not needed
    assert rn_post_fix_road_network_object_file == "_road_network_object_square.pkl"
rn_simplify = False

print ("Loaded config file: Prefix for name of road network object pickle file!")
####################################
#########   Scale Class   ##########
####################################
if BASE_FOLDER == BASE_FOLDER_server:
    scl_n_jobs_parallel = 45
else:
    scl_n_jobs_parallel = 5
scl_temp_file_counter = True

# if rn_truncate_method == "GDF_CUSTOM":
#     assert scl_n_jobs_parallel == 1

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

# test_small
if RUNNING_ON_LOCAL:
    if single_city == "Istanbul":
        scl_list_of_seeds = [75] # [37, 75, 150] #, 50]  # , 50, 25] # , 25, 50] # , 50, 100] # list(range(50, 121, 40)) # [10, 25, 30, 45, 50, 65, 70, 85, 90, 105]  # list(range(5, 6, 1))  # list(range(5, 50, 5)) + list(range(50, 300, 10))
    else:
        scl_list_of_seeds = [50] # [25, 50, 100]
elif RUNNING_ON_SERVER:
    if single_city == "Istanbul":
        scl_list_of_seeds = [75] # [37, 75, 150] #, 50]  # , 50, 25] # , 25, 50] # , 50, 100] # list(range(50, 121, 40)) # [10, 25, 30, 45, 50, 65, 70, 85, 90, 105]  # list(range(5, 6, 1))  # list(range(5, 50, 5)) + list(range(50, 300, 10))
    else:
        scl_list_of_seeds = [50] # [25, 50, 100]
# forward
# scl_list_of_seeds = list(range(5, 350, 10))

# backward
# scl_list_of_seeds = list(range(345, 120, -10))

scl_error_percentage_tolerance = 1
scl_basemap_zoom_level = 19

####################################
######## Tile Class configs ########
####################################
tls_garbage_Test_Speed = False   # tls_garbage_Test_Speed set to True for debugging, otherwise MUST BE FALSE;
                                 # if set to True, will return all zeros; Zeros are used to ensure that a
                                 # wrongly set config is easy to spot
tls_betweenness_features = True
tls_number_of_lanes = True
tls_add_edge_speed_and_tt = False
tls_add_metered_intersections = True

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
sd_temporal_combination_method = "mean"
assert sd_temporal_combination_method in ["mean", "max", "variance"]
sd_start_datetime_str = "2022-09-02T00:00:01"
sd_end_datetime_str = "2022-09-29T23:59:59"
sd_total_number_of_days = 30 # additional config item for sanity check
sd_total_number_of_data_points_for_each_segment = 30 * 24 * (60/sd_raw_speed_data_gran)


####################################
######   PreProcess Speed   ########
####################################
ps_spatial_combination_method = "length_weighted_mean"
assert ps_spatial_combination_method in ["mean", "max", "length_weighted_mean", "variance"]
if RUNNING_ON_LOCAL:
    ps_tod_list = [
        # [9, 10],
        # [6, 7, 8, 9, 10],  # peak hour morning
        # [16, 17, 18, 19, 20],  # peak hour morning
        list(range(1, 25))
    ]
elif RUNNING_ON_SERVER:
    ps_tod_list = [
        # [9, 10],
        # [6, 7, 8, 9, 10],  # peak hour morning
        # [16, 17, 18, 19, 20],  # peak hour morning
        list(range(1, 25))
    ]
assert isinstance(ps_tod_list, list)
ps_set_all_speed_zero = False


####################################
##### TRAIN DATA Class configs #####
####################################
td_tod_list = ps_tod_list
td_standard_scaler = True
td_min_max_scaler = False
td_plot_raw_variance_before_scaling = True
td_viz_y_hist = True

td_reuse_Graph_features = True

## maybe we should simply remove this
td_drop_feature_lists = [
    "self_loop_proportion",
    "streets_per_node_count_1",
    "streets_per_node_count_4",
    "street_length_avg",
    "street_segment_count",
    "streets_per_node_count_2",
    "mean_lanes",
]
td_drop_collinear_features = True
td_drop_collinear_features = True

shift_tile_marker = 2
network_folder = CONGESTION_TYPE + "network_tmean_smean_" +str(rn_square_from_city_centre)+ "x" +str(rn_square_from_city_centre)+ "_shifting_" + str(shift_tile_marker)
warnings_folder = "warnings"
results_folder = CONGESTION_TYPE + "results_network_tmean_smean_"+str(rn_square_from_city_centre)+ "x" +str(rn_square_from_city_centre)+ "_shifting_" # "results_50x50_max_" + ("full" if ppl_smallest_sample == -1 else str(ppl_smallest_sample)) + "_data" + "-fi-max-max"


temp_folder_for_robust_pickle_files = os.path.join(BASE_FOLDER, "pickle_slack_temp")
if not os.path.exists(temp_folder_for_robust_pickle_files):
    os.mkdir(temp_folder_for_robust_pickle_files)

# To ensure that we don't overwrite the network folder of max with mean or vice-versa
# When we use mean, we use mean in the network folder and
# When we use max, we use max in the network folder
# assert ps_spatial_combination_method == sd_temporal_combination_method
# assert sd_temporal_combination_method in network_folder