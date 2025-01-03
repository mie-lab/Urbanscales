# polybox password: "data-7-cities"

import os
import osmnx as ox

pickle_protocol = 5

verbose = 2

DEBUG = False
DEBUG_TRUNCATE = False
MASTER_VISUALISE_EACH_STEP = False
MASTER_VISUALISE_EACH_STEP_INSIDE_RN_class = False
MASTER_VISUALISE_EACH_STEP_INSIDE_PrepNetwork_class = True
MASTER_VISUALISE_EACH_STEP_INSIDE_ShapAnalysisScript = False
MASTER_VISUALISE_FEATURE_DIST_INSIDE_ShapAnalysisScript = True

CONGESTION_TYPE = "RECURRENT"
assert CONGESTION_TYPE in ["RECURRENT", "NON-RECURRENT", "NON-RECURRENT-MMM"]

BASE_FOLDER_local = "/Users/nishant/Documents/GitHub/WCS"  # "/Users/nishant/Downloads/WCS_from_server_local_copy_withy_new_plots" # "/Users/nishant/Documents/GitHub/WCS"
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
        "Auckland": [-36.681247, 174.925937, -36.965932, 174.63532],
        "Bogota": [4.837015, -73.996423, 4.4604, -74.223689],
        "Capetown": [-34.462, 18.1107, -33.3852, 19.0926],
        "Istanbul": [41.671, 29.9581, 40.7289, 27.9714],
        "MexicoCity": [19.592757, -98.940303, 19.048237, -99.364924],
        "Mumbai": [19.270177, 72.979731, 18.893957, 72.776333],
        "NewYorkCity": [40.916178, -73.700181, 40.477399, -74.25909],
    }
    single_city = "Auckland"
    rn_city_wise_bboxes = {single_city : rn_city_wise_bboxes[single_city]}


elif RUNNING_ON_LOCAL:
    rn_city_wise_bboxes = {
        "Auckland": [-36.681247, 174.925937, -36.965932, 174.63532],
        "Bogota": [4.837015, -73.996423, 4.4604, -74.223689],
        "Capetown": [-34.462, 18.1107, -33.3852, 19.0926],
        "Istanbul": [41.671, 29.9581, 40.7289, 27.9714],
        "MexicoCity": [19.592757, -98.940303, 19.048237, -99.364924],
        "Mumbai": [19.270177, 72.979731, 18.893957, 72.776333],
        "NewYorkCity": [40.916178, -73.700181, 40.477399, -74.25909],

    }
    single_city = "Istanbul"
    rn_city_wise_bboxes = {single_city : rn_city_wise_bboxes[single_city]}

if RUNNING_ON_SERVER:
    # rn_city_wise_tz_code can be commented out since we don't need to recompute the speed_data_object everytime.
    rn_city_wise_tz_code = {
        "Auckland": "Pacific/Auckland",
        "Bogota": "America/Bogota",
        "Capetown": "Africa/Johannesburg",
        "Istanbul": "Europe/Istanbul",
        "NewYorkCity": "America/New_York",
        "MexicoCity": "America/Mexico_City",
        "Mumbai": "Asia/Kolkata",
    }
elif RUNNING_ON_LOCAL:
    # rn_city_wise_tz_code can be commented out since we don't need to recompute the speed_data_object everytime.
    rn_city_wise_tz_code = {
        "Auckland": "Pacific/Auckland",
        "Bogota": "America/Bogota",
        "Capetown": "Africa/Johannesburg",
        "Istanbul": "Europe/Istanbul",
        "NewYorkCity": "America/New_York",
        "MexicoCity": "America/Mexico_City",
        "Mumbai": "Asia/Kolkata",
    }

rn_master_list_of_cities = list(rn_city_wise_bboxes.keys())
list.sort(rn_master_list_of_cities)

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
# elif single_city == "London":
#     rn_square_from_city_centre = 25  # 15 implies 15X15 sq.km.
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
running_extended_scales = 0 # 0 implies single scale (1 sq.km), 1 implies 3 scales (1 sqkm, 4 swkm and 0.25 sqkm.) and 2 implies all from 20 to 100

if RUNNING_ON_LOCAL:
    if single_city == "Istanbul":
        if running_extended_scales == 2:
            scl_list_of_seeds = [30, 45, 60, 90, 105, 120, 135, 37, 75, 150]
        elif running_extended_scales == 1:
            scl_list_of_seeds = [37, 75, 150] # [37, 75, 150] # [37, 75, 150] #, 50]  # , 50, 25] # , 25, 50] # , 50, 100] # list(range(50, 121, 40)) # [10, 25, 30, 45, 50, 65, 70, 85, 90, 105]  # list(range(5, 6, 1))  # list(range(5, 50, 5)) + list(range(50, 300, 10))
        elif running_extended_scales == 0:
            scl_list_of_seeds = [75]
    else:
        if running_extended_scales == 2:
            scl_list_of_seeds = [20, 30, 40, 60, 70, 80, 90, 25, 50, 100]
        elif running_extended_scales == 1:
            scl_list_of_seeds = [25, 50, 100] # [25, 50, 100] # [25, 50, 100]
        elif running_extended_scales == 0:
            scl_list_of_seeds = [50]
elif RUNNING_ON_SERVER:
    if single_city == "Istanbul":
        if running_extended_scales == 2:
            scl_list_of_seeds = [30, 45, 60, 90, 105, 120, 135, 37, 75, 150]
        elif running_extended_scales == 1:
            scl_list_of_seeds = [37, 75, 150] # [37, 75, 150] # [37, 75, 150] #, 50]  # , 50, 25] # , 25, 50] # , 50, 100] # list(range(50, 121, 40)) # [10, 25, 30, 45, 50, 65, 70, 85, 90, 105]  # list(range(5, 6, 1))  # list(range(5, 50, 5)) + list(range(50, 300, 10))
        elif running_extended_scales == 0:
            scl_list_of_seeds = [75]
    else:
        if running_extended_scales == 2:
            scl_list_of_seeds = [20, 30, 40, 60, 70, 80, 90, 25, 50, 100]
        elif running_extended_scales == 1:
            scl_list_of_seeds = [25, 50, 100] # [25, 50, 100] # [25, 50, 100]
        elif running_extended_scales == 0:
            scl_list_of_seeds = [50]


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
tls_add_max_speed = True
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
td_SIMPLIFY_GRAPH = True
td_min_max_scaler = False
td_plot_raw_variance_before_scaling = True
td_viz_y_hist = True

td_reuse_Graph_features = True

## drop features showing high collinearity, some of them were removed initially
td_drop_feature_lists = [
    # "self_loop_proportion",
    # "streets_per_node_count_1",
    # "streets_per_node_count_4",
    # "street_length_avg",
    # "street_segment_count",
    # "streets_per_node_count_2",
    # "mean_lanes",
]

shift_tile_marker = 3
SHAP_mode_spatial_CV = "horizontal"
assert SHAP_mode_spatial_CV in ["vertical", "horizontal", "grid"]

SHAP_additive_regression_model = False # Poor GoF; no longer used
SHAP_sort_features_alphabetical_For_heatmaps = False
FAST_GEN_PDPs_for_multiple_runs = True  # True only when fast PDPs are needed for a large number of scenarios; this must be False for normal runs

network_folder = CONGESTION_TYPE + "R3network_tmean_smean_" +str(rn_square_from_city_centre)+ "x" +str(rn_square_from_city_centre)+ "_shifting_" + str(shift_tile_marker)
warnings_folder = "warnings"
results_folder = CONGESTION_TYPE + "R3results_network_tmean_smean_"+str(rn_square_from_city_centre)+ "x" +str(rn_square_from_city_centre)+ "_shifting_" + str(shift_tile_marker)  # "results_50x50_max_" + ("full" if ppl_smallest_sample == -1 else str(ppl_smallest_sample)) + "_data" + "-fi-max-max"

temp_folder_for_robust_pickle_files = os.path.join(BASE_FOLDER, "pickle_slack_temp")
if not os.path.exists(temp_folder_for_robust_pickle_files):
    os.mkdir(temp_folder_for_robust_pickle_files)

SHAP_ScalingOfInputVector = True
SHAP_use_all_features_including_highly_correlated = False
SHAP_random_search_CV = False
if SHAP_random_search_CV:
    SHAP_HPT_num_iters = 50
    SHAP_Grid_search_CV = False
else:
    SHAP_Grid_search_CV = True

SHAP_values_disabled = False # True only when we are interested in capturing the just the GoF for a large number of scenarios
SHAP_disable_all_plots_for_server = True
SHAP_GBM_ENABLED = False
SHAP_generate_beeswarm = True

# To ensure that we don't overwrite the network folder of max with mean or vice-versa
# When we use mean, we use mean in the network folder and
# When we use max, we use max in the network folder
# assert ps_spatial_combination_method == sd_temporal_combination_method
# assert sd_temporal_combination_method in network_folder
