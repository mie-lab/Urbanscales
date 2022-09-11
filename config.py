pickle_protocol = 5


####################################
######   ROAD NETWORK Class   ######
####################################
# rn_master_list_of_cities = [
#     "Singapore",
#     "Zurich",
#     "Mumbai",
#     "Auckland",
#     "Istanbul",
#     "Mexico City",
#     "Bogota",
#     "New York City",
#     "Cape Town",
#     "London",
# ]
rn_master_list_of_cities = [
    "Singapore",
]
rn_plotting_enabled = False
rn_compute_graph_features = False


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
