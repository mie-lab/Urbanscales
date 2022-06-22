intermediate_files_path = "/Users/nishant/Documents/GitHub/WCS/intermediate_files/"
outputfolder = "/Users/nishant/Documents/GitHub/WCS/output_folder/"
temp_files = "/Users/nishant/Documents/GitHub/WCS/temp_files/"

# incident file name
combined_incidents_file = "combined_incidents_sample.csv"

custom_filter = None  # '["highway"~"motorway|motorway_link|primary"]'
# , "trunk","trunk_link", "motorway_link","primary","secondary"]
# custom_filter=["motorway", "motorway_link","motorway_junction","highway"],
# '["highway"~"motorway|motorway_link|primary"]'
#'["highway"~"motorway"]'


num_threads = 7
hierarchies = 2
best_fit_hierarchy = 2

base_for_merge = 5

use_route_path_curved = False
plotting_enabled = False

bbox_split_for_merge_optimisation = 8
reorder_dict_for_better_viz = True

merge_param_a = 0.75
merge_param_b = 0.5


# steps_combined.py
base_list = [2, 3]


# urban_merge.py file
thresh_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
single_threshold = 0.5


# compute_GOF_before_and_after
pca_components = 1
