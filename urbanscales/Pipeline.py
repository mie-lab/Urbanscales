import csv
import os
import shutil
import sys
import subprocess

from urbanscales.modelling.ML_Pipeline import Pipeline
from urbanscales.process_results.process_feature_importance import process_FI_file

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from urbanscales.io.road_network import RoadNetwork
import config
import urbanscales
from urbanscales.io.road_network import RoadNetwork
from urbanscales.io.speed_data import SpeedData
from urbanscales.modelling.LR import LR
from urbanscales.preprocessing.prep_network import Scale
from urbanscales.preprocessing.prep_speed import ScaleJF
from urbanscales.preprocessing.train_data import TrainDataVectors

# Instead of os.system for rm and mkdir, use shutil and os library respectively
# shutil.rmtree(os.path.join(config.BASE_FOLDER, config.results_folder))
# os.makedirs(os.path.join(config.BASE_FOLDER, config.results_folder))

# Removed the chdir since it's not a good idea to change working directory within script
# Use absolute paths instead

print ("python urbanscales/io/road_network.py")
RoadNetwork.generate_road_nw_object_for_all_cities()

print ("python urbanscales/io/speed_data.py")
SpeedData.preprocess_speed_data_for_all_cities()

print ("python urbanscales/preprocessing/prep_network.py")
Scale.generate_scales_for_all_cities()

print ("python urbanscales/preprocessing/prep_speed.py")
sys.path.append(config.home_folder_path)
ScaleJF.connect_speed_and_nw_data_for_all_cities()

print ("python urbanscales/preprocessing/train_data.py")
os.chdir(config.home_folder_path)
TrainDataVectors.compute_training_data_for_all_cities()

print ("python urbanscales/modelling/ML_Pipeline.py")
if config.delete_results_folder and os.path.exists(os.path.join(config.BASE_FOLDER, config.results_folder)):
    shutil.rmtree(os.path.join(config.BASE_FOLDER, config.results_folder))
if not os.path.exists(os.path.join(config.BASE_FOLDER, config.results_folder)):
    os.mkdir(os.path.join(config.BASE_FOLDER, config.results_folder))

with open(os.path.join(config.BASE_FOLDER, config.results_folder, "feature_importance.csv"), "w") as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(
        ["cityname", "scale", "tod", "marker", "plot_counter"] + ["feature" + str(i) for i in range(1, 17)])
Pipeline.compute_scores_for_all_cities()

print ("python urbanscales/process_results/process_feature_importance.py")
process_FI_file()