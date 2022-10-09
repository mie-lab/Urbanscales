import os
import shutil

import config
import urbanscales
from urbanscales.io.road_network import RoadNetwork

from urbanscales.io.speed_data import SpeedData
from urbanscales.modelling.LR import LR
from urbanscales.preprocessing.prep_network import Scale
from urbanscales.preprocessing.prep_speed import ScaleJF
from urbanscales.preprocessing.train_data import TrainDataVectors

os.system("rm -rf " + config.results_folder)
os.system("mkdir " + config.results_folder)


if config.rn_delete_existing_pickled_objects:
    RoadNetwork.generate_road_nw_object_for_all_cities()

if config.scl_delete_existing_pickle_objects:
    os.system("python urbanscales/preprocessing/prep_network.py")
    # Scale.generate_scales_for_all_cities()

if config.sd_delete_existing_pickle_objects:
    os.system("python urbanscales/io/speed_data.py")
    # SpeedData.preprocess_speed_data_for_all_cities()
    os.system("python urbanscales/preprocessing/prep_speed.py")
    # ScaleJF.connect_speed_and_nw_data_for_all_cities()

if config.td_delete_existing_pickle_objects:
    os.system("python urbanscales/preprocessing/train_data.py")
    # TrainDataVectors.compute_training_data_for_all_cities()

os.system("python urbanscales/modelling/LR.py")
# LR.compute_scores_for_all_cities()
