import os
import shutil
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import config
import urbanscales
from urbanscales.io.road_network import RoadNetwork

from urbanscales.io.speed_data import SpeedData
from urbanscales.modelling.LR import LR
from urbanscales.preprocessing.prep_network import Scale
from urbanscales.preprocessing.prep_speed import ScaleJF
from urbanscales.preprocessing.train_data import TrainDataVectors

# os.system("rm -rf " + config.BASE_FOLDER + config.results_folder)
# os.system("mkdir " + config.BASE_FOLDER + config.results_folder)
#
# print("Current working directory: ", os.getcwd())
# os.chdir("../")
# print("Current working directory: ", os.getcwd())
#
# os.system("python urbanscales/io/road_network.py")
# print("\n Complete: python urbanscales/io/road_network.py\n")
#
# os.system("python urbanscales/io/speed_data.py")
# print("\n Complete: python urbanscales/io/speed_data.py\n")
#
# os.system("python urbanscales/preprocessing/prep_network.py")
# print("\n Complete: python urbanscales/preprocessing/prep_network.py\n")
#
# os.system("python urbanscales/preprocessing/prep_speed.py")
# print("\n Complete: python urbanscales/preprocessing/prep_speed.py\n")
#
# os.system("python urbanscales/preprocessing/train_data.py")
# print("\n Complete: python urbanscales/preprocessing/train_data.py\n")
#
# # os.system("python urbanscales/modelling/ML_Pipeline.py")
# # print("\n Complete: python urbanscales/modelling/ML_Pipeline.py\n")

os.system("python urbanscales/process_results/process_feature_importance.py")
print("\n Complete: python urbanscales/process_results/process_feature_importance.py\n")

