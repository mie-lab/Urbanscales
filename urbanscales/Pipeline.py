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

os.system("rm -rf " + config.BASE_FOLDER + config.results_folder)
os.system("mkdir " + config.BASE_FOLDER + config.results_folder)

print ("Current working directory: ", os.getcwd())
os.chdir('../')
print ("Current working directory: ", os.getcwd())

os.system("python urbanscales/io/road_network.py")
os.system("python urbanscales/io/speed_data.py")
os.system("python urbanscales/preprocessing/prep_network.py")
os.system("python urbanscales/preprocessing/prep_speed.py")
os.system("python urbanscales/preprocessing/train_data.py")
os.system("python urbanscales/modelling/LR.py")
