import os
import sys
import subprocess

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

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

def run_command(command, message):
    result = subprocess.call(command, shell=True)
    if result == 0:
        print(f"\n Complete: {message}\n")
    else:
        print(f"\n Failed: {message}\n")


os.chdir("../")
run_command("python urbanscales/io/road_network.py", "python urbanscales/io/road_network.py")
run_command("python urbanscales/io/speed_data.py", "python urbanscales/io/speed_data.py")
run_command("python urbanscales/preprocessing/prep_network.py", "python urbanscales/preprocessing/prep_network.py")
run_command("python urbanscales/preprocessing/prep_speed.py", "python urbanscales/preprocessing/prep_speed.py")
run_command("python urbanscales/preprocessing/train_data.py", "python urbanscales/preprocessing/train_data.py")
run_command("python urbanscales/modelling/SHAP_analysis.py", "python urbanscales/modelling/SHAP_analysis.py")
# run_command("python urbanscales/modelling/ML_Pipeline.py", "python urbanscales/modelling/ML_Pipeline.py")
# run_command("python urbanscales/process_results/process_feature_importance.py", "python urbanscales/process_results/process_feature_importance.py")
