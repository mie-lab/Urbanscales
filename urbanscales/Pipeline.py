import os
import sys
import subprocess

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


def run_command(command, message):
    result = subprocess.call(command, shell=True)
    if result == 0:
        print(f"\n Complete: {message}\n")
    else:
        print(f"\n Failed: {message}\n")


os.chdir("../")
cityname = ["Singapore", "Zurich", "Mumbai", "Auckland","Istanbul","MexicoCity","Bogota", "NewYorkCity", "Capetown","London"]
counter = 5


# The simpler solution of one line as shown below does not work on Ubuntu; so we now have this fancy 3-step solution
# os.system('sed -i \'\' "s/single_city = .*/single_city = \\"' + cityname[counter] +'\\"/" config.py')

backup_extension = '.bak'
os.system('sed -i' + backup_extension + ' "s/single_city = .*/single_city = \\"' + cityname[counter] + '\\"/" config.py')
os.system('rm config.py' + backup_extension)
run_command("python urbanscales/io/road_network.py", "python urbanscales/io/road_network.py")

backup_extension = '.bak'
os.system('sed -i' + backup_extension + ' "s/single_city = .*/single_city = \\"' + cityname[counter] + '\\"/" config.py')
os.system('rm config.py' + backup_extension)
run_command("python urbanscales/io/speed_data.py", "python urbanscales/io/speed_data.py")

backup_extension = '.bak'
os.system('sed -i' + backup_extension + ' "s/single_city = .*/single_city = \\"' + cityname[counter] + '\\"/" config.py')
os.system('rm config.py' + backup_extension)
run_command("python urbanscales/preprocessing/prep_network.py", "python urbanscales/preprocessing/prep_network.py")

backup_extension = '.bak'
os.system('sed -i' + backup_extension + ' "s/single_city = .*/single_city = \\"' + cityname[counter] + '\\"/" config.py')
os.system('rm config.py' + backup_extension)
run_command("python urbanscales/preprocessing/prep_speed.py", "python urbanscales/preprocessing/prep_speed.py")

backup_extension = '.bak'
os.system('sed -i' + backup_extension + ' "s/single_city = .*/single_city = \\"' + cityname[counter] + '\\"/" config.py')
os.system('rm config.py' + backup_extension)
run_command("python urbanscales/preprocessing/train_data.py", "python urbanscales/preprocessing/train_data.py")

backup_extension = '.bak'
os.system('sed -i' + backup_extension + ' "s/single_city = .*/single_city = \\"' + cityname[counter] + '\\"/" config.py')
os.system('rm config.py' + backup_extension)
run_command("python urbanscales/modelling/SHAP_analysis.py", "python urbanscales/modelling/SHAP_analysis.py")

# run_command("python urbanscales/modelling/ML_Pipeline.py", "python urbanscales/modelling/ML_Pipeline.py")
# run_command("python urbanscales/process_results/process_feature_importance.py", "python urbanscales/process_results/process_feature_importance.py")
