import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# Define the file names
file_names = [
    'GOFtod_[6, 7, 8, 9, 10]_scale_25.csv',
    'GOFtod_[6, 7, 8, 9, 10]_scale_30.csv',
    'GOFtod_[6, 7, 8, 9, 10]_scale_40.csv',
    'GOFtod_[6, 7, 8, 9, 10]_scale_50.csv',
    'GOFtod_[6, 7, 8, 9, 10]_scale_60.csv',
    'GOFtod_[6, 7, 8, 9, 10]_scale_70.csv',
    'GOFtod_[6, 7, 8, 9, 10]_scale_80.csv',
    'GOFtod_[6, 7, 8, 9, 10]_scale_90.csv',
    'GOFtod_[6, 7, 8, 9, 10]_scale_100.csv',
    # Add all your file names here
]


# file_names = [
#     'GOFtod_[6, 7, 8, 9, 10]_scale_30.csv',
#     'GOFtod_[6, 7, 8, 9, 10]_scale_40.csv',
#     'GOFtod_[6, 7, 8, 9, 10]_scale_50.csv',
#     'GOFtod_[6, 7, 8, 9, 10]_scale_60.csv',
#     'GOFtod_[6, 7, 8, 9, 10]_scale_70.csv',
# ]
# Initialize a dictionary to store GoF values for each model and scale
gof_dict = {}

# Read each file and extract GoF values and scale
for file_name in file_names:
    df = pd.read_csv(file_name)
    # Extract scale from file name
    scale = int(re.search(r'scale_(\d+)', file_name).group(1))

    # Iterate through each model in the file
    for index, row in df.iterrows():
        model = row['model']  # Replace 'cityname' with the correct column name for your models
        gof_value = row['GoF_explained_Variance']

        # Add the GoF value to the dictionary
        if model == "Lasso":
            continue
        if model not in gof_dict:
            gof_dict[model] = {}
        gof_dict[model][scale] = gof_value

# Plotting
for model, scale_gof_dict in gof_dict.items():
    # Sort the dictionary by scale for each model
    sorted_scale_gof_dict = dict(sorted(scale_gof_dict.items()))
    plt.plot(sorted_scale_gof_dict.keys(), sorted_scale_gof_dict.values(), marker='o', label=model)

plt.xlabel('Scale')
plt.ylabel('GoF Explained Variance')
# plt.title('GoF vs Scale NYC - Recurrent')
plt.title('GoF vs Scale MexicoCity - Recurrent')
plt.legend()
plt.ylim(0, 1)
# plt.savefig("NYC-Recurrent-Scale-vs-GoF.png")
plt.savefig("MexicoCity-Recurrent-Scale-vs-GoF.png")
# plt.grid(True)
plt.show()
