import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from smartprint import smartprint as sprint

import pandas as pd
import glob
import os


# ON-RECURRENT-MMM Bogota 70 RF 0.014390624005569454 MSE
# NON-RECURRENT-MMM Bogota 70 GBM 0.027941862297532427 MSE
# NON-RECURRENT-MMM Bogota 70 LLR 0.028374573538975316 MSE
# NON-RECURRENT-MMM Bogota 70 RLR 0.016694790135212887 MSE
# NON-RECURRENT-MMM Bogota 70 LR 0.420176034740506 R2
# NON-RECURRENT-MMM Bogota 70 RF 0.49813574292000595 R2
# NON-RECURRENT-MMM Bogota 70 GBM 0.009735688983875921 R2
# NON-RECURRENT-MMM Bogota 70 LLR -0.005931713933676245 R2
# NON-RECURRENT-MMM Bogota 70 RLR 0.4188192163378041 R2


import pandas as pd
import os
import glob

# Initialize an empty DataFrame to store the combined data
combined_data = pd.DataFrame()


# Helper function to process files
def process_files(file_pattern, congestion_type):
    files = glob.glob(file_pattern)
    temp_data = pd.DataFrame()

    for file in files:
        # Create a temporary filtered CSV file
        temp_file = f"temp_file_{congestion_type}.csv"
        os.system(f"grep \"{congestion_type}\" {file} | grep \"R2\" | grep -v GBM > {temp_file}")

        # Read the data from the temporary CSV file
        data = pd.read_csv(temp_file, header=None, delimiter=r"\s+")
        data.columns = ['CongestionType', 'CityName', 'Scale', 'Model', 'Gof', 'Metric']

        # Calculate the adjusted scale and tiles
        data['Scale'] = data['Scale'].astype(int)
        data['n~#tiles=(nXn)'] = data['Scale'].astype(int)
        data['Scale'] = round(
            (75 / data['Scale'] if 'Istanbul' in data['CityName'].unique() else 50 / data['Scale']) ** 2, 2)

        # Specify the R2 type
        data['R2 Type'] = 'NRC' if 'NON-RECURRENT' in congestion_type else 'RC'

        data['Gof'] = pd.to_numeric(data['Gof'], errors='coerce')
        data['Gof'] = data['Gof'].round(3)

        data = data[(data['n~#tiles=(nXn)'] == 150) | (data['n~#tiles=(nXn)'] == 100)]

        # Append to the temporary DataFrame
        temp_data = pd.concat([temp_data, data[['CityName', 'n~#tiles=(nXn)', 'Scale', 'Model', 'Gof', 'R2 Type']]])

    return temp_data


# Process Non-Recurrent and Recurrent files
nrc_data = process_files('NONRECURRENTcity?.csv', 'NON-RECURRENT-MMM')
rc_data = process_files('RECURRENTcity?.csv', 'RECURRENT')

# Combine the data
combined_data = pd.concat([nrc_data, rc_data])

# Pivot the DataFrame to have separate columns for NRC and RC R2 values
final_data = combined_data.pivot_table(index=['CityName', 'n~#tiles=(nXn)', 'Scale', 'Model'], columns='R2 Type',
                                       values='Gof', aggfunc='first').reset_index()

# Save the combined data to a new CSV file
final_data.to_csv('final_combined_city_data.csv', index=False)

# Convert the DataFrame to a LaTeX table
from tabulate import tabulate

latex_table = tabulate(final_data, headers='keys', tablefmt='latex_booktabs', showindex=False)

# Print or save the LaTeX table
print(latex_table)



# Optionally, write the LaTeX table to a file
with open('table_output_GoF.tex', 'w') as f:
    f.write(latex_table)






def set_global_rc_params():
    plt.rcParams.update({
        'font.size': 20,          # General font size
        'axes.titlesize': 16,     # Title font size
        'axes.labelsize': 14,     # Axis label font size
        'xtick.labelsize': 12,    # X-tick label font size
        'ytick.labelsize': 12,    # Y-tick label font size
        'legend.fontsize': 12,    # Legend font size
        'figure.titlesize': 16,   # Figure title font size
        # 'cbar.labelsize': 14,     # Colorbar label font size
        # 'cbar.tick.size': 12,     # Colorbar tick label size
    })
set_global_rc_params();

NANCOUNTPERCITY = 5  # Set this to the desired threshold

# Load and prepare your data
def load_data(file_path):
    df = pd.read_csv(file_path, header=None, sep=" ")
    df.columns = ["congtype", "City", "Scale", "Model", "GoF", "R2"]
    df['Scale'] = df['Scale'].astype(str)
    df['Scale'] = df['Scale'].apply(lambda x: x.zfill(3))
    return df


# Process files
files = ['NONRECURRENTFigure1city' + str(x) + '.csv' for x in range(2, 9)]  # City indices from 2 to 8

all_data = pd.concat([load_data(f) for f in files])


# Calculate tile area with specific condition for Istanbul
def calculate_tile_area(row):
    base = 75 if row['City'] == 'Istanbul' else 50
    return (base / float(str(row['Scale']))) ** 2


all_data['TileArea'] = all_data.apply(calculate_tile_area, axis=1)

# Color-blind friendly palette for seven cities
colors = {
    "Mumbai": "#E69F00",  # orange
    "Auckland": "#56B4E9",  # sky blue
    "Istanbul": "#009E73",  # bluish green
    "MexicoCity": "#F0E442",  # yellow
    "Bogota": "#0072B2",  # blue
    "NewYorkCity": "#D55E00",  # vermilion
    "Capetown": "#CC79A7",  # reddish purple
}


# Plotting function for each model type with lines for each city
# Plotting function for each model type with lines for each city
goflist = []
def plot_data_by_model(df, model_types, city_list):
    for model in model_types:
        plt.figure(figsize=(7, 5))

        for city in city_list:
            subset = df[(df['City'] == city) & (df['Model'] == model)]
            subset = subset.sort_values(by='TileArea')
            if not subset.empty:
                plt.plot(subset['TileArea'], subset['GoF'], marker='o', linestyle='-', label=f"{city} ".replace("NewYorkCity", "New York City").replace("Capetown", "Cape Town").replace("MexicoCity", "Mexico City"),
                         color=colors[city], )
                sprint("NRC: ", city, round(np.mean(subset['GoF']), 2), round(np.std(subset['GoF']), 2))
                goflist.extend(subset['GoF'].tolist())
        sprint(round(np.mean(goflist), 2), round(np.std(goflist), 2))
        # plt.title(r'GoF (R$^2$) vs '+f'Tile Area for {model} Model Across Cities')
        # plt.xlabel(r'Tile Area (km$^2$)' + " " + r"coarse $\rightarrow$ fine", fontsize=25)
        plt.xlabel(r'Scale ' + " " + r"(coarse $\rightarrow$ fine)", fontsize=17)
        plt.ylabel('Goodness of Fit '+ r'($R^2$)', fontsize=17)
        plt.grid(True, alpha=0.2)
        plt.legend(ncol=2, fontsize=12.5, loc="upper right")
        plt.ylim(0, 1)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.tight_layout();
        plt.gca().invert_xaxis()
        plt.savefig("nonrecurrent_Fi_1_inverted.pdf",dpi=300)

        # plt.gca().invert_xaxis()
        # plt.savefig("nonrecurrent_Fi_1.pdf",dpi=300)
        plt.show()



# Example usage
city_list = [ "Auckland", "Bogota", "Capetown", "Istanbul", "MexicoCity", "Mumbai", "NewYorkCity"]
model_types = ['RF']  # , 'LR', 'RLR', 'GBM']

plot_data_by_model(all_data, model_types, city_list)
non_recurrent_gof_dict = {}
for i in range(all_data.shape[0]):
    if all_data.iloc[i].Model == "RF":
        non_recurrent_gof_dict[all_data.iloc[i].City.lower(), all_data.iloc[i].Scale] = all_data.iloc[i].GoF

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu


def load_data(file_path):
    df = pd.read_csv(file_path, delim_whitespace=True, header=None)
    df.columns = ["marker", "City-Scale-tod", "feature", "absshap"]

    def adjust_feature_name(feature_name):
        # if 'streets_per_node_count_1' == feature_name:
        #     return "streets_per_node_count_0"
        # elif 'streets_per_node_proportion_1' == feature_name:
        #     return "streets_per_node_count_1"
        # elif 'streets_per_node_count_2' == feature_name:
        #     return "streets_per_node_count_2"
        # elif 'streets_per_node_proportion_2' == feature_name:
        #     return "streets_per_node_count_3"
        # elif 'streets_per_node_count_3' == feature_name:
        #     return "streets_per_node_count_4"
        # elif 'streets_per_node_proportion_3' == feature_name:
        #     return "streets_per_node_count_5"
        return feature_name

    def adjust_feature_name_plotting(feature_name):
        feature_dict = {
            'n': '#nodes',
            'm': '#edges',
            'k_avg': 'avg degree',
            'streets_per_node_avg': 'SPN-avg',
            'circuity_avg': 'avg circuity',
            'metered_count': '#traffic lights',
            'non_metered_count': '#free turns',
            'total_crossings': '#total crossings',
            'betweenness': 'local centrality',
            'mean_lanes': 'avg lanes',
            'streets_per_node_count_1': '#SPN-1',
            'streets_per_node_count_2': '#SPN-2',
            'streets_per_node_count_3': '#SPN-3',
            'streets_per_node_count_4': '#SPN-4',
            'streets_per_node_count_5': '#SPN-5',
            'global_betweenness': 'global centrality'
        }
        return feature_dict.get(feature_name, feature_name)  # default to the original name if not found

    # Applying the renaming function to the 'feature' column
    df['feature'] = df['feature'].apply(adjust_feature_name)
    df['feature'] = df['feature'].apply(adjust_feature_name_plotting)


    # indices_to_drop = df[df['feature'] == 'global centrality'].index
    # df = df.drop(index=indices_to_drop)

    split_columns = df['City-Scale-tod'].str.split('-', expand=True)
    df['City'] = split_columns[3]
    df['Scale'] = split_columns[4].astype(int)
    df['Scale'] = split_columns[4].astype(str)
    df['Scale'] = df['Scale'].apply(lambda x: x.zfill(3))
    df['City-Scale'] = df['City'] + '-' + df['Scale'].astype(str)
    return df


files = ['NONRECURRENTFigure2city' + str(x) + '.csv' for x in range(2, 9)]  # City indices from 2 to 8
all_data = pd.concat([load_data(f) for f in files])
heatmap_data = all_data.pivot_table(index='feature', columns='City-Scale', values='absshap', aggfunc='mean')
heatmap_data_backup = heatmap_data.copy() # pd.DataFrame(heatmap_data)
# Choose the thresholding method: 'otsu', 'mean_std', 'quantile', 'top_n'
plt.figure(figsize=(15, 8))
sns.heatmap(heatmap_data, annot=False, cmap="viridis", cbar_kws={'label': 'Feature Importance (mean |SHAP|)'},
            yticklabels=True, xticklabels=True)
# plt.title("Raw heatmap without filtering")
plt.ylabel("Feature")
plt.xlabel("Scale")
plt.tight_layout();
plt.savefig("nonrecurrent_Fi_2a_orig.pdf", dpi=300)
plt.show()

aaa = []
for scalestringified in ["020", "025", "030", "040", "050", "060", "070", "080", "090", "100"]:
    aaa.extend (heatmap_data["capetown-" + scalestringified].tolist())
print ("min, max(aaa) :",  max(aaa), min(aaa))

plt.figure(figsize=(4, 5))
plt.hist(heatmap_data.to_numpy().flatten().tolist(), bins=10)
plt.xticks(rotation=90)
plt.title("Histogram of FI values (NRC)")
plt.grid(alpha=0.3)
# plt.savefig("NRC_histogram.pdf", dpi=300)
# plt.show()
plt.clf()


# Reshape the DataFrame from pivot table format to a long format
heatmap_data_long = heatmap_data.reset_index().melt(id_vars='feature', var_name='City-Scale', value_name='FI')
heatmap_data_long[['City', 'Scale']] = heatmap_data_long['City-Scale'].str.split('-', expand=True)
heatmap_data_long = heatmap_data_long[['City', 'Scale', 'feature', 'FI']]
heatmap_data_long.columns = ['city', 'scale', 'featurename', 'FI']
heatmap_data_long['scale'] = heatmap_data_long['scale'].astype(str).str.zfill(3)
heatmap_data_long.dropna(subset=['FI'], inplace=True)
heatmap_data_long.to_csv('feature_importance_NRC_Shifting_3.csv', index=False)



if "heatmapplotunfiltered" == "heatmapplotunfiltered":
    tick_positions = np.concatenate([np.arange(0, 8, 4), np.array([9]), np.arange(14, heatmap_data.columns.size, 10)])
    tick_labels = ["6.25 $km^2$", "4.0 $km^2$", "2.78 $km^2$", "1.56 $km^2$", "1.0 $km^2$", "0.69 $km^2$",
                   "0.51 $km^2$", "0.39 $km^2$", "0.31 $km^2$", "0.25 $km^2$"]
    tick_labels = tick_labels + tick_labels + tick_labels + tick_labels + tick_labels + tick_labels + tick_labels
    # Repeating the sequence for as many times as needed (or slice it to the number of ticks)
    tick_labels = np.array(tick_labels)[tick_positions]
    plt.figure(figsize=(15, 8))
    ax = sns.heatmap(heatmap_data, annot=False, cmap="viridis", cbar_kws={'label': 'Feature Importance (mean |SHAP|)'},
                     yticklabels=True, xticklabels=tick_positions)  # Pass only the positions where labels should be shown



    # Assuming heatmap_data is already loaded and processed
    # Let's compute the mid-points for the city labels
    city_positions = {}
    for col in heatmap_data.columns:
        city = col.split('-')[0]
        if city not in city_positions:
            city_positions[city] = []
        city_positions[city].append(heatmap_data.columns.get_loc(col))

    # Calculate mid-points for placing city labels
    city_labels = {}
    for city, positions in city_positions.items():
        city_labels[city] = np.mean(positions)

    plt.figure(figsize=(15, 8))
    ax = sns.heatmap(heatmap_data, annot=False, cmap="viridis", cbar_kws={'label': r'${Feature~Importance}_j = \mathbb{E} \left[ |\phi_j| \right]$'},
                     yticklabels=True, xticklabels=tick_positions)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=24)  # Adjust the tick font size as needed
    cbar.set_label(r'${Feature~Importance}_j = \mathbb{E} \left[ |\phi_j| \right]$',
                   size=24)  # Adjust the label font size as needed

    # Set custom x-tick labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=20)  # Rotate labels for better visibility
    ax.set_yticklabels(heatmap_data.index, rotation=0, fontsize=20)  # Rotate labels for better visibility

    # Annotate with city names
    for city, pos in city_labels.items():
        ax.text(pos, -0.5, city.title().replace("Newyorkcity", "New York City").replace("Mexicocity", "Mexico City").replace("Capetown", "Cape Town"), va='center', ha='center', color='black', rotation=0, fontsize=12)

    # Color-blind friendly palette for seven cities
    colors = {
        "Mumbai".lower(): "#E69F00",  # orange
        "Auckland".lower(): "#56B4E9",  # sky blue
        "Istanbul".lower(): "#009E73",  # bluish green
        "MexicoCity".lower(): "#F0E442",  # yellow
        "Bogota".lower(): "#0072B2",  # blue
        "NewYorkCity".lower(): "#D55E00",  # vermilion
        "Capetown".lower(): "#CC79A7",  # reddish purple
    }

    for city, pos in city_labels.items():
        ax.text(pos, -0.5, city.title().replace("Newyorkcity", "New York City").replace("Mexicocity", "Mexico City").replace("Capetown", "Cape Town"),
                va='center', ha='center', color='black', rotation=0, fontsize=12,
                bbox=dict(facecolor=colors[city.replace(" ", "").lower()], alpha=0.4, edgecolor='none', boxstyle='round,pad=0.5'))


    # Add vertical lines to separate groups of cities every 10 columns
    for i in range(10, len(heatmap_data.columns), 10):
        ax.axvline(x=i, color='white', linestyle='--', lw=1)  # Change color and style if needed
    for i in range(1, heatmap_data.shape[0]):
        ax.axhline(y=i, color='white', linestyle='--', lw=1)

    # Set custom x-tick labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90)  # Rotate labels for better visibility
    plt.ylabel("Feature", fontsize=25)
    plt.xlabel("Scale", fontsize=25)
    plt.tight_layout()
    # plt.savefig("nonrecurrent_Fi_2a.pdf", dpi=300)
    # plt.show()
    plt.clf()

    for citynum in range(7):
        sprint (heatmap_data.iloc[:,citynum*10:(citynum+1)*10].columns)
        sprint (np.mean(heatmap_data.iloc[:,citynum*10:(citynum+1)*10].to_numpy().flatten().tolist()),
               np.std(heatmap_data.iloc[:,citynum*10:(citynum+1)*10].to_numpy().flatten().tolist()))
        print ("===============================")
    for citynum in range(7):
        sprint (heatmap_data.iloc[:,citynum*10:(citynum+1)*10].columns)
        sprint (np.max(heatmap_data.iloc[:,citynum*10:(citynum+1)*10].to_numpy().flatten().tolist()))
        print ("===============================")



    # print for paper
    all_sum = heatmap_data.sum(axis=1).to_list()
    all_std = heatmap_data.std(axis=1).to_list()
    dict_FI = dict(zip(heatmap_data.index.to_list(), [(all_sum[i], all_std[i]) for i in range(len(all_sum))]))
    sprint(sorted(dict_FI.items(), key=lambda x: x[1][0])[:])



method = 'otsu'  # Change this variable to switch methods


for column in heatmap_data.columns:
    if method == 'otsu':
        thresh = 0 # threshold_otsu(np.array(heatmap_data[column].to_list()))
    elif method == 'mean_std':
        thresh = heatmap_data[column].mean() + heatmap_data[column].std()
    elif method == 'quantile':
        thresh = heatmap_data[column].quantile(0.75)  # 75th percentile
    elif method == 'top_n':
        sorted_values = np.sort(heatmap_data[column].dropna())
        if len(sorted_values) > 5:
            thresh = sorted_values[-5]  # Threshold to keep top 5 values
        else:
            thresh = sorted_values[0]  # If less than 5 values, take the smallest
    else:
        raise ValueError("Unsupported method")

    heatmap_data[column] = heatmap_data[column].apply(lambda x: x if x >= thresh else np.nan)

plt.figure(figsize=(15, 8))
sns.heatmap(heatmap_data, annot=False, cmap="viridis", cbar_kws={'label': 'Feature Importance (mean |SHAP|)'},
            yticklabels=True, xticklabels=True)
# plt.title("Heatmap with column wise Otsu thresholding")
plt.ylabel("Feature")
plt.xlabel("Scale")
plt.tight_layout();

# plt.savefig("nonrecurrent_Fi_2b_orig.pdf", dpi=300)
# plt.show()
plt.clf()




if "heatmapplotunOtsu" == "heatmapplotunOtsu":
    tick_positions = np.concatenate([np.arange(0, 8, 4), np.array([9]), np.arange(14, heatmap_data.columns.size, 10)])
    tick_labels = ["6.25 $km^2$", "4.0 $km^2$", "2.78 $km^2$", "1.56 $km^2$", "1.0 $km^2$", "0.69 $km^2$",
                   "0.51 $km^2$", "0.39 $km^2$", "0.31 $km^2$", "0.25 $km^2$"]
    tick_labels = tick_labels + tick_labels + tick_labels + tick_labels + tick_labels + tick_labels + tick_labels
    # Repeating the sequence for as many times as needed (or slice it to the number of ticks)
    tick_labels = np.array(tick_labels)[tick_positions]
    plt.figure(figsize=(15, 8))
    ax = sns.heatmap(heatmap_data, annot=False, cmap="viridis", cbar_kws={'label': 'Feature Importance (mean |SHAP|)'},
                     yticklabels=True, xticklabels=tick_positions)  # Pass only the positions where labels should be shown



    # Assuming heatmap_data is already loaded and processed
    # Let's compute the mid-points for the city labels
    city_positions = {}
    for col in heatmap_data.columns:
        city = col.split('-')[0]
        if city not in city_positions:
            city_positions[city] = []
        city_positions[city].append(heatmap_data.columns.get_loc(col))

    # Calculate mid-points for placing city labels
    city_labels = {}
    for city, positions in city_positions.items():
        city_labels[city] = np.mean(positions)

    plt.figure(figsize=(15, 8))
    ax = sns.heatmap(heatmap_data, annot=False, cmap="viridis", cbar_kws={'label': r'${Feature~Importance}_j = \mathbb{E} \left[ |\phi_j| \right]$'},
                     yticklabels=True, xticklabels=tick_positions)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=24)  # Adjust the tick font size as needed
    cbar.set_label(r'${Feature~Importance}_j = \mathbb{E} \left[ |\phi_j| \right]$',
                   size=24)  # Adjust the label font size as needed

    # Set custom x-tick labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=20)  # Rotate labels for better visibility
    ax.set_yticklabels(heatmap_data.index, rotation=0, fontsize=20)  # Rotate labels for better visibility

    # Annotate with city names
    for city, pos in city_labels.items():
        ax.text(pos, -0.5, city.title().replace("Newyorkcity", "New York City").replace("Mexicocity", "Mexico City").replace("Capetown", "Cape Town"), va='center', ha='center', color='black', rotation=0, fontsize=12)

    # Add vertical lines to separate groups of cities every 10 columns
    for i in range(10, len(heatmap_data.columns), 10):
        ax.axvline(x=i, color='white', linestyle='--', lw=1)  # Change color and style if needed
    for i in range(1, heatmap_data.shape[0]):
        ax.axhline(y=i, color='white', linestyle='--', lw=1)

    # Set custom x-tick labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90)  # Rotate labels for better visibility
    plt.ylabel("Feature", fontsize=25)
    plt.xlabel("Scale", fontsize=25)
    plt.tight_layout()
    # plt.savefig("nonrecurrent_Fi_2b.pdf", dpi=300)
    # plt.show()
    plt.clf()


backup_non_recurrent_nans = np.sign(pd.DataFrame(heatmap_data))

import pandas as pd
import numpy as np


def process_row(row, nancountpercity):
    num_cities = 7
    num_scales = 10
    result = []

    for i in range(num_cities):
        start_idx = i * num_scales
        end_idx = start_idx + num_scales
        city_values = row[start_idx:end_idx]
        nan_count = city_values.isna().sum()

        if nan_count >= nancountpercity+1:
            result.extend([1] * num_scales)
        else:
            result.extend([np.nan] * num_scales)

    return pd.Series(result, index=row.index)


def process_dataframe(df, nancountpercity):
    processed_df = df.apply(lambda row: process_row(row, nancountpercity), axis=1)
    return processed_df

backup_non_recurrent_nans = process_dataframe(backup_non_recurrent_nans, NANCOUNTPERCITY)


heatmap_data = heatmap_data_backup.where(backup_non_recurrent_nans.isna())
original_xticks = heatmap_data.columns.tolist()  # Save original x-tick labels

heatmap_data.columns = original_xticks
plt.figure(figsize=(15, 8))
sns.heatmap(heatmap_data, annot=False, cmap="viridis", cbar_kws={'label': 'Feature Importance (mean |SHAP|)'},
            yticklabels=True, xticklabels=True)
# plt.title("Heatmap with > Otsu in > 50% scales")
plt.ylabel("Feature")
plt.xlabel("Scale")
plt.tight_layout()
plt.tight_layout();
# plt.savefig("nonrecurrent_Fi_2c_orig.pdf", dpi=300)
# plt.show()
plt.clf()

if "heatmapplotunOtsuSelected50" == "heatmapplotunOtsuSelected50":
    tick_positions = np.concatenate([np.arange(0, 8, 4), np.array([9]), np.arange(14, heatmap_data.columns.size, 10)])
    tick_labels = ["6.25 $km^2$", "4.0 $km^2$", "2.78 $km^2$", "1.56 $km^2$", "1.0 $km^2$", "0.69 $km^2$",
                   "0.51 $km^2$", "0.39 $km^2$", "0.31 $km^2$", "0.25 $km^2$"]
    tick_labels = tick_labels + tick_labels + tick_labels + tick_labels + tick_labels + tick_labels + tick_labels
    # Repeating the sequence for as many times as needed (or slice it to the number of ticks)
    tick_labels = np.array(tick_labels)[tick_positions]
    plt.figure(figsize=(15, 8))
    ax = sns.heatmap(heatmap_data, annot=False, cmap="viridis", cbar_kws={'label': 'Feature Importance (mean |SHAP|)'},
                     yticklabels=True, xticklabels=tick_positions)  # Pass only the positions where labels should be shown



    # Assuming heatmap_data is already loaded and processed
    # Let's compute the mid-points for the city labels
    city_positions = {}
    for col in heatmap_data.columns:
        city = col.split('-')[0]
        if city not in city_positions:
            city_positions[city] = []
        city_positions[city].append(heatmap_data.columns.get_loc(col))

    # Calculate mid-points for placing city labels
    city_labels = {}
    for city, positions in city_positions.items():
        city_labels[city] = np.mean(positions)

    plt.figure(figsize=(15, 8))
    ax = sns.heatmap(heatmap_data, annot=False, cmap="viridis", cbar_kws={'label': r'${Feature~Importance}_j = \mathbb{E} \left[ |\phi_j| \right]$'},
                     yticklabels=True, xticklabels=tick_positions)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=24)  # Adjust the tick font size as needed
    cbar.set_label(r'${Feature~Importance}_j = \mathbb{E} \left[ |\phi_j| \right]$',
                   size=24)  # Adjust the label font size as needed

    # Set custom x-tick labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=20)  # Rotate labels for better visibility
    ax.set_yticklabels(heatmap_data.index, rotation=0, fontsize=20)  # Rotate labels for better visibility

    # Annotate with city names
    for city, pos in city_labels.items():
        ax.text(pos, -0.5, city.title().replace("Newyorkcity", "New York City").replace("Mexicocity", "Mexico City").replace("Capetown", "Cape Town"), va='center', ha='center', color='black', rotation=0, fontsize=12)

    # Add vertical lines to separate groups of cities every 10 columns
    for i in range(10, len(heatmap_data.columns), 10):
        ax.axvline(x=i, color='white', linestyle='--', lw=1)  # Change color and style if needed
    for i in range(1, heatmap_data.shape[0]):
        ax.axhline(y=i, color='white', linestyle='--', lw=1)

    # Set custom x-tick labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90)  # Rotate labels for better visibility
    plt.ylabel("Feature", fontsize=25)
    plt.xlabel("Scale", fontsize=25)
    # plt.tight_layout()
    # plt.savefig("nonrecurrent_Fi_2c.pdf", dpi=300)
    # plt.show()
    plt.clf()


################ FIGURE 2
if 1==1: # trick to allow code folding :)


    # trick to allow code folding :)

    import pandas as pd


    def create_city_feature_dict(df):
        city_feature_dict = {}

        for col in df.columns:
            city = col.split('-')[0].lower()
            if city not in city_feature_dict:
                city_feature_dict[city] = []

            for feature in df.index:
                if pd.isna(df.at[feature, col]) and feature not in city_feature_dict[city]:
                    city_feature_dict[city].append(feature)

        return city_feature_dict


    # Example usage
    # Assuming backup_non_recurrent_nans is your DataFrame
    city_feature_dict = create_city_feature_dict(backup_non_recurrent_nans)

    # Display the result
    import pprint

    pprint.pprint(city_feature_dict)

    # Create the dictionary structure
    city_feature_dict_true_values_for_scale_dependency = {}

    for col in original_xticks:
        # print(col)
        city, scale = col.split('-')
        scale = int(scale)
        if city not in city_feature_dict_true_values_for_scale_dependency:
            city_feature_dict_true_values_for_scale_dependency[city] = []

        # Identify features that are not NaN across all scales for this city
        valid_features = heatmap_data[col].notna()
        features_dict = {}

        for feature, is_valid in valid_features.items():
            if feature in city_feature_dict[city]:
                if feature not in features_dict:
                    features_dict[feature] = {}
                features_dict[feature][scale] = heatmap_data.at[feature, col]

        # Add non-empty feature dictionaries to the city's list
        city_feature_dict_true_values_for_scale_dependency[city].append(features_dict)

    import pprint

    # pprint.pprint(city_feature_dict_true_values_for_scale_dependency)
    plt.figure(figsize=(14, 8))
    # Collate data
    collated_data = {}

    for city, features in city_feature_dict_true_values_for_scale_dependency.items():
        for feature_list in features:
            for feature, scales in feature_list.items():
                if (city, feature) not in collated_data:
                    collated_data[(city, feature)] = {}
                collated_data[(city, feature)].update(scales)

    # Initialize the plot
    plt.figure(figsize=(5, 6))

    # Color-blind friendly palette for seven cities
    colors = {
        "Mumbai".lower(): "#E69F00",  # orange
        "Auckland".lower(): "#56B4E9",  # sky blue
        "Istanbul".lower(): "#009E73",  # bluish green
        "MexicoCity".lower(): "#F0E442",  # yellow
        "Bogota".lower(): "#0072B2",  # blue
        "NewYorkCity".lower(): "#D55E00",  # vermilion
        "Capetown".lower(): "#CC79A7",  # reddish purple
    }

    FI_as_timeseries = []
    keys_for_cluster_tracking = []
    # Plot the data
    for (city, feature), scales in collated_data.items():
        scales_list = sorted(scales.keys())
        values = [scales[scale] for scale in scales_list]
        if city.lower() != "istanbul":
            arealist = [(50 / x) ** 2 for x in scales_list]
            # arealist = [x * 1.5 for x in scales_list]
        else:
            arealist = [(75 / x) ** 2 for x in scales_list]
            # arealist = [x for x in scales_list]
        plt.plot(arealist, values, label=f"{city}-{feature}", color=colors[city], marker='o')
        # FI_as_timeseries.append(np.array(values) / np.array(arealist))
        FI_as_timeseries.append(values)
        keys_for_cluster_tracking.append((city.lower(), feature))
    FI_as_timeseries = np.array(FI_as_timeseries)

    # Add labels and title
    plt.xlabel(r'Tile Area (km$^2$)')
    plt.ylabel('Feature Importance')
    # plt.title('Feature Importance vs. Scale for Each City-Feature Combination')
    # plt.legend(loc='best', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout();
    # plt.savefig("nonrecurrent_Fi_3a.pdf", dpi=300)
    # plt.show()
    plt.clf()
    # Display the dictionary

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance
    from tslearn.clustering import silhouette_score as ts_silhouette_score
    from tslearn.metrics import dtw

    # Normalize the data
    # timeseries_data = TimeSeriesScalerMeanVariance().fit_transform(heatmap_data_backup.values)
    timeseries_data = TimeSeriesScalerMeanVariance().fit_transform(FI_as_timeseries)
    # timeseries_data = FI_as_timeseries

    HARDCODED_CLUSTER = 3
    if HARDCODED_CLUSTER == 0:
        # Determine the optimal number of clusters using the Elbow method and Silhouette analysis
        wcss = []
        silhouette_scores = []
        max_clusters = 10

        for k in range(2, max_clusters + 1):
            km = TimeSeriesKMeans(n_clusters=k, metric="dtw", random_state=0)
            clusters = km.fit_predict(timeseries_data)
            wcss.append(km.inertia_)  # WCSS
            silhouette_scores.append(ts_silhouette_score(timeseries_data, clusters, metric='dtw'))

        # Plot Elbow method
        # plt.figure(figsize=(14, 8))
        # plt.plot(range(2, max_clusters + 1), wcss, marker='o')
        # plt.xlabel('Number of Clusters')
        # plt.ylabel('WCSS (Within-cluster sum of squares)')
        # # plt.title('Elbow Method for Determining the Optimal Number of Clusters')
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout();
        # plt.show()

        # Plot Silhouette scores
        plt.figure(figsize=(14, 8))
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        # plt.title('Silhouette Analysis for Determining the Optimal Number of Clusters')
        plt.grid(True, alpha=0.3)
        plt.tight_layout();
        # plt.savefig("nonrecurrent_Fi_3b.pdf", dpi=300)
        # plt.show()
        plt.clf()

        # Choose the optimal number of clusters (based on the Elbow or Silhouette)
        optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because range starts from 2
        HARDCODED_CLUSTER = optimal_clusters

    # Perform K-means clustering with DTW as the distance metric and the optimal number of clusters
    km = TimeSeriesKMeans(n_clusters=HARDCODED_CLUSTER, metric="dtw", random_state=0)
    clusters = km.fit_predict(timeseries_data)

    # Calculate the representative time series for each cluster
    cluster_representatives = np.zeros((HARDCODED_CLUSTER, timeseries_data.shape[1]))
    for i in range(HARDCODED_CLUSTER):
        cluster_indices = np.where(clusters == i)[0]
        cluster_timeseries = timeseries_data[cluster_indices]
        try:
            # cluster_representatives[i] = np.median(cluster_timeseries, axis=0)
            cluster_representatives[i] = np.median(cluster_timeseries, axis=0).reshape(cluster_representatives[i].shape)
        except Exception as e:
            debug_pitstop = True
            raise e

    # Define colors for each city
    colors = {
        "Mumbai".lower(): "#E69F00",  # orange
        "Auckland".lower(): "#56B4E9",  # sky blue
        "Istanbul".lower(): "#009E73",  # bluish green
        "MexicoCity".lower(): "#F0E442",  # yellow
        "Bogota".lower(): "#0072B2",  # blue
        "NewYorkCity".lower(): "#D55E00",  # vermilion
        "Capetown".lower(): "#CC79A7",  # reddish purple
    }

    # Extract city names from labels
        # Extract city names from labels
    citylist_alphabetical = ['auckland',
                         'bogota',
                         'capetown',
                         'istanbul',
                         'mexicocity',
                         'mumbai',
                         'newyorkcity'
                             ]
    city_labels = []
    for i, citynamelowercase in enumerate(citylist_alphabetical):
        city_labels.extend([citynamelowercase] * heatmap_data.shape[0])

    ratio_of_last_to_first_element = []
    for row in cluster_representatives:
        ratio_of_last_to_first_element.append((row[-1] - row[0]))

    sorted_indices = sorted(range(len(ratio_of_last_to_first_element)), key=lambda k: ratio_of_last_to_first_element[k])
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]  # Tab Blue, Tab Green, Tab Orange
    cluster_color = {sorted_indices[i]: colors[i] for i in range(3)}

    # Plot the clustered time series
    plt.figure(figsize=(5, 6))
    for i in range(HARDCODED_CLUSTER):
        cluster_indices = np.where(clusters == i)[0]
        for idx in cluster_indices:
            try:
                city = city_labels[idx]
            except Exception as e:
                debug_pitstop = True
                raise e
            plt.plot(timeseries_data[idx].ravel(), alpha=0.1, color=cluster_color[i])
        plt.plot(cluster_representatives[i], label=f"Cluster {i + 1} (Representative)", linewidth=2, color=cluster_color[i])

    plt.xlabel('Scale')
    plt.ylabel('Feature Importance')
    # plt.title(f'Clustered Feature Importance vs. Scale (n_clusters={HARDCODED_CLUSTER})')
    # plt.legend(loc='best', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout();
    # plt.savefig("nonrecurrent_Fi_3c.pdf", dpi=300)
    # plt.show()
    plt.clf()

    # Display the number of time series in each cluster
    for i in range(HARDCODED_CLUSTER):
        print(f"Cluster {i + 1}: {len(np.where(clusters == i)[0])} time series")

    dictzip = dict(zip(keys_for_cluster_tracking, clusters.tolist()))
    inv_map = {}
    for k, v in dictzip.items():
        inv_map[v] = inv_map.get(v, []) + [k]
    print("============ ABS SHAP  ================")
    sprint (inv_map)
    print ("===================================")




    # Plot the data
    for (city, feature), scales in collated_data.items():
        scales_list = sorted(scales.keys())
        values = [scales[scale] for scale in scales_list]
        if city.lower() != "istanbul":
            arealist = [(50 / x) ** 2 for x in scales_list]
            # arealist = [x * 1.5 for x in scales_list]
        else:
            arealist = [(75 / x) ** 2 for x in scales_list]
            # arealist = [x for x in scales_list]
        plt.plot(arealist, values, label=f"{city}-{feature}", color=cluster_color[dictzip[city.lower(), feature]], marker='o')
        # FI_as_timeseries.append(np.array(values) / np.array(arealist))
    # Add labels and title
    plt.xlabel(r'Tile Area (km$^2$)')
    plt.ylabel('Feature Importance')
    # plt.title('ABSSHAP for Each City-Feature Combination')
    # plt.legend(loc='best', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout();
    # plt.savefig("nonrecurrent_Fi_3d.pdf", dpi=300)
    # plt.show()
    plt.clf()




########## FIGURE 3
if 1==1: # allow code folding
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt


    def load_data_direction(file_path):
        print(file_path)
        df = pd.read_csv(file_path, delim_whitespace=True, header=None)
        #     sprint (df.shape, len(["garbage" + str(x) for x in [0,1,2,3,4,5,6,7,8]] + ["City-Scale-tod", "garbage7", "feature",  "ratio", "num", "den"]))
        df.columns = ["garbage" + str(x) for x in [0]] + ["City-Scale-tod", "garbage7",
                                                                                      "feature",
                                                                                      "ratio", "num", "den"]


        def adjust_feature_name(feature_name):
            # if 'streets_per_node_count_1' == feature_name:
            #     return "streets_per_node_count_0"
            # elif 'streets_per_node_proportion_1' == feature_name:
            #     return "streets_per_node_count_1"
            # elif 'streets_per_node_count_2' == feature_name:
            #     return "streets_per_node_count_2"
            # elif 'streets_per_node_proportion_2' == feature_name:
            #     return "streets_per_node_count_3"
            # elif 'streets_per_node_count_3' == feature_name:
            #     return "streets_per_node_count_4"
            # elif 'streets_per_node_proportion_3' == feature_name:
            #     return "streets_per_node_count_5"
            return feature_name

        # Applying the renaming function to the 'feature' column
        def adjust_feature_name_plotting(feature_name):
            feature_dict = {
                'n': '#nodes',
                'm': '#edges',
                'k_avg': 'avg degree',
                'streets_per_node_avg': 'SPN-avg',
                'circuity_avg': 'avg circuity',
                'metered_count': '#traffic lights',
                'non_metered_count': '#free turns',
                'total_crossings': '#total crossings',
                'betweenness': 'local centrality',
                'mean_lanes': 'avg lanes',
                'streets_per_node_count_1': '#SPN-1',
                'streets_per_node_count_2': '#SPN-2',
                'streets_per_node_count_3': '#SPN-3',
                'streets_per_node_count_4': '#SPN-4',
                'streets_per_node_count_5': '#SPN-5',
                'global_betweenness': 'global centrality'
            }
            return feature_dict.get(feature_name, feature_name)  # default to the original name if not found

        # Applying the renaming function to the 'feature' column
        df['feature'] = df['feature'].apply(adjust_feature_name)
        df['feature'] = df['feature'].apply(adjust_feature_name_plotting)
        # indices_to_drop = df[df['feature'] == 'global centrality'].index
        # df = df.drop(index=indices_to_drop)

        split_columns = df['City-Scale-tod'].str.split('-', expand=True)

        df['City'] = split_columns[3].apply(lambda x: x.lower())
        df['Scale'] = split_columns[4].astype(int)
        df['Scale'] = split_columns[4].astype(str)
        df['Scale'] = df['Scale'].apply(lambda x: x.zfill(3))
        df['City-Scale'] = df['City'] + '-' + df['Scale'].astype(str)
        df["signeddenominator"] = np.sign(df["ratio"]) # * np.abs(df["num"])

        #     for i in df.index:
        #         city_scale_tuple = (df.loc[i, 'City'], df.loc[i, 'Scale'])
        # #         print (city_scale_tuple, )
        #         if city_scale_tuple in non_recurrent_gof_dict:
        #             df.loc[i, 'signeddenominator'] /= non_recurrent_gof_dict[city_scale_tuple]
        #             print (non_recurrent_gof_dict[city_scale_tuple])
        return df




    # Process files
    files = ['NONRECURRENTFigure3city' + str(x) + '.csv' for x in range(2, 9)]  # City indices from 2 to 8

    all_data = pd.concat([load_data_direction(f) for f in files])

    heatmap_data = all_data.pivot_table(index='feature', columns='City-Scale', values='signeddenominator', aggfunc='mean')
    # plt.hist(heatmap_data.to_numpy().flatten().tolist(), bins=20)
    # plt.tight_layout(); plt.show()
    plt.clf()

    heatmap_data = heatmap_data.where(backup_non_recurrent_nans.isna())
    original_xticks = heatmap_data.columns.tolist()  # Save original x-tick labels\
    # heatmap_data = heatmap_data.apply(lambda row: adjust_row_based_on_nan_count(row), axis=1)
    plt.figure(figsize=(14, 8))

    from matplotlib.colors import LinearSegmentedColormap

    colors = ["#fb8072", "#8dd3c7"]  # Crimson Red for -1, Teal Green for +1
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=2)

    # Create the heatmap
    sns.heatmap(heatmap_data, annot=False, cmap=cmap, cbar_kws={'label': 'Direction of Relationship'}, center=0,
                yticklabels=True, xticklabels=original_xticks)
    plt.tight_layout();
    plt.savefig("nonrecurrent_Fi_4a_orig.pdf", dpi=300)
    plt.show()

    if "heatmaptratioSign" == "heatmaptratioSign":
        tick_positions = np.concatenate(
            [np.arange(0, 8, 4), np.array([9]), np.arange(14, heatmap_data.columns.size, 10)])
        tick_labels = ["6.25 $km^2$", "4.0 $km^2$", "2.78 $km^2$", "1.56 $km^2$", "1.0 $km^2$", "0.69 $km^2$",
                       "0.51 $km^2$", "0.39 $km^2$", "0.31 $km^2$", "0.25 $km^2$"]
        tick_labels = tick_labels + tick_labels + tick_labels + tick_labels + tick_labels + tick_labels + tick_labels
        # Repeating the sequence for as many times as needed (or slice it to the number of ticks)
        tick_labels = np.array(tick_labels)[tick_positions]
        plt.figure(figsize=(15, 8))

        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd

        plt.figure(figsize=(15, 8))

        # Create the heatmap
        ax = sns.heatmap(heatmap_data, annot=False, cmap=cmap, cbar_kws={'label': 'Direction of Relationship'},
                         center=0,
                         yticklabels=True, xticklabels=tick_positions)

        # Assuming heatmap_data is already loaded and processed
        # Let's compute the mid-points for the city labels
        city_positions = {}
        for col in heatmap_data.columns:
            city = col.split('-')[0]
            if city not in city_positions:
                city_positions[city] = []
            city_positions[city].append(heatmap_data.columns.get_loc(col))

        # Calculate mid-points for placing city labels
        city_labels = {}
        for city, positions in city_positions.items():
            city_labels[city] = np.mean(positions)


        # Adjust the colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([-0.5, 0.5])  # Set ticks for your two values
        cbar.set_ticklabels(['Congestion\nAlleviating\nFeatures', 'Congestion\nExacerbating\n Features'])  # Label the ticks
        cbar.ax.tick_params(labelsize=24)  # Adjust the tick font size
        cbar.set_label('Direction of Relationship',
                       size=0.01)  # Adju


        # Set custom x-tick labels properly
        ax.set_xticks(tick_positions + 0.5)  # Center ticks in the middle of the cells
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=20)
        ax.set_yticklabels(heatmap_data.index, rotation=0, fontsize=20)  # Rotate labels for better visibility

        # Annotate with city names
        for city, pos in city_labels.items():
            ax.text(pos, -0.5,
                    city.title().replace("Newyorkcity", "New York City").replace("Mexicocity", "Mexico City").replace("Capetown", "Cape Town"),
                    va='center', ha='center', color='black', rotation=0, fontsize=12)

        # Color-blind friendly palette for seven cities
        colors = {
            "Mumbai".lower(): "#E69F00",  # orange
            "Auckland".lower(): "#56B4E9",  # sky blue
            "Istanbul".lower(): "#009E73",  # bluish green
            "MexicoCity".lower(): "#F0E442",  # yellow
            "Bogota".lower(): "#0072B2",  # blue
            "NewYorkCity".lower(): "#D55E00",  # vermilion
            "Capetown".lower(): "#CC79A7",  # reddish purple
        }

        for city, pos in city_labels.items():
            ax.text(pos, -0.5,
                    city.title().replace("Newyorkcity", "New York City").replace("Mexicocity", "Mexico City").replace(
                        "Capetown", "Cape Town"),
                    va='center', ha='center', color='black', rotation=0, fontsize=12,
                    bbox=dict(facecolor=colors[city.replace(" ", "").lower()], alpha=0.4, edgecolor='none',
                              boxstyle='round,pad=0.5'))



        # Add vertical lines to separate groups of cities every 10 columns
        for i in range(10, len(heatmap_data.columns), 10):
            ax.axvline(x=i, color='white', linestyle='--', lw=1)
        # Add vertical lines to separate groups of cities every 10 columns
        for i in range(1, heatmap_data.shape[0]):
            ax.axhline(y=i, color='white', linestyle='--', lw=1)

        plt.ylabel("Feature", fontsize=25)
        plt.xlabel("Scale", fontsize=25)
        plt.tight_layout()
        plt.savefig("nonrecurrent_Fi_4a.pdf", dpi=300)
        plt.show()


    # Load and prepare your data
    def load_data_direction(file_path):
        print(file_path)
        df = pd.read_csv(file_path, delim_whitespace=True, header=None)
        #     sprint (df.shape, len(["garbage" + str(x) for x in [0,1,2,3,4,5,6,7,8]] + ["City-Scale-tod", "garbage7", "feature",  "ratio", "num", "den"]))
        df.columns = ["garbage" + str(x) for x in [0]] + ["City-Scale-tod", "garbage7",
                                                                                      "feature",
                                                                                      "ratio", "num", "den"]

        def adjust_feature_name(feature_name):
            # if 'streets_per_node_count_1' == feature_name:
            #     return "streets_per_node_count_0"
            # elif 'streets_per_node_proportion_1' == feature_name:
            #     return "streets_per_node_count_1"
            # elif 'streets_per_node_count_2' == feature_name:
            #     return "streets_per_node_count_2"
            # elif 'streets_per_node_proportion_2' == feature_name:
            #     return "streets_per_node_count_3"
            # elif 'streets_per_node_count_3' == feature_name:
            #     return "streets_per_node_count_4"
            # elif 'streets_per_node_proportion_3' == feature_name:
            #     return "streets_per_node_count_5"
            return feature_name

        # Applying the renaming function to the 'feature' column
        def adjust_feature_name_plotting(feature_name):
            feature_dict = {
                'n': '#nodes',
                'm': '#edges',
                'k_avg': 'avg degree',
                'streets_per_node_avg': 'SPN-avg',
                'circuity_avg': 'avg circuity',
                'metered_count': '#traffic lights',
                'non_metered_count': '#free turns',
                'total_crossings': '#total crossings',
                'betweenness': 'local centrality',
                'mean_lanes': 'avg lanes',
                'streets_per_node_count_1': '#SPN-1',
                'streets_per_node_count_2': '#SPN-2',
                'streets_per_node_count_3': '#SPN-3',
                'streets_per_node_count_4': '#SPN-4',
                'streets_per_node_count_5': '#SPN-5',
                'global_betweenness': 'global centrality'
            }
            return feature_dict.get(feature_name, feature_name)  # default to the original name if not found

        # Applying the renaming function to the 'feature' column
        df['feature'] = df['feature'].apply(adjust_feature_name)
        df['feature'] = df['feature'].apply(adjust_feature_name_plotting)
        # indices_to_drop = df[df['feature'] == 'global centrality'].index
        # df = df.drop(index=indices_to_drop)

        split_columns = df['City-Scale-tod'].str.split('-', expand=True)

        df['City'] = split_columns[3].apply(lambda x: x.lower())
        df['Scale'] = split_columns[4].astype(int)
        df['Scale'] = split_columns[4].astype(str)
        df['Scale'] = df['Scale'].apply(lambda x: x.zfill(3))
        df['City-Scale'] = df['City'] + '-' + df['Scale'].astype(str)
        df["signeddenominator"] = df["ratio"]

        #     for i in df.index:
        #         city_scale_tuple = (df.loc[i, 'City'], df.loc[i, 'Scale'])
        # #         print (city_scale_tuple, )
        #         if city_scale_tuple in non_recurrent_gof_dict:
        #             df.loc[i, 'signeddenominator'] /= non_recurrent_gof_dict[city_scale_tuple]
        #             print (non_recurrent_gof_dict[city_scale_tuple])
        return df




    # Process files
    files = ['NONRECURRENTFigure3city' + str(x) + '.csv' for x in range(2, 9)]  # City indices from 2 to 8

    all_data = pd.concat([load_data_direction(f) for f in files])

    heatmap_data = all_data.pivot_table(index='feature', columns='City-Scale', values='signeddenominator', aggfunc='mean')
    # plt.hist(heatmap_data.to_numpy().flatten().tolist(), bins=20)
    # plt.tight_layout(); plt.show()
    plt.clf()

    heatmap_data = heatmap_data.where(backup_non_recurrent_nans.isna())
    original_xticks = heatmap_data.columns.tolist()  # Save original x-tick labels\
    # heatmap_data = heatmap_data.apply(lambda row: adjust_row_based_on_nan_count(row), axis=1)
    heatmap_data.columns = original_xticks
    # heatmap_data = heatmap_data.clip(-0.000000001, 0.000000001)
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, annot=False, cmap="seismic", cbar_kws={'label': 'Sensitivity Ratio'}, center=0,
                yticklabels=True, xticklabels=True)
    # plt.title("Sensitivity")
    plt.tight_layout();
    # plt.savefig("nonrecurrent_Fi_4b.pdf", dpi=300)
    # plt.show()
    plt.clf()



















    positive_vals = {}
    val_list = []
    for col in heatmap_data.columns:
        for feature in heatmap_data[col].index.to_list():
            city = col.split("-")[0]

            if heatmap_data[col][feature] > 0:
                val_list.append(heatmap_data[col][feature])

            if (city,feature) not in positive_vals:
                positive_vals[city, feature] = [(heatmap_data[col][feature])]
            else:
                positive_vals[city, feature].append((heatmap_data[col][feature]))
    print ("************** SELECTED FEATURES WITH POSITIVE SIGN *********\n\n\n\n\n\n")
    listkeys = list(positive_vals.keys())
    for key in listkeys:
        if not (len(set(([np.sign(x) for x in positive_vals[key]]))) == 1 and sum([np.sign(x) for x in positive_vals[key]]) == 10):
            del positive_vals[key]
        # else:
        #     print (key, positive_vals[key])

    # filter top 3
    count_per_city = {}
    for key in positive_vals:
        if key[0] not in count_per_city:
            count_per_city[key[0]] =  [ (key[1], np.mean(positive_vals[key])) ]
        else:
            count_per_city[key[0]].append((key[1], np.mean(positive_vals[key])))

    # Processing to retain only the top 3 features based on absolute values
    top_features_per_city = {}
    for city, features in count_per_city.items():
        # Sort features based on the absolute value of the second element in the tuple
        sorted_features = sorted(features, key=lambda x: np.abs(x[1]), reverse=True)
        sprint(city, sorted_features, "Positive")
        # Keep the top 3 features or all if there are 3 or fewer
        top_features_per_city[city] = sorted_features[:min(3, len(sorted_features))]

    # Print the processed dictionary
    for city, features in top_features_per_city.items():
        print(f"{city}: {[x[0] for x in features]}")
        top_features_per_city[city] = [x[0] for x in features]

    listkeys = list(positive_vals.keys())
    for key in listkeys:
        if key[1] not in top_features_per_city[key[0]]:
            del positive_vals[key]

    # print ("*******************************\n\n\n\n\n\n")
    # plt.hist(val_list, bins=20)
    # plt.title("Hist of Positive SR values")
    # plt.show()

    # Define colors for each city
    colors = {
        "Mumbai".lower(): "#E69F00",  # orange
        "Auckland".lower(): "#56B4E9",  # sky blue
        "Istanbul".lower(): "#009E73",  # bluish green
        "MexicoCity".lower(): "#F0E442",  # yellow
        "Bogota".lower(): "#0072B2",  # blue
        "NewYorkCity".lower(): "#D55E00",  # vermilion
        "Capetown".lower(): "#CC79A7",  # reddish purple
    }
    linestyles = {
        'betweenness': 'solid',  # Solid line
        'circuity_avg': 'dashed',  # Dashed line
        'global_betweenness': (0, (3, 1, 1, 1)),  # Dash-dot line
        'k_avg': 'dotted',  # Dotted line
        'm': (0, (5, 10)),  # Long dash line
        'mean_lanes': (0, (3, 5, 1, 5)),  # Dash-dot-dot line
        'metered_count': (0, (1, 1)),  # Fine dotted line
        'n': (0, (5, 5)),  # Evenly spaced dash line
        'non_metered_count': (0, (3, 10, 1, 10)),  # Long dash-dot line
        'streets_per_node_avg': 'dashdot',  # Dash-dot (predefined style)
        'streets_per_node_count_1': (0, (3, 1)),  # Short dash line
        'streets_per_node_count_2': (0, (1, 10)),  # Long dot line
        'streets_per_node_count_3': (0, (5, 1, 1, 1, 1, 1)),  # Dash followed by fine dots
        'streets_per_node_count_4': (0, (3, 5, 1, 5, 1, 5)),  # Complex pattern
        'streets_per_node_count_5': (0, (5, 1, 3, 1, 1, 1)),  # Varied pattern
        'total_crossings': (0, (1, 5))  # Sparse dotted line
    }
    plt.figure(figsize=(8, 6))
    plt.clf()
    list_of_features = heatmap_data.index.to_list()
    area_list = [(50 / x) ** 2 for x in [20, 25, 30, 40, 50, 60, 70, 80, 90, 100]]
    for key in positive_vals:
        plt.plot(area_list, positive_vals[key], label=key, color=colors[key[0]],
                 linewidth=list_of_features.index(key[1]) / 5)

    plt.legend(fontsize=9, ncol=2, loc="best")
    plt.yscale("log")
    plt.xlabel(r'Tile Area (km$^2$)', fontsize=20)
    plt.ylabel("|SR|", fontsize=20)
    plt.ylim(1e-6, 1)
    plt.title("Magnitude of SR for features\n with +ve SR for all scales", fontsize=20)
    plt.tight_layout()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.savefig("nonrecurrent_Fig5_var_abs_SR_Positive.pdf", dpi=300)
    # plt.show()
    plt.clf()





    positive_vals = {}
    val_list = []
    for col in heatmap_data.columns:
        for feature in heatmap_data[col].index.to_list():
            city = col.split("-")[0]

            if heatmap_data[col][feature] < 0:
                val_list.append(heatmap_data[col][feature])

            if (city,feature) not in positive_vals:
                positive_vals[city, feature] = [-(heatmap_data[col][feature])]
            else:
                positive_vals[city, feature].append(-(heatmap_data[col][feature]))
    print ("************** SELECTED FEATURES WITH NEGATIVE SIGN *********\n\n\n\n\n\n")
    listkeys = list(positive_vals.keys())
    for key in listkeys:
        if not (len(set(([np.sign(x) for x in positive_vals[key]]))) == 1 and sum([np.sign(x) for x in positive_vals[key]]) == 10):
            del positive_vals[key]
        # else:
        #     print (key, positive_vals[key])

    # filter top 3
    count_per_city = {}
    for key in positive_vals:
        if key[0] not in count_per_city:
            count_per_city[key[0]] =  [ (key[1], np.mean(positive_vals[key])) ]
        else:
            count_per_city[key[0]].append((key[1], np.mean(positive_vals[key])))

    # Processing to retain only the top 3 features based on absolute values
    top_features_per_city = {}
    for city, features in count_per_city.items():
        # Sort features based on the absolute value of the second element in the tuple
        sorted_features = sorted(features, key=lambda x: np.abs(x[1]), reverse=True)
        sprint(city, sorted_features, "Negative")
        # Keep the top 3 features or all if there are 3 or fewer
        top_features_per_city[city] = sorted_features[:min(3, len(sorted_features))]

    # Print the processed dictionary
    for city, features in top_features_per_city.items():
        print(f"{city}: {[x[0] for x in features]}")
        top_features_per_city[city] = [x[0] for x in features]

    listkeys = list(positive_vals.keys())
    for key in listkeys:
        if key[1] not in top_features_per_city[key[0]]:
            del positive_vals[key]

    print ("*******************************\n\n\n\n\n\n")
    plt.hist(val_list, bins=20)
    # plt.title("Hist of Positive SR values")
    # plt.show()
    plt.clf()

    # Define colors for each city
    colors = {
        "Mumbai".lower(): "#E69F00",  # orange
        "Auckland".lower(): "#56B4E9",  # sky blue
        "Istanbul".lower(): "#009E73",  # bluish green
        "MexicoCity".lower(): "#F0E442",  # yellow
        "Bogota".lower(): "#0072B2",  # blue
        "NewYorkCity".lower(): "#D55E00",  # vermilion
        "Capetown".lower(): "#CC79A7",  # reddish purple
    }
    linestyles = {
        'betweenness': 'solid',  # Solid line
        'circuity_avg': 'dashed',  # Dashed line
        'global_betweenness': (0, (3, 1, 1, 1)),  # Dash-dot line
        'k_avg': 'dotted',  # Dotted line
        'm': (0, (5, 10)),  # Long dash line
        'mean_lanes': (0, (3, 5, 1, 5)),  # Dash-dot-dot line
        'metered_count': (0, (1, 1)),  # Fine dotted line
        'n': (0, (5, 5)),  # Evenly spaced dash line
        'non_metered_count': (0, (3, 10, 1, 10)),  # Long dash-dot line
        'streets_per_node_avg': 'dashdot',  # Dash-dot (predefined style)
        'streets_per_node_count_1': (0, (3, 1)),  # Short dash line
        'streets_per_node_count_2': (0, (1, 10)),  # Long dot line
        'streets_per_node_count_3': (0, (5, 1, 1, 1, 1, 1)),  # Dash followed by fine dots
        'streets_per_node_count_4': (0, (3, 5, 1, 5, 1, 5)),  # Complex pattern
        'streets_per_node_count_5': (0, (5, 1, 3, 1, 1, 1)),  # Varied pattern
        'total_crossings': (0, (1, 5))  # Sparse dotted line
    }
    plt.figure(figsize=(8, 6))
    plt.clf()
    list_of_features = heatmap_data.index.to_list()
    area_list = [(50 / x) ** 2 for x in [20, 25, 30, 40, 50, 60, 70, 80, 90, 100]]
    for key in positive_vals:
        plt.plot(area_list, positive_vals[key], label=key, color=colors[key[0]],
                 linewidth=list_of_features.index(key[1]) / 5)

    plt.legend(fontsize=9, ncol=2, loc="best")
    plt.yscale("log")
    plt.xlabel(r'Tile Area (km$^2$)', fontsize=20)
    plt.ylabel("|SR|", fontsize=20)
    plt.ylim(1e-6, 1)
    plt.title("Magnitude of SR for features\n with -ve SR for all scales", fontsize=20)
    plt.tight_layout()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.savefig("nonrecurrent_Fig5_var_abs_SR_Negative.pdf", dpi=300)
    # plt.show()
    plt.clf()

    print ("\n\n\n\n")


























    positive_vals = {}
    val_list = []
    for col in heatmap_data.columns:
        for feature in heatmap_data[col].index.to_list():
            city = col.split("-")[0]

            if heatmap_data[col][feature] > -1000000000000:
                val_list.append(heatmap_data[col][feature])

            if (city,feature) not in positive_vals:
                positive_vals[city, feature] = [np.abs(heatmap_data[col][feature])]
            else:
                positive_vals[city, feature].append(np.abs(heatmap_data[col][feature]))
    listkeys = list(positive_vals.keys())
    # for key in listkeys:
    #     if not (len(set(([np.sign(x) for x in positive_vals[key]]))) == 1 and sum([np.sign(x) for x in positive_vals[key]]) == 10):
    #         del positive_vals[key]
    #     else:
    #         print (key, positive_vals[key])

    # filter top 3
    count_per_city = {}
    for key in positive_vals:
        if key[0] not in count_per_city:
            count_per_city[key[0]] =  [ (key[1], np.mean(positive_vals[key])) ]
        else:
            count_per_city[key[0]].append((key[1], np.mean(positive_vals[key])))

    # Processing to retain only the top 3 features based on absolute values
    top_features_per_city = {}
    for city, features in count_per_city.items():
        # Sort features based on the absolute value of the second element in the tuple
        sorted_features = sorted(features, key=lambda x: np.abs(x[1]), reverse=True)
        # Keep the top 3 features or all if there are 3 or fewer
        top_features_per_city[city] = sorted_features[:min(3, len(sorted_features))]

    # Print the processed dictionary
    for city, features in top_features_per_city.items():
        # print(f"{city}: {[x[0] for x in features]}")
        top_features_per_city[city] = [x[0] for x in features]

    listkeys = list(positive_vals.keys())
    # for key in listkeys:
    #     if key[1] not in top_features_per_city[key[0]]:
    #         del positive_vals[key]

    # print ("*******************************\n\n\n\n\n\n")
    # plt.hist(val_list, bins=20)
    # plt.title("Hist of Positive SR values")
    # plt.show()

    # Define colors for each city
    colors = {
        "Mumbai".lower(): "#E69F00",  # orange
        "Auckland".lower(): "#56B4E9",  # sky blue
        "Istanbul".lower(): "#009E73",  # bluish green
        "MexicoCity".lower(): "#F0E442",  # yellow
        "Bogota".lower(): "#0072B2",  # blue
        "NewYorkCity".lower(): "#D55E00",  # vermilion
        "Capetown".lower(): "#CC79A7",  # reddish purple
    }
    linestyles = {
        'betweenness': 'solid',  # Solid line
        'circuity_avg': 'dashed',  # Dashed line
        'global_betweenness': (0, (3, 1, 1, 1)),  # Dash-dot line
        'k_avg': 'dotted',  # Dotted line
        'm': (0, (5, 10)),  # Long dash line
        'mean_lanes': (0, (3, 5, 1, 5)),  # Dash-dot-dot line
        'metered_count': (0, (1, 1)),  # Fine dotted line
        'n': (0, (5, 5)),  # Evenly spaced dash line
        'non_metered_count': (0, (3, 10, 1, 10)),  # Long dash-dot line
        'streets_per_node_avg': 'dashdot',  # Dash-dot (predefined style)
        'streets_per_node_count_1': (0, (3, 1)),  # Short dash line
        'streets_per_node_count_2': (0, (1, 10)),  # Long dot line
        'streets_per_node_count_3': (0, (5, 1, 1, 1, 1, 1)),  # Dash followed by fine dots
        'streets_per_node_count_4': (0, (3, 5, 1, 5, 1, 5)),  # Complex pattern
        'streets_per_node_count_5': (0, (5, 1, 3, 1, 1, 1)),  # Varied pattern
        'total_crossings': (0, (1, 5))  # Sparse dotted line
    }
    plt.figure(figsize=(8, 6))
    plt.clf()
    list_of_features = heatmap_data.index.to_list()
    area_list = [(50 / x) ** 2 for x in [20, 25, 30, 40, 50, 60, 70, 80, 90, 100]]
    new_list = []
    labellist = []
    for key in positive_vals:
        # plt.plot(area_list, positive_vals[key], label=str(key) + str(np.sum(np.diff(positive_vals[key]))), color=colors[key[0]],
        #          linewidth=list_of_features.index(key[1]) / 5)
        a = np.array(positive_vals[key])

        if not (np.isnan(a).any() or np.isnan(np.std(a)) or np.std(a) == 0):
            mu = np.mean(a)
            sigma = np.std(a)
            a_dash = (a - mu) / sigma

            # new_list.append(a)

            new_list.append(a_dash)
            labellist.append(key)

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance
    from tslearn.clustering import silhouette_score as ts_silhouette_score
    from tslearn.metrics import dtw

    # Normalize the data
    # timeseries_data = TimeSeriesScalerMeanVariance().fit_transform(heatmap_data_backup.values)
    # timeseries_data = TimeSeriesScalerMeanVariance().fit_transform(FI_as_timeseries)
    timeseries_data = np.array(new_list)
    N_CLUSTERS = 2

    # Setting up K-means clustering with DTW
    model = TimeSeriesKMeans(n_clusters=N_CLUSTERS, metric="dtw", max_iter=10, random_state=0)
    clusters = model.fit_predict(timeseries_data)

    # Plotting results
    plt.figure(figsize=(10, 12))
    for yi in range(N_CLUSTERS):
        plt.subplot(N_CLUSTERS, 1, yi + 1)
        for xx in timeseries_data[clusters == yi]:
            plt.plot(area_list, xx.ravel(), "k-", alpha=0.2)
        plt.plot(area_list, model.cluster_centers_[yi].ravel(), "r-", linewidth=3)
        plt.xlim(0, 6.5)
        plt.xlabel(r'Tile Area (km$^2$)', fontsize=20)
        plt.ylabel("|SR|", fontsize=20)
        plt.ylim(np.min(timeseries_data), np.max(timeseries_data))
        plt.title(f"Cluster {yi + 1}")

    plt.tight_layout()
    # plt.savefig("DTW_nonrecurent_selected_postive_features.pdf", dpi=300)
    # plt.show()
    plt.clf()

    # Printing which label belongs to which cluster
    print("Cluster assignments:")
    for label, cluster_num in zip(labellist, clusters):
        print(f"{label} is in Cluster {cluster_num + 1}")

    # Print labels for each cluster
    for cluster_id in range(N_CLUSTERS):
        print(f"\nLabels in Cluster {cluster_id + 1}:")
        for label, cluster in zip(labellist, clusters):
            if cluster == cluster_id:
                print(label)




























    plt.figure(figsize=(14, 8))
    sns.heatmap(np.log(heatmap_data+0.127), annot=False, cmap="seismic", cbar_kws={'label': 'Sensitivity Ratio'},
                yticklabels=True, xticklabels=True)
    # plt.title("Log of sensitivity")
    # plt.savefig("nonrecurrent_Fi_4c.pdf", dpi=300)
    # plt.tight_layout();
    # plt.show()
    plt.clf()

    # trick to allow code folding :)

    import pandas as pd


    def create_city_feature_dict(df):
        city_feature_dict = {}

        for col in df.columns:
            city = col.split('-')[0].lower()
            if city not in city_feature_dict:
                city_feature_dict[city] = []

            for feature in df.index:
                if pd.isna(df.at[feature, col]) and feature not in city_feature_dict[city]:
                    city_feature_dict[city].append(feature)

        return city_feature_dict


    # Example usage
    # Assuming backup_non_recurrent_nans is your DataFrame
    city_feature_dict = create_city_feature_dict(backup_non_recurrent_nans)

    # Display the result
    import pprint

    # pprint.pprint(city_feature_dict)

    # Create the dictionary structure
    city_feature_dict_true_values_for_scale_dependency = {}

    for col in original_xticks:
        # print(col)
        city, scale = col.split('-')
        scale = int(scale)
        if city not in city_feature_dict_true_values_for_scale_dependency:
            city_feature_dict_true_values_for_scale_dependency[city] = []

        # Identify features that are not NaN across all scales for this city
        valid_features = heatmap_data[col].notna()
        features_dict = {}

        for feature, is_valid in valid_features.items():
            if feature in city_feature_dict[city]:
                if feature not in features_dict:
                    features_dict[feature] = {}
                features_dict[feature][scale] = heatmap_data.at[feature, col]

        # Add non-empty feature dictionaries to the city's list
        city_feature_dict_true_values_for_scale_dependency[city].append(features_dict)

    import pprint

    # pprint.pprint(city_feature_dict_true_values_for_scale_dependency)
    plt.figure(figsize=(14, 8))
    # Collate data
    collated_data = {}

    for city, features in city_feature_dict_true_values_for_scale_dependency.items():
        for feature_list in features:
            for feature, scales in feature_list.items():
                if (city, feature) not in collated_data:
                    collated_data[(city, feature)] = {}
                collated_data[(city, feature)].update(scales)

    # Initialize the plot
    plt.figure(figsize=(5, 6))

    # Color-blind friendly palette for seven cities
    colors = {
        "Mumbai".lower(): "#E69F00",  # orange
        "Auckland".lower(): "#56B4E9",  # sky blue
        "Istanbul".lower(): "#009E73",  # bluish green
        "MexicoCity".lower(): "#F0E442",  # yellow
        "Bogota".lower(): "#0072B2",  # blue
        "NewYorkCity".lower(): "#D55E00",  # vermilion
        "Capetown".lower(): "#CC79A7",  # reddish purple
    }

    FI_as_timeseries = []
    keys_for_cluster_tracking = []
    # Plot the data
    for (city, feature), scales in collated_data.items():
        scales_list = sorted(scales.keys())
        values = [scales[scale] for scale in scales_list]
        if city.lower() != "istanbul":
            # arealist = [(50 / x) ** 2 for x in scales_list]
            arealist = [x * 1.5 for x in scales_list]
        else:
            # arealist = [(75 / x) ** 2 for x in scales_list]
            arealist = [x for x in scales_list]
        plt.plot(arealist, values, label=f"{city}-{feature}", color=colors[city], marker='o')
        # FI_as_timeseries.append(np.array(values) / np.array(arealist))
        FI_as_timeseries.append(values)
        keys_for_cluster_tracking.append((city.lower(), feature))
    FI_as_timeseries = np.array(FI_as_timeseries)

    # Add labels and title
    plt.xlabel('Scale')
    plt.ylabel('Feature Importance')
    # plt.title('Feature Importance vs. Scale for Each City-Feature Combination')
    # plt.legend(loc='best', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout();
    # plt.savefig("nonrecurrent_Fi_4d.pdf", dpi=300)
    # plt.show()
    plt.clf()
    # Display the dictionary

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance
    from tslearn.clustering import silhouette_score as ts_silhouette_score
    from tslearn.metrics import dtw

    # Normalize the data
    # timeseries_data = TimeSeriesScalerMeanVariance().fit_transform(heatmap_data_backup.values)
    timeseries_data = TimeSeriesScalerMeanVariance().fit_transform(FI_as_timeseries)
    # timeseries_data = FI_as_timeseries

    HARDCODED_CLUSTER = 3
    if HARDCODED_CLUSTER == 0:
        # Determine the optimal number of clusters using the Elbow method and Silhouette analysis
        wcss = []
        silhouette_scores = []
        max_clusters = 10

        for k in range(2, max_clusters + 1):
            km = TimeSeriesKMeans(n_clusters=k, metric="dtw", random_state=0)
            clusters = km.fit_predict(timeseries_data)
            wcss.append(km.inertia_)  # WCSS
            silhouette_scores.append(ts_silhouette_score(timeseries_data, clusters, metric='dtw'))

        # Plot Elbow method
        plt.figure(figsize=(14, 8))
        plt.plot(range(2, max_clusters + 1), wcss, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS (Within-cluster sum of squares)')
        # plt.title('Elbow Method for Determining the Optimal Number of Clusters')
        plt.grid(True, alpha=0.3)
        plt.tight_layout();
        # plt.savefig("nonrecurrent_Fi_4e.pdf", dpi=300)
        # plt.show()
        plt.clf()

        # Plot Silhouette scores
        plt.figure(figsize=(14, 8))
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        # plt.title('Silhouette Analysis for Determining the Optimal Number of Clusters')
        plt.grid(True, alpha=0.3)
        plt.tight_layout();
        # plt.savefig("nonrecurrent_Fi_4f.pdf", dpi=300)
        # plt.show()'
        plt.clf()

        # Choose the optimal number of clusters (based on the Elbow or Silhouette)
        optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because range starts from 2
        HARDCODED_CLUSTER = optimal_clusters

    # Perform K-means clustering with DTW as the distance metric and the optimal number of clusters
    km = TimeSeriesKMeans(n_clusters=HARDCODED_CLUSTER, metric="dtw", random_state=0)
    clusters = km.fit_predict(timeseries_data)

    # Calculate the representative time series for each cluster
    cluster_representatives = np.zeros((HARDCODED_CLUSTER, timeseries_data.shape[1]))
    for i in range(HARDCODED_CLUSTER):
        cluster_indices = np.where(clusters == i)[0]
        cluster_timeseries = timeseries_data[cluster_indices]
        try:
            # cluster_representatives[i] = np.median(cluster_timeseries, axis=0)
            cluster_representatives[i] = np.median(cluster_timeseries, axis=0).reshape(cluster_representatives[i].shape)
        except Exception as e:
            debug_pitstop = True
            raise e

    # Define colors for each city
    colors = {
        "Mumbai".lower(): "#E69F00",  # orange
        "Auckland".lower(): "#56B4E9",  # sky blue
        "Istanbul".lower(): "#009E73",  # bluish green
        "MexicoCity".lower(): "#F0E442",  # yellow
        "Bogota".lower(): "#0072B2",  # blue
        "NewYorkCity".lower(): "#D55E00",  # vermilion
        "Capetown".lower(): "#CC79A7",  # reddish purple
    }

    # Extract city names from labels
        # Extract city names from labels
    citylist_alphabetical = ['auckland',
                         'bogota',
                         'capetown',
                         'istanbul',
                         'mexicocity',
                         'mumbai',
                         'newyorkcity'
                             ]
    city_labels = []
    for i, citynamelowercase in enumerate(citylist_alphabetical):
        city_labels.extend([citynamelowercase] * heatmap_data.shape[0])

    ratio_of_last_to_first_element = []
    for row in cluster_representatives:
        ratio_of_last_to_first_element.append((row[-1] - row[0]))

    sorted_indices = sorted(range(len(ratio_of_last_to_first_element)), key=lambda k: ratio_of_last_to_first_element[k])
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]  # Tab Blue, Tab Green, Tab Orange

    cluster_color = {sorted_indices[i]: colors[i] for i in range(3)}
    cluster_names = {}
    for key in cluster_color:
        if cluster_color[key] == "#1f77b4":
            # blue increasein
            cluster_names[key] = "Increasing"
        if cluster_color[key] == "#ff7f0e":
            # blue increasein
            cluster_names[key] = "Decreasing"
        if cluster_color[key] == "#2ca02c":
            # blue increasein
            cluster_names[key] = "Indeterminate"



    # Plot the clustered time series
    plt.clf()
    plt.close()
    plt.figure(figsize=(6, 6))

    area_list = [(50/x)**2 for x in [20, 25, 30, 40, 50, 60, 70, 80, 90, 100]]
    for i in range(HARDCODED_CLUSTER):
        cluster_indices = np.where(clusters == i)[0]
        for idx in cluster_indices:
            try:
                city = city_labels[idx]
            except Exception as e:
                debug_pitstop = True
                raise e
            plt.plot(area_list, timeseries_data[idx].ravel(), alpha=0.1, color=cluster_color[i])
        plt.plot(area_list, cluster_representatives[i], label=f"Cluster Representative ({cluster_names[i]})", linewidth=2, color=cluster_color[i])

    plt.xlabel(r'Tile Area (km$^2$)')
    plt.ylabel('Sensitivity Ratio (z-normalised)')
    # plt.title(f'Clustered Feature Importance vs. Scale (n_clusters={HARDCODED_CLUSTER})')
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.0)
    # plt.tight_layout();
    # plt.savefig("nonrecurrent_Fi_4g.pdf", dpi=300)
    # plt.show()
    plt.clf()

    # Display the number of time series in each cluster
    for i in range(HARDCODED_CLUSTER):
        print(f"Cluster {i + 1}: {len(np.where(clusters == i)[0])} time series")

    dictzip = dict(zip(keys_for_cluster_tracking, clusters.tolist()))
    inv_map = {}
    for k, v in dictzip.items():
        inv_map[v] = inv_map.get(v, []) + [k]

    print("============ SENSITIVITY SHAP  ================")
    sprint (inv_map)
    print ("===================================")




    # Plot the data
    for (city, feature), scales in collated_data.items():
        scales_list = sorted(scales.keys())
        values = [scales[scale] for scale in scales_list]
        plt.plot(arealist, values, label=f"{city}-{feature}", color=cluster_color[dictzip[city.lower(), feature]], marker='o')
        # FI_as_timeseries.append(np.array(values) / np.array(arealist))
    # Add labels and title
    plt.xlabel(r'Tile Area (km$^2$)')
    plt.ylabel('Feature Importance')
    # plt.title('Ratio vs. Scale for Each City-Feature Combination')
    # plt.legend(loc='best', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout();
    # plt.savefig("nonrecurrent_Fi_4h.pdf", dpi=300)
    # plt.show()
    plt.clf()

debug_pitstop = True


# K means and K medoids tested but doesnt work.  K medoids tested with both Euclidean and cosine similarities)