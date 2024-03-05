import os
import pandas as pd

# Define the root directory (where the script is located)
root_dir = '.'

# Hardcoded list of city names
cities = ['Auckland', 'Zurich', 'Mumbai', 'Bogota', 'Singapore', 'MexicoCity', 'Istanbul', 'Capetown', 'NewYorkCity', 'London']

# Function to create a dataframe with features as rows and city_scale as columns
def create_feature_city_scale_matrix(root_dir, cities):
    features = set()
    city_scale_combinations = set()

    # Scan through directories and find the PDP plot files
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if 'PDP_plots' in subdir and file.startswith('tod_evening_peak_shap_pdp_') and file.endswith('.png'):
                parts = subdir.split(os.sep)
                city = parts[-2]  # City is one directory up from the file
                if city in cities:
                    scale = file.split('_')[-1].split('.')[0]  # Assuming scale is the last part before .png
                    feature = file.split('shap_pdp_')[1].split('_scale')[0]
                    features.add(feature)
                    city_scale_combinations.add((city, scale))


    features = [
        'betweenness',
        'circuity_avg',
        'global_betweenness',
        'k_avg',
        'lane_density',
        'm',
        'metered_count',
        'n',
        'non_metered_count',
        'street_length_total',
        'streets_per_node_count_5',
        'total_crossings'
    ]
    # Create a dataframe with features as rows and city_scale as columns
    matrix_df = pd.DataFrame(0, index=sorted(list(features)), columns=pd.MultiIndex.from_tuples(sorted(city_scale_combinations), names=['City', 'Scale']))

    # Create a dataframe with features as rows and city_scale as combinations with the specified city order
    city_scale_combinations_ordered = [(city, scale) for city in cities for scale in sorted({s for c, s in city_scale_combinations if c == city})]
    matrix_df = pd.DataFrame(0, index=sorted(list(features)), columns=pd.MultiIndex.from_tuples(city_scale_combinations_ordered, names=['City', 'Scale']))


    # Fill in the dataframe
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if 'PDP_plots' in subdir and file.startswith('tod_evening_peak_shap_pdp_') and file.endswith('.png'):
                parts = subdir.split(os.sep)
                city = parts[-2]  # City is one directory up from the file
                if city in cities:
                    scale = file.split('_')[-1].split('.')[0]  # Assuming scale is the last part before .png
                    feature = file.split('shap_pdp_')[1].split('_scale')[0]
                    matrix_df.at[feature, (city, scale)] = 1

    return matrix_df

# Create the feature-city-scale matrix
feature_city_scale_matrix = create_feature_city_scale_matrix(root_dir, cities)

# Save the matrix to a CSV file
csv_file_path = './feature_city_scale_matrix.csv'
feature_city_scale_matrix.to_csv(csv_file_path)

csv_file_path



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the data from CSV
feature_city_scale_matrix = pd.read_csv('./feature_city_scale_matrix.csv', header=[0,1], index_col=0)

# Define a binary colormap (black and white)
binary_cmap = ListedColormap(['white', 'black'])

# Create a heatmap without annotations
plt.figure(figsize=(20, 10))
ax = sns.heatmap(feature_city_scale_matrix, cmap=binary_cmap, cbar=True, cbar_kws={'ticks': [0.25, 0.75], 'format': plt.FuncFormatter(lambda x, _: 'Below Otsu' if x < 0.5 else 'Above Otsu')})
plt.title('Feature-City-Scale Matrix',fontsize=15)
plt.xlabel('City, Scale', fontsize=15)
plt.ylabel('Feature', fontsize=15)
plt.xticks(rotation=90, ha='right')

# Add vertical dashed lines after each city (every 3 x-ticks)
for i in range(3, len(ax.get_xticks()), 3):
    plt.axvline(x=i, color='gray', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig("Heatmap_evening_peak.png", dpi=300)
plt.show()
