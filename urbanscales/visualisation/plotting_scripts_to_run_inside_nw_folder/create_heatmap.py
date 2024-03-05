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
            if 'PDP_plots' in subdir and file.startswith('tod_all_day_shap_pdp_') and file.endswith('.png'):
                parts = subdir.split(os.sep)
                city = parts[-2]  # City is one directory up from the file
                if city in cities:
                    scale = file.split('_')[-1].split('.')[0]  # Assuming scale is the last part before .png
                    feature = file.split('shap_pdp_')[1].split('_scale')[0]
                    features.add(feature)
                    city_scale_combinations.add((city, scale))

    # Create a dataframe with features as rows and city_scale as columns
    matrix_df = pd.DataFrame(0, index=sorted(list(features)), columns=pd.MultiIndex.from_tuples(sorted(city_scale_combinations), names=['City', 'Scale']))

    # Fill in the dataframe
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if 'PDP_plots' in subdir and file.startswith('tod_all_day_shap_pdp_') and file.endswith('.png'):
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
