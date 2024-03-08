import os
import pandas as pd

# List of cities
cities = ['London', 'Capetown', 'NewYorkCity', 'Bogota', 'MexicoCity', 'Auckland', 'Mumbai', 'Zurich', 'Singapore', 'Istanbul']

# Scales
scales = [25, 50, 100]

# Initialize an empty dictionary to store the data
feature_importance_dict = {}

# Loop through each city and scale, read the CSV file, and add the data to the dictionary
for city in cities:
    feature_importance_dict[city] = {}
    for scale in scales:
        file_path = f'{city}/spatial_tod_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]_scale_{scale}/spatial_total_feature_importance.csv'
        # file_path = file_path.replace(' ', '\ ')  # Replace spaces with '\ '
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            feature_importance_dict[city][scale] = df.set_index('Feature')['Total SHAP Value'].to_dict()
        else:
            print(f'File not found: {file_path}')

# Example usage: print the feature importance for Mexico City at scale 100
print(feature_importance_dict['MexicoCity'][100])
import matplotlib.pyplot as plt
import seaborn as sns

# Sort cities in alphabetical order
cities = sorted(cities)

# Prepare the data for the heatmap
heatmap_data = []
for city in cities:
    for scale in scales:
        for feature, value in feature_importance_dict[city][scale].items():
            heatmap_data.append([f'{city} Scale {scale}', feature, value])

# Convert the data to a DataFrame
heatmap_df = pd.DataFrame(heatmap_data, columns=['City & Scale', 'Feature', 'FI'])

# Pivot the DataFrame for the heatmap
heatmap_pivot = heatmap_df.pivot('Feature', 'City & Scale', 'FI')

# Manually set the order of the columns (City & Scale)
col_order = [f'{city} Scale {scale}' for city in cities for scale in scales]
heatmap_pivot = heatmap_pivot[col_order]

# Create the heatmap with a single-hue color map and without annotations
plt.figure(figsize=(15, 10))
sns.heatmap(heatmap_pivot, cmap='Blues', linewidths=.5, cbar_kws={'label': 'Feature Importance'})
plt.title('Feature Importance Heatmap; Recurrent, shifting 3, Spatial CV:4')
plt.xlabel('City & Scale')
plt.ylabel('Feature')
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.savefig("heatmap_spatial_recurrent.png", dpi=300)
plt.show()
