import copy

import numpy as np
from smartprint import smartprint as sprint

common_features_colors = {
    'betweenness': '#1f77b4',  # blue
    'circuity_avg': '#ff7f0e',  # orange
    'global_betweenness': '#2ca02c',  # green
    'k_avg': '#d62728',  # red
    'lane_density': '#9467bd',  # purple
    'metered_count': '#8c564b',  # brown
    'non_metered_count': '#e377c2',  # pink
    'street_length_total': '#7f7f7f',  # gray
    'm': '#bcbd22',  # lime
    'total_crossings': '#17becf',  # cyan
    "n": "black"
}
city_colors = {
    "Auckland": "#1f77b4",  # Muted blue
    "Bogota": "#ff7f0e",    # Safety orange
    "Capetown": "#2ca02c",  # Cooked asparagus green
    "Istanbul": "#d62728",  # Brick red
    "London": "#9467bd",    # Muted purple
    "MexicoCity": "#8c564b",# Chestnut brown
    "Mumbai": "#e377c2",    # Raspberry yogurt pink
    "NewYorkCity": "#7f7f7f",# Middle gray
    # "Singapore": "#bcbd22", # Curry yellow-green
    # "Zurich": "#17becf"     # Blue-teal
}
import matplotlib.pyplot as plt
import pandas as pd
import os

# Define lists of features, cities, and scales
features = list(common_features_colors.keys())  # Add more features as needed
cities = list(city_colors.keys())
scales = [25, 50, 100]

##### FILTERED FEATURE WISE
filter_important_features_recurrent = {25: {'betweenness': ['Bogota', 'London', 'Mumbai', 'Zurich'],
      'circuity_avg': ['London', 'Mumbai'],
      'global_betweenness': ['Bogota',
                             'Istanbul',
                             'London',
                             'Mumbai',
                             'Singapore'],
      'k_avg': ['Bogota',
                'Istanbul',
                'London',
                'Mumbai',
                'NewYorkCity',
                'Singapore'],
      'lane_density': ['London', 'Mumbai'],
      'm': ['Bogota', 'Istanbul', 'London', 'Zurich'],
      'metered_count': ['Auckland',
                        'Bogota',
                        'Capetown',
                        'MexicoCity',
                        'NewYorkCity',
                        'Zurich'],
      'n': ['London', 'Zurich'],
      'non_metered_count': ['Singapore', 'Zurich'],
      'street_length_total': ['Istanbul', 'London', 'MexicoCity', 'Mumbai'],
      'total_crossings': ['Auckland',
                          'Capetown',
                          'NewYorkCity',
                          'Singapore',
                          'Zurich']},
 50: {'betweenness': ['Bogota', 'Mumbai'],
      'circuity_avg': ['Istanbul', 'London', 'MexicoCity', 'Mumbai'],
      'global_betweenness': ['London', 'Zurich'],
      'k_avg': ['Auckland', 'London'],
      'lane_density': ['Bogota', 'London', 'MexicoCity'],
      'm': ['Istanbul', 'London', 'MexicoCity', 'Mumbai'],
      'metered_count': ['Bogota',
                        'Capetown',
                        'MexicoCity',
                        'NewYorkCity',
                        'Zurich'],
      'n': ['Istanbul', 'London', 'MexicoCity', 'Mumbai'],
      'non_metered_count': ['Singapore'],
      'street_length_total': ['Istanbul', 'MexicoCity'],
      'total_crossings': ['Auckland', 'Bogota', 'Capetown']},
 100: {'betweenness': ['Istanbul', 'MexicoCity', 'Mumbai'],
       'circuity_avg': ['Istanbul',
                        'London',
                        'Mumbai',
                        'NewYorkCity',
                        'Singapore',
                        'Zurich'],
       'global_betweenness': ['Istanbul', 'London', 'Zurich'],
       'k_avg': ['Auckland', 'London', 'Zurich'],
       'lane_density': ['Auckland',
                        'Bogota',
                        'Capetown',
                        'London',
                        'MexicoCity',
                        'NewYorkCity',
                        'Singapore',
                        'Zurich'],
       'm': ['London', 'MexicoCity', 'Zurich'],
       'metered_count': ['Auckland',
                         'Bogota',
                         'Capetown',
                         'London',
                         'MexicoCity',
                         'NewYorkCity',
                         'Zurich'],
       'n': ['Auckland', 'Mumbai'],
       'non_metered_count': ['London', 'Singapore'],
       'street_length_total': ['Istanbul', 'Mumbai'],
       'total_crossings': ['Bogota', 'NewYorkCity', 'Singapore']}}

filter_important_features_non_recurrent = {25: {'betweenness': ['Istanbul', 'NewYorkCity'],
      'circuity_avg': ['London', 'Zurich'],
      'global_betweenness': ['London', 'Mumbai'],
      'k_avg': ['NewYorkCity', 'Zurich'],
      'lane_density': ['London', 'NewYorkCity'],
      'm': ['Auckland',
            'Istanbul',
            'London',
            'MexicoCity',
            'NewYorkCity',
            'Zurich'],
      'metered_count': ['Auckland', 'Bogota', 'NewYorkCity'],
      'n': ['London'],
      'non_metered_count': ['Capetown'],
      'street_length_total': ['Capetown', 'Istanbul', 'Singapore'],
      'total_crossings': ['MexicoCity', 'Zurich']},
 50: {'betweenness': ['Auckland',
                      'Capetown',
                      'Istanbul',
                      'London',
                      'MexicoCity',
                      'Mumbai',
                      'NewYorkCity'],
      'circuity_avg': ['Istanbul', 'London', 'Mumbai', 'Zurich'],
      'global_betweenness': ['London', 'MexicoCity', 'Singapore'],
      'k_avg': ['Auckland',
                'Capetown',
                'Istanbul',
                'London',
                'MexicoCity',
                'NewYorkCity',
                'Zurich'],
      'lane_density': ['Auckland',
                       'Capetown',
                       'Istanbul',
                       'London',
                       'MexicoCity',
                       'NewYorkCity',
                       'Singapore',
                       'Zurich'],
      'm': ['Istanbul', 'London', 'MexicoCity', 'Mumbai', 'Zurich'],
      'metered_count': ['Auckland',
                        'Bogota',
                        'London',
                        'NewYorkCity',
                        'Zurich'],
      'n': ['Istanbul'],
      'non_metered_count': ['Auckland',
                            'Bogota',
                            'Capetown',
                            'London',
                            'NewYorkCity',
                            'Singapore',
                            'Zurich'],
      'street_length_total': ['Istanbul', 'MexicoCity', 'Mumbai'],
      'total_crossings': ['Bogota',
                          'Capetown',
                          'MexicoCity',
                          'Mumbai',
                          'Singapore',
                          'Zurich']},
 100: {'betweenness': ['Istanbul', 'MexicoCity'],
       'circuity_avg': ['Capetown',
                        'Istanbul',
                        'London',
                        'MexicoCity',
                        'Mumbai',
                        'NewYorkCity',
                        'Singapore',
                        'Zurich'],
       'global_betweenness': ['Capetown',
                              'Istanbul',
                              'London',
                              'MexicoCity',
                              'Mumbai',
                              'Singapore'],
       'k_avg': ['Auckland',
                 'Capetown',
                 'Istanbul',
                 'London',
                 'MexicoCity',
                 'Mumbai',
                 'NewYorkCity',
                 'Zurich'],
       'lane_density': ['Auckland',
                        'Bogota',
                        'Capetown',
                        'London',
                        'MexicoCity',
                        'NewYorkCity',
                        'Singapore',
                        'Zurich'],
       'm': ['Auckland',
             'Capetown',
             'Istanbul',
             'London',
             'MexicoCity',
             'Mumbai',
             'NewYorkCity',
             'Zurich'],
       'metered_count': ['Auckland',
                         'Capetown',
                         'Mumbai',
                         'NewYorkCity',
                         'Singapore',
                         'Zurich'],
       'n': ['Istanbul', 'Mumbai'],
       'non_metered_count': ['Capetown',
                             'London',
                             'NewYorkCity',
                             'Singapore',
                             'Zurich'],
       'street_length_total': ['Auckland',
                               'Capetown',
                               'Istanbul',
                               'London',
                               'MexicoCity',
                               'Mumbai',
                               'NewYorkCity',
                               'Singapore',
                               'Zurich'],
       'total_crossings': ['Bogota', 'Zurich']}}

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Extract all unique features and cities
all_features = set()
all_cities = set()
for scale in filter_important_features_recurrent:
    for feature, cities in filter_important_features_recurrent[scale].items():
        all_features.add(feature)
        all_cities.update(cities)

# Convert to list for indexing
all_features = sorted(list(all_features))
all_cities = sorted(list(all_cities))

for scale in [25, 50, 100]:
    # Create a binary matrix
    binary_matrix = []
    for feature in all_features:
        row = []
        for city in all_cities:
            row.append(city in filter_important_features_recurrent[scale].get(feature, []))
        binary_matrix.append(row)

    # Create a DataFrame for the heatmap
    df = pd.DataFrame(binary_matrix, index=all_features, columns=all_cities)

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=False, cmap="YlGnBu")
    plt.title('Recurrent Feature Heatmap Feature-wise cutoff @scale' + str(scale))
    plt.xlabel('Cities')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('Recurrent Feature Heatmap Feature-wise cutoff @scale' + str(scale) + ".png", dpi=300)
    plt.show(block=False)


# Combined scale

binary_matrix = []
for feature in all_features:
    row = []
    for city in all_cities:
        for scale in [25, 50, 100]:
            row.append(city in filter_important_features_recurrent[scale].get(feature, []))
    binary_matrix.append(row)

# Create a DataFrame for the heatmap
df = pd.DataFrame(binary_matrix, index=all_features, columns=[item for item in all_cities for i in range(3)])

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df, annot=False, cmap="YlGnBu")
plt.title('Recurrent Feature Heatmap Feature-wise cutoff @ALL_SCALES' + str(scale))
plt.xlabel('Cities')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('Recurrent Feature Heatmap Feature-wise cutoff @ALL_SCALES' + str(scale) + ".png", dpi=300)
plt.show(block=False)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Extract all unique features and cities
all_features = set()
all_cities = set()
for scale in filter_important_features_non_recurrent:
    for feature, cities in filter_important_features_non_recurrent[scale].items():
        all_features.add(feature)
        all_cities.update(cities)

# Convert to list for indexing
all_features = sorted(list(all_features))
all_cities = sorted(list(all_cities))

for scale in [25, 50, 100]:
    # Create a binary matrix
    binary_matrix = []
    for feature in all_features:
        row = []
        for city in all_cities:
            row.append(city in filter_important_features_non_recurrent[scale].get(feature, []))
        binary_matrix.append(row)

    # Create a DataFrame for the heatmap
    df = pd.DataFrame(binary_matrix, index=all_features, columns=all_cities)

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=False, cmap="YlGnBu")
    plt.title('Non Recurrent Feature Heatmap Feature-wise cutoff @scale' + str(scale))
    plt.xlabel('Cities')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('Non Recurrent Feature Heatmap Feature-wise cutoff @scale' + str(scale) + ".png", dpi=300)
    plt.show(block=False)



# Create a binary matrix
binary_matrix = []
for feature in all_features:
    row = []
    for city in all_cities:
        for scale in [25, 50, 100]:
            row.append(city in filter_important_features_non_recurrent[scale].get(feature, []))
    binary_matrix.append(row)

# Create a DataFrame for the heatmap
df = pd.DataFrame(binary_matrix, index=all_features, columns=[item for item in all_cities for i in range(3)])

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df, annot=False, cmap="YlGnBu")
plt.title('Non Recurrent Feature Heatmap Feature-wise cutoff @ALL_SCALES' + str(scale))
plt.xlabel('Cities')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('Non Recurrent Feature Heatmap Feature-wise cutoff @ALL_SCALES' + str(scale) + ".png", dpi=300)
plt.show(block=False)




##### FILTERED CITY WISE
filter_important_features_recurrent_city_wise={25: {'Auckland': ['metered_count', 'total_crossings'],
      'Bogota': ['metered_count'],
      'Capetown': ['metered_count', 'total_crossings'],
      'Istanbul': ['street_length_total'],
      'London': ['betweenness',
                 'circuity_avg',
                 'global_betweenness',
                 'k_avg',
                 'lane_density',
                 'street_length_total'],
      'MexicoCity': ['metered_count', 'street_length_total'],
      'Mumbai': ['betweenness',
                 'circuity_avg',
                 'k_avg',
                 'metered_count',
                 'street_length_total'],
      'NewYorkCity': ['metered_count', 'total_crossings'],
      'Singapore': ['global_betweenness', 'non_metered_count'],
      'Zurich': ['betweenness', 'metered_count', 'm']},
 50: {'Auckland': ['betweenness', 'k_avg', 'total_crossings'],
      'Bogota': ['betweenness',
                 'global_betweenness',
                 'metered_count',
                 'total_crossings'],
      'Capetown': ['metered_count', 'total_crossings'],
      'Istanbul': ['street_length_total'],
      'London': ['betweenness',
                 'circuity_avg',
                 'global_betweenness',
                 'k_avg',
                 'lane_density',
                 'street_length_total'],
      'MexicoCity': ['metered_count', 'street_length_total'],
      'Mumbai': ['betweenness',
                 'circuity_avg',
                 'global_betweenness',
                 'k_avg',
                 'street_length_total'],
      'NewYorkCity': ['global_betweenness', 'metered_count'],
      'Singapore': ['non_metered_count'],
      'Zurich': ['global_betweenness', 'metered_count']},
 100: {'Auckland': ['betweenness',
                    'circuity_avg',
                    'global_betweenness',
                    'k_avg',
                    'lane_density',
                    'metered_count',
                    'street_length_total',
                    'm',
                    'n'],
       'Bogota': ['betweenness',
                  'circuity_avg',
                  'global_betweenness',
                  'lane_density',
                  'metered_count',
                  'total_crossings'],
       'Capetown': ['betweenness',
                    'circuity_avg',
                    'global_betweenness',
                    'k_avg',
                    'lane_density',
                    'metered_count',
                    'street_length_total',
                    'total_crossings',
                    'n'],
       'Istanbul': ['betweenness',
                    'circuity_avg',
                    'global_betweenness',
                    'street_length_total'],
       'London': ['betweenness',
                  'circuity_avg',
                  'global_betweenness',
                  'k_avg',
                  'lane_density',
                  'street_length_total'],
       'MexicoCity': ['betweenness', 'street_length_total'],
       'Mumbai': ['betweenness',
                  'circuity_avg',
                  'global_betweenness',
                  'k_avg',
                  'street_length_total',
                  'n'],
       'NewYorkCity': ['circuity_avg', 'total_crossings'],
       'Singapore': ['circuity_avg', 'total_crossings'],
       'Zurich': ['betweenness',
                  'circuity_avg',
                  'global_betweenness',
                  'k_avg',
                  'lane_density']}}
filter_important_features_non_recurrent_city_wise = {25: {'Auckland': ['metered_count'],
      'Bogota': ['metered_count'],
      'Capetown': ['metered_count',
                   'non_metered_count',
                   'street_length_total',
                   'total_crossings'],
      'Istanbul': ['betweenness', 'street_length_total'],
      'London': ['n'],
      'MexicoCity': ['total_crossings'],
      'Mumbai': ['betweenness',
                 'global_betweenness',
                 'metered_count',
                 'street_length_total',
                 'total_crossings'],
      'NewYorkCity': ['metered_count'],
      'Singapore': ['betweenness', 'street_length_total'],
      'Zurich': ['k_avg', 'total_crossings']},
 50: {'Auckland': ['metered_count'],
      'Bogota': ['total_crossings'],
      'Capetown': ['betweenness', 'metered_count', 'total_crossings'],
      'Istanbul': ['street_length_total', 'n'],
      'London': ['metered_count'],
      'MexicoCity': ['betweenness', 'total_crossings'],
      'Mumbai': ['street_length_total', 'total_crossings'],
      'NewYorkCity': ['metered_count'],
      'Singapore': ['total_crossings'],
      'Zurich': ['k_avg', 'metered_count']},
 100: {'Auckland': ['metered_count'],
       'Bogota': ['total_crossings'],
       'Capetown': ['global_betweenness', 'metered_count'],
       'Istanbul': ['n'],
       'London': ['betweenness',
                  'circuity_avg',
                  'global_betweenness',
                  'k_avg',
                  'metered_count',
                  'street_length_total',
                  'n'],
       'MexicoCity': ['betweenness', 'k_avg'],
       'Mumbai': ['metered_count', 'n'],
       'NewYorkCity': ['global_betweenness', 'metered_count'],
       'Singapore': ['metered_count'],
       'Zurich': ['circuity_avg', 'metered_count']}}

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Extract all unique features and cities
all_features = set()
all_cities = set()
for scale, city_data in filter_important_features_recurrent_city_wise.items():
    for city, features in city_data.items():
        all_cities.add(city)
        all_features.update(features)

# Convert to list for indexing
all_features = sorted(list(all_features))
all_cities = sorted(list(all_cities))

# Iterate over scales
for scale in [25, 50, 100]:  # Add or remove scales as necessary
    # Create a binary matrix
    binary_matrix = []
    for feature in all_features:
        row = []
        for city in all_cities:
            row.append(feature in filter_important_features_recurrent_city_wise.get(scale, {}).get(city, []))
        binary_matrix.append(row)

    # Create a DataFrame for the heatmap
    df = pd.DataFrame(binary_matrix, index=all_features, columns=all_cities)

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=False, cmap="YlGnBu")
    plt.title('Recurrent Feature Heatmap @ Scale ' + str(scale))
    plt.xlabel('Cities')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('Recurrent Feature Heatmap City-wise cutoff @ Scale ' + str(scale) + ".png", dpi=300)
    plt.show(block=False)


# Iterate over scales
 # Add or remove scales as necessary
# Create a binary matrix
binary_matrix = []
for feature in all_features:
    row = []
    for city in all_cities:
        for scale in [25, 50, 100]:
            row.append(feature in filter_important_features_recurrent_city_wise.get(scale, {}).get(city, []))
    binary_matrix.append(row)

# Create a DataFrame for the heatmap
df = pd.DataFrame(binary_matrix, index=all_features, columns=[item for item in all_cities for i in range(3)])

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df, annot=False, cmap="YlGnBu")
plt.title('Recurrent Feature Heatmap @ ALL_SCALES ' + str(scale))
plt.xlabel('Cities')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('Recurrent Feature Heatmap City-wise cutoff @ ALL_SCALES ' + str(scale) + ".png", dpi=300)
plt.show(block=False)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Extract all unique features and cities
all_features = set()
all_cities = set()
for scale, city_data in filter_important_features_non_recurrent_city_wise.items():
    for city, features in city_data.items():
        all_cities.add(city)
        all_features.update(features)

# Convert to list for indexing
all_features = sorted(list(all_features))
all_cities = sorted(list(all_cities))

# Iterate over scales
for scale in [25, 50, 100]:  # Add or remove scales as necessary
    # Create a binary matrix
    binary_matrix = []
    for feature in all_features:
        row = []
        for city in all_cities:
            row.append(feature in filter_important_features_non_recurrent_city_wise.get(scale, {}).get(city, []))
        binary_matrix.append(row)

    # Create a DataFrame for the heatmap
    df = pd.DataFrame(binary_matrix, index=all_features, columns=all_cities)

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=False, cmap="YlGnBu")
    plt.title('Non Recurrent Feature Heatmap @ Scale ' + str(scale))
    plt.xlabel('Cities')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('Non Recurrent Feature Heatmap City-wise cutoff @ Scale ' + str(scale) + ".png", dpi=300)
    plt.show(block=False)


# Iterate over scales
  # Add or remove scales as necessary
    # Create a binary matrix
binary_matrix = []
for feature in all_features:
    row = []
    for city in all_cities:
        for scale in [25, 50, 100]:
            row.append(feature in filter_important_features_non_recurrent_city_wise.get(scale, {}).get(city, []))
    binary_matrix.append(row)

# Create a DataFrame for the heatmap
df = pd.DataFrame(binary_matrix, index=all_features, columns=[item for item in all_cities for i in range(3)])

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df, annot=False, cmap="YlGnBu")
plt.title('Non Recurrent Feature Heatmap @ ALL_SCALES ' + str(scale))
plt.xlabel('Cities')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('Non Recurrent Feature Heatmap City-wise cutoff @ ALL_SCALES ' + str(scale) + ".png", dpi=300)
plt.show(block=False)


filter_important_features_recurrent = copy.copy( filter_important_features_recurrent_city_wise )
filter_important_features_non_recurrent = copy.copy( filter_important_features_non_recurrent_city_wise )


def read_pdp_data(file_name):
    df = pd.read_csv(file_name, header=None)
    return df.T[1:][0], df.T[1:][1]


# for feature in features:
#
#     for scale in scales:
#         plt.clf()
#         # plt.figure(figsize=(10, 6))
#         for city in cities:
#             if city not in filter_important_features_non_recurrent[scale][feature]:
#                 linewidth = 0.5; linestyle='--'
#             else:
#                 linewidth = 2.3; linestyle='-'
#
#             file_name = f'PDP-MEAN_MAX_max_FI_Mean_{city}_Scale_{scale}_TOD_0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23_FEATURE_{feature}csv'
#
#             if os.path.exists(file_name):
#                 grid, pdp_values = read_pdp_data(file_name)
#                 plt.plot(grid, pdp_values, label=city, color=city_colors[city], linewidth=linewidth, linestyle=linestyle)
#             else:
#                 print(f'File not found: {file_name}')
#
#         plt.title(f'PDP for {feature} at Scale {scale}', fontsize=15)
#         plt.xlabel(feature, fontsize=11)
#         plt.ylabel("Jam Factor", fontsize=11)
#         plt.legend(fontsize=11, ncol=2)
#         plt.ylim(0, 10)
#         plt.tight_layout()
#         plt.savefig(f'Non-Recurrent_case_{feature}_scale_{scale}.png', dpi=300)
#         plt.show(block=False)
#
#
# for feature in features:
#
#     for scale in scales:
#         plt.clf()
#         # plt.figure(figsize=(10, 6))
#         for city in cities:
#             if city not in filter_important_features_recurrent[scale][feature]:
#                 linewidth = 0.5; linestyle='--'
#             else:
#                 linewidth = 2.3; linestyle='-'
#
#             file_name = f'PDP-MEAN_MAX_mean_FI_Mean_{city}_Scale_{scale}_TOD_0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23_FEATURE_{feature}csv'
#
#             if os.path.exists(file_name):
#                 grid, pdp_values = read_pdp_data(file_name)
#                 plt.plot(grid, pdp_values, label=city, color=city_colors[city], linewidth=linewidth, linestyle=linestyle)
#             else:
#                 print(f'File not found: {file_name}')
#
#         plt.title(f'PDP for {feature} at Scale {scale}', fontsize=15)
#         plt.xlabel(feature, fontsize=11)
#         plt.ylabel("Jam Factor", fontsize=11)
#         plt.legend(fontsize=11, ncol=2)
#         plt.ylim(0, 5)
#         plt.tight_layout()
#         plt.savefig(f'Recurrent_case_{feature}_scale_{scale}.png', dpi=300)
#         plt.show(block=False)

cities = list(city_colors.keys())
features = list(common_features_colors.keys())
for feature in features:


    for scale in scales:
        plt.clf()
        # plt.figure(figsize=(10, 6))
        for city in cities:
            print (city, scale, feature)
            if feature not in filter_important_features_non_recurrent[scale][city]:
                linewidth = 0.5;
                linestyle = '--'
            else:
                linewidth = 2.3;
                linestyle = '-'

            file_name = f'PDP-MEAN_MAX_max_FI_Mean_{city}_Scale_{scale}_TOD_0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23_FEATURE_{feature}csv'

            if os.path.exists(file_name):
                grid, pdp_values = read_pdp_data(file_name)
                plt.plot(grid, pdp_values, label=city, color=city_colors[city], linewidth=linewidth,
                         linestyle=linestyle)
            else:
                print(f'File not found: {file_name}')

        plt.title(f'PDP for {feature} at Scale {scale}', fontsize=15)
        plt.xlabel(feature, fontsize=11)
        plt.ylabel("Jam Factor", fontsize=11)
        plt.legend(fontsize=11, ncol=2)
        plt.ylim(0, 10)
        plt.tight_layout()
        plt.savefig(f'City-wise_Otsu-Non-Recurrent_case_{feature}_scale_{scale}.png', dpi=300)
        plt.show(block=False)

for feature in features:
    for scale in scales:
        plt.clf()
        # plt.figure(figsize=(10, 6))
        for city in cities:
            if feature not in filter_important_features_recurrent[scale][city]:
                linewidth = 0.5;
                linestyle = '--'
            else:
                linewidth = 2.3;
                linestyle = '-'

            file_name = f'PDP-MEAN_MAX_mean_FI_Mean_{city}_Scale_{scale}_TOD_0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23_FEATURE_{feature}csv'

            if os.path.exists(file_name):
                grid, pdp_values = read_pdp_data(file_name)
                plt.plot(grid, pdp_values, label=city, color=city_colors[city], linewidth=linewidth,
                         linestyle=linestyle)
            else:
                print(f'File not found: {file_name}')

        plt.title(f'PDP for {feature} at Scale {scale}', fontsize=15)
        plt.xlabel(feature, fontsize=11)
        plt.ylabel("Jam Factor", fontsize=11)
        plt.legend(fontsize=11, ncol=2)
        plt.ylim(0, 5)
        plt.tight_layout()
        plt.savefig(f'City-wise_Otsu-Recurrent_case_{feature}_scale_{scale}.png', dpi=300)
        plt.show(block=False)

non_recurrent_case_correlations = {}
non_recurrent_case_x_axis = {}
for feature in features:
    non_recurrent_case_correlations[feature] = {}
    non_recurrent_case_x_axis[feature] = {}
    for city in cities:
        non_recurrent_case_correlations[feature][city] = {}
        non_recurrent_case_x_axis[feature][city] = {}
        for scale in scales:
            non_recurrent_case_correlations[feature][city][scale] = []
            non_recurrent_case_x_axis[feature][city][scale] = []

            file_name = f'PDP-MEAN_MAX_max_FI_Mean_{city}_Scale_{scale}_TOD_0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23_FEATURE_{feature}csv'

            if os.path.exists(file_name):
                grid, pdp_values = read_pdp_data(file_name)
                # plt.plot(grid, pdp_values, label=city, color=city_colors[city], linewidth=linewidth,
                #          linestyle=linestyle)
                non_recurrent_case_correlations[feature][city][scale] = pdp_values.tolist()
                non_recurrent_case_x_axis[feature][city][scale] = grid.tolist()
            else:
                print(f'File not found: {file_name}')


recurrent_case_correlations = {}
recurrent_case_x_axis = {}
for feature in features:
    recurrent_case_correlations[feature] = {}
    recurrent_case_x_axis[feature] = {}
    for city in cities:
        recurrent_case_correlations[feature][city] = {}
        recurrent_case_x_axis[feature][city] = {}
        for scale in scales:
            recurrent_case_correlations[feature][city][scale] = []
            recurrent_case_x_axis[feature][city][scale] = []

            file_name = f'PDP-MEAN_MAX_mean_FI_Mean_{city}_Scale_{scale}_TOD_0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23_FEATURE_{feature}csv'

            if os.path.exists(file_name):
                grid, pdp_values = read_pdp_data(file_name)
                # plt.plot(grid, pdp_values, label=city, color=city_colors[city], linewidth=linewidth,
                #          linestyle=linestyle)
                recurrent_case_correlations[feature][city][scale] = pdp_values.tolist()
                recurrent_case_x_axis[feature][city][scale] = grid.tolist()
            else:
                print(f'File not found: {file_name}')

# sprint (non_recurrent_case_x_axis)
# sprint (non_recurrent_case_correlations)
# sprint (recurrent_case_x_axis)
# sprint (recurrent_case_correlations)



# Import Output of sprint above, shown below:
# from precomputeddicts import non_recurrent_case_x_axis, non_recurrent_case_correlations, recurrent_case_x_axis, recurrent_case_correlations


from scipy import stats
print ("Non recurrent case")
for feature in features:
    for city in cities:
        try:
            l25 = len(non_recurrent_case_correlations[feature][city][25])
            l50 = len(non_recurrent_case_correlations[feature][city][50])
            l100 = len(non_recurrent_case_correlations[feature][city][100])
            L = min([l25, l50, l100])
            # print ("================")
            # print (city, feature, l25, l50, l100, L)
            corr_50_25 = stats.pearsonr(
                non_recurrent_case_correlations[feature][city][50][:L],
                non_recurrent_case_correlations[feature][city][25][:L]
            )
            corr_50_100 = stats.pearsonr(
                non_recurrent_case_correlations[feature][city][50][:L],
                non_recurrent_case_correlations[feature][city][100][:L]
            )
            corr_25_100 = stats.pearsonr(
                non_recurrent_case_correlations[feature][city][25][:L],
                non_recurrent_case_correlations[feature][city][100][:L]
            )
            plt.clf()
            # print ("================")
            # if ((corr_50_100.statistic) < -0.3 or
            #         (corr_50_25.statistic) < -0.3 or
            #         (corr_25_100.statistic) < -0.3):
            sprint (city, feature, corr_50_25.statistic, corr_50_100.statistic)
            plt.plot(non_recurrent_case_x_axis[feature][city][25], non_recurrent_case_correlations[feature][city][25], label="scale 25")
            plt.plot(non_recurrent_case_x_axis[feature][city][50], non_recurrent_case_correlations[feature][city][50], label="scale 50")
            plt.plot(non_recurrent_case_x_axis[feature][city][100], non_recurrent_case_correlations[feature][city][100], label="scale 100")
            plt.title("Non- recurrent_" + city + "_" + feature)
            plt.legend()
            plt.xlabel(feature)
            plt.ylabel("Jam Factor")
            # plt.ylim(0, 10)
            plt.savefig("Non Recurrent PDP variation across scales" + city + "_" + feature + ".png", dpi=300)
            plt.show(block=False)
        except:
            continue
        # print ("================")


from scipy import stats
print ("Recurrent case")
for feature in features:
    for city in cities:
        try:
            l25 = len(recurrent_case_correlations[feature][city][25])
            l50 = len(recurrent_case_correlations[feature][city][50])
            l100 = len(recurrent_case_correlations[feature][city][100])
            L = min([l25, l50, l100])
            # print ("================")
            # print (city, feature, l25, l50, l100, L)
            corr_50_25 = stats.pearsonr(
                        recurrent_case_correlations[feature][city][50][:L],
                        recurrent_case_correlations[feature][city][25][:L]
            )
            corr_50_100 = stats.pearsonr(
                recurrent_case_correlations[feature][city][50][:L],
                recurrent_case_correlations[feature][city][100][:L]
            )
            corr_25_100 = stats.pearsonr(
                recurrent_case_correlations[feature][city][25][:L],
                recurrent_case_correlations[feature][city][100][:L]
            )
            plt.clf()
            # print ("================")
            # if (np.sign(corr_50_100.statistic) == -1 or np.sign(corr_50_25.statistic) == -1 or np.sign(corr_25_100.statistic)):
            # if ((corr_50_100.statistic) < -0.3 or
            #         (corr_50_25.statistic) < -0.3 or
            #             (corr_25_100.statistic) < -0.3):
            #     sprint (city, feature, corr_50_25.statistic, corr_50_100.statistic)
            plt.plot(recurrent_case_x_axis[feature][city][25], recurrent_case_correlations[feature][city][25], label="scale 25")
            plt.plot(recurrent_case_x_axis[feature][city][50], recurrent_case_correlations[feature][city][50], label="scale 50")
            plt.plot(recurrent_case_x_axis[feature][city][100], recurrent_case_correlations[feature][city][100], label="scale 100")
            plt.title("Recurrent_" + city + "_" + feature)
            plt.xlabel(feature)
            plt.ylabel("Jam Factor")
            plt.legend()
            # plt.ylim(0, 10)
            plt.savefig("Recurrent variation across scales" + city + "_" + feature + ".png", dpi=300)
            plt.show(block=False)
        except:
            continue
