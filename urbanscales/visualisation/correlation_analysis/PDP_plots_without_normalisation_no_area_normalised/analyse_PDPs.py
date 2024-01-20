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



def read_pdp_data(file_name):
    df = pd.read_csv(file_name, header=None)
    return df.T[1:][0], df.T[1:][1]


for feature in features:

    for scale in scales:
        plt.clf()
        # plt.figure(figsize=(10, 6))
        for city in cities:
            if city not in filter_important_features_non_recurrent[scale][feature]:
                linewidth = 0.5; linestyle='--'
            else:
                linewidth = 2.3; linestyle='-'

            file_name = f'PDP-MEAN_MAX_max_FI_Mean_{city}_Scale_{scale}_TOD_0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23_FEATURE_{feature}csv'
            
            if os.path.exists(file_name):
                grid, pdp_values = read_pdp_data(file_name)
                plt.plot(grid, pdp_values, label=city, color=city_colors[city], linewidth=linewidth, linestyle=linestyle)
            else:
                print(f'File not found: {file_name}')

        plt.title(f'PDP for {feature} at Scale {scale}', fontsize=15)
        plt.xlabel(feature, fontsize=11)
        plt.ylabel("Jam Factor", fontsize=11)
        plt.legend(fontsize=11, ncol=2)
        plt.ylim(0, 10)
        plt.tight_layout()
        plt.savefig(f'Non-Recurrent_case_{feature}_scale_{scale}.png', dpi=300)
        plt.show(block=False)


for feature in features:

    for scale in scales:
        plt.clf()
        # plt.figure(figsize=(10, 6))
        for city in cities:
            if city not in filter_important_features_recurrent[scale][feature]:
                linewidth = 0.5; linestyle='--'
            else:
                linewidth = 2.3; linestyle='-'

            file_name = f'PDP-MEAN_MAX_mean_FI_Mean_{city}_Scale_{scale}_TOD_0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23_FEATURE_{feature}csv'

            if os.path.exists(file_name):
                grid, pdp_values = read_pdp_data(file_name)
                plt.plot(grid, pdp_values, label=city, color=city_colors[city], linewidth=linewidth, linestyle=linestyle)
            else:
                print(f'File not found: {file_name}')

        plt.title(f'PDP for {feature} at Scale {scale}', fontsize=15)
        plt.xlabel(feature, fontsize=11)
        plt.ylabel("Jam Factor", fontsize=11)
        plt.legend(fontsize=11, ncol=2)
        plt.ylim(0, 5)
        plt.tight_layout()
        plt.savefig(f'Recurrent_case_{feature}_scale_{scale}.png', dpi=300)
        plt.show(block=False)
