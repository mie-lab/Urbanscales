import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

# Define cities and their features
# List of cities
from slugify import slugify

city_list = ["Auckland", "Bogota", "Capetown", "Istanbul", "London",
             "MexicoCity",
             "Mumbai", "NewYorkCity", "Singapore", "Zurich"]

common_features = [
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
        # 'streets_per_node_count_5',
        'total_crossings'
    ]

common_features_colors = {
    'betweenness': '#1f77b4',  # blue
    'circuity_avg': '#ff7f0e',  # orange
    'global_betweenness': '#2ca02c',  # green
    'k_avg': '#d62728',  # red
    'lane_density': '#9467bd',  # purple
    'metered_count': '#8c564b',  # brown
    'non_metered_count': '#e377c2',  # pink
    'street_length_total': '#7f7f7f',  # gray
    'streets_per_node_count_5': '#bcbd22',  # lime
    'total_crossings': '#17becf'  # cyan
}

common_features_acronyms = {
        "BW" :'betweenness',
        "CA": 'circuity_avg',
        "GBW":'global_betweenness',
        "KAVG": 'k_avg',
        "LD": 'lane_density',
        # 'm',
        "MC":'metered_count',
        # 'n',
        "NMC": 'non_metered_count',
        "SLT": 'street_length_total',
        "SPNC5": 'streets_per_node_count_5',
        "TC":'total_crossings'
    }
# Dictionary of features for each city
# feature_list = {
#     "Auckland": ["metered_count", "total_crossings"],
#     "Bogota": ["metered_count", "total_crossings"],
#     "Capetown": ["metered_count", "total_crossings"],
#     "Istanbul": ["street_length_total"],
#     "London": ["global_betweenness", "k_avg"],
#     # "Mexico City": [],  # No common features
#     "Mumbai": ["street_length_total", "circuity_avg"],
#     "NewYorkCity": ["metered_count"],
#     "Singapore": ["total_crossings"],
#     "Zurich": ["metered_count"]
# }

feature_list = {
    "Auckland": common_features,
    "Bogota": common_features,
    "Capetown": common_features,
    "Istanbul": common_features,
    "London": common_features,
    "MexicoCity": common_features,  # No common features
    "Mumbai": common_features,
    "NewYorkCity": common_features,
    "Singapore": common_features,
    "Zurich": common_features
}



def read_and_aggregate_data(city, feature, mean_max, scales):
    aggregated_data = {}
    for scale in scales:
        file_name = f"MEAN_MAX_{mean_max}_FI_Mean_{city}_Scale_{scale}_TOD_0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23.csv"
        column_names = ['feature', 'scale', 'city', 'tod', 'FI_mean']  # List your column names here
        try:
            df = pd.read_csv(file_name, header=None, names=column_names)
        except:
            print ("Missing filename: ", file_name, "\n ignored")
            continue
        aggregated_data[scale] = df[df['feature'] == feature]['FI_mean'].mean()
    return aggregated_data

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

# ... [Other parts of the script remain the same] ...

def plot_fi_vs_scale(city, feature_data, features, linestyle, Recurrent_non_recurrent):
    scales = [25, 50, 100]

      # Define more colors if needed

    for i, feature in enumerate(features):
        if Recurrent_non_recurrent == "Non R":
            mean_max = "max"
        elif Recurrent_non_recurrent == "R":
            mean_max = "mean"
        fi_mean = [feature_data[feature][mean_max][scale] for scale in scales]

        # fi_max = [feature_data[feature]['max'][scale] for scale in scales]

        plt.plot(scales, fi_mean, label=f'{feature}', color=common_features_colors[feature],
                 linestyle=linestyle, linewidth=2)
        # plt.plot(scales, fi_max, label=f'{feature} Max', color=color, linestyle='--', linewidth=3)


    # plt.show(block=False)


def analyze_monotonicity_and_correlation(city, feature, data, mean_max) :#,feature_ranges):
    scales = [25, 50, 100]
    fis = [data[scale] for scale in scales]

    # Monotonicity check
    if all(x <= y for x, y in zip(fis, fis[1:])):
        monotonicity = "Dependent"
    elif all(x >= y for x, y in zip(fis, fis[1:])):
        monotonicity = "Inverted"
    elif abs(max(fis) - min(fis))<0.2 * np.mean(fis) and np.mean(fis)>0.1: # :np.mean(feature_ranges[feature][mean_max]):
        monotonicity = "NM: Important"
    else:
        monotonicity = "NM: --"
    # print ("Range: ", mean_max, feature, abs(max(fis) - min(fis)))
    correlation = linregress(scales, fis).rvalue
    return city, feature, monotonicity, correlation

"""

for city in city_list:
    print (city)

    # using data from temp2 file:
    # feature_ranges = {
    #     "betweenness": {"mean": 0.04597491691686059, "max": 0.05108693180930041},
    #     "metered_count": {
    #         "mean": [0.31259602378059487, 0.2849257282843944, 0.32897317059036524, 0.12866392828905227],
    #         "max": [0.6293410191788906, 0.07684809604088638, 0.07546141720711702, 0.29031890533200455]
    #     },
    #     "total_crossings": {
    #         "mean": [0.1583207893919382, 0.16450482862432897, 0.14154647230925949],
    #         "max": [0.4826918917401068, 0.20801395798512212, 0.40450592852338646]
    #     },
    #     "street_length_total": {
    #         "mean": [0.08712109402733148, 0.07021212422246578],
    #         "max": [0.14275479244030853, 0.05793771121283378]
    #     },
    #     "global_betweenness": {"mean": 0.028538748836211533, "max": 0.03004274647575285},
    #     "k_avg": {"mean": 0.045280616993588776, "max": 0.04488598093563059},
    #     "circuity_avg": {"mean": 0.03650493456168592, "max": 0.06307857318043444},
    # }

    feature_data = {}
    for feature in feature_list[city]:
        data_mean = read_and_aggregate_data(city, feature, "mean", [25, 50, 100])
        data_max = read_and_aggregate_data(city, feature, "max", [25, 50, 100])
        feature_data[feature] = {'mean': data_mean, 'max': data_max}

        _, feature, monotonicity, correlation = analyze_monotonicity_and_correlation(city, feature, data_mean, mean_max="mean") #, feature_ranges=feature_ranges)
        print ("Case: ", "mean", feature, monotonicity, correlation)
        _, feature, monotonicity, correlation = analyze_monotonicity_and_correlation(city, feature, data_max, mean_max="max") # , feature_ranges=feature_ranges)
        print("Case: ", "max", feature, monotonicity, correlation)





################ SCALE 50
recurrent_congestion_feature_list = {
    "Auckland": ["TC", "SPNC5"],
    "Bogota": ["MC", "BW", "TC"],
    "Capetown": ["TC", "MC"],
    "Istanbul": ["SLT"],
    "London": ["SLT", "KAVG", "GBW", "LD", "SPNC5"],
    "MexicoCity": ["SLT", "MC", "SPNC5"],
    "Mumbai": ["BW", "CA", "SLT"],
    "NewYorkCity": ["MC"],
    "Singapore": ["NMC"],
    "Zurich": ["GBW", "MC"]
}

non_recurrent_congestion_feature_list = {
    "Auckland": ["MC"],
    "Bogota": ["TC", "MC"],
    "Capetown": ["TC", "MC"],
    "Istanbul": ["SLT"],
    "London": ["MC"],
    "MexicoCity": ["TC"],
    "Mumbai": ["TC", "SLT"],
    "NewYorkCity": ["MC"],
    "Singapore": ["TC"],
    "Zurich": ["MC"]
}
for key in recurrent_congestion_feature_list:
    temp = list(recurrent_congestion_feature_list[key])
    try:
        recurrent_congestion_feature_list[key] = [common_features_acronyms[x] for x in temp]
    except:
        debug_breakpoint = True

for key in non_recurrent_congestion_feature_list:
    temp = list(non_recurrent_congestion_feature_list[key])
    non_recurrent_congestion_feature_list[key] = [common_features_acronyms[x] for x in temp]




#
for city in city_list:
    plt.clf()
    plt.figure()
    feature_data = {}
    for feature in feature_list[city]:
        data_mean = read_and_aggregate_data(city, feature, "mean", [25, 50, 100])
        data_max = read_and_aggregate_data(city, feature, "max", [25, 50, 100])
        feature_data[feature] = {'mean': data_mean, 'max': data_max}
    plot_fi_vs_scale(city, feature_data, recurrent_congestion_feature_list[city], linestyle="-",Recurrent_non_recurrent="R")
    plot_fi_vs_scale(city, feature_data, non_recurrent_congestion_feature_list[city], linestyle="--",Recurrent_non_recurrent="Non R")
    plt.xlabel('Scale')
    plt.ylabel('Feature Importance (FI)')
    plt.title(f'FI vs Scale for {city}')
    plt.plot([],[],label="non-Recurrent", color="black",linestyle="--")
    plt.plot([], [], label="Recurrent", color="black", linestyle="-")
    plt.legend(ncol=2, fontsize=7)
    plt.ylim(0, 1)
    plt.savefig(f'Combined FI vs Scale for {city}_CUTOFF_50', dpi=300)



################ SCALE 25
recurrent_congestion_feature_list = {
    "Auckland": ["MC"],
    "Bogota": ["MC"],
    "Capetown": ["MC"],
    "Istanbul": ["SLT"],
    "London": ["SLT"],
    "MexicoCity": ["MC", "SLT"],
    "Mumbai": ["BW", "CA", "SLT", "MC"],
    "NewYorkCity": ["MC"],
    "Singapore": ["NMC"],
    "Zurich": ["BW", "MC"]
}
non_recurrent_congestion_feature_list = {
    "Auckland": ["MC"],
    "Bogota": ["MC"],
    "Capetown": ["SLT", "NMC", "MC"],
    "Istanbul": ["BW", "SLT"],
    "London": ["MC"],
    "MexicoCity": ["TC"],
    "Mumbai": ["GBW", "MC"],
    "NewYorkCity": ["MC"],
    "Singapore": ["SLT"],
    "Zurich": ["TC"]
}

for key in recurrent_congestion_feature_list:
    temp = list(recurrent_congestion_feature_list[key])
    recurrent_congestion_feature_list[key] = [common_features_acronyms[x] for x in temp]


for key in non_recurrent_congestion_feature_list:
    temp = list(non_recurrent_congestion_feature_list[key])
    non_recurrent_congestion_feature_list[key] = [common_features_acronyms[x] for x in temp]

#
for city in city_list:
    plt.clf()
    plt.figure()
    feature_data = {}
    for feature in feature_list[city]:
        data_mean = read_and_aggregate_data(city, feature, "mean", [25, 50, 100])
        data_max = read_and_aggregate_data(city, feature, "max", [25, 50, 100])
        feature_data[feature] = {'mean': data_mean, 'max': data_max}
    plot_fi_vs_scale(city, feature_data, recurrent_congestion_feature_list[city], linestyle="-",Recurrent_non_recurrent="R")
    plot_fi_vs_scale(city, feature_data, non_recurrent_congestion_feature_list[city], linestyle="--",Recurrent_non_recurrent="Non R")
    plt.xlabel('Scale')
    plt.ylabel('Feature Importance (FI)')
    plt.title(f'FI vs Scale for {city}')
    plt.plot([],[],label="non-Recurrent", color="black",linestyle="--")
    plt.plot([], [], label="Recurrent", color="black", linestyle="-")
    plt.legend(ncol=2, fontsize=7)
    plt.ylim(0, 1)
    plt.savefig(f'Combined FI vs Scale for {city}_CUTOFF_25', dpi=300)



for city in city_list:
    plt.clf()
    plt.figure()

    feature_data = {}
    for feature in feature_list[city]:
        data_mean = read_and_aggregate_data(city, feature, "mean", [25, 50, 100])
        data_max = read_and_aggregate_data(city, feature, "max", [25, 50, 100])
        feature_data[feature] = {'mean': data_mean, 'max': data_max}


    plot_fi_vs_scale(city, feature_data, feature_list[city], linestyle="-", Recurrent_non_recurrent="R")
    # plot_fi_vs_scale(city, feature_data, non_recurrent_congestion_feature_list[city], linestyle="--",Recurrent_non_recurrent="Non R")
    plt.xlabel('Scale')
    plt.ylabel('Feature Importance (FI)')
    plt.title(f'FI vs Scale for {city}')
    plt.plot([],[],label="non-Recurrent", color="black",linestyle="--")
    plt.plot([], [], label="Recurrent", color="black", linestyle="-")
    plt.legend(ncol=2, fontsize=7)
    plt.ylim(0, 1)
    plt.savefig(f'All_features_Recurrent FI vs Scale for {city}', dpi=300)



for city in city_list:
    plt.clf()
    plt.figure()
    print (city)

    feature_data = {}
    for feature in feature_list[city]:
        data_mean = read_and_aggregate_data(city, feature, "mean", [25, 50, 100])
        data_max = read_and_aggregate_data(city, feature, "max", [25, 50, 100])
        feature_data[feature] = {'mean': data_mean, 'max': data_max}

    # plot_fi_vs_scale(city, feature_data, recurrent_congestion_feature_list[city], linestyle="-",Recurrent_non_recurrent="R")
    plot_fi_vs_scale(city, feature_data, feature_list[city], linestyle="--", Recurrent_non_recurrent="Non R")
    plt.xlabel('Scale')
    plt.ylabel('Feature Importance (FI)')
    plt.title(f'FI vs Scale for {city}')
    plt.plot([],[],label="non-Recurrent", color="black",linestyle="--")
    plt.plot([], [], label="Recurrent", color="black", linestyle="-")
    plt.legend(ncol=2, fontsize=7)
    plt.ylim(0,1)
    plt.savefig(f'All_features_Non_Recurrent FI vs Scale for {city}', dpi=300)

"""

city_colors = {
    "Auckland": "#1f77b4",  # Muted blue
    "Bogota": "#ff7f0e",    # Safety orange
    "Capetown": "#2ca02c",  # Cooked asparagus green
    "Istanbul": "#d62728",  # Brick red
    "London": "#9467bd",    # Muted purple
    "MexicoCity": "#8c564b",# Chestnut brown
    "Mumbai": "#e377c2",    # Raspberry yogurt pink
    "NewYorkCity": "#7f7f7f",# Middle gray
    "Singapore": "#bcbd22", # Curry yellow-green
    "Zurich": "#17becf"     # Blue-teal
}

import matplotlib.pyplot as plt

# Initialize data structure
common_features = [
        'betweenness',
        'circuity_avg',
        'global_betweenness',
        'k_avg',
        'lane_density',
        # 'm',
        'metered_count',
        # 'n',
        'non_metered_count',
        'street_length_total',
        'streets_per_node_count_5',
        'total_crossings'
    ]

import matplotlib.pyplot as plt

# Initialize data structure
common_features = [
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
        # 'streets_per_node_count_5',
        'total_crossings'
    ]

import matplotlib.pyplot as plt
city_list = ["Auckland", "Bogota", "Capetown", "Istanbul", "London",
             "MexicoCity",
             "Mumbai", "NewYorkCity",
             # "Singapore", "Zurich"
             ]

import matplotlib.colors as mcolors

for scale_list in [[25], [50], [100], [25, 50, 100]]:
    # Create the box plot
    plt.figure(figsize=(12, 6))

    # Initialize data structure
    feature_data = {feature: {'recurrent': [], 'nonrecurrent': []} for feature in common_features}

    # Aggregate data across cities and features
    for city in city_list:
        for feature in feature_list[city]:
            data_recurrent = read_and_aggregate_data(city, feature, "mean", scale_list)
            # data_recurrent = read_and_aggregate_data(city, feature, "mean", [25, 50, 100])
            data_nonrecurrent = read_and_aggregate_data(city, feature, "max", scale_list)
            # data_nonrecurrent = read_and_aggregate_data(city, feature, "max", [25, 50, 100])

            # Combine data across scales for each city
            feature_data[feature]['recurrent'].extend(list(data_recurrent.values()))
            feature_data[feature]['nonrecurrent'].extend(list(data_nonrecurrent.values()))

    # Plot box plots and scatter points
    colorlist = []
    for city in city_list:
        colorlist.extend([city_colors[city]] * len(scale_list))

    for i, feature in enumerate(common_features, 1):
        plt.scatter([i - 0.2] * len(feature_data[feature]['recurrent']), feature_data[feature]['recurrent'],
                    color=colorlist, s=50)
        # print (colorlist)
        # print ("One list")
        plt.scatter([i + 0.2] * len(feature_data[feature]['nonrecurrent']), feature_data[feature]['nonrecurrent'],
                    color=colorlist, s=50)

    # Prepare data for plotting
    plot_data = []
    feature_labels = []
    positions = []
    for i, feature in enumerate(feature_data, 1):
        feature_labels.append(feature)
        plot_data.append(feature_data[feature]['recurrent'])   # Recurrent FI values
        plot_data.append(feature_data[feature]['nonrecurrent']) # Nonrecurrent FI values
        positions.extend([i - 0.2, i + 0.2])  # Adjust position for each pair of box plots




    box = plt.boxplot(plot_data, positions=positions, patch_artist=True, widths=0.39, showfliers=False)

    # Set colors for recurrent and nonrecurrent

    # Set colors for recurrent and nonrecurrent with alpha
    alpha_value = 0.6  # Set alpha value between 0 (transparent) and 1 (opaque)
    color_names = ['tab:blue', 'tab:orange']
    for patch, color_name in zip(box['boxes'], color_names * len(feature_data)):
        color_rgba = mcolors.to_rgba(color_name, alpha=alpha_value)
        patch.set_facecolor(color_rgba)

    # Set median line color to black
    for median in box['medians']:
        median.set_color('black')

    # Setting labels for x-axis
    plt.xticks(range(1, len(feature_labels) + 1), feature_labels, rotation=90)

    # Adding labels and title
    plt.xlabel('Feature')
    plt.ylabel('Feature Importance (FI)')
    title = 'Feature Importance Across Cities @Scale ' + "-".join(str(x) for x in scale_list)
    plt.title(title)

    colorlist = []
    for city in city_list:
        colorlist.append(city_colors[city])
        plt.scatter([], [], color=city_colors[city], label=city, s=60)

    # plt.scatter([] * len(colorlist), [None] * len(colorlist), label=city_list, color=colorlist)
    plt.plot([], [], label="Recurrent", color="tab:blue", linewidth=10, alpha=alpha_value)
    plt.plot([], [], label="Non Recurrent", color="tab:orange", linewidth=10, alpha=alpha_value)
    plt.legend(ncol=2, fontsize=12.4)
    # plt.legend([box['boxes'][0], box['boxes'][1]], ['Recurrent', 'Nonrecurrent'], loc='upper right')

    plt.ylim(0, 0.7)
    # Display the plot
    plt.tight_layout()  # Adjust layout
    plt.savefig(slugify(title) + ".png", dpi=300)
    plt.show(block=False)

