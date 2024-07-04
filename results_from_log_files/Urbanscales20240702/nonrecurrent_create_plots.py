import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from smartprint import smartprint as sprint

NANCOUNTPERCITY = 4  # Set this to the desired threshold

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
def plot_data_by_model(df, model_types, city_list):
    for model in model_types:
        plt.figure(figsize=(7, 5))
        for city in city_list:
            subset = df[(df['City'] == city) & (df['Model'] == model)]
            subset = subset.sort_values(by='TileArea')
            if not subset.empty:
                plt.plot(subset['TileArea'], subset['GoF'], marker='o', linestyle='-', label=f"{city} ({model})",
                         color=colors[city])

        plt.title(r'GoF (R$^2$) vs '+f'Tile Area for {model} Model Across Cities')
        plt.xlabel(r'Tile Area $km^2$')
        plt.ylabel('Goodness of Fit (GoF)')
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.ylim(0, 1)
        plt.tight_layout();
        plt.savefig("nonrecurrent_Fi_1.png",dpi=300)
        plt.show()


# Example usage
city_list = ["Mumbai", "Auckland", "Istanbul", "MexicoCity", "Bogota", "NewYorkCity", "Capetown"]
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
sns.heatmap(heatmap_data, annot=False, cmap="coolwarm", center=0, cbar_kws={'label': 'Feature Importance (mean |SHAP|)'},
            yticklabels=True, xticklabels=True)
plt.title("Raw heatmap without filtering")
plt.ylabel("Feature")
plt.xlabel("City-Scale Combination")
plt.tight_layout();
plt.savefig("nonrecurrent_Fi_2a.png", dpi=300)
plt.show()


method = 'otsu'  # Change this variable to switch methods


for column in heatmap_data.columns:
    if method == 'otsu':
        thresh = threshold_otsu(np.array(heatmap_data[column].to_list()))
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
sns.heatmap(heatmap_data, annot=False, cmap="coolwarm", center=0, cbar_kws={'label': 'Feature Importance (mean |SHAP|)'},
            yticklabels=True, xticklabels=True)
plt.title("Heatmap with column wise Otsu thresholding")
plt.ylabel("Feature")
plt.xlabel("City-Scale Combination")
plt.tight_layout();
plt.savefig("nonrecurrent_Fi_2b.png", dpi=300)
plt.show()


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
sns.heatmap(heatmap_data, annot=False, cmap="coolwarm", center=0, cbar_kws={'label': 'Feature Importance (mean |SHAP|)'},
            yticklabels=True, xticklabels=True)
plt.title("Heatmap with > Otsu in > 50% scales")
plt.ylabel("Feature")
plt.xlabel("City-Scale Combination")
plt.tight_layout()
plt.tight_layout();
plt.savefig("nonrecurrent_Fi_2c.png", dpi=300)
plt.show()





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
    plt.xlabel('Tile Area')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance vs. Scale for Each City-Feature Combination')
    # plt.legend(loc='best', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout();
    plt.savefig("nonrecurrent_Fi_3a.png", dpi=300)
    plt.show()
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
        # plt.title('Elbow Method for Determining the Optimal Number of Clusters')
        # plt.grid(True)
        # plt.tight_layout();
        # plt.show()

        # Plot Silhouette scores
        plt.figure(figsize=(14, 8))
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis for Determining the Optimal Number of Clusters')
        plt.grid(True)
        plt.tight_layout();
        plt.savefig("nonrecurrent_Fi_3b.png", dpi=300)
        plt.show()

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
    city_labels = [label.split('-')[0] for label in heatmap_data.columns]

    cluster_color = {0: "#E69F00", 1:"#F0E442", 2:"#CC79A7"}

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
    plt.title(f'Clustered Feature Importance vs. Scale (n_clusters={HARDCODED_CLUSTER})')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.tight_layout();
    plt.savefig("nonrecurrent_Fi_3c.png", dpi=300)
    plt.show()

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
    plt.xlabel('Tile Area')
    plt.ylabel('Feature Importance')
    plt.title('ABSSHAP for Each City-Feature Combination')
    # plt.legend(loc='best', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout();
    plt.savefig("nonrecurrent_Fi_3d.png", dpi=300)
    plt.show()




########## FIGURE 3
if 1==1: # allow code folding
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt


    # Load and prepare your data
    def load_data_direction(file_path):
        print(file_path)
        df = pd.read_csv(file_path, delim_whitespace=True, header=None)
        #     sprint (df.shape, len(["garbage" + str(x) for x in [0,1,2,3,4,5,6,7,8]] + ["City-Scale-tod", "garbage7", "feature",  "ratio", "num", "den"]))
        df.columns = ["garbage" + str(x) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8]] + ["City-Scale-tod", "garbage7", "feature",
                                                                                  "ratio", "num", "den"]
        split_columns = df['City-Scale-tod'].str.split('-', expand=True)

        df['City'] = split_columns[3].apply(lambda x: x.lower())
        df['Scale'] = split_columns[4].astype(int)
        df['Scale'] = split_columns[4].astype(str)
        df['Scale'] = df['Scale'].apply(lambda x: x.zfill(3))
        df['City-Scale'] = df['City'] + '-' + df['Scale'].astype(str)
        df["signeddenominator"] = np.sign(df["ratio"]) * np.abs(df["num"])

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
    sns.heatmap(heatmap_data, annot=False, cmap="seismic", cbar_kws={'label': 'Signed Numerator'}, center=0,
                yticklabels=True, xticklabels=original_xticks)
    plt.tight_layout();
    plt.savefig("nonrecurrent_Fi_4a.png", dpi=300)
    plt.show()




    # Load and prepare your data
    def load_data_direction(file_path):
        print(file_path)
        df = pd.read_csv(file_path, delim_whitespace=True, header=None)
        #     sprint (df.shape, len(["garbage" + str(x) for x in [0,1,2,3,4,5,6,7,8]] + ["City-Scale-tod", "garbage7", "feature",  "ratio", "num", "den"]))
        df.columns = ["garbage" + str(x) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8]] + ["City-Scale-tod", "garbage7", "feature",
                                                                                  "ratio", "num", "den"]
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
    plt.title("Sensitivity")
    plt.tight_layout();
    plt.savefig("nonrecurrent_Fi_4b.png", dpi=300)
    plt.show()


    plt.figure(figsize=(14, 8))
    sns.heatmap(np.log(heatmap_data+0.127), annot=False, cmap="seismic", cbar_kws={'label': 'Sensitivity Ratio'},
                yticklabels=True, xticklabels=True)
    plt.title("Log of sensitivity")
    plt.savefig("nonrecurrent_Fi_4c.png", dpi=300)
    plt.tight_layout();
    plt.show()


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
    plt.title('Feature Importance vs. Scale for Each City-Feature Combination')
    # plt.legend(loc='best', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout();
    plt.savefig("nonrecurrent_Fi_4d.png", dpi=300)
    plt.show()
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
        plt.title('Elbow Method for Determining the Optimal Number of Clusters')
        plt.grid(True)
        plt.tight_layout();
        plt.savefig("nonrecurrent_Fi_4e.png", dpi=300)
        plt.show()

        # Plot Silhouette scores
        plt.figure(figsize=(14, 8))
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis for Determining the Optimal Number of Clusters')
        plt.grid(True)
        plt.tight_layout();
        plt.savefig("nonrecurrent_Fi_4f.png", dpi=300)
        plt.show()

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
    city_labels = [label.split('-')[0] for label in heatmap_data.columns]

    cluster_color = {0: "#E69F00", 1:"#F0E442", 2:"#CC79A7"}

    # Plot the clustered time series
    plt.figure(figsize=(5, 6))

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
        plt.plot(area_list, cluster_representatives[i], label=f"Cluster {i + 1} (Representative)", linewidth=2, color=cluster_color[i])

    plt.xlabel('Tile Area')
    plt.ylabel('Feature Importance')
    plt.title(f'Clustered Feature Importance vs. Scale (n_clusters={HARDCODED_CLUSTER})')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.tight_layout();
    plt.savefig("nonrecurrent_Fi_4g.png", dpi=300)
    plt.show()

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
    plt.xlabel('Tile Area')
    plt.ylabel('Feature Importance')
    plt.title('Ratio vs. Scale for Each City-Feature Combination')
    # plt.legend(loc='best', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout();
    plt.savefig("nonrecurrent_Fi_4h.png", dpi=300)
    plt.show()

debug_pitstop = True


# K means and K medoids tested but doesnt work.  K medoids tested with both Euclidean and cosine similarities)