import csv
import os
import config
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from slugify import slugify


marker_type = "train"
model = "RF"

for model in ["RF"]:# ["RF", "GB"]:
    for marker_type in ["val", "train"]:
        # Define the path to your CSV file
        # csv_file_path = os.path.join(config.BASE_FOLDER, config.results_folder, "feature_importance_"+model+".csv")
        csv_file_path = os.path.join(config.BASE_FOLDER, config.results_folder, "feature_importance.csv")

        # Create a defaultdict to store the feature counts for each key combination
        feature_counts = defaultdict(lambda: defaultdict(int))


        # Read the CSV file and iterate over each row
        with open(csv_file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Extract the key combination (cityname, scale, marker)
                key = slugify( " ".join([row['cityname'], row['scale'], row['marker'], row['tod']]) ) # row['plot_counter']

                if row['marker'] == marker_type:
                    continue
                # if row['cityname'] != "London":
                #     continue

                # Extract the ordered list of features from the row
                features = list(row.values())[5:]

                # Calculate the counts for each feature within the specified slots
                for i, feature in enumerate(features):
                    # Increment the count for each top-k value
                    for k in [1, 2, 3, 4, 5]:
                        feature_counts[key][f"Top_{k}_{feature}"] += 0
                        if i < k:
                            feature_counts[key][f"Top_{k}_{feature}"] += 1

        # Convert the feature counts to a dataframe
        feature_counts_df = pd.DataFrame(feature_counts)

        # Generate a heatmap for each top-k value
        for k in [1, 2, 3, 4, 5]:
            # Get the top-k features
            top_k_features = [f"Top_{k}_{feature.split('_')[2]}" for feature in feature_counts_df.index]
            top_k_features = list(set(top_k_features))
            list.sort(top_k_features)

            # Subset the feature counts dataframe for the top-k features
            heatmap_data = feature_counts_df.loc[top_k_features]

            top_k_features = [k.split("_")[2] for k in top_k_features]

            plt.figure(figsize=(10, 6))
            sns.heatmap(heatmap_data, annot=False, cmap='YlGnBu')
            plt.title(f'Top-{k} Feature Heatmap')
            # plt.xlabel('Key Combination')
            plt.ylabel('Feature')
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.gca().set_yticklabels([tick.get_text().split("_")[2] for tick in plt.gca().get_yticklabels()])
            plt.tight_layout()
            plt.savefig(os.path.join(config.BASE_FOLDER, config.results_folder, model + "_" + marker_type + "_Top_" + str(k) + ".png"), dpi=300)
            plt.show()
