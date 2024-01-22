import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sys

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.append("../../../")
from urbanscales.preprocessing.train_data import TrainDataVectors
import pickle
import copy
from sklearn.model_selection import KFold
from smartprint import smartprint as sprint

current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.join(current_dir, '..')
os.chdir(current_dir)
sprint (os.getcwd())

mean_max = "mean"
ZIPPEDfoldername = "train_data_three_scales_" + mean_max
PLOTS_foldername = mean_max + "_case_corr_analysis/"

if "mean" in ZIPPEDfoldername:
    assert "mean" in PLOTS_foldername
if "max" in ZIPPEDfoldername:
    assert "max" in PLOTS_foldername

os.system("tar -xf ../" + ZIPPEDfoldername + ".tar.gz")

class CustomUnpicklerTrainDataVectors(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module, name)

class CorrelationAnalysis:
    def __init__(self, X_data=None, Y_data=None, fname=None):
        if fname:
            temp = copy.deepcopy(CustomUnpicklerTrainDataVectors(open(fname, "rb")).load())
            self.__dict__.update(temp.__dict__)
            self.nparrayX = np.array(self.X)
            self.nparrayY = np.array(self.Y)
            self.columns = self.X.columns.tolist()
        elif X_data is not None and Y_data is not None:
            if isinstance(X_data, pd.DataFrame):
                self.nparrayX = X_data.to_numpy()
                self.columns = X_data.columns.tolist()
            else:  # assume it's a numpy array
                self.nparrayX = X_data
                self.columns = [f"feature_{i}" for i in range(X_data.shape[1])]  # generic names
            self.nparrayY = Y_data
        else:
            raise ValueError("Either provide a filename or both X_data and Y_data arrays for initialization.")
        print(self.nparrayX.shape, self.nparrayY.shape)


    def random_forest_feature_importance(self, common_features):
        n_splits = 7

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        total_feature_importances = np.zeros(len(common_features))

        for train_index, test_index in kfold.split(self.nparrayX, self.nparrayY):
            X_train, X_test = self.nparrayX[train_index], self.nparrayX[test_index]
            Y_train, Y_test = self.nparrayY[train_index], self.nparrayY[test_index]
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, Y_train)
            total_feature_importances += rf.feature_importances_
        total_feature_importances /= n_splits

        return total_feature_importances

    def plot_feature_importances(self, importances, scale, common_features, list_of_cities):
        plt.title(f"Feature Importances RF " + list_of_cities)
        plt.plot(range(len(common_features)), importances, marker='o', linestyle='-', label=f"Scale: {scale}")
        plt.xticks(range(len(common_features)), common_features, rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.legend()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from itertools import product


if __name__ == "__main__":
    list_of_cities = "Singapore|Zurich|Mumbai|Auckland|Istanbul|MexicoCity|Bogota|NewYorkCity|Capetown|London".split("|")
    list_of_cities_list_of_list = [
                                    # list_of_cities[:2],
                                    # list_of_cities[2:]
                                    [list_of_cities[0]],
                                    [list_of_cities[1]],
                                    [list_of_cities[2]],
                                    [list_of_cities[3]],
                                    [list_of_cities[4]],
                                    [list_of_cities[5]],
                                    [list_of_cities[6]],
                                    [list_of_cities[7]],
                                    [list_of_cities[8]],
                                    [list_of_cities[9]],

                                ]

    tod_list_of_list = [
        [6,7,8,9],
        # [0,1,2,3,4,5],
        # [10,11,12,13,14,15],
        # [16,17,18,19],
        # [20,21,22,23]
    # range(24),
    ]

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

    scale_list = [25, 50, 100]

    results = {}
    for list_of_cities in list_of_cities_list_of_list:
        for tod_list in tod_list_of_list:
            for scale in scale_list:
                combined_X_for_scale, combined_Y_for_scale = [], []
                for city in list_of_cities:
                    combined_X_for_city, combined_Y_for_city = [], []
                    for tod in tod_list:
                        fname = f"{ZIPPEDfoldername}/{city}/_scale_{scale}_train_data_{tod}.pkl"
                        try:
                            temp_obj = CustomUnpicklerTrainDataVectors(open(fname, "rb")).load()
                            if isinstance(temp_obj.X, pd.DataFrame):
                                filtered_X = temp_obj.X[list(common_features)]
                                combined_X_for_city.append(filtered_X)
                                combined_Y_for_city.append(temp_obj.Y)

                                # Compute and store mean and variance for each column in temp_obj.X
                                for column in filtered_X.columns:
                                    column_mean = np.mean(filtered_X[column])
                                    column_variance = np.var(filtered_X[column])

                                    # Storing the computed values for each city, scale, and column combination
                                    results[city, scale, column] = {"mean": column_mean, "variance": column_variance}



                        except FileNotFoundError:
                            print("Error in fname:", fname)

                    # After processing each city and time of day, concatenate data
                    X = pd.concat(combined_X_for_city, axis=0)
                    Y = np.concatenate(combined_Y_for_city)

                    # Splitting data
                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                                        random_state=42)

                    # Initialize and fit the scaler on training data
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Train the linear regression model with scaled data
                    lr = LinearRegression()
                    lr.fit(X_train_scaled, Y_train)
                    lr_pred = lr.predict(X_test_scaled)
                    # lr_r2 = r2_score(Y_test, lr_pred)
                    lr_r2 = np.mean( (Y_test - lr_pred) ** 2 )

                    # Extract feature importance for linear regression
                    feature_importance_lr = lr.coef_
                    feature_importance_dict_lr = dict(zip(common_features, feature_importance_lr))
                    results[city, scale, "feature_importance_lr"] = feature_importance_dict_lr

                    feature_importance_rf = 0
                    rf_r2 = 0
                    repeating_runs = 10
                    for run_num in tqdm(range(repeating_runs), desc="RF run count" + city + " scale: " + str(scale)):
                        # Splitting data
                        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                                            )

                        # Initialize and fit the scaler on training data
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)

                        # Train the RandomForest regression model with scaled data
                        rf = RandomForestRegressor(n_estimators=100)

                        rf.fit(X_train_scaled, Y_train)
                        rf_pred = rf.predict(X_test_scaled)
                        # rf_r2 = r2_score(Y_test, rf_pred)
                        rf_r2 += np.mean( (Y_test-rf_pred) ** 2 )

                        # Extract feature importance for random forest
                        feature_importance_rf += rf.feature_importances_

                    feature_importance_rf /= repeating_runs
                    rf_r2 /= repeating_runs

                    feature_importance_dict_rf = dict(zip(common_features, feature_importance_rf))
                    results[city, scale, "feature_importance_rf"] = feature_importance_dict_rf

                    # Store R^2 values
                    results[city, scale, "lr_r2"] = lr_r2
                    results[city, scale, "rf_r2"] = rf_r2



                    print(city, scale, "std_Y", np.std(temp_obj.Y))
                    results[city, scale, "Y"] = {"mean": np.mean(temp_obj.Y), "variance": np.std(temp_obj.Y) ** 2}


    # Define a helper function to print tables neatly
    def print_table(city, scale, lr_r2, rf_r2, importance_lr, importance_rf):
        header = f"{city} - Scale {scale}"
        print(header)
        print('-' * len(header))
        print(f"{'Metric':<30} {'Linear Regression':<20} {'RandomForest':<20}")
        print('-' * 70)  # 30 + 20 + 20 characters wide
        print(f"{'R^2':<30} {lr_r2:<20.3f} {rf_r2:<20.3f}")
        for feature in common_features:
            imp_lr = importance_lr.get(feature, 0)
            imp_rf = importance_rf.get(feature, 0)
            print(f"{feature:<30} {imp_lr:<20.3f} {imp_rf:<20.3f}")
        print('\n')


    # After processing all data, print feature importances and R^2 values
    for citylist in list_of_cities_list_of_list:
        city = citylist[0]
        for scale in scale_list:
            lr_r2 = results.get((city, scale, "lr_r2"), 0)
            rf_r2 = results.get((city, scale, "rf_r2"), 0)
            importance_lr = results.get((city, scale, "feature_importance_lr"), {})
            importance_rf = results.get((city, scale, "feature_importance_rf"), {})
            print_table(city, scale, lr_r2, rf_r2, importance_lr, importance_rf)


    # For a given scale (e.g., 25)
    selected_scale = 25
    import matplotlib.pyplot as plt
    import numpy as np


    # Function to plot feature importances for a given model
    def plot_feature_importances(model_key, model_name):
        plt.figure(figsize=(12, 6))

        # For each city, extract its feature importances and plot them
        for citylist in list_of_cities_list_of_list:
            city = citylist[0]
            importances = [results.get((city, selected_scale, model_key), {}).get(feature, 0) for feature in
                           common_features]
            plt.plot(common_features, importances, label=city, marker='o')

        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title(f'Feature Importances for {model_name}' + " case: " + mean_max)
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig("LRvsRF/" + "feature_importance" + model_name + "-scale-" + str(selected_scale) + "-case-" + mean_max + ".png", dpi=300)
        plt.show(block=False)


    # Plot for Linear Regression
    plot_feature_importances("feature_importance_lr", "Linear Regression")

    # Plot for Random Forest
    plot_feature_importances("feature_importance_rf", "Random Forest")

    # for col in common_features + ["Y"]:
    #     for mean_Var in ["mean", "variance"]:
    #         plt.clf()
    #
    #         cities = sorted(set([key[0] for key in results.keys()]))
    #         scales = [25, 50, 100]
    #
    #         # Extracting variance values for plotting
    #         variances_for_scales = {scale: [results[city, scale, col][mean_Var] for city in cities] for scale in scales}
    #
    #         barWidth = 0.25
    #         r1 = np.arange(len(cities))
    #         r2 = [x + barWidth for x in r1]
    #         r3 = [x + barWidth for x in r2]
    #
    #         plt.bar(r1, variances_for_scales[25], color='b', width=barWidth, edgecolor='grey', label='scale 25')
    #         plt.bar(r2, variances_for_scales[50], color='r', width=barWidth, edgecolor='grey', label='scale 50')
    #         plt.bar(r3, variances_for_scales[100], color='g', width=barWidth, edgecolor='grey', label='scale 100')
    #
    #         plt.xlabel('Cities', fontweight='bold')
    #         plt.ylabel(mean_Var, fontweight='bold')
    #         plt.xticks([r + barWidth for r in range(len(cities))], cities, rotation=90)
    #         plt.legend()
    #         if col == "Y":
    #             plt.title(mean_max + " case" + "-" + col)
    #             plt.tight_layout()
    #             plt.savefig("train_data_plots/" + col + "-" + mean_Var + "-" + mean_max)
    #         else:
    #             plt.title(col)
    #             plt.tight_layout()
    #             plt.savefig("train_data_plots/" + col + "-" + mean_Var)
